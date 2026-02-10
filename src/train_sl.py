"""
train_sl.py
=========================================
NeuroChess Zero - Phase 1: Supervised Learning
=========================================

Description:
    This script performs Supervised Fine-Tuning (SFT) on the ChessResNet model.
    It streams high-level chess games from a local PGN file, encodes the board state
    into an 8x8x19 tensor, and minimizes the error between the network's prediction
    and the grandmaster's move.

Usage:
    python src/train_sl.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import chess
import chess.pgn
import numpy as np
import os
import time
import glob
import random
import logging
from pathlib import Path

# ==========================================
# 0. CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths - Adjusted for Local PC
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "supervised"
DATA_DIR = BASE_DIR / "data" / "raw"
PGN_FILENAME = "grandmaster_games.pgn"
PGN_PATH = DATA_DIR / PGN_FILENAME

# Hyperparameters
BATCH_SIZE = 1024           # Lower this if you get CUDA Out of Memory
LEARNING_RATE = 1e-4        # 0.0001
TOTAL_STEPS = 150_000       # Max training steps
BUFFER_SIZE = 20_000        # Number of positions to hold in RAM shuffle buffer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logger.info(f"📂 Checkpoint Dir: {CHECKPOINT_DIR}")
logger.info(f"♟️  PGN Source: {PGN_PATH}")
logger.info(f"🖥️  Device: {DEVICE}")

# ==========================================
# 1. ARCHITECTURE (ResNet)
# ==========================================
class ResidualBlock(nn.Module):
    """
    Standard ResNet block with two 3x3 convolutions and BatchNorm.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessResNet(nn.Module):
    """
    AlphaZero-style dual-headed network.
    Input: 8x8x19 Board State
    Output:
        - Policy: 4096 move probabilities
        - Value: Scalar [-1, 1] game evaluation
    """
    def __init__(self, num_res_blocks=6, num_channels=128, input_planes=19):
        super().__init__()
        # Backbone
        self.conv_input = nn.Conv2d(input_planes, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.res_tower = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        # Permute from NHWC to NCHW for PyTorch
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_tower:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

# ==========================================
# 2. HELPER FUNCTIONS & DATASET
# ==========================================
move_to_idx = { (i, j): count for count, (i, j) in enumerate((x, y) for x in range(64) for y in range(64)) }

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encodes the python-chess board object into an 8x8x19 float32 tensor.
    Planes 0-11: Pieces
    Planes 12-16: Aux (Turn, Castling)
    Plane 17: En Passant
    Plane 18: Move Clock
    """
    tensor = np.zeros((8, 8, 19), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row, col = 7 - (square // 8), square % 8
        c = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        tensor[row, col, c] = 1
    
    if board.turn == chess.WHITE: tensor[:, :, 12] = 1
    if board.has_kingside_castling_rights(chess.WHITE): tensor[:, :, 13] = 1
    if board.has_queenside_castling_rights(chess.WHITE): tensor[:, :, 14] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[:, :, 15] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[:, :, 16] = 1
    
    if board.ep_square:
        row, col = 7 - (board.ep_square // 8), board.ep_square % 8
        tensor[row, col, 17] = 1
    
    tensor[:, :, 18] = board.halfmove_clock / 100.0
    return tensor

def get_result(header) -> float:
    r = header.get("Result")
    if r == "1-0": return 1.0
    elif r == "0-1": return -1.0
    return 0.0

class StreamDataset(IterableDataset):
    """
    Streams games from a massive PGN file using a shuffle buffer.
    Efficient for datasets larger than RAM.
    """
    def __init__(self, pgn_path, buffer_size=20000):
        self.pgn_path = pgn_path
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        if not os.path.exists(self.pgn_path):
            logger.error(f"PGN file not found at {self.pgn_path}")
            return

        with open(self.pgn_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None: break # End of file
                
                result = get_result(game.headers)
                board = game.board()
                
                for move in game.mainline_moves():
                    state = encode_board(board)
                    action = move_to_idx.get((move.from_square, move.to_square), 0)
                    
                    # Perspective: Value is relative to current player
                    val = result if board.turn == chess.WHITE else -result

                    if len(buffer) < self.buffer_size:
                        buffer.append((state, action, val))
                    else:
                        idx = random.randint(0, len(buffer)-1)
                        yield buffer[idx]
                        buffer[idx] = (state, action, val)

                    board.push(move)

        # Yield remaining items in buffer
        random.shuffle(buffer)
        for item in buffer:
            yield item

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def save_checkpoint(model, optimizer, scheduler, step, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    state = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }
    torch.save(state, path)
    logger.info(f"✅ Saved Checkpoint: {filename}")

def load_latest_checkpoint(model, optimizer, scheduler):
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    if not files:
        logger.info("No previous checkpoints found. Starting from scratch.")
        return 0

    def get_step_num(filename):
        try:
            if "latest" in os.path.basename(filename): return -1
            return int(os.path.basename(filename).split('_')[-1].replace('.pth', ''))
        except:
            return -1

    latest_file = max(files, key=get_step_num)
    logger.info(f"🔄 Resuming from: {latest_file}")

    try:
        checkpoint = torch.load(latest_file, map_location=DEVICE)
        
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            return checkpoint['step']
        else:
            model.load_state_dict(checkpoint) # Legacy format support
            return get_step_num(latest_file)
            
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        return 0

def train():
    if not os.path.exists(PGN_PATH):
        logger.error(f"❌ ERROR: Please place your .pgn file at: {PGN_PATH}")
        return

    model = ChessResNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    global_step = load_latest_checkpoint(model, optimizer, scheduler)

    dataset = StreamDataset(PGN_PATH, buffer_size=BUFFER_SIZE)
    # Num_workers > 0 works on PC (Linux/Win), usually 0 is safer for stability
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=0)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    model.train()
    t0 = time.time()
    running_loss = 0.0

    try:
        for i, (states, actions, values) in enumerate(dataloader):
            current_step = global_step + i + 1
            
            # Move to device
            states = states.to(DEVICE, non_blocking=True)
            actions = actions.to(DEVICE, non_blocking=True)
            values = values.float().unsqueeze(1).to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                p_pred, v_pred = model(states)
                loss_p = ce_loss(p_pred, actions)
                loss_v = mse_loss(v_pred, values)
                loss = loss_p + loss_v

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            if current_step % 100 == 0:
                dt = time.time() - t0
                speed = (100 * BATCH_SIZE) / dt
                avg_loss = running_loss / 100
                logger.info(f"Step {current_step} | Loss: {avg_loss:.4f} | Speed: {speed:.0f} pos/s | LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0
                t0 = time.time()

            if current_step % 2000 == 0:
                save_checkpoint(model, optimizer, scheduler, current_step, f"resnet_step_{current_step}.pth")

    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted. Saving emergency checkpoint...")
        save_checkpoint(model, optimizer, scheduler, current_step, "resnet_latest.pth")
    except Exception as e:
        logger.error(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    train()
