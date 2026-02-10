"""
train_rl.py
=========================================
NeuroChess Zero - Phase 2: Reinforcement Learning
=========================================

Description:
    This script executes the Self-Play loop using Monte Carlo Tree Search (MCTS).
    1.  Loads a 'generation' model (starts with gen_0.pth).
    2.  Model plays against itself to generate game data (SFT + Noise).
    3.  Training loop updates the model based on game results.
    4.  Evaluates new model vs previous model.
    5.  Saves new generation and repeats.

Usage:
    Ensure 'gen_0.pth' exists in checkpoints/rl/
    python src/train_rl.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import os
import time
import random
import math
import copy
import glob
import pickle
import logging
from collections import deque
from pathlib import Path

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RL_CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "rl"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Ensure directories exist
os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Important Files
INITIAL_MODEL_NAME = "gen_0.pth"
BUFFER_FILE = DATA_PROCESSED_DIR / "replay_buffer.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RL Hyperparameters
SIMULATIONS = 200         # MCTS Sims per move (Higher = Stronger but Slower)
EPISODES_PER_ITER = 10    # Games to play before training
TOTAL_ITERATIONS = 500    # Total generations to produce
MAX_BUFFER_SIZE = 10000   # Sliding window of game positions
EVAL_EVERY_N = 5          # How often to run Arena Mode
CPUCT = 1.2               # Exploration constant

# ==========================================
# 1. MODEL ARCHITECTURE (Must match SL)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessResNet(nn.Module):
    def __init__(self, num_res_blocks=6, num_channels=128, input_planes=19):
        super().__init__()
        self.conv_input = nn.Conv2d(input_planes, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.res_tower = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_tower:
            x = block(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
move_to_idx = { (i, j): count for count, (i, j) in enumerate((x, y) for x in range(64) for y in range(64)) }

def encode_board(board):
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

# ==========================================
# 3. MCTS ENGINE
# ==========================================
class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, cpuct=CPUCT):
        u = cpuct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value() + u

class MCTS:
    def __init__(self, model, device="cuda", num_simulations=SIMULATIONS):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations

    def search(self, board, exploration=True):
        root = MCTSNode(prior=0)
        
        # Initial Expansion
        self._expand(root, board)

        # Add Dirichlet Noise to root for exploration diversity
        if exploration:
            self._add_noise(root)

        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy()
            
            # Selection
            while node.children:
                move, node = max(node.children.items(), key=lambda item: item[1].ucb_score())
                scratch_board.push(move)

            # Expansion & Evaluation
            value = 0
            if not scratch_board.is_game_over():
                value = self._expand(node, scratch_board)
            else:
                if scratch_board.is_checkmate():
                    value = -1.0 # Current player lost
                else:
                    value = 0.0 # Draw

            # Backpropagation
            while node is not None:
                node.value_sum += value
                node.visit_count += 1
                value = -value # Switch perspective
                node = node.parent
        return root

    def _expand(self, node, board):
        state = encode_board(board)
        tensor = torch.tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(tensor)
        
        policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.item()
        
        for move in board.legal_moves:
            idx = move_to_idx.get((move.from_square, move.to_square), 0)
            if idx < 4096:
                node.children[move] = MCTSNode(parent=node, prior=policy_probs[idx])
        return value

    def _add_noise(self, node):
        actions = list(node.children.keys())
        if not actions: return
        noise = np.random.dirichlet([0.6] * len(actions)) # Gamma=0.6 (High Chaos)
        for i, move in enumerate(actions):
            node.children[move].prior = 0.75 * node.children[move].prior + 0.25 * noise[i]

# ==========================================
# 4. SELF-PLAY & TRAINING LOGIC
# ==========================================
class ChessRLDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def execute_self_play(mcts):
    """Plays one game against itself and returns training examples."""
    board = chess.Board()
    examples = []
    move_count = 0
    MAX_MOVES = 150 # Prevent infinite games

    while not board.is_game_over() and move_count < MAX_MOVES:
        root = mcts.search(board, exploration=True)

        # Create Policy Target Vector
        policy_probs = np.zeros(4096, dtype=np.float32)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        for move, child in root.children.items():
            idx = move_to_idx.get((move.from_square, move.to_square), 0)
            policy_probs[idx] = child.visit_count / total_visits

        examples.append([encode_board(board), policy_probs, 0])

        # Temperature: Play probabilistically for first 45 moves, then strictly
        if move_count < 45:
            moves = list(root.children.keys())
            visits = [root.children[m].visit_count for m in moves]
            probs = [v/total_visits for v in visits]
            move = np.random.choice(moves, p=probs)
        else:
            move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]

        board.push(move)
        move_count += 1

    # Assign Rewards
    result_type = "Draw"
    reward = 0.0

    if board.is_checkmate():
        reward = 1.0 # The player who just moved won
        result_type = "Checkmate"
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_repetition():
        reward = 0.0
        result_type = "Draw"
    else:
        reward = -0.5 # Slight punishment for reaching move limit without winning
        result_type = "Move Limit"

    # Backpropagate reward to history
    processed_examples = []
    for i, (state, policy, _) in enumerate(examples):
        # Alternating turns means alternating rewards
        # If the game ended on turn N with reward +1, turn N-1 (opponent) gets -1
        turn_reward = reward if ((len(examples) - 1 - i) % 2 == 0) else -reward
        processed_examples.append((state, policy, turn_reward))

    return processed_examples, {"result": result_type, "length": move_count}

def run_evaluation(current_model, best_model_path, device, num_games=4):
    """Compares current generation vs the baseline (Gen 0 or previous best)."""
    logger.info(f"⚔️ EVALUATION ARENA ⚔️")
    
    baseline_model = ChessResNet().to(device)
    try:
        baseline_ckpt = torch.load(best_model_path, map_location=device)
        # Handle state dict wrapping
        if 'model_state' in baseline_ckpt: baseline_model.load_state_dict(baseline_ckpt['model_state'])
        else: baseline_model.load_state_dict(baseline_ckpt)
        baseline_model.eval()
    except Exception as e:
        logger.error(f"Failed to load baseline for eval: {e}")
        return

    mcts_curr = MCTS(current_model, device, num_simulations=50) # Fast MCTS for eval
    mcts_base = MCTS(baseline_model, device, num_simulations=50)
    results = {"W": 0, "L": 0, "D": 0}

    # Play X games
    for i in range(num_games):
        # Alternate colors
        if i % 2 == 0:
            white, black = mcts_curr, mcts_base
            p1_name, p2_name = "Current", "Baseline"
        else:
            white, black = mcts_base, mcts_curr
            p1_name, p2_name = "Baseline", "Current"

        board = chess.Board()
        while not board.is_game_over() and board.fullmove_number < 100:
            player = white if board.turn == chess.WHITE else black
            root = player.search(board, exploration=False) # Greedy play
            move = max(root.children.items(), key=lambda i: i[1].visit_count)[0]
            board.push(move)

        # Determine winner
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
        else:
            winner = "Draw"

        # Log result relative to Current Model
        if (winner == "White" and p1_name == "Current") or (winner == "Black" and p2_name == "Current"):
            results["W"] += 1
        elif winner == "Draw":
            results["D"] += 1
        else:
            results["L"] += 1
            
        logger.info(f"  Game {i+1}: White({p1_name}) vs Black({p2_name}) -> {winner}")

    logger.info(f"🏆 Match Result: +{results['W']} -{results['L']} ={results['D']}")

def train_network(model, examples, device, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    dataset = ChessRLDataset(examples)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    total_loss = 0
    steps = 0
    
    for epoch in range(epochs):
        for state, policy_target, value_target in dataloader:
            state = state.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            p_pred, v_pred = model(state)
            
            # Policy Loss: Cross Entropy
            loss_p = -torch.mean(torch.sum(policy_target * F.log_softmax(p_pred, dim=1), dim=1))
            # Value Loss: MSE
            loss_v = mse(v_pred, value_target)
            
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
    return total_loss / steps if steps > 0 else 0

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    logger.info(f"🖥️  Device: {DEVICE}")
    model = ChessResNet().to(DEVICE)

    # 1. Load Replay Buffer
    if os.path.exists(BUFFER_FILE):
        logger.info(f"📂 Loading Buffer from {BUFFER_FILE}")
        with open(BUFFER_FILE, "rb") as f:
            replay_buffer = pickle.load(f)
        logger.info(f"✅ Loaded {len(replay_buffer)} positions.")
    else:
        logger.info("🆕 Starting with Empty Buffer.")
        replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)

    # 2. Determine Generation
    existing_gens = glob.glob(str(RL_CHECKPOINT_DIR / "gen_*.pth"))
    start_iteration = 0

    if existing_gens:
        def get_gen_num(filename):
            try: return int(os.path.basename(filename).split('_')[-1].replace('.pth', ''))
            except: return -1
        latest_gen = max(existing_gens, key=get_gen_num)
        start_iteration = get_gen_num(latest_gen)
        logger.info(f"🚀 Resuming from Gen {start_iteration}")
        model.load_state_dict(torch.load(latest_gen, map_location=DEVICE))
    else:
        # Check for initial bootstrap model
        bootstrap_path = RL_CHECKPOINT_DIR / INITIAL_MODEL_NAME
        if os.path.exists(bootstrap_path):
            logger.info("🔰 Loading Bootstrap Model (Gen 0)...")
            ckpt = torch.load(bootstrap_path, map_location=DEVICE)
            if 'model_state' in ckpt: model.load_state_dict(ckpt['model_state'])
            else: model.load_state_dict(ckpt)
        else:
            logger.error(f"❌ ERROR: {INITIAL_MODEL_NAME} not found in {RL_CHECKPOINT_DIR}")
            logger.error("Please copy your supervised model there and rename it to gen_0.pth")
            return

    mcts = MCTS(model, device=DEVICE, num_simulations=SIMULATIONS)

    # 3. RL Loop
    for iteration in range(start_iteration, TOTAL_ITERATIONS):
        logger.info(f"--- Iteration {iteration+1} (Gen {iteration} -> Gen {iteration+1}) ---")

        model.eval()
        new_examples = []
        stats = {"Checkmate": 0, "Draw": 0, "Move Limit": 0}

        start_time = time.time()
        for i in range(EPISODES_PER_ITER):
            # Print progress in place
            print(f"Playing Game {i+1}/{EPISODES_PER_ITER}...", end="\r")
            try:
                game_data, info = execute_self_play(mcts)
                new_examples.extend(game_data)
                res = info["result"]
                if res in stats: stats[res] += 1
            except Exception as e:
                logger.error(f"Game Error: {e}")

        duration = time.time() - start_time
        print(f"\n✅ Finished {EPISODES_PER_ITER} games in {duration:.1f}s")
        logger.info(f"📊 Stats: {stats}")

        # Update Buffer
        replay_buffer.extend(new_examples)
        with open(BUFFER_FILE, "wb") as f: pickle.dump(replay_buffer, f)

        # Train if we have enough data
        if len(replay_buffer) > 500:
            logger.info("🧠 Training Network...")
            loss = train_network(model, list(replay_buffer), DEVICE, epochs=2)
            logger.info(f"📉 Loss: {loss:.4f}")

            new_gen_num = iteration + 1
            save_path = RL_CHECKPOINT_DIR / f"gen_{new_gen_num}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"💾 Saved Gen {new_gen_num}")

            # Update MCTS model
            mcts.model = model

            # Periodic Eval
            if new_gen_num % EVAL_EVERY_N == 0:
                bootstrap_path = RL_CHECKPOINT_DIR / INITIAL_MODEL_NAME
                run_evaluation(model, bootstrap_path, DEVICE, num_games=4)
        else:
            logger.warning(f"Buffer too small ({len(replay_buffer)}). Skipping training.")

if __name__ == "__main__":
    main()
