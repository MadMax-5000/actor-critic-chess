import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.pgn
import numpy as np
import zstandard as zstd
import io
import os


def board_to_tensor(board):
    tensor = np.zeros((8, 8, 13), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        tensor[row, col, idx] = 1
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1
    return tensor


class ActorCritic(nn.Module):
    def __init__(self, board_size=8, num_channels=13):
        super().__init__()

        # 1. Convolutional Block (Extract features)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Added layer
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 128, kernel_size=1)  # Bottleneck layer

        # 2. Fully Connected Layers
        self.fc_input_dim = 128 * board_size * board_size
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)  # Increased size
        self.fc2 = nn.Linear(1024, 512)

        # Heads
        self.policy_head = nn.Linear(512, 4096)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # x: (batch, 8, 8, 13) -> (batch, 13, 8, 8)
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))  # No BN usually needed on 1x1 here, but okay

        # Flatten
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


move_to_idx = {
    (i, j): count
    for count, (i, j) in enumerate((x, y) for x in range(64) for y in range(64))
}


def encode_move(move: chess.Move):
    return move_to_idx.get((move.from_square, move.to_square), 0)


def get_game_result(header):
    res = header.get("Result", "*")
    if res == "1-0":
        return 1.0
    elif res == "0-1":
        return -1.0
    elif res == "1/2-1/2":
        return 0.0
    return None


def train_from_large_file(zst_path, save_path="pretrained_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    BATCH_SIZE = 256
    MIN_ELO = 2000
    MAX_GAMES = 50000

    print(f"Opening {zst_path}...")
    print(f"Filtering for ELO > {MIN_ELO}. This might take a moment to find matches...")

    games_used = 0
    total_loss = 0
    batch_states, batch_actions, batch_values = [], [], []

    # Open the compressed file
    with open(zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            while games_used < MAX_GAMES:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break

                headers = game.headers
                try:
                    w_elo = int(headers.get("WhiteElo", 0))
                    b_elo = int(headers.get("BlackElo", 0))
                except:
                    continue

                if w_elo < MIN_ELO or b_elo < MIN_ELO:
                    continue  # Skip this game

                # 2. Get Result
                result = get_game_result(headers)
                if result is None:
                    continue

                # 3. Process Moves
                board = game.board()
                for move in game.mainline_moves():
                    # Create Input Tensor
                    state = board_to_tensor(board)

                    # Create Target Action
                    action = encode_move(move)

                    # Create Target Value
                    val = result if board.turn == chess.WHITE else -result

                    batch_states.append(state)
                    batch_actions.append(action)
                    batch_values.append(val)

                    board.push(move)

                    # 4. Train when batch is full
                    if len(batch_states) >= BATCH_SIZE:
                        # Convert to Tensors
                        s = torch.tensor(np.array(batch_states)).to(device)
                        a = torch.tensor(batch_actions).to(device)
                        v = (
                            torch.tensor(batch_values, dtype=torch.float32)
                            .unsqueeze(1)
                            .to(device)
                        )

                        optimizer.zero_grad()
                        p_pred, v_pred = model(s)

                        loss = policy_loss_fn(p_pred, a) + value_loss_fn(v_pred, v)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        batch_states, batch_actions, batch_values = [], [], []

                games_used += 1
                if games_used % 100 == 0:
                    print(
                        f"Learned from {games_used} games. Avg Loss: {total_loss / 100:.4f}"
                    )
                    total_loss = 0

                    # Save periodically
                    torch.save(model.state_dict(), save_path)

    print("Training Finished!")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# RUNNNIIINNGGG
if __name__ == "__main__":
    # Note: Use forward slashes '/' or double backslashes '\\' for Windows paths

    file_path = "/content/lichess_db_standard_rated_2025-11.pgn.zst"

    if os.path.exists(file_path):
        train_from_large_file(file_path)
    else:
        print(f"Error: Could not find file at {file_path}")
        print("Please check the path inside the code.")
