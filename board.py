import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
import random

board = chess.Board()


# transforming the board to a tensor
def board_to_tensor(board):
    tensor = np.zeros((8, 8, 13), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        tensor[row, col, idx] = 1

    # add turn channel
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1

    return tensor


# δ = r + γ * V(s_prime) - V(s)


# Defining the actor critic network
class ActorCritic(nn.Module):
    def __init__(self, board_size=8, num_channels=13):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)

        # Actor head
        self.policy_head = nn.Linear(512, 4096)  # max possible chess moves

        # Critic head
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # x: (batch, 8, 8, 13) -> convert to (batch, 13, 8, 8)
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # converts (batch, 128, 8, 8) into (batch, 128 * 8 * 8) which is (batch, 8192)
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))

        policy = self.policy_head(x)
        value = self.value_head(x)  # scalar value
        return policy, value


# mapping moves to indices and vice versa
def create_move_mapping():
    move_to_idx = {}
    idx_to_move = {}
    count = 0

    for i in range(64):
        for j in range(64):
            move_to_idx[(i, j)] = count
            idx_to_move[count] = (i, j)
            count += 1
    return move_to_idx, idx_to_move


move_to_idx, idx_to_move = create_move_mapping()


def encode_move(move: chess.Move):
    return move_to_idx[(move.from_square, move.to_square)]


def decode_move(idx: int, board: chess.Board):
    from_sq, to_sq = idx_to_move[idx]

    # if pawn in last rank we assume it's a queen
    # TODO : handle other promotions

    promotion = None
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and chess.square_rank(to_sq) == 7) or (
            piece.color == chess.BLACK and chess.square_rank(to_sq) == 0
        ):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


# verifying the model takes only legal moves
def select_action(model, board, device="cuda"):
    model.eval()

    # 1. preparing input
    state = board_to_tensor(board)
    state_tensor = (
        torch.from_numpy(state).unsqueeze(0).to(device)
    )  # shape (1, 8, 8, 13)

    # 2. Forward pass
    policy_logits, value = model(state_tensor)  # logits = (1, 4096)

    # 3. create mask for legal moves
    # setting the probability of illegal moves to -infinity to that softmax makes them 0 later
    mask = torch.full((1, 4096), -float("inf")).to(device)

    legal_moves = list(board.legal_moves)
    legal_indices = [encode_move(m) for m in legal_moves]

    # unmasking legal moves
    mask[0, legal_indices] = 0

    # applying mask and softmax
    masked_logits = policy_logits + mask
    probs = F.softmax(masked_logits, dim=-1)

    # sample an action
    dist = torch.distributions.Categorical(probs)
    action_idx = dist.sample()

    # decode move
    chosen_move = decode_move(action_idx.item(), board)

    # validation
    if chosen_move not in legal_moves:
        print("AI made an illegal move falling back to random")

        chosen_move = random.choice(legal_moves)

    return chosen_move, dist.log_prob(action_idx), value


# ==================
# Training loop
# ==================


def Astra_training(model, optimizer, gamma=0.99, device="cuda"):
    board = chess.Board()
    model.train()

    log_probs = []
    rewards = []
    values = []

    # playing one full game of chess
    while not board.is_game_over():
        move, log_prob, value = select_action(model, board, device)
        board.push(move)

        # storing data for training
        log_probs.append(log_prob)
        values.append(value)

        # reward shaping
        if board.is_checkmate():
            rewards.append(1 if board.turn == chess.BLACK else -1)
        elif board.is_stalemate() or board.is_insufficient_material():
            rewards.append(0)
        else:
            rewards.append(0)
    # calculating returns
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # Normalize

    # Calculate loss
    policy_loss = 0
    value_loss = 0

    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()

        # Actor loss: -log_prob * advantage
        policy_loss += -log_prob * advantage

        # Critic loss : MSE(value, return)
        value_loss += F.mse_loss(value.squeeze(), R)
    total_loss = policy_loss + value_loss

    # backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


# RUNNNNINNNGGGG
# initializing the model with 4096 outputs (64*64)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )

# 1. Resize Model FIRST
model = ActorCritic(num_channels=13)  # Note: 13 channels now
model.to(device)

# 2. Define Optimizer SECOND
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("starting training for 10000 episodes")
for episode in range(10000):
    loss = Astra_training(model, optimizer, device=device)
    print(f"Episode {episode}, Loss : {loss:.4f}")

    # Save checkpoint every 1000 episodes
    if (episode + 1) % 1000 == 0:
        torch.save(
            {
                "episode": episode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"checkpoint_episode_{episode + 1}.pth",
        )
        print(f"Checkpoint saved at episode {episode + 1}")

print("Training complete!")
