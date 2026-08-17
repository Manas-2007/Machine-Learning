import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# =========================================================
# ENVIRONMENT
# =========================================================

class WarehouseRobot:

    def __init__(self):

        self.rows = 5
        self.cols = 5

        self.start = (0, 0)
        self.goal = (4, 4)

        self.obstacles = {
            (1, 1),
            (1, 2),
            (2, 2),
            (3, 2)
        }

        self.actions = 4

        self.max_steps = 50

        self.reset()


    def reset(self):

        self.position = self.start

        self.steps = 0

        return self.get_state()


    def get_state(self):

        row, col = self.position

        goal_row, goal_col = self.goal

        distance_row = goal_row - row
        distance_col = goal_col - col

        state = np.array([
            row / (self.rows - 1),
            col / (self.cols - 1),
            distance_row / (self.rows - 1),
            distance_col / (self.cols - 1)
        ], dtype=np.float32)

        return state


    def step(self, action):

        self.steps += 1

        old_row, old_col = self.position

        new_row = old_row
        new_col = old_col


        # -------------------------------------------------
        # ACTIONS
        # -------------------------------------------------

        if action == 0:

            new_row -= 1

        elif action == 1:

            new_row += 1

        elif action == 2:

            new_col -= 1

        elif action == 3:

            new_col += 1


        # -------------------------------------------------
        # CHECK GRID BOUNDARY
        # -------------------------------------------------

        if (
            new_row < 0
            or new_row >= self.rows
            or new_col < 0
            or new_col >= self.cols
        ):

            reward = -5

            new_row = old_row
            new_col = old_col

        else:

            new_position = (new_row, new_col)

            # -------------------------------------------------
            # OBSTACLE
            # -------------------------------------------------

            if new_position in self.obstacles:

                reward = -100

                self.position = new_position

                return self.get_state(), reward, True


            # -------------------------------------------------
            # NORMAL MOVEMENT
            # -------------------------------------------------

            old_distance = abs(
                self.goal[0] - old_row
            ) + abs(
                self.goal[1] - old_col
            )

            new_distance = abs(
                self.goal[0] - new_row
            ) + abs(
                self.goal[1] - new_col
            )


            if new_distance < old_distance:

                reward = 2

            elif new_distance > old_distance:

                reward = -2

            else:

                reward = -1


            self.position = new_position


        # -------------------------------------------------
        # GOAL
        # -------------------------------------------------

        if self.position == self.goal:

            reward = 100

            return self.get_state(), reward, True


        # -------------------------------------------------
        # MAX STEPS
        # -------------------------------------------------

        if self.steps >= self.max_steps:

            reward = -20

            return self.get_state(), reward, True


        return self.get_state(), reward, False


# =========================================================
# DQN NETWORK
# =========================================================

class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, action_size)
        )


    def forward(self, state):

        return self.network(state)


# =========================================================
# REPLAY MEMORY
# =========================================================

class ReplayMemory:

    def __init__(self, capacity):

        self.memory = deque(
            maxlen=capacity
        )


    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


    def sample(self, batch_size):

        return random.sample(
            self.memory,
            batch_size
        )


    def __len__(self):

        return len(self.memory)


# =========================================================
# PARAMETERS
# =========================================================

STATE_SIZE = 4

ACTION_SIZE = 4

EPISODES = 1000

GAMMA = 0.95

LEARNING_RATE = 0.001

BATCH_SIZE = 64

MEMORY_SIZE = 10000

EPSILON = 1.0

EPSILON_DECAY = 0.995

EPSILON_MIN = 0.05

TARGET_UPDATE = 20


# =========================================================
# CREATE ENVIRONMENT
# =========================================================

env = WarehouseRobot()


# =========================================================
# CREATE NETWORKS
# =========================================================

policy_network = DQN(
    STATE_SIZE,
    ACTION_SIZE
)


target_network = DQN(
    STATE_SIZE,
    ACTION_SIZE
)


target_network.load_state_dict(
    policy_network.state_dict()
)


target_network.eval()


# =========================================================
# LOSS AND OPTIMIZER
# =========================================================

loss_function = nn.MSELoss()


optimizer = optim.Adam(
    policy_network.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# REPLAY MEMORY
# =========================================================

memory = ReplayMemory(
    MEMORY_SIZE
)


# =========================================================
# ACTION NAMES
# =========================================================

action_names = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}


# =========================================================
# TRAINING STATISTICS
# =========================================================

successful_episodes = 0

episode_rewards = []

episode_losses = []


# =========================================================
# TRAINING
# =========================================================

for episode in range(1, EPISODES + 1):

    state = env.reset()

    total_reward = 0

    total_loss = 0

    loss_count = 0

    done = False

    steps = 0


    while not done:

        steps += 1


        # -------------------------------------------------
        # SELECT ACTION
        # -------------------------------------------------

        if random.random() < EPSILON:

            action = random.randint(
                0,
                ACTION_SIZE - 1
            )

        else:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():

                q_values = policy_network(
                    state_tensor
                )

            action = torch.argmax(
                q_values
            ).item()


        # -------------------------------------------------
        # ENVIRONMENT
        # -------------------------------------------------

        next_state, reward, done = env.step(
            action
        )


        # -------------------------------------------------
        # STORE EXPERIENCE
        # -------------------------------------------------

        memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )


        state = next_state

        total_reward += reward


        # -------------------------------------------------
        # TRAIN NETWORK
        # -------------------------------------------------

        if len(memory) >= BATCH_SIZE:

            batch = memory.sample(
                BATCH_SIZE
            )


            states = torch.tensor(
                np.array(
                    [item[0] for item in batch]
                ),
                dtype=torch.float32
            )


            actions = torch.tensor(
                [item[1] for item in batch],
                dtype=torch.long
            )


            rewards = torch.tensor(
                [item[2] for item in batch],
                dtype=torch.float32
            )


            next_states = torch.tensor(
                np.array(
                    [item[3] for item in batch]
                ),
                dtype=torch.float32
            )


            dones = torch.tensor(
                [item[4] for item in batch],
                dtype=torch.float32
            )


            # -------------------------------------------------
            # CURRENT Q VALUES
            # -------------------------------------------------

            current_q_values = policy_network(
                states
            )


            current_q = current_q_values.gather(
                1,
                actions.unsqueeze(1)
            ).squeeze(1)


            # -------------------------------------------------
            # FUTURE Q VALUES
            # -------------------------------------------------

            with torch.no_grad():

                next_q_values = target_network(
                    next_states
                )

                best_next_q = torch.max(
                    next_q_values,
                    dim=1
                ).values


            # -------------------------------------------------
            # Q TARGET
            # -------------------------------------------------

            target_q = rewards + (
                GAMMA
                * best_next_q
                * (1 - dones)
            )


            # -------------------------------------------------
            # LOSS
            # -------------------------------------------------

            loss = loss_function(
                current_q,
                target_q
            )


            # -------------------------------------------------
            # BACKPROPAGATION
            # -------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_loss += loss.item()

            loss_count += 1


    # =====================================================
    # EPSILON DECAY
    # =====================================================

    EPSILON = max(
        EPSILON_MIN,
        EPSILON * EPSILON_DECAY
    )


    # =====================================================
    # SUCCESS CHECK
    # =====================================================

    if env.position == env.goal:

        successful_episodes += 1


    # =====================================================
    # TARGET NETWORK UPDATE
    # =====================================================

    if episode % TARGET_UPDATE == 0:

        target_network.load_state_dict(
            policy_network.state_dict()
        )


    # =====================================================
    # STATISTICS
    # =====================================================

    episode_rewards.append(
        total_reward
    )


    if loss_count > 0:

        average_loss = (
            total_loss / loss_count
        )

    else:

        average_loss = 0


    episode_losses.append(
        average_loss
    )


    # =====================================================
    # PRINT
    # =====================================================

    if episode % 50 == 0:

        success_rate = (
            successful_episodes
            / episode
        ) * 100


        print(
            f"Episode: {episode:4d} | "
            f"Steps: {steps:2d} | "
            f"Reward: {total_reward:7.1f} | "
            f"Loss: {average_loss:.4f} | "
            f"Epsilon: {EPSILON:.3f} | "
            f"Success: {success_rate:6.2f}%"
        )


# =========================================================
# TRAINING COMPLETE
# =========================================================

print("\n")
print("=" * 65)
print("                 TRAINING COMPLETE")
print("=" * 65)

print(
    f"Total Episodes : {EPISODES}"
)

print(
    f"Successful     : {successful_episodes}"
)

print(
    f"Success Rate   : "
    f"{(successful_episodes / EPISODES) * 100:.2f}%"
)

print(
    f"Final Epsilon  : {EPSILON:.3f}"
)


# =========================================================
# TEST TRAINED AGENT
# =========================================================

print("\n")
print("=" * 65)
print("                  TESTING DQN")
print("=" * 65)


state = env.reset()

done = False

test_reward = 0

test_steps = 0

path = [env.position]


while not done:

    test_steps += 1


    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)


    with torch.no_grad():

        q_values = policy_network(
            state_tensor
        )


    action = torch.argmax(
        q_values
    ).item()


    next_state, reward, done = env.step(
        action
    )


    test_reward += reward

    state = next_state

    path.append(
        env.position
    )


    print(
        f"Step: {test_steps:2d} | "
        f"Position: {env.position} | "
        f"Action: {action_names[action]:5s} | "
        f"Reward: {reward:4d} | "
        f"Q-values: "
        f"{q_values.squeeze().numpy()}"
    )


# =========================================================
# FINAL TEST RESULT
# =========================================================

print("\n")
print("=" * 65)

if env.position == env.goal:

    print("             🎯 GOAL REACHED!")

else:

    print("             ❌ GOAL NOT REACHED")

print("=" * 65)

print(
    f"Test Steps  : {test_steps}"
)

print(
    f"Test Reward : {test_reward}"
)

print(
    f"Path        : {path}"
)

print("=" * 65)