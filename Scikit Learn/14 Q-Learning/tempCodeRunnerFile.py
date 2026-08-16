import numpy as np


# =========================================================
# Q-LEARNING PARAMETERS
# =========================================================

TOTAL_STATES = 16
TOTAL_ACTIONS = 4
TOTAL_EPISODES = 1000

ALPHA = 0.1
GAMMA = 0.9

EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05


# Q-Table
Q_TABLE = np.zeros((TOTAL_STATES, TOTAL_ACTIONS))

print("Initial Q-Table:\n")
print(Q_TABLE)


# =========================================================
# GRID WORLD ENVIRONMENT
# =========================================================

class GridWorld:

    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.total_states = self.rows * self.cols
        self.total_actions = 4

        self.start_state = 0
        self.goal_state = 15

        self.state = self.start_state

    # -----------------------------------------------------
    # RESET ENVIRONMENT
    # -----------------------------------------------------

    def reset(self):
        self.state = self.start_state
        return self.state

    # -----------------------------------------------------
    # TAKE ACTION
    # -----------------------------------------------------

    def step(self, action):

        row = self.state // self.cols
        col = self.state % self.cols

        # 0 = Up
        if action == 0:
            row -= 1

        # 1 = Down
        elif action == 1:
            row += 1

        # 2 = Left
        elif action == 2:
            col -= 1

        # 3 = Right
        elif action == 3:
            col += 1

        # Keep agent inside the grid
        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))

        # Convert row and column into state number
        next_state = row * self.cols + col

        # -------------------------------------------------
        # REWARD SYSTEM
        # -------------------------------------------------

        if next_state == self.goal_state:
            reward = 10
            done = True

        else:
            reward = -1
            done = False

        # Update current state
        self.state = next_state

        return next_state, reward, done


# =========================================================
# CREATE ENVIRONMENT
# =========================================================

env = GridWorld()


# =========================================================
# Q-LEARNING TRAINING
# =========================================================

for episode in range(TOTAL_EPISODES):

    # Reset environment
    state = env.reset()

    done = False

    while not done:

        # -------------------------------------------------
        # EXPLORATION VS EXPLOITATION
        # -------------------------------------------------

        if np.random.random() < EPSILON:

            # Exploration
            action = np.random.randint(TOTAL_ACTIONS)

        else:

            # Exploitation
            action = np.argmax(Q_TABLE[state])

        # -------------------------------------------------
        # TAKE ACTION
        # -------------------------------------------------

        next_state, reward, done = env.step(action)

        # -------------------------------------------------
        # Q-LEARNING UPDATE
        # -------------------------------------------------

        old_q = Q_TABLE[state, action]

        best_future_q = np.max(Q_TABLE[next_state])

        target = reward + GAMMA * best_future_q

        new_q = old_q + ALPHA * (target - old_q)

        Q_TABLE[state, action] = new_q

        # Move to next state
        state = next_state

    # -----------------------------------------------------
    # EPSILON DECAY
    # -----------------------------------------------------

    EPSILON = max(
        EPSILON_MIN,
        EPSILON * EPSILON_DECAY
    )


# =========================================================
# FINAL Q-TABLE
# =========================================================

print("\nFinal Q-Table:\n")
print(Q_TABLE)

print("\nTraining Complete!")

print("Final Epsilon:", round(EPSILON, 3))