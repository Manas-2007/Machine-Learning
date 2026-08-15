import numpy as np
import matplotlib.pyplot as plt
import random

# =========================================================
# ENVIRONMENT
# =========================================================

ROWS = 5
COLS = 5

START = (0, 0)
GOAL = (4, 4)

OBSTACLES = {
    (1, 1),
    (1, 2),
    (2, 2),
    (3, 2),
    (3, 3)
}

ACTIONS = {
    0: (-1, 0),   # Up
    1: (1, 0),    # Down
    2: (0, -1),   # Left
    3: (0, 1)     # Right
}

ACTION_NAMES = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right"
}

# =========================================================
# Q-LEARNING PARAMETERS
# =========================================================

alpha = 0.1
gamma = 0.9
epsilon = 1.0

epsilon_decay = 0.995
epsilon_min = 0.05

episodes = 10000
max_steps = 100

# 25 states × 4 actions
q_table = np.zeros((ROWS * COLS, 4))

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def state_to_number(state):
    row, col = state
    return row * COLS + col


def number_to_state(number):
    return divmod(number, COLS)


def get_reward(next_state):

    if next_state == GOAL:
        return 100

    if next_state in OBSTACLES:
        return -10

    return -1


def take_action(state, action):

    row, col = state

    move_row, move_col = ACTIONS[action]

    new_row = row + move_row
    new_col = col + move_col

    # Outside grid
    if new_row < 0 or new_row >= ROWS:
        return state, -10, False

    if new_col < 0 or new_col >= COLS:
        return state, -10, False

    next_state = (new_row, new_col)

    # Obstacle
    if next_state in OBSTACLES:
        return state, -10, False

    reward = get_reward(next_state)

    done = next_state == GOAL

    return next_state, reward, done


# =========================================================
# TRAINING
# =========================================================

episode_rewards = []

print("\n================ Q-LEARNING TRAINING ================\n")

for episode in range(episodes):

    state = START
    total_reward = 0

    for step in range(max_steps):

        state_number = state_to_number(state)

        # Exploration vs Exploitation
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state_number])

        next_state, reward, done = take_action(state, action)

        next_state_number = state_to_number(next_state)

        # Q-Learning update
        old_q_value = q_table[state_number, action]

        best_future_q = np.max(q_table[next_state_number])

        new_q_value = old_q_value + alpha * (
            reward + gamma * best_future_q - old_q_value
        )

        q_table[state_number, action] = new_q_value

        state = next_state

        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {total_reward:4d} | "
            f"Epsilon: {epsilon:.3f}"
        )


# =========================================================
# DISPLAY Q-TABLE
# =========================================================

print("\n================ LEARNED Q-TABLE ================\n")

for state_number in range(ROWS * COLS):

    state = number_to_state(state_number)

    if state in OBSTACLES:
        continue

    values = q_table[state_number]

    best_action = np.argmax(values)

    print(
        f"State {state} | "
        f"Up: {values[0]:7.2f} | "
        f"Down: {values[1]:7.2f} | "
        f"Left: {values[2]:7.2f} | "
        f"Right: {values[3]:7.2f} | "
        f"Best: {ACTION_NAMES[best_action]}"
    )


# =========================================================
# TEST TRAINED AGENT
# =========================================================

print("\n================ TRAINED AGENT PATH ================\n")

state = START
path = [state]

for step in range(max_steps):

    state_number = state_to_number(state)

    action = np.argmax(q_table[state_number])

    next_state, reward, done = take_action(state, action)

    print(
        f"Step {step + 1:2d} | "
        f"State: {state} | "
        f"Action: {ACTION_NAMES[action]:7s} | "
        f"Reward: {reward:4d}"
    )

    state = next_state
    path.append(state)

    if done:
        print("\n🎯 GOAL REACHED!")
        break


# =========================================================
# DISPLAY GRID
# =========================================================

print("\n================ LEARNED PATH ON GRID ================\n")

for row in range(ROWS):

    line = ""

    for col in range(COLS):

        position = (row, col)

        if position == START:
            symbol = " S "

        elif position == GOAL:
            symbol = " G "

        elif position in OBSTACLES:
            symbol = " X "

        elif position in path:
            symbol = " * "

        else:
            symbol = " . "

        line += symbol

    print(line)


# =========================================================
# REWARD GRAPH
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(episode_rewards)

plt.title("Q-Learning Training Progress")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.grid(alpha=0.3)

plt.show()


# =========================================================
# FINAL INFORMATION
# =========================================================

print("\n================ TRAINING COMPLETE ================\n")

print("Total Episodes :", episodes)
print("Learning Rate  :", alpha)
print("Discount Factor:", gamma)
print("Final Epsilon  :", round(epsilon, 3))

print("\nStart :", START)
print("Goal  :", GOAL)

print("\nFinal Path:")
print(path)