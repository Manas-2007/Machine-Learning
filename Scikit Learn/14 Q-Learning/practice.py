import numpy as np
import random
import matplotlib.pyplot as plt


# =========================================================
# TRAFFIC ENVIRONMENT
# =========================================================

TRAFFIC_LEVELS = 5

ACTIONS = {
    0: "Keep Current Phase",
    1: "North-South Green",
    2: "East-West Green"
}

START_STATE = 2

alpha = 0.1
gamma = 0.9

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

episodes = 1000
max_steps = 30

q_table = np.zeros((TRAFFIC_LEVELS, len(ACTIONS)))


# =========================================================
# REALISTIC TRAFFIC PATTERNS
# =========================================================

traffic_patterns = [
    {
        "name": "Morning Rush",
        "ns_traffic": 85,
        "ew_traffic": 55
    },
    {
        "name": "Office Hours",
        "ns_traffic": 60,
        "ew_traffic": 50
    },
    {
        "name": "Evening Rush",
        "ns_traffic": 50,
        "ew_traffic": 90
    },
    {
        "name": "Night",
        "ns_traffic": 20,
        "ew_traffic": 15
    },
    {
        "name": "Weekend",
        "ns_traffic": 40,
        "ew_traffic": 45
    }
]


# =========================================================
# CONVERT TRAFFIC INTO STATE
# =========================================================

def get_state(total_traffic):

    if total_traffic < 30:
        return 0

    elif total_traffic < 60:
        return 1

    elif total_traffic < 100:
        return 2

    elif total_traffic < 140:
        return 3

    else:
        return 4


# =========================================================
# ENVIRONMENT STEP
# =========================================================

def environment_step(state, action, ns_traffic, ew_traffic):

    if action == 1:

        ns_traffic -= random.randint(8, 15)
        ew_traffic += random.randint(2, 6)

    elif action == 2:

        ew_traffic -= random.randint(8, 15)
        ns_traffic += random.randint(2, 6)

    else:

        ns_traffic += random.randint(-3, 5)
        ew_traffic += random.randint(-3, 5)

    ns_traffic = max(0, ns_traffic)
    ew_traffic = max(0, ew_traffic)

    total_traffic = ns_traffic + ew_traffic

    next_state = get_state(total_traffic)

    if total_traffic < 50:
        reward = 10

    elif total_traffic < 80:
        reward = 5

    elif total_traffic < 120:
        reward = 0

    elif total_traffic < 160:
        reward = -5

    else:
        reward = -10

    return next_state, reward, ns_traffic, ew_traffic


# =========================================================
# TRAINING
# =========================================================

episode_rewards = []

print("\n================ TRAFFIC RL TRAINING ================\n")

for episode in range(episodes):

    pattern = random.choice(traffic_patterns)

    ns_traffic = pattern["ns_traffic"] + random.randint(-10, 10)
    ew_traffic = pattern["ew_traffic"] + random.randint(-10, 10)

    state = get_state(ns_traffic + ew_traffic)

    total_reward = 0

    for step in range(max_steps):

        if random.random() < epsilon:

            action = random.randint(0, 2)

        else:

            action = np.argmax(q_table[state])

        next_state, reward, ns_traffic, ew_traffic = environment_step(
            state,
            action,
            ns_traffic,
            ew_traffic
        )

        old_q = q_table[state, action]

        best_future_q = np.max(q_table[next_state])

        new_q = old_q + alpha * (
            reward + gamma * best_future_q - old_q
        )

        q_table[state, action] = new_q

        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {total_reward:4d} | "
            f"Epsilon: {epsilon:.3f}"
        )


# =========================================================
# LEARNED POLICY
# =========================================================

print("\n================ LEARNED TRAFFIC POLICY ================\n")

traffic_names = {
    0: "Very Low",
    1: "Low",
    2: "Medium",
    3: "High
    4: "Very High"
}

for state in range(TRAFFIC_LEVELS):

    best_action = np.argmax(q_table[state])

    print(
        f"Traffic Level: {traffic_names[state]:10s} | "
        f"Keep: {q_table[state, 0]:7.2f} | "
        f"NS Green: {q_table[state, 1]:7.2f} | "
        f"EW Green: {q_table[state, 2]:7.2f} | "
        f"Best Action: {ACTIONS[best_action]}"
    )


# =========================================================
# TEST THE TRAINED AGENT
# =========================================================

print("\n================ TESTING TRAINED AGENT ================\n")

for pattern in traffic_patterns:

    ns_traffic = pattern["ns_traffic"]
    ew_traffic = pattern["ew_traffic"]

    total_traffic = ns_traffic + ew_traffic

    state = get_state(total_traffic)

    action = np.argmax(q_table[state])

    print(
        f"{pattern['name']:15s} | "
        f"NS Traffic: {ns_traffic:3d} | "
        f"EW Traffic: {ew_traffic:3d} | "
        f"State: {traffic_names[state]:10s} | "
        f"Decision: {ACTIONS[action]}"
    )


# =========================================================
# FINAL Q-TABLE
# =========================================================

print("\n================ FINAL Q-TABLE ================\n")

print(
    "              Keep Phase     NS Green      EW Green"
)

for state in range(TRAFFIC_LEVELS):

    print(
        f"{traffic_names[state]:10s} "
        f"{q_table[state, 0]:12.2f} "
        f"{q_table[state, 1]:12.2f} "
        f"{q_table[state, 2]:12.2f}"
    )


# =========================================================
# TRAINING GRAPH
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(episode_rewards)

plt.title("Traffic Signal Q-Learning Training")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.grid(alpha=0.3)

plt.show()