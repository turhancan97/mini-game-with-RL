# %%
"""
RL GAME plot - Two cases: eat and avoid and two observation types: vector and pixels

Run examples:
  python plot.py --scenario eat --episode 60 --num-enemies 4 --obs-type pixels
  python plot.py --scenario avoid --episodes 60 --num-enemies 8 --obs-type pixels
"""
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from scipy.interpolate import griddata
EPISODES = 2000

# %% CLI args and file resolution
parser = argparse.ArgumentParser(description="Plot training results for a scenario.")
parser.add_argument("--scenario", choices=["eat", "avoid"], default="eat", help="Scenario to plot")
parser.add_argument("--obs-type", choices=["vector", "pixels"], default="vector", help="Observation type: low-dim vector or raw pixels")
parser.add_argument("--episodes", type=int, default=None, help="Episode window to plot (optional)")
parser.add_argument("--num-enemies", type=int, default=3, help="Number of enemies to plot")
args = parser.parse_args()

def resolve_candidates(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate files exist: {candidates}")

# Prefer obs-type + scenario; fall back to scenario-only; then legacy
reward_path = resolve_candidates([
    f"plot/reward_history_{args.obs_type}_{args.scenario}.pickle",
    f"plot/reward_history_{args.scenario}.pickle",
    "plot/reward_history.pickle",
])
qvalues_path = resolve_candidates([
    f"plot/q_values_{args.obs_type}_{args.scenario}.pickle",
    f"plot/q_values_{args.scenario}.pickle",
    "plot/q_values.pickle",
])

# %% Read the data from pickle files
with open(reward_path, "rb") as f:
    reward_history = pickle.load(f)
with open(qvalues_path, "rb") as f:
    q_values = pickle.load(f)

# Determine EPISODES window
if args.episodes is not None:
    EPISODES = args.episodes

# %% Plot the reward history
# 5 episode moving average
mov_average = 100
reward_history_arr = np.array(reward_history, dtype=float)
if len(reward_history_arr) >= mov_average:
    reward_history_av = np.convolve(reward_history_arr, np.ones(mov_average), 'valid') / mov_average
    plt.plot(reward_history_av)
else:
    plt.plot(reward_history_arr)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward vs. Episode')
plt.show()

# %% Plot the heatmap for state vs Q-values
# Reshaping and stacking the q_values for heatmap
if args.obs_type == "pixels":
    # Pixels: q_values entries are per-action arrays/lists; no state features.
    try:
        q_values_list = [list(item)[:3] for item in q_values]
    except Exception:
        q_values_list = [np.array(item).ravel()[:3].tolist() for item in q_values]
else:
    # Vector: entries contain [q0,q1,q2, state...]
    q_values_list = []
    state_values_list = []
    for item in q_values:
        arr = np.array(item).ravel()
        if arr.size >= 3:
            q_values_list.append(arr[:3].tolist())
            state_values_list.append(tuple(arr[3:].tolist()))
    if state_values_list:
        # Heatmap for state vs Q-values
        state_to_qvalues_dict = {state: q for state, q in zip(state_values_list, q_values_list)}
        heatmap_data = list(state_to_qvalues_dict.values())
        window = max(1, int(EPISODES / 10))
        heatmap_data_last = heatmap_data[-window:]
        state_values_list_last = state_values_list[-window:]
        plt.figure(figsize=(10, 10))
        try:
            sns.heatmap(heatmap_data_last, annot=True, cmap='coolwarm', yticklabels=state_values_list_last)
        except Exception:
            sns.heatmap(heatmap_data_last, cmap='coolwarm', yticklabels=False)
        plt.xlabel('Actions')
        plt.ylabel('States')
        plt.title('State vs Q-values Heatmap')
        plt.show()
# %% Plot the Q-values over time
# Extracting Q-values for each action over time
q_values_action_1 = [item[0] for item in q_values_list if len(item) >= 3]
q_values_action_2 = [item[1] for item in q_values_list if len(item) >= 3]
q_values_action_3 = [item[2] for item in q_values_list if len(item) >= 3]

# Plotting Q-values over time
plt.figure(figsize=(12, 8))
plt.plot(q_values_action_1, label="Action 1", marker='o')
plt.plot(q_values_action_2, label="Action 2", marker='o')
plt.plot(q_values_action_3, label="Action 3", marker='o')
plt.xlabel('Time Step or Episode')
plt.ylabel('Q-value')
plt.title('Q-value Evolution Over Time for Each Action')
plt.legend()
plt.grid(True)
plt.show()

# %% Plot the Q-value surface plot
# Extracting state values
if args.obs_type == "vector" and 'state_values_list' in locals() and state_values_list:
    for enemy_number in range(args.num_enemies):
        q_values_action_3_last = q_values_action_3[-EPISODES:]
        state_x_values = [item[(enemy_number * 2)] for item in state_values_list][-EPISODES:]
        state_y_values = [item[(enemy_number * 2) + 1] for item in state_values_list][-EPISODES:]
        if not state_x_values or not state_y_values:
            continue
        x = np.linspace(min(state_x_values), max(state_x_values), len(state_x_values))
        y = np.linspace(min(state_y_values), max(state_y_values), len(state_y_values))
        X, Y = np.meshgrid(x, y)
        try:
            Z = griddata((state_x_values, state_y_values), q_values_action_3_last, (X, Y), method='cubic')
        except Exception:
            continue
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
        fig.colorbar(surf)
        plt.xlabel('State Dimension 1')
        plt.ylabel('State Dimension 2')
        ax.set_zlabel('Q-value for Action 3')
        plt.title(f"{enemy_number + 1}th Enemy's Q-value Surface Plot for Action 3")
        plt.show()

# %%
