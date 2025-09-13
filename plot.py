# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from scipy.interpolate import griddata
EPISODES = 2000
# %% Read the data from pickle files
# Open pickle file and load data
with open("plot/reward_history.pickle", "rb") as f:
    reward_history = pickle.load(f)

# Open q_values pickle file and load data
with open("plot/q_values.pickle", "rb") as f:
    q_values = pickle.load(f)

# %% Plot the reward history
# 5 episode moving average
mov_average = 25
reward_history_av = np.array(reward_history)
reward_history_av = np.convolve(reward_history_av, np.ones(mov_average), 'valid') / mov_average
plt.plot(reward_history_av)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward vs. Episode')
plt.show()

# %% Plot the heatmap for state vs Q-values
# Reshaping and stacking the q_values for heatmap
# Splitting Q-values and state values
q_values_list = [item[:3] for item in q_values]
state_values_list = [tuple(item[3:]) for item in q_values]

# Creating a dictionary with state values as keys and Q-values as values
state_to_qvalues_dict = {state: q_values for state, q_values in zip(state_values_list, q_values_list)}

# For the heatmap, we only need the Q-values
heatmap_data = list(state_to_qvalues_dict.values())
heatmap_data_last = heatmap_data[-int(EPISODES / 10):]
state_values_list_last = state_values_list[-int(EPISODES / 10):]
# Plotting the heatmap for state vs Q-values
plt.figure(figsize=(10, 10))
sns.heatmap(heatmap_data_last, annot=True, cmap='coolwarm', yticklabels=state_values_list_last)
plt.xlabel('Actions')
plt.ylabel('States')
plt.title('State vs Q-values Heatmap')
plt.show()
# %% Plot the Q-values over time
# Extracting Q-values for each action over time
q_values_action_1 = [item[0] for item in q_values_list]
q_values_action_2 = [item[1] for item in q_values_list]
q_values_action_3 = [item[2] for item in q_values_list]

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
for enemy_number in range(3):
    q_values_action_3 = q_values_action_3[-EPISODES:]
    state_x_values = [item[(enemy_number * 2)] for item in state_values_list][-EPISODES:]
    state_y_values = [item[(enemy_number * 2) + 1] for item in state_values_list][-EPISODES:]
    # Creating a grid of state values
    x = np.linspace(min(state_x_values), max(state_x_values), len(state_x_values))
    y = np.linspace(min(state_y_values), max(state_y_values), len(state_y_values))

    X, Y = np.meshgrid(x, y)

    # Using Q-values of the first action as an example
    Z = griddata((state_x_values, state_y_values), q_values_action_3, (X, Y), method='cubic')

    # Plotting the surface plot
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
