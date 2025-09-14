from collections import deque
import random
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

class DQLAgent:
    def __init__(self, num_enemies):
        # parameter / hyperparameter
        self.num_enemies = num_enemies
        self.state_size = 2 * self.num_enemies
        self.action_size = 3  # right, left, no move

        self.gamma = 0.95
        self.learning_rate = 0.001  # 0.001

        # Epsilon-greedy: linear decay per episode
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.eps_linear_episodes = 750
        self.eps_linear_delta = (1.0 - self.epsilon_min) / max(1, self.eps_linear_episodes)

        # Replay buffer
        self.memory_size = 50000
        self.memory = deque(maxlen=self.memory_size)
        self.train_start = 2000
        self.updates_per_step = 2
        self.q_values = []
        self.model = self.build_model()
        # Target network for stability
        self.target_model = self.build_model()
        self.update_target_model()
        # Soft update rate for target network
        self.tau = 0.005
        self.train_steps = 0

    def build_model(self):
        # neural network for deep q learning (wider MLP; Huber loss)
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def soft_update_target(self):
        main_w = self.model.get_weights()
        targ_w = self.target_model.get_weights()
        new_w = [self.tau * mw + (1.0 - self.tau) * tw for mw, tw in zip(main_w, targ_w)]
        self.target_model.set_weights(new_w)

    def load_model(self, model_path):
        self.model = load_model(model_path)
        # keep target in sync
        self.update_target_model()

    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        # merge q_values with state for plotting
        q_values_for_state = np.append(act_values, state)
        self.q_values.append(q_values_for_state)
        if len(self.q_values) > 10000:
            self.q_values = self.q_values[-10000:]
        return int(np.argmax(act_values[0]))

    def replay(self, batch_size):
        # training with batch updates and target network
        if len(self.memory) < max(batch_size, self.train_start):
            return

        for _ in range(self.updates_per_step):
            minibatch = random.sample(self.memory, batch_size)

            states = np.array([np.array(s).reshape(-1) for (s, _, _, _, _) in minibatch])
            next_states = np.array([np.array(ns).reshape(-1) for (_, _, _, ns, _) in minibatch])
            actions = np.array([a for (_, a, _, _, _) in minibatch])
            rewards = np.array([r for (_, _, r, _, _) in minibatch])
            dones = np.array([d for (_, _, _, _, d) in minibatch]).astype(np.float32)

            # Current and next Q-values
            q_values_curr = self.model.predict(states, verbose=0)
            q_values_next = self.target_model.predict(next_states, verbose=0)

            # Targets: r + gamma * max_a' Q_target(s', a') for non-terminal; r for terminal
            max_next = np.max(q_values_next, axis=1)
            targets = q_values_curr.copy()
            updated = rewards + (1 - dones) * (self.gamma * max_next)
            for i, a in enumerate(actions):
                targets[i, a] = updated[i]

            self.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0)

            # Soft update target after each gradient step
            self.soft_update_target()

            self.train_steps += 1

    def adaptiveEGreedy(self):
        # Linear decay per episode
        self.epsilon = max(self.epsilon_min, self.epsilon - self.eps_linear_delta)


class CNNDQLAgent:
    def __init__(self, num_enemies):
        # Hyperparameters
        self.num_enemies = num_enemies
        self.state_shape = (84, 84, 4)
        self.action_size = 3
        self.gamma = 0.95
        self.learning_rate = 0.00025

        # Epsilon-greedy linear decay per episode
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.eps_linear_episodes = 750
        self.eps_linear_delta = (1.0 - self.epsilon_min) / max(1, self.eps_linear_episodes)

        # Replay buffer
        self.memory_size = 100000
        self.memory = deque(maxlen=self.memory_size)
        self.train_start = 5000
        self.updates_per_step = 2

        self.q_values = []  # keep lightweight or empty for CNN
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.tau = 0.005
        self.train_steps = 0

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=4, activation="relu", input_shape=self.state_shape))
        model.add(Conv2D(32, (4, 4), strides=2, activation="relu"))
        # model.add(Conv2D(64, (3, 3), strides=1, activation="relu"))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def soft_update_target(self):
        main_w = self.model.get_weights()
        targ_w = self.target_model.get_weights()
        new_w = [self.tau * mw + (1.0 - self.tau) * tw for mw, tw in zip(main_w, targ_w)]
        self.target_model.set_weights(new_w)

    def load_model(self, model_path):
        self.model = load_model(model_path)
        self.update_target_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state, verbose=0)
        # Optionally store only Qs to keep memory small
        self.q_values.append(q[0].tolist())
        if len(self.q_values) > 10000:
            self.q_values = self.q_values[-10000:]
        return int(np.argmax(q[0]))

    def replay(self, batch_size):
        if len(self.memory) < max(batch_size, self.train_start):
            return
        for _ in range(self.updates_per_step):
            minibatch = random.sample(self.memory, batch_size)

            # states are stored with a leading batch dim (1, H, W, C); stack them
            states = np.concatenate([s for (s, _, _, _, _) in minibatch], axis=0)
            next_states = np.concatenate([ns for (_, _, _, ns, _) in minibatch], axis=0)
            actions = np.array([a for (_, a, _, _, _) in minibatch])
            rewards = np.array([r for (_, _, r, _, _) in minibatch])
            dones = np.array([d for (_, _, _, _, d) in minibatch]).astype(np.float32)

            q_curr = self.model.predict(states, verbose=0)
            q_next = self.target_model.predict(next_states, verbose=0)
            max_next = np.max(q_next, axis=1)

            targets = q_curr.copy()
            updated = rewards + (1 - dones) * (self.gamma * max_next)
            for i, a in enumerate(actions):
                targets[i, a] = updated[i]

            self.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0)
            self.soft_update_target()
            self.train_steps += 1

    def adaptiveEGreedy(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.eps_linear_delta)
