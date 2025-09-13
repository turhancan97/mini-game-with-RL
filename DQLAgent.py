from collections import deque
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class DQLAgent:
    def __init__(self, num_enemies):
        # parameter / hyperparameter
        self.num_enemies = num_enemies
        self.state_size = 2 * self.num_enemies
        self.action_size = 3  # right, left, no move

        self.gamma = 0.95
        self.learning_rate = 0.001  # 0.001

        self.epsilon = 1  # explore
        self.epsilon_decay = 0.995  # 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=1000)
        self.q_values = []
        self.model = self.build_model()

    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation="relu"))
        # model.add(Dropout(0.2))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(16, activation='tanh'))
        # model.add(Dropout(0.2))
        # model.add(Dense(16, activation='tanh'))
        # model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # merge q_values with state
        q_values_for_state = np.append(act_values, state)
        self.q_values.append(q_values_for_state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            train_target = self.model.predict(state)
            train_target[0][action] = target

            # It's more efficient to use fit() in batches, but for this example, keeping the original structure:
            # Added epochs for clarity, even though it defaults to 1
            self.model.fit(state, train_target, verbose=0, epochs=1)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
