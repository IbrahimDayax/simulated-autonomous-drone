import numpy as np
import random
import os

class QLearningAgent:
    def __init__(self, state_size, action_size, q_table_file="../data/q_table.npy"):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.q_table_file = q_table_file
        self.load_q_table()
        self.learning_rate = 0.1
        self.discount_rate = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_rate * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self):
        np.save(self.q_table_file, self.q_table)

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            self.q_table = np.load(self.q_table_file)
