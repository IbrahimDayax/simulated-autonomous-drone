import numpy as np
from environment import DroneEnv
from q_learning_agent import QLearningAgent

def test_agent(episodes):
    env = DroneEnv()
    agent = QLearningAgent(env.state_space, env.action_space)
    
    # Load the best Q-table
    agent.q_table = np.load("../data/best_q_table.npy")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(agent.q_table[state])
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward

            if done:
                print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")
                break

    env.close()

if __name__ == "__main__":
    test_agent(episodes=1)
