import numpy as np
from environment import DroneEnv
from q_learning_agent import QLearningAgent
import signal
import sys
import os

def save_and_exit(agent, env, episode, total_rewards, best_q_table, best_reward):
    print(f"Saving Q-table and current episode {episode} and shutting down.")
    agent.save_q_table()
    np.save("../data/last_episode.npy", episode)
    np.save("../data/total_rewards.npy", total_rewards)
    np.save("../data/best_q_table.npy", best_q_table)
    np.save("../data/best_reward.npy", best_reward)
    env.close()
    sys.exit(0)

def train_agent(episodes):
    env = DroneEnv()
    agent = QLearningAgent(env.state_space, env.action_space)
    
    start_episode = 0
    total_rewards = []
    best_q_table = None
    best_reward = -np.inf

    if os.path.exists("../data/last_episode.npy"):
        start_episode = int(np.load("../data/last_episode.npy"))
        total_rewards = list(np.load("../data/total_rewards.npy"))
        best_q_table = np.load("../data/best_q_table.npy")
        best_reward = float(np.load("../data/best_reward.npy"))
        print(f"Resuming training from episode {start_episode}")

    # Register the signal handler
    signal.signal(signal.SIGINT, lambda sig, frame: save_and_exit(agent, env, start_episode, total_rewards, best_q_table, best_reward))

    try:
        for episode in range(start_episode, episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward

                if done:
                    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
                    break

            total_rewards.append(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward
                best_q_table = np.copy(agent.q_table)

            # Save the Q-table and current episode periodically
            if episode % 10 == 0:
                agent.save_q_table()
                np.save("../data/last_episode.npy", episode)
                np.save("../data/total_rewards.npy", total_rewards)
                np.save("../data/best_q_table.npy", best_q_table)
                np.save("../data/best_reward.npy", best_reward)

            start_episode = episode

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        save_and_exit(agent, env, episode, total_rewards, best_q_table, best_reward)

    agent.save_q_table()
    np.save("../data/last_episode.npy", episodes)
    np.save("../data/total_rewards.npy", total_rewards)
    np.save("../data/best_q_table.npy", best_q_table)
    np.save("../data/best_reward.npy", best_reward)
    env.close()

    # Find the episode with the highest reward
    max_reward_episode = np.argmax(total_rewards)
    print(f"The episode with the highest reward is Episode {max_reward_episode + 1} with a reward of {total_rewards[max_reward_episode]}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    train_agent(episodes=100)
