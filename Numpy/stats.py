from Agent import *
import numpy as np
from environment import *
import matplotlib.pyplot as plt

def compute_moving_avg(rewards, window=100):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def run_experiments(env_name, algo_fn, num_runs=5, episodes=10000, steps=500, **kwargs):
    all_rewards = []
    
    for run in range(num_runs):
        print(run)
        np.random.seed(run)  # Set different seeds
        env = Environment(env_name)
        agent = Agent(env)
        rewards = algo_fn(agent, episodes=episodes, steps=steps, **kwargs)
        all_rewards.append(compute_moving_avg(rewards))  # Apply moving average
    
    return np.array(all_rewards)  # Shape: (num_runs, episodes-window+1)

def compute_stats(all_rewards):
    mean_rewards = np.mean(all_rewards, axis=0)  # Average over runs
    std_rewards = np.std(all_rewards, axis=0)    # Compute standard deviation
    return mean_rewards, std_rewards

def plot_results(mean_sarsa, std_sarsa, mean_q, std_q, env_name):
    plt.figure(figsize=(10,6))
    
    plt.plot(mean_sarsa, label="SARSA (Mean Reward, Moving Avg)", color="blue")
    plt.fill_between(range(len(mean_sarsa)), mean_sarsa - std_sarsa, mean_sarsa + std_sarsa, alpha=0.3, color="blue")
    
    plt.plot(mean_q, label="Q-Learning (Mean Reward, Moving Avg)", color="red")
    plt.fill_between(range(len(mean_q)), mean_q - std_q, mean_q + std_q, alpha=0.3, color="red")
    
    plt.title(f'Comparison of SARSA and Q-Learning in {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Episodic Return (Moving Avg)')
    plt.legend()
    plt.savefig(f'{env_name}_comparison.png')
    plt.show()

if __name__ == '__main__':
    # CartPole
    env_name = 'CartPole-v1'
    sarsa_rewards = run_experiments(env_name, lambda agent, **kwargs: agent.train_Sarsa(
        policy=EpsilonGreedy(agent.env.action_space),eps_start=1, eps_end=0.01,eps_decay=0.9995, **kwargs),episodes=10000,steps=500)
    
    qlearning_rewards = run_experiments(env_name, lambda agent, **kwargs: agent.train_Qlearnig(
        policy=Softmax(agent.env.action_space), tau_start=2, tau_end=0.1,tau_decay=0.9995, **kwargs),episodes=10000,steps=500)
    sarsa_mean, sarsa_std = compute_stats(sarsa_rewards)
    qlearning_mean, qlearning_std = compute_stats(qlearning_rewards)
    plot_results(sarsa_mean, sarsa_std, qlearning_mean, qlearning_std, env_name)
    
    # MountainCar
    env_name = 'MountainCar-v0'
    sarsa_rewards = run_experiments(env_name, lambda agent, **kwargs: agent.train_Sarsa(
        policy=EpsilonGreedy(agent.env.action_space), eps_start=1, eps_end=0.05, eps_decay=0.9997, **kwargs),episodes=20000,steps=500)
    qlearning_rewards = run_experiments(env_name, lambda agent, **kwargs: agent.train_Qlearnig(
        policy=Softmax(agent.env.action_space), tau_start=2, tau_end=0.1, tau_decay=0.9995, **kwargs),episodes=20000,steps=500)
    sarsa_mean, sarsa_std = compute_stats(sarsa_rewards)
    qlearning_mean, qlearning_std = compute_stats(qlearning_rewards)
    plot_results(sarsa_mean, sarsa_std, qlearning_mean, qlearning_std, env_name)
