from Agent import *
import numpy as np
from environment import *
import matplotlib.pyplot as plt
from config import *

def compute_moving_avg(rewards, window=100):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def run_experiments(env_name, algo_fn, num_runs=5, episodes=RUNS, steps=NUM_STEPS, **kwargs):
    all_rewards = []
    
    for run in range(num_runs):
        print(run)
        np.random.seed(run)  
        env = Environment(env_name)
        agent = Agent(env)
        rewards = algo_fn(agent, episodes=episodes, steps=steps, **kwargs)
        all_rewards.append(compute_moving_avg(rewards)) 
    
    return np.array(all_rewards) 

def compute_stats(all_rewards):
    mean_rewards = np.mean(all_rewards, axis=0)  
    std_rewards = np.std(all_rewards, axis=0)    
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
    env_name = 'CartPole-v1'
    sarsa_rewards = run_experiments(env_name, lambda agent, **kwargs: agent.train_Sarsa(
        policy=EpsilonGreedy(agent.env.action_space),eps_start=EPS_START, eps_end=EPS_END,**kwargs),episodes=RUNS,steps=NUM_STEPS)
    
    qlearning_rewards = run_experiments(env_name, lambda agent, **kwargs: agent.train_Qlearnig(
        policy=Softmax(agent.env.action_space), tau_start=TAU_START, tau_end=TAU_END, **kwargs),episodes=RUNS,steps=NUM_STEPS)
    sarsa_mean, sarsa_std = compute_stats(sarsa_rewards)
    qlearning_mean, qlearning_std = compute_stats(qlearning_rewards)
    plot_results(sarsa_mean, sarsa_std, qlearning_mean, qlearning_std, env_name)