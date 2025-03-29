import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def discretize_state(state):
    position = np.round(state[0], decimals=1)
    velocity = np.round(state[1], decimals=2)
    
    position_bins = np.linspace(-1.2, 0.6, 20)
    velocity_bins = np.linspace(-0.07, 0.07, 20)
    
    position_index = np.digitize(position, position_bins) - 1
    velocity_index = np.digitize(velocity, velocity_bins) - 1
    
    return position_index, velocity_index

def softmax(q_values, temp=0.4):
    temp = max(temp, 1e-3)  # Prevent division by zero
    logits = np.exp((q_values - np.max(q_values)) / temp)  # Stable softmax
    probs = logits / np.clip(np.sum(logits), 1e-6, None)  # Normalize
    return np.random.choice(len(q_values), p=probs)

def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, tau=1.0, tau_decay=0.995, tau_min=0.01):
    q_table = np.zeros((20, 20, env.action_space.n))
    rewards_per_episode = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        position_index, velocity_index = discretize_state(state)
        
        episode_reward = 0
        episode_length = 0
        tau = max(tau_min, tau * tau_decay)  # Decay temperature
        
        for step in range(200):
            action = softmax(q_table[position_index, velocity_index], tau)
            next_state, reward, done, _, _ = env.step(action)
            
            next_position_index, next_velocity_index = discretize_state(next_state)
            best_next_action = softmax(q_table[next_position_index, next_velocity_index], tau)
            
            q_table[position_index, velocity_index, action] += alpha * (
                reward + gamma * np.max(q_table[next_position_index, next_velocity_index]) -
                q_table[position_index, velocity_index, action]
            )
            
            position_index, velocity_index = next_position_index, next_velocity_index
            episode_reward += reward
            episode_length += 1
            if done:
                break
        
        rewards_per_episode.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Length: {episode_length}")
    
    return rewards_per_episode, episode_lengths, q_table

# Run Q-learning with Softmax policy
env = gym.make('MountainCar-v0')
rewards, episode_lengths, q_table = q_learning(env)
env.close()

plt.figure(figsize=(12,6))
plt.subplot(2, 1, 1)
plt.plot(rewards)
plt.title('Rewards per Episode (Q-learning with Softmax)')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(2, 1, 2)
plt.plot(episode_lengths)
plt.title('Episode Length per Episode (Q-learning with Softmax)')
plt.xlabel('Episode')
plt.ylabel('Steps Taken')

plt.tight_layout()
plt.show()

def test_agent(env, q_table):
    env = gym.make("MountainCar-v0", render_mode='human')
    state, _ = env.reset()
    position_index, velocity_index = discretize_state(state)
    
    done = False
    episode_reward = 0
    episode_length = 0
    
    while not done:
        action = np.argmax(q_table[position_index, velocity_index])
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        episode_length += 1
        
        position_index, velocity_index = discretize_state(next_state)
        done = terminated or truncated
    
    print(f"Test Episode Reward: {episode_reward}, Steps Taken: {episode_length}")

# Run the test
test_agent(env, q_table)