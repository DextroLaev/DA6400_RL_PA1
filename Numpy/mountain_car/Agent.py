import numpy as np
from environment import *
from Policy import *
import tqdm 
import matplotlib.pyplot as plt
import wandb
from config import *

# Sarsa -> Epsilon-Greedy Policy
# Q-Learning -> Softmax Policy

def moving_avg(reward,WindowSize=100):
    return np.convolve(reward,np.ones(WindowSize)/WindowSize,mode='valid')

def plot_reward_curve(reward,WindowSize=100):
    plt.plot(reward,label='actual reward')
    plt.plot(moving_avg(reward,WindowSize),label='moving avg')    
    plt.legend()
    plt.show()

def comparison_plot(sarsa_reward,qlearning_reward,windowSize=100,plot_name=None):
    plt.plot(moving_avg(sarsa_reward,windowSize),label='sarsa moving avg')
    plt.plot(moving_avg(qlearning_reward,windowSize),label='qlearning moving avg')
    plt.legend()
    plt.title("{}".format(plot_name))
    plt.xlabel("Episodes")
    plt.ylabel("Penalty")
    plt.savefig('{}.png'.format(plot_name))
    plt.show()

import numpy as np

class Agent:
    def __init__(self,env):
        self.env = env
        self.Q = self.env.Q.copy()

    def train_Sarsa(self,policy,episodes = 40000,steps=200,eps_start=0.8,eps_end=0.01,alpha=0.1,gamma=0.99):
        reward_per_eps = []
        eps = eps_start
        print("Training using Sarsa...")
        for ep in (range(episodes)):
            state,_ = self.env.reset()
            done = False
            action = policy.choose(self.Q,state,eps)
            eps = max(eps_end,eps_start * (1 - ep/(episodes*0.5)))
            total_reward = 0

            for step in range(steps):
                next_state,reward,done,_ = self.env.step(action)                
                next_action = policy.choose(self.Q,next_state,eps)
                self.Q[state][action] = self.Q[state][action] + alpha*(reward + gamma*self.Q[next_state][next_action] - self.Q[state][action])
                state = next_state
                action = next_action
                total_reward += reward
                if done:
                    break
                
            if ep%1000 == 0:
                print("Episode:",ep,"Reward:- ",total_reward)        
            
            reward_per_eps.append(total_reward)
        return reward_per_eps

    def train_Qlearnig(self,policy,episodes = 40000,steps=200,tau_start=2,tau_end=0.1,alpha=0.1,gamma=0.99):
        reward_per_eps = []
        tau = tau_start
        print("Training using Qlearning...")
        for ep in (range(episodes)):
            state,_ = self.env.reset()
            done = False
            action = policy.choose(self.Q,state,tau)
            tau = max(tau_end,tau_start * (1 - ep/(episodes*0.5)))
            total_reward = 0
            
            for step in range(steps):
                next_state,reward,done,_, = self.env.step(action)
                next_action = policy.choose(self.Q,next_state,tau)
                self.Q[state][action] = self.Q[state][action] + alpha*(reward + gamma*np.max(self.Q[next_state]) - self.Q[state][action])
                state = next_state
                action = next_action
                total_reward += reward
                if done:
                    break
            
            if ep%1000 == 0:                
                print("Episode:",ep,"Reward:- ",total_reward)   
            
            reward_per_eps.append(total_reward)

        return reward_per_eps

    def reset_agent(self):
        self.Q = self.env.Q.copy()

    def simulate(self,env_name):
        env = gym.make(env_name,render_mode='human')
        state,_ = env.reset()
        state = self.env.get_state(state)
        done = False
        steps = 0
        t_reward = 0
        for i in range(200):
            action = np.argmax(self.Q[state])
            state,reward,done,_,_ = env.step(action)
            state = self.env.get_state(state)
            steps += 1
            t_reward += reward            
            if done:
                break
        env.close()
        print("Total steps taken:- ",steps)
        print("Total reward:- ",t_reward)


if __name__ == '__main__':
    env = Environment('MountainCar-v0')
     
    # sarsa -> eps_greedy
    eps_greedy = EpsilonGreedy(env.action_space)
    agent = Agent(env)    
    sarsa_reward = agent.train_Sarsa(episodes=RUNS,steps=NUM_STEPS,eps_end=EPS_END,
                                    policy=eps_greedy,eps_start=EPS_START)
    agent.simulate(env.env_name)

    agent.reset_agent()

    # Q-learning -> softmax
    Softmax_policy = Softmax(env.action_space)
    qlearning_reward = agent.train_Qlearnig(episodes=RUNS,steps=NUM_STEPS,tau_start=TAU_START,
                                            tau_end=TAU_END,
                                            policy=Softmax_policy)
    agent.simulate(env.env_name)
    comparison_plot(sarsa_reward,qlearning_reward)
    


    
    