from Agent import *
import numpy as np
from environment import *

if __name__ == '__main__':
    env = Environment('CartPole-v1',bins=50)
     
    # sarsa -> eps_greedy
    eps_greedy = EpsilonGreedy(env.action_space)
    agent = Agent(env)
    steps = 500
    sarsa_reward = agent.train_Sarsa(episodes=10000,steps=steps,eps_end=0.01,
                                    policy=eps_greedy,eps_start=1,eps_decay=0.9995)
    
    agent.reset_agent()

    # Q-learning -> softmax
    Softmax_policy = Softmax(env.action_space)
    qlearning_reward = agent.train_Qlearnig(episodes=10000,steps=steps,tau_start=2,
                                            tau_end=0.01,
                                            policy=Softmax_policy,tau_decay=0.9995)
    
    comparison_plot(sarsa_reward,qlearning_reward,plot_name='CartPole')

    # 
    

