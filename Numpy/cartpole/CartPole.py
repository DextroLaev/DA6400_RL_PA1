from Agent import *
import numpy as np
from environment import *
import wandb
from config import *

if __name__ == '__main__':
    env = Environment('CartPole-v1')
     
    # sarsa -> eps_greedy
    eps_greedy = EpsilonGreedy(env.action_space)
    agent = Agent(env)
    sarsa_reward = agent.train_Sarsa(episodes=RUNS,steps=NUM_STEPS,eps_end=EPS_END,
                                    policy=eps_greedy,eps_start=EPS_START)
    
    
    agent.reset_agent()

    # Q-learning -> softmax
    Softmax_policy = Softmax(env.action_space)
    qlearning_reward = agent.train_Qlearnig(episodes=RUNS,steps=NUM_STEPS,tau_start=TAU_START,
                                            tau_end=TAU_END,
                                            policy=Softmax_policy)
    
    comparison_plot(sarsa_reward,qlearning_reward,plot_name='CartPole')

