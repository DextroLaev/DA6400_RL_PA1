from Agent import *
import numpy as np
from environment import *
from config import *

if __name__ == '__main__':
    env = Environment('MountainCar-v0',bins=BINS)
     
    # sarsa -> eps_greedy
    eps_greedy = EpsilonGreedy(env.action_space)
    agent = Agent(env)

    sarsa_reward = agent.train_Sarsa(episodes=RUNS,eps_end=EPS_END,steps=NUM_STEPS,alpha=ALPHA,
                                    policy=eps_greedy,eps_start=EPS_START)
    
    agent.reset_agent()

    # Q-learning -> softmax
    Softmax_policy = Softmax(env.action_space)
    qlearning_reward = agent.train_Qlearnig(episodes=RUNS,tau_start=TAU_START,
                                            tau_end=TAU_END,alpha=ALPHA,
                                            policy=Softmax_policy,steps=NUM_STEPS)
    
    comparison_plot(sarsa_reward,qlearning_reward,plot_name='MountainCar')