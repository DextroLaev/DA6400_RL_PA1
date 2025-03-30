from Agent import *
import numpy as np
from environment import *

if __name__ == '__main__':
    sweep_config_sarsa = {
        "name":"cartpole_sarsa",
        "method": "bayes",
        "metric": {"name": "reward", "goal": "maximize"},
        "parameters": {
            "eps_start": {"values": [1.0, 0.9, 0.8,0.7,0.6,0.5]},
            "eps_end": {"values": [0.01, 0.05]},
            "alpha": {"values": [0.1, 0.05, 0.2]},
            "episodes":{'values':[10000,20000]},
            "steps":{"values":[200,300,500,1000]},            
        }
    }

    sweep_config_qlearning = {
        "name":"cartpole_qlearning",
        "method": "bayes",
        "metric": {"name": "reward", "goal": "maximize"},
        "parameters": {            
            "alpha": {"values": [0.1, 0.05, 0.2]},
            "episodes":{'values':[10000,20000]},
            "steps":{"values":[200,300,500,1000]},
            "tau_start":{"values":[1,2,3]},
            "tau_end":{"values":[0.1,0.2,0.3,0.01,0.02]},
        }
    }

    def run_sweep_sarsa():
        wandb.init(mode='online',)
        config = wandb.config
        run_name = f"SARSA_eps{config.eps_start}_eps_end{config.eps_end}"
        wandb.run.name = run_name  

        env = Environment('CartPole-v1')
        policy = EpsilonGreedy(env.action_space)
        agent = Agent(env)

        
        sarsa_reward_per_eps = agent.train_Sarsa(policy,eps_start=config.eps_start,eps_end=config.eps_end,
                                           alpha=config.alpha,episodes=config.episodes,steps=config.steps)
        

        sarsa_reward_per_eps = moving_avg(sarsa_reward_per_eps)
        
        for reward in sarsa_reward_per_eps:
            wandb.log({"reward": reward})

        wandb.log({"final_reward": sarsa_reward_per_eps[-1]})
    

    def run_sweep_qleanring():
        wandb.init(mode='online')
        config = wandb.config
        run_name = f"qlearning_tau{config.tau_start}_tauend{config.tau_end}_steps{config.steps}"
        wandb.run.name = run_name

        env = Environment('CartPole-v1')
        policy = EpsilonGreedy(env.action_space)
        agent = Agent(env)

        Softmax_policy = Softmax(env.action_space)
        qlearning_reward_per_eps = agent.train_Qlearnig(Softmax_policy,tau_start=config.tau_start,tau_end=config.tau_end,
                                           alpha=config.alpha,episodes=config.episodes,steps=config.steps)
        
        
        
        qlearning_reward_per_eps = moving_avg(qlearning_reward_per_eps)
        
        for reward in qlearning_reward_per_eps:
            wandb.log({"reward": reward})
        

        wandb.log({"qlearning_final_reward": qlearning_reward_per_eps[-1]})
    
    sweep_id = wandb.sweep(sweep_config_sarsa, project="CartPole")
    wandb.agent(sweep_id, function=run_sweep_sarsa, count=20)

    sweep_id = wandb.sweep(sweep_config_qlearning, project="CartPole")
    wandb.agent(sweep_id, function=run_sweep_qleanring, count=20)


