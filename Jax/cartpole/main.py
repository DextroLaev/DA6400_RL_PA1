import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import wandb

from q_learning import *
from sarsa import *
from environment import *
from config import *

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update("jax_enable_x64", True)


@jax.jit
def running_avg_rewards(rewards, window=100):
    cumsum = jnp.cumsum(rewards)
    cumsum = jnp.concatenate([jnp.zeros(1), cumsum])
    moving_avg = (cumsum[window:] - cumsum[:-window]) / window
    return moving_avg


def train_qlearning_softmax(rng, env, qlearning, num_episodes_steps=NUM_SIM_STEPS, test_interval=TEST_INTERVAL, debug=False,wandb_log = False):
    rewards = jnp.zeros(num_episodes_steps)
    test_rewards = jnp.zeros(num_episodes_steps // test_interval)
    

    @jax.jit
    def run_episode(carry, episode):
        rng, qlearning_Q, rewards = carry
        rng, ep_rng = jax.random.split(rng)
        state, obs = env.reset(ep_rng)
        tau = jnp.maximum(qlearning.tau_end, qlearning.tau_start *
                          (1 - episode / (num_episodes_steps*0.5)))

        def step_fn(step_carry):
            state, obs, qlearning_Q, ep_rng, episode_reward, done = step_carry
            ep_rng, action_rng = jax.random.split(ep_rng)
            (action_rng, qlearning_Q), action = qlearning.act(
                (action_rng, qlearning_Q), state, tau)
            next_state, next_obs, reward, done = env.step(
                ep_rng, state, obs, action)
            qlearning_Q = qlearning.update(
                qlearning_Q, (state, action, reward, next_state, done))

            episode_reward += reward
            return (next_state, next_obs, qlearning_Q, ep_rng, episode_reward, done)

        init_carry = (state, obs, qlearning_Q, ep_rng, jnp.float32(0), False)
        final_carry = jax.lax.while_loop(
            lambda c: jnp.logical_not(c[5]), step_fn, init_carry)
        _, _, qlearning_Q, _, episode_reward, _ = final_carry
        rewards = rewards.at[episode].set(episode_reward)
        return (rng, qlearning_Q, rewards), episode_reward

    @jax.jit
    def test_run_episode(carry, episode):
        rng, qlearning_Q, test_rewards = carry
        rng, ep_rng = jax.random.split(rng)
        state, obs = env.reset(ep_rng)
        tau = qlearning.tau_end

        def test_step_fn(test_carry):
            state, obs, qlearning_Q, ep_rng, episode_reward, done = test_carry
            ep_rng, action_rng = jax.random.split(ep_rng)
            (action_rng, qlearning_Q), action = qlearning.act(
                (action_rng, qlearning_Q), state, tau)
            next_state, next_obs, reward, done = env.step(
                ep_rng, state, obs, action)

            episode_reward += reward
            return (next_state, next_obs, qlearning_Q, ep_rng, episode_reward, done)

        init_carry = (state, obs, qlearning_Q, ep_rng, jnp.float32(0), False)
        final_carry = jax.lax.while_loop(
            lambda c: jnp.logical_not(c[5]), test_step_fn, init_carry)
        _, _, qlearning_Q, _, episode_reward, _ = final_carry
        test_rewards = test_rewards.at[episode].set(episode_reward)
        return (rng, qlearning_Q, test_rewards), episode_reward

    for start in tqdm.tqdm(range(0, num_episodes_steps, test_interval)):
        end = min(start + test_interval, num_episodes_steps)
        test_index = start // test_interval

        (rng, qlearning.Q, rewards), _ = jax.lax.scan(
            run_episode, (rng, qlearning.Q, rewards), jnp.arange(start, end)
        )
        (rng, qlearning.Q, test_rewards), _ = jax.lax.scan(
            test_run_episode, (rng, qlearning.Q,
                               test_rewards), jnp.array([test_index])
        )
        if wandb_log:
            wandb.log({
                'train_episode_reward_averaged': np.mean(np.array(rewards[start:end])),
                'test_episode_reward_averaged': np.mean(np.array(test_rewards[:test_index+1])),
                'episode': start
            })
        print(
            f'Episode {start}-{end}: Average Reward: {np.mean(np.array(rewards[start:end]))}')
        print(
            f'Test Episode {test_index}: Average Reward: {np.mean(np.array(test_rewards[:test_index+1]))}')
        print()

    return rewards, test_rewards


def train_sarsa_epsilon(rng, env, sarsa, num_episodes_steps=NUM_SIM_STEPS, test_interval=TEST_INTERVAL, debug=False, wandb_log=False):
    rewards = jnp.zeros(num_episodes_steps)
    test_rewards = jnp.zeros(num_episodes_steps // test_interval)
    

    @jax.jit
    def run_episode(carry, episode):
        rng, sarsa_Q, rewards = carry
        rng, ep_rng = jax.random.split(rng)
        state, obs = env.reset(ep_rng)
        epsilon = jnp.maximum(sarsa.epsilon_end, sarsa.epsilon_start *
                              (1 - episode / (num_episodes_steps*0.5)))

        def step_fn(step_carry):
            state, obs, sarsa_Q, ep_rng, episode_reward, done = step_carry
            ep_rng, action_rng = jax.random.split(ep_rng)
            (action_rng, sarsa_Q), action = sarsa.act(
                (action_rng, sarsa_Q), state, epsilon)
            next_state, next_obs, reward, done = env.step(
                ep_rng, state, obs, action)
            (action_rng, sarsa_Q), next_action = sarsa.act(
                (action_rng, sarsa_Q), next_state, epsilon)
            sarsa_Q = sarsa.update(
                sarsa_Q, (state, action, reward, next_state, next_action, done))

            episode_reward += reward
            return (next_state, next_obs, sarsa_Q, ep_rng, episode_reward, done)

        init_carry = (state, obs, sarsa_Q, ep_rng, jnp.float32(0), False)
        final_carry = jax.lax.while_loop(
            lambda c: jnp.logical_not(c[5]), step_fn, init_carry)
        _, _, sarsa_Q, _, episode_reward, _ = final_carry
        rewards = rewards.at[episode].set(episode_reward)
        return (rng, sarsa_Q, rewards), episode_reward

    @jax.jit
    def test_run_episode(carry, episode):
        rng, sarsa_Q, test_rewards = carry
        rng, ep_rng = jax.random.split(rng)
        state, obs = env.reset(ep_rng)
        epsilon = sarsa.epsilon_end

        def test_step_fn(test_carry):
            state, obs, sarsa_Q, ep_rng, episode_reward, done = test_carry
            ep_rng, action_rng = jax.random.split(ep_rng)
            (action_rng, sarsa_Q), action = sarsa.act(
                (action_rng, sarsa_Q), state, epsilon)
            next_state, next_obs, reward, done = env.step(
                ep_rng, state, obs, action)

            episode_reward += reward
            return (next_state, next_obs, sarsa_Q, ep_rng, episode_reward, done)

        init_carry = (state, obs, sarsa_Q, ep_rng, jnp.float32(0), False)
        final_carry = jax.lax.while_loop(
            lambda c: jnp.logical_not(c[5]), test_step_fn, init_carry)
        _, _, sarsa_Q, _, episode_reward, _ = final_carry
        test_rewards = test_rewards.at[episode].set(episode_reward)
        return (rng, sarsa_Q, test_rewards), episode_reward

    for start in tqdm.tqdm(range(0, num_episodes_steps, test_interval)):
        end = min(start + test_interval, num_episodes_steps)
        test_index = start // test_interval

        (rng, sarsa.Q, rewards), _ = jax.lax.scan(
            run_episode, (rng, sarsa.Q, rewards), jnp.arange(start, end)
        )
        (rng, sarsa.Q, test_rewards), _ = jax.lax.scan(
            test_run_episode, (rng, sarsa.Q, test_rewards), jnp.array(
                [test_index])
        )
        if wandb_log:
            wandb.log({
                'train_episode_reward_averaged': np.mean(np.array(rewards[start:end])),
                'test_episode_reward_averaged': np.mean(np.array(test_rewards[:test_index+1])),
                'episode': start
            })
        print(
            f'Episode {start}-{end}: Average Reward: {np.mean(np.array(rewards[start:end]))}')
        print(
            f'Test Episode {test_index}: Average Reward: {np.mean(np.array(test_rewards[:test_index+1]))}')
        print()

    return rewards, test_rewards

def sweep_run_sarsa_epsilon():
    sweep_config = {
        "method": "bayes",  # Choose "grid", "random", or "bayes"
        "metric": {"name": "train_episode_reward_averaged", "goal": "maximize"},
        "parameters": {
            "eps_start": {"values": [1.0, 0.9, 0.8,0.7,0.6]},
            "eps_end": {"values": [0.01, 0.05]},
            "alpha": {"values": [0.1, 0.05, 0.2]},
            "episodes":{'values':[10000,20000]},
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="CartPole_SARSA")
    
    def run_sweep():
        wandb.init(mode='online')
        config = wandb.config
        wandb.run.name = f'SARSA epsilon, Alpha: {config.alpha}, Epsilon Start: {config.eps_start}, Epsilon End: {config.eps_end}'
        env = Environment()
        agent = SARSAEpsGreedy(env,alpha=config.alpha,epsilon_start=config.eps_start, epsilon_end=config.eps_end)
        rewards, test_rewards = train_sarsa_epsilon(jax.random.PRNGKey(0), env, agent, num_episodes_steps=config.episodes, test_interval=1000,wandb_log=True)
        
    wandb.agent(sweep_id = sweep_id,function=run_sweep,count = 20)
    
def sweep_run_qlearning_softmax():
    sweep_config = {
        "method": "bayes",  # Choose "grid", "random", or "bayes"
        "metric": {"name": "train_episode_reward_averaged", "goal": "maximize"},
        "parameters": {
            "tau_start":{"values":[1,2,3]},
            "tau_end":{"values":[0.1,0.2,0.3,0.01,0.02]},
            "alpha": {"values": [0.1, 0.05, 0.2]},
            "episodes":{'values':[10000,20000]},
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="CartPole_QLEARNING")
    
    def run_sweep():
        wandb.init(mode='online')
        config = wandb.config
        wandb.run.name = f'Q-Learning Softmax, Alpha: {config.alpha}, Tau Start: {config.tau_start}, Tau End: {config.tau_end}'
        env = Environment()
        agent = QLearningSoftmax(env,alpha=config.alpha,tau_start=config.tau_start, tau_end=config.tau_end)
        rewards, test_rewards = train_qlearning_softmax(jax.random.PRNGKey(0), env, agent, num_episodes_steps=config.episodes, test_interval=1000,wandb_log=True)
        
    wandb.agent(sweep_id = sweep_id,function=run_sweep,count = 20)
    


if __name__ == '__main__':
    # sweep_run_qlearning_softmax()
    # sweep_run_sarsa_epsilon()
    
    os.makedirs('results_seeds', exist_ok=True)
    env = Environment()
        
    for i in range(5):
        rng = jax.random.PRNGKey(i)  # Set random seed for reproducibility
        
        agent = QLearningSoftmax(env=env, tau_start=1, tau_end=0.02)
        rewards, eval_rewards = train_qlearning_softmax(rng, env, agent, num_episodes_steps=20_000)
        np.savetxt(f'results_seeds/q_learning_softmax_rewards_{i}.txt', rewards)
        
        agent = SARSAEpsGreedy(env=env, epsilon_start=0.9, epsilon_end=0.01)
        rewards, eval_rewards = train_sarsa_epsilon(rng, env, agent, num_episodes_steps=20_000)
        np.savetxt(f'results_seeds/sarsa_epsilon_rewards_{i}.txt', rewards)
        
    for tau_start in [2,1]:
        for tau_end in [0.01,0.02]:
            agent = QLearningSoftmax(env=env, tau_start=tau_start, tau_end=tau_end)
            rewards, eval_rewards = train_qlearning_softmax(rng, env, agent)
            np.savetxt(f'results/q_learning_softmax_rewards_{tau_start}_{tau_end}.txt', rewards)
            
    for eps_start in [1,0.9]:
        for eps_end in [0.01,0.02]:
            agent = SARSAEpsGreedy(env=env, epsilon_start=eps_start, epsilon_end=eps_end)
            rewards, eval_rewards = train_sarsa_epsilon(rng, env, agent)
            np.savetxt(f'results/sarsa_epsilon_rewards_{eps_start}_{eps_end}.txt', rewards)
            

