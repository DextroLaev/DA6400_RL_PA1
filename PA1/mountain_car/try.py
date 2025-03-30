import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os

from q_learning import *
from sarsa import *
from environment import *
from config import *

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update("jax_enable_x64", False)


@jax.jit
def running_avg_rewards(rewards, window=100):
    cumsum = jnp.cumsum(rewards)
    cumsum = jnp.concatenate([jnp.zeros(1), cumsum])
    moving_avg = (cumsum[window:] - cumsum[:-window]) / window
    return moving_avg

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

def train_qlearning_softmax(env, qlearning, num_episodes_steps=NUM_SIM_STEPS, test_interval=TEST_INTERVAL, debug=False):
    rewards = jnp.zeros(num_episodes_steps)
    test_rewards = jnp.zeros(num_episodes_steps // test_interval)
    rng = jax.random.PRNGKey(0)

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
        print(
            f'Episode {start}-{end}: Average Reward: {np.mean(np.array(rewards[start:end]))}')
        print(
            f'Test Episode {test_index}: Average Reward: {np.mean(np.array(test_rewards[:test_index+1]))}')
        print()

    return rewards, test_rewards
