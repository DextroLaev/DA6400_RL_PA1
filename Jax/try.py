

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from config import *
from environment import *
from q_learning import *


# def progress_bar_scan(num_samples, message=None):
#     "Progress bar for a JAX scan"
#     if message is None:
#         message = f"Running for {num_samples:,} iterations"
#     tqdm_bars = {}

#     if num_samples > 20:
#         print_rate = int(num_samples / 20)
#     else:
#         print_rate = 1  # if you run the sampler for less than 20 iterations
#     remainder = num_samples % print_rate

#     def _define_tqdm(arg, transform):
#         tqdm_bars[0] = tqdm.tqdm(range(num_samples))
#         tqdm_bars[0].set_description(message, refresh=False)

#     def _update_tqdm(arg, transform):
#         tqdm_bars[0].update(arg)

#     def _update_progress_bar(iter_num):
#         "Updates tqdm progress bar of a JAX scan or loop"
#         _ = lax.cond(
#             iter_num == 0,
#             lambda _: jax.experimental.host_callback.id_tap(
#                 _define_tqdm, None, result=iter_num),
#             lambda _: iter_num,
#             operand=None,
#         )

#         _ = lax.cond(
#             # update tqdm every multiple of `print_rate` except at the end
#             (iter_num % print_rate == 0) & (iter_num != num_samples-remainder),
#             lambda _: jax.experimental.host_callback.id_tap(
#                 _update_tqdm, print_rate, result=iter_num),
#             lambda _: iter_num,
#             operand=None,
#         )

#         _ = lax.cond(
#             # update tqdm by `remainder`
#             iter_num == num_samples-remainder,
#             lambda _: jax.experimental.host_callback.id_tap(
#                 _update_tqdm, remainder, result=iter_num),
#             lambda _: iter_num,
#             operand=None,
#         )

#     def _close_tqdm(arg, transform):
#         tqdm_bars[0].close()

#     def close_tqdm(result, iter_num):
#         return lax.cond(
#             iter_num == num_samples-1,
#             lambda _: jax.experimental.host_callback.id_tap(
#                 _close_tqdm, None, result=result),
#             lambda _: result,
#             operand=None,
#         )

#     def _progress_bar_scan(func):
#         """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
#         Note that `body_fun` must either be looping over `np.arange(num_samples)`,
#         or be looping over a tuple who's first element is `np.arange(num_samples)`
#         This means that `iter_num` is the current iteration number
#         """

#         def wrapper_progress_bar(carry, x):
#             if type(x) is tuple:
#                 iter_num, *_ = x
#             else:
#                 iter_num = x
#             _update_progress_bar(iter_num)
#             result = func(carry, x)
#             return close_tqdm(result, iter_num)

#         return wrapper_progress_bar

#     return _progress_bar_scan


def train_qlearning(env, qlearning, num_episodes_steps=NUM_SIM_STEPS, test_interval=1000):
    rewards = jnp.zeros(num_episodes_steps)
    test_rewards = jnp.zeros(num_episodes_steps // test_interval)
    rng = jax.random.PRNGKey(0)

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
            (action_rng, qlearning_Q), action = qlearning.softmax_policy(
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
            (action_rng, qlearning_Q), action = qlearning.softmax_policy(
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
        
        (rng, qlearning.Q, rewards), _ = jax.lax.scan(
            run_episode, (rng, qlearning.Q, rewards), jnp.arange(start, end)
        )
        print(f'Episode {start}-{end}: Average Reward: {np.mean(np.array(rewards[start:end]))}')
        
        test_index = start // test_interval
        (rng, qlearning.Q, test_rewards), _ = jax.lax.scan(
            test_run_episode, (rng, qlearning.Q, test_rewards), jnp.array([test_index])
        )
        # print(test_rewards)
        print(f'Test Episode {test_index}: Average Reward: {np.mean(np.array(test_rewards[:test_index+1]))}')
        print()
        
    return rewards

# # @jax.jit
# def train_qlearning(env, qlearning, num_episodes_steps=NUM_SIM_STEPS):
#     rewards = jnp.zeros(num_episodes_steps)
#     rng = jax.random.PRNGKey(0)

#     for episode in tqdm.trange(num_episodes_steps):
#         state, obs = env.reset(rng)
#         done = False
#         episode_reward = 0
#         tau = jnp.maximum(qlearning.tau_end, qlearning.tau_start * (1 - episode / num_episodes_steps))
#         for step in range(env.env_params.max_steps_in_episode):
#             (rng, qlearning.Q), action = qlearning.softmax_policy(
#                 (rng, qlearning.Q), state, tau)
#             next_state, next_obs, reward, done = env.step(
#                 rng, state, obs, action)
#             episode_reward += reward
#             qlearning.Q = qlearning.update(
#                 qlearning.Q, (state, action, reward, next_state, done))
#             state = next_state
#             obs = next_obs
#             if done:
#                 break
#         rewards = rewards.at[episode].set(episode_reward)

#         if episode % 1000 == 0:
#             print(
#                 f'Episode: {episode}, Average Reward: {np.mean(rewards[max(0, episode-1000):episode])}')

#             state, obs = env.reset(rng)
#             done = False
#             episode_reward = 0
#             while not done:
#                 (rng, qlearning.Q), action = qlearning.softmax_policy(
#                     (rng, qlearning.Q), state, tau)
#                 next_state, next_obs, reward, done = env.step(
#                 rng, state, obs, action)
#                 state = next_state
#                 obs = next_obs
#                 episode_reward += reward
#                 if done:
#                     break
#             print('Final Reward:', episode_reward, 'Tau: ', tau)

#     return rewards

# def train_qlearning(env, qlearning, num_episodes_steps=NUM_SIM_STEPS, test_interval=5000, debug=False):
#     rewards = jnp.zeros(num_episodes_steps)
#     test_rewards = jnp.zeros(num_episodes_steps // test_interval)
#     rng = jax.random.PRNGKey(0)

#     @jax.jit
#     def run_episode(carry, episode):
#         rng, qlearning_Q, rewards = carry
#         rng, ep_rng = jax.random.split(rng)
#         state, obs = env.reset(ep_rng)
#         tau = jnp.maximum(qlearning.tau_end, qlearning.tau_start * (1 - episode / num_episodes_steps))
        
#         def step_fn(step_carry):
#             state, obs, qlearning_Q, ep_rng, episode_reward, done = step_carry
#             ep_rng, action_rng = jax.random.split(ep_rng)
#             (action_rng, qlearning_Q), action = qlearning.softmax_policy((action_rng, qlearning_Q), state, tau)
#             next_state, next_obs, reward, done = env.step(ep_rng, state, obs, action)
#             qlearning_Q = qlearning.update(qlearning_Q, (state, action, reward, next_state, done))
            
#             episode_reward += reward
#             return (next_state, next_obs, qlearning_Q, ep_rng, episode_reward, done)
        
#         init_carry = (state, obs, qlearning_Q, ep_rng, jnp.float32(0), False)
#         final_carry = jax.lax.while_loop(lambda c: jnp.logical_not(c[5]), step_fn, init_carry)
#         _, _, qlearning_Q, _, episode_reward, _ = final_carry
#         rewards = rewards.at[episode].set(episode_reward)
#         return (rng, qlearning_Q, rewards), episode_reward
    
#     @jax.jit
#     def test_run_episode(carry, episode):
#         rng, qlearning_Q, test_rewards = carry
#         rng, ep_rng = jax.random.split(rng)
#         state, obs = env.reset(ep_rng)
#         tau = qlearning.tau_end
        
#         def test_step_fn(test_carry):
#             state, obs, qlearning_Q, ep_rng, episode_reward, done = test_carry
#             ep_rng, action_rng = jax.random.split(ep_rng)
#             (action_rng, qlearning_Q), action = qlearning.softmax_policy((action_rng, qlearning_Q), state, tau)
#             next_state, next_obs, reward, done = env.step(ep_rng, state, obs, action)
            
#             episode_reward += reward
#             return (next_state, next_obs, qlearning_Q, ep_rng, episode_reward, done)
        
#         init_carry = (state, obs, qlearning_Q, ep_rng, jnp.float32(0), False)
#         final_carry = jax.lax.while_loop(lambda c: jnp.logical_not(c[5]), test_step_fn, init_carry)
#         _, _, qlearning_Q, _, episode_reward, _ = final_carry
#         test_rewards = test_rewards.at[episode].set(episode_reward)
#         return (rng, qlearning_Q, test_rewards), episode_reward

#     # @jax.jit
#     @jax.jit
#     def train_loop(carry, start):
#         rng, qlearning_Q, rewards, test_rewards = carry
#         end = jnp.minimum(start + test_interval, num_episodes_steps)
#         episode_range = jnp.arange(start, end)
        
#         (rng, qlearning_Q, rewards), _ = jax.lax.scan(
#             run_episode, (rng, qlearning_Q, rewards), episode_range
#         )
        
#         test_index = start // test_interval
#         (rng, qlearning_Q, test_rewards), _ = jax.lax.scan(
#             test_run_episode, (rng, qlearning_Q, test_rewards), jnp.array([test_index])
#         )
#         return (rng, qlearning_Q, rewards, test_rewards), None

#     episode_starts = jnp.arange(0, num_episodes_steps, test_interval)
#     (rng, qlearning.Q, rewards, test_rewards), _ = jax.lax.scan(
#         train_loop, (rng, qlearning.Q, rewards, test_rewards), episode_starts
#     )
    
#     return rewards, test_rewards


if __name__ == '__main__':
    env = Environment()
    qlearning = QLearning(env)
    rewards = train_qlearning(env, qlearning)
