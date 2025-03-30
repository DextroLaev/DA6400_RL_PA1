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
jax.config.update("jax_enable_x64", True)


@jax.jit
def running_avg_rewards(rewards, window=100):
    cumsum = jnp.cumsum(rewards)
    cumsum = jnp.concatenate([jnp.zeros(1), cumsum])
    moving_avg = (cumsum[window:] - cumsum[:-window]) / window
    return moving_avg


def train_qlearning_softmax(rng, env, qlearning, num_episodes_steps=NUM_SIM_STEPS, test_interval=TEST_INTERVAL, debug=False):
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
        print(
            f'Episode {start}-{end}: Average Reward: {np.mean(np.array(rewards[start:end]))}')
        print(
            f'Test Episode {test_index}: Average Reward: {np.mean(np.array(test_rewards[:test_index+1]))}')
        print()

    return rewards, test_rewards


def train_qlearning_epsilon(rng, env, qlearning, num_episodes_steps=NUM_SIM_STEPS, test_interval=TEST_INTERVAL, debug=False):
    rewards = jnp.zeros(num_episodes_steps)
    test_rewards = jnp.zeros(num_episodes_steps // test_interval)
    

    @jax.jit
    def run_episode(carry, episode):
        rng, qlearning_Q, rewards = carry
        rng, ep_rng = jax.random.split(rng)
        state, obs = env.reset(ep_rng)
        epsilon = jnp.maximum(qlearning.epsilon_end, qlearning.epsilon_start *
                              (1 - episode / (num_episodes_steps*0.5)))

        def step_fn(step_carry):
            state, obs, qlearning_Q, ep_rng, episode_reward, done = step_carry
            ep_rng, action_rng = jax.random.split(ep_rng)
            (action_rng, qlearning_Q), action = qlearning.act(
                (action_rng, qlearning_Q), state, epsilon)
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
        epsilon = qlearning.epsilon_end

        def test_step_fn(test_carry):
            state, obs, qlearning_Q, ep_rng, episode_reward, done = test_carry
            ep_rng, action_rng = jax.random.split(ep_rng)
            (action_rng, qlearning_Q), action = qlearning.act(
                (action_rng, qlearning_Q), state, epsilon)
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
        print(
            f'Episode {start}-{end}: Average Reward: {np.mean(np.array(rewards[start:end]))}')
        print(
            f'Test Episode {test_index}: Average Reward: {np.mean(np.array(test_rewards[:test_index+1]))}')
        print()

    return rewards, test_rewards


def train_sarsa_epsilon(rng, env, sarsa, num_episodes_steps=NUM_SIM_STEPS, test_interval=TEST_INTERVAL, debug=False):
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
        print(
            f'Episode {start}-{end}: Average Reward: {np.mean(np.array(rewards[start:end]))}')
        print(
            f'Test Episode {test_index}: Average Reward: {np.mean(np.array(test_rewards[:test_index+1]))}')
        print()

    return rewards, test_rewards


def train_sarsa_softmax(rng, env, sarsa, num_episodes_steps=NUM_SIM_STEPS, test_interval=TEST_INTERVAL, debug=False):
    rewards = jnp.zeros(num_episodes_steps)
    test_rewards = jnp.zeros(num_episodes_steps // test_interval)
    

    @jax.jit
    def run_episode(carry, episode):
        rng, sarsa_Q, rewards = carry
        rng, ep_rng = jax.random.split(rng)
        state, obs = env.reset(ep_rng)
        tau = jnp.maximum(sarsa.tau_end, sarsa.tau_start *
                          (1 - episode / (num_episodes_steps*0.5)))

        def step_fn(step_carry):
            state, obs, sarsa_Q, ep_rng, episode_reward, done = step_carry
            ep_rng, action_rng = jax.random.split(ep_rng)
            (action_rng, sarsa_Q), action = sarsa.act(
                (action_rng, sarsa_Q), state, tau)
            next_state, next_obs, reward, done = env.step(
                ep_rng, state, obs, action)
            (action_rng, sarsa_Q), next_action = sarsa.act(
                (action_rng, sarsa_Q), next_state, tau)
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
        tau = sarsa.tau_end

        def test_step_fn(test_carry):
            state, obs, sarsa_Q, ep_rng, episode_reward, done = test_carry
            ep_rng, action_rng = jax.random.split(ep_rng)
            (action_rng, sarsa_Q), action = sarsa.act(
                (action_rng, sarsa_Q), state, tau)
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
            test_run_episode, (rng, sarsa.Q, test_rewards), jnp.array([test_index])
        )
        print(f'Episode {start}-{end}: Average Reward: {np.mean(np.array(rewards[start:end]))}')
        print(f'Test Episode {test_index}: Average Reward: {np.mean(np.array(test_rewards[:test_index+1]))}')
        print()
    
    return rewards, test_rewards


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    env = Environment()
    
    for i in range(5):
        rng = jax.random.PRNGKey(i)  # Set random seed for reproducibility
        
        agent = QLearningSoftmax(env=env)
        rewards, eval_rewards = train_qlearning_softmax(rng, env, agent)
        np.savetxt(f'results/q_learning_softmax_rewards_{i}.txt', rewards)
        np.savetxt(f'results/q_learning_softmax_eval_rewards_{i}.txt', eval_rewards)
        avg_rewards = running_avg_rewards(rewards)
        eval_avg_rewards = running_avg_rewards(eval_rewards)
        plt.plot(rewards, label = 'rewards')
        plt.plot(avg_rewards, label = 'average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.title('q_learning_softmax_rewards'.replace('_', '-').upper())
        plt.savefig('q_learning_softmax_rewards.png')
        plt.close()

        agent = SARSASoftmax(env=env)
        rewards, eval_rewards = train_sarsa_softmax(rng, env, agent)
        avg_rewards = running_avg_rewards(rewards)
        eval_avg_rewards = running_avg_rewards(eval_rewards)
        np.savetxt(f'results/sarsa_softmax_rewards_{i}.txt', rewards)
        np.savetxt(f'results/sarsa_softmax_eval_rewards_{i}.txt', eval_rewards)
        plt.plot(rewards, label='rewards')
        plt.plot(avg_rewards, label='average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.title('sarsa_softmax_rewards'.replace('_', '-').upper())
        plt.savefig('sarsa_softmax_rewards.png')
        plt.close()

        agent = QLearningEpsGreedy(env=env)
        rewards, eval_rewards = train_qlearning_epsilon(rng, env, agent)
        avg_rewards = running_avg_rewards(rewards)
        eval_avg_rewards = running_avg_rewards(eval_rewards)
        np.savetxt(f'results/q_learning_epsilon_rewards_{i}.txt', rewards)
        np.savetxt(f'results/q_learning_epsilon_eval_rewards_{i}.txt', eval_rewards)
        plt.plot(rewards, label='rewards')
        plt.plot(avg_rewards, label='average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.title('q_learning_epsilon_rewards'.replace('_', '-').upper())
        plt.savefig('q_learning_epsilon_rewards.png')
        plt.close()

        agent = SARSAEpsGreedy(env=env)
        rewards, eval_rewards = train_sarsa_epsilon(rng, env, agent)
        avg_rewards = running_avg_rewards(rewards)
        eval_avg_rewards = running_avg_rewards(eval_rewards)
        np.savetxt(f'results/sarsa_epsilon_rewards_{i}.txt', rewards)
        np.savetxt(f'results/sarsa_epsilon_eval_rewards_{i}.txt', eval_rewards)
        plt.plot(rewards, label='rewards')
        plt.plot(avg_rewards, label='average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.title('sarsa_epsilon_rewards'.replace('_', '-').upper())
        plt.savefig('sarsa_epsilon_rewards.png')
        plt.close()
