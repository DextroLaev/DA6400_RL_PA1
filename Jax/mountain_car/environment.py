import jax
import jax.numpy as jnp
import gymnax
import dataclasses
import functools

from config import *


class Environment:
    def __init__(self, env_name='MountainCar-v0', bins=BINS):
        # self.rng = jax.random.PRNGKey(rng_seed)
        self.env, self.env_params = gymnax.make(env_name)
        self.env_params = dataclasses.replace(
            self.env_params, max_steps_in_episode=500)
        self.max_bound = jnp.array([0.6, 0.07])
        self.min_bound = jnp.array([-1.2, -0.07])
        self.bins = jnp.array([jnp.linspace(
            self.min_bound[i], self.max_bound[i], bins) for i in range(len(self.max_bound))])

    @functools.partial(jax.jit, static_argnums=(0,))
    def discretize_state(self, state):
        state_index = jax.vmap(
            lambda s, b: jnp.digitize(s, b) - 1)(state, self.bins)
        return tuple(state_index)
        # return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_action(self, key):
        action_space = self.env.action_space(self.env_params).n
        action = jax.random.randint(key, (1,), 0, action_space)[0]
        return action

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        state, obs = self.env.reset(key, self.env_params)
        return self.discretize_state(state), obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, obs, action):
        new_state, new_obs, reward, done, _ = self.env.step(
            key, obs, action, self.env_params)
        return self.discretize_state(new_state), new_obs, reward, done

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_total_rewards_vectorized(self, rewards, dones):
        episode_starts = jnp.concatenate(
            [jnp.array([1], dtype=dones.dtype), dones[:-1]])
        episode_ids = jnp.cumsum(episode_starts) - 1
        return jax.ops.segment_sum(rewards, episode_ids, num_segments=rewards.shape[0])

    @functools.partial(jax.jit, static_argnums=(0,))
    def simulate(self, rng, num_steps=NUM_SIM_STEPS):
        def body_fn(carry, _):
            rng, state, obs = carry
            rng, key_act, key_step = jax.random.split(rng, 3)
            action = self.sample_action(key_act)
            next_state, next_obs, reward, done = self.step(
                key_step, state, obs, action)
            return (rng, next_state, next_obs), (state, next_state, action, reward, done)

        rng, key_reset = jax.random.split(rng)
        state, obs = self.reset(key_reset)
        (rng, _, _), traj = jax.lax.scan(
            body_fn, (rng, state, obs), None, length=num_steps)

        states, next_states, actions, rewards, dones = traj
        states = jnp.stack(states, axis=1).reshape(
            num_steps, -1)  # Shape (4, 200)
        next_states = jnp.stack(next_states, axis=1).reshape(
            num_steps, -1)  # Shape (4, 200)
        # actions = jnp.array(actions).reshape(num_steps, -1)
        # rewards = jnp.array(rewards).reshape(num_steps, -1)
        # dones = jnp.array(dones).reshape(num_steps, -1)
        return states, next_states, actions, rewards, dones


if __name__ == "__main__":
    for i in range(0, 5):
        env = Environment()
        rng = jax.random.PRNGKey(i)
        states, next_states, actions, rewards, dones = env.simulate(rng)
        print(states.shape, next_states.shape,
              actions.shape, rewards.shape, dones.shape)

        total_rewards = env.compute_total_rewards_vectorized(rewards, dones)
        print(actions)
        print(total_rewards)
        print(dones)

        for idx, state in enumerate(states):
            if dones[idx]:
                print(state,idx)
    # print(states, next_states, actions, rewards, dones)
    # for s, ns, a, r, d in zip(states, next_states, actions, rewards, dones):
    #     print(
    #         f"State: {s}, Next State: {ns}, Action: {a}, Reward: {r}, Done: {d}")
