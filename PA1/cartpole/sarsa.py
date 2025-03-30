import jax
import jax.numpy as jnp
import numpy as np
import functools

from config import *

class SARSA:
    def __init__(self, env, bins=BINS, alpha=ALPHA, gamma=GAMMA):
        self.env = env
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.Q = jnp.zeros((bins, bins, bins, bins, env.env.action_space(env.env_params).n))
        
    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, carry, transition):
        Q = carry
        state, action, reward, next_state, next_action, done = transition
        q_current = Q[tuple(state) + (action,)]
        q_next = Q[tuple(next_state) + (next_action,)]
        target = reward + (1 - done) * self.gamma * q_next
        updated_q = q_current + self.alpha * (target - q_current)
        Q = Q.at[tuple(state) + (action,)].set(updated_q)
        return Q
        
        
class SARSASoftmax(SARSA):
    def __init__(self, env, bins=BINS, alpha=ALPHA, gamma=GAMMA, tau_start=2.0, tau_end=0.1):
        super().__init__(env, bins, alpha, gamma)
        self.tau_start = tau_start
        self.tau_end = tau_end
        
    @functools.partial(jax.jit, static_argnums=(0,))
    def act(self, carry, state, tau):
        rng, Q = carry
        q_values = Q[tuple(state)] / tau
        probabilities = jax.nn.softmax(q_values)
        rng, subkey = jax.random.split(rng)
        action = jax.random.categorical(subkey, jnp.log(probabilities))
        return (rng, Q), action
    

class SARSAEpsGreedy(SARSA):
    def __init__(self, env, bins=BINS, alpha=ALPHA, gamma=GAMMA, epsilon_start=0.1, epsilon_end=0.01):
        super().__init__(env, bins, alpha, gamma)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

    @functools.partial(jax.jit, static_argnums=(0,))
    def act(self, carry, state, epsilon):
        rng, Q = carry
        q_values = Q[tuple(state)]
        
        rng, subkey = jax.random.split(rng)
        explore = jax.random.uniform(subkey) < epsilon
        @jax.jit
        def random_action():
            return jax.random.randint(subkey, (), 0, q_values.shape[0])
        @jax.jit
        def greedy_action():
            return jnp.argmax(q_values)
        
        action = jax.lax.cond(explore, random_action, greedy_action)
        return (rng, Q), action
        
        