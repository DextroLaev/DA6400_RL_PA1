import jax
import jax.numpy as jnp
import numpy as np
import functools

from config import *

class QLearning:
    def __init__(self, env, bins=BINS, alpha=ALPHA, gamma=GAMMA):
        self.env = env
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.Q = jnp.zeros((bins, bins, env.env.action_space(env.env_params).n))
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, carry, transition):
        Q = carry
        state, action, reward, next_state, done = transition
        q_current = Q[tuple(state) + (action,)]
        q_next = jnp.max(Q[tuple(next_state)])
        target = reward + (1 - done) * self.gamma * q_next
        updated_q = q_current + self.alpha * (target - q_current)
        Q = Q.at[tuple(state) + (action,)].set(updated_q)
        return Q
    
class QLearningSoftmax(QLearning):
    def __init__(self, env, bins=BINS, alpha=ALPHA, gamma=GAMMA, tau_start=2, tau_end=0.1):
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
    
class QLearningEpsGreedy(QLearning):
    def __init__(self, env, bins=BINS, alpha=ALPHA, gamma=GAMMA, eps_start=1, eps_end=0.05):
        super().__init__(env, bins, alpha, gamma)
        self.epsilon_start = eps_start
        self.epsilon_end = eps_end
        
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

if __name__ == "__main__":
    q_learning = QLearning()
