import numpy as np

class Policy:
    def __init__(self,action_space):
        self.action_space = action_space

class Softmax(Policy):
    def __init__(self,action_space):
        super().__init__(action_space)
        self.action_space = action_space

    def choose(self,Q,state,temp=0.4):
        logits = np.exp( (Q[state] - np.max(Q[state])) /temp)
        sums = np.sum(logits)
        return np.random.choice(self.action_space,p = logits/sums)

class EpsilonGreedy(Policy):
    def __init__(self,action_space):
        super().__init__(action_space)
        self.action_space = action_space        

    def choose(self,Q,state,eps=0.3):
        prob = np.random.uniform(0,1)
        if prob <= eps:
            return np.random.choice(np.arange(self.action_space))
        else:
            return np.argmax(Q[state])