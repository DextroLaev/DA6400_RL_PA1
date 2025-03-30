import numpy as np
import gymnasium as gym

class Environment:
    def __init__(self,env_name,bins=50):
        self.bins = bins
        self.env_name = env_name
        self.env = self.create_env()
        self.action_space = self.env.action_space.n
        self.init_env_vars()
    
    def create_env(self,render=None):        
        return gym.make(self.env_name,render_mode=render)
    
    def close_env(self):
        self.env.close()

    def render(self):
        self.env.render()

    def init_env_vars(self):
        self.cart_position = np.linspace(-4.8,4.8,15)
        self.cart_velocity = np.linspace(-2,2,10)
        self.pole_angle = np.linspace(-0.418,0.418,30)
        self.pole_ang_velociy = np.linspace(-2,2,25)
        
        self.state_bins = [self.cart_position,self.cart_velocity,self.pole_angle,self.pole_ang_velociy]
        self.Q = np.zeros((self.cart_position.shape[0]+1,
                            self.cart_velocity.shape[0]+1,
                            self.pole_angle.shape[0]+1,
                            self.pole_ang_velociy.shape[0]+1,
                            self.action_space))

    def get_state(self,state):
        return tuple(np.digitize(state[i], self.state_bins[i]) for i in range(len(self.state_bins)))


    def step(self,action):
        next_state,reward,done,_,_ = self.env.step(action)
        return self.get_state(next_state),reward,done,_

    def sample_action(self):
        self.env.reset()
        action = self.env.action_space.sample()
    
    def reset(self):
        state,_ = self.env.reset()
        return self.get_state(state),_
    
if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env = Environment(env_name)
    env.sample_action()