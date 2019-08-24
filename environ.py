import numpy as np
import gym

class environment():

    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        self.env._max_episode_steps = 1200
    
    def start(self, render=False):
        self.render = render
        return self.env.reset()

    def step(self, action):
        
        if self.render:
            self.env.render()

        state, reward, done, _ = self.env.step(action)

        reward = state[0] + 1.0
        
        if (state[0] > 0.5):
            reward = 10

        return state, reward, done

    def close(self):
        self.env.close()

        
