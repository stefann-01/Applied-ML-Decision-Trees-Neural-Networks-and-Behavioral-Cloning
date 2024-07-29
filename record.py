import gymnasium as gym
import numpy as np
import pygame

from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper


class ManualControl:
    def __init__(self, env):
        self.env = env
        self.closed = False
        self.last_obs = None
        self.observations = []
        self.actions = []

    def start(self):
        self.reset()

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    key = pygame.key.name(int(event.key))
                    self.key_handler(key)
        
        self.env.close()

        if len(self.observations) > 0 and len(self.actions) == len(self.observations):
            np.save('observations.npy', self.observations)
            np.save('actions.npy', self.actions)
        else:
            print("No data saved")

    def step(self, action):
        self.observations.append(self.last_obs)
        self.actions.append(action)

        self.last_obs, _, terminated, truncated, _ = self.env.step(action)

        if terminated or truncated:
            self.reset()
        else:
            self.env.render()

    def reset(self):
        self.last_obs, _ = self.env.reset()
        self.env.render()

    def key_handler(self, key):
        # print(self.observations)
        if key == "escape":
            self.closed = True
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "tab": Actions.pickup,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)


env = gym.make(
    "MiniGrid-Empty-6x6-v0",
    render_mode="human",
    highlight=False,
    screen_size=640
)

env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

manual_control = ManualControl(env)
manual_control.start()
