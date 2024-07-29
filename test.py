import tensorflow as tf
import numpy as np
import gymnasium as gym

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

LOCAL = True

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

env = gym.make(
    "MiniGrid-Empty-Random-6x6-v0",
    render_mode="human" if LOCAL else "rgb_array",
    highlight=False,
    screen_size=640
)

env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

rewards = []

for episode in range(10):
    obs, _ = env.reset()
    step = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and step < 30:
        if LOCAL:
            env.render()
        action = model.predict(obs.reshape(-1, *obs.shape), verbose=0)
        obs, reward, terminated, truncated, _ = env.step(np.argmax(action))
        step += 1
        # print("step")

    print(f"{episode=} {reward=}")
    rewards.append(reward)

env.close()
print(f"mean reward: {np.mean(rewards)}")

