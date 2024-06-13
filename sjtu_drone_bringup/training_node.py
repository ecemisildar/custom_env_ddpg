from stable_baselines3 import PPO, DDPG, TD3, DDPG, SAC
import os
from drone_env import DroneEnv
import time
import tensorflow as tf
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.policies import ActorCriticCnnPolicy
from typing import Callable

import torch
import numpy as np
import torch.nn as nn
import gym


models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = DroneEnv()


# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to True for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set the visible GPU devices
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("GPU enabled")
    except RuntimeError as e:
        print(e)


n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))


TIMESTEPS = int(1e6)
SAVE_INTERVAL = int(1e5)


model = DDPG("CnnPolicy", env, verbose=1, action_noise=action_noise, learning_rate=linear_schedule(1e-5), gamma = 0.99, tensorboard_log=logdir)

# print(model.policy)
for i in range(0, TIMESTEPS, SAVE_INTERVAL):
    model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=True, tb_log_name="DDPG")
    model.save(f"{models_dir}/DDPG_{(i+1) + SAVE_INTERVAL}")

# Final save to ensure the last model is saved
model.save(f"{models_dir}/DDPG_final")
env.close()


# del model # remove to demonstrate saving and loading

# model = DDPG.load("/home/ei_admin/ros2_ws/src/sjtu_drone/sjtu_drone_bringup/sjtu_drone_bringup/models/1716995850/DDPG_1000000")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)

#     if dones:
#          obs = env.reset()






# MAX_ITERS = 1e7  # Number of iterations
# iters = 0
# while iters < MAX_ITERS:
#     iters += 1
#     print(f"Starting iteration {iters} out of {MAX_ITERS}")
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DDPG")
#     model.save(f"{models_dir}/{TIMESTEPS*iters}")
#     print(f"Completed iteration {iters}")

# print("Training complete")



# model = load_model(saved_model_path, env)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} +/- {std_reward}")


# # Example of using the loaded model for prediction
# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)      
      



         
	

