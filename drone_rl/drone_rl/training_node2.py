from stable_baselines3 import PPO, DDPG, TD3, DDPG, SAC, A2C
import os
from drone_env2 import DroneEnv2
import time
import tensorflow as tf
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback

from typing import Callable


import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import gym


models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
name = f"DDPG"


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

env = make_vec_env(lambda: DroneEnv2(), n_envs=1)


n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# action_noise = OrnsteinUhlenbeckActionNoise()


TIMESTEPS = int(2e6)
SAVE_INTERVAL = int(1e4)


# model = DDPG("MlpPolicy", env, 
#              verbose=1, device='cuda:0', 
#              batch_size=128, action_noise=action_noise,
#              learning_rate=1e-4,
#              tensorboard_log=logdir)

# for i in range(0, TIMESTEPS, SAVE_INTERVAL):
#     model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False, tb_log_name=f"{name}")

#     model.save(f"{models_dir}/{name}_{(i+1) + SAVE_INTERVAL}")

# model.save(f"{models_dir}/{name}_final")
# env.close()


model = DDPG.load("/home/ei_admin/rl_ws/drone_trainings_tests/trainings/DDPG/1722084092/DDPG_final")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated,  info = env.step(action)

    if terminated :
        obs = env.reset()

    
      



         
	

