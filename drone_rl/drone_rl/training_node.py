from stable_baselines3 import PPO, DDPG, TD3, DDPG, SAC, A2C
import os
from drone_env import DroneEnv
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

env = make_vec_env(lambda: DroneEnv(), n_envs=1)


n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# action_noise = OrnsteinUhlenbeckActionNoise()


TIMESTEPS = int(2e6)
SAVE_INTERVAL = int(1e4)

# 1721404353  unsuccesful
# 1721397943 -> lr = 1e-3 / eps = 200 successful but backwards batch=64
# 1721564297 -> lr = 1e-3 / eps = 100 successful ( reward = -1*distance_to_goal + penalty ) batch=64
# 1721569046  -> lr = 1e-4 / eps = 250 successful (reward = -1*distance_to_goal + penalty ) random spawned goal pos (y axis random -2,2) batch=256
# 1721576347 & 1721577292  -> lr = 1e-4 / eps = batch=128 successful (learned after 15 eps) (reward = -1*distance_to_goal + penalty ) random spawned goal pos (x and y axis random -3,3) 
# 1721647903 -> lr = 1e-4 successful in room env (learned after 20 eps ) 
# 1721725490 -> lr = 1e-4 successful in 3d without any other object (learned after 80 eps)
# 1721733957 -> lr = 1e-4 almost successful in collided building (with 5 meters long cylinder in a relatively easy position) (cant find the correct height)
# 1721742936 -> lr = 1e-4 successfull in collided building (learned after 80 eps with 7 observation pos and yaw angle of agent-4) 1.50 metes long cylinder
# 1721806541 -> lr = 1e-4 unsuccessful in collided building 7 obs



model = DDPG("MlpPolicy", env, 
             verbose=1, device='cuda:0', 
             batch_size=128, action_noise=action_noise,
             learning_rate=1e-4,
             tensorboard_log=logdir)

for i in range(0, TIMESTEPS, SAVE_INTERVAL):
    model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False, tb_log_name=f"{name}")

    model.save(f"{models_dir}/{name}_{(i+1) + SAVE_INTERVAL}")

model.save(f"{models_dir}/{name}_final")
env.close()


# model = DDPG.load("/home/ei_admin/rl_ws/models/1721742936/DDPG_final")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, terminated,  info = env.step(action)

#     if terminated :
#         obs = env.reset()

    
      



         
	

