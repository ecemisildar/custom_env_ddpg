import gym
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve
from drone_env import DroneEnv
import tensorflow as tf

if __name__ == '__main__':
    env = DroneEnv()
    alpha=float(1e-5)
    beta=float(1e-4)
    agent = Agent(input_dims=env.observation_space.shape, alpha=alpha, beta=beta, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 1000
    
    input_dims = env.observation_space.shape
    action_dims = env.action_space.shape[0]


    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    
    
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:

            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        if len(score_history) > 0:
            avg_score = sum(score_history) / len(score_history)
        else:
            avg_score = 0
        # avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()  


        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
        if (i + 1) % 10 == 0:
            x = [j+1 for j in range(i+1)]
            plot_learning_curve(x, score_history, f'plots/learning_curve_{i+1}.png')
            # agent.plot_actor_losses(f'loss_plots/actor_loss_{i+1}.png')
            # agent.plot_critic_losses(f'loss_plots/critic_loss_{i+1}.png')


    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, 'plots/learning_curve_final.png')
        
