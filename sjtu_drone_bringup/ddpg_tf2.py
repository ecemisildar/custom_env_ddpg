import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
import numpy as np
import matplotlib.pyplot as plt
import os


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

class Agent:
    def __init__(self, input_dims, alpha, beta, env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=400, fc2=300, batch_size=32, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.seq_l = 3
        self.memory = ReplayBuffer(max_size, self.seq_l, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

        self.actor_losses = []
        self.critic_losses = []


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        print("remember")
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        epsilon = 0.01
        if not evaluate:
            if np.random.rand() < epsilon:
                # Exploration: Choose a random action within the valid action space
                random_action = tf.random.uniform(shape=[self.n_actions], minval=self.min_action, maxval=self.max_action)
                actions += random_action
            else:
                # Exploitation: Choose the action suggested by the actor network and add noise if not evaluating
                actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)

        # if not evaluate:
        #     actions += tf.random.normal(shape=[self.n_actions],
        #                                 mean=0.0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        # print(f"actions {actions}")

        return actions[0]

    def learn(self):
        print("learn")
        if self.memory.mem_cntr < self.batch_size:
            print("condition 1")
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        
        num_states = state.shape[1]
        print(f"num of states: {num_states}")

        if num_states <= self.seq_l:
            print("condition 2")
            return
        
        batch_indices = np.random.randint(self.seq_l, state.shape[1], size=self.batch_size)

        states, actions, rewards, new_states, dones = [], [], [], [], []
        for idx in batch_indices:
            print("for loop")
            states.append(state[:, idx - self.seq_l:idx, :])
            actions.append(action[:, idx - self.seq_l:idx, :])
            rewards.append(reward[:, idx - self.seq_l:idx])
            new_states.append(new_state[:, idx - self.seq_l:idx, :])
            dones.append(done[:, idx - self.seq_l:idx])

        states = np.array(states)
        print(f"states shape {states.shape}")
        actions = np.array(actions)
        rewards = np.array(rewards)
        new_states = np.array(new_states)
        dones = np.array(dones)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            # critic_loss = keras.losses.MSE(target, critic_value)
            critic_loss = tf.math.reduce_mean(tf.math.abs(target - critic_value))

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def plot_actor_losses(self, figure_file):
        x = range(len(self.actor_losses))
        plt.plot(x, self.actor_losses, label='Actor Loss')

        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Actor Loss during Training')
        directory = os.path.dirname(figure_file)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(figure_file)

    def plot_critic_losses(self, figure_file):
        x = range(len(self.critic_losses))
        plt.plot(x, self.critic_losses, label='Critic Loss')

        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Critic Loss during Training')
        directory = os.path.dirname(figure_file)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(figure_file)   