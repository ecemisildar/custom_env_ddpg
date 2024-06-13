import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate, LSTM, Reshape, MaxPooling2D


class CriticNetwork(keras.Model):
    def __init__(self, conv_filters=(32, 64, 64), fc1_dims=1024, fc2_dims=512,
                 name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.conv_filters = conv_filters
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.weights.h5')

        self.conv1 = Conv2D(self.conv_filters[0], (8, 8), strides=(4, 4), activation='gelu')
        self.conv2 = Conv2D(self.conv_filters[1], (4, 4), strides=(2, 2), activation='gelu')
        self.conv3 = Conv2D(self.conv_filters[2], (3, 3), strides=(1, 1), activation='gelu')
        self.flatten = Flatten()

        self.reshape = Reshape((1, -1))
        self.lstm = LSTM(512, return_sequences=False)


        self.concat = Concatenate()
        self.fc1 = Dense(self.fc1_dims, activation='gelu')
        self.fc2 = Dense(self.fc2_dims, activation='gelu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        batch_size, sequence_length, height, width, channels = state.shape
        state = tf.reshape(state, (-1, height, width, channels))
        

        conv_state = self.conv1(state)
        conv_state = self.conv2(conv_state)
        conv_state = self.conv3(conv_state)
        flatten_state = self.flatten(conv_state)

        flatten_state = tf.reshape(flatten_state, (batch_size, sequence_length, -1))
        

        # x = self.reshape(flatten_state)
        x = self.lstm(flatten_state)
        concat_input = self.concat([x, action])
       
        # concat_input = self.concat([flatten_state, action])
    
        fc1_output = self.fc1(concat_input)
        fc2_output = self.fc2(fc1_output)
        q = self.q(fc2_output)
        return q    

class ActorNetwork(keras.Model):
    def __init__(self, conv_filters=(32, 64, 64), fc1_dims=512, fc2_dims=512, n_actions=2, name='actor',
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.conv_filters = conv_filters
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.weights.h5')

        self.conv1 = Conv2D(self.conv_filters[0], (8, 8), strides=(4, 4), activation='gelu')
        self.conv2 = Conv2D(self.conv_filters[1], (4, 4), strides=(2, 2), activation='gelu')
        self.conv3 = Conv2D(self.conv_filters[2], (3, 3), strides=(1, 1), activation='gelu')
        self.flatten = Flatten()

        self.reshape = Reshape((1, -1))
        self.lstm = LSTM(512, return_sequences=False)

       
        self.fc1 = Dense(self.fc1_dims, activation='gelu')
        self.fc2 = Dense(self.fc2_dims, activation='gelu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        batch_size, sequence_length, height, width, channels = state.shape
        state = tf.reshape(state, (-1, height, width, channels))
        
        conv_output = self.conv1(state)
        conv_output = self.conv2(conv_output)
        conv_output = self.conv3(conv_output)
        flatten_output = self.flatten(conv_output)

        flatten_output = tf.reshape(flatten_output, (batch_size, sequence_length, -1))
      
        
        # x = self.reshape(flatten_output)
        x = self.lstm(flatten_output)

        fc1_output = self.fc1(x)
        fc2_output = self.fc2(fc1_output)

        mu = self.mu(fc2_output)
        return mu   