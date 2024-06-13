import tensorflow as tf

class A3Clstm(tf.keras.Model):
    def __init__(self, num_inputs, action_space):
        super(A3Clstm, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')
        self.maxp1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')
        self.maxp2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=1, padding='same', activation='relu')
        self.maxp3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')
        self.conv4 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.maxp4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')

        self.lstm = tf.keras.layers.LSTMCell(512)
        num_outputs = action_space.n
        self.critic_linear = tf.keras.layers.Dense(1)
        self.actor_linear = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, states):
        inputs, (hx, cx) = inputs
        x = self.conv1(inputs)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.maxp3(x)
        x = self.conv4(x)
        x = self.maxp4(x)

        x = tf.reshape(x, (x.shape[0], -1))

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)