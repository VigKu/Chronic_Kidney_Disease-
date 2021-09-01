from abc import ABC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Classifier(layers.Layer):
    def __init__(self, channels=None):  # [10,2,]
        super(Classifier, self).__init__()
        self.linear1 = layers.Dense(channels[0])
        self.linear2 = layers.Dense(channels[1], activation='softmax')
        self.dropout = layers.Dropout(0.2)

    def call(self, x, training=False):
        x = tf.nn.relu(self.linear1(x, training=training))
        x = self.dropout(x, training=training)
        x = self.linear2(x, training=training)
        return x


class LinearEncoder(layers.Layer):
    def __init__(self, channels=None):  # [16,5]
        super(LinearEncoder, self).__init__()
        self.linear1 = layers.Dense(channels[0])
        self.linear2 = layers.Dense(channels[1], kernel_regularizer=regularizers.l2(0.00001))
        self.dropout = layers.Dropout(0.1)

    def call(self, x, training=False):
        x = tf.nn.relu(self.linear1(x, training=training))
        x = self.dropout(x, training=training)
        x = self.linear2(x, training=training)
        # x = self.dropout(x, training=training)
        return x


class ConvBlock(layers.Layer):
    def __init__(self, out_channels=None, kernel_size=None):  # kernel_size = (1,3)
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding="same",
                                  use_bias=False, kernel_regularizer=regularizers.l2(0.00001))
        self.bn = layers.BatchNormalization()
        self.pool = layers.MaxPooling2D((1, 2), padding="valid")

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)
        return x


class WaveEncoder(layers.Layer):
    def __init__(self, channels=None):
        super(WaveEncoder, self).__init__()
        self.cb1 = ConvBlock(channels[0], (1, 3))  # maxpool seq: 35 -> 18
        self.cb2 = ConvBlock(channels[1], (1, 3))  # maxpool seq: 18 -> 9
        self.cb3 = ConvBlock(channels[2], (1, 3))  # maxpool seq: 9 -> 5
        self.dropout1 = layers.Dropout(0.2)
        self.dropout2 = layers.Dropout(0.2)

    def call(self, x, training=False):
        x = self.cb1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.cb2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.cb3(x, training=training)
        # x = self.dropout(x, training=training)
        return x


# Final model
class CombinedModel(keras.Model, ABC):
    def __init__(self, seq_len, num_days):
        super(CombinedModel, self).__init__()
        self.encoder1 = WaveEncoder([64, 32, 1])  # [32,32,1]
        self.encoder2 = LinearEncoder([32, 5])  # [16, 16, 5]
        self.fc = Classifier([10, 2])
        self.seq_len = seq_len
        self.num_days = num_days
        self.gru = layers.GRU(5, stateful=True)

    def call(self, data, training=False):
        # loop over the sequences
        x1, x2 = data
        output = None
        for i in range(0, self.num_days, self.seq_len):
            # encoder
            x2_sub = self.encoder1(x2[:, :, i:i + self.seq_len, :], training=training)
            x2_sub = tf.squeeze(x2_sub, axis=1)
            # GRU
            output = self.gru(x2_sub, training=training)
        # print(f"output shape: {output.shape}")
        x1 = self.encoder2(x1, training=training)
        # print(f"x1 shape: {x1.shape}")
        out = tf.concat([x1, output], axis=1)
        # print(f"out shape: {out.shape}")
        # out = tf.reshape(out, [out.shape[0], -1])
        out = self.fc(out, training=training)
        return out

    # def model(self):  # optional to get output shape in model summary
    #    x = keras.Input(shape=(1, 35, 12))
    #    return keras.Model(inputs=[x], outputs=self.call(x))
#  model = CombinedModel().model()
# print(model.summary())
