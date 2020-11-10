import math, random, pickle, itertools

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, add, ReLU
from tensorflow.keras import Model
import matplotlib
import matplotlib.pyplot as plt

train_data = pd.read_csv("./mitbih_train.csv", header=None)
test_data = pd.read_csv("./mitbih_test.csv", header=None)

'''
# 그래프를 그리기 위한 과정. 한 샘플만 집어서 그림. 
l_train = train_data.iloc[0, :-1].tolist()
x_axis = range(0, 187)

plt.figure(figsize=(20,12))

COLOR = 'white'
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR

plt.plot(x_axis, l_train, label="Example")
plt.legend()

plt.title("1-beat ECG", fontsize=20)
plt.ylabel("Amplitude", fontsize=15)
plt.xlabel("Time (ms)", fontsize=15)
plt.show()
'''

# 데이터셋 분리하기
train_information = train_data.iloc[:, :-1].to_numpy
train_label = train_data.iloc[:, -1].to_numpy

test_information = test_data.iloc[:, :-1].to_numpy
test_label = test_data.iloc[:, -1].to_numpy

train_information = train_information[..., tf.newaxis]
train_label = train_label[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_information, train_label))
train_ds = [..., tf.newaxis]
train_ds = train_ds.shuffle(10000).batch(500)
test_ds = tf.data.Dataset.from_tensor_slices(
    (test_information, test_label)).batch(500)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_in = Conv1D(32, input_shape=(None, 187, 1), kernel_size=(5), strides=1,
                              padding='same', activation='relu')
        self.conv_relu = Conv1D(32, kernel_size=(5), strides=1,
                           padding='same', activation='relu')
        self.conv_raw = Conv1D(32, kernel_size=(5), strides=1,
                              padding='same')
        self.maxpool = MaxPooling1D(pool_size=5, strides=2)
        self.flatten = Flatten()
        self.relu = ReLU()
        self.dense = Dense(32, activation='relu')
        self.dense_out = Dense(5, activation='softmax')

    def routine(self, x):
        input_param = x
        x = self.conv_relu(x)
        x = self.conv_raw(x)
        x = add([input_param, x])
        x = self.relu(x)
        return self.maxpool(x)

    def call(self, x):
        x = self.conv_in(x)
        for _ in range(5):
            x = self.routine(x)

        x = self.flatten(x)

        x = self.dense(x)
        return self.dense_out(x)


model = MyModel()

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 75

train_loss_results = []
train_accuracy_results = []

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    train_loss_results.append(train_loss.result())
    train_accuracy_results.append(train_accuracy.result())

    template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
    if (epoch - 1) % 5 == 0:
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

# 그래프를 그리자
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('훈련 지표')

axes[0].set_ylabel("손실", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("정확도", fontsize=14)
axes[1].set_xlabel("에포크", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# 리샘플링을 해보자
C0 = np.argwhere(y == 0).flatten()
C1 = np.argwhere(y == 1).flatten()
C2 = np.argwhere(y == 2).flatten()
C3 = np.argwhere(y == 3).flatten()
C4 = np.argwhere(y == 4).flatten()

subC0 = np.random.choice(C0, 800)
subC1 = np.random.choice(C1, 800)
subC2 = np.random.choice(C2, 800)
subC3 = np.random.choice(C3, 800)
subC4 = np.random.choice(C4, 800)
