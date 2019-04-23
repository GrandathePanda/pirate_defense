from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

class Agent():
  def __init__(self):
    self.memory  = deque(maxlen=100000)
    
    self.gamma = 0.95
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.0333
    self.model = self.create_model()
    self.target_model = self.create_model()
    self.actions = range(0, 100)
    self.tau = 0.125

  def create_model(self):
    model = Sequential()
    state_shape  = (10, 10)
    model.add(Dense(128, input_dim=state_shape[0], 
        activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(100))
    model.compile(loss="mean_squared_error",
        optimizer=Adam(lr=self.learning_rate))
    return model

  def act(self, state):
    if np.random.random() < self.epsilon:
        return random.choice(self.actions)
    return np.argmax(self.model.predict(state)[0])

  def remember(self, state, action, reward, new_state, done):
    self.memory.append([state, action, reward, new_state, done])

  def replay(self, batch_size):
    samples = random.sample(self.memory, batch_size)

    for state, action, reward, new_state, done in samples:
        target = self.target_model.predict(state)

        if done:
            target[0][action] = reward
        else:
            Q_future = max(self.target_model.predict(new_state)[0])
            target[0][action] = reward + Q_future * self.gamma

        self.model.fit(state, target, epochs=1, verbose=0)

    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon_min, self.epsilon)

  def target_train(self):
    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
    self.target_model.set_weights(target_weights)

  def save_model(self, fn):
    self.model.save(fn)