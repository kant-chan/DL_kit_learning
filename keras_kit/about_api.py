import numpy as np
from keras import backend
from keras.models import Model, Sequential
from keras.layers import Dense, Input

##########
#  1
##########
# def rmse(y_true, y_pred):
# 	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# model = Sequential()
# model.add(Dense(2, input_dim=1))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam', metrics=[rmse])
# history = model.fit(X, X, epochs=10, batch_size=len(X))

#########
# 2
#########
# X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
dt = 0.1
X = np.arange(-30, 30, dt)
y_1 = np.sin(2 * np.pi * X)
y_2 = np.array([1.0 if i >= 0 else -1.0 for i in X])

batch_size = 50

inputs = Input(shape=(1,))
x = Dense(16)(inputs)
x = Dense(16)(x)
x = Dense(8)(x)
x_1 = Dense(8)(x)
x_2 = Dense(4)(x_1)
outputs_1 = Dense(1, name='outputs_1')(x_2)
x = Dense(4)(x)
outputs_2 = Dense(1, name='outputs_2')(x)

model = Model(inputs=inputs, outputs=[outputs_1, outputs_2])
# model.name = 'fuck'  # default is model_1
model.compile(optimizer='adam',
              loss='mse')

model.fit(X, [y_1, y_2], batch_size=batch_size, epochs=10)

X_val = np.array([0.05, 0.78, 0.24, 0.45])
y_pred = model.predict(X_val, batch_size=len(X_val))
print(y_pred)