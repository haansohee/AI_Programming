import tensorflow as tf
import numpy as np

# XOR
x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([[0], [1], [1], [0]])

# OR
# x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
# y = np.array([[0], [1], [1], [1]])

# 2개의 뉴런을 가진 레이어 하나 더 추가
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(units=2, activation='sigmoid'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mse')

# (1) OR epoch = 1000
# history = model.fit(x, y, epochs=1000, batch_size=1)

# (1) OR epoch =3000
history = model.fit(x, y, epochs=3000, batch_size=1)

model.summary()

print("XOR 연산 epoch=3000 학습 예측 결과 : ", model.predict(x))
