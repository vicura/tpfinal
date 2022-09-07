import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # From https://www.tensorflow.org/guide/gpu

# LET'S BUILD A SIMPLE MODEL TO TEST GPU ACCELERATION
print('\nBuilding data\n')
tf.random.set_seed(42)
# Build data
N = 1e7
X = tf.random.uniform(shape=(1,int(N)),dtype='float32')
Y = tf.math.sin(X**2) + X**3

# Build model
print('\nCreating model\n')
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(1,activation='relu'),
])

# Compile model
model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae','mse'])

# Fit model
print('\nTraining model\n')
history_1 = model.fit(x=X,y=Y,epochs=10)

