import tensorflow as tf
import numpy as np

# Učitava zabeležene podatke
observations = np.load('observations.npy', allow_pickle=True)
actions = np.load('actions.npy', allow_pickle=True)

# Određuje oblik ulaza i broj mogućih akcija
input_shape = observations.shape[1:]  # Oblik jedne opservacije
num_actions = len(np.unique(actions))  # Broj jedinstvenih akcija

# Definiše model neuronske mreže
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),  # Izravnava ulaz
    tf.keras.layers.Dense(128, activation='relu'),     # Skriveni sloj sa 128 neurona
    tf.keras.layers.Dense(64, activation='relu'),      # Skriveni sloj sa 64 neurona
    tf.keras.layers.Dense(num_actions, activation='softmax')  # Izlazni sloj sa brojem neurona jednakim broju akcija
])

# Kompajlira model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trenira model
model.fit(observations, actions, epochs=100, batch_size=32)

# Čuva trenirani model
model.save('trained_model.h5')

print("Trening je završen i model je sačuvan.")
