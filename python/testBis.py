import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape

# Paramètres
latent_dim = 100  # Dimension de l'espace latent
img_shape = (28, 28, 1)  # Taille de l'image générée

# Modèle générateur
def build_generator():
    model = Sequential()
    model.add(Dense(128 * 7 * 7, input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(Dense(128, activation='relu'))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid'))
    return model

# Générer des images aléatoires
def generate_images(generator, n_samples=1):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_images = generator.predict(noise)
    return generated_images

# Afficher des images générées
def show_generated_images(generated_images, n_cols=1):
    n_rows = int(np.ceil(len(generated_images) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), sharex=True, sharey=True)
    axs = axs.flatten()
    for img, ax in zip(generated_images, axs):
        ax.imshow(img[:, :, 0], cmap='gray')
        ax.axis('off')
    plt.show()

# Construire le générateur
generator = build_generator()

# Générer et afficher des images aléatoires
num_images_to_generate = 5
generated_images = generate_images(generator, num_images_to_generate)
show_generated_images(generated_images)
