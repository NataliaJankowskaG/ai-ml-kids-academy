# ==============================================
# GAN Fashion MNIST - Entrenamiento y Guardado
# ==============================================

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, UpSampling2D, LeakyReLU, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img
from matplotlib import pyplot as plt

# ============================
# 1. Configurar rutas
# ============================

model_dir = r'C:\Users\iblan\Desktop\Proyecto_final\assets\models'
image_dir = r'C:\Users\iblan\Desktop\Proyecto_final\assets\generated_images'

os.makedirs(model_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# ============================
# 2. Preparar datos
# ============================

def scale_images(data):
    image = data['image']
    return image / 255.0

ds = tfds.load('fashion_mnist', split='train')
ds = ds.map(scale_images)
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(128)
ds = ds.prefetch(64)

# ============================
# 3. Generador
# ============================

def build_generator():
    model = Sequential()
    model.add(Dense(7 * 7 * 128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))
    return model

generator = build_generator()

# ============================
# 4. Discriminador
# ============================

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator()

# ============================
# 5. Subclase GAN
# ============================

class FashionGAN(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss):
        super().compile()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128)), training=False)

        # Discriminator
        with tf.GradientTape() as d_tape:
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
            y_realfake = tf.concat([
                tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)
            ], axis=0)
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Generator
        with tf.GradientTape() as g_tape:
            gen_images = self.generator(tf.random.normal((128, 128)), training=True)
            predicted_labels = self.discriminator(gen_images, training=False)
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}

# ============================
# 6. Callback para imágenes
# ============================

class ModelMonitor(Callback):
    def __init__(self, output_dir, num_img=3, latent_dim=128):
        super().__init__()
        self.output_dir = output_dir
        self.num_img = num_img
        self.latent_dim = latent_dim
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        latent_vectors = tf.random.normal((self.num_img, self.latent_dim))
        generated_images = self.model.generator(latent_vectors)
        generated_images *= 255
        generated_images = generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join(self.output_dir, f'generated_img_{epoch}_{i}.png'))

# ============================
# 7. Compilar y Entrenar
# ============================

g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

fashgan = FashionGAN(generator, discriminator)
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

monitor = ModelMonitor(output_dir=image_dir)

history = fashgan.fit(ds, epochs=25, callbacks=[monitor])

# ============================
# 8. Guardar modelos
# ============================

generator.save(os.path.join(model_dir, 'generator.h5'))
discriminator.save(os.path.join(model_dir, 'discriminator.h5'))

# ============================
# 9. Visualizar pérdidas
# ============================

plt.plot(history.history['d_loss'], label='d_loss')
plt.plot(history.history['g_loss'], label='g_loss')
plt.title("Training Losses")
plt.legend()
plt.show()

