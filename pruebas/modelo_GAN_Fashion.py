# !pip install tensorflow matplotlib tensorflow-datasets ipywidgets

# !pip list

# Brining in tensorflow datasets for fashion mnist
import tensorflow_datasets as tfds
# Bringing in matplotlib for viz stuff
from matplotlib import pyplot as plt

# Use the tensorflow datasets api to bring in the data source
ds = tfds.load('fashion_mnist', split='train')

ds.as_numpy_iterator().next().keys()
ds.as_numpy_iterator().next()['label']

import numpy as np

# El iterador del dataset devuelve un diccionario
dataiterator = ds.as_numpy_iterator()

print(type(dataiterator.next()))
dataiterator.next().keys() #Con esto tenemos imágenes y etiquetas

# Getting data out of the pipeline  devuelbe el código de color de los picces por línea
print(dataiterator.next()['image'].shape)
img_tensor = dataiterator.next()['image'][0] # una sola fila o columna
print(img_tensor)

# Setup the subplot formatting
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# Loop four times and get images
for idx in range(4):
    # Grab an image and label
    sample = dataiterator.next()
    # Plot the image using a specific subplot
    ax[idx].imshow(np.squeeze(sample['image']))
    # Appending the image label as the plot title
    ax[idx].title.set_text(sample['label'])
    
# Scale and return images only
def scale_images(data):
    image = data['image']
    return image / 255

# Reload the dataset
ds = tfds.load('fashion_mnist', split='train')
# Running the dataset through the scale_images preprocessing step
ds = ds.map(scale_images)
# Cache the dataset for that batch
ds = ds.cache()
# Shuffle it up
ds = ds.shuffle(60000)
# Batch into 128 images per sample
ds = ds.batch(128)
# Reduces the likelihood of bottlenecking
ds = ds.prefetch(64)

ds.as_numpy_iterator().next().shape
# 128 imagenes de 28*28 en blanco y negro


# 3. Build Neural Network

# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Sequential
# Bring in the layers for the neural network
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D



# 3.2 Build Generator

def build_generator():
    model = Sequential()

    # Takes in random values and reshapes it to 7x7x128
    # Beginnings of a generated image
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))

    # Upsampling block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Upsampling block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional block 1
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional block 2
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Conv layer to get to one channel
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    return model


generator = build_generator()


generator.summary()


img = generator.predict(np.random.randn(4,128,1))


# Generate new fashion
img = generator.predict(np.random.randn(4,128,1))
# Setup the subplot formatting
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# Loop four times and get images
for idx, img in enumerate(img):
    # Plot the image using a specific subplot
    ax[idx].imshow(np.squeeze(img))
    # Appending the image label as the plot title
    ax[idx].title.set_text(idx)
    


# 3.3 Build Discriminator


def build_discriminator():
    model = Sequential()

    # First Conv Block
    model.add(Conv2D(32, 5, input_shape = (28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Second Conv Block
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Third Conv Block
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Fourth Conv Block
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model



discriminator = build_discriminator()


discriminator.summary()


img.shape


img = generator.predict(np.random.randn(4,128,1))
discriminator.predict(img)



# 4. Construct Training Loop

# 4.1 Setup Losses and Optimizers

# Adam is going to be the optimizer for both
from tensorflow.keras.optimizers import Adam
# Binary cross entropy is going to be the loss for both
from tensorflow.keras.losses import BinaryCrossentropy



g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()


# 4.2 Build Subclassed Model

# Importing the base model class to subclass our training step
from tensorflow.keras.models import Model

# Bueno, esta es la parte más dificil de entender ¿Cómo se entrenan ambas Redes Neuronales?


class FashionGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        # Pass through args and kwargs to base class
        super().__init__(*args, **kwargs)
        # esto permite que la clase FashionGAN pueda heredar argumentos y
        # argumentos clave de su clase "superior", en este caso
        # tf.keras.model

        # Create attributes for gen and disc
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):

        # Aquí sobrescribimos el compile de tf.keras.model y lo adaptamos a
        # nuestro modelo que tiene dos redes neuronales.

        # Compile with base class
        # esto permite que si se le pasan argumentos o argumentos clave al
        # .compile de FashionGan como "metrics=..." o "lose_weights=..."
        # estos se suban a su clase superior, es decir al compile de tf.keras.model
        super().compile(*args, **kwargs)

        # Create attributes for losses and optimizers
        # aquí simplemente creamos algunos atributos para la clase
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        # Get the data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        # Train the discriminator
        # todas las operaciones que hagamos con esto se empiezan a guardar
        # para despues calcular el grandiente
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Add some noise to the TRUE outputs
            # se añade algo de ruido a las etiquetas, esto hace que el
            # entrenamiento del discriminador se vuelva más robusto
            # es decir, que empiece a "dudar" de las decisiones que toma.
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate loss - BINARYCROSS
            # calculamos el Binarycrossentropy del discriminador
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # Apply backpropagation - nn learn
        # con esto calculamos el grandiente
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)

        # con esto aplicamos la retropropagación
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate some new images
            gen_images = self.generator(tf.random.normal((128,128,1)), training=True)

            # Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)

            # Calculate loss - trick to training to fake out the discriminator
            # Premiamos si detecta un cero:
            # Se usa una función de pérdida BinaryCrossentropy
            # que da un valor más bajo cuando las predicciones se acercan a 0
            # (o sea, cuando el generador engaña al discriminador).
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        # Apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss":total_d_loss, "g_loss":total_g_loss}




# Create instance of subclassed model
fashgan = FashionGAN(generator, discriminator)


# Compile the model
fashgan.compile(g_opt, d_opt, g_loss, d_loss)


# 4.3 Build Callback

import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback


os.mkdir("images")


class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))


# Ejecutar para cargar el generador y el discriminador guardados en el Drive 
# (yo lo querre guardado en mi ruta para modelos guardados: C:\Users\iblan\Desktop\Proyecto_final\assets\models)


# 4.3 Train

fashgan = FashionGAN(generator, discriminator)
# Compile the model
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

# Recommend 2000 epochs
hist = fashgan.fit(ds, epochs=25, callbacks=[ModelMonitor()])


# 4.4 Review Performance


plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()



# 5. Test Out the Generator


generator.load_weights('/content/drive/MyDrive/Colab Notebooks/generatormodel.h5')


imgs = generator.predict(tf.random.normal((16, 128, 1)))


fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10,10))
for r in range(4):
    for c in range(4):
        ax[r][c].imshow(imgs[(r+1)*(c+1)-1])
        
        
# 5.2 Save the Model

generator.save('generator.h5')
discriminator.save('discriminator.h5')






