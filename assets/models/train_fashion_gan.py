import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, UpSampling2D, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img
from matplotlib import pyplot as plt
from PIL import Image

print("Iniciando script de entrenamiento de GAN Fashion MNIST...")

# ============================
# 1. Configurar rutas
# ============================

# Rutas proporcionadas por el usuario (absolutas)
model_dir = r'C:\Users\iblan\Desktop\Proyecto_final\assets\models'
image_dir = r'C:\Users\iblan\Desktop\Proyecto_final\assets\generated_images'

# Asegurarse de que las carpetas existen
os.makedirs(model_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

print(f"Los modelos se guardarán en: {model_dir}")
print(f"Las imágenes generadas durante el entrenamiento se guardarán en: {image_dir}")

# ============================
# 2. Preparar datos
# ============================

def scale_images(data):
    """Escala las imágenes de 0-255 a 0-1."""
    image = data['image']
    return tf.cast(image, tf.float32) / 255.0 # Asegurarse de que el tipo de dato es float32

print("Cargando y preparando el dataset Fashion MNIST...")
ds = tfds.load('fashion_mnist', split='train')
ds = ds.map(scale_images)
ds = ds.cache()
ds = ds.shuffle(60000) # Tamaño completo del dataset para un buen shuffle

# *** CAMBIO AQUI: AUMENTAR EL BATCH_SIZE PARA REDUCIR PASOS POR EPOCA ***
BATCH_SIZE = 512 # Aumentado de 128 a 512 para acelerar la época
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(tf.data.AUTOTUNE) # Optimización para pre-cargar batches

print(f"Dataset Fashion MNIST preparado con tamaño de lote: {BATCH_SIZE}")

# ============================
# 3. Generador
# ============================

def build_generator(latent_dim=128):
    """
    Construye el modelo Generador.
    input_dim: Dimensión del vector de ruido (latent space).
    """
    model = Sequential(name="Generator")
    # Capa densa inicial que expande el ruido a un tamaño adecuado para la remodelación
    # Aumentar la capacidad inicial del generador
    model.add(Dense(7 * 7 * 256, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization()) # CRÍTICO para la estabilidad de GANs

    # Remodelar a un tensor 3D
    model.add(Reshape((7, 7, 256))) # 7x7 es el tamaño inicial de las imágenes

    # Bloque de UpSampling y Convolución
    model.add(UpSampling2D()) # Duplica la resolución a 14x14
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(UpSampling2D()) # Duplica la resolución a 28x28
    model.add(Conv2D(64, 5, padding='same')) # Reducir filtros a medida que aumenta la resolución
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    # Capas convolucionales adicionales para refinar la imagen
    model.add(Conv2D(32, 4, padding='same')) # Capa convolucional final antes de la salida
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization()) # Última BatchNormalization antes de la salida de activación

    # Capa de salida con activación sigmoid para generar imágenes en el rango [0, 1]
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    return model

LATENT_DIM = 128 # Define la dimensión del vector de ruido
generator = build_generator(latent_dim=LATENT_DIM)
print("Modelo Generador construido.")
# generator.summary() # Descomenta para ver el resumen del modelo

# ============================
# 4. Discriminador
# ============================

def build_discriminator():
    """
    Construye el modelo Discriminador.
    input_shape: Forma de las imágenes (Fashion MNIST 28x28x1).
    """
    model = Sequential(name="Discriminator")
    # Capas convolucionales para extraer características de las imágenes
    # Primera capa Conv2D sin BatchNormalization
    model.add(Conv2D(64, 5, strides=(2, 2), padding='same', input_shape=(28, 28, 1))) # Recibe imágenes de 28x28x1
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3)) # Menos dropout para el discriminador

    model.add(Conv2D(128, 5, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, 5, padding='same')) # Sin strides para mantener info
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    # Aplanar para la capa densa de clasificación
    model.add(Flatten())
    model.add(Dropout(0.3)) # Último dropout antes de la capa de salida

    # Capa de salida con activación sigmoid para la clasificación binaria (real/falsa)
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = build_discriminator()
print("Modelo Discriminador construido.")
# discriminator.summary() # Descomenta para ver el resumen del modelo

# ============================
# 5. Subclase GAN
# ============================
# Esta subclase es CRÍTICA y debe ser idéntica a la que se use para cargar el modelo en Streamlit.
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

    @tf.function # Para acelerar el entrenamiento usando gráficos de TensorFlow
    def train_step(self, batch):
        real_images = batch

        # Generar imágenes falsas usando el Generador (sin entrenamiento del Generador aquí)
        fake_images = self.generator(tf.random.normal((BATCH_SIZE, LATENT_DIM)), training=False)

        # Entrenamiento del Discriminador
        with tf.GradientTape() as d_tape:
            # Clasificar imágenes reales y falsas
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)

            # Concatenar predicciones y crear etiquetas (con label smoothing)
            # Etiquetas para reales (0.9, comunmente) y falsas (0.0)
            y_real = tf.ones_like(yhat_real) * 0.9 # Etiqueta suavizada para imágenes reales
            y_fake = tf.zeros_like(yhat_fake)     # Etiqueta para imágenes falsas

            # Calcular la pérdida total del Discriminador
            loss_real = self.d_loss(y_real, yhat_real)
            loss_fake = self.d_loss(y_fake, yhat_fake)
            total_d_loss = (loss_real + loss_fake) / 2 # Promedio de las pérdidas

        # Aplicar gradientes al Discriminador
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Entrenamiento del Generador
        with tf.GradientTape() as g_tape:
            # Generar nuevas imágenes (ahora con entrenamiento del Generador)
            gen_images = self.generator(tf.random.normal((BATCH_SIZE, LATENT_DIM)), training=True)

            # El Generador quiere engañar al Discriminador, así que sus imágenes falsas
            # deberían ser clasificadas como "reales" (etiqueta 1 en la convención usual,
            # pero dado que el Discriminador recibe 0 para reales, el Generador quiere 0)
            predicted_labels = self.discriminator(gen_images, training=False) # Discriminador en modo inferencia

            # Calcular la pérdida del Generador (quiere que las predicciones sean 0, como imágenes reales)
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels) # Generador quiere 0 para sus generadas

        # Aplicar gradientes al Generador
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}

print("Clase FashionGAN definida.")

# ============================
# 6. Callback para imágenes y guardado de modelo
# ============================

class ModelMonitor(Callback):
    """
    Callback para generar y guardar imágenes de muestra al final de cada época,
    y guardar el modelo del Generador y Discriminador.
    """
    def __init__(self, output_image_dir, output_model_dir, num_img=3, latent_dim=128):
        super().__init__()
        self.output_image_dir = output_image_dir
        self.output_model_dir = output_model_dir # Nueva ruta para guardar modelos
        self.num_img = num_img
        self.latent_dim = latent_dim
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_model_dir, exist_ok=True) # Asegurarse de que la carpeta de modelos existe
        # Generar un conjunto fijo de vectores de ruido para seguimiento consistente
        self.fixed_latent_vectors = tf.random.normal((self.num_img, self.latent_dim))

    def on_epoch_end(self, epoch, logs=None):
        # 1. Generar y guardar imágenes de muestra
        generated_images = self.model.generator(self.fixed_latent_vectors, training=False)
        generated_images = (generated_images * 255).numpy().astype(np.uint8)

        fig, axes = plt.subplots(1, self.num_img, figsize=(self.num_img * 3, 3))
        for i in range(self.num_img):
            img_array = generated_images[i, :, :, 0]
            axes[i].imshow(img_array, cmap='gray')
            axes[i].axis('off')

            img_pil = Image.fromarray(img_array)
            img_pil.save(os.path.join(self.output_image_dir, f'generated_img_epoch_{epoch+1}_{i}.png'))

        plt.suptitle(f'Imágenes Generadas en Época {epoch+1}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_image_dir, f'epoch_{epoch+1}_summary.png'))
        plt.close(fig)
        print(f"Imágenes de muestra guardadas para la época {epoch+1} en {self.output_image_dir}")

        # 2. Guardar el modelo del Generador en cada época
        generator_epoch_save_path = os.path.join(self.output_model_dir, f'fashion_mnist_generator_epoch_{epoch+1:03d}.h5')
        try:
            self.model.generator.save(generator_epoch_save_path)
            print(f"Generador guardado exitosamente para la época {epoch+1} en: {generator_epoch_save_path}")
        except Exception as e:
            print(f"Error al guardar el generador para la época {epoch+1}: {e}")

        # 3. Guardar el modelo del Discriminador en cada época
        discriminator_epoch_save_path = os.path.join(self.output_model_dir, f'fashion_mnist_discriminator_epoch_{epoch+1:03d}.h5')
        try:
            self.model.discriminator.save(discriminator_epoch_save_path)
            print(f"Discriminador guardado exitosamente para la época {epoch+1} en: {discriminator_epoch_save_path}")
        except Exception as e:
            print(f"Error al guardar el discriminador para la época {epoch+1}: {e}")

print("Callback ModelMonitor definido.")

# ============================
# 7. Compilar y Entrenar
# ============================

# Optimizadores
g_opt = Adam(learning_rate=0.0001, beta_1=0.5)
d_opt = Adam(learning_rate=0.00005, beta_1=0.5)

# Funciones de pérdida
g_loss = BinaryCrossentropy(from_logits=False)
d_loss = BinaryCrossentropy(from_logits=False)

# Crear la instancia de la GAN
fashgan = FashionGAN(generator, discriminator)
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

# Crear el callback de monitoreo
monitor = ModelMonitor(output_image_dir=image_dir, output_model_dir=model_dir, latent_dim=LATENT_DIM)

EPOCHS = 50 # Puedes ajustar esto, pero las épocas serán mucho más rápidas ahora
print(f"Iniciando entrenamiento por {EPOCHS} épocas...")

# *** CAMBIO AQUI: AÑADIR steps_per_epoch PARA ACELERAR AUN MAS LA EPOCA ***
# Esto limitará cuántos batches se procesan por época, haciendo la época más corta.
# Por ejemplo, 100 pasos significa que cada época solo procesará 100 * BATCH_SIZE muestras.
# Con BATCH_SIZE = 512, esto sería 100 * 512 = 5120 muestras por época.
STEPS_PER_EPOCH = 100 # Puedes ajustar este valor
history = fashgan.fit(ds, epochs=EPOCHS, callbacks=[monitor], steps_per_epoch=STEPS_PER_EPOCH)
print("Entrenamiento completado.")

# ============================
# 8. Guardar modelos (opcional, ya se guardan en el callback)
# ============================

# Nombres de archivo sugeridos
generator_save_path = os.path.join(model_dir, 'fashion_mnist_generator_final.h5')
discriminator_save_path = os.path.join(model_dir, 'fashion_mnist_discriminator_final.h5')

# Puedes seguir guardando el modelo final si quieres un archivo sin el número de época
try:
    generator.save(generator_save_path)
    print(f"Generador final guardado exitosamente en: {generator_save_path}")
except Exception as e:
    print(f"Error al guardar el generador final: {e}")

try:
    discriminator.save(discriminator_save_path)
    print(f"Discriminador final guardado exitosamente en: {discriminator_save_path}")
except Exception as e:
    print(f"Error al guardar el discriminador final: {e}")

# ============================
# 9. Visualizar pérdidas
# ============================

print("Generando gráfico de pérdidas de entrenamiento...")
plt.figure(figsize=(10, 6))
plt.plot(history.history['d_loss'], label='Pérdida del Discriminador (d_loss)')
plt.plot(history.history['g_loss'], label='Pérdida del Generador (g_loss)')
plt.title("Pérdidas de Entrenamiento de GAN Fashion MNIST")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.grid(True)
plt.show()

print("Script de entrenamiento finalizado.")

# ============================
# 10. Probar el generador
# ============================

print("Generando imágenes finales para prueba visual...")

def mostrar_imagenes_generadas(modelo_generador, latent_dim=128, n=5):
    """
    Genera y muestra imágenes con el generador entrenado.
    También las guarda en la carpeta de imágenes si se desea.
    """
    ruido = tf.random.normal((n, latent_dim))  # Generar vectores de ruido
    imagenes_generadas = modelo_generador(ruido, training=False)

    imagenes_generadas = imagenes_generadas.numpy() * 255.0
    imagenes_generadas = imagenes_generadas.astype(np.uint8)

    plt.figure(figsize=(n * 2, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(imagenes_generadas[i, :, :, 0], cmap='gray')
        plt.axis('off')

        # Guardar también la imagen si quieres mantener registro
        img_pil = Image.fromarray(imagenes_generadas[i, :, :, 0])
        img_pil.save(os.path.join(image_dir, f'test_final_{i}.png'))

    plt.suptitle("Muestras Generadas por el Generador Entrenado")
    plt.tight_layout()
    plt.show()

# Mostrar y guardar 5 imágenes de prueba
mostrar_imagenes_generadas(generator, latent_dim=LATENT_DIM, n=5)