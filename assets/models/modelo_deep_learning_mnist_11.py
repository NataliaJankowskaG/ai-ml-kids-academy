import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import os

# --- Configuración de la ruta para guardar el modelo ---
# Asegúrate de que esta ruta exista o créala.
MODEL_SAVE_PATH = r"C:\Users\iblan\Desktop\Proyecto_final\assets\models\mnist_dl_model.h5"

def load_and_prepare_mnist_data():
    """Carga y preprocesa el dataset MNIST."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalizar las imágenes a un rango de 0-1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Añadir una dimensión para el canal de color (imágenes en escala de grises tienen 1 canal)
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    
    # Convertir etiquetas a one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def build_and_train_deep_learning_model(x_train, y_train):
    """Construye y entrena un modelo de Red Neuronal Convolucional (CNN) simple."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Entrenando el modelo... Esto puede tardar unos minutos.")
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1) # Usar más épocas y batch_size para un mejor entrenamiento
    print("Entrenamiento completado.")
    return model

if __name__ == "__main__":
    print("Cargando y preparando datos MNIST...")
    x_train, y_train, x_test, y_test = load_and_prepare_mnist_data()
    print("Datos MNIST cargados.")

    if x_train is not None and y_train is not None:
        model = build_and_train_deep_learning_model(x_train, y_train)
        
        # Evaluar el modelo en el conjunto de prueba
        print("Evaluando el modelo en el conjunto de prueba...")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Precisión del modelo en el conjunto de prueba: {accuracy:.4f}")

        # Asegurarse de que el directorio exista
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        
        # Guardar el modelo
        model.save(MODEL_SAVE_PATH)
        print(f"Modelo guardado exitosamente en: {MODEL_SAVE_PATH}")
    else:
        print("No se pudieron cargar los datos MNIST. No se entrenará ni guardará el modelo.")