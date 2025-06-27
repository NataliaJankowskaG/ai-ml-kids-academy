# train_model.py (Versión mejorada)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam # Importar Adam para ajustar learning_rate
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Para Data Augmentation
import numpy as np
import os

# Ruta donde guardar el modelo
MODEL_SAVE_PATH = r"C:\Users\iblan\Desktop\Proyecto_final\assets\models\deep_learning_model.h5"

def load_and_prepare_mnist_data():
    """Carga y preprocesa el dataset MNIST para el ejemplo de Deep Learning."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalizar las imágenes a un rango de 0-1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Añadir una dimensión para el canal de color (imágenes en escala de grises tienen 1 canal)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Convertir etiquetas a one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def build_and_train_model(x_train, y_train):
    """Construye y entrena un modelo de Red Neuronal Convolucional (CNN) mejorado."""
    model = Sequential([
        # Primera capa Conv2D + BatchNormalization + MaxPooling
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25), # Dropout después del pooling
        
        # Segunda capa Conv2D + BatchNormalization + MaxPooling
        Conv2D(64, (3, 3), activation='relu'), # Aumentamos filtros
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25), # Dropout después del pooling
        
        Flatten(), # Aplanar para las capas densas
        
        # Capa densa oculta
        Dense(128, activation='relu'), # Aumentamos neuronas
        BatchNormalization(),
        Dropout(0.5), # Dropout más alto en la capa densa
        
        # Capa de salida
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), # Optimizador Adam con learning rate explícito
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Entrenando el modelo...")

    # Data Augmentation (Opcional, descomentar para usar)
    # datagen = ImageDataGenerator(
    #     rotation_range=10,
    #     zoom_range=0.1,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1
    # )
    # # Ajustar el generador a los datos de entrenamiento
    # datagen.fit(x_train)
    # 
    # model.fit(datagen.flow(x_train, y_train, batch_size=64),
    #           epochs=15, # Más épocas con augmentation
    #           verbose=1)

    # Entrenamiento sin Data Augmentation (usar si no usas lo de arriba)
    model.fit(x_train, y_train, epochs=15, batch_size=64, verbose=1) # Aumento de épocas y batch_size

    print("Entrenamiento completado.")
    return model

if __name__ == "__main__":
    print("Iniciando el script de entrenamiento del modelo Deep Learning...")
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    x_train, y_train, x_test, y_test = load_and_prepare_mnist_data() # Cargamos test para evaluación final si quieres

    if x_train is not None:
        model = build_and_train_model(x_train, y_train)
        
        if model:
            # Evaluar el modelo en los datos de prueba
            print("\nEvaluando el modelo en los datos de prueba...")
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            print(f"Precisión en datos de prueba: {accuracy*100:.2f}%")
            
            try:
                model.save(MODEL_SAVE_PATH)
                print(f"Modelo guardado exitosamente en: {MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"Error al guardar el modelo: {e}")
        else:
            print("No se pudo entrenar el modelo.")
    else:
        print("No se pudieron cargar los datos MNIST para el entrenamiento.")