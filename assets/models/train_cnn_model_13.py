import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # Importaciones añadidas
import os
import numpy as np
import matplotlib.pyplot as plt # Importar matplotlib para las gráficas

print("TensorFlow Version:", tf.__version__)

# --- Rutas de archivos y directorios ---
# Asume que este script se ejecuta desde la raíz de tu proyecto (C:\Users\iblan\Desktop\Proyecto_final\)
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) # La raíz del proyecto
MODEL_DIR = os.path.join(BASE_PROJECT_DIR, 'assets', 'models')
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_defect_detector_model.h5')
DATASET_BASE_DIR = os.path.join(BASE_PROJECT_DIR, 'datasets', 'piece_defects')

# Asegúrate de que el directorio para guardar el modelo exista
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Parámetros para el modelo CNN y el entrenamiento ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20 # Aumentamos las épocas máximas para dar margen a EarlyStopping
# PACIENCIA_EARLY_STOPPING = 3 # Puedes ajustar este valor si ves que el modelo sigue sobreajustándose muy rápido

# --- Función para crear el modelo CNN ---
def create_cnn_model():
    """Define y crea una arquitectura CNN sencilla para clasificación binaria."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), # 3 para RGB
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.6), # Tasa de Dropout aumentada de 0.5 a 0.6 para combatir sobreajuste
        Dense(1, activation='sigmoid') # Sigmoid para clasificación binaria (defecto/no defecto)
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

# --- Función para cargar y preprocesar los datos ---
def load_and_preprocess_cnn_data(dataset_path):
    """Carga y preprocesa los datos de imagen para la CNN usando ImageDataGenerator.
    Ahora carga datos de train y validación.
    """
    train_dir = os.path.join(dataset_path, 'train')
    valid_dir = os.path.join(dataset_path, 'valid') # Nueva carpeta de validación

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        print(f"ERROR: ¡Los directorios de entrenamiento o validación del dataset no existen! Por favor, verifica la ruta: '{dataset_path}' y que contenga 'train' y 'valid'.")
        print("Asegúrate de que tienes subcarpetas 'good' y 'defect' dentro de 'train' y 'valid'.")
        return None, None, None

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,    # Ligeramente aumentado para más variabilidad
        width_shift_range=0.15, # Ligeramente aumentado
        height_shift_range=0.15, # Ligeramente aumentado
        shear_range=0.1,
        zoom_range=0.15,      # Ligeramente aumentado
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Solo escalado para validación
    valid_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True
        )
        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False # No mezclar para una evaluación consistente
        )
        class_names = list(train_generator.class_indices.keys())
        print(f"Clases detectadas en el dataset: {class_names}. Mapeo: {train_generator.class_indices}")
        return train_generator, valid_generator, class_names # Devolvemos valid_generator
    except Exception as e:
        print(f"ERROR: Error al cargar datos con ImageDataGenerator. Asegúrate de que '{train_dir}' y '{valid_dir}' contengan subcarpetas (ej. 'good' y 'defect'). Error: {e}")
        return None, None, None

# --- Ejecución del entrenamiento ---
if __name__ == "__main__":
    print(f"Iniciando el script de entrenamiento de CNN. El modelo se guardará en: {CNN_MODEL_PATH}")
    print(f"Buscando datos en: {DATASET_BASE_DIR}")

    # 1. Cargar datos de entrenamiento y validación
    train_generator, valid_generator, class_names = load_and_preprocess_cnn_data(DATASET_BASE_DIR)

    if train_generator is None or valid_generator is None:
        print("No se pudieron cargar los datos de entrenamiento o validación. Abortando entrenamiento.")
    else:
        # 2. Crear o cargar el modelo
        model = create_cnn_model()

        # --- Definir Callbacks para el entrenamiento ---
        # EarlyStopping: Detiene el entrenamiento si la val_loss no mejora después de 'patience' épocas
        early_stopping = EarlyStopping(
            monitor='val_loss',          # Monitoreamos la pérdida de validación
            patience=3,                  # Número de épocas sin mejora antes de detener
            restore_best_weights=True,   # Restaura los pesos del modelo de la mejor época
            verbose=1                    # Muestra un mensaje cuando se detiene
        )

        # ModelCheckpoint: Guarda el mejor modelo automáticamente durante el entrenamiento
        model_checkpoint = ModelCheckpoint(
            filepath=CNN_MODEL_PATH,     # Ruta donde se guardará el modelo
            monitor='val_loss',          # Monitoreamos la pérdida de validación
            save_best_only=True,         # Solo guarda si el rendimiento es mejor que el anterior
            mode='min',                  # Queremos minimizar la pérdida de validación
            verbose=1                    # Muestra un mensaje cuando guarda un modelo
        )

        # Agrupa los callbacks
        callbacks_list = [early_stopping, model_checkpoint]

        # 3. Entrenar el modelo
        print(f"\nEntrenando el modelo por un máximo de {EPOCHS} épocas (con Early Stopping)...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=valid_generator, # Usamos valid_generator para validación
            callbacks=callbacks_list,        # ¡Añadimos los callbacks aquí!
            verbose=1
        )

        # El modelo ya ha sido guardado por ModelCheckpoint, y EarlyStopping ha restaurado los mejores pesos.
        print(f"\n¡Entrenamiento finalizado! El mejor modelo ha sido guardado automáticamente en '{CNN_MODEL_PATH}'.")

        # Opcional: Mostrar gráficas de pérdida y precisión del entrenamiento
        print("\nResultados del entrenamiento:")
        if history:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
            # Asegúrate de que 'val_accuracy' existe en el historial antes de intentar plotearla
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Precisión Validación')
            plt.title('Precisión del Modelo')
            plt.xlabel('Época')
            plt.ylabel('Precisión')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
            # Asegúrate de que 'val_loss' existe en el historial antes de intentar plotearla
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Pérdida Validación')
            plt.title('Pérdida del Modelo')
            plt.xlabel('Época')
            plt.ylabel('Pérdida')
            plt.legend()

            plt.tight_layout()
            plt.show() # Esto mostrará la gráfica en una ventana aparte
            print("Se han generado gráficas de precisión y pérdida.")

    print("\nScript de entrenamiento finalizado.")