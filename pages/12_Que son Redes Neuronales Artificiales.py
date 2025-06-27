import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from streamlit_lottie import st_lottie
from openai import OpenAI
from gtts import gTTS
import io
import random
import time
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="¿Qué son las Redes Neuronales Artificiales?",
    layout="wide"
)

# ---- Función para cargar animación Lottie desde un archivo local ----
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo Lottie en la ruta: {filepath}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: El archivo Lottie '{filepath}' no es un JSON válido o está corrupto.")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar el archivo Lottie '{filepath}': {e}. Asegúrate de que el archivo no esté corrupto y sea un JSON válido.")
        return None

# --- Rutas a Lottie (¡NECESITAS DESCARGAR ESTOS ARCHIVOS!) ---
# Sugerencias: Busca en lottiefiles.com "neural network", "brain", "AI", "neurons", "robot"
LOTTIE_NEURONS_PATH = os.path.join("assets", "lottie_animations", "neuron_network.json")
LOTTIE_BRAIN_PATH = os.path.join("assets", "lottie_animations", "brain_ai.json")
LOTTIE_TRAINING_PATH = os.path.join("assets", "lottie_animations", "data_training.json")
LOTTIE_ROBOT_PATH = os.path.join("assets", "lottie_animations", "robot.json")

# --- Ruta a la imagen de Red Neuronal local ---
NEURAL_NETWORK_IMAGE_PATH = os.path.join("assets", "imagenes", "neural_network_diagram.jpg") # O una imagen más específica de CNN si tienes.

# --- Ruta al modelo CNN pre-entrenado ---
MODEL_PATH = os.path.join("assets", "models", "mnist_cnn_model.h5")

# --- Cargar el modelo CNN pre-entrenado (una vez) ---
@st.cache_resource
def load_mnist_model():
    """Carga el modelo CNN de MNIST y lo cachea."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"¡Error! No se encontró el modelo CNN en la ruta: {MODEL_PATH}")
        st.warning("Por favor, asegúrate de haber entrenado y guardado el modelo TensorFlow/Keras con un script separado y guardarlo en la ruta especificada.")
        return None
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.warning("Asegúrate de que el archivo del modelo no esté corrupto y que TensorFlow esté correctamente instalado.")
        return None

# Cargar el modelo al inicio de la aplicación
model_mnist_cnn = load_mnist_model()


# --- Configuración de la API de OpenAI ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai_api_key = None
    st.warning("Advertencia: La clave de la API de OpenAI no está configurada en `secrets.toml`. El chatbot Neo no funcionará.")

client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Inicializar estados para el "laboratorio" de RNA interactivo
if 'rna_lab_config' not in st.session_state:
    st.session_state.rna_lab_config = {
        'conv_layers': 1, # Número de capas convolucionales simuladas
        'dense_layers': 1, # Número de capas densas simuladas (Fully Connected)
        'filters_per_conv_layer': 16, # Número de filtros en la primera capa convolucional
        'neurons_per_dense_layer': 128, # Neuronas en la primera capa densa
        'epochs': 5,
        'current_accuracy': 0.1, # Precisión inicial (adivinar 1 de 10 dígitos)
        'current_loss': 2.3, # Pérdida inicial
        'drawn_digit_input': None, # Para guardar la imagen dibujada por el usuario
        'predicted_digit': None, # Almacenar el dígito predicho real
        'prediction_probabilities': None, # Almacenar las probabilidades reales
        'simulated_activations': {} # Para guardar las activaciones simuladas
    }

st.subheader("¡Descubre los 'cerebros' de la Inteligencia Artificial!")

st.write("---")

# Sección 1: Introducción: ¿Qué son las Redes Neuronales?
st.header("¡El Cerebro Mágico de la IA: Redes Neuronales!")
st.markdown("""
Imagina que la Inteligencia Artificial tiene su propio "cerebro". No es un cerebro de verdad,
¡sino uno hecho de matemáticas y códigos! Este "cerebro" se llama **Red Neuronal Artificial (RNA)**.

Piensa en cómo aprendes tú: ves cosas, escuchas, tocas, y tu cerebro hace conexiones para entender el mundo.
Una Red Neuronal Artificial hace algo parecido, pero con ¡**datos**!

Son como un equipo de pequeños detectives conectados, que trabajan juntos para reconocer patrones,
tomar decisiones o incluso crear cosas nuevas, ¡todo a partir de la información que les damos!
""")

col_intro_left_rna, col_intro_right_rna = st.columns([1, 1])
with col_intro_left_rna:
    lottie_neurons = load_lottiefile(LOTTIE_NEURONS_PATH)
    if lottie_neurons:
        st_lottie(lottie_neurons, height=200, width=200, key="rna_intro")
    else:
        st.info("Consejo: Descarga una animación de red neuronal (ej. 'neuron_network.json') de LottieFiles.com y ponla en 'assets/lottie_animations/'.")
with col_intro_right_rna:
    st.image(NEURAL_NETWORK_IMAGE_PATH, caption="Un diagrama simple de una Red Neuronal", width=300)
    st.markdown("¡Nuestro objetivo es entender cómo estos 'cerebritos' funcionan!")

st.write("---")

# Sección 2: Los Componentes Secretos de una RNA (Ajustar para mencionar CNNs)
st.header("Los Ingredientes Secretos de las Redes Neuronales (¡Especialmente para Imágenes!)")
st.markdown("""
Una Red Neuronal está hecha de muchas partes, como un edificio con ladrillos y cables:

1.  **Neuronas (o Nodos):** Son como los ladrillos individuales. Cada una recibe información, la procesa y envía un resultado.
    ¡Son las unidades básicas de procesamiento!
2.  **Capas:** Las neuronas se agrupan en "capas". Hay una **capa de entrada** (donde entran los datos),
    una o más **capas ocultas** (donde ocurre la "magia" del procesamiento) y una **capa de salida** (donde obtenemos el resultado final).
    * **¡Novedad para las imágenes! Capas Convolucionales (Conv):** Son como "detectives" que buscan características específicas en pequeñas partes de una imagen (bordes, curvas, texturas). Tienen "filtros" que se mueven por toda la imagen.
    * **Capas de Agrupación (Pooling):** Reducen el tamaño de la imagen, manteniendo las características importantes. ¡Como resumir una historia muy larga!
    * **Capas Densas (Fully Connected):** Las neuronas aquí están conectadas a *todas* las neuronas de la capa anterior. Son las que toman las características encontradas por las capas convolucionales y hacen la decisión final.
3.  **Conexiones y Pesos:** Cada neurona está conectada a otras neuronas. Cada conexión tiene un "peso",
    que es un número que indica la importancia de esa conexión. Es como el volumen de un sonido:
    un peso alto significa que esa conexión es muy importante.
4.  **Sesgos (Bias):** Es como un "empujoncito" extra que recibe una neurona, para que le sea más fácil (o más difícil) activarse.
5.  **Función de Activación:** Imagina que una neurona decide si "dispara" o no. La función de activación
    es la regla que decide si el resultado final de la neurona es lo suficientemente fuerte como para pasar a la siguiente.
    Es como un interruptor de luz: ¿se enciende o no?
""")

col_components_left_rna, col_components_right_rna = st.columns([1, 1])
with col_components_left_rna:
    lottie_brain = load_lottiefile(LOTTIE_BRAIN_PATH)
    if lottie_brain:
        st_lottie(lottie_brain, height=180, width=180, key="rna_brain")
    else:
        st.info("Consejo: Descarga una animación de cerebro IA (ej. 'brain_ai.json') de LottieFiles.com.")
with col_components_right_rna:
    lottie_training = load_lottiefile(LOTTIE_TRAINING_PATH)
    if lottie_training:
        st_lottie(lottie_training, height=180, width=180, key="rna_training")
    else:
        st.info("Consejo: Descarga una animación de entrenamiento de datos (ej. 'data_training.json') de LottieFiles.com.")

st.write("---")

# --- Sección 3: ¡Tu Laboratorio de Construcción y Entrenamiento de CNN para Dígitos MNIST! ---
st.header("Tu Laboratorio de Construcción de CNN: ¡Enseña a tu IA a Leer Dígitos!")
st.markdown("""
¡Aquí es donde te conviertes en un arquitecto de la IA! Vamos a construir una Red Neuronal Convolucional (CNN) muy simplificada.
Esta red intentará reconocer los **dígitos escritos a mano** (como los del famoso dataset MNIST).

**¡Tú controlas la estructura de la red y cómo se 'entrena' para que aprenda mejor!**
""")

col_cnn_controls, col_cnn_viz = st.columns([1, 2])

with col_cnn_controls:
    st.markdown("### Configura tu Red Neuronal Convolucional (CNN):")

    num_conv_layers = st.slider(
        "Número de Capas Convolucionales:",
        min_value=1, max_value=3, value=st.session_state.rna_lab_config['conv_layers'], step=1,
        help="Las capas convolucionales son las primeras en procesar la imagen, buscando características."
    )
    st.session_state.rna_lab_config['conv_layers'] = num_conv_layers

    filters_per_conv_layer = st.slider(
        "Número de Filtros por Capa Conv.:",
        min_value=8, max_value=32, value=st.session_state.rna_lab_config['filters_per_conv_layer'], step=4,
        help="Cada filtro es como un 'detective' que busca un patrón específico en la imagen (bordes, líneas, etc.). Más filtros, más patrones detectados."
    )
    st.session_state.rna_lab_config['filters_per_conv_layer'] = filters_per_conv_layer

    num_dense_layers = st.slider(
        "Número de Capas Densas (Finales):",
        min_value=1, max_value=2, value=st.session_state.rna_lab_config['dense_layers'], step=1,
        help="Estas capas toman las características aprendidas y hacen la clasificación final (qué dígito es)."
    )
    st.session_state.rna_lab_config['dense_layers'] = num_dense_layers

    neurons_per_dense_layer = st.slider(
        "Neuronas por Capa Densa (Final):",
        min_value=64, max_value=256, value=st.session_state.rna_lab_config['neurons_per_dense_layer'], step=32,
        help="Más neuronas en estas capas pueden procesar más combinaciones de características."
    )
    st.session_state.rna_lab_config['neurons_per_dense_layer'] = neurons_per_dense_layer

    num_epochs = st.slider(
        "Épocas de Entrenamiento (¡Cuántas veces la red 'revisa' los datos!):",
        min_value=1, max_value=20, value=st.session_state.rna_lab_config['epochs'], step=1,
        help="Cada época es una ronda completa de aprendizaje. Más épocas, más aprendizaje (pero cuidado con el 'sobreajuste')."
    )
    st.session_state.rna_lab_config['epochs'] = num_epochs

    if st.button("¡Entrenar mi CNN!", key="train_cnn_button"):
        st.info("Simulando el entrenamiento de tu CNN en el dataset MNIST...")
        
        # --- Lógica de simulación de entrenamiento para MNIST ---
        # Simulamos cómo cambiarían las métricas para una CNN real en MNIST
        # Una red más profunda, más ancha, con más filtros y más épocas, tiende a mejorar la precisión
        # y reducir la pérdida, hasta un punto.

        # Parámetros que influyen positivamente
        impact_conv = num_conv_layers * 0.03
        impact_filters = filters_per_conv_layer * 0.005
        impact_dense = num_dense_layers * 0.02
        impact_neurons = neurons_per_dense_layer * 0.0005
        
        # Simulación de la curva de aprendizaje de las épocas (logarítmico para simular convergencia)
        accuracy_gain_from_epochs = 1 - np.exp(-num_epochs / 8.0) # Crece y se estabiliza
        loss_reduction_from_epochs = np.exp(-num_epochs / 8.0) # Decrece y se estabiliza

        # Base para la precisión y pérdida (valores típicos para MNIST)
        max_possible_accuracy = 0.98 # Una CNN muy buena puede llegar a esto
        min_possible_loss = 0.05

        # Calcular una precisión y pérdida "objetivo" ideal para la arquitectura dada
        # Estos valores se obtienen empíricamente de cómo se comportan las CNNs en MNIST
        target_accuracy = 0.5 + (impact_conv + impact_filters + impact_dense + impact_neurons) * 2 # Ajustar factor
        target_accuracy = min(max_possible_accuracy, target_accuracy)

        target_loss = 1.5 - (impact_conv + impact_filters + impact_dense + impact_neurons) * 2 # Ajustar factor
        target_loss = max(min_possible_loss, target_loss)
        
        # Aplicar el efecto de las épocas a los valores objetivo
        final_accuracy = st.session_state.rna_lab_config['current_accuracy'] + (target_accuracy - st.session_state.rna_lab_config['current_accuracy']) * accuracy_gain_from_epochs
        final_loss = st.session_state.rna_lab_config['current_loss'] + (target_loss - st.session_state.rna_lab_config['current_loss']) * loss_reduction_from_epochs
        
        # Añadir algo de ruido para que no sea predecible al 100%
        final_accuracy += random.uniform(-0.01, 0.01)
        final_loss += random.uniform(-0.02, 0.02)

        # Simular sobreajuste (overfitting) si la red es demasiado grande o entrenada por muchas épocas
        # para un dataset "simple" como MNIST (aunque MNIST es grande, simulamos el efecto)
        if num_conv_layers >= 3 and filters_per_conv_layer >= 24 and num_epochs >= 18:
            st.warning("¡Cuidado! Tu red es muy compleja y entrenada por muchas épocas. Podría estar 'memorizando' demasiado y no 'aprendiendo' bien (sobreajuste). La precisión en datos nuevos podría bajar.")
            final_accuracy -= random.uniform(0.02, 0.05) # Reducir precisión por sobreajuste
            final_loss += random.uniform(0.02, 0.05) # Aumentar pérdida
            
        final_accuracy = max(0.1, min(0.995, final_accuracy)) # Limitar rango
        final_loss = max(0.005, final_loss) # Limitar rango

        st.session_state.rna_lab_config['current_accuracy'] = round(final_accuracy, 3)
        st.session_state.rna_lab_config['current_loss'] = round(final_loss, 3)
        
        st.success("¡Entrenamiento simulado completado!")
        st.rerun()

with col_cnn_viz:
    st.markdown("### ¡Visualización de tu CNN y Métricas!")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Capa de Entrada (imagen MNIST 28x28)
    ax.add_patch(plt.Rectangle((-0.1, 0.4), 0.2, 0.2, color='blue', ec='black', alpha=0.7))
    ax.text(-0.2, 0.5, 'Entrada\n(Imagen 28x28)', ha='right', va='center')
    
    current_x = 0.2
    
    # Capas Convolucionales
    for i in range(st.session_state.rna_lab_config['conv_layers']):
        # Dibujar una caja para representar el volumen de la capa convolucional
        # El número de filtros se representa con la "profundidad" de la caja
        # Simplificación: el tamaño espacial se reduce en cada capa Conv/Pooling, pero visualmente lo mantenemos simple.
        rect_width = 0.2
        rect_height = 0.4 # Altura que simula el tamaño espacial
        rect_depth = st.session_state.rna_lab_config['filters_per_conv_layer'] / 64 * 0.15 + 0.05 # Simula profundidad (filtros)

        ax.add_patch(plt.Rectangle((current_x, 0.5 - rect_height/2), rect_width, rect_height,
                                   color='orange', ec='black', alpha=0.6))
        # Para simular la "profundidad" de los filtros
        ax.plot([current_x, current_x + rect_depth], [0.5 - rect_height/2, 0.5 - rect_height/2], 'k-', alpha=0.5)
        ax.plot([current_x + rect_width, current_x + rect_width + rect_depth], [0.5 - rect_height/2, 0.5 - rect_height/2], 'k-', alpha=0.5)
        ax.plot([current_x, current_x + rect_depth], [0.5 + rect_height/2, 0.5 + rect_height/2], 'k-', alpha=0.5)
        ax.plot([current_x + rect_width, current_x + rect_width + rect_depth], [0.5 + rect_height/2, 0.5 + rect_height/2], 'k-', alpha=0.5)
        ax.plot([current_x + rect_depth, current_x + rect_width + rect_depth], [0.5 - rect_height/2, 0.5 - rect_height/2], 'k-', alpha=0.5)
        ax.plot([current_x + rect_depth, current_x + rect_width + rect_depth], [0.5 + rect_height/2, 0.5 + rect_height/2], 'k-', alpha=0.5)
        ax.plot([current_x + rect_width + rect_depth, current_x + rect_width + rect_depth], [0.5 - rect_height/2, 0.5 + rect_height/2], 'k-', alpha=0.5)

        ax.text(current_x + rect_width/2, 0.95, f'Capa Conv {i+1}\n({st.session_state.rna_lab_config["filters_per_conv_layer"]} filtros)', ha='center', va='bottom', fontsize=8)
        
        # Conexiones conceptuales
        if i == 0:
            ax.plot([0.1, current_x], [0.5, 0.5], 'k-', alpha=0.2)
        else:
            ax.plot([current_x - 0.2, current_x], [0.5, 0.5], 'k-', alpha=0.2)
            
        current_x += rect_width + 0.1 # Espacio entre capas

    # Capas Densas (Fully Connected)
    for i in range(st.session_state.rna_lab_config['dense_layers']):
        neurons_count = st.session_state.rna_lab_config['neurons_per_dense_layer'] if i == 0 else max(32, st.session_state.rna_lab_config['neurons_per_dense_layer'] // 2) # Reducir neuronas en capas subsiguientes
        
        neuron_size = 0.08
        if neurons_count > 100: neuron_size = 0.05
        
        y_positions = np.linspace(0.1, 0.9, min(neurons_count, 10)) # Muestra un máximo de 10 neuronas para visualización
        
        for j, y_pos in enumerate(y_positions):
            ax.add_patch(plt.Circle((current_x, y_pos), neuron_size, color='green', ec='black', alpha=0.7))
        if neurons_count > 10:
            ax.text(current_x, y_positions[0] - 0.1, f'{neurons_count} Neuronas', ha='center', va='top', fontsize=8)

        ax.text(current_x, 0.95, f'Capa Densa {i+1}', ha='center', va='bottom', fontsize=8)
        
        # Conexiones
        if i == 0: # De última capa Conv a primera densa
            ax.plot([current_x - 0.1, current_x - neuron_size], [0.5, y_positions[0]], 'k-', alpha=0.1)
            ax.plot([current_x - 0.1, current_x - neuron_size], [0.5, y_positions[-1]], 'k-', alpha=0.1)
        else: # Entre capas densas
            ax.plot([current_x - 0.1 - (0.2 if st.session_state.rna_lab_config['dense_layers'] > 1 else 0), current_x - neuron_size], [0.5, y_positions[0]], 'k-', alpha=0.05)
            ax.plot([current_x - 0.1 - (0.2 if st.session_state.rna_lab_config['dense_layers'] > 1 else 0), current_x - neuron_size], [0.5, y_positions[-1]], 'k-', alpha=0.05)
            
        current_x += 0.3 # Espacio entre capas densas

    # Capa de Salida (10 neuronas para dígitos 0-9)
    output_x_pos = current_x + 0.1
    output_y_positions = np.linspace(0.1, 0.9, 10) # 10 neuronas para 0-9
    for j, y_pos in enumerate(output_y_positions):
        ax.add_patch(plt.Circle((output_x_pos, y_pos), 0.06, color='red', ec='black', alpha=0.8))
        ax.text(output_x_pos + 0.08, y_pos, str(j), ha='left', va='center', fontsize=8) # Etiquetar con el dígito

    ax.text(output_x_pos, 0.95, 'Salida\n(Dígito 0-9)', ha='center', va='bottom', fontsize=8)

    # Conexiones de la última capa densa a la salida
    last_dense_y_positions = np.linspace(0.1, 0.9, min(st.session_state.rna_lab_config['neurons_per_dense_layer'] if st.session_state.rna_lab_config['dense_layers'] == 1 else max(32, st.session_state.rna_lab_config['neurons_per_dense_layer'] // 2), 10))
    for prev_y in last_dense_y_positions:
        for out_y in output_y_positions:
            ax.plot([current_x - 0.06, output_x_pos - 0.06], [prev_y, out_y], 'k-', alpha=0.02)


    ax.set_xlim(-0.3, output_x_pos + 0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    ax.set_title(f"Estructura de la Red Neuronal Convolucional (CNN)\n"
                 f"Capas Conv: {st.session_state.rna_lab_config['conv_layers']}, Filtros: {st.session_state.rna_lab_config['filters_per_conv_layer']}\n"
                 f"Capas Densas: {st.session_state.rna_lab_config['dense_layers']}, Neuronas: {st.session_state.rna_lab_config['neurons_per_dense_layer']}")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("### Rendimiento de la Red (Después de entrenar):")
    st.metric(label="Precisión (Accuracy)", value=f"{st.session_state.rna_lab_config['current_accuracy']*100:.1f}%",
              help="¿Qué tan bien predice la red los dígitos? ¡Más alto es mejor!")
    st.metric(label="Pérdida (Loss)", value=f"{st.session_state.rna_lab_config['current_loss']:.3f}",
              help="¿Qué tan equivocada está la red al predecir los dígitos? ¡Más bajo es mejor!")
    
    st.markdown("""
    **Observa:**
    * **Más épocas:** Normalmente mejoran la precisión y reducen la pérdida.
    * **Más capas convolucionales o filtros:** Ayudan a la red a detectar patrones más complejos en las imágenes.
    * **Más capas densas o neuronas:** Permiten a la red aprender relaciones más sofisticadas entre las características detectadas.
    * **¡Cuidado con el 'sobreajuste'!** Si la red es demasiado grande o entrena por muchas épocas para la cantidad de datos, podría memorizar los ejemplos en lugar de aprender a generalizar.
    """)

st.write("---")

st.subheader("¡Neuronas que se Encienden! ¿Cómo 've' tu CNN un Dígito?")
st.markdown("""
Cuando tu CNN 've' una imagen (como un dígito dibujado por ti), diferentes
neuronas en sus capas se "encienden" o se activan. Las neuronas en las
primeras capas convolucionales se especializan en detectar cosas simples
como bordes, líneas o curvas. Las neuronas en capas más profundas detectan
patrones más complejos, hasta que las últimas neuronas deciden qué dígito es.

**¡Dibuja un dígito y veamos qué 'piensa' tu CNN!**
""")

col_draw_input, col_simulated_activations = st.columns([1, 1])

with col_draw_input:
    st.markdown("#### ✍️ Dibuja un dígito (0-9) aquí:")
    
    # Crear un lienzo para dibujar
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # Color de fondo blanco
        stroke_width=15, # Ancho del lápiz
        stroke_color="#000000", # Color del lápiz (negro)
        background_color="#FFFFFF", # Fondo blanco para la visualización final
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas_mnist",
    )

    if st.button("¡Mi CNN lo clasifica!", key="classify_drawn_digit_button"):
        if canvas_result.image_data is not None:
            if model_mnist_cnn is None:
                st.error("¡El modelo CNN no está cargado! No puedo clasificar el dígito. Asegúrate de que el archivo del modelo exista y sea válido.")
                st.session_state.rna_lab_config['predicted_digit'] = None
                st.session_state.rna_lab_config['prediction_probabilities'] = None
                st.session_state.rna_lab_config['drawn_digit_input'] = None
                st.session_state.rna_lab_config['simulated_activations'] = {} # Limpiar activaciones
                st.rerun() # Volver a cargar para mostrar el error
            
            # Convertir el resultado del canvas a una imagen de PIL
            drawn_image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
            
            # Convertir a escala de grises y redimensionar a 28x28 (como MNIST)
            drawn_image_L = drawn_image.convert("L") # L for Luminance (grayscale)
            drawn_image_resized = drawn_image_L.resize((28, 28), Image.LANCZOS)
            
            # Invertir colores: negro -> blanco, blanco -> negro
            # Esto convierte el trazo negro a blanco para que se parezca a los datos MNIST
            img_array = np.array(drawn_image_resized)
            img_array = 255 - img_array # Invertir: 0 (negro) se vuelve 255 (blanco), 255 (blanco) se vuelve 0 (negro)
            
            # Normalizar a 0-1 y añadir una dimensión para el batch y el canal (1, 28, 28, 1)
            img_for_prediction = img_array.astype('float32') / 255.0
            img_for_prediction = np.expand_dims(img_for_prediction, axis=0) # Añadir dimensión de batch
            img_for_prediction = np.expand_dims(img_for_prediction, axis=-1) # Añadir dimensión de canal si es necesario (el modelo Keras lo espera)

            st.session_state.rna_lab_config['drawn_digit_input'] = img_array # Guardar para visualización
            
            st.info("La CNN está clasificando tu dígito...")
            
            # --- REALIZAR LA PREDICCIÓN CON EL MODELO CARGADO ---
            predictions = model_mnist_cnn.predict(img_for_prediction)
            predicted_digit = np.argmax(predictions)
            prediction_probabilities = predictions[0] # Obtener las probabilidades de la primera (y única) imagen
            
            st.session_state.rna_lab_config['predicted_digit'] = predicted_digit
            st.session_state.rna_lab_config['prediction_probabilities'] = prediction_probabilities
            
            # --- Generar activaciones simuladas (mantenemos la simulación para la visualización interna) ---
            simulated_activations = {}
            
            # Capas Convolucionales: Mostrar pseudo-filtros/características
            # Para una visualización más precisa de activaciones reales, se requeriría un modelo de Keras que soporte
            # la extracción de capas intermedias y visualización más compleja (ej. Keras-vis).
            # Aquí, simplemente simulamos cómo se verían las "activaciones" basado en el dibujo.
            
            # Simulación de activación de la primera capa convolucional
            # Crear una imagen simulada de las "características" detectadas
            # Más oscuro donde hay "detección" (cercano al trazo)
            sim_conv_output = np.zeros((28, 28, st.session_state.rna_lab_config['filters_per_conv_layer']))
            for f_idx in range(st.session_state.rna_lab_config['filters_per_conv_layer']):
                # Simular que los filtros detectan el dígito (activación alta donde hay píxeles del dígito)
                sim_conv_output[:, :, f_idx] = img_array / 255.0 + random.uniform(-0.1, 0.1)
                sim_conv_output[:, :, f_idx] = np.clip(sim_conv_output[:, :, f_idx], 0, 1)
            st.session_state.rna_lab_config['simulated_activations']['conv1'] = sim_conv_output
            
            # Simulación de activación de la primera capa densa
            # Las neuronas se "encienden" más si la predicción es del dígito correspondiente
            sim_dense_output = np.random.rand(st.session_state.rna_lab_config['neurons_per_dense_layer']) * 0.2
            # Dar un "empujón" a las neuronas que llevarían a la clase predicha
            if predicted_digit is not None:
                # Simular que las neuronas relevantes para el dígito predicho tienen mayor activación
                # Esto es una simplificación, en realidad son muchas neuronas las que contribuyen
                num_boosted_neurons = min(10, st.session_state.rna_lab_config['neurons_per_dense_layer'])
                boost_indices = random.sample(range(st.session_state.rna_lab_config['neurons_per_dense_layer']), num_boosted_neurons)
                for idx in boost_indices:
                     sim_dense_output[idx] = min(1.0, sim_dense_output[idx] + random.uniform(0.5, 0.9)) # Aumentar activación
            
            st.session_state.rna_lab_config['simulated_activations']['dense1'] = sim_dense_output

            # Simulación de activación de la capa de salida (las probabilidades reales)
            st.session_state.rna_lab_config['simulated_activations']['output'] = prediction_probabilities
            
            st.success("¡Clasificación completada!")
            st.rerun() # Recargar para mostrar los resultados

        else:
            st.warning("Por favor, dibuja algo en el lienzo antes de clasificar.")

with col_simulated_activations:
    st.markdown("#### ¿Qué 'vio' tu CNN?")

    if st.session_state.rna_lab_config['drawn_digit_input'] is not None:
        st.markdown("##### Tu dígito dibujado (redimensionado a 28x28 para la CNN):")
        # Mostrar el dígito preprocesado (invertido y 28x28)
        fig_drawn_preview, ax_drawn_preview = plt.subplots(figsize=(2, 2))
        ax_drawn_preview.imshow(st.session_state.rna_lab_config['drawn_digit_input'], cmap='gray_r') # gray_r para blanco en negro
        ax_drawn_preview.axis('off')
        st.pyplot(fig_drawn_preview)
        plt.close(fig_drawn_preview)

        if st.session_state.rna_lab_config['predicted_digit'] is not None:
            st.markdown(f"**¡Tu CNN predice que es el dígito: {st.session_state.rna_lab_config['predicted_digit']}!**")
            
            st.markdown("---")
            st.markdown("##### Probabilidades de la predicción (Capa de Salida):")
            # Mostrar barras de probabilidad de cada dígito
            probabilities = st.session_state.rna_lab_config['prediction_probabilities']
            if probabilities is not None:
                fig_probs, ax_probs = plt.subplots(figsize=(6, 3))
                digits = np.arange(10)
                bars = ax_probs.bar(digits, probabilities, color='skyblue')
                ax_probs.set_xticks(digits)
                ax_probs.set_xlabel("Dígito")
                ax_probs.set_ylabel("Probabilidad")
                ax_probs.set_title("Probabilidad de cada dígito")
                ax_probs.set_ylim(0, 1)
                
                # Resaltar el dígito predicho
                bars[st.session_state.rna_lab_config['predicted_digit']].set_color('salmon')
                
                # Añadir etiquetas de porcentaje
                for bar in bars:
                    yval = bar.get_height()
                    ax_probs.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval*100:.1f}%', ha='center', va='bottom', fontsize=8)
                
                st.pyplot(fig_probs)
                plt.close(fig_probs)

            st.markdown("---")
            st.markdown("##### Activaciones Internas (Simuladas):")
            st.info("Estas visualizaciones son una simulación conceptual de cómo las neuronas se 'activan' internamente, no una representación exacta de las activaciones reales del modelo TensorFlow para simplicidad educativa.")

            # Visualización de la primera capa convolucional simulada
            if 'conv1' in st.session_state.rna_lab_config['simulated_activations']:
                st.markdown("###### Capa Convolucional (Filtros buscando características):")
                num_filters_to_show = min(8, st.session_state.rna_lab_config['filters_per_conv_layer']) # Mostrar hasta 8 filtros
                cols = st.columns(num_filters_to_show)
                for i in range(num_filters_to_show):
                    with cols[i]:
                        fig_conv, ax_conv = plt.subplots(figsize=(1.5, 1.5))
                        # Usamos 'gray_r' para que las activaciones más altas (cerca de 1) se vean más blancas
                        ax_conv.imshow(st.session_state.rna_lab_config['simulated_activations']['conv1'][:, :, i], cmap='gray_r', vmin=0, vmax=1)
                        ax_conv.axis('off')
                        ax_conv.set_title(f"Filtro {i+1}", fontsize=8)
                        st.pyplot(fig_conv)
                        plt.close(fig_conv)
                st.markdown(f"*(Mostrando {num_filters_to_show} de {st.session_state.rna_lab_config['filters_per_conv_layer']} filtros simulados)*")

            # Visualización de la primera capa densa simulada
            if 'dense1' in st.session_state.rna_lab_config['simulated_activations']:
                st.markdown("###### Capa Densa (Neuronas combinando características):")
                dense_activations = st.session_state.rna_lab_config['simulated_activations']['dense1']
                
                # Visualización como un heatmap de activaciones neuronales
                fig_dense, ax_dense = plt.subplots(figsize=(8, 1)) # Una fila, muchas columnas
                
                # Redimensionar a una forma más visual (ej. 1xN o 2x(N/2))
                display_neurons = min(50, len(dense_activations)) # Mostrar hasta 50 neuronas para no saturar
                
                # Crear un array 2D para el heatmap
                if display_neurons > 0:
                    rows = 1 # Puedes ajustar esto para más filas si es necesario
                    cols = int(np.ceil(display_neurons / rows))
                    
                    # Rellenar con las activaciones, y padding si no es un cuadrado perfecto
                    reshaped_activations = np.zeros((rows, cols))
                    reshaped_activations.flat[:display_neurons] = dense_activations[:display_neurons]
                    
                    sns.heatmap(reshaped_activations, cmap='viridis', cbar=False, ax=ax_dense, vmin=0, vmax=1, xticklabels=False, yticklabels=False)
                    ax_dense.set_title(f"Activación de Neuronas (Simulada) en la primera Capa Densa (top {display_neurons})", fontsize=10)
                    st.pyplot(fig_dense)
                    plt.close(fig_dense)
                
                st.markdown(f"*(Mostrando activación simulada para las primeras {display_neurons} neuronas de {st.session_state.rna_lab_config['neurons_per_dense_layer']})*")


    else:
        st.info("Dibuja un dígito y haz clic en '¡Mi CNN lo clasifica!' para ver las activaciones.")

st.write("---")

# --- Sección de Chatbot de Juego con Neo el Neurona ---
st.header("¡Juega y Aprende con Neo el Neurona sobre Redes Neuronales!")
st.markdown("¡Hola! Soy Neo, tu neurona amiga. ¿Listo para conectar tus conocimientos y aprender sobre Redes Neuronales?")

if client:
    # Inicializa el estado del juego y los mensajes del chat
    if "rna_game_active" not in st.session_state:
        st.session_state.rna_game_active = False
    if "rna_game_messages" not in st.session_state:
        st.session_state.rna_game_messages = []
    if "rna_current_question" not in st.session_state:
        st.session_state.rna_current_question = None
    if "rna_current_options" not in st.session_state:
        st.session_state.rna_current_options = {}
    if "rna_correct_answer" not in st.session_state:
        st.session_state.rna_correct_answer = None
    if "rna_awaiting_next_game_decision" not in st.session_state:
        st.session_state.rna_awaiting_next_game_decision = False
    if "rna_game_needs_new_question" not in st.session_state:
        st.session_state.rna_game_needs_new_question = False
    if "rna_correct_streak" not in st.session_state: # <-- This is crucial
        st.session_state.rna_correct_streak = 0

    # System prompt para el juego de preguntas de Neo el Neurona
    # MUEVE ESTA DEFINICIÓN AQUÍ, DESPUÉS DE QUE TODO EL SESSION_STATE ESTÉ INICIALIZADO
    rna_game_system_prompt = f"""
    Eres un **experto consumado en Inteligencia Artificial y Machine Learning**, con una especialización profunda en el diseño, entrenamiento y comprensión de las **Redes Neuronales Artificiales (RNA)**. Comprendes a fondo sus fundamentos biológicos inspiradores, su arquitectura matemática, los algoritmos de aprendizaje clave (como el backpropagation), las funciones de activación, los optimizadores y sus diversas aplicaciones prácticas en problemas de clasificación, regresión y más. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de las RNA mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Activación neuronal exitosa! Tu red de conocimiento es robusta." o "Esa conexión sináptica necesita reforzarse. Revisemos los pesos."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "Las Redes Neuronales Artificiales son modelos computacionales inspirados en el cerebro humano, diseñados para reconocer patrones y aprender de los datos..."]
    [Pregunta para continuar, ej: "¿Listo para optimizar tus arquitecturas neuronales?" o "¿Quieres profundizar en cómo el backpropagation ajusta los pesos?"]

    **Reglas adicionales para el Experto en Redes Neuronales Artificiales:**
    * **Enfoque Riguroso en RNA:** Todas tus preguntas y explicaciones deben girar en torno a las Redes Neuronales Artificiales (incluyendo Perceptrón, Multi-Layer Perceptron/DNNs). Cubre sus fundamentos (inspiración biológica, neurona artificial/perceptrón), componentes (entradas, pesos, sesgos, sumador, función de activación), arquitecturas (capas de entrada, ocultas, salida), el **proceso de forward propagation**, el **cálculo de la función de pérdida**, el algoritmo de **backpropagation** para el ajuste de pesos, los **optimizadores** (Descenso de Gradiente, SGD, Adam), el **sobreajuste** y técnicas de **regularización** (Dropout, Batch Normalization), y la importancia del **preprocesamiento de datos** para las RNA.
    * **¡VARIEDAD, VARIADAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de RNA que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General e Inspiración:** ¿Qué es una RNA? Inspiración biológica (neuronas, sinapsis). Diferencias con el cerebro biológico.
        * **La Neurona Artificial (Perceptrón):** Entradas, pesos, sesgos, suma ponderada, función de activación (básica: escalón, sigmoide).
        * **Arquitectura de RNA:**
            * **Capas:** Entrada, Oculta (número de capas, neuronas por capa), Salida.
            * **Conexiones:** Adelante (feedforward).
            * Redes Densamente Conectadas (Fully Connected / Perceptrón Multicapa - MLP).
        * **Funciones de Activación:** Propósito (introducir no linealidad), tipos (Sigmoide, ReLU, Tanh, Softmax), cuándo usar cada una.
        * **Forward Propagation:** Cómo se calcula la salida de la red para una entrada dada.
        * **Entrenamiento de una RNA:**
            * **Función de Pérdida (Loss Function):** MSE (regresión), Cross-Entropy (clasificación).
            * **Descenso de Gradiente:** Concepto de minimizar la pérdida.
            * **Backpropagation:** Cómo se calculan los gradientes y se actualizan los pesos de forma eficiente (cadena de reglas).
            * **Optimizadores:** SGD (Stochastic Gradient Descent), Adam, RMSProp.
        * **Sobreajuste y Regularización:**
            * **Dropout:** Concepto y propósito.
            * **Batch Normalization:** Concepto y propósito.
            * Regularización L1/L2.
        * **Preprocesamiento de Datos:** Escalado (normalización/estandarización) para RNA, codificación de variables categóricas.
        * **Ventajas y Desventajas:** Capacidad de aprender patrones complejos vs. interpretabilidad (caja negra), necesidad de muchos datos, costo computacional.
        * **Tipos de Problemas:** Clasificación, Regresión (cómo se adaptan las RNA).

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.rna_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Explorador Neuronal – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de cómo una "neurona" artificial toma una decisión simple y ejemplos generales de lo que puede hacer una RNA.
            * *Tono:* "Estás activando tus primeras conexiones para entender cómo aprenden las máquinas."
        * **Nivel 2 (Constructor de Perceptrones – 3-5 respuestas correctas):** Tono más técnico. Introduce los componentes de una neurona (pesos, sesgos, activación) y el concepto de capas. Preguntas sobre el funcionamiento básico de un perceptrón o una red sencilla.
            * *Tono:* "Tu comprensión de los bloques fundamentales de las Redes Neuronales está en pleno desarrollo."
        * **Nivel 3 (Arquitecto de RNA – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en el proceso de **forward propagation**, la necesidad de funciones de activación no lineales, la intuición del **backpropagation** (sin entrar en los cálculos detallados de gradientes), y la importancia del escalado de datos.
            * *Tono:* "Tu habilidad para diseñar y comprender la mecánica interna de las Redes Neuronales es crucial para construir modelos complejos."
        * **Nivel Maestro (Científico de Redes Profundas – 9+ respuestas correctas):** Tono de **especialista en Machine Learning y optimización de RNA**. Preguntas sobre la elección de funciones de activación para diferentes capas/problemas, el impacto de diferentes optimizadores en la convergencia, la aplicación de técnicas de regularización avanzadas, o el diagnóstico y solución de problemas como el vanishing/exploding gradient (conceptual). Se esperan respuestas que demuestren una comprensión teórica y práctica robusta, incluyendo sus limitaciones y cómo diseñar redes eficientes.
            * *Tono:* "Tu maestría en Redes Neuronales Artificiales te posiciona a la vanguardia del desarrollo de sistemas inteligentes, transformando datos en decisiones."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Una red que aprende a clasificar imágenes simples como "perro" o "gato".
        * **Nivel 2:** Una red neuronal que predice el precio de una vivienda basándose en múltiples características, o que clasifica comentarios como positivos o negativos.
        * **Nivel 3:** Diseñar una red neuronal densamente conectada para reconocer dígitos escritos a mano (MNIST), explicando la función de activación de la capa de salida y la función de pérdida.
        * **Nivel Maestro:** Optimizar una RNA para una tarea de detección de anomalías en datos de sensores con ruido, eligiendo el optimizador adecuado, implementando Batch Normalization y explicando cómo mitigar el sobreajuste en un contexto de datos limitados.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre REDES NEURONALES ARTIFICIALES, y asegúrate de que no se parezca a las anteriores.
    """

    # ... el resto de tu código

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_rna_question_response(raw_text):
        question = ""
        options = {}
        correct_answer_key = ""
        lines = raw_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("pregunta:"):
                question = line[len("pregunta:"):].strip()
            elif line.lower().startswith("a)"):
                options['A'] = line[len("a):"):].strip()
            elif line.lower().startswith("b)"):
                options['B'] = line[len("b):"):].strip()
            elif line.lower().startswith("c)"):
                options['C'] = line[len("c):"):].strip()
            elif line.lower().startswith("respuestacorrecta:"):
                correct_answer_key = line[len("respuestacorrecta:"):].strip().upper()
        if not (question and len(options) == 3 and correct_answer_key in options):
            st.warning(f"DEBUG: Formato de pregunta inesperado de la API. Texto recibido:\n{raw_text}")
            return None, {}, ""
        return question, options, correct_answer_key

    # Función para parsear la respuesta de feedback de la IA
    def parse_rna_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"
    
    # --- Funciones para subir de nivel directamente ---
    def set_rna_level(target_streak, level_name):
        st.session_state.rna_correct_streak = target_streak
        st.session_state.rna_game_active = True
        st.session_state.rna_game_messages = []
        st.session_state.rna_current_question = None
        st.session_state.rna_current_options = {}
        st.session_state.rna_correct_answer = None
        st.session_state.rna_game_needs_new_question = True
        st.session_state.rna_awaiting_next_game_decision = False
        st.session_state.rna_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}**! Prepárate para preguntas más desafiantes. ¡Aquí va tu primera!"})
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons_rna, col_level_up_buttons_rna = st.columns([1, 2])

    with col_game_buttons_rna:
        if st.button("¡Vamos a jugar con Neo el Neurona!", key="start_neo_game_button"):
            st.session_state.rna_game_active = True
            st.session_state.rna_game_messages = []
            st.session_state.rna_current_question = None
            st.session_state.rna_current_options = {}
            st.session_state.rna_correct_answer = None
            st.session_state.rna_game_needs_new_question = True
            st.session_state.rna_awaiting_next_game_decision = False
            st.session_state.rna_correct_streak = 0
            st.rerun()
    
    with col_level_up_buttons_rna:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un experto neuronal? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_rna, col_lvl2_rna, col_lvl3_rna = st.columns(3)
        with col_lvl1_rna:
            if st.button("Subir a Nivel Medio (RNA)", key="level_up_medium_rna"):
                set_rna_level(3, "Medio")
        with col_lvl2_rna:
            if st.button("Subir a Nivel Avanzado (RNA)", key="level_up_advanced_rna"):
                set_rna_level(6, "Avanzado")
        with col_lvl3_rna:
            if st.button("¡Maestro Neuronal! (RNA)", key="level_up_champion_rna"):
                set_rna_level(9, "Campeón")


    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.rna_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Lógica del juego del chatbot si está activo
    if st.session_state.rna_game_active:
        if st.session_state.rna_current_question is None and st.session_state.rna_game_needs_new_question and not st.session_state.rna_awaiting_next_game_decision:
            with st.spinner("Neo está preparando una pregunta..."):
                try:
                    rna_game_messages_for_api = [{"role": "system", "content": rna_game_system_prompt}]
                    # Añadir mensajes anteriores al historial del chat para la API (útil para el seguimiento de dificultad)
                    for msg in st.session_state.rna_game_messages[-6:]: # Limitar historial para no exceder tokens
                        if msg["role"] == "assistant" and msg["content"].startswith("**"):
                            rna_game_messages_for_api.append({"role": "assistant", "content": f"PREGUNTA ANTERIOR: {msg['content'].splitlines()[0]}"})
                        elif msg["role"] == "user" and "MI RESPUESTA:" not in msg["content"]:
                            rna_game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {msg['content']}"})

                    rna_game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre Redes Neuronales Artificiales o Redes Convolucionales (CNNs). Recuerda el nivel de dificultad actual basado en la racha de preguntas correctas."})

                    rna_response = client.chat.completions.create(
                        model="gpt-4o-mini", # Puedes usar "gpt-3.5-turbo" si prefieres
                        messages=rna_game_messages_for_api,
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_rna_question_text = rna_response.choices[0].message.content
                    
                    question, options, correct_answer_key = parse_rna_question_response(raw_rna_question_text)

                    if question and options and correct_answer_key:
                        st.session_state.rna_current_question = question
                        st.session_state.rna_current_options = options
                        st.session_state.rna_correct_answer = correct_answer_key
                        st.session_state.rna_game_needs_new_question = False
                        st.session_state.rna_awaiting_next_game_decision = False # Reset this if a new question is generated
                        st.rerun() # Refresh to display the new question
                    else:
                        st.error("Neo no pudo generar una pregunta. Inténtalo de nuevo.")
                        st.session_state.rna_game_messages.append({"role": "assistant", "content": "¡Uy! Parece que hubo un problema y no pude generar la pregunta. ¿Puedes intentarlo de nuevo por favor?"})
                        st.session_state.rna_game_needs_new_question = True # Allow retrying
                        st.session_state.rna_awaiting_next_game_decision = False
                        time.sleep(1)
                        st.rerun()

                except Exception as e:
                    st.error(f"¡Oops! Neo tuvo un problema al generar la pregunta. Error: {e}")
                    st.info("Asegúrate de que tu clave de API de OpenAI es válida y tienes conexión a internet.")
                    st.session_state.rna_game_active = False # Desactivar juego si hay un error grave
                    st.session_state.rna_game_messages.append({"role": "assistant", "content": "¡Lo siento! No puedo conectar mis circuitos neuronales ahora mismo. Por favor, revisa mi conexión o inténtalo más tarde."})
                    time.sleep(1)
                    st.rerun()

        # Display current question if available
        if st.session_state.rna_current_question:
            st.markdown(f"**Nivel de Dificultad Actual:** {st.session_state.rna_correct_streak + 1} (Racha: {st.session_state.rna_correct_streak} correctas)")
            with st.chat_message("assistant"):
                st.markdown(f"**{st.session_state.rna_current_question}**")
                # Add TTS button
                try:
                    tts = gTTS(text=st.session_state.rna_current_question, lang='es', slow=False)
                    audio_bytes = io.BytesIO()
                    tts.write_to_fp(audio_bytes)
                    st.audio(audio_bytes, format='audio/mp3', start_time=0)
                except Exception as e:
                    st.warning(f"No se pudo generar el audio para la pregunta: {e}")

            # Collect user's answer
            user_answer = st.radio(
                "Elige tu respuesta:",
                options=[f"{key}) {value}" for key, value in st.session_state.rna_current_options.items()],
                key="rna_question_radio_options"
            )

            if st.button("Enviar Respuesta", key="submit_rna_answer"):
                chosen_key = user_answer[0] # Get 'A', 'B', or 'C'
                is_correct = (chosen_key == st.session_state.rna_correct_answer)
                
                # Add user's answer to chat history
                st.session_state.rna_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {chosen_key}) {st.session_state.rna_current_options[chosen_key]}"})
                
                feedback_prompt = f"El usuario respondió {chosen_key}. La respuesta correcta era {st.session_state.rna_correct_answer}. ¿Fue correcto: {is_correct}? Dame feedback."
                
                with st.spinner("Neo está pensando en tu respuesta..."):
                    try:
                        rna_game_messages_for_api = [{"role": "system", "content": rna_game_system_prompt}]
                        # Incluir la pregunta original y la respuesta del usuario para el feedback
                        rna_game_messages_for_api.append({"role": "assistant", "content": f"PREGUNTA: {st.session_state.rna_current_question}"})
                        rna_game_messages_for_api.append({"role": "user", "content": f"RESPUESTA DEL USUARIO: {chosen_key}"})
                        rna_game_messages_for_api.append({"role": "user", "content": feedback_prompt})

                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=rna_game_messages_for_api,
                            temperature=0.7,
                            max_tokens=200
                        )
                        raw_rna_feedback_text = feedback_response.choices[0].message.content
                        feedback_msg, explanation_msg, continue_q_msg = parse_rna_feedback_response(raw_rna_feedback_text)
                        
                        st.session_state.rna_game_messages.append({"role": "assistant", "content": feedback_msg})
                        st.session_state.rna_game_messages.append({"role": "assistant", "content": explanation_msg})
                        st.session_state.rna_game_messages.append({"role": "assistant", "content": continue_q_msg})
                        
                        # Add TTS for feedback
                        try:
                            feedback_text_for_audio = f"{feedback_msg}. {explanation_msg}. {continue_q_msg}"
                            tts_feedback = gTTS(text=feedback_text_for_audio, lang='es', slow=False)
                            audio_bytes_feedback = io.BytesIO()
                            tts_feedback.write_to_fp(audio_bytes_feedback)
                            st.audio(audio_bytes_feedback, format='audio/mp3', start_time=0)
                        except Exception as e:
                            st.warning(f"No se pudo generar el audio para el feedback: {e}")

                        if is_correct:
                            st.session_state.rna_correct_streak += 1
                        else:
                            st.session_state.rna_correct_streak = 0 # Reset streak on incorrect answer
                            
                        st.session_state.rna_current_question = None # Clear question
                        st.session_state.rna_current_options = {}
                        st.session_state.rna_correct_answer = None
                        st.session_state.rna_awaiting_next_game_decision = True
                        st.rerun() # Refresh to show feedback and prompt for next question
                    except Exception as e:
                        st.error(f"¡Oops! Neo tuvo un problema al darte feedback. Error: {e}")
                        st.session_state.rna_game_messages.append({"role": "assistant", "content": "Lo siento, no pude darte feedback ahora mismo. ¿Quieres intentar otra pregunta?"})
                        st.session_state.rna_game_needs_new_question = True
                        st.session_state.rna_awaiting_next_game_decision = False
                        st.rerun()

        # Botones para continuar el juego después del feedback
        if st.session_state.rna_awaiting_next_game_decision:
            col_decision_buttons = st.columns(2)
            with col_decision_buttons[0]:
                if st.button("¡Siguiente pregunta!", key="next_rna_question"):
                    st.session_state.rna_game_needs_new_question = True
                    st.session_state.rna_awaiting_next_game_decision = False
                    st.rerun()
            with col_decision_buttons[1]:
                if st.button("Terminar el juego", key="end_rna_game"):
                    st.session_state.rna_game_active = False
                    st.session_state.rna_game_messages.append({"role": "assistant", "content": "¡Gracias por jugar con Neo el Neurona! ¡Espero que hayas aprendido mucho sobre las Redes Neuronales! ¡Hasta la próxima conexión!"})
                    st.session_state.rna_correct_streak = 0
                    st.rerun()
else:
    st.info("Para jugar con Neo el Neurona, por favor, configura tu clave de la API de OpenAI en `secrets.toml`.")

