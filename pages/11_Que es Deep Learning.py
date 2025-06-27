import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import io
import time
from gtts import gTTS

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model # Importar load_model
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
    from tensorflow.keras.utils import to_categorical
except ImportError:
    st.error("Las librerías 'tensorflow' y 'keras' no están instaladas. Por favor, instálalas usando: pip install tensorflow")
    tf = None
    keras = None
    Sequential = None
    Dense = None
    Flatten = None
    Conv2D = None
    MaxPooling2D = None
    to_categorical = None
    load_model = None

# Importar OpenAI si se usa el chatbot
try:
    from openai import OpenAI
except ImportError:
    st.error("La librería 'openai' no está instalada. Por favor, instálala usando: pip install openai")
    OpenAI = None

# --- Configuración de la página ---
st.set_page_config(
    page_title="¿Qué es el Deep Learning?",
    layout="wide"
)

# --- Ruta donde se espera encontrar el modelo guardado ---
MODEL_LOAD_PATH = os.path.join("assets", "models", "deep_learning_model.h5")

# --- Funciones auxiliares para Deep Learning ---

@st.cache_resource
def load_mnist_test_data():
    """Carga y preprocesa el dataset MNIST (solo test) para el ejemplo de Deep Learning."""
    if tf is None or keras is None:
        return None, None
    try:
        _, (x_test, y_test) = keras.datasets.mnist.load_data() # Cargar solo los datos de test directamente
        x_test = x_test.astype('float32') / 255.0
        x_test = np.expand_dims(x_test, -1)
        y_test = to_categorical(y_test, 10)
        
        return x_test, y_test
    except Exception as e:
        st.error(f"Error al cargar o preprocesar datos MNIST de prueba: {e}")
        return None, None

@st.cache_resource
def load_deep_learning_model():
    """Carga el modelo de Red Neuronal Convolucional (CNN) guardado."""
    if load_model is None:
        st.warning("`load_model` de Keras no está disponible. No se puede cargar el modelo.")
        return None
    if not os.path.exists(MODEL_LOAD_PATH):
        st.error(f"¡Error! El archivo del modelo no se encontró en: {MODEL_LOAD_PATH}. Por favor, entrena el modelo primero ejecutando 'train_model.py'.")
        return None
    try:
        model = load_model(MODEL_LOAD_PATH)
        st.success(f"Modelo de Deep Learning (CNN) cargado exitosamente desde: {MODEL_LOAD_PATH}")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo de Deep Learning desde {MODEL_LOAD_PATH}: {e}. Asegúrate de que el modelo fue guardado correctamente y que tienes las versiones compatibles de TensorFlow/Keras.")
        return None


# --- Inicialización robusta de session_state ---

if 'deep_learning_module_config' not in st.session_state:
    st.session_state.deep_learning_module_config = {
        'dl_model': None,
        'dl_data_test': None,
        'dl_labels_test': None,
        'dl_game_correct_count': 0,
        'dl_game_total_count': 0,
        'current_game_image_dl': None,
        'current_game_label_dl': None,
        'game_awaiting_guess_dl': False,
        'show_dl_explanation': False,
        'chatbot_messages': [{"role": "assistant", "content": "¡Hola! Soy el **Cerebro Digital**. ¿Listo para explorar el mundo del Deep Learning? Pregúntame lo que quieras."}]
    }

if "dl_chatbot_game_active" not in st.session_state:
    st.session_state.dl_chatbot_game_active = False
if "dl_chatbot_game_messages" not in st.session_state:
    st.session_state.dl_chatbot_game_messages = []
if "dl_chatbot_current_question" not in st.session_state:
    st.session_state.dl_chatbot_current_question = None
if "dl_chatbot_current_options" not in st.session_state:
    st.session_state.dl_chatbot_current_options = {}
if "dl_chatbot_correct_answer" not in st.session_state:
    st.session_state.dl_chatbot_correct_answer = None
if "dl_chatbot_awaiting_next_game_decision" not in st.session_state:
    st.session_state.dl_chatbot_awaiting_next_game_decision = False
if "dl_chatbot_game_needs_new_question" not in st.session_state:
    st.session_state.dl_chatbot_game_needs_new_question = False
if "dl_chatbot_correct_streak" not in st.session_state:
    st.session_state.dl_chatbot_correct_streak = 0
if "dl_chatbot_last_played_question" not in st.session_state:
    st.session_state.dl_chatbot_last_played_question = None


# Cargar datos de test y modelo de Deep Learning al inicio
if tf is not None and keras is not None:
    if st.session_state.deep_learning_module_config['dl_data_test'] is None:
        x_test_dl, y_test_dl = load_mnist_test_data()
        st.session_state.deep_learning_module_config['dl_data_test'] = x_test_dl
        st.session_state.deep_learning_module_config['dl_labels_test'] = y_test_dl

    if st.session_state.deep_learning_module_config['dl_model'] is None:
        model_temp = load_deep_learning_model()
        st.session_state.deep_learning_module_config['dl_model'] = model_temp
    else:
        pass


# --- Rutinas de parseo de la API (Mantener igual) ---
def parse_api_response(raw_text, mode="question"):
    if mode == "question":
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
            return None, {}, ""
        return question, options, correct_answer_key
    elif mode == "feedback":
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        else:
            return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"

# --- Configuración de la API de OpenAI (Mantener igual) ---
client = None
openai_api_key_value = None

if "openai_api_key" in st.secrets:
    openai_api_key_value = st.secrets['openai_api_key']
elif "OPENAI_API_KEY" in st.secrets:
    openai_api_key_value = st.secrets['OPENAI_API_KEY']

if openai_api_key_value:
    try:
        client = OpenAI(api_key=openai_api_key_value)
    except Exception as e:
        st.error(f"Error al inicializar cliente OpenAI con la clave proporcionada: {e}")
        client = None
else:
    st.warning("¡ATENCIÓN! La clave de la API de OpenAI no se ha encontrado en `secrets.toml`.")
    st.info("""
    Para usar el chatbot del Cerebro Digital o el juego de preguntas, necesitas añadir tu clave de la API de OpenAI a tu archivo `secrets.toml`.
    """)
    OpenAI = None

# --- Título y Explicación del Módulo ---
st.title("Laboratorio Interactivo: ¿Qué es el Deep Learning?")

st.markdown("""
¡Bienvenido al laboratorio donde desentrañaremos los misterios del **Deep Learning**!

---

### ¿Qué es el Deep Learning? ¡Es como enseñar a un robot a pensar como nosotros!

Imagina que quieres enseñarle a un robot a reconocer animales en fotos. En lugar de darle una lista de características (como "tiene orejas puntiagudas" o "tiene cola larga"), lo que haces es mostrarle **millones de fotos** de perros, gatos, pájaros, etc. El robot, por sí mismo, empieza a descubrir qué patrones en las fotos corresponden a cada animal.

¡Eso es **Deep Learning**! Es una parte del Machine Learning que usa **Redes Neuronales Artificiales** (inspiradas en el cerebro humano) con **muchas capas** para aprender de cantidades GIGANTES de datos. Estas redes son capaces de encontrar patrones muy complejos y abstractos en los datos, algo que otros algoritmos no pueden hacer.

#### ¿Por qué es tan "Profundo" (Deep)?
* **Muchas Capas:** Las redes neuronales profundas tienen muchas "capas" de neuronas (como los cerebros tienen muchas capas de neuronas). Cada capa aprende a reconocer características diferentes: la primera puede ver bordes y líneas, la segunda formas, la tercera partes de objetos, hasta que la última capa reconoce el objeto completo.
* **Aprende por sí mismo:** A diferencia de la programación tradicional donde le dices a la computadora qué buscar, en Deep Learning la red descubre las características importantes por sí misma a medida que ve más y más datos.

**Es especialmente bueno para:**
* **Reconocimiento de imágenes:** Identificar objetos, caras, o incluso diagnosticar enfermedades en radiografías.
* **Procesamiento de Lenguaje Natural (PLN):** Entender lo que decimos, traducir idiomas, o generar texto.
* **Reconocimiento de voz:** Convertir nuestra voz en texto.
* **Juegos y robótica:** Aprender a jugar a videojuegos o controlar robots.
""")

st.write("---")

# --- Sección de Explicación de Redes Neuronales ---
st.header("1. Redes Neuronales: Los Cimientos del Deep Learning")
st.markdown("""
Una **Red Neuronal Artificial** es el componente fundamental del Deep Learning. Imagina un conjunto de "neuronas" interconectadas, organizadas en capas:

* **Capas de Entrada:** Reciben los datos iniciales (por ejemplo, los píxeles de una imagen).
* **Capas Ocultas (¡Muchas!):** Aquí es donde ocurre la "magia". Cada neurona en estas capas toma decisiones simples y pasa su resultado a las neuronas de la siguiente capa. Al tener muchas capas, la red puede aprender representaciones de los datos cada vez más complejas y abstractas.
* **Capas de Salida:** Producen el resultado final (por ejemplo, la probabilidad de que una imagen sea un perro o un gato).

**Cómo aprenden:** A través de un proceso llamado **propagación hacia atrás (backpropagation)**, la red ajusta las conexiones entre sus neuronas (llamadas "pesos") basándose en lo bien que predice. Si se equivoca, ajusta los pesos para intentar hacerlo mejor la próxima vez. Es como si un estudiante revisara sus errores en un examen para no cometerlos de nuevo.
""")

if st.session_state.deep_learning_module_config['dl_model']:
    st.subheader("Visualizando el Aprendizaje: Ejemplo con Números Escritos a Mano (MNIST)")
    st.markdown("""
    Aquí te mostramos cómo una red neuronal "ve" y clasifica números escritos a mano. El modelo ha sido entrenado en el famoso conjunto de datos MNIST.
    """)
    
    x_test_dl = st.session_state.deep_learning_module_config['dl_data_test']
    y_test_dl = st.session_state.deep_learning_module_config['dl_labels_test']
    dl_model = st.session_state.deep_learning_module_config['dl_model']

    if x_test_dl is not None and y_test_dl is not None and dl_model is not None:
        st.write("Algunas predicciones del modelo:")
        
        fig_predictions, axes_predictions = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            idx = random.randint(0, len(x_test_dl) - 1)
            image = x_test_dl[idx]
            true_label = np.argmax(y_test_dl[idx])
            
            prediction_probs = dl_model.predict(image[np.newaxis, ...], verbose=0)[0]
            predicted_label = np.argmax(prediction_probs)
            
            axes_predictions[i].imshow(image.squeeze(), cmap='gray')
            axes_predictions[i].set_title(f"Real: {true_label}\nPred: {predicted_label}", color='green' if true_label == predicted_label else 'red')
            axes_predictions[i].axis('off')
        st.pyplot(fig_predictions)
        st.markdown("""
        Observa cómo el modelo, incluso simple, puede reconocer los dígitos.
        """)
    else:
        st.warning("No se pudieron cargar los datos de prueba o el modelo de Deep Learning. Asegúrate de que el modelo haya sido entrenado y guardado, y que los datos MNIST sean accesibles.")
else:
    st.warning(f"El modelo de Deep Learning no está cargado. Asegúrate de haber ejecutado el script 'train_model.py' y que el archivo {MODEL_LOAD_PATH} exista.")


st.write("---")

# --- Sección de Explicación de Redes Convolucionales (CNNs) ---
st.header("2. Redes Convolucionales (CNNs): Los Ojos del Deep Learning")
st.markdown("""
Las **Redes Neuronales Convolucionales (CNNs)** son un tipo especial de red neuronal profunda, ¡perfectas para trabajar con imágenes!

* **Capas de Convolución:** Estas capas actúan como "filtros" que recorren la imagen buscando patrones específicos, como bordes, esquinas o texturas. Es como si un detective examinara una foto en busca de huellas dactilares o detalles clave.
* **Capas de Agrupación (Pooling)::** Después de la convolución, estas capas reducen el tamaño de la imagen, manteniendo solo la información más importante. Esto ayuda a que la red sea más eficiente y a que las características aprendidas sean menos sensibles a pequeños cambios en la imagen (por ejemplo, si un objeto está ligeramente girado).
* **Jerarquía de Características:** Al apilar muchas capas convolucionales y de agrupación, una CNN aprende una jerarquía de características: las primeras capas detectan patrones simples, y las capas más profundas combinan esos patrones para detectar formas más complejas (ojos, narices, etc.) hasta reconocer objetos completos.

**Beneficios:** Extremadamente efectivas para tareas de visión por computador, como reconocimiento facial, coches autónomos y análisis médico de imágenes.
""")

st.write("---")

# --- Sección de Juego Interactivo: El Juego del Cerebro Digital ---
st.header("¡Juego Interactivo: Adivina el Número con el Cerebro Digital!")
st.markdown(f"""
¡Es hora de poner a prueba tu propio "cerebro digital"! Te mostraremos una imagen de un número escrito a mano y tendrás que adivinar qué número es. ¡Comprueba si eres tan bueno como una CNN!
**Aciertos: {st.session_state.deep_learning_module_config['dl_game_correct_count']} / {st.session_state.deep_learning_module_config['dl_game_total_count']}**
""")

if (tf is None or keras is None or
    st.session_state.deep_learning_module_config['dl_model'] is None or
    st.session_state.deep_learning_module_config['dl_data_test'] is None):
    st.warning("El juego no está disponible. Asegúrate de que `tensorflow` esté instalado y el modelo de Deep Learning esté cargado (entrenado previamente).")
else:
    def generate_new_game_point_dl():
        x_test_dl = st.session_state.deep_learning_module_config['dl_data_test']
        y_test_dl = st.session_state.deep_learning_module_config['dl_labels_test']
        
        idx = random.randint(0, len(x_test_dl) - 1)
        new_image = x_test_dl[idx]
        true_label = np.argmax(y_test_dl[idx])

        st.session_state.deep_learning_module_config['current_game_image_dl'] = new_image
        st.session_state.deep_learning_module_config['current_game_label_dl'] = true_label
        st.session_state.deep_learning_module_config['game_awaiting_guess_dl'] = True
        st.session_state.deep_learning_module_config['show_dl_explanation'] = False

    if not st.session_state.deep_learning_module_config['game_awaiting_guess_dl']:
        if st.button("¡Empezar una nueva ronda del juego DL!", key="start_dl_game_button"):
            generate_new_game_point_dl()
            st.rerun()

    if st.session_state.deep_learning_module_config['current_game_image_dl'] is not None:
        st.subheader("Observa la imagen y adivina el número:")
        
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(st.session_state.deep_learning_module_config['current_game_image_dl'].squeeze(), cmap='gray')
        ax.set_title("¿Qué número es este?", fontsize=16)
        ax.axis('off')
        st.pyplot(fig)

        if st.session_state.deep_learning_module_config['game_awaiting_guess_dl']:
            user_guess = st.number_input(
                "Mi adivinanza es el número...",
                min_value=0, max_value=9, value=0, step=1,
                key="dl_user_guess"
            )

            if st.button("¡Verificar mi adivinanza!", key="check_dl_guess_button"):
                st.session_state.deep_learning_module_config['dl_game_total_count'] += 1
                
                if user_guess == st.session_state.deep_learning_module_config['current_game_label_dl']:
                    st.session_state.deep_learning_module_config['dl_game_correct_count'] += 1
                    st.success(f"¡Correcto! El número era el **{st.session_state.deep_learning_module_config['current_game_label_dl']}**.")
                    st.balloons()
                else:
                    st.error(f"¡Incorrecto! El número era el **{st.session_state.deep_learning_module_config['current_game_label_dl']}**.")
                
                st.session_state.deep_learning_module_config['game_awaiting_guess_dl'] = False
                st.session_state.deep_learning_module_config['show_dl_explanation'] = True
                st.markdown(f"**Resultado actual del juego: {st.session_state.deep_learning_module_config['dl_game_correct_count']} aciertos de {st.session_state.deep_learning_module_config['dl_game_total_count']} intentos.**")
                st.button("¡Siguiente número!", key="next_dl_point_button", on_click=generate_new_game_point_dl)
                st.rerun()
        else:
            st.write("Haz clic en '¡Siguiente número!' para una nueva ronda.")
            if st.button("¡Siguiente número!", key="next_dl_point_after_reveal", on_click=generate_new_game_point_dl):
                st.rerun()

# --- Nueva Sección: ¿Por qué el Deep Learning? (Explicación Post-Juego) ---
if st.session_state.deep_learning_module_config['show_dl_explanation']:
    st.write("---")
    st.header("¿Por qué el Deep Learning es tan revolucionario?")
    st.markdown("""
    En el juego, habrás visto que reconocer un número, incluso escrito a mano, es una tarea que nuestro cerebro hace sin pensar. Para una máquina, esto es increíblemente difícil si no la "entrenamos" bien.

    * **Aprendizaje de Características Automático:** La mayor ventaja del Deep Learning es que la red aprende por sí misma las características relevantes de los datos. En el pasado, teníamos que "decirle" a la computadora qué buscar (por ejemplo, para reconocer un gato, le programaríamos que buscara "orejas puntiagudas", "bigotes", etc.). Las redes profundas descubren estas características por sí solas, incluso las que nosotros no podríamos imaginar.
    * **Escalabilidad con Grandes Datos:** Cuando tienes cantidades masivas de datos (como miles de millones de imágenes o horas de audio), el Deep Learning es inigualable. Cuantos más datos le des, ¡más inteligente se vuelve!
    * **Rendimiento de Vanguardia:** Ha roto récords en casi todas las áreas donde se aplica, superando con creces a otros métodos de Machine Learning en tareas como la visión por computador y el procesamiento del lenguaje.

    En resumen, el **Deep Learning** es como tener un "cerebro" digital que aprende y mejora constantemente, permitiendo a las máquinas realizar tareas complejas que antes solo los humanos podían hacer. ¡Es el motor detrás de muchas de las tecnologías más asombrosas que vemos hoy en día!
    """)
    st.write("---")


# --- Sección de Chatbot de Juego con Neo el Neurona para "Qué es el Deep Learning" ---
st.header("¡Despierta tu Neurona con Neo y el Deep Learning!")
st.markdown("¡Hola! Soy Neo, tu neurona particular que adora explorar las profundidades del aprendizaje de máquina. ¿Listo para construir cerebros artificiales?")

if client:
    # Inicializa el estado del juego y los mensajes del chat para Neo el Neurona
    if "neo_game_active" not in st.session_state:
        st.session_state.neo_game_active = False
    if "neo_game_messages" not in st.session_state:
        st.session_state.neo_game_messages = []
    if "neo_current_question" not in st.session_state:
        st.session_state.neo_current_question = None
    if "neo_current_options" not in st.session_state:
        st.session_state.neo_current_options = {}
    if "neo_correct_answer" not in st.session_state:
        st.session_state.neo_correct_answer = None
    if "neo_awaiting_next_game_decision" not in st.session_state:
        st.session_state.neo_awaiting_next_game_decision = False
    if "neo_game_needs_new_question" not in st.session_state:
        st.session_state.neo_game_needs_new_question = False
    if "neo_correct_streak" not in st.session_state:
        st.session_state.neo_correct_streak = 0
    if "last_played_question_neo_dl" not in st.session_state:
        st.session_state.last_played_question_neo_dl = None

    # System prompt para el juego de preguntas de Neo el Neurona
    neo_dl_game_system_prompt = f"""
    Eres un **experto consumado en Deep Learning y Redes Neuronales Artificiales**, con una especialización profunda en los **fundamentos de las redes neuronales, su arquitectura, el proceso de entrenamiento y sus aplicaciones clave**. Comprendes a fondo conceptos como neuronas, capas (entrada, oculta, salida), funciones de activación, pesos y sesgos, forward propagation, backpropagation, optimizadores, overfitting, y tipos de redes (FFNN, CNN, RNN). Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio del Deep Learning mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Activación perfecta! Tu neurona ha disparado con acierto." o "Esa conexión no fue la más fuerte. Repasemos cómo aprenden las neuronas."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "El Deep Learning es una rama del Machine Learning que utiliza redes neuronales con múltiples capas para aprender representaciones complejas de datos..."]
    [Pregunta para continuar, ej: "¿Listo para entrenar tus conexiones neuronales?" o "¿Quieres profundizar en la arquitectura de las redes?"]

    **Reglas adicionales para el Experto en Deep Learning (Redes Neuronales):**
    * **Enfoque Riguroso en Deep Learning:** Todas tus preguntas y explicaciones deben girar en torno a Deep Learning y Redes Neuronales. Cubre sus fundamentos (neuronas, conexiones, capas), el proceso de entrenamiento (forward pass, cálculo de error, backpropagation, optimización), funciones de activación (ReLU, Sigmoid, Tanh, Softmax), funciones de pérdida (MSE, Cross-Entropy), optimizadores (SGD, Adam), regularización (Dropout, L1/L2), overfitting y underfitting, y los tipos principales de redes:
        * **Redes Neuronales Feedforward (FFNN/MLP):** Concepto básico, capas.
        * **Redes Neuronales Convolucionales (CNN):** Para imágenes (filtros, pooling).
        * **Redes Neuronales Recurrentes (RNN):** Para secuencias (memoria, bucles).
    * **¡VARIEDAD, VARIEDAD, VARIEDAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de Deep Learning que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General de Deep Learning:** ¿Qué es? Diferencia con Machine Learning tradicional. La analogía con el cerebro.
        * **Neuronas y Capas:** Función de una neurona, tipos de capas (entrada, oculta, salida).
        * **Pesos y Sesgos:** Su rol en el aprendizaje.
        * **Funciones de Activación:** Propósito y ejemplos (ReLU, Sigmoid).
        * **Forward Propagation:** Cómo viaja la información.
        * **Backpropagation:** El mecanismo de aprendizaje.
        * **Optimizadores:** (e.g., Gradiente Descendente, Adam) su rol en ajustar pesos.
        * **Función de Pérdida/Costo:** Qué mide y por qué es importante.
        * **Overfitting y Underfitting:** Cómo identificarlos y mitigarlos.
        * **Regularización:** Dropout, L1/L2.
        * **Tipos de Redes:**
            * **FFNN/MLP:** Cuándo se usan.
            * **CNN:** Para qué son ideales (visión artificial).
            * **RNN:** Para qué son ideales (secuencias, lenguaje natural).
        * **Aplicaciones Prácticas:** Visión por computador, procesamiento del lenguaje natural (NLP), sistemas de recomendación, vehículos autónomos.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.neo_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Neurona Despierta – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de una neurona y cómo aprenden los modelos.
            * *Tono:* "Estás activando tus primeras conexiones neuronales. ¡El viaje del conocimiento ha comenzado!"
        * **Nivel 2 (Conexionista Curioso – 3-5 respuestas correctas):** Tono más técnico. Introduce conceptos como capas, pesos, y el flujo de información. Preguntas sobre los componentes básicos y el proceso fundamental.
            * *Tono:* "Tus conexiones se hacen más fuertes. Estás entendiendo cómo las redes construyen conocimiento."
        * **Nivel 3 (Arquitecto de Redes – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en funciones de activación, backpropagation, optimizadores, y la diferencia entre overfitting/underfitting.
            * *Tono:* "Estás diseñando arquitecturas que piensan. Tu comprensión del Deep Learning es profunda y precisa."
        * **Nivel Maestro (Científico de Datos Neuronal – 9+ respuestas correctas):** Tono de **especialista en diseño y entrenamiento de redes profundas**. Preguntas sobre tipos avanzados de capas (CNN, RNN), regularización, o la optimización de hiperparámetros. Se esperan respuestas que demuestren una comprensión teórica y práctica profunda, incluyendo cómo elegir la arquitectura adecuada para diferentes problemas.
            * *Tono:* "Tu maestría en el Deep Learning te permite construir cerebros artificiales capaces de aprender y resolver los problemas más complejos. ¡Un verdadero genio neuronal!"
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Cómo una neurona "aprende" a reconocer un perro en una foto simple, o a decidir si es de día o de noche.
        * **Nivel 2:** Explicar cómo varias capas ayudan a una red a distinguir diferentes razas de perros, o a clasificar correos electrónicos en "importantes" o "no importantes".
        * **Nivel 3:** Cómo una CNN puede detectar objetos específicos en una imagen compleja (coches, personas), o cómo una RNN puede predecir la siguiente palabra en una frase.
        * **Nivel Maestro:** Diseñar una red generativa adversaria (GAN) para crear imágenes realistas, o implementar una red de transformers para la traducción automática de idiomas, abordando los desafíos de escala y eficiencia.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre DEEP LEARNING (Redes Neuronales), y asegúrate de que no se parezca a las anteriores.
    """

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_neo_dl_question_response(raw_text):
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
    def parse_neo_dl_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"

    # --- Funciones para subir de nivel directamente ---
    def set_neo_dl_level(target_streak, level_name):
        st.session_state.neo_correct_streak = target_streak
        st.session_state.neo_game_active = True
        st.session_state.neo_game_messages = []
        st.session_state.neo_current_question = None
        st.session_state.neo_current_options = {}
        st.session_state.neo_correct_answer = None
        st.session_state.neo_game_needs_new_question = True
        st.session_state.neo_awaiting_next_game_decision = False
        st.session_state.neo_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}** de Neo! Prepárate para preguntas más desafiantes sobre Deep Learning. ¡Aquí va tu primera!"})
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons_neo_dl, col_level_up_buttons_neo_dl = st.columns([1, 2])

    with col_game_buttons_neo_dl:
        if st.button("¡Vamos a jugar con Neo!", key="start_neo_dl_game_button"):
            st.session_state.neo_game_active = True
            st.session_state.neo_game_messages = []
            st.session_state.neo_current_question = None
            st.session_state.neo_current_options = {}
            st.session_state.neo_correct_answer = None
            st.session_state.neo_game_needs_new_question = True
            st.session_state.neo_awaiting_next_game_decision = False
            st.session_state.neo_correct_streak = 0
            st.session_state.last_played_question_neo_dl = None
            st.rerun()
    
    with col_level_up_buttons_neo_dl:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un arquitecto neuronal? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_neo_dl, col_lvl2_neo_dl, col_lvl3_neo_dl = st.columns(3) # Tres columnas para los botones de nivel
        with col_lvl1_neo_dl:
            if st.button("Subir a Nivel Medio (Neo - Conexionista)", key="level_up_medium_neo_dl"):
                set_neo_dl_level(3, "Medio") # 3 respuestas correctas para Nivel Medio
        with col_lvl2_neo_dl:
            if st.button("Subir a Nivel Avanzado (Neo - Arquitecto)", key="level_up_advanced_neo_dl"):
                set_neo_dl_level(6, "Avanzado") # 6 respuestas correctas para Nivel Avanzado
        with col_lvl3_neo_dl:
            if st.button("👑 ¡Científico de Datos Neuronal! (Neo)", key="level_up_champion_neo_dl"):
                set_neo_dl_level(9, "Campeón") # 9 respuestas correctas para Nivel Campeón

    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.neo_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica del juego del chatbot si está activo
    if st.session_state.neo_game_active:
        if st.session_state.neo_current_question is None and st.session_state.neo_game_needs_new_question and not st.session_state.neo_awaiting_next_game_decision:
            with st.spinner("Neo está preparando una pregunta sobre Deep Learning..."):
                try:
                    # Incluimos el prompt del sistema actualizado con el nivel de dificultad
                    game_messages_for_api = [{"role": "system", "content": neo_dl_game_system_prompt}]
                    # Limita el historial para evitar prompts demasiado largos, tomando las últimas interacciones relevantes
                    if st.session_state.neo_game_messages:
                        last_message = st.session_state.neo_game_messages[-1]
                        if last_message["role"] == "user":
                            game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {last_message['content']}"})
                        elif last_message["role"] == "assistant":
                            # Si el último mensaje fue del asistente (feedback), lo añadimos para que sepa dónde se quedó
                            game_messages_for_api.append({"role": "assistant", "content": last_message['content']})

                    game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre DEEP LEARNING (Redes Neuronales) siguiendo el formato exacto. ¡Recuerda, la pregunta debe ser muy VARIADA y CREATIVA, y no se debe parecer a las anteriores!"})

                    game_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=game_messages_for_api,
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_neo_dl_question_text = game_response.choices[0].message.content
                    question, options, correct_answer_key = parse_neo_dl_question_response(raw_neo_dl_question_text)

                    if question:
                        st.session_state.neo_current_question = question
                        st.session_state.neo_current_options = options
                        st.session_state.neo_correct_answer = correct_answer_key

                        display_question_text = f"**Nivel {int(st.session_state.neo_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.neo_correct_streak}**\n\n**Pregunta de Neo:** {question}\n\n"
                        for key in sorted(options.keys()):
                            display_question_text += f"{key}) {options[key]}\n"

                        st.session_state.neo_game_messages.append({"role": "assistant", "content": display_question_text})
                        st.session_state.neo_game_needs_new_question = False
                        st.rerun()
                    else:
                        st.session_state.neo_game_messages.append({"role": "assistant", "content": "¡Lo siento! Neo no pudo preparar la pregunta en el formato correcto. ¿Puedes pulsar '¡Vamos a jugar!' de nuevo?"})
                        st.session_state.neo_game_active = False
                        st.rerun()

                except Exception as e:
                    st.error(f"¡Oops! Neo no pudo hacer la pregunta. Error: {e}")
                    st.session_state.neo_game_messages.append({"role": "assistant", "content": "¡Lo siento! Neo tiene un pequeño problema para hacer preguntas ahora. ¿Puedes intentarlo de nuevo?"})
                    st.session_state.neo_game_active = False
                    st.rerun()


        if st.session_state.neo_current_question is not None and not st.session_state.neo_awaiting_next_game_decision:
            # Audio de la pregunta
            if st.session_state.get('last_played_question_neo_dl') != st.session_state.neo_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.neo_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.neo_correct_streak}. Pregunta de Neo: {st.session_state.neo_current_question}. Opción A: {st.session_state.neo_current_options.get('A', '')}. Opción B: {st.session_state.neo_current_options.get('B', '')}. Opción C: {st.session_state.neo_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                    st.session_state.last_played_question_neo_dl = st.session_state.neo_current_question
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")

            with st.form("neo_dl_game_form", clear_on_submit=True):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_choice = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.neo_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.neo_current_options[x]}",
                        key="neo_dl_answer_radio_buttons",
                        label_visibility="collapsed"
                    )

                submit_button = st.form_submit_button("Enviar Respuesta")

            if submit_button:
                st.session_state.neo_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_choice}) {st.session_state.neo_current_options[user_choice]}"})
                prev_streak = st.session_state.neo_correct_streak

                # Lógica para actualizar el contador de respuestas correctas
                if user_choice == st.session_state.neo_correct_answer:
                    st.session_state.neo_correct_streak += 1
                else:
                    st.session_state.neo_correct_streak = 0

                radio_placeholder.empty()

                # --- Lógica de subida de nivel ---
                if st.session_state.neo_correct_streak > 0 and \
                   st.session_state.neo_correct_streak % 3 == 0 and \
                   st.session_state.neo_correct_streak > prev_streak:
                    
                    if st.session_state.neo_correct_streak < 9: # Niveles Básico, Medio, Avanzado
                        current_level_text = ""
                        if st.session_state.neo_correct_streak == 3:
                            current_level_text = "Medio (como un joven que entiende cómo se interconectan las ideas)"
                        elif st.session_state.neo_correct_streak == 6:
                            current_level_text = "Avanzado (como un Data Scientist que construye cerebros artificiales)"
                        
                        level_up_message = f"🎉 ¡Increíble! ¡Has respondido {st.session_state.neo_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de Deep Learning. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a activador/a de neuronas!"
                        st.session_state.neo_game_messages.append({"role": "assistant", "content": level_up_message})
                        st.balloons()
                        # Generar audio
                        try:
                            tts_level_up = gTTS(text=level_up_message, lang='es', slow=False)
                            audio_fp_level_up = io.BytesIO()
                            tts_level_up.write_to_fp(audio_fp_level_up)
                            audio_fp_level_up.seek(0)
                            st.audio(audio_fp_level_up, format="audio/mp3", start_time=0, autoplay=True)
                            time.sleep(2) # Pequeña pausa para que se reproduzca
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de subida de nivel: {e}")
                    elif st.session_state.neo_correct_streak >= 9:
                        medals_earned = (st.session_state.neo_correct_streak - 6) // 3
                        medal_message = f"🏅 ¡FELICITACIONES, CIENTÍFICO DE DATOS NEURONAL! ¡Has ganado tu {medals_earned}ª Medalla del Deep Learning! ¡Tu habilidad para dominar las redes neuronales es asombrosa y digna de un verdadero EXPERTO en IA! ¡Sigue así!"
                        st.session_state.neo_game_messages.append({"role": "assistant", "content": medal_message})
                        st.balloons()
                        st.snow()
                        try:
                            tts_medal = gTTS(text=medal_message, lang='es', slow=False)
                            audio_fp_medal = io.BytesIO()
                            tts_medal.write_to_fp(audio_fp_medal)
                            audio_fp_medal.seek(0)
                            st.audio(audio_fp_medal, format="audio/mp3", start_time=0, autoplay=True)
                            time.sleep(3)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de medalla: {e}")

                        if prev_streak < 9:
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Campeón (Científico de Datos Neuronal)**! ¡Las preguntas ahora son solo para los verdaderos genios que programan el futuro de la IA! ¡Adelante!"
                            st.session_state.neo_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                audio_fp_level_up_champion = io.BytesIO()
                                tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                                audio_fp_level_up_champion.seek(0)
                                st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0, autoplay=True)
                                time.sleep(2)
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón: {e}")

                # Generar feedback de Neo
                with st.spinner("Neo está revisando tu respuesta..."):
                    try:
                        feedback_prompt = f"""
                        El usuario respondió '{user_choice}'. La pregunta era: '{st.session_state.neo_current_question}'.
                        La respuesta correcta era '{st.session_state.neo_correct_answer}'.
                        Da feedback como Neo.
                        Si es CORRECTO, el mensaje es "¡Activación perfecta! Tu neurona ha disparado con acierto." o similar.
                        Si es INCORRECTO, el mensaje es "¡Esa conexión no fue la más fuerte. Repasemos cómo aprenden las neuronas!" o similar.
                        Luego, una explicación concisa y clara.
                        Finalmente, pregunta: "¿Quieres seguir entrenando tus conexiones neuronales?".
                        **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                        """
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": neo_dl_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.8,
                            max_tokens=300
                        )
                        raw_neo_dl_feedback_text = feedback_response.choices[0].message.content

                        feedback_msg, explanation_msg, next_question_prompt = parse_neo_dl_feedback_response(raw_neo_dl_feedback_text)

                        st.session_state.neo_game_messages.append({"role": "assistant", "content": feedback_msg})
                        st.session_state.neo_game_messages.append({"role": "assistant", "content": explanation_msg})
                        st.session_state.neo_game_messages.append({"role": "assistant", "content": next_question_prompt})

                        try:
                            tts = gTTS(text=f"{feedback_msg}. {explanation_msg}. {next_question_prompt}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")

                        st.session_state.neo_current_question = None
                        st.session_state.neo_current_options = {}
                        st.session_state.neo_correct_answer = None
                        st.session_state.neo_game_needs_new_question = False
                        st.session_state.neo_awaiting_next_game_decision = True

                        st.rerun()

                    except Exception as e:
                        st.error(f"Ups, Neo no pudo procesar tu respuesta. Error: {e}")
                        st.session_state.neo_game_messages.append({"role": "assistant", "content": "Lo siento, Neo tiene un pequeño problema técnico ahora mismo. ¡Pero me encantaría ver tu respuesta!"})


        if st.session_state.neo_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Sí, quiero más desafíos neuronales", key="play_more_questions_neo_dl"):
                    st.session_state.neo_game_needs_new_question = True
                    st.session_state.neo_awaiting_next_game_decision = False
                    st.session_state.neo_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío!"})
                    st.rerun()
            with col2:
                if st.button("👎 No, ya he activado suficiente mi cerebro", key="stop_playing_neo_dl"):
                    st.session_state.neo_game_active = False
                    st.session_state.neo_awaiting_next_game_decision = False
                    st.session_state.neo_game_messages.append({"role": "assistant", "content": "¡De acuerdo! ¡Gracias por entrenar conmigo! Espero que hayas aprendido mucho sobre Deep Learning. ¡Nos vemos pronto!"})
                    st.rerun()

else:
    st.info("Para usar la sección de preguntas de Neo, necesitas configurar tu clave de API de OpenAI en `secrets.toml`.")