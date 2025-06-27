# pages/01_Que es Inteligencia Artificial.py
# KNN clasifier

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from streamlit_lottie import st_lottie
from openai import OpenAI
from gtts import gTTS
import io
import random
import time
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

st.set_page_config(
    page_title="¿Qué es la IA?",
    layout="wide"
)

# ---- Función para cargar animación Lottie desde un archivo local ----
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo Lottie en la ruta: {filepath}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: El archivo Lottie '{filepath}' no es un JSON válido.")
        return None

# --- Rutas a Lottie ---
# Asegúrate de que este archivo exista en la estructura de tu proyecto: assets/lottie_animations/Math.json
LOTTIE_THINKING_ROBOT_PATH = os.path.join("assets", "lottie_animations", "Math.json")

# --- Configuración de la API de OpenAI ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai_api_key = None

client = OpenAI(api_key=openai_api_key) if openai_api_key else None


st.subheader("¡Descubre la magia de las máquinas que piensan!")

st.write("---")

# Sección 1: ¿Qué es la Inteligencia Artificial?
st.header("¿Qué es la Inteligencia Artificial?")
st.markdown("""
Imagina que tienes un amigo robot muy, muy inteligente. Este amigo no solo sigue instrucciones,
sino que también puede **aprender**, **entender** y **tomar decisiones** por sí mismo, ¡casi como tú!

Eso es la **Inteligencia Artificial (IA)**: hacer que las máquinas sean tan inteligentes que puedan
resolver problemas, entender lo que decimos, ver lo que pasa a su alrededor y hasta crear cosas nuevas.
¡Es como darles un cerebro a las computadoras!
""")

# Pequeña animación para la introducción
col_intro_left, col_intro_right = st.columns([1, 1])
with col_intro_right:
    lottie_thinking_robot = load_lottiefile(LOTTIE_THINKING_ROBOT_PATH)
    if lottie_thinking_robot:
        st_lottie(lottie_thinking_robot, height=200, width=200, key="thinking_robot_intro")
    else:
        st.info("Consejo: Asegúrate de que 'Math.json' esté en 'assets/lottie_animations/' para esta animación.")


st.write("---")

# Sección 2: ¿Cómo Aprende la IA? (Visualización Interactiva)
st.header("¿Cómo Aprende la Inteligencia Artificial?")
st.markdown("""
Las máquinas con IA aprenden de una forma parecida a como lo hacemos nosotros: ¡observando y practicando!
Cuanta más información les damos, más listas se vuelven. Es como si les diéramos muchos ejemplos
para que entiendan patrones.

**¡Vamos a simular cómo aprende una IA a clasificar cosas!**
""")

# --- Visualización interactiva con Matplotlib ---
st.subheader("Clasificación de Frutas: ¡Ayuda a la IA a aprender!")

st.markdown("""
Puedes ajustar los controles para ver cómo cambia la forma en que la IA podría aprender.
""")

# --- CONTROLES PARA LA CLASIFICACIÓN DE FRUTAS ---
col_params1, col_params2, col_params3 = st.columns(3)

with col_params1:
    num_datos_entrenamiento = st.slider(
        "Cantidad de ejemplos de entrenamiento:",
        min_value=10, max_value=100, value=40, step=10,
        help="Cuantos más ejemplos, mejor aprenderá la IA a clasificar."
    )
with col_params2:
    dispersión_datos = st.slider(
        "Dispersión de los datos (dificultad):",
        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        help="Un valor más alto hace que los datos estén más mezclados y sea más difícil para la IA."
    )
with col_params3:
    num_clases = st.selectbox(
        "Número de tipos de 'frutas':",
        options=[2, 3], index=0, # Por defecto 2 (Manzanas y Limones)
        help="Elige cuántos tipos de 'frutas' quieres que la IA intente clasificar."
    )

# Crear algunos datos de ejemplo (frutas con dos características: dulzura y tamaño)
np.random.seed(42)

# Inicializar listas para datos de entrenamiento del modelo
X_train = [] # Características (Dulzura, Tamaño)
y_train = [] # Etiquetas (0: Manzana, 1: Limón, 2: Naranja)
class_names = {0: 'Manzana 🍎', 1: 'Limón 🍋', 2: 'Naranja 🍊'}
class_colors = {0: 'green', 1: 'yellow', 2: 'orange'}

fig, ax = plt.subplots(figsize=(8, 6))

# Datos para la primera clase (ej. Manzanas)
manzanas_dulzura = np.random.normal(loc=7, scale=dispersión_datos, size=num_datos_entrenamiento)
manzanas_tamano = np.random.normal(loc=8, scale=dispersión_datos, size=num_datos_entrenamiento)
ax.scatter(manzanas_dulzura, manzanas_tamano,
            color='green', label='Manzanas 🍎', alpha=0.7)
X_train.extend(list(zip(manzanas_dulzura, manzanas_tamano)))
y_train.extend([0] * num_datos_entrenamiento)

# Datos para la segunda clase (ej. Limones)
limones_dulzura = np.random.normal(loc=3, scale=dispersión_datos, size=num_datos_entrenamiento)
limones_tamano = np.random.normal(loc=4, scale=dispersión_datos, size=num_datos_entrenamiento)
ax.scatter(limones_dulzura, limones_tamano,
            color='yellow', label='Limones 🍋', alpha=0.7)
X_train.extend(list(zip(limones_dulzura, limones_tamano)))
y_train.extend([1] * num_datos_entrenamiento)

# Datos para la tercera clase (ej. Naranjas), solo si se selecciona 3 clases
if num_clases == 3:
    naranjas_dulzura = np.random.normal(loc=5, scale=dispersión_datos, size=num_datos_entrenamiento)
    naranjas_tamano = np.random.normal(loc=6, scale=dispersión_datos, size=num_datos_entrenamiento)
    ax.scatter(naranjas_dulzura, naranjas_tamano,
                color='orange', label='Naranjas 🍊', alpha=0.7)
    X_train.extend(list(zip(naranjas_dulzura, naranjas_tamano)))
    y_train.extend([2] * num_datos_entrenamiento)

X_train = np.array(X_train)
y_train = np.array(y_train)

# --- Entrenamiento del modelo predictivo (KNN) ---
k_neighbors = int(np.sqrt(len(X_train))) if len(X_train) > 0 else 1
if k_neighbors % 2 == 0:
    k_neighbors += 1
k_neighbors = max(1, k_neighbors)

model = None
# Solo entrenar si hay al menos dos clases únicas en y_train, y al menos K_neighbors puntos
if len(np.unique(y_train)) > 1 and len(X_train) >= k_neighbors:
    model = KNeighborsClassifier(n_neighbors=k_neighbors)
    model.fit(X_train, y_train)

# --- Visualización de las fronteras de decisión del modelo ---
# Definir límites de la cuadrícula para el meshgrid (0 a 10 para dulzura y tamaño)
plot_x_min, plot_x_max = 0, 10
plot_y_min, plot_y_max = 0, 10

# Crear la cuadrícula de puntos para el meshgrid
xx, yy = np.meshgrid(np.linspace(plot_x_min, plot_x_max, 100),
                      np.linspace(plot_y_min, plot_y_max, 100))

# Solo dibujar las fronteras si el modelo ha sido entrenado
if model is not None:
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Crear mapa de colores para las regiones de decisión
    if num_clases == 2:
        cmap_background = ListedColormap(['#90EE90', '#FFFF99'])
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
    elif num_clases == 3:
        cmap_background = ListedColormap(['#90EE90', '#FFFF99', '#FFDAB9'])
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)


ax.set_xlabel("Nivel de Dulzura")
ax.set_ylabel("Tamaño")
ax.set_title(f"Clasificación de Frutas por IA (con {num_datos_entrenamiento} ejemplos por fruta)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(plot_x_min, plot_x_max)
ax.set_ylim(plot_y_min, plot_y_max)


# --- Sección para que el niño añada su propia fruta ---
st.markdown("---")
st.subheader("¡Añade tu propia fruta y mira cómo la clasificaría la IA!")

# Límite de frutas a añadir
MAX_FRUTAS_USUARIO = 10

# Inicializar la lista de frutas añadidas por el usuario en session_state
if 'user_added_fruits' not in st.session_state:
    st.session_state.user_added_fruits = []

col_add_fruit1, col_add_fruit2, col_add_fruit3 = st.columns(3)

with col_add_fruit1:
    user_dulzura = st.slider("Dulzura de tu fruta:", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key="user_fruit_dulzura")
with col_add_fruit2:
    user_tamano = st.slider("Tamaño de tu fruta:", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key="user_fruit_tamano")
with col_add_fruit3:
    st.markdown(" ")
    st.markdown(" ")
    add_fruit_button = st.button("➕ Añadir mi Fruta al gráfico", key="add_user_fruit")
    
    if add_fruit_button:
        if len(st.session_state.user_added_fruits) >= MAX_FRUTAS_USUARIO:
            st.session_state.user_added_fruits = [] # Reiniciar la lista si se alcanza el límite
            st.warning(f"¡Has añadido {MAX_FRUTAS_USUARIO} frutas! Se han borrado para que puedas añadir más.")
        st.session_state.user_added_fruits.append({'dulzura': user_dulzura, 'tamano': user_tamano})
        st.rerun() # Recargar para que la fruta aparezca en el gráfico

# Dibujar las frutas añadidas por el usuario
for i, fruit in enumerate(st.session_state.user_added_fruits):
    ax.scatter(fruit['dulzura'], fruit['tamano'],
                color='purple', marker='*', s=200, edgecolor='black', linewidth=1.5,
                label=f'Mi Fruta {i+1} 🍇' if i == 0 else "_nolegend_",
                zorder=5)
    
    # Predecir la clase de la fruta añadida usando el modelo entrenado
    predicted_class_name = "¡No hay suficientes datos para clasificarla!"

    if model is not None:
        prediction_input = np.array([[fruit['dulzura'], fruit['tamano']]])
        predicted_class_id = model.predict(prediction_input)[0]
        predicted_class_name = class_names.get(predicted_class_id, "¡es una fruta misteriosa!") # Obtener el nombre
    
    st.markdown(f"La IA diría que tu fruta (Dulzura: {fruit['dulzura']:.1f}, Tamaño: {fruit['tamano']:.1f}) {predicted_class_name}")


st.pyplot(fig) # Vuelve a mostrar el gráfico con las nuevas frutas añadidas

st.markdown("""
¿Ves las áreas de colores en el fondo? Esas son las **"fronteras de decisión"** de la IA.
La IA ha aprendido de todos los puntos que le diste, y ahora sabe qué tipo de fruta
es probable que sea en cada parte del gráfico. ¡Si tu fruta cae en una de esas áreas,
la IA la clasificará como esa fruta!

Cuantos más ejemplos le des, y menos mezclados estén (baja dispersión), más claras
serán esas fronteras y mejor clasificará la IA. ¡Es como si aprendiera las reglas del juego!
""")

st.write("---")

# --- Sección de Chatbot de Juego con Byte ---
st.header("¡Juega y Aprende con Byte sobre la Inteligencia Artificial!")
st.markdown("¡Hola! Soy **Byte**, tu compañero digital que sabe todo sobre cómo las máquinas aprenden y predicen. ¿Listo para descubrir cómo los modelos predictivos nos ayudan a 'adivinar' el futuro con datos?")

if client:
    # Inicializa el estado del juego y los mensajes del chat
    if "inteligent_game_active" not in st.session_state:
        st.session_state.inteligent_game_active = False
    if "inteligent_game_messages" not in st.session_state:
        st.session_state.inteligent_game_messages = []
    if "inteligent_current_question" not in st.session_state:
        st.session_state.inteligent_current_question = None
    if "inteligent_current_options" not in st.session_state:
        st.session_state.inteligent_current_options = {}
    if "inteligent_correct_answer" not in st.session_state:
        st.session_state.inteligent_correct_answer = None
    if "inteligent_awaiting_next_game_decision" not in st.session_state:
        st.session_state.inteligent_awaiting_next_game_decision = False
    if "inteligent_game_needs_new_question" not in st.session_state:
        st.session_state.inteligent_game_needs_new_question = False
    if "inteligent_correct_streak" not in st.session_state:
        st.session_state.inteligent_correct_streak = 0
    if "last_played_question_inteligent" not in st.session_state: # Nueva clave para audio
        st.session_state.last_played_question_inteligent = None


    # System prompt para el juego de preguntas de Byte
    inteligent_game_system_prompt = f"""
    Eres un **experto y líder de opinión en el campo de la Inteligencia Artificial (IA)**, con un profundo conocimiento de sus fundamentos, aplicaciones y desafíos éticos. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de la IA mediante un **juego de preguntas adaptativo**. Aunque el entorno inicial pueda parecer "amigable", tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Análisis impecable! Has optimizado tu comprensión del concepto de IA." o "Revisa tu algoritmo mental. Esa no era la respuesta óptima."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "La IA se define como la simulación de procesos de inteligencia humana por máquinas..."]
    [Pregunta para continuar, ej: "¿Listo para el siguiente desafío en el ámbito de los sistemas inteligentes?" o "¿Quieres profundizar más en la evolución de los modelos de IA?"]

    **Reglas adicionales para el Experto en Inteligencia Artificial:**
    * **Enfoque Riguroso en Inteligencia Artificial:** Todas tus preguntas y explicaciones deben girar en torno a la IA. Cubre sus fundamentos (definición, tipos de IA), subcampos (Machine Learning, Deep Learning, Procesamiento del Lenguaje Natural, Visión por Computadora, Robótica), algoritmos clave (redes neuronales, algoritmos de clustering), aplicaciones prácticas, historia, desafíos éticos y sociales, y el futuro de la IA.
    * **¡VARIEDAD, VARIEDAD, VARIEDAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de IA que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General y Definiciones:** ¿Qué es IA? Tipos de IA (IA débil/fuerte, ANI/AGI/ASI).
        * **Historia y Hitos:** Eventos clave, figuras influyentes, "inviernos" de la IA.
        * **Machine Learning (ML):** Aprendizaje supervisado/no supervisado/por refuerzo, algoritmos básicos (regresión, clasificación, clustering).
        * **Deep Learning (DL):** Redes neuronales (DNN, CNN, RNN), GPUs, Big Data como habilitadores.
        * **Procesamiento del Lenguaje Natural (NLP):** Comprensión del lenguaje, traducción automática, chatbots, modelos de lenguaje grandes (LLMs).
        * **Visión por Computadora (CV):** Reconocimiento de imágenes, detección de objetos, visión robótica.
        * **Robótica e IA:** Robots autónomos, interacción humano-robot.
        * **Ética y Sesgos en IA:** Fairness, transparencia, privacidad, responsabilidad.
        * **Aplicaciones Prácticas:** Ejemplos en medicina, finanzas, transporte, entretenimiento.
        * **Desafíos y Futuro de la IA:** Limitaciones actuales, IA explicable (XAI), regulación.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.inteligent_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Curioso – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la definición básica de IA y ejemplos cotidianos. Analogías simples para ilustrar conceptos fundamentales.
            * *Tono:* "Estás dando tus primeros pasos en el vasto universo de la Inteligencia Artificial."
        * **Nivel 2 (Desarrollador Junior de IA – 3-5 respuestas correctas):** Tono más técnico. Introduce subcampos como Machine Learning, conceptos de datos y entrenamiento de modelos de forma directa. Preguntas sobre las capacidades básicas de los sistemas de IA.
            * *Tono:* "Tu comprensión de los fundamentos de la IA es sólida, estás listo para aplicar tus conocimientos."
        * **Nivel 3 (Ingeniero de IA – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Introduce algoritmos específicos (ej. tipos de redes neuronales, clustering), conceptos de rendimiento de modelos, y la importancia de los datos. Preguntas que requieren una comprensión de cómo funcionan los sistemas de IA a un nivel más profundo.
            * *Tono:* "Tu análisis demuestra una comprensión profunda de los algoritmos y arquitecturas de IA."
        * **Nivel Maestro (Científico de Investigación en IA – 9+ respuestas correctas):** Tono de **especialista en investigación y desarrollo de vanguardia en IA**. Preguntas sobre desafíos abiertos, implicaciones éticas complejas, arquitecturas avanzadas de modelos (Transformers, GANs), o el impacto socioeconómico de la IA. Se esperan respuestas que demuestren una comprensión crítica, teórica y práctica robusta.
            * *Tono:* "Tu maestría en el diseño, implementación y evaluación de sistemas inteligentes es excepcional. Estás en la vanguardia de la innovación en IA."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Analogías (Adaptadas al Nivel):**
        * **Nivel 1:** Un asistente de voz que te ayuda con tareas diarias.
        * **Nivel 2:** Un sistema de recomendación que sugiere películas o música.
        * **Nivel 3:** Un algoritmo de visión por computadora que detecta enfermedades en radiografías, o un modelo de lenguaje que genera texto coherente.
        * **Nivel Maestro:** El desarrollo de una IA robusta y ética para vehículos autónomos, o la creación de nuevos paradigmas de aprendizaje automático inspirados en la neurociencia.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre la INTELIGENCIA ARTIFICIAL, y asegúrate de que no se parezca a las anteriores.
    """

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_inteligent_question_response(raw_text):
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
            st.warning(f"DEBUG (Byte): Formato de pregunta inesperado de la API. Texto recibido:\n{raw_text}")
            return None, {}, ""
        return question, options, correct_answer_key

    # Función para parsear la respuesta de feedback de la IA
    def parse_inteligent_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG (Byte): Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"
    
    # --- Funciones para subir de nivel directamente ---
    def set_inteligent_level(target_streak, level_name):
        st.session_state.inteligent_correct_streak = target_streak
        st.session_state.inteligent_game_active = True
        st.session_state.inteligent_game_messages = []
        st.session_state.inteligent_current_question = None
        st.session_state.inteligent_current_options = {}
        st.session_state.inteligent_correct_answer = None
        st.session_state.inteligent_game_needs_new_question = True
        st.session_state.inteligent_awaiting_next_game_decision = False
        st.session_state.inteligent_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}**! Prepárate para preguntas más desafiantes. ¡Aquí va tu primera!"})
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons, col_level_up_buttons = st.columns([1, 2])

    with col_game_buttons:
        if st.button("¡Vamos a jugar con Byte!", key="start_byte_game_button"):
            st.session_state.inteligent_game_active = True
            st.session_state.intelient_game_messages = [] # Corrected typo here as well
            st.session_state.inteligent_current_question = None
            st.session_state.inteligent_current_options = {}
            st.session_state.inteligent_correct_answer = None
            st.session_state.inteligent_game_needs_new_question = True
            st.session_state.inteligent_awaiting_next_game_decision = False
            st.session_state.inteligent_correct_streak = 0 # Reiniciar el contador al inicio del juego
            st.session_state.last_played_question_inteligent = None # Reiniciar también esta clave
            st.rerun()
    
    with col_level_up_buttons:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un experto? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1, col_lvl2, col_lvl3 = st.columns(3) # Tres columnas para los botones de nivel
        with col_lvl1:
            if st.button("Subir a Nivel Constructor", key="level_up_medium_byte"):
                set_inteligent_level(3, "Constructor de Byte") # 3 respuestas correctas para Nivel Medio
        with col_lvl2:
            if st.button("Subir a Nivel Arquitecto", key="level_up_advanced_byte"):
                set_inteligent_level(6, "Arquitecto de Byte") # 6 respuestas correctas para Nivel Avanzado
        with col_lvl3:
            if st.button("¡Nivel Maestro de Byte!", key="level_up_champion_byte"):
                set_inteligent_level(9, "Maestro de Byte") # 9 respuestas correctas para Nivel Campeón


    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.inteligent_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica del juego del chatbot si está activo
    if st.session_state.inteligent_game_active:
        if st.session_state.inteligent_current_question is None and st.session_state.inteligent_game_needs_new_question and not st.session_state.inteligent_awaiting_next_game_decision:
            with st.spinner("Byte está preparando una pregunta..."):
                try:
                    # Incluimos el prompt del sistema actualizado con el nivel de dificultad
                    game_messages_for_api = [{"role": "system", "content": inteligent_game_system_prompt}]
                    # Limita el historial para evitar prompts demasiado largos, tomando las últimas interacciones relevantes
                    for msg in st.session_state.inteligent_game_messages[-6:]:
                        if msg["role"] == "assistant" and msg["content"].startswith("**"):
                            game_messages_for_api.append({"role": "assistant", "content": f"PREGUNTA ANTERIOR: {msg['content'].splitlines()[0]}"})
                        elif msg["role"] == "user" and "MI RESPUESTA:" not in msg["content"]:
                            game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {msg['content']}"})

                    game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre INTELIGENCIA ARTIFICIAL siguiendo el formato exacto."}) # Cambiado prompt aquí

                    game_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=game_messages_for_api,
                        temperature=0.7,
                        max_tokens=250
                    )
                    raw_inteligent_question_text = game_response.choices[0].message.content
                    question, options, correct_answer_key = parse_inteligent_question_response(raw_inteligent_question_text)

                    if question:
                        st.session_state.inteligent_current_question = question
                        st.session_state.inteligent_current_options = options
                        st.session_state.inteligent_correct_answer = correct_answer_key

                        display_question_text = f"**{question}**\n\n"
                        for key in sorted(options.keys()):
                            display_question_text += f"{key}) {options[key]}\n"

                        st.session_state.inteligent_game_messages.append({"role": "assistant", "content": display_question_text})
                        st.session_state.inteligent_game_needs_new_question = False
                        st.session_state.last_played_question_inteligent = None # Resetear para forzar el audio
                        st.rerun()
                    else:
                        st.session_state.inteligent_game_messages.append({"role": "assistant", "content": "¡Lo siento! Byte no pudo preparar la pregunta en el formato correcto. ¿Puedes pulsar '¡Vamos a jugar con Byte!' de nuevo?"})
                        st.session_state.inteligent_game_active = False
                        st.rerun()

                except Exception as e:
                    st.error(f"¡Oops! Byte no pudo hacer la pregunta. Error: {e}")
                    st.session_state.inteligent_game_messages.append({"role": "assistant", "content": "¡Lo siento! Byte tiene un pequeño problema para hacer preguntas ahora. ¡Pero puedes intentarlo de nuevo!"})
                    st.session_state.inteligent_game_active = False # Corregido typo
                    st.rerun()


        if st.session_state.inteligent_current_question is not None and not st.session_state.inteligent_awaiting_next_game_decision:
            # Audio de la pregunta
            if st.session_state.get('last_played_question_inteligent') != st.session_state.inteligent_current_question:
                try:
                    tts_text = f"{st.session_state.inteligent_current_question}. Opción A: {st.session_state.inteligent_current_options.get('A', '')}. Opción B: {st.session_state.inteligent_current_options.get('B', '')}. Opción C: {st.session_state.inteligent_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                    st.session_state.last_played_question_inteligent = st.session_state.inteligent_current_question # Guardar la pregunta reproducida
                except Exception as e:
                    st.warning(f"Error al generar o reproducir el audio de la pregunta de Byte: {e}")


            with st.form("byte_game_form", clear_on_submit=True):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_choice = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.inteligent_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.inteligent_current_options[x]}",
                        key="inteligent_answer_radio_buttons", # Corregido de "pinteligent" a "inteligent"
                        label_visibility="collapsed"
                    )

                submit_button = st.form_submit_button("Enviar Respuesta")

            if submit_button:
                st.session_state.inteligent_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_choice}) {st.session_state.inteligent_current_options[user_choice]}"})
                prev_streak = st.session_state.inteligent_correct_streak # Guardar el streak anterior

                # Lógica para actualizar el contador de respuestas correctas
                if user_choice == st.session_state.inteligent_correct_answer:
                    st.session_state.inteligent_correct_streak += 1
                else:
                    st.session_state.inteligent_correct_streak = 0 # Resetear si falla

                radio_placeholder.empty()

                # --- Lógica de subida de nivel y confeti ---
                if st.session_state.inteligent_correct_streak > 0 and \
                   st.session_state.inteligent_correct_streak % 3 == 0 and \
                   st.session_state.inteligent_correct_streak > prev_streak:
                    
                    if st.session_state.inteligent_correct_streak < 9: # Niveles Aprendiz, Constructor, Arquitecto
                        current_level_text = ""
                        if st.session_state.inteligent_correct_streak == 3:
                            current_level_text = "Constructor de Byte (¡como un ingeniero en prácticas!)"
                        elif st.session_state.inteligent_correct_streak == 6:
                            current_level_text = "Arquitecto de Byte (¡como un profesional diseñando sistemas!)"
                        
                        level_up_message = f"🎉 ¡Genial! ¡Has respondido {st.session_state.inteligent_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}**. ¡Las preguntas serán un poco más desafiantes ahora! ¡Estás aprendiendo super rápido! 🚀"
                        st.session_state.inteligent_game_messages.append({"role": "assistant", "content": level_up_message})
                        st.balloons()
                        # Generar audio para el mensaje de subida de nivel
                        try:
                            tts_level_up = gTTS(text=level_up_message, lang='es', slow=False)
                            audio_fp_level_up = io.BytesIO()
                            tts_level_up.write_to_fp(audio_fp_level_up)
                            audio_fp_level_up.seek(0)
                            st.audio(audio_fp_level_up, format="audio/mp3", start_time=0, autoplay=True)
                            time.sleep(2)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de subida de nivel (Byte): {e}")
                    elif st.session_state.inteligent_correct_streak >= 9:
                        medals_earned = (st.session_state.inteligent_correct_streak - 6) // 3 # (9-6)//3 = 1ª medalla, (12-6)//3 = 2ª medalla
                        medal_message = f"🏅 ¡FELICITACIONES, MAESTRO/A DE BYTE! ¡Has ganado tu {medals_earned}ª Medalla de Programación Predictiva! ¡Tu conocimiento es asombroso y digno de un verdadero EXPERTO en modelos predictivos! ¡Sigue así! 🌟"
                        st.session_state.inteligent_game_messages.append({"role": "assistant", "content": medal_message})
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
                            st.warning(f"No se pudo reproducir el audio de medalla (Byte): {e}")
                        
                        # Mensaje de "subida de nivel" al pasar a Maestro de Byte
                        if prev_streak < 9:
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Maestro de Byte (Experto en Modelos Predictivos)**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros científicos de datos! ¡Adelante!"
                            st.session_state.inteligent_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                audio_fp_level_up_champion = io.BytesIO()
                                tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                                audio_fp_level_up_champion.seek(0)
                                st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0, autoplay=True)
                                time.sleep(2)
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón (Byte): {e}")


                feedback_prompt = f"""
                El usuario ha respondido '{user_choice}'. La pregunta era: '{st.session_state.inteligent_current_question}'.
                La respuesta correcta era '{st.session_state.inteligent_correct_answer}'.
                Da feedback como Byte.
                Si es CORRECTO, el mensaje es "¡Genial! ¡Lo has clavado!" o similar.
                Si es INCORRECTO, el mensaje es "¡Uhm, casi!" o similar.
                Luego, una explicación sencilla para el usuario.
                Finalmente, pregunta: "¿Quieres seguir jugando?".
                **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                """

                with st.spinner("Byte está revisando tu respuesta..."):
                    try:
                        # Usamos el prompt del sistema actualizado con el nivel de dificultad aquí también
                        feedback_messages_for_api = [{"role": "system", "content": inteligent_game_system_prompt}]
                        feedback_messages_for_api.append({"role": "user", "content": feedback_prompt})

                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=feedback_messages_for_api,
                            temperature=0.8,
                            max_tokens=300
                        )
                        raw_byte_feedback_text = feedback_response.choices[0].message.content

                        feedback_msg, explanation_msg, next_question_prompt = parse_inteligent_feedback_response(raw_byte_feedback_text)

                        st.session_state.inteligent_game_messages.append({"role": "assistant", "content": feedback_msg})
                        st.session_state.inteligent_game_messages.append({"role": "assistant", "content": explanation_msg})
                        st.session_state.inteligent_game_messages.append({"role": "assistant", "content": next_question_prompt})

                        try:
                            tts = gTTS(text=f"{feedback_msg}. {explanation_msg}. {next_question_prompt}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback (Byte): {e}")


                        st.session_state.inteligent_current_question = None
                        st.session_state.inteligent_current_options = {}
                        st.session_state.inteligent_correct_answer = None
                        st.session_state.inteligent_game_needs_new_question = False
                        st.session_state.inteligent_awaiting_next_game_decision = True

                        st.rerun()

                    except Exception as e:
                        st.error(f"Ups, Byte no pudo procesar tu respuesta. Error: {e}")
                        st.session_state.inteligent_game_messages.append({"role": "assistant", "content": "Lo siento, Byte tiene un pequeño problema técnico ahora mismo. ¡Pero me encantaría ver tu respuesta!"})


        if st.session_state.inteligent_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Sí, quiero jugar más preguntas", key="play_more_questions_inteligent"):
                    st.session_state.inteligent_game_needs_new_question = True
                    st.session_state.inteligent_awaiting_next_game_decision = False
                    st.session_state.inteligent_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío!"})
                    st.rerun()
            with col2:
                if st.button("👎 No, ya no quiero jugar más", key="stop_playing_inteligent"):
                    st.session_state.inteligent_game_active = False
                    st.session_state.inteligent_awaiting_next_game_decision = False
                    st.session_state.inteligent_game_messages.append({"role": "assistant", "content": "¡De acuerdo! ¡Gracias por jugar conmigo! Espero que hayas aprendido mucho sobre los Modelos Predictivos. ¡Nos vemos pronto!"})
                    st.rerun()
else:
    st.info("Para usar la sección de preguntas de Byte, necesitas configurar tu clave de API de OpenAI en `secrets.toml`.")