# pages/05_Que_es_Clasificacion.py

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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="¿Qué es la Clasificación?",
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
    except Exception as e:
        st.error(f"Error inesperado al cargar el archivo Lottie '{filepath}': {e}. Asegúrate de que el archivo no esté corrupto y sea un JSON válido.")
        return None

# --- Rutas a Lottie ---
LOTTIE_CLASSIFY_PATH = os.path.join("assets", "lottie_animations", "Categorization.json")

# --- Configuración de la API de OpenAI ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai_api_key = None
    st.error("Error: La clave de API de OpenAI no está configurada en `secrets.toml`.")
    st.info("Para configurarla, crea un archivo `.streamlit/secrets.toml` en la raíz de tu proyecto y añade: `OPENAI_API_KEY = 'tu_clave_aqui'`")

client = OpenAI(api_key=openai_api_key) if openai_api_key else None


st.subheader("¡Organiza el futuro en categorías!")

st.write("---")

# Sección 1: ¿Qué es la Clasificación?
st.header("¿Qué es la Clasificación?")
st.markdown("""
¿Alguna vez has separado tus juguetes por tipo, o las frutas por color?
¡Eso es **clasificar**!

En Inteligencia Artificial, la **Clasificación** es cuando una máquina
aprende a poner cosas en diferentes **grupos o categorías** basándose
en lo que ya ha aprendido de otros ejemplos.

No predice un número (como la regresión), ¡sino un **nombre o una etiqueta**!
""")

# Pequeña animación para la introducción
col_intro_left, col_intro_right = st.columns([1, 1])
with col_intro_right:
    lottie_classify = load_lottiefile(LOTTIE_CLASSIFY_PATH)
    if lottie_classify:
        st_lottie(lottie_classify, height=200, width=200, key="classify_intro")
    else:
        st.info("Consejo: Asegúrate de que 'Categorization.json' (o una mejor) esté en 'assets/lottie_animations/' para esta animación.")

st.write("---")

# Sección 2: ¿Cómo Clasifica la IA? (Visualización Interactiva)
st.header("¿Cómo Clasifica la IA?")
st.markdown("""
Imagina que la IA está aprendiendo a distinguir entre manzanas (🍎) y plátanos (🍌)
basándose en su color y su forma.

¡Vamos a ver cómo una IA puede aprender a clasificarlos!
""")

# --- Visualización interactiva con Matplotlib para clasificación ---
st.subheader("Clasificando Frutas: ¡Ayuda a la IA a aprender las diferencias!")

st.markdown("""
¡Define el "Color" y la "Forma" de tu fruta, elige si es una manzana o un plátano, y añádela al gráfico!
La IA intentará encontrar una línea (frontera) que los separe a medida que añades más ejemplos.
""")

# Inicializar los datos del juego en session_state
if 'fruit_data' not in st.session_state:
    st.session_state.fruit_data = [] # Lista de diccionarios: [{'x': x, 'y': y, 'class': 'apple'/'banana'}]
if 'current_fruit_class' not in st.session_state:
    st.session_state.current_fruit_class = 'apple' # Por defecto, añadir manzanas

# Alternar entre añadir manzanas y plátanos
col_fruit_toggle_left, col_fruit_toggle_right = st.columns([1, 2])
with col_fruit_toggle_left:
    if st.button("Manzana 🍎", key="set_apple_class"):
        st.session_state.current_fruit_class = 'apple'
    if st.button("Plátano 🍌", key="set_banana_class"):
        st.session_state.current_fruit_class = 'banana'
with col_fruit_toggle_right:
    st.info(f"Ahora añadirás: **{st.session_state.current_fruit_class.capitalize()}**")

# Crear el gráfico para la clasificación
fig_cls, ax_cls = plt.subplots(figsize=(9, 7))
ax_cls.set_xlabel("Color (e.g., 0=Verde, 1=Rojo, 2=Amarillo)")
ax_cls.set_ylabel("Forma (e.g., 0=Redondo, 1=Alargado)")
ax_cls.set_title("Clasificación de Frutas (Manzana vs. Plátano)")
ax_cls.set_xlim(-0.5, 2.5)
ax_cls.set_ylim(-0.5, 1.5)
ax_cls.set_xticks([0, 1, 2])
ax_cls.set_xticklabels(['Verde', 'Rojo', 'Amarillo'])
ax_cls.set_yticks([0, 1])
ax_cls.set_yticklabels(['Redondo', 'Alargado'])
ax_cls.grid(True, linestyle='--', alpha=0.6)


if not st.session_state.fruit_data:
    ax_cls.text((ax_cls.get_xlim()[0] + ax_cls.get_xlim()[1]) / 2,
                (ax_cls.get_ylim()[0] + ax_cls.get_ylim()[1]) / 2,
                "¡Usa los sliders de abajo para añadir frutas!",
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='gray', alpha=0.6)

# Preparar los datos para el modelo
X_cls = np.array([[d['x'], d['y']] for d in st.session_state.fruit_data])
y_cls = np.array([1 if d['class'] == 'apple' else 0 for d in st.session_state.fruit_data]) 
# 1 para manzana, 0 para plátano

# Plotear los puntos existentes
colors_map = {'apple': 'red', 'banana': 'gold'}
markers_map = {'apple': 'o', 'banana': 's'}
for d in st.session_state.fruit_data:
    ax_cls.scatter(d['x'], d['y'], color=colors_map[d['class']],
                   marker=markers_map[d['class']], s=200, edgecolor='black', zorder=3,
                   label=d['class'].capitalize() if d['class'] not in [item.get_label() for item in ax_cls.collections] else "") # Evitar duplicar etiquetas

model_cls = None
# Entrenar el modelo de clasificación si hay suficientes puntos y al menos una de cada clase
if len(np.unique(y_cls)) > 1 and len(X_cls) >= 2:
    try:
        # pipeline para escalar y luego entrenar SVC
        model_cls = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=42, C=1000)) # C alto
        model_cls.fit(X_cls, y_cls)

        # Dibujar la frontera de decisión
        xlim = ax_cls.get_xlim()
        ylim = ax_cls.get_ylim()

        # Crear una malla para dibujar la frontera
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                             np.linspace(ylim[0], ylim[1], 100))
        Z = model_cls.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax_cls.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu, levels=[-0.5, 0.5, 1.5])
        ax_cls.contour(xx, yy, Z, colors='k', levels=[0.5], alpha=0.7, linestyles=['--'])
        
        st.markdown("La línea punteada es la **frontera de decisión** que la IA ha aprendido.")

    except Exception as e:
        st.info(f"Necesitas añadir más puntos de **ambas** frutas para que la IA pueda dibujar una línea de separación. (Error: {e})")
        model_cls = None # Resetear modelo si falla el entrenamiento

ax_cls.legend()
# Mostrar el gráfico
st.pyplot(fig_cls, use_container_width=True)

st.markdown("---")
st.subheader("¡Añade tus propias frutas de ejemplo y ve cómo la IA aprende a separarlas!")

col_add_fruit1, col_add_fruit2, col_add_fruit3 = st.columns(3)

with col_add_fruit1:
    fruit_color_val = st.slider("Define el **Color** (0=Verde, 1=Rojo, 2=Amarillo):", min_value=0.0, max_value=2.0, value=1.0 if st.session_state.current_fruit_class == 'apple' else 2.0, step=0.1, key="fruit_color_cls")
with col_add_fruit2:
    fruit_shape_val = st.slider("Define la **Forma** (0=Redondo, 1=Alargado):", min_value=0.0, max_value=1.0, value=0.0 if st.session_state.current_fruit_class == 'apple' else 1.0, step=0.1, key="fruit_shape_cls")
with col_add_fruit3:
    st.markdown(" ")
    st.markdown(" ")
    add_fruit_button = st.button(f"➕ Añadir {st.session_state.current_fruit_class.capitalize()} al gráfico", key="add_cls_point")
    if add_fruit_button:
        st.session_state.fruit_data.append({'x': fruit_color_val, 'y': fruit_shape_val, 'class': st.session_state.current_fruit_class})
        st.rerun()

if st.button("Borrar todas las frutas del gráfico", key="clear_cls_points"):
    st.session_state.fruit_data = []
    st.rerun()

if model_cls:
    st.markdown("---")
    st.subheader("¡Haz una predicción con la IA!")
    st.markdown("Ahora que la IA ha aprendido, ¡dale las características de una nueva fruta para ver cómo la clasifica!")
    predict_fruit_color = st.slider("Color de la fruta a predecir:", min_value=0.0, max_value=2.0, value=1.5, step=0.1, key="predict_fruit_color_slider")
    predict_fruit_shape = st.slider("Forma de la fruta a predecir:", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="predict_fruit_shape_slider")
    
    # Realizar la predicción
    predicted_class_num = model_cls.predict(np.array([[predict_fruit_color, predict_fruit_shape]]))[0]
    predicted_class_name = 'Manzana 🍎' if predicted_class_num == 1 else 'Plátano 🍌'
    
    st.markdown(f"La IA predice que esta fruta es: **{predicted_class_name}**.")
    
    # Añadir el punto de predicción al gráfico
    ax_cls.plot(predict_fruit_color, predict_fruit_shape, 'X', color='blue', markersize=15, markeredgecolor='black', label='Nueva Fruta', zorder=4)
    ax_cls.legend()
    st.pyplot(fig_cls)


st.markdown("""
¿Ves cómo la IA intenta dibujar una línea que separe las manzanas de los plátanos?
A esa línea se le llama **frontera de decisión**. Una vez que la aprende, puede
**clasificar** nuevas frutas, ¡incluso si nunca las ha visto antes!

Esto es útil para clasificar correos electrónicos como 'spam' o 'no spam',
fotos como 'perro' o 'gato', o diagnosticar enfermedades como 'presente' o 'ausente'.
""")

st.write("---")

# Explicación sencilla del modelo
st.header("¿Cómo sabe la IA si es una manzana o un plátano?")
st.markdown("""
¡Increíble, ¿verdad?! Has visto cómo la IA aprende a separar las manzanas de los plátanos. Pero, ¿cómo lo hace?

Imagina que la IA es como un detective muy listo.

1.  **Tus Ejemplos son Pistas:** Cada vez que le dices "esto es una manzana con este color y esta forma" o "esto es un plátano con este color y esta forma", ¡le estás dando pistas muy importantes! La IA guarda todas esas pistas en su cerebro.

2.  **Aprende una "Regla Secreta":** Con todas esas pistas, la IA intenta encontrar una "regla secreta" para separar las manzanas de los plátanos. En nuestro juego, esa regla es la **línea punteada** que aparece en el gráfico. Esa línea es como la frontera mágica que dice: "A este lado, ¡son todas manzanas! Y a este otro, ¡todos plátanos!".

3.  **Predice con la Regla:** Cuando le das una nueva fruta y le preguntas "¿qué es esto?", la IA no la adivina. ¡Usa su "regla secreta" (esa línea que aprendió)! Mira dónde cae la nueva fruta con su color y forma, y dependiendo de qué lado de la línea caiga, te dice si cree que es una manzana o un plátano.

¡Así, la IA no necesita que le digas una y otra vez qué es cada fruta! Una vez que aprende su regla, puede clasificar millones de frutas nuevas por sí misma. ¡Es como magia, pero es ciencia y matemáticas! Y a ese "cerebro" que aprende las reglas lo llamamos **modelo de clasificación**.
""")
st.write("---")

# --- Sección de Chatbot de Juego con Etiquetín para "Qué es la Clasificación" ---
st.header("¡Juega y Aprende con Etiquetín sobre la Clasificación!")
st.markdown("¡Hola! Soy Etiquetín, la robot que ama organizar y etiquetar cosas. ¿Listo para aprender a clasificar con la IA?")

if client:
    # Inicializa el estado del juego y los mensajes del chat
    if "cls_game_active" not in st.session_state:
        st.session_state.cls_game_active = False
    if "cls_game_messages" not in st.session_state:
        st.session_state.cls_game_messages = []
    if "cls_current_question" not in st.session_state:
        st.session_state.cls_current_question = None
    if "cls_current_options" not in st.session_state:
        st.session_state.cls_current_options = {}
    if "cls_correct_answer" not in st.session_state:
        st.session_state.cls_correct_answer = None
    if "cls_awaiting_next_game_decision" not in st.session_state:
        st.session_state.cls_awaiting_next_game_decision = False
    if "cls_game_needs_new_question" not in st.session_state:
        st.session_state.cls_game_needs_new_question = False
    if "cls_correct_streak" not in st.session_state:
        st.session_state.cls_correct_streak = 0
    if "last_played_question_etiquetin" not in st.session_state:
        st.session_state.last_played_question_etiquetin = None


    # System prompt para el juego de preguntas de Etiquetín
    etiquetin_game_system_prompt = f"""
    Eres un **experto consumado en Machine Learning y Reconocimiento de Patrones**, con una especialización profunda en los **Algoritmos de Clasificación**. Comprendes a fondo sus fundamentos teóricos, métricas de rendimiento, aplicaciones prácticas y desafíos. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de la Clasificación mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Clasificación exitosa! Tu modelo discernió correctamente." o "Esa predicción de clase fue incorrecta. Revisemos el criterio."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "La clasificación es una tarea de aprendizaje automático donde el modelo asigna elementos a una de varias categorías predefinidas..."]
    [Pregunta para continuar, ej: "¿Listo para optimizar tus clasificadores?" o "¿Quieres explorar las complejidades del balance de clases?"]

    **Reglas adicionales para el Experto en Clasificación:**
    * **Enfoque Riguroso en Clasificación:** Todas tus preguntas y explicaciones deben girar en torno a los Algoritmos de Clasificación. Cubre sus fundamentos (problemas binarios vs. multiclase), algoritmos clave (Regresión Logística, SVM, Árboles de Decisión, Random Forests, Naive Bayes, K-NN, Redes Neuronales básicas), métricas de evaluación (Matriz de Confusión, Precisión, Recall, F1-score, AUC-ROC, Curva PR), balance de clases, sobreajuste/subajuste, validación cruzada y preprocesamiento de datos específico para clasificación.
    * **¡VARIEDAD, VARIEDAD, VARIEDAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de Clasificación que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General:** ¿Qué es la clasificación? ¿Para qué sirve? Diferencia entre clasificación y regresión.
        * **Tipos de Problemas de Clasificación:** Binaria, multiclase, multietiqueta.
        * **Algoritmos Fundamentales:**
            * **Regresión Logística:** Función Sigmoide, interpretación de probabilidades.
            * **Máquinas de Vectores de Soporte (SVM):** Hiperplano, margen, kernel (intuitivo).
            * **Árboles de Decisión / Random Forests / Boosting:** Criterios de división (Gini, Entropía), ensembles.
            * **Naive Bayes:** Teorema de Bayes (intuitivo), independencia condicional.
            * **K-Nearest Neighbors (K-NN):** Concepto de cercanía, elección de K.
            * **Redes Neuronales (básicas):** Capas, activación, backpropagation (concepto).
        * **Preprocesamiento de Datos para Clasificación:** Escalado, codificación de categóricas, manejo de datos desbalanceados (oversampling, undersampling).
        * **Evaluación del Modelo (Crucial):**
            * **Matriz de Confusión:** Verdaderos/Falsos Positivos/Negativos.
            * **Métricas Derivadas:** Precisión (Precision), Exhaustividad (Recall), Puntuación F1 (F1-score).
            * **Curvas ROC y AUC:** Interpretación del rendimiento del clasificador.
            * **Puntos de Corte (Thresholds):** Optimización para diferentes objetivos.
        * **Validación y Generalización:** Validación cruzada, sobreajuste y subajuste.
        * **Sesgo y Varianza:** Trade-off en clasificadores.
        * **Selección de Características:** Impacto en el rendimiento del clasificador.
        * **Interpretación de Modelos Clasificadores:** Importancia de características, explicabilidad.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.cls_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Iniciador en Clasificación – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de categorizar elementos y ejemplos sencillos de problemas de clasificación.
            * *Tono:* "Estás empezando a organizar el mundo en categorías con el poder del Machine Learning."
        * **Nivel 2 (Analista de Clasificadores – 3-5 respuestas correctas):** Tono más técnico. Introduce conceptos como clases, características, y algoritmos básicos como Regresión Logística o Árboles de Decisión de forma intuitiva. Preguntas sobre la aplicación inicial de estos modelos.
            * *Tono:* "Tu habilidad para distinguir patrones está en desarrollo, construyendo las bases de clasificadores efectivos."
        * **Nivel 3 (Ingeniero de Clasificación – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en las métricas de evaluación (matriz de confusión, precisión, recall, F1-score), el manejo de datos desbalanceados, y la lógica detrás de algoritmos más avanzados (SVM, Random Forests). Preguntas que requieren una comprensión de la evaluación y optimización de clasificadores.
            * *Tono:* "Tu capacidad para diseñar y evaluar clasificadores de alto rendimiento es fundamental para la toma de decisiones basada en datos."
        * **Nivel Maestro (Científico de Datos de Clasificación – 9+ respuestas correctas):** Tono de **especialista en Machine Learning y optimización de clasificadores**. Preguntas sobre la interpretación avanzada de curvas ROC/PR, la elección del algoritmo óptimo para problemas con requisitos específicos (ej. detección de fraudes), el impacto del balance de clases en el rendimiento, o el ajuste de hiperparámetros para maximizar métricas clave. Se esperan respuestas que demuestren una comprensión teórica y práctica robusta, incluyendo sus limitaciones y sesgos.
            * *Tono:* "Tu maestría en el desarrollo y despliegue de soluciones de clasificación te posiciona como un referente en la extracción de inteligencia de los datos."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Clasificar correos electrónicos como "spam" o "no spam".
        * **Nivel 2:** Determinar si un cliente es propenso a darse de baja de un servicio, o clasificar frutas por su tipo.
        * **Nivel 3:** Construir un clasificador para diagnosticar una enfermedad (presente/ausente) a partir de síntomas, optimizando el recall para minimizar falsos negativos.
        * **Nivel Maestro:** Desarrollar un sistema de clasificación multiclase para detectar diferentes tipos de anomalías en una red, considerando el desbalance extremo de clases y la interpretabilidad del modelo para los operadores.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre CLASIFICACIÓN, y asegúrate de que no se parezca a las anteriores.
    """

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_cls_question_response(raw_text):
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
    def parse_cls_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"

    # --- Funciones para subir de nivel directamente ---
    def set_etiquetin_level(target_streak, level_name):
        st.session_state.cls_correct_streak = target_streak
        st.session_state.cls_game_active = True
        st.session_state.cls_game_messages = []
        st.session_state.cls_current_question = None
        st.session_state.cls_current_options = {}
        st.session_state.cls_correct_answer = None
        st.session_state.cls_game_needs_new_question = True
        st.session_state.cls_awaiting_next_game_decision = False
        st.session_state.cls_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}** de Etiquetín! Prepárate para preguntas más desafiantes. ¡Aquí va tu primera!"})
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons_etiquetin, col_level_up_buttons_etiquetin = st.columns([1, 2])

    with col_game_buttons_etiquetin:
        if st.button("¡Vamos a jugar con Etiquetín!", key="start_etiquetin_game_button"):
            st.session_state.cls_game_active = True
            st.session_state.cls_game_messages = []
            st.session_state.cls_current_question = None
            st.session_state.cls_current_options = {}
            st.session_state.cls_correct_answer = None
            st.session_state.cls_game_needs_new_question = True
            st.session_state.cls_awaiting_next_game_decision = False
            st.session_state.cls_correct_streak = 0
            st.session_state.last_played_question_etiquetin = None
            st.rerun()
    
    with col_level_up_buttons_etiquetin:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un experto en clasificar? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_etiquetin, col_lvl2_etiquetin, col_lvl3_etiquetin = st.columns(3) # Tres columnas para los botones de nivel
        with col_lvl1_etiquetin:
            if st.button("Subir a Nivel Medio (Etiquetín)", key="level_up_medium_etiquetin"):
                set_etiquetin_level(3, "Medio") # 3 respuestas correctas para Nivel Medio
        with col_lvl2_etiquetin:
            if st.button("Subir a Nivel Avanzado (Etiquetín)", key="level_up_advanced_etiquetin"):
                set_etiquetin_level(6, "Avanzado") # 6 respuestas correctas para Nivel Avanzado
        with col_lvl3_etiquetin:
            if st.button("👑 ¡Maestro Clasificador! (Etiquetín)", key="level_up_champion_etiquetin"):
                set_etiquetin_level(9, "Campeón") # 9 respuestas correctas para Nivel Campeón


    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.cls_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica del juego del chatbot si está activo
    if st.session_state.cls_game_active:
        if st.session_state.cls_current_question is None and st.session_state.cls_game_needs_new_question and not st.session_state.cls_awaiting_next_game_decision:
            with st.spinner("Etiquetín está preparando una pregunta..."):
                try:
                    # Incluimos el prompt del sistema actualizado con el nivel de dificultad
                    game_messages_for_api = [{"role": "system", "content": etiquetin_game_system_prompt}]
                    # Limita el historial para evitar prompts demasiado largos, tomando las últimas interacciones relevantes
                    if st.session_state.cls_game_messages:
                        last_message = st.session_state.cls_game_messages[-1]
                        if last_message["role"] == "user":
                            game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {last_message['content']}"})
                        elif last_message["role"] == "assistant":
                            # Si el último mensaje fue del asistente (feedback), lo añadimos para que sepa dónde se quedó
                            game_messages_for_api.append({"role": "assistant", "content": last_message['content']})

                    game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre QUÉ ES LA CLASIFICACIÓN siguiendo el formato exacto. ¡Recuerda, la pregunta debe ser muy VARIADA y CREATIVA, y no se debe parecer a las anteriores!"})

                    game_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=game_messages_for_api,
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_cls_question_text = game_response.choices[0].message.content
                    question, options, correct_answer_key = parse_cls_question_response(raw_cls_question_text)

                    if question:
                        st.session_state.cls_current_question = question
                        st.session_state.cls_current_options = options
                        st.session_state.cls_correct_answer = correct_answer_key

                        display_question_text = f"**Nivel {int(st.session_state.cls_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.cls_correct_streak}**\n\n**Pregunta de Etiquetín:** {question}\n\n"
                        for key in sorted(options.keys()):
                            display_question_text += f"{key}) {options[key]}\n"

                        st.session_state.cls_game_messages.append({"role": "assistant", "content": display_question_text})
                        st.session_state.cls_game_needs_new_question = False
                        st.rerun()
                    else:
                        st.session_state.cls_game_messages.append({"role": "assistant", "content": "¡Lo siento! Etiquetín no pudo preparar la pregunta en el formato correcto. ¿Puedes pulsar 'VAMOS A JUGAR' de nuevo?"})
                        st.session_state.cls_game_active = False
                        st.rerun()

                except Exception as e:
                    st.error(f"¡Oops! Etiquetín no pudo hacer la pregunta. Error: {e}")
                    st.session_state.cls_game_messages.append({"role": "assistant", "content": "¡Lo siento! Etiquetín tiene un pequeño problema para hacer preguntas ahora. ¿Puedes intentarlo de nuevo?"})
                    st.session_state.cls_game_active = False
                    st.rerun()


        if st.session_state.cls_current_question is not None and not st.session_state.cls_awaiting_next_game_decision:
            # Audio de la pregunta
            if st.session_state.get('last_played_question_etiquetin') != st.session_state.cls_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.cls_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.cls_correct_streak}. Pregunta de Etiquetín: {st.session_state.cls_current_question}. Opción A: {st.session_state.cls_current_options.get('A', '')}. Opción B: {st.session_state.cls_current_options.get('B', '')}. Opción C: {st.session_state.cls_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                    st.session_state.last_played_question_etiquetin = st.session_state.cls_current_question
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")


            with st.form("etiquetin_game_form", clear_on_submit=True):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_choice = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.cls_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.cls_current_options[x]}",
                        key="cls_answer_radio_buttons",
                        label_visibility="collapsed"
                    )

                submit_button = st.form_submit_button("Enviar Respuesta")

            if submit_button:
                st.session_state.cls_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_choice}) {st.session_state.cls_current_options[user_choice]}"})
                prev_streak = st.session_state.cls_correct_streak

                # Lógica para actualizar el contador de respuestas correctas
                if user_choice == st.session_state.cls_correct_answer:
                    st.session_state.cls_correct_streak += 1
                else:
                    st.session_state.cls_correct_streak = 0 # Resetear si falla

                radio_placeholder.empty()

                # --- Lógica de subida de nivel y confeti ---
                if st.session_state.cls_correct_streak > 0 and \
                   st.session_state.cls_correct_streak % 3 == 0 and \
                   st.session_state.cls_correct_streak > prev_streak:
                    
                    if st.session_state.cls_correct_streak < 9: # Niveles Básico, Medio, Avanzado
                        current_level_text = ""
                        if st.session_state.cls_correct_streak == 3:
                            current_level_text = "Medio (como un adolescente que ya entiende de organizar)"
                        elif st.session_state.cls_correct_streak == 6:
                            current_level_text = "Avanzado (como un trabajador de Data Science senior)"
                        
                        level_up_message = f"🎉 ¡Increíble! ¡Has respondido {st.session_state.cls_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de Clasificación. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a organizador/a de categorías! 🚀"
                        st.session_state.cls_game_messages.append({"role": "assistant", "content": level_up_message})
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
                            st.warning(f"No se pudo reproducir el audio de subida de nivel: {e}")
                    elif st.session_state.cls_correct_streak >= 9:
                        medals_earned = (st.session_state.cls_correct_streak - 6) // 3 
                        medal_message = f"🏅 ¡FELICITACIONES, MAESTRO CLASIFICADOR! ¡Has ganado tu {medals_earned}ª Medalla de Clasificación! ¡Tu habilidad para etiquetar y organizar es asombrosa y digna de un verdadero EXPERTO en Clasificación! ¡Sigue así! 🌟"
                        st.session_state.cls_game_messages.append({"role": "assistant", "content": medal_message})
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
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Campeón (Maestro Clasificador)**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros científicos de datos que organizan el futuro con sus etiquetas! ¡Adelante!"
                            st.session_state.cls_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                audio_fp_level_up_champion = io.BytesIO()
                                tts_level_up_champion.write_to_fp(audio_fp_level_up_champion) 
                                audio_fp_level_up_champion.seek(0)
                                st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0, autoplay=True)
                                time.sleep(2)
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón: {e}")


                # Generar feedback de Etiquetín
                with st.spinner("Etiquetín está revisando tu respuesta..."):
                    try:
                        feedback_prompt = f"""
                        El usuario respondió '{user_choice}'. La pregunta era: '{st.session_state.cls_current_question}'.
                        La respuesta correcta era '{st.session_state.cls_correct_answer}'.
                        Da feedback como Etiquetín.
                        Si es CORRECTO, el mensaje es "¡Clasificación perfecta! ¡Lo has etiquetado bien!" o similar.
                        Si es INCORRECTO, el mensaje es "¡Revisa tu categoría!" o similar.
                        Luego, una explicación sencilla para niños y adolescentes.
                        Finalmente, pregunta: "¿Quieres seguir clasificando cosas?".
                        **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                        """
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": etiquetin_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=300
                        )
                        raw_etiquetin_feedback_text = feedback_response.choices[0].message.content

                        feedback_msg, explanation_msg, next_question_prompt = parse_cls_feedback_response(raw_etiquetin_feedback_text)

                        st.session_state.cls_game_messages.append({"role": "assistant", "content": feedback_msg})
                        st.session_state.cls_game_messages.append({"role": "assistant", "content": explanation_msg})
                        st.session_state.cls_game_messages.append({"role": "assistant", "content": next_question_prompt})

                        try:
                            tts = gTTS(text=f"{feedback_msg}. {explanation_msg}. {next_question_prompt}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")


                        st.session_state.cls_current_question = None
                        st.session_state.cls_current_options = {}
                        st.session_state.cls_correct_answer = None
                        st.session_state.cls_game_needs_new_question = False
                        st.session_state.cls_awaiting_next_game_decision = True

                        st.rerun()

                    except Exception as e:
                        st.error(f"Ups, Etiquetín no pudo procesar tu respuesta. Error: {e}")
                        st.session_state.cls_game_messages.append({"role": "assistant", "content": "Lo siento, Etiquetín tiene un pequeño problema técnico ahora mismo. ¡Pero me encantaría ver tu respuesta!"})


        if st.session_state.cls_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Sí, quiero jugar más preguntas", key="play_more_questions_cls"):
                    st.session_state.cls_game_needs_new_question = True
                    st.session_state.cls_awaiting_next_game_decision = False
                    st.session_state.cls_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío!"})
                    st.rerun()
            with col2:
                if st.button("👎 No, ya no quiero jugar más", key="stop_playing_cls"):
                    st.session_state.cls_game_active = False
                    st.session_state.cls_awaiting_next_game_decision = False
                    st.session_state.cls_game_messages.append({"role": "assistant", "content": "¡De acuerdo! ¡Gracias por jugar conmigo! Espero que hayas aprendido mucho sobre la Clasificación. ¡Nos vemos pronto!"})
                    st.rerun()

else:
    st.info("Para usar la sección de preguntas de Etiquetín, necesitas configurar tu clave de API de OpenAI en `secrets.toml`.")