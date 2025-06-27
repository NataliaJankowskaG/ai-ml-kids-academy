# Regresión Logística

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
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(
    page_title="¿Qué son los Modelos Predictivos?",
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
LOTTIE_PREDICT_PATH = os.path.join("assets", "lottie_animations", "Predict.json")

# --- Configuración de la API de OpenAI ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai_api_key = None

client = OpenAI(api_key=openai_api_key) if openai_api_key else None


st.subheader("¡Descubre cómo la IA adivina el futuro!")

st.write("---")

# Sección 1: ¿Qué son los Modelos Predictivos?
st.header("¿Qué son los Modelos Predictivos?")
st.markdown("""
Imagina que eres un detective y quieres adivinar qué pasará después.
¿Va a llover mañana? ¿Qué número saldrá en los dados?

Los **Modelos Predictivos** son como tus herramientas de detective superpoderosas.
Son programas de ordenador que aprenden de cosas que ya han pasado (¡los datos!)
para intentar **adivinar qué sucederá en el futuro**.

No siempre aciertan, ¡pero son muy buenos intentándolo!
""")

# Pequeña animación para la introducción
col_intro_left, col_intro_right = st.columns([1, 1])
with col_intro_right:
    lottie_predict = load_lottiefile(LOTTIE_PREDICT_PATH)
    if lottie_predict:
        st_lottie(lottie_predict, height=200, width=200, key="predict_models_intro")
    else:
        st.info("Consejo: Asegúrate de que 'Math.json' (o una mejor) esté en 'assets/lottie_animations/' para esta animación.")

st.write("---")

# --- NUEVA SECCIÓN 2: El Predictor del Clima ---
st.header("¡Juega con el Predictor del Clima: ¿Lloverá Mañana?!")
st.markdown("""
¡Vamos a construir un **Predictor del Clima**! Imagina que tenemos un registro de días pasados
con su **temperatura**, **humedad** y si **llovió o no**.

Nuestro Predictor aprenderá de estos días y luego intentará adivinar si lloverá mañana
basándose en la temperatura y la humedad que le des.
""")

# Inicializando datos en session_state
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = [] # Lista de diccionarios: {'temp': temp, 'humidity': humidity, 'rain': True/False}
if 'last_weather_prediction' not in st.session_state:
    st.session_state.last_weather_prediction = None
if 'trained_model' not in st.session_state: # Para almacenar el modelo entrenado
    st.session_state.trained_model = None

# Crear el gráfico para la clasificación del clima
fig_weather, ax_weather = plt.subplots(figsize=(9, 7))
ax_weather.set_xlabel("Temperatura (°C)")
ax_weather.set_ylabel("Humedad (%)")
ax_weather.set_title("Predictor del Clima: ¿Lluvia o Sol?")
ax_weather.set_xlim(0, 40) # Temperatura de 0 a 40 grados
ax_weather.set_ylim(0, 100) # Humedad de 0 a 100%
ax_weather.grid(True, linestyle='--', alpha=0.6)

# Añadir un mensaje si no hay datos
if not st.session_state.weather_data:
    ax_weather.text((ax_weather.get_xlim()[0] + ax_weather.get_xlim()[1]) / 2,
                    (ax_weather.get_ylim()[0] + ax_weather.get_ylim()[1]) / 2,
                    "¡Añade ejemplos de días pasados para que el Predictor aprenda!",
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14, color='gray', alpha=0.6)

plotted_labels = set()
for day_data in st.session_state.weather_data:
    color_map = {True: 'blue', False: 'gold'}
    marker_map = {True: 's', False: 'o'}
    label_text = 'Llovió' if day_data['rain'] else 'No Llovió'
    
    if label_text not in plotted_labels:
        ax_weather.scatter(day_data['temp'], day_data['humidity'],
                           color=color_map[day_data['rain']],
                           marker=marker_map[day_data['rain']],
                           s=150, label=label_text, zorder=3, alpha=0.8)
        plotted_labels.add(label_text)
    else:
        ax_weather.scatter(day_data['temp'], day_data['humidity'],
                           color=color_map[day_data['rain']],
                           marker=marker_map[day_data['rain']],
                           s=150, zorder=3, alpha=0.8)

model_trained_weather = False

# Lógica Logistic Regression - se requieren al menos 2 muestras para 2 clases diferentes, y preferiblemente más para un buen entrenamiento.
if len(st.session_state.weather_data) >= 5: # Mínimo de 5 puntos para entrenar
    X = np.array([[d['temp'], d['humidity']] for d in st.session_state.weather_data])
    y = np.array([1 if d['rain'] else 0 for d in st.session_state.weather_data]) # 1 para lluvia, 0 para no lluvia

    # Asegurarse de que haya al menos dos clases presentes para el entrenamiento
    if len(np.unique(y)) > 1:
        st.session_state.trained_model = LogisticRegression(random_state=42)
        try:
            st.session_state.trained_model.fit(X, y)
            model_trained_weather = True
            st.markdown("**¡El Predictor ha aprendido!** Ahora puede usar la temperatura y la humedad para intentar adivinar si lloverá. ¡Mira la línea que ha encontrado para separar los días!")

            # Visualizar la frontera de decisión del modelo
            x_min, x_max = ax_weather.get_xlim()
            y_min, y_max = ax_weather.get_ylim()

            disp = DecisionBoundaryDisplay.from_estimator(
                st.session_state.trained_model,
                X,
                response_method="predict",
                cmap=plt.cm.RdBu,
                alpha=0.3,
                ax=ax_weather,
            )
           
            handles, labels = ax_weather.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_weather.legend(by_label.values(), by_label.keys())

        except Exception as e:
            st.warning(f"No se pudo entrenar el modelo de regresión logística con los datos actuales. Asegúrate de tener suficientes ejemplos de AMBOS tipos de días (Lluvia y No Lluvia) y que no todos los puntos sean iguales. Error: {e}")
            model_trained_weather = False
    else:
        st.info("Necesitas añadir ejemplos de DÍAS CON LLUVIA y DÍAS SIN LLUVIA (al menos uno de cada tipo) para que el Predictor pueda aprender.")
        model_trained_weather = False
else:
    st.info(f"Necesitas al menos 5 ejemplos de días (tienes {len(st.session_state.weather_data)}) para que el Predictor empiece a aprender.")
    model_trained_weather = False

# UI para añadir nuevos datos del clima
st.markdown("---")
st.subheader("¡Añade ejemplos de días pasados al Predictor del Clima!")

col_add_weather1, col_add_weather2, col_add_weather3 = st.columns(3)

with col_add_weather1:
    day_temp = st.slider("Temperatura del día (°C):", min_value=0.0, max_value=40.0, value=20.0, step=1.0, key="day_temp_slider")
with col_add_weather2:
    day_humidity = st.slider("Humedad del día (%):", min_value=0.0, max_value=100.0, value=50.0, step=5.0, key="day_humidity_slider")
with col_add_weather3:
    did_it_rain = st.radio("¿Llovió ese día?:", ('Sí', 'No'), key="did_it_rain_radio")
    add_weather_button = st.button("Añadir este ejemplo", key="add_weather_point")
    if add_weather_button:
        st.session_state.weather_data.append({'temp': day_temp, 'humidity': day_humidity, 'rain': (did_it_rain == 'Sí')})
        st.session_state.last_weather_prediction = None # Borra la predicción anterior al añadir nuevos datos
        st.rerun()

if st.button("Borrar todos los ejemplos del clima", key="clear_weather_points"):
    st.session_state.weather_data = []
    st.session_state.last_weather_prediction = None
    st.session_state.trained_model = None # También borra el modelo entrenado
    st.rerun()

# UI para hacer una predicción de clima
if model_trained_weather and st.session_state.trained_model:
    st.markdown("---")
    st.subheader("¡Haz que el Predictor adivine el clima de mañana!")
    col_predict_weather1, col_predict_weather2, col_predict_weather3 = st.columns(3)
    with col_predict_weather1:
        predict_temp = st.slider("Temperatura de mañana (°C):", min_value=0.0, max_value=40.0, value=25.0, step=1.0, key="predict_weather_temp")
    with col_predict_weather2:
        predict_humidity = st.slider("Humedad de mañana (%):", min_value=0.0, max_value=100.0, value=70.0, step=5.0, key="predict_weather_humidity")
    with col_predict_weather3:
        st.markdown(" ")
        st.markdown(" ")
        if st.button("🔮 ¡Adivinar el clima!", key="predict_weather_button"):
            new_day_features = np.array([[predict_temp, predict_humidity]])
            predicted_class = st.session_state.trained_model.predict(new_day_features)[0]
            predicted_rain = (predicted_class == 1) # Convertir la clase de vuelta a booleano

            st.session_state.last_weather_prediction = {
                'temp': predict_temp,
                'humidity': predict_humidity,
                'predicted_rain': predicted_rain
            }
            st.rerun()

    # Muestra el resultado de la última predicción si está disponible
    if st.session_state.last_weather_prediction:
        predicted_data = st.session_state.last_weather_prediction
        predicted_type_text = "¡Va a llover! ☔" if predicted_data['predicted_rain'] else "¡No va a llover, hará sol! ☀️"
        predict_temp = predicted_data['temp']
        predict_humidity = predicted_data['humidity']

        st.markdown(f"El Predictor del Clima adivina: **{predicted_type_text}**")
        marker_color = 'purple'
        marker_symbol = 'X'
        ax_weather.plot(predict_temp, predict_humidity, marker_symbol, color=marker_color, markersize=15, label=f'Predicción: {predicted_type_text}', zorder=4)
        
        handles, labels = ax_weather.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_weather.legend(by_label.values(), by_label.keys())

st.pyplot(fig_weather, use_container_width=True)


st.markdown("""
¿Ves cómo el 'Predictor' intenta encontrar una forma de separar los días de lluvia de los días de sol?
Cuando le das suficientes ejemplos de Días de Lluvia y Días de Sol, ¡aprende a diferenciar!
Luego, cuando le presentas un nuevo día con su temperatura y humedad, usa lo que ha aprendido para **predecir**
si lloverá o no.

Así es como los Modelos Predictivos pueden adivinar si un email es spam, si una foto
tiene un perro o un gato, ¡o incluso ayudar a los médicos a predecir enfermedades!
""")

st.write("---")

# --- Sección de Chatbot de Juego con Adivino ---
st.header("¡Juega y Aprende con Adivino sobre los Modelos Predictivos!")
st.markdown("¡Hola! Soy Adivino, tu compañero que predice el futuro con datos. ¿Listo para descubrir cómo las máquinas adivinan cosas?")

if client:
    # Inicializa el estado del juego y los mensajes del chat
    if "adivino_game_active" not in st.session_state:
        st.session_state.adivino_game_active = False
    if "adivino_game_messages" not in st.session_state:
        st.session_state.adivino_game_messages = []
    if "adivino_current_question" not in st.session_state:
        st.session_state.adivino_current_question = None
    if "adivino_current_options" not in st.session_state:
        st.session_state.adivino_current_options = {}
    if "adivino_correct_answer" not in st.session_state:
        st.session_state.adivino_correct_answer = None
    if "adivino_awaiting_next_game_decision" not in st.session_state:
        st.session_state.adivino_awaiting_next_game_decision = False
    if "adivino_game_needs_new_question" not in st.session_state:
        st.session_state.adivino_game_needs_new_question = False
    if "adivino_correct_streak" not in st.session_state:
        st.session_state.adivino_correct_streak = 0
    if "last_played_question_adivino" not in st.session_state:
        st.session_state.last_played_question_adivino = None


    adivino_game_system_prompt = f"""
    Eres un **experto consumado en Modelos Predictivos y Análisis de Datos Avanzado**, con una profunda comprensión de las metodologías, algoritmos y aplicaciones prácticas en diversos dominios. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de los Modelos Predictivos mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Predicción acertada! Tu modelo de pensamiento es robusto." o "Esa no era la hipótesis más probable. Repensemos el enfoque."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "Los modelos predictivos buscan inferir resultados futuros basándose en datos históricos..."]
    [Pregunta para continuar, ej: "¿Listo para afinar tus habilidades en la construcción de modelos?" o "¿Quieres explorar las complejidades de la validación de modelos?"]

    **Reglas adicionales para el Experto en Modelos Predictivos:**
    * **Enfoque Riguroso en Modelos Predictivos:** Todas tus preguntas y explicaciones deben girar en torno a los Modelos Predictivos. Cubre sus fundamentos (definición, tipos, ciclo de vida), técnicas clave (regresión, clasificación, series temporales, clustering), métricas de evaluación (RMSE, MAE, R-cuadrado, precisión, recall, F1-score, AUC), sobreajuste/subajuste, validación (cruzada, hold-out), selección de características (feature engineering), interpretación de modelos y ética.
    * **¡VARIEDAD, VARIEDAD, VARIEDAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de Modelos Predictivos que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General y Ciclo de Vida:** ¿Qué es un modelo predictivo? Fases (recopilación, preprocesamiento, modelado, evaluación, despliegue).
        * **Tipos de Problemas Predictivos:** Regresión (predicción de valores continuos), Clasificación (predicción de categorías), Series Temporales.
        * **Algoritmos Fundamentales:** Regresión Lineal/Logística, Árboles de Decisión, Random Forests, Gradient Boosting, SVM, K-NN, K-Means (como pre-procesamiento o para segmentación).
        * **Preprocesamiento de Datos:** Manejo de valores nulos, escalado, codificación de variables, detección de outliers.
        * **Evaluación de Modelos:** Métricas clave para regresión (RMSE, MAE, R2) y clasificación (matriz de confusión, precisión, recall, F1-score, curva ROC/AUC).
        * **Validación y Generalización:** Validación cruzada (K-fold), hold-out, sobreajuste (overfitting) y subajuste (underfitting), sesgo/varianza.
        * **Selección e Ingeniería de Características:** Importancia de características (feature importance), creación de nuevas características.
        * **Interpretación y Explicabilidad (XAI):** Métodos para entender las predicciones del modelo (LIME, SHAP).
        * **Despliegue y Monitoreo:** Cómo se ponen en producción los modelos y se supervisa su rendimiento.
        * **Ética y Sesgos en Modelos Predictivos:** Fairness, implicaciones sociales de las predicciones.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.adivino_correct_streak} preguntas correctas consecutivas. # Cambiado
        * **Nivel 1 (Analista de Datos Principiante – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de predecir el futuro con datos. Analogías simples.
            * *Tono:* "Estás dando tus primeros pasos en el vasto universo de las predicciones."
        * **Nivel 2 (Constructor de Modelos – 3-5 respuestas correctas):** Tono más técnico. Introduce tipos de modelos (regresión, clasificación) y la idea de "entrenar" con datos.
            * *Tono:* "Tu comprensión de los fundamentos predictivos es sólida, estás listo para aplicar tus conocimientos."
        * **Nivel 3 (Arquitecto de Predicciones – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Introduce algoritmos específicos, métricas de evaluación, y conceptos de sobreajuste.
            * *Tono:* "Tu análisis demuestra una comprensión profunda de los algoritmos y arquitecturas de modelos predictivos."
        * **Nivel Maestro (Científico de Datos Principal – 9+ respuestas correctas):** Tono de **especialista en investigación y desarrollo de vanguardia**. Preguntas sobre desafíos abiertos, implicaciones éticas complejas, arquitecturas avanzadas, o el impacto socioeconómico.
            * *Tono:* "Tu maestría en el diseño, implementación y evaluación de sistemas predictivos es excepcional. Estás en la vanguardia de la innovación en Ciencia de Datos."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Analogías (Adaptadas al Nivel):**
        * **Nivel 1:** Predecir si lloverá mañana basándose en nubes pasadas.
        * **Nivel 2:** Un sistema que predice qué producto te gustará comprar en una tienda online.
        * **Nivel 3:** Un modelo que predice la probabilidad de que un paciente desarrolle una enfermedad, o el precio de una acción en el futuro.
        * **Nivel Maestro:** El desarrollo de un modelo predictivo robusto para la detección de fraudes financieros a gran escala, o la creación de nuevos algoritmos para predecir eventos climáticos extremos con mayor precisión.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre MODELOS PREDICTIVOS, y asegúrate de que no se parezca a las anteriores.
    """

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_adivino_question_response(raw_text):
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
            st.warning(f"DEBUG (Adivino): Formato de pregunta inesperado de la API. Texto recibido:\n{raw_text}")
            return None, {}, ""
        return question, options, correct_answer_key

    # Función para parsear la respuesta de feedback de la IA
    def parse_adivino_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG (Adivino): Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        return "Predicción procesada.", "Aquí tienes la explicación.", "¿Listo para tu próxima predicción?"
    
    # --- Funciones para subir de nivel directamente ---
    def set_adivino_level(target_streak, level_name):
        st.session_state.adivino_correct_streak = target_streak
        st.session_state.adivino_game_active = True
        st.session_state.adivino_game_messages = []
        st.session_state.adivino_current_question = None
        st.session_state.adivino_current_options = {}
        st.session_state.adivino_correct_answer = None
        st.session_state.adivino_game_needs_new_question = True
        st.session_state.adivino_awaiting_next_game_decision = False 
        st.session_state.adivino_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}**! Prepárate para predicciones más desafiantes. ¡Aquí va tu primera!"}) # Cambiado
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons_adivino, col_level_up_buttons_adivino = st.columns([1, 2])

    with col_game_buttons_adivino:
        if st.button("¡Vamos a predecir con Adivino!", key="start_adivino_game_button"):
            st.session_state.adivino_game_active = True
            st.session_state.adivino_game_messages = []
            st.session_state.adivino_current_question = None
            st.session_state.adivino_current_options = {}
            st.session_state.adivino_correct_answer = None
            st.session_state.adivino_game_needs_new_question = True
            st.session_state.adivino_awaiting_next_game_decision = False
            st.session_state.adivino_correct_streak = 0
            st.session_state.last_played_question_adivino = None
            st.rerun()
    
    with col_level_up_buttons_adivino:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un as de las predicciones? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_adivino, col_lvl2_adivino, col_lvl3_adivino = st.columns(3) # Tres columnas para los botones de nivel
        with col_lvl1_adivino:
            if st.button("Subir a Nivel Constructor (Adivino - Aprendiz)", key="level_up_medium_adivino"):
                set_adivino_level(3, "Constructor de Modelos") # 3 respuestas correctas para Nivel Intermedio
        with col_lvl2_adivino:
            if st.button("Subir a Nivel Arquitecto (Adivino - Experto)", key="level_up_advanced_adivino"):
                set_adivino_level(6, "Arquitecto de Predicciones") # 6 respuestas correctas para Nivel Avanzado
        with col_lvl3_adivino:
            if st.button("👑 ¡Maestro de Predicciones! (Adivino - Experto)", key="level_up_champion_adivino"):
                set_adivino_level(9, "Científico de Datos Principal") # 9 respuestas correctas para Nivel Maestro


    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.adivino_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica del juego del chatbot si está activo
    if st.session_state.adivino_game_active:
        if st.session_state.adivino_current_question is None and st.session_state.adivino_game_needs_new_question and not st.session_state.adivino_awaiting_next_game_decision:
            with st.spinner("Adivino está preparando una nueva predicción..."):
                try:
                    game_messages_for_api = [{"role": "system", "content": adivino_game_system_prompt}]
                    # Limita el historial para evitar prompts demasiado largos
                    for msg in st.session_state.adivino_game_messages[-6:]:
                        if msg["role"] == "assistant" and msg["content"].startswith("**"):
                            game_messages_for_api.append({"role": "assistant", "content": f"PREGUNTA ANTERIOR: {msg['content'].splitlines()[0]}"})
                        elif msg["role"] == "user" and "MI RESPUESTA:" not in msg["content"]:
                            game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {msg['content']}"})

                    game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre MODELOS PREDICTIVOS siguiendo el formato exacto. ¡Recuerda, la pregunta debe ser muy VARIADA y CREATIVA, y no se debe parecer a las anteriores!"})

                    game_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=game_messages_for_api,
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_adivino_question_text = game_response.choices[0].message.content
                    question, options, correct_answer_key = parse_adivino_question_response(raw_adivino_question_text)

                    if question:
                        st.session_state.adivino_current_question = question
                        st.session_state.adivino_current_options = options
                        st.session_state.adivino_correct_answer = correct_answer_key 

                        display_question_text = f"**Nivel {int(st.session_state.adivino_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.adivino_correct_streak}**\n\n**Pregunta de Adivino:** {question}\n\n" # Cambiado
                        for key in sorted(options.keys()):
                            display_question_text += f"{key}) {options[key]}\n"

                        st.session_state.adivino_game_messages.append({"role": "assistant", "content": display_question_text})
                        st.session_state.adivino_game_needs_new_question = False
                        st.session_state.last_played_question_adivino = None # Resetear para forzar el audio
                        st.rerun()
                    else:
                        st.session_state.adivino_game_messages.append({"role": "assistant", "content": "¡Lo siento! Adivino no pudo preparar la pregunta en el formato correcto. ¿Puedes pulsar '¡Vamos a predecir con Adivino!' de nuevo?"}) # Cambiado
                        st.session_state.adivino_game_active = False
                        st.rerun()

                except Exception as e:
                    st.error(f"¡Oops! Adivino no pudo hacer la pregunta. Error: {e}")
                    st.session_state.adivino_game_messages.append({"role": "assistant", "content": "¡Lo siento! Adivino tiene un pequeño problema para hacer preguntas ahora. ¿Puedes intentarlo de nuevo?"}) # Cambiado
                    st.session_state.adivino_game_active = False
                    st.rerun()


        if st.session_state.adivino_current_question is not None and not st.session_state.adivino_awaiting_next_game_decision:
            # Audio de la pregunta
            if st.session_state.get('last_played_question_adivino') != st.session_state.adivino_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.adivino_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.adivino_correct_streak}. Pregunta de Adivino: {st.session_state.adivino_current_question}. Opción A: {st.session_state.adivino_current_options.get('A', '')}. Opción B: {st.session_state.adivino_current_options.get('B', '')}. Opción C: {st.session_state.adivino_current_options.get('C', '')}." # Cambiado
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                    st.session_state.last_played_question_adivino = st.session_state.adivino_current_question # Guardar la pregunta reproducida
                except Exception as e:
                    st.warning(f"Error al generar o reproducir el audio de la pregunta de Adivino: {e}")


            with st.form("adivino_game_form", clear_on_submit=True):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_choice = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.adivino_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.adivino_current_options[x]}",
                        key="adivino_answer_radio_buttons",
                        label_visibility="collapsed"
                    )

                submit_button = st.form_submit_button("Enviar Respuesta")

            if submit_button:
                st.session_state.adivino_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_choice}) {st.session_state.adivino_current_options[user_choice]}"}) # Cambiado
                prev_streak = st.session_state.adivino_correct_streak

                # Lógica para actualizar el contador de respuestas correctas
                if user_choice == st.session_state.adivino_correct_answer:
                    st.session_state.adivino_correct_streak += 1
                else:
                    st.session_state.adivino_correct_streak = 0 # Resetear si falla

                radio_placeholder.empty()

                # --- Lógica de subida de nivel ---
                if st.session_state.adivino_correct_streak > 0 and \
                   st.session_state.adivino_correct_streak % 3 == 0 and \
                   st.session_state.adivino_correct_streak > prev_streak:
                    
                    if st.session_state.adivino_correct_streak < 9: # Niveles Básico, Intermedio, Avanzado
                        current_level_text = ""
                        if st.session_state.adivino_correct_streak == 3:
                            current_level_text = "Constructor de Modelos (como un estudiante que ya entiende un poco el futuro)"
                        elif st.session_state.adivino_correct_streak == 6:
                            current_level_text = "Arquitecto de Predicciones (como un científico de datos novato)"
                        
                        level_up_message = f"🎉 ¡Increíble! ¡Has acertado {st.session_state.adivino_correct_streak} predicciones seguidas! ¡Felicidades! Has subido al **Nivel {current_level_text}** de Modelos Predictivos. ¡Las predicciones serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a previsor/a! 🚀" # Cambiado
                        st.session_state.adivino_game_messages.append({"role": "assistant", "content": level_up_message})
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
                            st.warning(f"No se pudo reproducir el audio de subida de nivel (Adivino): {e}")
                    elif st.session_state.adivino_correct_streak >= 9:
                        medals_earned = (st.session_state.adivino_correct_streak - 6) // 3 # (9-6)//3 = 1ª medalla, (12-6)//3 = 2ª medalla
                        medal_message = f"🏅 ¡FELICITACIONES, MAESTRO DE PREDICCIONES! ¡Has ganado tu {medals_earned}ª Medalla de la Bola de Cristal! ¡Tu habilidad para ver el futuro con datos es asombrosa y digna de un verdadero EXPERTO en Modelos Predictivos! ¡Sigue así! 🌟" # Cambiado
                        st.session_state.adivino_game_messages.append({"role": "assistant", "content": medal_message})
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
                            st.warning(f"No se pudo reproducir el audio de medalla (Adivino): {e}")
                        
                        if prev_streak < 9:
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Maestro de Predicciones**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros futuristas que entienden los secretos de cómo predecir el mundo! ¡Adelante!"
                            st.session_state.adivino_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                audio_fp_level_up_champion = io.BytesIO()
                                tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                                audio_fp_level_up_champion.seek(0)
                                st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0, autoplay=True)
                                time.sleep(2)
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón (Adivino): {e}")


                feedback_prompt = f"""
                El usuario respondió '{user_choice}'. La pregunta era: '{st.session_state.adivino_current_question}'. 
                La respuesta correcta era '{st.session_state.adivino_correct_answer}'. 
                Da feedback como Adivino.
                Si es CORRECTO, el mensaje es "¡Predicción acertada! ¡Lo has clavado!" o similar.
                Si es INCORRECTO, el mensaje es "¡Hmm, esa predicción no fue del todo precisa! ¡No pasa nada!" o similar.
                Luego, una explicación sencilla para el usuario.
                Finalmente, pregunta: "¿Listo para tu próxima predicción?".
                **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                """

                with st.spinner("Adivino está revisando tu predicción..."):
                    try:
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": adivino_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.8,
                            max_tokens=300
                        )
                        raw_adivino_feedback_text = feedback_response.choices[0].message.content

                        feedback_msg, explanation_msg, next_question_prompt = parse_adivino_feedback_response(raw_adivino_feedback_text)

                        st.session_state.adivino_game_messages.append({"role": "assistant", "content": feedback_msg})
                        st.session_state.adivino_game_messages.append({"role": "assistant", "content": explanation_msg})
                        st.session_state.adivino_game_messages.append({"role": "assistant", "content": next_question_prompt})

                        try:
                            tts = gTTS(text=f"{feedback_msg}. {explanation_msg}. {next_question_prompt}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback (Adivino): {e}")


                        st.session_state.adivino_current_question = None # Cambiado
                        st.session_state.adivino_current_options = {} # Cambiado
                        st.session_state.adivino_correct_answer = None # Cambiado
                        st.session_state.adivino_game_needs_new_question = False # Cambiado
                        st.session_state.adivino_awaiting_next_game_decision = True # Cambiado

                        st.rerun()

                    except Exception as e:
                        st.error(f"Ups, Adivino no pudo procesar tu predicción. Error: {e}")
                        st.session_state.adivino_game_messages.append({"role": "assistant", "content": "Lo siento, Adivino tiene un pequeño problema técnico ahora mismo. ¡Pero me encantaría ver tu predicción!"}) # Cambiado


        if st.session_state.adivino_awaiting_next_game_decision: # Cambiado
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col1_adivino, col2_adivino = st.columns(2) # Cambiado
            with col1_adivino: # Cambiado
                if st.button("👍 Sí, quiero más predicciones", key="play_more_questions_adivino"): # Cambiado key
                    st.session_state.adivino_game_needs_new_question = True # Cambiado
                    st.session_state.adivino_awaiting_next_game_decision = False # Cambiado
                    st.session_state.adivino_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío de predicción!"}) # Cambiado
                    st.rerun()
            with col2_adivino: # Cambiado
                if st.button("👎 No, ya no quiero predecir más", key="stop_playing_adivino"): # Cambiado key
                    st.session_state.adivino_game_active = False # Cambiado
                    st.session_state.adivino_awaiting_next_game_decision = False # Cambiado
                    st.session_state.adivino_game_messages.append({"role": "assistant", "content": "¡De acuerdo! ¡Gracias por practicar tus predicciones conmigo! Espero que hayas aprendido mucho. ¡Hasta la próxima predicción!"}) # Cambiado
                    st.rerun()

else:
    # Solo mostrar este mensaje si el juego de Adivino NO está activo
    # y si no hay una clave de API de OpenAI configurada.
    if client is None:
        st.info("Para usar la sección de preguntas de Adivino, necesitas configurar tu clave de API de OpenAI en `secrets.toml`.")