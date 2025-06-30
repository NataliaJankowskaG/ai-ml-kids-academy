# pages/04_Que_es_Regresion_Lineal.py

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="¿Qué es la Regresión Lineal?",
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
LOTTIE_PREDICT_PATH = os.path.join("assets", "lottie_animations", "Math.json")


# --- Configuración de la API de OpenAI ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai_api_key = None
    st.error("Error: La clave de API de OpenAI no está configurada en `secrets.toml`.")
    st.info("Para configurarla, crea un archivo `.streamlit/secrets.toml` en la raíz de tu proyecto y añade: `OPENAI_API_KEY = 'tu_clave_aqui'`")

client = OpenAI(api_key=openai_api_key) if openai_api_key else None


st.subheader("¡Adivina el futuro con matemáticas simples!")

st.write("---")

# Sección 1: ¿Qué es la Regresión Lineal?
st.header("¿Qué es la Regresión Lineal?")
st.markdown("""
Imagina que quieres saber cuánto crecerá una planta la próxima semana,
o cuántos helados se venderán si hace mucho calor.

La **Regresión Lineal** es como tener una bola de cristal matemática.
Nos ayuda a **predecir un número** basándonos en otros números que ya conocemos.
¡Es como encontrar la "línea mágica" que conecta los puntos de nuestros datos!
""")

# Pequeña animación para la introducción
col_intro_left, col_intro_right = st.columns([1, 1])
with col_intro_right:
    lottie_predict = load_lottiefile(LOTTIE_PREDICT_PATH)
    if lottie_predict:
        st_lottie(lottie_predict, height=200, width=200, key="predict_intro")
    else:
        st.info("Consejo: Asegúrate de que 'Math.json' (o una mejor) esté en 'assets/lottie_animations/' para esta animación.")

st.write("---")

# Sección 2: ¿Cómo Predice la Regresión Lineal? (Visualización Interactiva)
st.header("¿Cómo Predice la Regresión Lineal?")
st.markdown("""
Las máquinas usan la regresión lineal para encontrar una línea recta que se ajuste lo mejor posible a los puntos de datos.
¡Así pueden hacer predicciones!

**¡Vamos a simular cómo una IA predice el tamaño de un zapato en función de la altura!**
""")

st.subheader("Predicción del Tamaño del Zapato: ¡Ayuda a la IA a dibujar la línea!")

st.markdown("""
Puedes añadir "ejemplos" (puntos) haciendo clic en el gráfico. La IA intentará dibujar una línea
que se ajuste a tus puntos y así hará predicciones.
""")

# Inicializar juego en session_state
if 'shoe_data' not in st.session_state:
    st.session_state.shoe_data = [] # Lista de diccionarios: [{'height': x, 'shoe_size': y}]

# Crear el gráfico
fig_reg, ax_reg = plt.subplots(figsize=(9, 7))
ax_reg.set_xlabel("Altura (cm)")
ax_reg.set_ylabel("Talla de Zapato")
ax_reg.set_title("Predicción de Talla de Zapato vs. Altura")
ax_reg.set_xlim(100, 200) # Altura típica
ax_reg.set_ylim(20, 50) # Talla de zapato típica
ax_reg.grid(True, linestyle='--', alpha=0.6)

if not st.session_state.shoe_data:
    ax_reg.text((ax_reg.get_xlim()[0] + ax_reg.get_xlim()[1]) / 2,
                (ax_reg.get_ylim()[0] + ax_reg.get_ylim()[1]) / 2,
                "¡Haz clic en el gráfico para añadir puntos!",
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='gray', alpha=0.6)

# Convertir los datos a numpy arrays para el modelo
X_reg_data = np.array([d['height'] for d in st.session_state.shoe_data]).reshape(-1, 1)
y_reg_data = np.array([d['shoe_size'] for d in st.session_state.shoe_data])

# Plotear los puntos existentes
if len(st.session_state.shoe_data) > 0:
    ax_reg.scatter(X_reg_data, y_reg_data, color='blue', s=100, label='Ejemplos (Talla de Zapato)', zorder=3)
    ax_reg.legend()

model_reg = None
# Entrenar el modelo de regresión lineal si hay suficientes puntos (al menos 2)
if len(st.session_state.shoe_data) >= 2:
    model_reg = LinearRegression()
    model_reg.fit(X_reg_data, y_reg_data)

    # Dibujar la línea de regresión
    x_line = np.array([ax_reg.get_xlim()[0], ax_reg.get_xlim()[1]]).reshape(-1, 1)
    y_line = model_reg.predict(x_line)
    ax_reg.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Línea de Predicción de la IA', zorder=2)
    ax_reg.legend()

    # Evaluar el modelo
    y_pred = model_reg.predict(X_reg_data)
    mse = mean_squared_error(y_reg_data, y_pred)
    r2 = r2_score(y_reg_data, y_pred)
    st.markdown(f"**La línea de la IA:** Cuanto mejor se ajusta a los puntos, mejor puede predecir.")

# Mostrar el gráfico
clicked_point = st.pyplot(fig_reg, use_container_width=True)


st.markdown("---")
st.subheader("¡Añade tus propios puntos de datos y ve la línea mágica!")

col_add_point1, col_add_point2, col_add_point3 = st.columns(3)

with col_add_point1:
    user_height = st.slider("Altura (cm):", min_value=100.0, max_value=200.0, value=150.0, step=1.0, key="user_height_reg")
with col_add_point2:
    user_shoe_size = st.slider("Talla de Zapato:", min_value=20.0, max_value=50.0, value=35.0, step=0.5, key="user_shoe_size_reg")
with col_add_point3:
    st.markdown(" ")
    st.markdown(" ")
    add_point_button = st.button("Añadir este punto al gráfico", key="add_reg_point")
    if add_point_button:
        st.session_state.shoe_data.append({'height': user_height, 'shoe_size': user_shoe_size})
        st.rerun()

if st.button("Borrar todos los puntos", key="clear_reg_points"):
    st.session_state.shoe_data = []
    st.rerun()

if model_reg is not None:
    st.markdown("---")
    st.subheader("¡Haz una predicción con la línea de la IA!")
    predict_height = st.slider("¿Para qué altura quieres predecir la talla de zapato?", min_value=100.0, max_value=200.0, value=170.0, step=1.0, key="predict_height_slider")
    predicted_shoe_size = model_reg.predict(np.array([[predict_height]]))[0]
    st.markdown(f"Si la altura es **{predict_height:.0f} cm**, la IA predice que la talla de zapato será **{predicted_shoe_size:.1f}**.")
    ax_reg.plot(predict_height, predicted_shoe_size, 'o', color='purple', markersize=12, label='Predicción de IA 🔮', zorder=4)
    ax_reg.legend()
    st.pyplot(fig_reg) # Volver a mostrar el gráfico con la predicción


st.markdown("""
¿Ves cómo la línea roja se mueve cuando añades más puntos? La IA intenta encontrar la mejor línea recta
que pase por el medio de todos los puntos. Una vez que tiene esa línea, puede **predecir**
nuevos valores, ¡incluso para alturas que no le has dado antes!

Esto es útil para predecir precios de casas, ventas de productos, ¡o incluso el clima!
""")

st.write("---")

# --- Sección de Chatbot de Juego con Líneo para "Qué es la Regresión Lineal" ---
st.header("¡Juega y Aprende con Líneo sobre la Regresión Lineal!")
st.markdown("¡Hola! Soy Líneo, tu compañero que dibuja el futuro. ¿Listo para descubrir cómo las máquinas adivinan números?")

if client:
    # Inicializa el estado del juego y los mensajes del chat
    if "reg_game_active" not in st.session_state:
        st.session_state.reg_game_active = False
    if "reg_game_messages" not in st.session_state:
        st.session_state.reg_game_messages = []
    if "reg_current_question" not in st.session_state:
        st.session_state.reg_current_question = None
    if "reg_current_options" not in st.session_state:
        st.session_state.reg_current_options = {}
    if "reg_correct_answer" not in st.session_state:
        st.session_state.reg_correct_answer = None
    if "reg_awaiting_next_game_decision" not in st.session_state:
        st.session_state.reg_awaiting_next_game_decision = False
    if "reg_game_needs_new_question" not in st.session_state:
        st.session_state.reg_game_needs_new_question = False
    if "reg_correct_streak" not in st.session_state:
        st.session_state.reg_correct_streak = 0
    if "last_played_question_lineo" not in st.session_state:
        st.session_state.last_played_question_lineo = None


    # System prompt para el juego de preguntas
    reg_game_system_prompt = f"""
    Eres un **experto consumado en Modelado Estadístico y Machine Learning**, con una especialización profunda en el **Análisis de Regresión Lineal**. Comprendes a fondo sus fundamentos teóricos, supuestos, aplicaciones prácticas y limitaciones. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de la Regresión Lineal mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Ajuste perfecto! Tu predicción es precisa." o "Esa estimación necesita revisarse. Repasemos los coeficientes."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "La Regresión Lineal es un método estadístico para modelar la relación lineal entre una variable dependiente continua y una o más variables independientes..."]
    [Pregunta para continuar, ej: "¿Listo para optimizar tus modelos de predicción?" o "¿Quieres explorar los supuestos críticos de la regresión lineal?"]

    **Reglas adicionales para el Experto en Regresión Lineal:**
    * **Enfoque Riguroso en Regresión Lineal:** Todas tus preguntas y explicaciones deben girar en torno a la Regresión Lineal (simple y múltiple). Cubre sus fundamentos (ecuación, coeficientes, intercepción), supuestos (linealidad, independencia, homocedasticidad, normalidad de los residuos), interpretación de resultados (coeficientes, p-valores, $R^2$), evaluación del modelo (RMSE, MAE, $R^2$, $R^2$ ajustado), manejo de outliers y multicolinealidad, y aplicaciones prácticas.
    * **¡VARIEDAD, VARIEDAD, VARIEDAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de Regresión Lineal que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General:** ¿Qué es la regresión lineal? ¿Para qué sirve? (predicción de valores continuos).
        * **Ecuación de Regresión:** Comprensión de $y = \beta_0 + \beta_1 x + \epsilon$, qué representan $\beta_0$, $\beta_1$, $x$, $y$, y $\epsilon$.
        * **Estimación de Parámetros:** Mínimos Cuadrados Ordinarios (OLS) de forma intuitiva.
        * **Supuestos del Modelo:**
            * **Linealidad:** La relación entre variables es lineal.
            * **Independencia de Residuos:** No autocorrelación.
            * **Homocedasticidad:** Varianza constante de los residuos.
            * **Normalidad de Residuos:** Residuos distribuidos normalmente.
            * **No Multicolinealidad** (para regresión múltiple).
        * **Interpretación de Coeficientes:** Cómo se interpreta $\beta_1$ y $\beta_0$.
        * **Evaluación del Modelo:**
            * **Métricas:** RMSE, MAE, $R^2$ (coeficiente de determinación), $R^2$ ajustado.
            * **Significancia Estadística:** P-valores, intervalos de confianza.
        * **Diagnóstico del Modelo:** Análisis de residuos (gráficos de residuos vs. predichos), gráficos Q-Q.
        * **Manejo de Problemas:** Outliers, heterocedasticidad, multicolinealidad (VIF).
        * **Regresión Lineal Múltiple:** Añadir más predictores, diferencias con la simple.
        * **Ventajas y Limitaciones:** Simplicidad, interpretabilidad vs. rigidez de supuestos.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.reg_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Aprendiz de Estadístico – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de encontrar una línea para predecir un valor y ejemplos simples de relaciones lineales.
            * *Tono:* "Estás trazando tus primeras líneas en el mapa de las predicciones estadísticas."
        * **Nivel 2 (Analista de Regresión – 3-5 respuestas correctas):** Tono más técnico. Introduce la ecuación básica, los conceptos de variable dependiente e independiente, y la interpretación fundamental de los coeficientes.
            * *Tono:* "Tu análisis de la relación entre variables es cada vez más preciso."
        * **Nivel 3 (Modelador de Regresión – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en los supuestos del modelo, las métricas de evaluación ($R^2$, RMSE), la detección de problemas (outliers, heterocedasticidad) y la regresión lineal múltiple.
            * *Tono:* "Tu habilidad para construir, evaluar y diagnosticar modelos de regresión lineal es fundamental para el análisis predictivo."
        * **Nivel Maestro (Científico de Datos Cuantitativo – 9+ respuestas correctas):** Tono de **especialista en modelado estadístico avanzado**. Preguntas sobre la violación de supuestos y sus consecuencias, la corrección de problemas complejos (transformaciones de Box-Cox, weighted least squares), la comparación con otros modelos lineales generalizados, o las implicaciones de la multicolinealidad en la inferencia. Se esperan respuestas que demuestren una comprensión teórica y práctica robusta.
            * *Tono:* "Tu maestría en el análisis de regresión lineal te permite desentrañar relaciones complejas y construir modelos predictivos con gran rigor estadístico."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Predecir el peso de una persona basándose en su altura.
        * **Nivel 2:** Estimar el precio de una casa en función de su tamaño en metros cuadrados, o la calificación de un estudiante según las horas de estudio.
        * **Nivel 3:** Modelar el impacto de la inversión publicitaria y el precio en las ventas de un producto, analizando los residuos para verificar la bondad del ajuste.
        * **Nivel Maestro:** Desarrollar un modelo de regresión lineal robusto para predecir el rendimiento de cosechas agrícolas considerando múltiples variables climáticas y de suelo, evaluando cuidadosamente la multicolinealidad y la homocedasticidad.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre REGRESIÓN LINEAL, y asegúrate de que no se parezca a las anteriores.
    """

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_reg_question_response(raw_text):
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
    def parse_reg_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"

    # --- Funciones para subir de nivel directamente ---
    def set_lineo_level(target_streak, level_name):
        st.session_state.reg_correct_streak = target_streak
        st.session_state.reg_game_active = True
        st.session_state.reg_game_messages = []
        st.session_state.reg_current_question = None
        st.session_state.reg_current_options = {}
        st.session_state.reg_correct_answer = None
        st.session_state.reg_game_needs_new_question = True
        st.session_state.reg_awaiting_next_game_decision = False
        st.session_state.reg_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}** de Líneo! Prepárate para preguntas más desafiantes. ¡Aquí va tu primera!"})
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons_lineo, col_level_up_buttons_lineo = st.columns([1, 2])

    with col_game_buttons_lineo:
        if st.button("¡Vamos a jugar con Líneo!", key="start_lineo_game_button"):
            st.session_state.reg_game_active = True
            st.session_state.reg_game_messages = []
            st.session_state.reg_current_question = None
            st.session_state.reg_current_options = {}
            st.session_state.reg_correct_answer = None
            st.session_state.reg_game_needs_new_question = True
            st.session_state.reg_awaiting_next_game_decision = False
            st.session_state.reg_correct_streak = 0
            st.session_state.last_played_question_lineo = None
            st.rerun()
    
    with col_level_up_buttons_lineo:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un experto en líneas? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_lineo, col_lvl2_lineo, col_lvl3_lineo = st.columns(3) # Tres columnas para los botones de nivel
        with col_lvl1_lineo:
            if st.button("Subir a Nivel Medio (Líneo)", key="level_up_medium_lineo"):
                set_lineo_level(3, "Medio") # 3 respuestas correctas para Nivel Medio
        with col_lvl2_lineo:
            if st.button("Subir a Nivel Avanzado (Líneo)", key="level_up_advanced_lineo"):
                set_lineo_level(6, "Avanzado") # 6 respuestas correctas para Nivel Avanzado
        with col_lvl3_lineo:
            if st.button("👑 ¡Maestro de Líneas! (Líneo)", key="level_up_champion_lineo"):
                set_lineo_level(9, "Campeón") # 9 respuestas correctas para Nivel Campeón


    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.reg_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica del juego del chatbot si está activo
    if st.session_state.reg_game_active:
        if st.session_state.reg_current_question is None and st.session_state.reg_game_needs_new_question and not st.session_state.reg_awaiting_next_game_decision:
            with st.spinner("Líneo está preparando una pregunta..."):
                try:
                    # Incluimos el prompt del sistema actualizado con el nivel de dificultad
                    game_messages_for_api = [{"role": "system", "content": reg_game_system_prompt}]
                    if st.session_state.reg_game_messages:
                        last_message = st.session_state.reg_game_messages[-1]
                        if last_message["role"] == "user":
                            game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {last_message['content']}"})
                        elif last_message["role"] == "assistant":
                            game_messages_for_api.append({"role": "assistant", "content": last_message['content']})

                    game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre QUÉ ES LA REGRESIÓN LINEAL siguiendo el formato exacto. ¡Recuerda, la pregunta debe ser muy VARIADA y CREATIVA, y no se debe parecer a las anteriores!"})

                    game_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=game_messages_for_api,
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_reg_question_text = game_response.choices[0].message.content
                    question, options, correct_answer_key = parse_reg_question_response(raw_reg_question_text)

                    if question:
                        st.session_state.reg_current_question = question
                        st.session_state.reg_current_options = options
                        st.session_state.reg_correct_answer = correct_answer_key

                        display_question_text = f"**Nivel {int(st.session_state.reg_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.reg_correct_streak}**\n\n**Pregunta de Líneo:** {question}\n\n"
                        for key in sorted(options.keys()):
                            display_question_text += f"{key}) {options[key]}\n"

                        st.session_state.reg_game_messages.append({"role": "assistant", "content": display_question_text})
                        st.session_state.reg_game_needs_new_question = False
                        st.rerun()
                    else:
                        st.session_state.reg_game_messages.append({"role": "assistant", "content": "¡Lo siento! Líneo no pudo preparar la pregunta en el formato correcto. ¿Puedes pulsar 'VAMOS A JUGAR' de nuevo?"})
                        st.session_state.reg_game_active = False
                        st.rerun()

                except Exception as e:
                    st.error(f"¡Oops! Líneo no pudo hacer la pregunta. Error: {e}")
                    st.session_state.reg_game_messages.append({"role": "assistant", "content": "¡Lo siento! Líneo tiene un pequeño problema para hacer preguntas ahora. ¿Puedes intentarlo de nuevo?"})
                    st.session_state.reg_game_active = False
                    st.rerun()


        if st.session_state.reg_current_question is not None and not st.session_state.reg_awaiting_next_game_decision:
            # Audio de la pregunta
            if st.session_state.get('last_played_question_lineo') != st.session_state.reg_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.reg_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.reg_correct_streak}. Pregunta de Líneo: {st.session_state.reg_current_question}. Opción A: {st.session_state.reg_current_options.get('A', '')}. Opción B: {st.session_state.reg_current_options.get('B', '')}. Opción C: {st.session_state.reg_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                    st.session_state.last_played_question_lineo = st.session_state.reg_current_question
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")


            with st.form("lineo_game_form", clear_on_submit=True):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_choice = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.reg_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.reg_current_options[x]}",
                        key="reg_answer_radio_buttons",
                        label_visibility="collapsed"
                    )

                submit_button = st.form_submit_button("Enviar Respuesta")

            if submit_button:
                st.session_state.reg_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_choice}) {st.session_state.reg_current_options[user_choice]}"})
                prev_streak = st.session_state.reg_correct_streak

                # Lógica para actualizar el contador de respuestas correctas
                if user_choice == st.session_state.reg_correct_answer:
                    st.session_state.reg_correct_streak += 1
                else:
                    st.session_state.reg_correct_streak = 0 # Resetear si falla

                radio_placeholder.empty()

                # --- Lógica de subida de nivel y confeti ---
                if st.session_state.reg_correct_streak > 0 and \
                   st.session_state.reg_correct_streak % 3 == 0 and \
                   st.session_state.reg_correct_streak > prev_streak:
                    
                    if st.session_state.reg_correct_streak < 9: # Niveles Básico, Medio, Avanzado
                        current_level_text = ""
                        if st.session_state.reg_correct_streak == 3:
                            current_level_text = "Medio (como un adolescente que ya sabe algo sobre el tema del colegio)"
                        elif st.session_state.reg_correct_streak == 6:
                            current_level_text = "Avanzado (como un trabajador de Data Science senior)"
                        
                        level_up_message = f"🎉 ¡Increíble! ¡Has respondido {st.session_state.reg_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de Regresión Lineal. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a predictor/a de líneas! 🚀"
                        st.session_state.reg_game_messages.append({"role": "assistant", "content": level_up_message})
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
                    elif st.session_state.reg_correct_streak >= 9: # Nivel Campeón o superior
                        medals_earned = (st.session_state.reg_correct_streak - 6) // 3 
                        medal_message = f"🏅 ¡FELICITACIONES, MAESTRO DE LÍNEAS! ¡Has ganado tu {medals_earned}ª Medalla de Regresión Lineal! ¡Tu conocimiento es asombroso y digno de un verdadero EXPERTO en Regresión Lineal! ¡Sigue así! 🌟"
                        st.session_state.reg_game_messages.append({"role": "assistant", "content": medal_message})
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
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Campeón (Maestro de Regresión Lineal)**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros científicos de datos que dibujan el futuro con sus líneas! ¡Adelante!"
                            st.session_state.reg_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                audio_fp_level_up_champion = io.BytesIO()
                                tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                                audio_fp_level_up_champion.seek(0)
                                st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0, autoplay=True)
                                time.sleep(2)
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón: {e}")


                # Generar feedback de Líneo
                with st.spinner("Líneo está revisando tu respuesta..."):
                    try:
                        feedback_prompt = f"""
                        El usuario respondió '{user_choice}'. La pregunta era: '{st.session_state.reg_current_question}'.
                        La respuesta correcta era '{st.session_state.reg_correct_answer}'.
                        Da feedback como Líneo.
                        Si es CORRECTO, el mensaje es "¡Línea perfecta! ¡Lo has entendido!" o similar.
                        Si es INCORRECTO, el mensaje es "¡Revisa tu trazo!" o similar.
                        Luego, una explicación sencilla para niños y adolescentes.
                        Finalmente, pregunta: "¿Quieres seguir dibujando líneas de predicción?".
                        **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                        """
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": reg_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.8,
                            max_tokens=300
                        )
                        raw_lineo_feedback_text = feedback_response.choices[0].message.content

                        feedback_msg, explanation_msg, next_question_prompt = parse_reg_feedback_response(raw_lineo_feedback_text)

                        st.session_state.reg_game_messages.append({"role": "assistant", "content": feedback_msg})
                        st.session_state.reg_game_messages.append({"role": "assistant", "content": explanation_msg})
                        st.session_state.reg_game_messages.append({"role": "assistant", "content": next_question_prompt})

                        try:
                            tts = gTTS(text=f"{feedback_msg}. {explanation_msg}. {next_question_prompt}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")


                        st.session_state.reg_current_question = None
                        st.session_state.reg_current_options = {}
                        st.session_state.reg_correct_answer = None
                        st.session_state.reg_game_needs_new_question = False
                        st.session_state.reg_awaiting_next_game_decision = True

                        st.rerun()

                    except Exception as e:
                        st.error(f"Ups, Líneo no pudo procesar tu respuesta. Error: {e}")
                        st.session_state.reg_game_messages.append({"role": "assistant", "content": "Lo siento, Líneo tiene un pequeño problema técnico ahora mismo. ¡Pero me encantaría ver tu respuesta!"})


        if st.session_state.reg_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Sí, quiero jugar más preguntas", key="play_more_questions_reg"):
                    st.session_state.reg_game_needs_new_question = True
                    st.session_state.reg_awaiting_next_game_decision = False
                    st.session_state.reg_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío!"})
                    st.rerun()
            with col2:
                if st.button("👎 No, ya no quiero jugar más", key="stop_playing_reg"):
                    st.session_state.reg_game_active = False
                    st.session_state.reg_awaiting_next_game_decision = False
                    st.session_state.reg_game_messages.append({"role": "assistant", "content": "¡De acuerdo! ¡Gracias por jugar conmigo! Espero que hayas aprendido mucho sobre la Regresión Lineal. ¡Nos vemos pronto!"})
                    st.rerun()

else:
    st.info("Para usar la sección de preguntas de Líneo, necesitas configurar tu clave de API de OpenAI en `secrets.toml`.")