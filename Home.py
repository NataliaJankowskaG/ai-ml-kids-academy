# plataforma_educativa_IA/Home.py

import streamlit as st
import os
import json
from streamlit_lottie import st_lottie
from openai import OpenAI # Importa la librería de OpenAI
from gtts import gTTS # Para la síntesis de voz (Google Text-to-Speech)
from pydub import AudioSegment
from pydub.playback import play
import io

st.set_page_config(
    page_title="Academia de Agentes IA para Niños",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
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


# ---- Rutas a tus archivos de animaciones Lottie locales ----
LOTTIE_TOP_ANIMATION_PATH = os.path.join("assets", "lottie_animations", "Nat1.json")
LOTTIE_CENTER_ANIMATION_PATH = os.path.join("assets", "lottie_animations", "Nat.json")

# ---- Configuración de la API de OpenAI ----
# Asegúrate de configurar tu clave de API de forma segura.
# Una buena práctica es usar secrets de Streamlit.
# Para el desarrollo, puedes usar una variable de entorno o directamente aquí (¡no recomendado para producción!)
# st.secrets["OPENAI_API_KEY"]
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Error: La clave de API de OpenAI no está configurada en `secrets.toml`.")
    st.info("Para configurarla, crea un archivo `.streamlit/secrets.toml` en la raíz de tu proyecto y añade: `OPENAI_API_KEY = 'tu_clave_aqui'`")
    openai_api_key = None # Si no hay clave, no inicializamos el cliente

if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None

# ---- Contenido de la página de inicio ----

# Alineación de la animación superior a la derecha
col_top_left, col_top_right = st.columns([2, 1]) # 2 para espacio vacío a la izquierda, 1 para la animación a la derecha

with col_top_right: # Colocamos la animación en la columna de la derecha
    lottie_top_json = load_lottiefile(LOTTIE_TOP_ANIMATION_PATH)
    if lottie_top_json:
        st_lottie(
            lottie_top_json,
            speed=1,
            loop=True,
            quality="high",
            height=150,
            width=150,
            key="top_greeting_animation",
        )
    else:
        st.warning("No se pudo cargar la animación Lottie superior (Nat1.json).")

st.title("¡Bienvenido a la Academia de Agentes IA!")
st.markdown(
    """
        ¡Hola, futuro experto en Inteligencia Artificial! Aquí aprenderás cómo las máquinas pueden pensar, ver, escuchar y hasta crear.
        ¿Estás listo para explorar un mundo lleno de algoritmos, redes neuronales y agentes inteligentes?
        """
)

# ---- Inserta la animación Lottie central y céntrala más a la derecha ----
st.write("---")

col1, col2, col3 = st.columns([0.5, 2, 1]) # Proporciones para moverla más a la derecha

with col2:
    lottie_center_json = load_lottiefile(LOTTIE_CENTER_ANIMATION_PATH)

    if lottie_center_json:
        st_lottie(
            lottie_center_json,
            speed=1,
            loop=True,
            quality="high",
            height=350,
            width=350,
            key="kid_investigate_animation",
        )
        st.markdown(
            "<h3 style='text-align: center; color: #6A5ACD;'>¡Descubre el emocionante mundo de la Inteligencia Artificial!</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "No se pudo cargar la animación Lottie central (Nat.json). Asegúrate de que el archivo sea válido."
        )

st.write("---")

st.subheader("Tu Aventura Comienza Aquí:")
st.markdown(
    """
        Utiliza el menú de la izquierda para navegar por los diferentes niveles y módulos.
        Cada módulo te traerá nuevas ideas, visualizaciones interactivas y emocionantes desafíos.
        """
)

st.info(
    "💡 **Consejo:** ¡No tengas miedo de experimentar con los controles interactivos en cada página!"
)

st.write("---")
st.markdown("### ¡Conoce a nuestro guía, 'Byte'!")
st.write(
    """
        Byte es un pequeño robot IA muy curioso que te acompañará en su aprendizaje.
        Él te explicará los conceptos más difíciles y te dará pistas para resolver los ejercicios.
        """
)

# --- Sección de Chatbot con Byte ---
st.write("---")
st.header("Habla con Byte")
st.markdown(
    """
    ¿Tienes alguna pregunta sobre la Inteligencia Artificial o algo que te pique la curiosidad?
    ¡Pregúntale a Byte! Él está aquí para ayudarte a entender mejor el mundo de la IA.
    """
)

if client: # Solo muestra el chat si la API key está configurada
    # Inicializa el historial de chat si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Mensaje de bienvenida de Byte
        st.session_state.messages.append({"role": "assistant", "content": "¡Hola! Soy Byte. ¿En qué puedo ayudarte hoy sobre la IA?"})

    # Muestra los mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de texto para el usuario
    if prompt := st.chat_input("Pregúntale a Byte..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Byte está pensando..."):
                try:
                    # Preparar el historial para la API
                    messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

                    # Puedes añadir un "system" role para darle contexto a Byte
                    messages_for_api.insert(0, {"role": "system", "content": """
Eres Byte, un pequeño robot IA muy curioso y amable, diseñado para ser el guía de niños y niñas en la "Academia de Agentes IA".
Tu misión es explicarles el emocionante mundo de la Inteligencia Artificial y el Machine Learning de forma sencilla, ¡como si fuera un juego o una aventura!

**Reglas para Byte:**
1.  **Lenguaje súper claro y divertido:** Usa palabras fáciles de entender. Si un concepto es complicado, explícalo con ejemplos de la vida real que un niño pueda conocer (juguetes, animales, dibujos, juegos).
2.  **Sé entusiasta y animado:** ¡Demuestra que te encanta la IA! Usa exclamaciones, emojis (pero no demasiados) y un tono positivo.
3.  **Fomenta la curiosidad:** Haz preguntas que inviten a pensar, pero sin poner a prueba al niño. Por ejemplo: "¿Te imaginas qué más podría hacer una IA?", o "¿Qué crees que pasaría si...?".
4.  **Respuestas cortas y directas:** Los niños tienen una capacidad de atención limitada. Sé conciso. Si la pregunta es compleja, divídela en partes o invítales a explorar un tema más adelante.
5.  **Siempre positivo y de apoyo:** Evita respuestas negativas o frustrantes. Si no sabes algo, puedes decir: "¡Esa es una pregunta muy interesante! ¡Aún estoy aprendiendo sobre eso, como tú! ¿Qué otra cosa te gustaría saber?"
6.  **No uses jergas técnicas:** Si es absolutamente necesario mencionar un término técnico (como "algoritmo" o "red neuronal"), explícalo inmediatamente con un ejemplo infantil.
7.  **Humor suave y apropiado:** Un poco de humor puede hacer la interacción más amena.
8.  **Siempre responde en español de España.**

**Ejemplos de cómo Byte explicaría:**
* **IA:** "La IA es como un cerebro súper inteligente para máquinas, que les ayuda a pensar y aprender, ¡como cuando tú aprendes a atarte los cordones!"
* **Machine Learning:** "Es como si enseñáramos a un robot a reconocer un gato mostrándole muchas fotos de gatos, ¡hasta que sepa distinguirlos él solito!"
* **Algoritmo:** "Es como una receta de cocina, pero para el ordenador. ¡Le dice paso a paso qué hacer!"
* **Red Neuronal:** "Imagina que es como un grupo de amigos que se pasan mensajes muy rápido para resolver un problema, ¡como tu cerebro tiene neuronas que se hablan!"
"""})

                    response = client.chat.completions.create(
                        model="gpt-4o-mini", 
                        messages=messages_for_api,
                        temperature=0.7, # Controla la creatividad de la respuesta
                        max_tokens=200 # Limita la longitud de la respuesta
                    )
                    byte_response_text = response.choices[0].message.content
                    st.markdown(byte_response_text)
                    st.session_state.messages.append({"role": "assistant", "content": byte_response_text})

                    # ---- Funcionalidad de Audio ----
                    # Crea un objeto gTTS
                    tts = gTTS(text=byte_response_text, lang='es', slow=False)
                    # Guarda el audio en un buffer en memoria
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0) # Vuelve al inicio del buffer

                    # Muestra un reproductor de audio
                    st.audio(audio_fp, format="audio/mp3", start_time=0)

                except Exception as e:
                    st.error(f"Ups, Byte no pudo responder ahora mismo. Error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Lo siento, tengo un pequeño problema técnico ahora mismo. ¿Podrías intentar de nuevo más tarde?"})
else:
    st.warning("El chatbot de Byte no está disponible. Por favor, configura tu clave de API de OpenAI.")