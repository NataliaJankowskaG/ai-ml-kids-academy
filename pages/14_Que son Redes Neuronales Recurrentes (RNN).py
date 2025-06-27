import streamlit as st
import os
import json
from streamlit_lottie import st_lottie
from openai import OpenAI
from gtts import gTTS
import io
import random
import time
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
# import matplotlib.pyplot as plt # Ya no se usa
# import plotly.graph_objects as go # Ya no se usa


# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Predicción de Precios con RNN", page_icon="📈", layout="wide")

# ---- Función para cargar animación Lottie desde un archivo local ----
@st.cache_data(ttl=3600) # Se añade el decorador para cachear la función
def load_lottiefile(filepath: str):
    """Carga un archivo JSON de Lottie con manejo de errores mejorado."""
    if not os.path.exists(filepath):
        st.error(f"Error: No se encontró el archivo Lottie en la ruta: {filepath}")
        return None
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error: El archivo Lottie '{filepath}' no es un JSON válido o está corrupto.")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar el archivo Lottie '{filepath}': {e}. Asegúrate de que el archivo no esté corrupto y sea un JSON válido.")
        return None

# --- Definir la raíz del proyecto para una gestión de rutas más robusta ---
# os.path.abspath(__file__) obtiene la ruta absoluta del script actual.
# os.path.dirname(...) se usa para obtener el directorio padre.
# Si el script está en Proyecto_final/pages/script.py, PROJECT_ROOT será Proyecto_final/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Rutas a Lottie y otras imágenes (¡CORREGIDAS USANDO PROJECT_ROOT!) ---
LOTTIE_NEURONS_PATH = os.path.join(PROJECT_ROOT, "assets", "lottie_animations", "neuron_network.json")
LOTTIE_BRAIN_PATH = os.path.join(PROJECT_ROOT, "assets", "lottie_animations", "brain_ai.json")
LOTTIE_TRAINING_PATH = os.path.join(PROJECT_ROOT, "assets", "lottie_animations", "data_training.json")
LOTTIE_ROBOT_PATH = os.path.join(PROJECT_ROOT, "assets", "lottie_animations", "neuron_network.json") # Corregido a robot.json

# --- Ruta a la imagen de Red Neuronal local (¡CORREGIDA USANDO PROJECT_ROOT!) ---
NEURAL_NETWORK_IMAGE_PATH = os.path.join(PROJECT_ROOT, "assets", "imagenes", "neural_network_diagram.jpg")


# --- Cargar Animaciones Lottie ---
lottie_brain_ai = load_lottiefile(LOTTIE_BRAIN_PATH)
lottie_neuron_network = load_lottiefile(LOTTIE_NEURONS_PATH)
lottie_data_training = load_lottiefile(LOTTIE_TRAINING_PATH)
lottie_robot = load_lottiefile(LOTTIE_ROBOT_PATH)


# --- Parámetros de la RNN ---
LOOK_BACK = 10 # Número de días anteriores que la RNN usa para hacer una predicción


# --- Inicialización del cliente de OpenAI ---
client = None
if "OPENAI_API_KEY" in st.secrets:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error(f"Error al inicializar el cliente de OpenAI: {e}. Asegúrate de que la clave de la API sea válida.")
else:
    st.warning("Clave de API de OpenAI no encontrada en Streamlit Secrets. Algunas funcionalidades del chatbot de Detective Pixel no estarán disponibles.")


# --- Rutas del Modelo y Scaler (¡CORREGIDAS USANDO PROJECT_ROOT!) ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'assets', 'models', 'rnn_ibex_predictor_v2.h5')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'assets', 'models', 'scaler_ibex_predictor_v2.pkl')

# --- INICIO: LÍNEAS DE DEPURACIÓN DE RUTAS (Actualizadas para verificar las nuevas rutas) ---
st.sidebar.header("Depuración de Rutas de Archivos")
st.sidebar.markdown(f"**Directorio actual del script (`__file__`):** `{os.path.dirname(__file__)}`")
st.sidebar.markdown(f"**Raíz del Proyecto (`PROJECT_ROOT`):** `{PROJECT_ROOT}`")
st.sidebar.markdown("---") # Separador para claridad

st.sidebar.markdown(f"**Ruta Calculada del Modelo:** `{MODEL_PATH}`")
st.sidebar.markdown(f"**Ruta Calculada del Scaler:** `{SCALER_PATH}`")
st.sidebar.markdown(f"**Ruta Calculada Lottie Brain:** `{LOTTIE_BRAIN_PATH}`")
st.sidebar.markdown(f"**Ruta Calculada Lottie Neurons:** `{LOTTIE_NEURONS_PATH}`")
st.sidebar.markdown(f"**Ruta Calculada Lottie Training:** `{LOTTIE_TRAINING_PATH}`")
st.sidebar.markdown(f"**Ruta Calculada Lottie Robot:** `{LOTTIE_ROBOT_PATH}`")
st.sidebar.markdown(f"**Ruta Calculada Imagen Red Neuronal:** `{NEURAL_NETWORK_IMAGE_PATH}`")


st.sidebar.markdown(f"**¿Existe el Archivo del Modelo?** {'✅ Sí' if os.path.exists(MODEL_PATH) else '❌ No'}")
st.sidebar.markdown(f"**¿Existe el Archivo del Scaler?** {'✅ Sí' if os.path.exists(SCALER_PATH) else '❌ No'}")
st.sidebar.markdown(f"**¿Existe Lottie Brain?** {'✅ Sí' if os.path.exists(LOTTIE_BRAIN_PATH) else '❌ No'}")
st.sidebar.markdown(f"**¿Existe Lottie Neurons?** {'✅ Sí' if os.path.exists(LOTTIE_NEURONS_PATH) else '❌ No'}")
st.sidebar.markdown(f"**¿Existe Lottie Training?** {'✅ Sí' if os.path.exists(LOTTIE_TRAINING_PATH) else '❌ No'}")
st.sidebar.markdown(f"**¿Existe Lottie Robot?** {'✅ Sí' if os.path.exists(LOTTIE_ROBOT_PATH) else '❌ No'}")
st.sidebar.markdown(f"**¿Existe Imagen Red Neuronal?** {'✅ Sí' if os.path.exists(NEURAL_NETWORK_IMAGE_PATH) else '❌ No'}")
st.sidebar.markdown("---")
# --- FIN: LÍNEAS DE DEPURACIÓN DE RUTAS ---


# --- Cargar el Modelo y el Scaler (Cacheado para Rendimiento) ---
@st.cache_resource
def load_rnn_model_and_scaler():
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error al cargar el modelo o el scaler. Asegúrate de que los archivos existan en las rutas correctas y no estén corruptos. Intenté cargar desde: Modelo: `{MODEL_PATH}`, Scaler: `{SCALER_PATH}`. Error: {e}")
        st.stop()

model, scaler = load_rnn_model_and_scaler()

# --- Lista de Tickers IBEX 35 y sus nombres de empresa ---
IBEX35_TICKERS_APP = {
    "ACX.MC": "Acerinox",
    "ACS.MC": "ACS",
    "AENA.MC": "Aena",
    "ALM.MC": "Almirall",
    "AMS.MC": "Amadeus",
    "BBVA.MC": "BBVA",
    "BKT.MC": "Bankinter",
    "CABK.MC": "CaixaBank",
    "CLNX.MC": "Cellnex",
    "COL.MC": "Colonial",
    "ELE.MC": "Endesa",
    "ENG.MC": "Enagás",
    "FER.MC": "Ferrovial",
    "GRF.MC": "Grifols",
    "IAG.MC": "IAG",
    "IBE.MC": "Iberdrola",
    "IDR.MC": "Indra",
    "ITX.MC": "Inditex",
    "MAP.MC": "Mapfre",
    "MEL.MC": "Meliá Hotels",
    "MRL.MC": "Merlin Properties",
    "MTS.MC": "ArcelorMittal",
    "NTGY.MC": "Naturgy",
    "RED.MC": "Red Eléctrica (Redeia)",
    "REP.MC": "Repsol",
    "ROVI.MC": "Laboratorios Rovi",
    "SAB.MC": "Banco Sabadell",
    "SAN.MC": "Banco Santander",
    "SCYR.MC": "Sacyr",
    "SLR.MC": "Solaria Energía",
    "TEF.MC": "Telefónica",
    "UNI.MC": "Unicaja Banco"
}

# Obtener solo los tickers para el selectbox
TICKER_OPTIONS = list(IBEX35_TICKERS_APP.keys())

# --- Función para obtener datos históricos (Cacheada para Rendimiento) ---
@st.cache_data(ttl=3600)
def get_historical_data(ticker_symbol, start_date, end_date):
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No se encontraron datos para {ticker_symbol} en el rango seleccionado. Por favor, ajusta las fechas.")
            return None
        return data
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker_symbol} de Yahoo Finance: {e}")
        return None

# --- Función para preparar los datos para la predicción ---
def prepare_data_for_prediction(data, scaler_obj, look_back):
    """
    Prepara los datos de cierre históricos para la predicción de la RNN.
    Escala los datos y crea la secuencia de entrada.
    """
    # Asegúrate de que haya suficientes datos para LOOK_BACK + 1 para crear al menos una secuencia de X e y
    if len(data) < look_back + 1: # Añadimos +1 aquí porque necesitamos 'look_back' para X y el siguiente para 'y' (aunque aquí 'y' no se usa, es para consistencia)
        st.warning(f"No hay suficientes datos históricos ({len(data)} puntos) para crear una secuencia de entrada con look_back={look_back}. Se necesitan al menos {look_back + 1} días de datos de cierre para una predicción significativa.")
        return None, None
    
    # Nos aseguramos de usar solo la columna 'Close' para la transformación
    scaled_data = scaler_obj.transform(data['Close'].values.reshape(-1, 1))
    
    # Tomar la última 'look_back' secuencia de datos para la predicción del día siguiente
    X_predict = scaled_data[-look_back:].reshape(1, look_back, 1) # Reshape para Keras [1, timesteps, features]
    
    return X_predict, scaled_data

# --- Funciones para la voz de Detective Pixel ---
def text_to_speech(text):
    if text:
        try:
            tts = gTTS(text=text, lang='es')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            st.audio(fp, format='audio/mp3', start_time=0)
        except Exception as e:
            st.error(f"Error al generar audio: {e}")

# --- Funciones para la interacción con OpenAI (Detective Pixel) ---
def get_detective_pixel_response(prompt_text):
    if client:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres Detective Pixel, un detective de IA experto en análisis de datos y predicción de mercados. Tu tono es entusiasta, curioso y un poco misterioso, como si estuvieras resolviendo un caso. Siempre anímate al usuario y usa analogías de detective."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=100,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"No pude obtener una respuesta de Detective Pixel (OpenAI): {e}. Usaré un mensaje predeterminado.")
            return None
    return None

def speak_as_detective_pixel(message_key, default_message, prompt_for_ai=None):
    if st.session_state.get(message_key) == default_message and not prompt_for_ai:
        return

    if prompt_for_ai and client:
        response_text = get_detective_pixel_response(prompt_for_ai)
        if response_text:
            st.info(f"**Detective Pixel:** {response_text}")
            text_to_speech(response_text)
            st.session_state[message_key] = response_text
            return
    
    st.info(f"**Detective Pixel:** {default_message}")
    text_to_speech(default_message)
    st.session_state[message_key] = default_message

# --- Inicialización de st.session_state para el juego ---
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'current_level' not in st.session_state:
    st.session_state.current_level = 1
if 'consecutive_hits' not in st.session_state:
    st.session_state.consecutive_hits = 0
if 'game_mode' not in st.session_state:
    st.session_state.game_mode = 'demo'
if 'current_ticker_challenge' not in st.session_state:
    st.session_state.current_ticker_challenge = None
if 'predicted_price' not in st.session_state:
    st.session_state.predicted_price = None
if 'actual_price' not in st.session_state:
    st.session_state.actual_price = None
if 'user_guess' not in st.session_state:
    st.session_state.user_guess = None
if 'dp_message_challenge' not in st.session_state:
    st.session_state.dp_message_challenge = ""
if 'dp_message_level_up' not in st.session_state:
    st.session_state.dp_message_level_up = ""
if 'dp_message_correct' not in st.session_state:
    st.session_state.dp_message_correct = ""
if 'dp_message_incorrect' not in st.session_state:
    st.session_state.dp_message_incorrect = ""

# --- Helper para resetear estado del juego ---
def reset_game_state_full():
    st.session_state.game_started = False
    st.session_state.current_level = 1
    st.session_state.consecutive_hits = 0
    st.session_state.game_mode = 'demo'
    st.session_state.current_ticker_challenge = None
    st.session_state.predicted_price = None
    st.session_state.actual_price = None
    st.session_state.user_guess = None
    st.session_state.dp_message_challenge = ""
    st.session_state.dp_message_level_up = ""
    st.session_state.dp_message_correct = ""
    st.session_state.dp_message_incorrect = ""

def reset_game_data_for_new_challenge():
    st.session_state.game_mode = 'game'
    st.session_state.current_ticker_challenge = None
    st.session_state.predicted_price = None
    st.session_state.actual_price = None
    st.session_state.user_guess = None
    st.session_state.dp_message_challenge = ""
    st.session_state.dp_message_correct = ""
    st.session_state.dp_message_incorrect = ""
    st.session_state.dp_message_level_up = ""

def start_new_challenge():
    reset_game_data_for_new_challenge()
    # Seleccionar un ticker aleatorio que no sea el mismo que el último si es posible
    available_tickers = [t for t in TICKER_OPTIONS if t != st.session_state.current_ticker_challenge]
    st.session_state.current_ticker_challenge = random.choice(available_tickers) if available_tickers else random.choice(TICKER_OPTIONS)
    
    
# --- Títulos y Descripciones Generales ---
col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Predicción de Precios de Acciones con RNNinja")
    st.markdown("""
    ¡Hola, jóvenes **Maestros de Redes Neuronales**!

    ¿Están listos para un nuevo desafío con **RNNinja**? Aquí vamos a usar los poderes de las **Redes Neuronales Recurrentes** para hacer algo genial: ¡**predecir los precios de las acciones**!

    Imagina que las acciones son como una **secuencia secreta de movimientos** en un tablero de juego. **RNNinja** ha estado estudiando estos movimientos, aprendiendo de cómo se comportan los precios a lo largo del tiempo. Ha visto muchísimas acciones del **IBEX 35**, ¡como si hubiera analizado cientos de partidas!

    Ahora, con su entrenamiento, **RNNinja** puede usar lo que ha aprendido para intentar adivinar cuál será el **próximo movimiento** o, en nuestro caso, ¡el **precio de cierre** de una acción!

    ¡Es como si **RNNinja** tuviera una **bola de cristal de datos** que le ayuda a ver el futuro de las secuencias! ¡Pero recuerden, es un juego y las predicciones son como pistas, no magia!
    """)
with col2:
    if lottie_brain_ai:
        st_lottie(
            lottie_brain_ai,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=200,
            width=200,
            key="brain_ai_lottie",
        )
    else:
        st.warning("Lottie animation 'brain_ai.json' could not be loaded.")

st.markdown("---")

# Display the neuron network animation below the main introduction
if lottie_neuron_network:
    st.markdown("### ¿Cómo funciona RNNinja?")
    col_lottie, col_text = st.columns([0.3, 0.7])
    with col_lottie:
        st_lottie(
            lottie_neuron_network,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=150,
            width=150,
            key="neuron_network_lottie",
        )
    with col_text:
        st.markdown("""
        RNNinja utiliza una **Red Neuronal Recurrente (RNN)**, una especie de cerebro digital especializado en secuencias.
        Piensa en ella como un detective que no solo ve la foto del momento, sino que recuerda toda la película.
        Cada vez que ve un nuevo dato de precio, lo compara con los anteriores para entender el patrón y predecir lo que viene.
        """)
else:
    st.warning("Lottie animation 'neuron_network.json' could not be loaded.")

st.markdown("---")


# --- SIDEBAR para controles generales del modo ---
st.sidebar.title("Modos de la Aplicación")
mode_selection = st.sidebar.radio(
    "Selecciona un modo:",
    ("Demostración del Modelo", "Juego de Detective Pixel")
)

if mode_selection == "Demostración del Modelo":
    st.session_state.game_mode = 'demo'
    st.sidebar.info("Estás en el modo de demostración. Explora el rendimiento del modelo libremente.")
elif mode_selection == "Juego de Detective Pixel":
    st.session_state.game_mode = 'game'
    st.sidebar.info("¡Prepárate para ser Detective Pixel! Tu misión: predecir movimientos de precios.")
    if st.sidebar.button("Reiniciar Juego Completo", key="reset_game_sidebar"):
        reset_game_state_full()
        st.rerun()

# --- DEMONSTRATION MODE ---
if st.session_state.game_mode == 'demo':
    st.header("Modo Demostración: Prueba el Modelo RNNinja")
    st.markdown("Aquí puedes seleccionar una acción del IBEX 35 y ver la predicción de RNNinja para el próximo día de cierre bursátil, basándose en los datos históricos recientes.")

    with st.expander("Instrucciones para el Modo Demostración", expanded=False):
        st.markdown("""
        1.  **Selecciona una Acción:** Elige una de las empresas del IBEX 35 de la lista desplegable.
        2.  **Define el Rango de Fechas:** Selecciona una fecha de inicio y una fecha de fin para los datos históricos que se utilizarán. Asegúrate de que haya al menos 11 días de datos para que el modelo pueda hacer una predicción (ya que `LOOK_BACK` es 10).
        3.  **Haz la Predicción:** Haz clic en el botón "Predecir Próximo Cierre".
        4.  **Analiza los Resultados:** Verás el precio de cierre real del último día y la predicción del modelo para el día siguiente.
        """)

    st.subheader("Opciones de Predicción y Visualización")
    col_sel, col_dates = st.columns([1, 2])

    with col_sel:
        selected_ticker_demo = st.selectbox(
            "Selecciona un ticker del IBEX 35:",
            TICKER_OPTIONS,
            key='demo_ticker_select',
            format_func=lambda x: f"{x} - {IBEX35_TICKERS_APP.get(x, 'Desconocido')}"
        )
        st.info(f"Empresa seleccionada: **{IBEX35_TICKERS_APP.get(selected_ticker_demo, 'Desconocido')}**")


    with col_dates:
        today = pd.to_datetime('today').date()
        default_start_date = today - pd.DateOffset(years=2)
        date_range = st.date_input(
            "Selecciona el rango de fechas para los datos históricos:",
            value=(default_start_date, today),
            key='demo_date_range'
        )

        start_date_demo = date_range[0]
        end_date_demo = date_range[1]

    if st.button("Realizar Predicción y Ver Historial", key="demo_predict_button"):
        if selected_ticker_demo and start_date_demo and end_date_demo:
            if start_date_demo >= end_date_demo:
                st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
            else:
                with st.spinner(f"Descargando y preparando datos para {selected_ticker_demo} ({IBEX35_TICKERS_APP.get(selected_ticker_demo, 'Desconocido')})..."):
                    historical_data_demo = get_historical_data(selected_ticker_demo, start_date_demo, end_date_demo)

                if historical_data_demo is not None and not historical_data_demo.empty:
                    # Convertir a float explícitamente para evitar TypeError en f-string
                    last_known_price_demo = float(historical_data_demo['Close'].iloc[-1])
                    
                    # Explicación clara de la predicción
                    with st.expander("¿Cómo se genera esta predicción?", expanded=False):
                        st.markdown(f"""
                        1.  **Recopilación de datos**: Primero, esta aplicación descarga automáticamente los **últimos datos de cierre** de la acción seleccionada (`yfinance`) hasta el día de hoy.
                        2.  **Preparación de la secuencia**: El modelo RNN (Red Neuronal Recurrente) no predice un valor aislado, sino que aprende de secuencias. Para predecir el **siguiente día hábil**, el modelo toma los **últimos `LOOK_BACK` ({LOOK_BACK} en este caso) días** de los datos históricos.
                        3.  **Normalización**: Estos datos se transforman usando un "escalador" (`MinMaxScaler`) que fue entrenado junto con el modelo. Esto asegura que los números estén en un rango que la red neuronal entiende mejor.
                        4.  **Predicción de la RNN**: La secuencia de `LOOK_BACK` días normalizados se introduce en el modelo RNN. El modelo, basado en los patrones que aprendió durante su entrenamiento, predice el precio de cierre para el siguiente día en la secuencia.
                        5.  **Desnormalización**: La predicción del modelo está en formato normalizado, por lo que se "desescala" de nuevo a la moneda real (euros en este caso) para que sea comprensible.
                        
                        Por lo tanto, la predicción que verás a continuación es siempre para el **próximo día hábil** después de los datos más recientes disponibles.
                        """)


                    X_input_for_prediction_demo, all_scaled_data_demo = prepare_data_for_prediction(historical_data_demo, scaler, LOOK_BACK)

                    if X_input_for_prediction_demo is not None:
                        predicted_scaled_price_demo = model.predict(X_input_for_prediction_demo)[0][0]
                        # Asegurar que inverse_transform recibe la forma correcta [[valor]]
                        predicted_price_demo = float(scaler.inverse_transform([[predicted_scaled_price_demo]])[0][0])

                        # --- INICIO: LÍNEAS DE DEPURACIÓN (AÑADIDAS AQUÍ) ---
                        st.markdown("---")
                        st.subheader("Pistas de Detective Pixel (Depuración):")
                        st.write(f"**scaler.n_features_in_:** `{scaler.n_features_in_}`")
                        st.write(f"**scaler.data_min_:** `{scaler.data_min_}`")
                        st.write(f"**scaler.data_max_:** `{scaler.data_max_}`")
                        st.write(f"**Últimos {LOOK_BACK} precios de cierre históricos (antes de escalar):** `{historical_data_demo['Close'].iloc[-LOOK_BACK:].values.tolist()}`")
                        st.write(f"**Últimos {LOOK_BACK} precios de cierre históricos (ESCALADOS, entrada a la RNN):** `{X_input_for_prediction_demo.flatten().tolist()}`")
                        st.write(f"**Predicción del modelo (ESCALADA - valor raw de la RNN):** `{predicted_scaled_price_demo}`")
                        st.write(f"**Array dummy usado para desescalar (¡AHORA YA NO SE USA PARA UN SOLO VALOR!):** `[[{predicted_scaled_price_demo}]]`") # Texto actualizado
                        st.markdown("---")
                        # --- FIN: LÍNEAS DE DEPURACIÓN ---

                        st.success(f"Predicción completada para {selected_ticker_demo} ({IBEX35_TICKERS_APP.get(selected_ticker_demo, 'Desconocido')})!")
                        st.subheader(f"Predicción del Precio de Cierre para el Próximo Día Hábil de {selected_ticker_demo}:")
                        st.metric(label="Precio Predicho", value=f"{predicted_price_demo:.2f} €", delta=f"{predicted_price_demo - last_known_price_demo:.2f} €")
                        
                        st.info(f"**Nota Importante:** El precio predicho de **€{predicted_price_demo:.2f}** es para el **siguiente día hábil** (el primer día de trading después de {historical_data_demo.index[-1].strftime('%Y-%m-%d')}).")
                        
                    else:
                        st.error("No se pudieron preparar los datos para la predicción. Esto puede ocurrir si no hay suficientes datos históricos para el 'look_back' configurado o si el rango de fechas es demasiado pequeño.")
                else:
                    st.error("No se pudieron obtener datos históricos para la acción y el rango de fechas seleccionados. Por favor, verifica el ticker y el rango de fechas.")

    st.divider() # Separador visual

    st.header("Fuentes de Noticias Clave para Inversores")
    st.markdown("""
    Para tomar decisiones de inversión informadas, la predicción de precios es solo una parte.
    Es crucial complementar el análisis técnico con el **análisis fundamental** y estar al tanto
    de las **noticias económicas y empresariales** más recientes.

    Aunque esta aplicación no puede procesar y resumir noticias en tiempo real,
    aquí te presentamos algunas de las fuentes más relevantes y consultadas por los inversores
    para seguir la actualidad del mercado y de las empresas:
    """)

    col_es, col_int = st.columns(2)

    with col_es:
        st.subheader("🇪🇸 Fuentes en Español (España/Latam):")
        st.markdown("""
        * **[Expansión](https://www.expansion.com/)**: Uno de los principales periódicos económicos de España, con amplia cobertura de empresas cotizadas, mercados y macroeconomía.
        * **[Cinco Días](https://cincodias.elpais.com/)**: Periódico económico de El País, ofrece análisis, noticias de empresas y mercados.
        * **[El Economista](https://www.eleconomista.es/)**: Noticias económicas, financieras y de bolsa con un enfoque en el mercado español.
        * **[Bolsamanía](https://www.bolsamania.com/)**: Portal especializado en bolsa, análisis técnico y fundamental, noticias de última hora sobre empresas y cotizaciones.
        * **[Investing.com (España)](https://es.investing.com/)**: Portal global con noticias, datos de mercado, análisis y herramientas para inversores.
        """)
    
    with col_int:
        st.subheader("🌍 Fuentes Internacionales (Inglés - Imprescindibles):") # Icono cambiado para ser más genérico
        st.markdown("""
        * **[Financial Times](https://www.ft.com/)**: Reconocido globalmente por su profundidad en noticias financieras y económicas. (Puede requerir suscripción)
        * **[The Wall Street Journal](https://www.wsj.com/)**: Cobertura exhaustiva de mercados financieros, empresas y economía a nivel mundial. (Puede requerir suscripción)
        * **[Bloomberg](https://www.bloomberg.com/markets)**: Noticias en tiempo real, datos de mercado y análisis, muy utilizado por profesionales.
        * **[Reuters](https://www.reuters.com/)**: Agencia de noticias global que ofrece información económica y de mercados muy rápidamente.
        * **[CNBC](https://www.cnbc.com/)**: Canal de noticias financieras con información en vivo del mercado y comentarios de expertos.
        """)
    
    st.markdown("""
    ---
    **Consejo de Detective Pixel:** "¡Un buen detective siempre busca todas las pistas!
    Combinar la predicción de patrones (`LOOK_BACK`) con las últimas noticias te dará una visión
    más completa para descifrar el mercado. ¡Siempre verifica múltiples fuentes!"
    """)

# --- GAME MODE (Detective Pixel) ---
elif st.session_state.game_mode == 'game':
    st.header("Modo Juego: ¡Conviértete en Detective Pixel!")
    st.markdown("""
    ¡Bienvenido, aspirante a detective! Tu misión es simple pero desafiante: **Predecir si el precio de una acción subirá o bajará**
    después de la predicción de RNNinja. Cada acierto consecutivo te acerca a ser un verdadero Maestro de Datos.
    """)

    if not st.session_state.game_started:
        st.info("Haz clic en 'Iniciar Desafío' para que Detective Pixel te dé tu primer caso.")
        if st.button("Iniciar Desafío", key="start_challenge", help="Comienza una nueva partida del juego de predicción."):
            st.session_state.game_started = True
            reset_game_data_for_new_challenge()
            start_new_challenge()
            st.rerun()
    else:
        st.subheader(f"Nivel Actual: {st.session_state.current_level} | Aciertos Consecutivos: {st.session_state.consecutive_hits}")

        if not st.session_state.current_ticker_challenge:
            start_new_challenge()

        current_ticker_name = IBEX35_TICKERS_APP.get(st.session_state.current_ticker_challenge, st.session_state.current_ticker_challenge)

        if not st.session_state.predicted_price:
            speak_as_detective_pixel(
                'dp_message_challenge',
                f"¡Caso nuevo, {current_ticker_name}! Necesito que investigues este caso. Vamos a ver los datos para descifrar el futuro de {current_ticker_name}. ¿Crees que subirá o bajará?",
                f"Genera una frase de detective de IA entusiasta y curiosa para iniciar un desafío de predicción de precios para la acción {current_ticker_name}. Anímale al usuario a ver si subirá o bajará."
            )
            st.markdown(f"**Empresa a investigar:** **{current_ticker_name}** ({st.session_state.current_ticker_challenge})")

            # Definir un rango de fechas para el desafío (ej. últimos 60 días para el cálculo y predicción)
            challenge_end_date = pd.Timestamp.today().normalize()
            challenge_start_date = challenge_end_date - pd.Timedelta(days=60) # Necesitamos suficientes días para LOOK_BACK

            with st.spinner(f"Detective Pixel está reuniendo pistas... descargando datos para {current_ticker_name}..."):
                challenge_data = get_historical_data(st.session_state.current_ticker_challenge, challenge_start_date, challenge_end_date)

            if challenge_data is not None and not challenge_data.empty:
                X_predict_challenge, _ = prepare_data_for_prediction(challenge_data, scaler, LOOK_BACK)

                if X_predict_challenge is not None:
                    try:
                        predicted_scaled_price = model.predict(X_predict_challenge)[0][0]
                        # Asegurar que inverse_transform recibe la forma correcta [[valor]]
                        predicted_price_unscaled = float(scaler.inverse_transform([[predicted_scaled_price]])[0][0])
                        
                        st.session_state.predicted_price = predicted_price_unscaled
                        # Convertir a float explícitamente el último precio real
                        st.session_state.actual_price = float(challenge_data['Close'].iloc[-1])

                        st.write(f"**Último precio de cierre real conocido de {current_ticker_name}:** **€{st.session_state.actual_price:.2f}**")
                        st.write(f"**RNNinja ha predicho que el próximo precio de cierre será:** **€{st.session_state.predicted_price:.2f}**")

                        # No hay visualización de gráfica en este modo

                        st.markdown("---")
                        st.markdown("### ¡Tu Turno, Detective!")
                        st.markdown(f"Dada la predicción de RNNinja (del último precio real de **€{st.session_state.actual_price:.2f}** al predicho de **€{st.session_state.predicted_price:.2f}**),")
                        
                        guess_col1, guess_col2, guess_col3 = st.columns([1,1,1])
                        with guess_col1:
                            if st.button("📈 Subirá", key="guess_up", help="Predice que el precio subirá."):
                                st.session_state.user_guess = "up"
                                st.rerun()
                        with guess_col2:
                            if st.button("📉 Bajará", key="guess_down", help="Predice que el precio bajará."):
                                st.session_state.user_guess = "down"
                                st.rerun()
                        with guess_col3:
                            if st.button("↔️ Se Mantendrá", key="guess_same", help="Predice que el precio se mantendrá estable."):
                                st.session_state.user_guess = "same"
                                st.rerun()

                    except Exception as e:
                        st.error(f"Error en la predicción del desafío: {e}. Por favor, reinicia el juego.")
                        reset_game_state_full()
                else:
                    st.warning("No hay suficientes datos para este desafío. Seleccionando un nuevo caso...")
                    start_new_challenge()
                    st.rerun()
            else:
                st.warning("No se pudieron obtener datos para el desafío. Seleccionando un nuevo caso...")
                start_new_challenge()
                st.rerun()

        # Check user's guess and display result
        if st.session_state.user_guess:
            # Lógica de "se mantendrá": un rango porcentual alrededor del precio actual
            predicted_movement = "same"
            # Define un umbral pequeño para considerar que sube o baja significativamente (ej. 0.5%)
            up_threshold = st.session_state.actual_price * 1.005
            down_threshold = st.session_state.actual_price * 0.995

            if st.session_state.predicted_price > up_threshold:
                predicted_movement = "up"
            elif st.session_state.predicted_price < down_threshold:
                predicted_movement = "down"
            
            is_correct = (st.session_state.user_guess == predicted_movement)

            st.markdown("---")
            st.subheader("¡Resultado del Caso!")
            
            # Display result with emojis and clear message
            if is_correct:
                st.session_state.consecutive_hits += 1
                st.success(f"✅ ¡CORRECTO! Tu suposición fue **{st.session_state.user_guess.capitalize()}** y la predicción de RNNinja indica que el precio **{predicted_movement}**.")
                speak_as_detective_pixel(
                    'dp_message_correct',
                    f"¡Elementar, mi querido colega! ¡Acertaste! Parece que mis pistas fueron útiles. Llevas {st.session_state.consecutive_hits} aciertos consecutivos. ¡Sigamos con el caso!",
                    f"Felicita al usuario por acertar una predicción de mercado. Menciona que es un 'caso resuelto' y que lleva {st.session_state.consecutive_hits} aciertos consecutivos. Anímale para el siguiente caso."
                )
                
                if st.session_state.consecutive_hits % 3 == 0: # Nivel cada 3 aciertos
                    st.session_state.current_level += 1
                    st.balloons() # Pequeña animación de celebración
                    st.success(f"🎉 ¡FELICIDADES, DETECTIVE! Has descifrado suficientes casos para subir al Nivel **{st.session_state.current_level}**.")
                    speak_as_detective_pixel(
                        'dp_message_level_up',
                        f"¡Felicidades, Detective! Has descifrado suficientes casos para subir al Nivel {st.session_state.current_level}. ¡Los misterios se hacen más interesantes!",
                        f"Felicita al usuario por subir al Nivel {st.session_state.current_level} en el juego de predicción de mercados, usando un tono de detective que sugiere que los casos se vuelven más complejos e interesantes."
                    )
            else:
                st.session_state.consecutive_hits = 0
                st.error(f"❌ ¡INCORRECTO! Tu suposición fue **{st.session_state.user_guess.capitalize()}** pero la predicción de RNNinja indica que el precio **{predicted_movement}**.")
                speak_as_detective_pixel(
                    'dp_message_incorrect',
                    f"¡Mmm, un pequeño desvío en el caso! No te preocupes, incluso los mejores detectives tienen días así. Hemos vuelto al Nivel 1. ¡Analicemos de nuevo las pistas!",
                    f"Consuela al usuario por fallar una predicción de mercado, diciéndole que es normal, que 'hemos vuelto al Nivel 1' y que debe 'analizar de nuevo las pistas' en tono de detective."
                )
            
            st.markdown(f"**Tu suposición:** **{st.session_state.user_guess.capitalize()}**")
            st.markdown(f"**Movimiento según RNNinja:** **{predicted_movement.capitalize()}**")
            st.markdown(f"**Precio Real Anterior:** **€{st.session_state.actual_price:.2f}**")
            st.markdown(f"**Precio Predicho por RNNinja:** **€{st.session_state.predicted_price:.2f}**")

            st.divider() # Separador visual

            if st.button("Siguiente Desafío", key="next_challenge", help="Empieza un nuevo desafío para seguir jugando."):
                start_new_challenge()
                st.rerun()


# --- Sección del Juego de Detective Pixel ---
elif st.session_state.game_mode == 'game':
    st.header("¡Juega y Aprende con Detective Pixel sobre RNNs!")
    st.markdown("""
    ¡Hola! Soy Detective Pixel, tu amigo detective de la IA. ¿Listo para desentrañar los secretos
    de las secuencias temporales y cómo las RNNs las interpretan para predecir el futuro del mercado?
    """)

    # Mostrar el contador de aciertos consecutivos y nivel
    col_hits, col_level = st.columns(2)
    with col_hits:
        st.metric(label="Aciertos consecutivos", value=st.session_state.consecutive_hits)
    with col_level:
        st.metric(label="Nivel Actual", value=st.session_state.current_level)

    # Botones de inicio del juego
    if not st.session_state.game_started:
        st.subheader("¡Elige tu nivel de desafío!")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Nivel Novato (1 acierto por subir)"):
                st.session_state.game_started = True
                st.session_state.consecutive_hits = 0
                st.session_state.current_level = 1
                start_new_challenge()
                st.rerun()
        with col2:
            if st.button("Nivel Medio (3 aciertos por subir)"):
                st.session_state.game_started = True
                st.session_state.current_level = 2
                st.session_state.consecutive_hits = 0
                start_new_challenge()
                st.rerun()
        with col3:
            if st.button("Nivel Avanzado (5 aciertos por subir)"):
                st.session_state.game_started = True
                st.session_state.current_level = 3
                st.session_state.consecutive_hits = 0
                start_new_challenge()
                st.rerun()

    else: # El juego ha empezado
        if st.session_state.current_ticker_challenge is None:
            start_new_challenge()

        # --- Lógica para el desafío de predicción ---
        if st.session_state.game_state == 'ask_prediction':
            
            ticker_name = IBEX35_TICKERS_APP.get(st.session_state.current_ticker_challenge, 'Desconocido')
            prompt_ai = f"El usuario está en el nivel {st.session_state.current_level}. Dale un desafío de predicción para la acción {st.session_state.current_ticker_challenge} ({ticker_name}). Pídele que adivine si el precio de cierre subirá o bajará. Usa un lenguaje de detective, entusiasta y un poco misterioso."
            speak_as_detective_pixel(
                "dp_message_challenge",
                f"¡Atención, aspirante a detective! Nuestro caso de hoy es la acción **{st.session_state.current_ticker_challenge}** ({ticker_name}) del IBEX 35. ¿Cuál es tu primera hipótesis? Basado en su comportamiento reciente, ¿crees que el precio de cierre de mañana **subirá o bajará**?",
                prompt_ai
            )

            st.subheader(f"Desafío de Predicción para: **{st.session_state.current_ticker_challenge}** ({ticker_name})")

            st.markdown("---")
            st.subheader(f"Historial de Precios para {st.session_state.current_ticker_challenge} ({ticker_name})")
            
            today = pd.to_datetime('today').date()
            game_end_date = today 
            game_start_date = today - pd.DateOffset(years=1)

            historical_data_game = get_historical_data(st.session_state.current_ticker_challenge, game_start_date, game_end_date)

            if historical_data_game is not None and not historical_data_game.empty:
                fig_game = go.Figure()
                fig_game.add_trace(go.Scatter(x=historical_data_game.index, y=historical_data_game['Close'],
                                         mode='lines', name='Precio de Cierre Histórico'))
                fig_game.update_layout(
                    title=f"Historial de Cierre de {st.session_state.current_ticker_challenge} ({ticker_name})",
                    xaxis_title="Fecha",
                    yaxis_title="Precio de Cierre (€)",
                    hovermode="x unified",
                    height=400
                )
                st.plotly_chart(fig_game, use_container_width=True)
                st.info("Analiza la tendencia reciente para hacer tu hipótesis, detective.")
            else:
                st.warning("No se pudieron cargar datos históricos para este desafío. ¡Intentemos con otro caso!")
                start_new_challenge()
                st.rerun()

            st.markdown("---")
            st.write("Tu misión: ¿El precio de cierre de **Mañana** será más alto o más bajo que el **último precio conocido**?")
            user_guess = st.radio(
                "Haz tu predicción:",
                ("Subirá", "Bajará"),
                key="user_direction_guess"
            )

            if st.button("¡Revelar el Misterio!", key="reveal_button"):
                st.session_state.user_guess = user_guess
                with st.spinner(f"Detective Pixel está investigando los datos de {st.session_state.current_ticker_challenge} ({ticker_name})..."):
                    historical_data_for_pred = get_historical_data(st.session_state.current_ticker_challenge, game_start_date, game_end_date)

                    if historical_data_for_pred is not None and not historical_data_for_pred.empty:
                        st.session_state.actual_price = historical_data_for_pred['Close'].iloc[-1]
                        
                        X_input_for_prediction, all_scaled_data = prepare_data_for_prediction(historical_data_for_pred, scaler, LOOK_BACK)

                        if X_input_for_prediction is not None:
                            predicted_scaled_price = model.predict(X_input_for_prediction)[0][0]
                            dummy_array = np.zeros((1, scaler.n_features_in_))
                            dummy_array[0, 0] = predicted_scaled_price
                            predicted_price = scaler.inverse_transform(dummy_array)[0][0]
                            
                            st.session_state.predicted_price = predicted_price

                            st.session_state.game_state = 'show_result'
                            st.rerun()
                        else:
                            st.error("Detective Pixel no encontró suficientes pistas (datos) para hacer una predicción. Intentemos con otro caso.")
                            start_new_challenge()
                            st.rerun()
                    else:
                        st.error("Detective Pixel no pudo obtener datos históricos para este caso. Intentemos con otra acción.")
                        start_new_challenge()
                        st.rerun()

        # --- Lógica para mostrar el resultado y el gráfico ---
        elif st.session_state.game_state == 'show_result':
            ticker_name = IBEX35_TICKERS_APP.get(st.session_state.current_ticker_challenge, 'Desconocido')
            st.subheader(f"¡Resultados del Caso **{st.session_state.current_ticker_challenge}** ({ticker_name})!")
            
            last_known_price = st.session_state.actual_price
            predicted_price = st.session_state.predicted_price
            user_guess = st.session_state.user_guess

            st.write(f"Último precio de cierre conocido: **{last_known_price:.2f} €**")
            st.metric(label="Precio Predicho para el Próximo Día Hábil", value=f"{predicted_price:.2f} €")

            prediction_direction = "Subirá" if predicted_price >= last_known_price else "Bajará"
            
            st.write(f"Tu hipótesis: **{user_guess}**")
            st.write(f"La predicción de la RNN indica que el precio **{prediction_direction}**.")

            correct_guess = (user_guess == prediction_direction)

            level_hit_threshold = 0
            if st.session_state.current_level == 1:
                level_hit_threshold = 1 
            elif st.session_state.current_level == 2:
                level_hit_threshold = 3 
            elif st.session_state.current_level == 3:
                level_hit_threshold = 5 

            if correct_guess:
                st.success("¡Felicidades, detective! ¡Tu hipótesis fue correcta! 🕵️‍♀️")
                st.session_state.consecutive_hits += 1

                if st.session_state.consecutive_hits >= level_hit_threshold:
                    if st.session_state.current_level < 3:
                        st.session_state.current_level += 1
                        st.session_state.consecutive_hits = 0
                        speak_as_detective_pixel(
                            "dp_message_level_up",
                            f"¡Excelente trabajo, detective! Has ascendido a Nivel {st.session_state.current_level}. ¡Tu agudeza es impresionante!",
                            f"El usuario ha acertado y ha subido al nivel {st.session_state.current_level}. Dale un mensaje de felicitación por su ascenso y anímale a seguir."
                        )
                    else:
                         speak_as_detective_pixel(
                            "dp_message_correct",
                            "¡Increíble! Has descifrado las señales. ¡Un paso más cerca de ser un gran analista de datos!",
                            f"El usuario acertó la predicción para {st.session_state.current_ticker_challenge}. Felicítale de forma entusiasta y de detective. Menciona que sigue acumulando experiencia en el maestro."
                        )
                else:
                    speak_as_detective_pixel(
                        "dp_message_correct",
                        "¡Increíble! Has descifrado las señales. ¡Un paso más cerca de ser un gran analista de datos!",
                        f"El usuario acertó la predicción para {st.session_state.current_ticker_challenge}. Felicítale de forma entusiasta y de detective."
                    )
            else:
                st.error("¡Caso cerrado... por ahora! Parece que las pistas nos llevaron en una dirección diferente. ¡No te desanimes, sigue practicando!")
                st.session_state.consecutive_hits = 0
                speak_as_detective_pixel(
                    "dp_message_incorrect",
                    "¡Vaya! La trama se complica, ¿verdad? No siempre es fácil prever el siguiente giro. ¡Pero cada error es una pista para el futuro!",
                    f"El usuario no acertó la predicción para {st.session_state.current_ticker_challenge}. Anímale a seguir jugando y que el error es parte del aprendizaje, de forma empática."
                )

            st.markdown("---")
            st.subheader("Visualización del Caso Resuelto")

            today = pd.to_datetime('today').date()
            game_end_date = today 
            game_start_date = today - pd.DateOffset(years=1)
            historical_data_for_plot = get_historical_data(st.session_state.current_ticker_challenge, game_start_date, game_end_date)

            if historical_data_for_plot is not None and not historical_data_for_plot.empty:
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=historical_data_for_plot.index, y=historical_data_for_plot['Close'],
                                         mode='lines', name='Precio de Cierre Histórico'))

                last_date = historical_data_for_plot.index[-1]
                next_day_date = last_date + pd.Timedelta(days=1)
                while next_day_date.weekday() > 4:
                    next_day_date += pd.Timedelta(days=1)

                fig.add_trace(go.Scatter(x=[next_day_date], y=[predicted_price],
                                         mode='markers+text', name='Predicción Siguiente Día Hábil',
                                         marker=dict(size=10, color='red', symbol='circle'),
                                         text=[f"{predicted_price:.2f} €"],
                                         textposition="top center"))
                
                fig.add_trace(go.Scatter(x=[last_date], y=[last_known_price],
                                         mode='markers', name='Último Precio Conocido',
                                         marker=dict(size=10, color='blue', symbol='x')))

                fig.update_layout(
                    title=f"Precio de Cierre de {st.session_state.current_ticker_challenge} ({ticker_name}) y Predicción",
                    xaxis_title="Fecha",
                    yaxis_title="Precio de Cierre (€)",
                    hovermode="x unified",
                    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.5)'),
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No se pudieron cargar los datos históricos para la visualización del gráfico de resultados.")

            st.info(f"**Nota:** El precio predicho es para el **siguiente día hábil** (el primer día de trading después de la última fecha de datos disponibles).")
            st.info("Las predicciones de precios de acciones son inherentemente difíciles y están sujetas a la volatilidad del mercado. Este modelo es para fines demostrativos y no debe usarse como asesoramiento financiero.")
            
            if st.button("¡Siguiente Caso, Detective!", key="next_case_button"):
                start_new_challenge()
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Controles del Juego")
    if st.sidebar.button("Reiniciar Juego de Detective Pixel"):
        reset_game_state_full()
        st.rerun()
                        
# --- Sección de Chatbot de Juego con RNNinja (RNNs) ---
st.header("¡Juega y Aprende con RNNinja sobre RNNs!")
st.markdown("¡Saludos, aspirante a Maestro de Secuencias! Soy **RNNinja**, el experto sigiloso en Redes Neuronales Recurrentes. ¿Listo para afilar tus habilidades y dominar el arte de las secuencias temporales?")

if client: # Solo muestra el chatbot si la API está configurada
    # Inicialización de estados de sesión para el juego de RNN
    if "rnn_game_active" not in st.session_state:
        st.session_state.rnn_game_active = False
    if "rnn_game_messages" not in st.session_state:
        st.session_state.rnn_game_messages = []
    if "rnn_current_question" not in st.session_state:
        st.session_state.rnn_current_question = None
    if "rnn_current_options" not in st.session_state:
        st.session_state.rnn_current_options = {}
    if "rnn_correct_answer" not in st.session_state:
        st.session_state.rnn_correct_answer = None
    if "rnn_awaiting_next_game_decision" not in st.session_state:
        st.session_state.rnn_awaiting_next_game_decision = False
    if "rnn_game_needs_new_question" not in st.session_state:
        st.session_state.rnn_game_needs_new_question = False
    if "rnn_correct_streak" not in st.session_state:
        st.session_state.rnn_correct_streak = 0
    if "last_played_rnn_question" not in st.session_state:
        st.session_state.last_played_rnn_question = None

    # Prompt del sistema para el juego de RNN
    rnn_game_system_prompt = f"""
    Eres un **experto consumado en Procesamiento de Secuencias y Deep Learning**, con una especialización profunda en el diseño, entrenamiento y comprensión de las **Redes Neuronales Recurrentes (RNN)**. Comprendes a fondo sus fundamentos (recurrencia, estados ocultos, secuencias, problemas de gradiente), su arquitectura específica para datos secuenciales (series temporales, texto, audio), sus ventajas en el reconocimiento de patrones temporales y sus diversas aplicaciones prácticas en predicción de series temporales, Procesamiento de Lenguaje Natural (PLN) y reconocimiento de voz. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de las RNN mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**
    **Para asegurar la máxima variedad, aborda los conceptos de RNN desde diferentes ángulos: teoría, aplicación práctica, comparación entre arquitecturas, escenarios de resolución de problemas o incluso depuración de un caso hipotético. Procura combinar subtemas cuando sea posible para crear preguntas más complejas y únicas.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Tu RNN ha captado la secuencia! Excelente comprensión." o "Esa recurrencia necesita afinarse. Revisemos los estados ocultos."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "Una RNN es una arquitectura de red neuronal especialmente diseñada para procesar datos secuenciales, manteniendo una "memoria" de entradas anteriores a través de su estado oculto..."]
    [Pregunta para continuar, ej: "¿Listo para profundizar en las arquitecturas de memoria a largo plazo?" o "¿Quieres explorar cómo las RNNs manejan las dependencias largas?"]

    **Reglas adicionales para el Experto en Redes Neuronales Recurrentes:**
    * **Enfoque Riguroso en RNN:** Todas tus preguntas y explicaciones deben girar en torno a las Redes Neuronales Recurrentes. Cubre sus fundamentos (manejo de secuencias, memoria, bucles recurrentes), los componentes clave (estado oculto, pesos recurrentes), el proceso de propagación a través del tiempo, el **entrenamiento** (Backpropagation Through Time - BPTT), los **problemas de gradiente** (desvanecimiento y explosión), las soluciones (LSTM, GRU), el **sobreajuste** y técnicas de **regularización** (Dropout en capas recurrentes), y sus aplicaciones principales.
    * **¡VARIEDAD, VARIADAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de RNN que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General e Inspiración:** ¿Qué es una RNN? ¿Por qué son especiales para secuencias? Ejemplos de datos secuenciales.
        * **Recurrencia y Memoria:** Funcionamiento del estado oculto (hidden state), cómo se propaga la información a través del tiempo, el bucle recurrente.
        * **Backpropagation Through Time (BPTT):** Cómo se entrena una RNN.
        * **Problemas de Gradiente:** Desvanecimiento (vanishing gradients) y explosión (exploding gradients) de gradientes en RNNs largas.
        * **Soluciones a Problemas de Gradiente:**
            * **LSTM (Long Short-Term Memory):** Estructura de celda, puertas (input, forget, output), estado de celda.
            * **GRU (Gated Recurrent Unit):** Simplificación de LSTM, puertas (reset, update).
        * **Arquitecturas de RNN Típicas:** Capas recurrentes (SimpleRNN, LSTM, GRU), seguido de capas densas para la salida.
        * **Tipos de RNNs:** One-to-one, one-to-many, many-to-one, many-to-many (ej. encoder-decoder).
        * **Sobreajuste y Regularización:** Dropout en capas recurrentes.
        * **Ventajas de las RNNs:** Efectivas para dependencias temporales, manejo de secuencias de longitud variable.
        * **Limitaciones/Desafíos:** Dificultad para capturar dependencias a muy largo plazo (sin LSTM/GRU), alto costo computacional para secuencias muy largas.
        * **Aplicaciones Principales:** Predicción de series temporales (bolsa, clima), Procesamiento de Lenguaje Natural (traducción, generación de texto, análisis de sentimientos), reconocimiento de voz, generación de música.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.rnn_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Aprendiz de Secuencias – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea de que una máquina "recuerda" secuencias y ejemplos sencillos de lo que puede hacer una RNN (ej., predecir el siguiente número en una serie simple).
            * *Tono:* "Estás dando tus primeros pasos en el fascinante mundo donde las máquinas entienden el tiempo y las secuencias."
        * **Nivel 2 (Analista Temporal – 3-5 respuestas correctas):** Tono más técnico. Introduce los conceptos de **estado oculto** y cómo la información "fluye" a través del tiempo en una RNN simple. Preguntas sobre cómo se procesa una secuencia paso a paso.
            * *Tono:* "Tu habilidad para seguir la pista a los datos temporales está mejorando con cada paso recurrente."
        * **Nivel 3 (Arquitecto de Memoria – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en los detalles de los **problemas de gradiente** (desvanecimiento/explosión) y presenta las soluciones clave: **LSTM y GRU**. Preguntas sobre la función de las puertas en LSTM o cómo BPTT entrena la red.
            * *Tono:* "Tu comprensión profunda de las arquitecturas de memoria te permite diseñar soluciones robustas para las dependencias a largo plazo."
        * **Nivel Maestro (Especialista en Series Temporales – 9+ respuestas correctas):** Tono de **especialista en la vanguardia del Deep Learning aplicado a secuencias**. Preguntas sobre el diseño de arquitecturas complejas (Encoder-Decoder, atención para RNNs), el manejo de secuencias muy largas, la optimización de LSTMs/GRUs para tareas específicas, o las implicaciones de aplicar RNNs a datos muy ruidosos. Se esperan respuestas que demuestren una comprensión teórica y práctica robusta, incluyendo sus limitaciones y el estado del arte.
            * *Tono:* "Tu maestría en Redes Neuronales Recurrentes te permite no solo entender, sino también modelar y transformar el mundo de los datos secuenciales."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Predecir la siguiente palabra en una frase sencilla, reconocer una secuencia de notas musicales básica.
        * **Nivel 2:** Análisis de sentimiento en reseñas cortas, predicción de la demanda de productos en un corto plazo.
        * **Nivel 3:** Traducción automática de frases, generación de texto coherente, forecasting de precios de acciones o clima a medio plazo.
        * **Nivel Maestro:** Desarrollo de sistemas de diálogo conversacional avanzados, modelado de genomas para predecir propiedades, forecasting de series temporales financieras con alta frecuencia y múltiples características.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre REDES NEURONALES RECURRENTES (RNN), y asegúrate de que no se parezca a las anteriores.
    """

    # --- Funciones de Parsing para la respuesta de la API ---
    def parse_rnn_question_response(raw_text):
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

    def parse_rnn_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"
    
    # --- Función para establecer el nivel del juego ---
    def set_rnn_level(target_streak, level_name):
        st.session_state.rnn_correct_streak = target_streak
        st.session_state.rnn_game_active = True
        st.session_state.rnn_game_messages = []
        st.session_state.rnn_current_question = None
        st.session_state.rnn_current_options = {}
        st.session_state.rnn_correct_answer = None
        st.session_state.rnn_game_needs_new_question = True
        st.session_state.rnn_awaiting_next_game_decision = False
        st.session_state.rnn_game_messages.append({"role": "assistant", "content": f"¡Saludos, Guerrero de las Secuencias! ¡Has saltado directamente al **Nivel {level_name}**! Prepara tu mente para desafíos avanzados. ¡Aquí va tu primera misión!"})
        st.rerun()

    # --- Botones de inicio y salto de nivel ---
    col_game_buttons_rnn, col_level_up_buttons_rnn = st.columns([1, 2])

    with col_game_buttons_rnn:
        if st.button("¡Desafía a RNNinja!", key="start_rnn_game_button"):
            st.session_state.rnn_game_active = True
            st.session_state.rnn_game_messages = []
            st.session_state.rnn_current_question = None
            st.session_state.rnn_current_options = {}
            st.session_state.rnn_correct_answer = None
            st.session_state.rnn_game_needs_new_question = True
            st.session_state.rnn_awaiting_next_game_decision = False
            st.session_state.rnn_correct_streak = 0
            st.session_state.last_played_rnn_question = None # Resetear el último audio reproducido
            st.rerun()
    
    with col_level_up_buttons_rnn:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya dominas las sombras de las RNNs? ¡Demuéstralo! 👇</p>", unsafe_allow_html=True)
        col_lvl1_rnn, col_lvl2_rnn, col_lvl3_rnn = st.columns(3)
        with col_lvl1_rnn:
            if st.button("Subir a Nivel Iniciado (RNN)", key="level_up_medium_rnn"):
                set_rnn_level(3, "Iniciado")
        with col_lvl2_rnn:
            if st.button("Subir a Nivel Kage (RNN)", key="level_up_advanced_rnn"):
                set_rnn_level(6, "Kage")
        with col_lvl3_rnn:
            if st.button("👑 ¡Gran Maestro RNNinja! (RNN)", key="level_up_champion_rnn"):
                set_rnn_level(9, "Gran Maestro RNNinja")

    # --- Mostrar mensajes del chat ---
    for message in st.session_state.rnn_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Lógica del juego de preguntas ---
    if st.session_state.rnn_game_active:
        if st.session_state.rnn_current_question is None and st.session_state.rnn_game_needs_new_question and not st.session_state.rnn_awaiting_next_game_decision:
            with st.spinner("RNNinja está meditando tu próximo desafío..."):
                try:
                    # Preparar historial de mensajes para la API, incluyendo el prompt del sistema
                    rnn_game_messages_for_api = [{"role": "system", "content": rnn_game_system_prompt}]
                    # Añadir un historial limitado para que el LLM evite repetir preguntas
                    for msg in st.session_state.rnn_game_messages[-6:]: # Últimos 6 mensajes para contexto
                        if msg["role"] == "assistant" and msg["content"].startswith("**"):
                            rnn_game_messages_for_api.append({"role": "assistant", "content": f"PREGUNTA ANTERIOR: {msg['content'].splitlines()[0]}"})
                        elif msg["role"] == "user" and "MI RESPUESTA:" not in msg["content"]:
                            rnn_game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {msg['content']}"})

                    rnn_game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre QUÉ SON LAS RNNs siguiendo el formato exacto."})

                    rnn_response = client.chat.completions.create(
                        model="gpt-4o-mini", # Puedes ajustar el modelo según tus necesidades y límites
                        messages=rnn_game_messages_for_api,
                        temperature=0.8, # Creatividad del modelo (0.0 a 1.0)
                        max_tokens=300 # Límite de tokens para la respuesta de la pregunta
                    )
                    raw_rnn_question_text = rnn_response.choices[0].message.content
                    question, options, correct_answer_key = parse_rnn_question_response(raw_rnn_question_text)

                    if question:
                        st.session_state.rnn_current_question = question
                        st.session_state.rnn_current_options = options
                        st.session_state.rnn_correct_answer = correct_answer_key
                        st.session_state.rnn_game_needs_new_question = False
                        
                        question_content = f"**Nivel {int(st.session_state.rnn_correct_streak / 3) + 1} - Desafíos completados: {st.session_state.rnn_correct_streak}**\n\n**Pregunta de RNNinja:** {question}\n\n"
                        for k, v in options.items():
                            question_content += f"**{k})** {v}\n"
                        
                        st.session_state.rnn_game_messages.append({"role": "assistant", "content": question_content})
                        st.rerun()
                    else:
                        st.error("RNNinja no pudo generar un desafío válido. Intenta de nuevo.")
                        st.session_state.rnn_game_active = False
                        st.session_state.rnn_game_messages.append({"role": "assistant", "content": "RNNinja no pudo generar un desafío válido. Parece que hay un problema con el formato de la API. Por favor, reinicia el juego."})

                except Exception as e:
                    st.error(f"Error al comunicarse con la API de OpenAI para la pregunta: {e}")
                    st.session_state.rnn_game_active = False
                    st.session_state.rnn_game_messages.append({"role": "assistant", "content": "Lo siento, tengo un problema para conectar con mi dojo (la API). ¡Por favor, reinicia el juego!"})
                    st.rerun()

        # Reproducir audio de la pregunta
        if st.session_state.rnn_current_question and not st.session_state.rnn_awaiting_next_game_decision:
            if st.session_state.get('last_played_rnn_question') != st.session_state.rnn_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.rnn_correct_streak / 3) + 1}. Desafíos completados: {st.session_state.rnn_correct_streak}. Pregunta de RNNinja: {st.session_state.rnn_current_question}. Opción A: {st.session_state.rnn_current_options.get('A', '')}. Opción B: {st.session_state.rnn_current_options.get('B', '')}. Opción C: {st.session_state.rnn_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.audio(fp, format='audio/mp3', start_time=0)
                    st.session_state.last_played_rnn_question = st.session_state.rnn_current_question
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")

            # Formulario para la respuesta del usuario
            with st.form(key="rnn_game_form"):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu técnica de respuesta:")
                    user_answer = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.rnn_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.rnn_current_options[x]}",
                        key="rnn_answer_radio",
                        label_visibility="collapsed"
                    )
                submit_button = st.form_submit_button("¡Ejecutar Técnica!")

            if submit_button:
                st.session_state.rnn_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_answer}) {st.session_state.rnn_current_options[user_answer]}"})
                prev_streak = st.session_state.rnn_correct_streak
                is_correct = (user_answer == st.session_state.rnn_correct_answer)

                if is_correct:
                    st.session_state.rnn_correct_streak += 1
                else:
                    st.session_state.rnn_correct_streak = 0

                radio_placeholder.empty() # Limpiar el radio button después de enviar

                # Lógica de subida de nivel y felicitaciones
                if st.session_state.rnn_correct_streak > 0 and \
                   st.session_state.rnn_correct_streak % 3 == 0 and \
                   st.session_state.rnn_correct_streak > prev_streak: # Asegurar que solo se active una vez por subida de nivel
                    
                    if st.session_state.rnn_correct_streak < 9:
                        current_level_text = ""
                        if st.session_state.rnn_correct_streak == 3:
                            current_level_text = "Iniciado (como un hábil rastreador de patrones temporales)"
                        elif st.session_state.rnn_correct_streak == 6:
                            current_level_text = "Kage (como un estratega que domina la memoria de las RNNs)"
                        
                        level_up_message = f"¡Impresionante, Guerrero de las Secuencias! ¡Has completado {st.session_state.rnn_correct_streak} desafíos seguidos! ¡Enhorabuena! Has ascendido al **Nivel {current_level_text}** de RNNs. ¡Los próximos desafíos pondrán a prueba tu dominio! ¡Eres un/a verdadero/a prodigio de las secuencias! 🚀"
                        st.session_state.rnn_game_messages.append({"role": "assistant", "content": level_up_message})
                        st.balloons()
                        try:
                            tts_level_up = gTTS(text=level_up_message, lang='es', slow=False)
                            audio_fp_level_up = io.BytesIO()
                            tts_level_up.write_to_fp(audio_fp_level_up)
                            audio_fp_level_up.seek(0)
                            st.audio(audio_fp_level_up, format="audio/mp3", start_time=0)
                            time.sleep(2) # Esperar un poco para que el audio se reproduzca
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de ascenso de nivel: {e}")
                    elif st.session_state.rnn_correct_streak >= 9:
                        medals_earned = (st.session_state.rnn_correct_streak - 6) // 3 # Ajustado para calcular medallas después del nivel 6
                        medal_message = f"🏅 ¡FELICITACIONES, GRAN MAESTRO RNNINJA! ¡Has forjado tu {medals_earned}ª Medalla del Ciclo Recurrente! ¡Tu dominio es legendario y digno de un verdadero MAESTRO de las RNNs! ¡Que el flujo de información te guíe! 🌟"
                        st.session_state.rnn_game_messages.append({"role": "assistant", "content": medal_message})
                        st.balloons()
                        st.snow()
                        try:
                            tts_medal = gTTS(text=medal_message, lang='es', slow=False)
                            audio_fp_medal = io.BytesIO()
                            tts_medal.write_to_fp(audio_fp_medal)
                            audio_fp_medal.seek(0)
                            st.audio(audio_fp_medal, format="audio/mp3", start_time=0)
                            time.sleep(3)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de medalla: {e}")
                    
                    if st.session_state.rnn_correct_streak >= 9 and prev_streak < 9: # Solo activa el mensaje de Maestro la primera vez
                        level_up_message_champion = f"¡Has desbloqueado el **Nivel Gran Maestro RNNinja**! ¡Los desafíos ahora son solo para aquellos que viven y respiran las secuencias temporales! ¡Adelante con tu camino ninja!"
                        st.session_state.rnn_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                        try:
                            tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                            audio_fp_level_up_champion = io.BytesIO()
                            tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                            audio_fp_level_up_champion.seek(0)
                            st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0)
                            time.sleep(2)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de Gran Maestro: {e}")


                # Obtener feedback de la API
                with st.spinner("RNNinja está analizando tu respuesta..."):
                    try:
                        feedback_prompt = f"""
                        El usuario respondió '{user_answer}'. La pregunta era: '{st.session_state.rnn_current_question}'.
                        La respuesta correcta era '{st.session_state.rnn_correct_answer}'.
                        Da feedback como RNNinja.
                        Si es CORRECTO, el mensaje es "¡Dominas la secuencia! ¡Excelente precisión, aspirante a ninja!" o similar.
                        Si es INCORRECTO, el mensaje es "¡Esa técnica necesita más entrenamiento, joven ninja! ¡No te rindas!" o similar.
                        Luego, una explicación sencilla para el usuario.
                        Finalmente, pregunta: "¿Listo para más entrenamiento en el arte de las RNNs?".
                        **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                        """
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": rnn_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=300
                        )
                        raw_rnn_feedback_text = feedback_response.choices[0].message.content
                        feedback_message, explanation_message, continue_question = parse_rnn_feedback_response(raw_rnn_feedback_text)
                        
                        st.session_state.rnn_game_messages.append({"role": "assistant", "content": feedback_message})
                        st.session_state.rnn_game_messages.append({"role": "assistant", "content": explanation_message})
                        st.session_state.rnn_game_messages.append({"role": "assistant", "content": continue_question})

                        try:
                            tts = gTTS(text=f"{feedback_message}. {explanation_message}. {continue_question}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")

                        # Limpiar la pregunta actual y preparar para la siguiente decisión
                        st.session_state.rnn_current_question = None
                        st.session_state.rnn_current_options = {}
                        st.session_state.rnn_correct_answer = None
                        st.session_state.rnn_game_needs_new_question = False
                        st.session_state.rnn_awaiting_next_game_decision = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error al comunicarse con la API de OpenAI para el feedback: {e}")
                        st.session_state.rnn_game_active = False
                        st.session_state.rnn_game_messages.append({"role": "assistant", "content": "Lo siento, no puedo darte feedback ahora mismo. ¡Mi sensei (la API) no responde! ¡Por favor, reinicia el juego!"})
                        st.rerun()

        # Botones para continuar o terminar el juego
        if st.session_state.rnn_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué harás a continuación, joven ninja?")
            col_continue, col_end = st.columns(2)
            with col_continue:
                if st.button("👍 ¡Sí, a perfeccionar mis técnicas!", key="continue_rnn_game"):
                    st.session_state.rnn_awaiting_next_game_decision = False
                    st.session_state.rnn_game_needs_new_question = True
                    st.session_state.rnn_game_messages.append({"role": "assistant", "content": "¡Excelente! ¡El camino del conocimiento de las RNNs es infinito! ¡Aquí va tu próximo desafío!"})
                    st.rerun()
            with col_end:
                if st.button("👎 Por hoy, el entrenamiento ha terminado.", key="end_rnn_game"):
                    st.session_state.rnn_game_active = False
                    st.session_state.rnn_awaiting_next_game_decision = False
                    st.session_state.rnn_game_messages.append({"role": "assistant", "content": "¡Hasta la próxima sesión de entrenamiento, aspirante a Gran Maestro! ¡Recuerda los principios de las secuencias!"})
                    st.rerun()

else:
    st.info("El chatbot RNNinja no está disponible porque la clave de la API de OpenAI no está configurada o no es válida. Por favor, asegúrate de añadirla en Streamlit Secrets.")

st.write("---")