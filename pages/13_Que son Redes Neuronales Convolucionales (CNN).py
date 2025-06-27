import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image
import os
import json
import time
import io
from gtts import gTTS

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    st.error("Las librerías 'tensorflow' y 'keras' no están instaladas. Por favor, instálalas usando: pip install tensorflow")
    tf = None

try:
    from openai import OpenAI
except ImportError:
    st.error("La librería 'openai' no está instalada. Por favor, instálala usando: pip install openai")
    OpenAI = None

# --- Configuración de la página ---
st.set_page_config(
    page_title="Laboratorio de Detección de Fallos (CNNs)",
    layout="wide"
)

# --- Rutas para el modelo CNN y el dataset ---
current_script_dir = os.path.dirname(__file__)
project_root_dir = os.path.normpath(os.path.join(current_script_dir, '..'))

CNN_MODEL_PATH = os.path.normpath(os.path.join(project_root_dir, 'assets', 'models', 'cnn_defect_detector_model.h5'))
DATASET_BASE_DIR = os.path.normpath(os.path.join(project_root_dir, 'datasets', 'piece_defects'))


# --- Parámetros para el modelo CNN ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# --- Funciones auxiliares para el modelo CNN ---

@st.cache_resource
def load_cnn_model(path):
    if tf is not None:
        if os.path.exists(path):
            try:
                model = load_model(path, compile=False)
                st.success(f"¡Modelo CNN de detección de fallos cargado con éxito desde: {path}!")
                return model
            except Exception as e:
                st.error(f"Error al cargar el modelo CNN desde '{path}'. Asegúrate de que el archivo existe y es un modelo Keras válido. Error: {e}")
                return None
        else:
            st.error(f"¡Error! No se encontró el archivo del modelo CNN en la ruta especificada: '{path}'. Por favor, verifica la ubicación y el nombre del archivo.")
            return None
    return None

@st.cache_resource
def load_test_cnn_data(dataset_path):
    """Carga los datos de prueba del dataset y las rutas para la selección."""
    test_dir = os.path.join(dataset_path, 'test')
    train_dir = os.path.join(dataset_path, 'train')

    if not os.path.exists(test_dir):
        st.error(f"¡El directorio de prueba del dataset NO EXISTE! Por favor, verifica la ruta: '{test_dir}'.")
        return None, None, None

    found_test_images_count = 0
    test_image_paths = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif') 

    expected_class_folders = ['good', 'defect']
    for class_folder in expected_class_folders:
        class_path = os.path.join(test_dir, class_folder)
        if os.path.isdir(class_path):
            current_class_images = 0
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(supported_extensions):
                    full_img_path = os.path.join(class_path, img_name)
                    test_image_paths.append(full_img_path)
                    found_test_images_count += 1
                    current_class_images += 1
        else:
            st.warning(f"La subcarpeta de clase esperada '{class_folder}' no se encontró en '{test_dir}'. Asegúrate de la estructura de tu dataset.")

    if found_test_images_count == 0:
        st.warning(f"¡Alerta! La verificación manual no encontró ninguna imagen válida en '{test_dir}' o sus subcarpetas 'good'/'defect'. Asegúrate de que hay imágenes con extensiones {supported_extensions} dentro.")

    datagen_minimal = ImageDataGenerator(rescale=1./255)
    
    class_names = []
    test_generator = None
    
    try:
        if os.path.exists(train_dir) and any(os.scandir(train_dir)):
            try:
                temp_train_generator = datagen_minimal.flow_from_directory(
                    train_dir,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=1,
                    class_mode='binary',
                    shuffle=False
                )
                if temp_train_generator.samples == 0:
                    st.warning(f"DEBUG: train_generator no encontró imágenes en '{train_dir}'. Las subcarpetas pueden estar vacías o faltar.")
                    class_names = ['defect', 'good'] # Fallback
                else:
                    class_names = sorted(list(temp_train_generator.class_indices.keys()))
            except Exception as e:
                st.warning(f"No se pudieron obtener los class_indices del directorio de entrenamiento '{train_dir}'. Error: {e}. Usando clases por defecto.")
                class_names = ['defect', 'good']
        else:
            st.warning(f"El directorio de entrenamiento '{train_dir}' no existe o está vacío. Las clases se establecerán a 'defect', 'good' por defecto.")
            class_names = ['defect', 'good']

        if os.path.exists(test_dir) and any(os.scandir(test_dir)):
            try:
                test_generator = datagen_minimal.flow_from_directory(
                    test_dir,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=BATCH_SIZE,
                    class_mode='binary',
                    shuffle=False
                )
                if test_generator.samples == 0:
                    st.warning(f"¡El ImageDataGenerator no encontró ninguna imagen en '{test_dir}'! Asegúrate de que las subcarpetas 'good' y 'defect' contengan imágenes válidas y con las extensiones correctas.")
                    test_generator = None 
            except Exception as e:
                st.error(f"ERROR: Fallo al ejecutar flow_from_directory en '{test_dir}'. Posiblemente por estructura incorrecta o falta de imágenes. Error: {e}")
                test_generator = None
        else:
            st.warning(f"ADVERTENCIA: El directorio de test '{test_dir}' no existe o está vacío para ImageDataGenerator. No se cargarán imágenes para el juego.")

        return test_generator, class_names, test_image_paths
    except Exception as e:
        st.error(f"Error general al cargar datos de prueba para el juego. Asegúrate de que '{test_dir}' y '{train_dir}' contengan subcarpetas (ej. 'good' y 'defect'). Error: {e}")
        return None, None, None

# --- Carga del modelo CNN al inicio ---
cnn_model = load_cnn_model(CNN_MODEL_PATH)

# --- Inicialización más robusta de session_state para el módulo CNN ---
if 'cnn_module_config' not in st.session_state:
    st.session_state.cnn_module_config = {}

if 'trained_model' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['trained_model'] = cnn_model
if 'last_prediction_image' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['last_prediction_image'] = None
if 'last_prediction_result' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['last_prediction_result'] = None
if 'last_prediction_probability' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['last_prediction_probability'] = None
if 'test_generator' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['test_generator'] = None
if 'class_names' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['class_names'] = ['defect', 'good']
if 'test_image_paths' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['test_image_paths'] = []
if 'game1_correct_count' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['game1_correct_count'] = 0
if 'game1_total_count' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['game1_total_count'] = 0
if 'current_game2_image_path' not in st.session_state.cnn_module_config:
    st.session_state.cnn_module_config['current_game2_image_path'] = None

if (st.session_state.cnn_module_config['trained_model'] is not None and
    (st.session_state.cnn_module_config['test_generator'] is None or
     not st.session_state.cnn_module_config['test_image_paths'])):
    test_gen, class_names, test_img_paths = load_test_cnn_data(DATASET_BASE_DIR)
    st.session_state.cnn_module_config['test_generator'] = test_gen
    st.session_state.cnn_module_config['test_image_paths'] = test_img_paths
    if class_names:
        st.session_state.cnn_module_config['class_names'] = class_names

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
    Para usar el chatbot del Detective Pixel, necesitas añadir tu clave de la API de OpenAI a tu archivo `secrets.toml`.
    """)
    OpenAI = None

# --- Título y Explicación del Módulo (Mantener igual) ---
st.title("Laboratorio Interactivo de Detección de Fallos con Redes Convolucionales (CNNs)")

st.markdown("""
¡Bienvenido al laboratorio donde una **Red Convolucional (CNN)** nos ayudará a encontrar piezas defectuosas en una fábrica!

---

### ¿Cómo funciona la CNN? ¡Es como un detective de imágenes!

Imagina que eres un detective de calidad en una fábrica. Tu trabajo es mirar miles de piezas y decir si están **perfectas** o si tienen algún **fallo**. ¡Una CNN hace exactamente eso, pero a toda velocidad!

* **Entrenando al Detective:** Le mostramos a la CNN muchísimas fotos de piezas: algunas perfectas y otras con fallos. Le decimos: "¡Esta está bien!", "¡Esta tiene un arañazo!", "¡Esta tiene un golpe!". La CNN aprende a reconocer los patrones de las piezas buenas y los patrones de los defectos.
* **Inspeccionando piezas nuevas:** Cuando le mostramos una foto de una pieza nueva que nunca ha visto, la CNN usa lo que ha aprendido para decirnos si cree que está **bien** o **defectuosa**.

¡Es como si la CNN tuviera unos "ojos" súper especiales para ver los detalles más pequeños en las imágenes!
""")

st.write("---")

# --- Sección de Configuración del Modelo CNN (simplificada y explicada para niños) ---
st.header("Tu Detective de Piezas (CNN) - ¡Listo para Usar!")
st.markdown("El modelo de detección de fallos ya ha sido entrenado y cargado. ¡Está listo para su misión!")

if st.session_state.cnn_module_config['trained_model'] is not None:
    st.success("¡Modelo CNN de detección de fallos preentrenado cargado con éxito!")
    st.markdown("### ¿Cómo ve y aprende el Detective Pixel?")

    st.write("""
    Imagina que el Detective Pixel tiene unos **superpoderes visuales** para analizar las piezas.
    No las ve como nosotros, ¡sino que las descompone en muchos detalles!
    """)

    st.subheader("1. Los 'Ojos Detectives' (Capas Convolucionales)")
    st.write("""
    Piensa que el Detective Pixel tiene muchos **pequeños ojos mágicos** (filtros).
    Cada ojo busca una cosa diferente en la imagen:
    * Un ojo busca **bordes** (líneas).
    * Otro ojo busca **texturas** (rugosidad).
    * Otro busca **formas** (círculos, cuadrados).

    Cuando uno de sus ojos encuentra lo que busca, lo marca. ¡Así la imagen se llena de "marcas" de lo que ha visto!
    """)

    st.subheader("2. Resumiendo la Información (Capas de Pooling)")
    st.write("""
    Después de que todos los ojos marcan los detalles, el Detective Pixel necesita **resumir** lo que ha encontrado.
    Es como si en lugar de recordar cada detalle minúsculo, solo se quedara con los más importantes y los agrupara.
    ¡Esto le ayuda a no confundirse con tanto detalle y a ser más rápido!
    """)

    st.subheader("3. El 'Cerebro Decisor' (Capas Densas)")
    st.write("""
    Finalmente, toda esa información resumida llega al **cerebro del Detective Pixel**.
    Aquí es donde realmente "piensa" y junta todas las pistas.
    Al final, toma una decisión:

    * "¡Esta pieza es **BUENA**!" ✅
    * "¡Esta pieza es **DEFECTUOSA**!" ❌

    ¡Y nos dice su veredicto!
    """)

    st.markdown("""
    Así, paso a paso, el Detective Pixel analiza la imagen y nos ayuda a saber si una pieza está en perfecto estado o tiene algún fallo.
    ¡Es un trabajo muy importante en la fábrica!
    """)

    with st.expander("Mira una simulación muy simple de cómo ve la CNN (idea conceptual)"):
        st.write("Imagina que la CNN ve tu imagen como una cuadrícula de números:")
        original_pixel_grid = np.array([[0, 0, 0, 0, 0],
                                        [0, 1, 1, 1, 0],
                                        [0, 1, 0, 1, 0],
                                        [0, 1, 1, 1, 0],
                                        [0, 0, 0, 0, 0]])
        st.dataframe(original_pixel_grid, hide_index=True)

        st.write("Uno de sus 'ojos detectivess' (un filtro) busca, por ejemplo, líneas verticales:")
        filter_vertical = np.array([[1, -1], [1, -1]])
        st.dataframe(filter_vertical, hide_index=True)
        st.write("Cuando el 'ojo' pasa por la imagen, 'ilumina' (da un número más alto) donde encuentra ese patrón. ¡Así crea un nuevo mapa de 'marcas'!")

        st.write("Luego, para resumir, de cada 4 cuadritos se queda con el número más grande (MaxPool):")
        st.markdown("`[[1, 2], [3, 4]]` se convierte en `[4]`")
        st.write("Esto ayuda a la CNN a enfocarse en lo más importante, sin los detalles pequeños.")

        st.write("Al final, toda esa información procesada llega a la parte 'cerebro' que toma la decisión final:")
        st.progress(85, text="Detective Pixel: ¡Confianza del 85% de que está buena!")

else:
    st.error("¡El modelo CNN no está cargado! Asegúrate de haber ejecutado el script de entrenamiento (`train_cnn_model.py`) para crear el modelo.")

st.write("---")

# --- Sección de Detección de Fallos (Juego 1: Clasificación de Imagen Subida/Seleccionada) ---
st.header("¡Juego 1: Detective de Fallos de Piezas!")
st.markdown(f"""
¡A ver si el Detective Pixel es bueno! Puedes elegir una foto del laboratorio o subir la tuya propia.
**Aciertos: {st.session_state.cnn_module_config['game1_correct_count']} / {st.session_state.cnn_module_config['game1_total_count']}**
""")

if st.session_state.cnn_module_config['trained_model'] is None:
    st.warning("No hay un modelo CNN cargado. Por favor, asegúrate de que el modelo ha sido entrenado y guardado correctamente.")
else:
    col_input_method, _ = st.columns([0.4, 0.6])
    with col_input_method:
        input_method = st.radio(
            "¿Cómo quieres darle una foto al Detective Pixel?",
            ("Subir mi propia foto", "Elegir una foto del laboratorio"),
            key="cnn_image_input_method"
        )

    image_to_predict = None
    original_image_path = None

    if input_method == "Subir mi propia foto":
        uploaded_file = st.file_uploader("Sube una imagen de una pieza (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"], key="cnn_game1_file_uploader")
        if uploaded_file is not None:
            try:
                image_to_predict = Image.open(uploaded_file).convert('RGB')
            except Exception as e:
                st.error(f"Error al cargar la imagen subida: {e}")
                image_to_predict = None

    elif input_method == "Elegir una foto del laboratorio":
        test_img_paths = st.session_state.cnn_module_config['test_image_paths']
        if not test_img_paths:
            st.warning("No hay imágenes disponibles en el laboratorio. Asegúrate de que tu carpeta `datasets/piece_defects/test` contenga subcarpetas de clase (ej. 'good' y 'defect') con imágenes dentro.")
        else:
            unique_paths = list(set(test_img_paths))
            num_display_images = min(20, len(unique_paths)) # Sigue siendo 20 por defecto, cámbialo si quieres más
            
            # --- SELECCIÓN ALEATORIA DE MUESTRAS ---
            random_sample_paths = random.sample(unique_paths, num_display_images)
            # --- ELIMINA ESTA LÍNEA PARA QUE EL ORDEN SEA COMPLETAMENTE ALEATORIO EN CADA RECARGA ---
            # random_sample_paths.sort() # Comentada para que el orden sea más aleatorio

            # --- Opciones para mostrar en el radio, ocultando la clase ---
            display_options = [f"Imagen de muestra #{i+1}" for i in range(len(random_sample_paths))]

            # Mapea las opciones de vuelta a la ruta completa
            option_to_path_map = {display_options[i]: random_sample_paths[i] for i in range(len(display_options))}

            selected_display_option = st.radio(
                "Selecciona una imagen de muestra:",
                options=display_options,
                key="cnn_game1_sample_selector_radio"
            )

            if selected_display_option:
                selected_path = option_to_path_map[selected_display_option]
                original_image_path = selected_path
                try:
                    image_to_predict = Image.open(selected_path).convert('RGB')
                except Exception as e:
                    st.error(f"Error al cargar la imagen seleccionada: {e}")
                    image_to_predict = None

                if image_to_predict:
                    st.image(image_to_predict, caption=f'Imagen seleccionada: {selected_display_option}', use_container_width=False, width=200)

    col_img, col_pred = st.columns([1, 2])
    with col_img:
        if image_to_predict is not None:
            st.image(image_to_predict, caption='Imagen a inspeccionar.', use_container_width=True)

    with col_pred:
        if image_to_predict is not None:
            if st.button("¡Pregúntale al Detective Pixel!", key="predict_cnn_button_game1"):
                with st.spinner("Detective Pixel está inspeccionando la pieza..."):
                    try:
                        img_array = np.array(image_to_predict.resize((IMG_WIDTH, IMG_HEIGHT))) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        predictions = st.session_state.cnn_module_config['trained_model'].predict(img_array)
                        
                        class_indices_map = st.session_state.cnn_module_config['test_generator'].class_indices if st.session_state.cnn_module_config['test_generator'] else {'defect': 0, 'good': 1}
                        
                        predicted_class_label = ''
                        confidence = 0.0

                        if predictions.shape[-1] == 1:
                            probability_of_class_1 = predictions[0][0]
                            
                            if 0 in class_indices_map.values() and 1 in class_indices_map.values():
                                label_for_0 = next((k for k, v in class_indices_map.items() if v == 0), 'defect')
                                label_for_1 = next((k for k, v in class_indices_map.items() if v == 1), 'good')

                                if probability_of_class_1 > 0.5:
                                    predicted_class_label = label_for_1
                                    confidence = probability_of_class_1
                                else:
                                    predicted_class_label = label_for_0
                                    confidence = 1 - probability_of_class_1
                            else:
                                predicted_class_label = 'good' if probability_of_class_1 > 0.5 else 'defect'
                                confidence = probability_of_class_1 if predicted_class_label == 'good' else (1 - probability_of_class_1)
                        else:
                            predicted_class_index = np.argmax(predictions[0])
                            if predicted_class_index < len(st.session_state.cnn_module_config['class_names']):
                                predicted_class_label = st.session_state.cnn_module_config['class_names'][predicted_class_index]
                            else:
                                predicted_class_label = 'desconocido'
                            confidence = predictions[0][predicted_class_index]

                        st.session_state.cnn_module_config['last_prediction_image'] = image_to_predict
                        st.session_state.cnn_module_config['last_prediction_result'] = predicted_class_label
                        st.session_state.cnn_module_config['last_prediction_probability'] = confidence

                        st.subheader("¡El Detective Pixel ha hablado!")
                        if predicted_class_label == 'good':
                            st.success(f"🔍 **¡Parece una pieza en buen estado!** (Confianza: {confidence*100:.2f}%)")
                        else:
                            st.error(f"🚨 **¡Alerta! Parece una pieza defectuosa.** (Confianza: {confidence*100:.2f}%)")
                        
                        if original_image_path:
                            st.session_state.cnn_module_config['game1_total_count'] += 1
                            true_label = os.path.basename(os.path.dirname(original_image_path))
                            if predicted_class_label == true_label:
                                st.session_state.cnn_module_config['game1_correct_count'] += 1
                                st.success(f"¡Has acertado! La imagen era realmente: **{true_label}**.")
                            else:
                                st.error(f"¡Fallaste! La imagen era realmente: **{true_label}**.")
                            st.markdown(f"**Resultado actual: {st.session_state.cnn_module_config['game1_correct_count']} aciertos de {st.session_state.cnn_module_config['game1_total_count']} intentos.**")
                        else:
                            st.info("Para que el juego cuente aciertos, debes elegir una imagen 'del laboratorio'.")

                        st.markdown("---")

                    except Exception as e:
                        st.error(f"Error al realizar la predicción con la CNN: {e}")
                        
# --- Sección del Juego 2: Adivina el Defecto ---
st.header("¡Juego 2: Pon a prueba tu ojo de Detective!")
st.markdown("""
¡Aquí el desafío es para TI! Te mostraré una pieza y tendrás que adivinar si está **buena** o **defectuosa** *antes* de que el Detective Pixel dé su veredicto.
""")

if st.session_state.cnn_module_config['trained_model'] is None or not st.session_state.cnn_module_config['test_image_paths']:
    st.warning("Este juego no está disponible. Asegúrate de que el modelo esté cargado y haya imágenes de test disponibles.")
else:
    if 'game2_current_image_path' not in st.session_state:
        st.session_state.game2_current_image_path = None
    if 'game2_awaiting_user_guess' not in st.session_state:
        st.session_state.game2_awaiting_user_guess = False
    if 'game2_last_guess_result' not in st.session_state:
        st.session_state.game2_last_guess_result = None
    if 'game2_total_count' not in st.session_state:
        st.session_state.game2_total_count = 0
    if 'game2_correct_count' not in st.session_state:
        st.session_state.game2_correct_count = 0
    if 'game2_user_guess_made' not in st.session_state:
        st.session_state.game2_user_guess_made = False

    def load_new_game2_image():
        available_paths = st.session_state.cnn_module_config['test_image_paths']
        if available_paths:
            st.session_state.game2_current_image_path = random.choice(available_paths)
            st.session_state.game2_awaiting_user_guess = True
            st.session_state.game2_last_guess_result = None
            st.session_state.game2_user_guess_made = False
        else:
            st.error("No hay imágenes disponibles para el Juego 2.")

    if not st.session_state.game2_awaiting_user_guess and not st.session_state.game2_user_guess_made:
        if st.button("¡Empezar nueva ronda de Juego 2!", key="start_game2_button"):
            load_new_game2_image()
            st.rerun()

    if st.session_state.game2_current_image_path:
        st.subheader("¿Cómo es esta pieza, Detective Humano?")
        st.markdown(f"**Aciertos en Juego 2: {st.session_state.game2_correct_count} / {st.session_state.game2_total_count}**")
        
        col_game2_img, col_game2_guess = st.columns([1, 1])
        with col_game2_img:
            try:
                img_display = Image.open(st.session_state.game2_current_image_path).convert('RGB')
                st.image(img_display, caption="¡Observa con atención!", use_container_width=True)
            except Exception as e:
                st.error(f"No se pudo cargar la imagen para el juego: {e}")
                st.session_state.game2_current_image_path = None
                st.session_state.game2_awaiting_user_guess = False
                st.session_state.game2_user_guess_made = False
                st.rerun()

        with col_game2_guess:
            if st.session_state.game2_awaiting_user_guess:
                user_guess = st.radio(
                    "¿Es una pieza...?",
                    ("Buena", "Defectuosa"),
                    key="game2_user_guess"
                )
                if st.button("¡Revelar el veredicto del Detective Pixel!", key="game2_reveal_button"):
                    st.session_state.game2_awaiting_user_guess = False
                    st.session_state.game2_user_guess_made = True
                    
                    true_label_folder_name = os.path.basename(os.path.dirname(st.session_state.game2_current_image_path))
                    
                    user_guess_normalized = "good" if user_guess == "Buena" else "defect"
                    
                    st.session_state.game2_total_count += 1
                    if user_guess_normalized == true_label_folder_name:
                        st.session_state.game2_last_guess_result = 'correct'
                        st.session_state.game2_correct_count += 1
                        st.success(f"¡Felicidades, Detective! ¡Tu ojo es tan bueno como el de Pixel! Era una pieza **{true_label_folder_name}**.")
                    else:
                        st.session_state.game2_last_guess_result = 'incorrect'
                        st.error(f"¡Casi, casi! Pero el Detective Pixel dice que era **{true_label_folder_name}**. ¡No te rindas!")
                    
                    img_for_pred = Image.open(st.session_state.game2_current_image_path).convert('RGB')
                    img_array = np.array(img_for_pred.resize((IMG_WIDTH, IMG_HEIGHT))) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    predictions = st.session_state.cnn_module_config['trained_model'].predict(img_array)

                    test_gen = st.session_state.cnn_module_config['test_generator']
                    class_indices_map = test_gen.class_indices if test_gen else {'defect': 0, 'good': 1}

                    cnn_prediction_label = ''
                    cnn_confidence = 0.0

                    if predictions.shape[-1] == 1:
                        probability_of_class_1 = predictions[0][0]
                        if 0 in class_indices_map.values() and 1 in class_indices_map.values():
                            label_for_0 = next((k for k, v in class_indices_map.items() if v == 0), 'defect')
                            label_for_1 = next((k for k, v in class_indices_map.items() if v == 1), 'good')

                            if probability_of_class_1 > 0.5:
                                predicted_class_label = label_for_1
                                confidence = probability_of_class_1
                            else:
                                predicted_class_label = label_for_0
                                confidence = 1 - probability_of_class_1
                        else:
                            predicted_class_label = 'good' if probability_of_class_1 > 0.5 else 'defect'
                            confidence = probability_of_class_1 if predicted_class_label == 'good' else (1 - probability_of_class_1)
                    else:
                        predicted_class_index = np.argmax(predictions[0])
                        if predicted_class_index < len(st.session_state.cnn_module_config['class_names']):
                            predicted_class_label = st.session_state.cnn_module_config['class_names'][predicted_class_index]
                        else:
                            predicted_class_label = 'desconocido'
                        confidence = predictions[0][predicted_class_index]
                    
                    st.info(f"**El veredicto del Detective Pixel es:** ¡Es **{predicted_class_label}**! (Con una confianza del {cnn_confidence*100:.2f}%)")
                    st.markdown("---")
                    st.button("¡Siguiente ronda de Juego 2!", key="next_game2_button", on_click=load_new_game2_image)
            else:
                st.write("Selecciona tu respuesta y luego haz clic para ver si acertaste.")
    elif st.session_state.game2_user_guess_made:
        st.button("¡Siguiente ronda de Juego 2!", key="next_game2_after_reveal_button", on_click=load_new_game2_image)

# --- Sección de Chatbot de Juego con Detective Pixel ---
st.header("¡Juega y Aprende con Detective Pixel sobre CNNs!")
st.markdown("¡Hola! Soy Detective Pixel, tu amigo detective de la IA. ¿Listo para desentrañar los secretos de la visión por computadora y cómo las CNNs ven el mundo?")

if client:
    if "cnn_game_active" not in st.session_state:
        st.session_state.cnn_game_active = False
    if "cnn_game_messages" not in st.session_state:
        st.session_state.cnn_game_messages = []
    if "cnn_current_question" not in st.session_state:
        st.session_state.cnn_current_question = None
    if "cnn_current_options" not in st.session_state:
        st.session_state.cnn_current_options = {}
    if "cnn_correct_answer" not in st.session_state:
        st.session_state.cnn_correct_answer = None
    if "cnn_awaiting_next_game_decision" not in st.session_state:
        st.session_state.cnn_awaiting_next_game_decision = False
    if "cnn_game_needs_new_question" not in st.session_state:
        st.session_state.cnn_game_needs_new_question = False
    if "cnn_correct_streak" not in st.session_state:
        st.session_state.cnn_correct_streak = 0

    cnn_game_system_prompt = f"""
    Eres un **experto consumado en Visión por Computadora y Deep Learning**, con una especialización profunda en el diseño, entrenamiento y comprensión de las **Redes Neuronales Convolucionales (CNN)**. Comprendes a fondo sus fundamentos (operaciones de convolución, pooling), su arquitectura específica para datos de imagen y vídeo, sus ventajas en el reconocimiento de patrones espaciales y sus diversas aplicaciones prácticas en clasificación de imágenes, detección de objetos y segmentación. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de las CNN mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Filtro de características activado! Tu CNN ha reconocido el patrón." o "Esa convolución necesita ajuste. Revisemos los mapas de características."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "Una CNN es una arquitectura de red neuronal especialmente diseñada para procesar datos con una topología conocida, como imágenes, aprovechando la correlación espacial de los píxeles..."]
    [Pregunta para continuar, ej: "¿Listo para optimizar tus arquitecturas convolucionales?" o "¿Quieres profundizar en la transferencia de conocimiento en CNNs?"]

    **Reglas adicionales para el Experto en Redes Neuronales Convolucionales:**
    * **Enfoque Riguroso en CNN:** Todas tus preguntas y explicaciones deben girar en torno a las Redes Neuronales Convolucionales. Cubre sus fundamentos (diseño inspirado en la visión biológica), los componentes clave (capas convolucionales, filtros/kernels, mapas de características, pooling, capas densas), el proceso de extracción jerárquica de características, el **entrenamiento** (backpropagation, optimizadores), el **sobreajuste** y técnicas de **regularización** (Dropout, Batch Normalization), la **transferencia de aprendizaje (Transfer Learning)** con CNNs (preentrenamiento, fine-tuning), y sus aplicaciones principales.
    * **¡VARIEDAD, VARIADAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de CNN que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General e Inspiración:** ¿Qué es una CNN? ¿Por qué son especiales para imágenes? Inspiración biológica.
        * **Capas Convolucionales:**
            * **Operación de Convolución:** Funcionamiento del filtro/kernel, stride, padding.
            * **Filtros/Kernels:** Su propósito (detección de bordes, texturas, patrones).
            * **Mapas de Características (Feature Maps):** Cómo se generan.
            * **Parámetros Aprendibles:** Pesos de los filtros y sesgos.
        * **Capas de Pooling (Submuestreo):**
            * **Propósito:** Reducción dimensional, invariancia traslacional, reducción de parámetros.
            * **Tipos:** Max Pooling, Average Pooling.
        * **Arquitectura de una CNN Típica:** Secuencia de capas convolucionales y de pooling, seguido de capas densas (Fully Connected) al final para clasificación.
        * **Funciones de Activación:** ReLU como estándar en capas ocultas.
        * **Entrenamiento de CNNs:**
            * **Backpropagation:** Adaptación para CNNs.
            * **Optimizadores:** Adam, SGD.
            * **Función de Pérdida:** Cross-Entropy para clasificación.
        * **Sobreajuste y Regularización en CNNs:** Dropout, Batch Normalization, Data Augmentation.
        * **Transfer Learning:**
            * **Concepto:** Reutilización de modelos preentrenados (ImageNet).
            * **Fine-tuning:** Ajuste de las últimas capas o de la red completa.
            * **Extracción de Características:** Uso de CNNs como extractores.
        * **Ventajas de las CNNs:** Eficaces para datos con estructura espacial, reducción de parámetros (compartir pesos), invariancia.
        * **Limitaciones/Desafíos:** Necesidad de grandes datasets etiquetados, alto costo computacional de entrenamiento, interpretabilidad limitada.
        * **Aplicaciones Principales:** Clasificación de imágenes, detección de objetos, segmentación semántica, reconocimiento facial.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.cnn_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Visionario Digital – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea de que una máquina "ve" imágenes y ejemplos sencillos de lo que puede hacer una CNN (ej., reconocer una cara).
            * *Tono:* "Estás abriendo tus ojos al mundo de cómo las máquinas perciben e interpretan las imágenes."
        * **Nivel 2 (Constructor de Detectores – 3-5 respuestas correctas):** Tono más técnico. Introduce los conceptos de **filtros/kernels** y cómo "barren" la imagen para detectar patrones básicos. Preguntas sobre la función de las capas convolucionales o de pooling.
            * *Tono:* "Tu habilidad para descomponer y analizar los elementos visuales está mejorando con cada capa."
        * **Nivel 3 (Arquitecto de CNN – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en los detalles de la operación de **convolución** (stride, padding), la generación de **mapas de características**, la justificación de las capas de **pooling**, y el proceso de **transfer learning** con modelos preentrenados.
            * *Tono:* "Tu comprensión profunda de las arquitecturas convolucionales te permite diseñar soluciones robustas para el procesamiento de imágenes."
        * **Nivel Maestro (Científico de Visión por Computadora – 9+ respuestas correctas):** Tono de **especialista en la vanguardia del Deep Learning aplicado a la visión**. Preguntas sobre el diseño de arquitecturas complejas (Inception, ResNet, VGG de forma conceptual), el manejo de datasets muy grandes o desbalanceados en visión, el impacto de diferentes optimizadores para CNNs, o las implicaciones de aplicar CNNs a datos no visuales. Se esperan respuestas que demuestren una comprensión teórica y práctica robusta, incluyendo sus limitaciones y el estado del arte.
            * *Tono:* "Tu maestría en Redes Neuronales Convolucionales te permite no solo ver, sino también entender y transformar el mundo visual digitalmente."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Una aplicación que clasifica fotos de animales o plantas.
        * **Nivel 2:** Un sistema que detecta caras en una imagen o reconoce objetos específicos (coches, señales) en vídeos para vehículos autónomos.
        * **Nivel 3:** Entrenar una CNN para diagnosticar enfermedades en radiografías, o usar una CNN preentrenada (ej. VGG16) para extraer características de imágenes para una tarea de clasificación personalizada.
        * **Nivel Maestro:** Diseñar y optimizar una arquitectura CNN para la segmentación semántica de imágenes médicas 3D (MRI), gestionando la complejidad computacional y el tamaño del dataset, o desarrollar un sistema de detección de anomalías visuales en líneas de producción industrial usando CNNs.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre REDES NEURONALES CONVOLUCIONALES (CNN), y asegúrate de que no se parezca a las anteriores.
    """

    def parse_cnn_question_response(raw_text):
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
            # st.warning(f"DEBUG: Formato de pregunta inesperado de la API. Texto recibido:\n{raw_text}") # Mantenemos esto para depuración interna
            return None, {}, ""
        return question, options, correct_answer_key

    def parse_cnn_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        # st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}") # Mantenemos esto para depuración interna
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"
    
    def set_cnn_level(target_streak, level_name):
        st.session_state.cnn_correct_streak = target_streak
        st.session_state.cnn_game_active = True
        st.session_state.cnn_game_messages = []
        st.session_state.cnn_current_question = None
        st.session_state.cnn_current_options = {}
        st.session_state.cnn_correct_answer = None
        st.session_state.cnn_game_needs_new_question = True
        st.session_state.cnn_awaiting_next_game_decision = False
        st.session_state.cnn_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}**! Prepárate para preguntas más desafiantes. ¡Aquí va tu primera!"})
        st.rerun()

    col_game_buttons_cnn, col_level_up_buttons_cnn = st.columns([1, 2])

    with col_game_buttons_cnn:
        if st.button("¡Vamos a jugar con Detective Pixel!", key="start_cnn_game_button"):
            st.session_state.cnn_game_active = True
            st.session_state.cnn_game_messages = []
            st.session_state.cnn_current_question = None
            st.session_state.cnn_current_options = {}
            st.session_state.cnn_correct_answer = None
            st.session_state.cnn_game_needs_new_question = True
            st.session_state.cnn_awaiting_next_game_decision = False
            st.session_state.cnn_correct_streak = 0
            st.rerun()
    
    with col_level_up_buttons_cnn:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un detective experto? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_cnn, col_lvl2_cnn, col_lvl3_cnn = st.columns(3)
        with col_lvl1_cnn:
            if st.button("Subir a Nivel Medio (CNN)", key="level_up_medium_cnn"):
                set_cnn_level(3, "Medio")
        with col_lvl2_cnn:
            if st.button("Subir a Nivel Avanzado (CNN)", key="level_up_advanced_cnn"):
                set_cnn_level(6, "Avanzado")
        with col_lvl3_cnn:
            if st.button("👑 ¡Maestro CNN! (CNN)", key="level_up_champion_cnn"):
                set_cnn_level(9, "Maestro CNN")


    for message in st.session_state.cnn_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if st.session_state.cnn_game_active:
        if st.session_state.cnn_current_question is None and st.session_state.cnn_game_needs_new_question and not st.session_state.cnn_awaiting_next_game_decision:
            with st.spinner("Detective Pixel está preparando una pregunta..."):
                try:
                    cnn_game_messages_for_api = [{"role": "system", "content": cnn_game_system_prompt}]
                    for msg in st.session_state.cnn_game_messages[-6:]:
                        if msg["role"] == "assistant" and msg["content"].startswith("**"):
                            cnn_game_messages_for_api.append({"role": "assistant", "content": f"PREGUNTA ANTERIOR: {msg['content'].splitlines()[0]}"})
                        elif msg["role"] == "user" and "MI RESPUESTA:" not in msg["content"]:
                            cnn_game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {msg['content']}"})

                    cnn_game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre QUÉ SON LAS CNNs siguiendo el formato exacto."})

                    cnn_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=cnn_game_messages_for_api,
                        temperature=0.7,
                        max_tokens=250
                    )
                    raw_cnn_question_text = cnn_response.choices[0].message.content
                    question, options, correct_answer_key = parse_cnn_question_response(raw_cnn_question_text)

                    if question:
                        st.session_state.cnn_current_question = question
                        st.session_state.cnn_current_options = options
                        st.session_state.cnn_correct_answer = correct_answer_key
                        st.session_state.cnn_game_needs_new_question = False
                        
                        question_content = f"**Nivel {int(st.session_state.cnn_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.cnn_correct_streak}**\n\n**Pregunta de Detective Pixel:** {question}\n\n"
                        for k, v in options.items():
                            question_content += f"**{k})** {v}\n"
                        
                        st.session_state.cnn_game_messages.append({"role": "assistant", "content": question_content})
                        st.rerun()
                    else:
                        st.error("Detective Pixel no pudo generar una pregunta válida. Intenta de nuevo.")
                        st.session_state.cnn_game_active = False
                        st.session_state.cnn_game_messages.append({"role": "assistant", "content": "Detective Pixel no pudo generar una pregunta válida. Parece que hay un problema. Por favor, reinicia el juego."})

                except Exception as e:
                    st.error(f"Error al comunicarse con la API de OpenAI para la pregunta: {e}")
                    st.session_state.cnn_game_active = False
                    st.session_state.cnn_game_messages.append({"role": "assistant", "content": "Lo siento, tengo un problema para conectar con mi cerebro (la API). ¡Por favor, reinicia el juego!"})
                    st.rerun()

        if st.session_state.cnn_current_question and not st.session_state.cnn_awaiting_next_game_decision:
            if st.session_state.get('last_played_cnn_question') != st.session_state.cnn_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.cnn_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.cnn_correct_streak}. Pregunta de Detective Pixel: {st.session_state.cnn_current_question}. Opción A: {st.session_state.cnn_current_options.get('A', '')}. Opción B: {st.session_state.cnn_current_options.get('B', '')}. Opción C: {st.session_state.cnn_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.audio(fp, format='audio/mp3', start_time=0)
                    st.session_state.last_played_cnn_question = st.session_state.cnn_current_question
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")

            with st.form(key="cnn_game_form"):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_answer = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.cnn_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.cnn_current_options[x]}",
                        key="cnn_answer_radio",
                        label_visibility="collapsed"
                    )
                submit_button = st.form_submit_button("¡Enviar Respuesta!")

            if submit_button:
                st.session_state.cnn_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_answer}) {st.session_state.cnn_current_options[user_answer]}"})
                prev_streak = st.session_state.cnn_correct_streak
                is_correct = (user_answer == st.session_state.cnn_correct_answer)

                if is_correct:
                    st.session_state.cnn_correct_streak += 1
                else:
                    st.session_state.cnn_correct_streak = 0

                radio_placeholder.empty()

                if st.session_state.cnn_correct_streak > 0 and \
                   st.session_state.cnn_correct_streak % 3 == 0 and \
                   st.session_state.cnn_correct_streak > prev_streak:
                    
                    if st.session_state.cnn_correct_streak < 9:
                        current_level_text = ""
                        if st.session_state.cnn_correct_streak == 3:
                            current_level_text = "Medio (como un adolescente que ya entiende cómo ven las CNNs)"
                        elif st.session_state.cnn_correct_streak == 6:
                            current_level_text = "Avanzado (como un buen detective de CNNs)"
                        
                        level_up_message = f"¡Increíble! ¡Has respondido {st.session_state.cnn_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de CNNs. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a detective de imágenes! 🚀"
                        st.session_state.cnn_game_messages.append({"role": "assistant", "content": level_up_message})
                        st.balloons()
                        try:
                            tts_level_up = gTTS(text=level_up_message, lang='es', slow=False)
                            audio_fp_level_up = io.BytesIO()
                            tts_level_up.write_to_fp(audio_fp_level_up)
                            audio_fp_level_up.seek(0)
                            st.audio(audio_fp_level_up, format="audio/mp3", start_time=0)
                            time.sleep(2)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de subida de nivel: {e}")
                    elif st.session_state.cnn_correct_streak >= 9:
                        medals_earned = (st.session_state.cnn_correct_streak - 6) // 3 
                        medal_message = f"🏅 ¡FELICITACIONES, MAESTRO/A CNN! ¡Has ganado tu {medals_earned}ª Medalla de Detección de Imágenes! ¡Tu habilidad es asombrosa y digna de un verdadero EXPERTO en CNNs! ¡Sigue así! 🌟"
                        st.session_state.cnn_game_messages.append({"role": "assistant", "content": medal_message})
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
                        
                        if prev_streak < 9:
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Maestro CNN**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros ingenieros de visión por computadora! ¡Adelante!"
                            st.session_state.cnn_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                audio_fp_level_up_champion = io.BytesIO()
                                tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                                audio_fp_level_up_champion.seek(0)
                                st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0)
                                time.sleep(2)
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón: {e}")


                with st.spinner("Detective Pixel está pensando su respuesta..."):
                    try:
                        feedback_prompt = f"""
                        El usuario respondió '{user_answer}'. La pregunta era: '{st.session_state.cnn_current_question}'.
                        La respuesta correcta era '{st.session_state.cnn_correct_answer}'.
                        Da feedback como Detective Pixel.
                        Si es CORRECTO, el mensaje es "¡Acertaste como un detective experto!" o similar.
                        Si es INCORRECTO, el mensaje es "¡Necesitas más pistas, aspirante a detective!" o similar.
                        Luego, una explicación sencilla para el usuario.
                        Finalmente, pregunta: "¿Quieres seguir explorando el mundo de las CNNs?".
                        **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                        """
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": cnn_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=300
                        )
                        raw_cnn_feedback_text = feedback_response.choices[0].message.content
                        feedback_message, explanation_message, continue_question = parse_cnn_feedback_response(raw_cnn_feedback_text)
                        
                        st.session_state.cnn_game_messages.append({"role": "assistant", "content": feedback_message})
                        st.session_state.cnn_game_messages.append({"role": "assistant", "content": explanation_message})
                        st.session_state.cnn_game_messages.append({"role": "assistant", "content": continue_question})

                        try:
                            tts = gTTS(text=f"{feedback_message}. {explanation_message}. {continue_question}", lang='es', slow=False)
                            audio_fp = io.BytesIO() # Cambio de 'fp' a 'audio_fp' para evitar conflicto
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")


                        st.session_state.cnn_current_question = None
                        st.session_state.cnn_current_options = {}
                        st.session_state.cnn_correct_answer = None
                        st.session_state.cnn_game_needs_new_question = False
                        st.session_state.cnn_awaiting_next_game_decision = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error al comunicarse con la API de OpenAI para el feedback: {e}")
                        st.session_state.cnn_game_active = False
                        st.session_state.cnn_game_messages.append({"role": "assistant", "content": "Lo siento, no puedo darte feedback ahora mismo. ¡Por favor, reinicia el juego!"})
                        st.rerun()

        if st.session_state.cnn_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col_continue, col_end = st.columns(2)
            with col_continue:
                if st.button("👍 Sí, quiero seguir explorando!", key="continue_cnn_game"):
                    st.session_state.cnn_awaiting_next_game_decision = False
                    st.session_state.cnn_game_needs_new_question = True
                    st.session_state.cnn_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío!"})
                    st.rerun()
            with col_end:
                if st.button("👎 No, gracias! Quiero descansar.", key="end_cnn_game"):
                    st.session_state.cnn_game_active = False
                    st.session_state.cnn_awaiting_next_game_decision = False
                    st.session_state.cnn_game_messages.append({"role": "assistant", "content": "¡Gracias por jugar! ¡Vuelve pronto para seguir investigando el mundo de las CNNs!"})
                    st.rerun()

else:
    st.info("El chatbot Detective Pixel no está disponible porque la clave de la API de OpenAI no está configurada.")

st.write("---")