import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pydotplus
import json
from streamlit_lottie import st_lottie
from openai import OpenAI
from gtts import gTTS
import io
import time
from io import StringIO # Importación necesaria para export_graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz # Importaciones necesarias para el árbol de decisión

st.set_page_config(
    page_title="Árboles de Decisión - Academia de Agentes IA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Configuración de la API de OpenAI (inicializada una vez por sesión en st.session_state) ---
if "openai_client" not in st.session_state:
    try:
        # Intenta obtener la clave secreta
        api_key = st.secrets["OPENAI_API_KEY"]
        # Inicializa el cliente de OpenAI y lo guarda en session_state
        st.session_state.openai_client = OpenAI(api_key=api_key) # <--- ¡Aquí el cambio! Quitar 'openai.'
    except KeyError:
        # Si la clave no se encuentra, establece el cliente en None y muestra un mensaje de error
        st.session_state.openai_client = None
        st.error("¡Advertencia! La clave de API de OpenAI no está configurada. Algunas funcionalidades (como los chatbots) no estarán disponibles.")
        st.info("Por favor, configura 'OPENAI_API_KEY' en la sección 'Secrets' de tu aplicación en Streamlit Cloud.")

# --- Obtener la ruta base del proyecto ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Función para cargar animaciones Lottie locales ---
def load_lottiefile(filepath: str):
    """Carga un archivo JSON de animación Lottie desde una ruta local."""
    absolute_filepath = os.path.join(PROJECT_ROOT, filepath)

    try:
        with open(absolute_filepath, "r", encoding="utf8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Advertencia: No se encontró el archivo Lottie en la ruta: {absolute_filepath}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: El archivo Lottie '{absolute_filepath}' no es un JSON válido.")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar el archivo Lottie '{absolute_filepath}': {e}. Asegúrate de que el archivo no esté corrupto y sea un JSON válido.")
        return None

# --- Rutas a Lottie y Imágenes ---
LOTTIE_ARBOL_RELATIVE_PATH = os.path.join("assets", "lottie_animations", "arbol.json")
GAME_IMAGES_RELATIVE_PATH = os.path.join("assets", "imagenes")

# --- Definición del Árbol de Decisión del Juego ---
decision_tree_data = {
    "q1": {
        "text": "¿Tu animal vive principalmente en el agua (océano, río, lago)?",
        "options": {
            "Sí": {"next_question": "q2_agua"},
            "No": {"next_question": "q2_tierra_aire"}
        }
    },
    "q2_agua": {
        "text": "¿Tu animal tiene aletas o es un animal nadador sin patas ni alas?",
        "options": {
            "Sí": {"next_question": "q3_marino"},
            "No": {"result": "Tortuga", "image": "tortuga.png"}
        }
    },
    "q3_marino": {
        "text": "¿Tu animal tiene muchos brazos (tentáculos)?",
        "options": {
            "Sí": {"result": "Pulpo", "image": "pulpo.png"},
            "No": {"next_question": "q4_pez_grande"}
        }
    },
    "q4_pez_grande": {
        "text": "¿Tu animal es un pez muy grande y temible con dientes afilados?",
        "options": {
            "Sí": {"result": "Tiburón", "image": "tiburon.png"},
            "No": {"result": "Pez", "image": "pez.png"}
        }
    },
    "q2_tierra_aire": {
        "text": "¿Tu animal puede volar o tiene alas?",
        "options": {
            "Sí": {"next_question": "q3_vuela"},
            "No": {"next_question": "q3_no_vuela"}
        }
    },
    "q3_vuela": {
        "text": "¿Tu animal vuela principalmente de noche y usa sus orejas para ver?",
        "options": {
            "Sí": {"result": "Murciélago", "image": "murcielago.png"},
            "No": {"next_question": "q4_vuela_plumas"}
        }
    },
    "q4_vuela_plumas": {
        "text": "¿Tu animal tiene plumas?",
        "options": {
            "Sí": {"result": "Pájaro", "image": "pajaro.png"},
            "No": {"result": "Mariposa", "image": "mariposa.png"}
        }
    },
    "q3_no_vuela": {
        "text": "¿Tu animal tiene la piel cubierta de pelo o pelaje?",
        "options": {
            "Sí": {"next_question": "q4_pelaje"},
            "No": {"next_question": "q4_sin_pelaje"}
        }
    },
    "q4_pelaje": {
        "text": "¿Tu animal tiene rayas blancas y negras?",
        "options": {
            "Sí": {"result": "Cebra", "image": "cebra.png"},
            "No": {"next_question": "q5_grande_felino"}
        }
    },
    "q5_grande_felino": {
        "text": "¿Tu animal es un felino grande y ruge?",
        "options": {
            "Sí": {"result": "León", "image": "leon.png"},
            "No": {"next_question": "q6_trompa"}
        }
    },
    "q6_trompa": {
        "text": "¿Tu animal tiene una trompa muy larga?",
        "options": {
            "Sí": {"result": "Elefante", "image": "elefante.png"},
            "No": {"result": "Oso", "image": "oso.png"}
        }
    },
    "q4_sin_pelaje": {
        "text": "¿Tu animal tiene escamas o piel muy dura y arrastra el cuerpo?",
        "options": {
            "Sí": {"next_question": "q5_reptil_forma"},
            "No": {"result": "Rana", "image": "rana.png"}
        }
    },
    "q5_reptil_forma": {
        "text": "¿Tu animal tiene una forma alargada y no tiene patas?",
        "options": {
            "Sí": {"result": "Serpiente", "image": "serpiente.png"},
            "No": {"result": "Cocodrilo", "image": "cocodrilo.png"}
        }
    }
}


# --- Funciones del Juego (sin cambios) ---
def start_game():
    st.session_state.game_state = "playing"
    st.session_state.current_question_id = "q1" # Empezar por la primera pregunta
    st.session_state.animal_path = [] # Para guardar las respuestas dadas
    st.session_state.adivinado_animal_info = None # Para guardar el resultado final

def reset_game():
    # Eliminar todas las claves relacionadas con el juego de session_state
    keys_to_delete = ["game_state", "current_question_id", "animal_path", "adivinado_animal_info"]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun() # Reiniciar la página para empezar de cero

# --- Sidebar (Menú de navegación) ---
st.sidebar.title("Home")
st.sidebar.markdown("""
- Que es Inteligencia Artificial
- Que son Modelos Predictivos
- Que es EDA
- Que es Regresión Lineal
- Que es Clasificación
- Que es Clustering (k-means)
- Que son Árboles de Decisión
- Que es K-Nearest Neighbors (KNN)
- Que son Máquinas de Vectores de So...
- Que son Métodos Ensemble (Random...
- Que es Deep Learning
- Que son Redes Neuronales Artificiales
- Que son Redes Neuronales Convoluci...
- Que son Redes Neuronales Recurrent...
- Que son Mapas Autoorganizados (SOM)
- Que son Redes Generativas Antagónic...
- Que es IA Generativa
""")

# --- Contenido Principal de la Página ---
st.title("🌳 ¿Qué son los Árboles de Decisión?")

st.markdown("""
¡Hola, futuro experto en Inteligencia Artificial! Hoy vamos a aprender sobre un tipo de "árbol" muy especial: ¡los Árboles de Decisión!
""")

# --- la animación Lottie principal ---
lottie_arbol = load_lottiefile(LOTTIE_ARBOL_RELATIVE_PATH)

if lottie_arbol:
    st_lottie(
        lottie_arbol,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        height=300, # Ajustar altura para dejar espacio
        width=None,
        key="arbol_decision_lottie",
    )
else:
    st.warning(f"No se pudo cargar la animación Lottie principal. Asegúrate de que la ruta {LOTTIE_ARBOL_RELATIVE_PATH} es correcta y que el archivo JSON es válido.")


st.markdown("""
Imagina que un Árbol de Decisión es como un juego de "Adivina Quién". Cada vez que llegas a una "rama" del árbol, haces una pregunta. Dependiendo de la respuesta, sigues un camino diferente hasta llegar a una "hoja" que te da la respuesta final.

**¿Para qué sirven?** Los Árboles de Decisión nos ayudan a tomar decisiones o a clasificar cosas basándose en diferentes características. Por ejemplo, pueden ayudarnos a decidir si debemos llevar un paraguas (¿está lloviendo?), si un animal es un perro o un gato (¿tiene bigotes?), o incluso a predecir si un cliente comprará un producto.
""")

st.subheader("¡Vamos a jugar a un juego de Árbol de Decisión para niños!")

st.markdown("""
Imagina que eres un detective de animales y tienes que adivinar qué animal es. ¡Te haremos preguntas y tú seguirás el camino correcto!
""")

# --- Juego Interactivo de Adivina el Animal ---
st.write("---")
st.subheader("El Detective de Animales: ¡Adivina el Animal!")

# Inicializar el estado del juego si no existe
if "game_state" not in st.session_state:
    st.session_state.game_state = "start"

if st.session_state.game_state == "start":
    st.info("¡Presiona el botón para empezar a adivinar tu animal secreto!")
    if st.button("🚀 ¡Empezar a Jugar!", key="start_game_button"):
        start_game()
        st.rerun() # Recargar para ir al estado de juego

elif st.session_state.game_state == "playing":
    current_question = decision_tree_data.get(st.session_state.current_question_id)

    if current_question:
        st.markdown(f"**Pregunta: {current_question['text']}**")

        options = list(current_question["options"].keys())

        # Usamos st.radio para que el niño elija una opción
        selected_option = st.radio("Elige tu respuesta:", options, key=st.session_state.current_question_id)

        if st.button("➡️ ¡Siguiente!", key=f"next_q_{st.session_state.current_question_id}"):
            st.session_state.animal_path.append(selected_option) # Guardar la respuesta

            next_step = current_question["options"][selected_option]

            if "next_question" in next_step:
                st.session_state.current_question_id = next_step["next_question"]
            elif "result" in next_step:
                st.session_state.game_state = "result"
                st.session_state.adivinado_animal_info = next_step
            st.rerun() # Recargar para mostrar la siguiente pregunta o el resultado

    else:
        st.error("¡Ups! Parece que algo salió mal con las preguntas. ¡Reinicia el juego!")
        if st.button("Reiniciar Juego", key="error_reset_game"):
            reset_game()


elif st.session_state.game_state == "result":
    animal_info = st.session_state.adivinado_animal_info
    if animal_info:
        st.success(f"🥳 ¡Has adivinado! ¡Eres un **{animal_info['result']}**!")
        st.markdown("¿Ves? Cada pregunta te lleva a un camino diferente, ¡igual que en un árbol de decisión!")

        # Mostrar imagen del animal
        # Construimos la ruta absoluta para la imagen
        image_absolute_path = os.path.join(PROJECT_ROOT, GAME_IMAGES_RELATIVE_PATH, animal_info['image'])

        if os.path.exists(image_absolute_path):
            st.image(image_absolute_path, caption=animal_info['result'], use_container_width=False, width=300) # Ajusta el ancho
        else:
            st.warning(f"Imagen '{animal_info['image']}' no encontrada en la ruta: '{image_absolute_path}'.") # Mensaje más útil

        # Opcional: Generar dato curioso con OpenAI y reproducirlo
        if st.session_state.openai_client: # Asegurarse de que el cliente esté disponible
            st.markdown("---")
            st.subheader("¡Dato curioso de tu animal!")
            try:
                with st.spinner("Generando dato curioso..."):
                    response = st.session_state.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo", # Puedes probar con "gpt-4" si tienes acceso y quieres más calidad
                        messages=[
                            {"role": "system", "content": "Actúa como un experto en animales muy amigable para niños."},
                            {"role": "user", "content": f"Dame una curiosidad muy corta y divertida sobre un {animal_info['result']} para niños (máximo 20 palabras)."}
                        ],
                        max_tokens=50,
                        temperature=0.7
                    )
                    curiosidad = response.choices[0].message.content
                    st.info(f"¡Sabías que... {curiosidad}!")

                # Reproducir audio del dato curioso
                with st.spinner("Generando audio..."):
                    tts = gTTS(curiosidad, lang='es', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format='audio/mp3', start_time=0)
            except Exception as e:
                st.warning(f"No pude generar un dato curioso o audio en este momento: {e}")


        st.markdown("---")
        st.write("¿Quieres volver a jugar y adivinar otro animal?")
        if st.button("¡Jugar de Nuevo!", key="reset_game_button_result"):
            reset_game()
    else:
        st.error("No se pudo determinar el resultado. ¡Reinicia el juego!")
        if st.button("Reiniciar Juego", key="no_result_reset_game"):
            reset_game()

st.write("---")


st.subheader("Un ejemplo real de cómo funciona un Árbol de Decisión (¡para futuros científicos de datos!)")
st.markdown("""
Ahora que hemos jugado, te mostraremos cómo los científicos usan los árboles de decisión para hacer predicciones.
Vamos a usar un ejemplo donde queremos saber **"¿Qué tipo de mascota es?"** basándonos en sus características.
""")

st.markdown("### Nuestros datos de ejemplo:")

pet_data_current_behavior = {
    'Tiene_Pelo': [
        'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí',  # 10 con pelo
        'No', 'No', 'No', 'No', 'No',                              # 5 sin pelo (Pájaros)
        'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', # Otros 10 con pelo
        'No', 'No', 'No', 'No', 'No'                               # Otros 5 sin pelo (Pájaros)
    ],
    'Tamaño_Pequeño': [
        'No', 'Sí', 'No', 'Sí', 'No', 'Sí', 'No', 'Sí', 'No', 'Sí', # Variedad para con pelo
        'Sí', 'No', 'Sí', 'No', 'Sí',                               # Variedad para sin pelo
        'No', 'Sí', 'No', 'Sí', 'No', 'Sí', 'No', 'Sí', 'No', 'Sí',
        'Sí', 'No', 'Sí', 'No', 'Sí'
    ],
    'Hace_Sonido': [
        'Guau', 'Miau', 'Guau', 'Miau', 'Guau', 'Miau', 'Guau', 'Miau', 'Guau', 'Miau', # Mezcla de sonidos para con pelo
        'Pío', 'Pío', 'Pío', 'Pío', 'Pío',                                            # Solo "Pío" para sin pelo (garantiza pureza de pájaro)
        'Guau', 'Miau', 'Guau', 'Miau', 'Guau', 'Miau', 'Guau', 'Miau', 'Guau', 'Miau',
        'Pío', 'Pío', 'Pío', 'Pío', 'Pío'
    ],
    'Mascota': [
        'Perro', 'Gato', 'Perro', 'Gato', 'Perro', 'Gato', 'Perro', 'Gato', 'Perro', 'Gato', # Perros y Gatos (con pelo)
        'Pájaro', 'Pájaro', 'Pájaro', 'Pájaro', 'Pájaro',                                            # Solo Pájaros (sin pelo, grupo puro)
        'Perro', 'Gato', 'Perro', 'Gato', 'Perro', 'Gato', 'Perro', 'Gato', 'Perro', 'Gato',
        'Pájaro', 'Pájaro', 'Pájaro', 'Pájaro', 'Pájaro'
    ]
}
df_pet = pd.DataFrame(pet_data_current_behavior)
st.dataframe(df_pet)

st.markdown("""
Para que el ordenador entienda estos datos y pueda "dibujar" el árbol, necesita convertirlos a números. ¡Es como traducir un idioma! Cada característica (como 'Tiene Pelo' o 'Guau') se convierte en un número.
""")

df_pet_encoded = df_pet.copy()

# Mapeos explícitos para mayor claridad en la visualización y consistencia con class_names
pelo_mapping = {'No': 0, 'Sí': 1}
tamano_mapping = {'No': 0, 'Sí': 1}
sonido_mapping = {'Miau': 0, 'Guau': 1, 'Pío': 2}
mascota_output_mapping = {'Gato': 0, 'Perro': 1, 'Pájaro': 2}

df_pet_encoded['Tiene_Pelo'] = df_pet_encoded['Tiene_Pelo'].map(pelo_mapping)
df_pet_encoded['Tamaño_Pequeño'] = df_pet_encoded['Tamaño_Pequeño'].map(tamano_mapping)
df_pet_encoded['Hace_Sonido'] = df_pet_encoded['Hace_Sonido'].map(sonido_mapping)
df_pet_encoded['Mascota'] = df_pet_encoded['Mascota'].map(mascota_output_mapping)

st.dataframe(df_pet_encoded)

# Separar características (X) y objetivo (y)
X_pet = df_pet_encoded[['Tiene_Pelo', 'Tamaño_Pequeño', 'Hace_Sonido']]
y_pet = df_pet_encoded['Mascota']

# Entrenar el Árbol de Decisión
model_pet = DecisionTreeClassifier(criterion='entropy', random_state=42)
model_pet.fit(X_pet, y_pet)

try:
    # Visualizar el árbol
    dot_data_pet = StringIO()
    export_graphviz(model_pet, out_file=dot_data_pet,
                    feature_names=X_pet.columns,
                    class_names=['Gato', 'Perro', 'Pájaro'],
                    filled=True, rounded=True,
                    special_characters=True)

    graph_pet = pydotplus.graph_from_dot_data(dot_data_pet.getvalue())
    tree_image_path_pet = 'decision_tree_pet.png' # Esto creará el PNG en el directorio de trabajo actual
    graph_pet.write_png(tree_image_path_pet)

    st.image(tree_image_path_pet, caption='Nuestro Árbol de Decisión para "Adivinar la Mascota"', use_container_width=True)

    # --- EXPLICACIÓN DEL GRÁFICO ---
    st.markdown("---")
    st.subheader("¡Entendiendo el Árbol de Decisión para adivinar mascotas!")
    st.markdown("""
    Mira el gráfico del árbol que aparece arriba. ¡Es un mapa para adivinar qué mascota es!

    **Cada caja (o "nodo") es una pregunta.** Las preguntas te guían por el árbol hasta que llegas a una respuesta final.

    **Vamos a ver cómo funciona, paso a paso, como si estuviéramos buscando una mascota:**

    1.  **Empezamos arriba, en la primera caja (el "nodo raíz").** Aquí se hace la pregunta más importante para diferenciar a los animales. En nuestro árbol, la primera pregunta es: **`Hace_Sonido <= 0.5`**
        * ¿Recuerdas que tradujimos 'Miau' a 0, 'Guau' a 1 y 'Pío' a 2 para 'Hace_Sonido'? Esta pregunta se traduce a: **"¿El animal hace 'Miau'?"** (es decir, el valor para 'Hace_Sonido' es 0, que es menor o igual a 0.5).
        * Si la respuesta es **SÍ** (el animal hace 'Miau'), seguimos la flecha `True` (hacia la izquierda).
            * Este camino lleva a una **hoja final** donde la `class` es **Gato**. ¡Así que si hace 'Miau', es un Gato!
        * Si la respuesta es **NO** (el animal hace 'Guau' o 'Pío'), seguimos la flecha `False` (hacia la derecha).

    3.  **Si fuimos por la derecha (el animal NO hace 'Miau', es decir, hace 'Guau' o 'Pío'):** Llegamos a otra nueva caja. Esta caja nos pregunta sobre el **"Tiene_Pelo"**.
        * La pregunta es `Tiene_Pelo <= 0.5`. ¿Recuerdas que 'No' es 0 y 'Sí' es 1 para 'Tiene_Pelo' ? Esta pregunta se traduce a: **"¿El animal NO tiene pelo?"** (es decir, el valor para 'Tiene_Pelo' es 0, que es menor o igual a 0.5).
        * Si la respuesta es **SÍ** (el animal NO tiene pelo), seguimos la flecha `True` (hacia la izquierda).
            * Este camino lleva a una **hoja final** donde la `class` es **Pájaro**. ¡Si no hace 'Miau' y no tiene pelo, es un Pájaro!
        * Si la respuesta es **NO** (el animal SÍ tiene pelo), seguimos la flecha `False` (hacia la derecha).
            * Este camino lleva a una **hoja final** donde la `class` es **Perro**. ¡Si no hace 'Miau' pero sí tiene pelo, es un Perro!

    **Las "hojas" (las cajas al final de las ramas que no se dividen más) son las respuestas finales.** La `class` que ves en cada hoja te dice qué tipo de mascota predice el árbol.

    **En resumen:** Este árbol usa preguntas sobre el sonido y el pelo para ayudarnos a adivinar qué tipo de mascota es, ¡empezando por el sonido!
    """)

except Exception as e:
    st.warning(f"No se pudo generar la imagen del Árbol de Decisión. Asegúrate de tener Graphviz instalado y configurado correctamente. Error: {e}")
    st.markdown("Puedes aprender más sobre la visualización de árboles de decisión en la documentación de scikit-learn o pydotplus.")


st.markdown("""
Esperamos que este ejemplo te haya ayudado a entender un poco mejor cómo funcionan los Árboles de Decisión. ¡Son como un mapa que te guía hacia la mejor decisión!
""")


# --- Sección de Chatbot de Juego con Arbolín ---
st.header("¡Juega y Aprende con Arbolín sobre los Árboles de Decisión!")
st.markdown("¡Hola! Soy Arbolín, el guardián de las decisiones. ¿Listo para recorrer los caminos de las preguntas y encontrar la respuesta correcta?")

# Inicializa el estado del juego y los mensajes del chat
if "tree_game_active" not in st.session_state:
    st.session_state.tree_game_active = False
if "tree_game_messages" not in st.session_state:
    st.session_state.tree_game_messages = []
if "tree_current_question" not in st.session_state:
    st.session_state.tree_current_question = None
if "tree_current_options" not in st.session_state:
    st.session_state.tree_current_options = {}
if "tree_correct_answer" not in st.session_state:
    st.session_state.tree_correct_answer = None
if "tree_awaiting_next_game_decision" not in st.session_state:
    st.session_state.tree_awaiting_next_game_decision = False
if "tree_game_needs_new_question" not in st.session_state:
    st.session_state.tree_game_needs_new_question = False
if "tree_correct_streak" not in st.session_state:
    st.session_state.tree_correct_streak = 0
if "last_played_question_arbolin_tree" not in st.session_state:
    st.session_state.last_played_question_arbolin_tree = None

arbolin_tree_game_system_prompt = f"""
Eres un **experto en Machine Learning y Ciencia de Datos**, especializado en el campo de los **Árboles de Decisión**. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de los Árboles de Decisión mediante un **juego de preguntas adaptativo**. Aunque el entorno inicial sea "para niños", tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

**TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
**¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

**Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
Pregunta: [Tu pregunta aquí]
A) [Opción A]
B) [Opción B]
C) [Opción C]
RespuestaCorrecta: [A, B o C]

**Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
[Mensaje de Correcto/Incorrecto, ej: "¡Análisis impecable! Has optimizado tu modelo de conocimiento." o "Revisa tu algoritmo. Esa no era la decisión óptima."]
[Breve explicación del concepto, adecuada al nivel del usuario, ej: "Un árbol de decisión particiona el espacio de características..."]
[Pregunta para continuar, ej: "¿Listo para el siguiente desafío en el ámbito de los clasificadores?" o "¿Quieres profundizar más en la teoría de la información aplicada a los árboles?"]

**Reglas adicionales para el Experto en Árboles de Decisión:**
* **Enfoque Riguroso en Árboles de Decisión:** Todas tus preguntas y explicaciones deben girar en torno a los Árboles de Decisión. Cubre sus fundamentos (nodos, ramas, hojas, atributos), algoritmos de construcción (ID3, C4.5, CART), métricas de división (ganancia de información, impureza Gini), sobreajuste (overfitting), poda (pruning), y su aplicación en clasificación y regresión.
* **¡VARIEDAD, VARIEDAD, VARIEDAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de Árboles de Decisión que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
    * **Concepto General y Usos:** Definición, casos de uso (clasificación, regresión), ventajas (interpretabilidad).
    * **Estructura del Árbol:** Nodos (raíz, internos, hoja), ramas, condiciones de división (splits), atributos.
    * **Algoritmos de Construcción:** Principios de ID3, C4.5, CART.
    * **Métricas de Impureza/Ganancia:** Entropía, Ganancia de Información, Impureza Gini (qué miden y por qué se usan).
    * **Proceso de Decisión y Clasificación:** Cómo un dato atraviesa el árbol hasta una clase o valor.
    * **Preprocesamiento y Datos:** Manejo de variables categóricas/numéricas, valores perdidos.
    * **Sobreajuste y Poda:** Qué es el overfitting en árboles, métodos de poda (pre-pruning, post-pruning).
    * **Ensembles (Introducción):** Breve mención de Random Forests o Gradient Boosting como extensiones.
    * **Ventajas y Desventajas:** Robustez, sesgo, varianza, estabilidad.

* **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.tree_correct_streak} preguntas correctas consecutivas.
    * **Nivel 1 (Aprendiz – 0-2 respuestas correctas):** Tono introductorio, analogías simples. Preguntas sobre la función básica de un árbol de decisión (tomar decisiones secuenciales). Ejemplos muy claros y conceptuales.
        * *Tono:* "Eres un explorador que empieza a entender el mapa de decisiones."
    * **Nivel 2 (Desarrollador Junior – 3-5 respuestas correctas):** Tono más técnico. Introduce conceptos como nodos, ramas y hojas, pero de forma directa. Preguntas sobre la estructura y el flujo de decisión.
        * *Tono:* "Has completado tu primer sprint de modelado de datos."
    * **Nivel 3 (Científico de Datos – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Introduce métricas de impureza (sin entrar en fórmulas complejas inicialmente), sobreajuste, y cómo el árbol "aprende". Preguntas que requieren una comprensión más profunda de los mecanismos internos y desafíos.
        * *Tono:* "Tu análisis demuestra una comprensión sólida de los algoritmos de clasificación."
    * **Nivel Maestro (Experto en ML – 9+ respuestas correctas):** Tono de **especialista en Machine Learning**. Preguntas sobre algoritmos específicos (ID3 vs CART), poda, manejo de datos complejos, sesgos, o la base intuitiva de la entropía/ganancia de información. Se esperan respuestas que demuestren una comprensión teórica y práctica robusta.
        * *Tono:* "Tu maestría en el diseño de modelos predictivos es excepcional. Estás listo para enfrentar cualquier conjunto de datos."
    * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
    * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
    * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

* **Ejemplos y Analogías (Adaptadas al Nivel):**
    * **Nivel 1:** Un diagrama de flujo para organizar juguetes.
    * **Nivel 2:** Un algoritmo para clasificar correos electrónicos como spam/no spam.
    * **Nivel 3:** Un modelo para predecir la propensión de un cliente a comprar un producto, analizando atributos demográficos y de comportamiento.
    * **Nivel Maestro:** La optimización de un árbol de decisión para un problema de diagnóstico médico, considerando la interpretabilidad y la robustez frente a datos ruidosos.

* **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
* **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
* **Siempre responde en español de España.**
* **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre los ÁRBOLES DE DECISIÓN, y asegúrate de que no se parezca a las anteriores.
"""

# Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
def parse_tree_question_response(raw_text):
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
def parse_tree_feedback_response(raw_text):
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    if len(lines) >= 3:
        return lines[0], lines[1], lines[2]
    st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
    return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"

# --- Funciones para subir de nivel directamente ---
def set_arbolin_tree_level(target_streak, level_name):
    st.session_state.tree_correct_streak = target_streak
    st.session_state.tree_game_active = True
    st.session_state.tree_game_messages = []
    st.session_state.tree_current_question = None
    st.session_state.tree_current_options = {}
    st.session_state.tree_correct_answer = None
    st.session_state.tree_game_needs_new_question = True
    st.session_state.tree_awaiting_next_game_decision = False
    st.session_state.tree_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}** de Arbolín! Prepárate para preguntas más desafiantes sobre los Árboles de Decisión. ¡Aquí va tu primera!"})
    st.rerun()

# Botones para iniciar o reiniciar el juego y subir de nivel
col_game_buttons_arbolin_tree, col_level_up_buttons_arbolin_tree = st.columns([1, 2])

with col_game_buttons_arbolin_tree:
    if st.button("¡Vamos a jugar con Arbolín!", key="start_arbolin_tree_game_button"):
        st.session_state.tree_game_active = True
        st.session_state.tree_game_messages = []
        st.session_state.tree_current_question = None
        st.session_state.tree_current_options = {}
        st.session_state.tree_correct_answer = None
        st.session_state.tree_game_needs_new_question = True
        st.session_state.tree_awaiting_next_game_decision = False
        st.session_state.tree_correct_streak = 0
        st.session_state.last_played_question_arbolin_tree = None
        st.rerun()

with col_level_up_buttons_arbolin_tree:
    st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un experto en decisiones? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
    col_lvl1_arbolin_tree, col_lvl2_arbolin_tree, col_lvl3_arbolin_tree = st.columns(3) # Tres columnas para los botones de nivel
    with col_lvl1_arbolin_tree:
        if st.button("Subir a Nivel Medio (Arbolín - Explorador)", key="level_up_medium_arbolin_tree"):
            set_arbolin_tree_level(3, "Medio") # 3 respuestas correctas para Nivel Medio
    with col_lvl2_arbolin_tree:
        if st.button("Subir a Nivel Avanzado (Arbolín - Guardián)", key="level_up_advanced_arbolin_tree"):
            set_arbolin_tree_level(6, "Avanzado") # 6 respuestas correctas para Nivel Avanzado
    with col_lvl3_arbolin_tree:
        if st.button("👑 ¡Maestro de Decisiones! (Arbolín)", key="level_up_champion_arbolin_tree"):
            set_arbolin_tree_level(9, "Campeón") # 9 respuestas correctas para Nivel Campeón


# Mostrar mensajes del juego del chatbot
for message in st.session_state.tree_game_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Lógica del juego del chatbot si está activo
if st.session_state.tree_game_active:
    if st.session_state.tree_current_question is None and st.session_state.tree_game_needs_new_question and not st.session_state.tree_awaiting_next_game_decision:
        with st.spinner("Arbolín está preparando una pregunta sobre árboles de decisión..."):
            try:
                # Usa st.session_state.openai_client
                if st.session_state.openai_client is None:
                    st.error("Error: El cliente de OpenAI no está inicializado. Asegúrate de que tu clave de API esté configurada.")
                    st.session_state.tree_game_active = False
                    st.rerun()

                game_messages_for_api = [{"role": "system", "content": arbolin_tree_game_system_prompt}]
                if st.session_state.tree_game_messages:
                    last_message = st.session_state.tree_game_messages[-1]
                    if last_message["role"] == "user":
                        game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {last_message['content']}"})
                    elif last_message["role"] == "assistant":
                        game_messages_for_api.append({"role": "assistant", "content": last_message['content']})

                game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre QUÉ SON LOS ÁRBOLES DE DECISIÓN siguiendo el formato exacto. ¡Recuerda, la pregunta debe ser muy VARIADA y CREATIVA, y no se debe parecer a las anteriores!"})

                game_response = st.session_state.openai_client.chat.completions.create( # Referencia directa a st.session_state.openai_client
                    model="gpt-4o-mini",
                    messages=game_messages_for_api,
                    temperature=0.8,
                    max_tokens=300
                )
                raw_tree_question_text = game_response.choices[0].message.content
                question, options, correct_answer_key = parse_tree_question_response(raw_tree_question_text)

                if question:
                    st.session_state.tree_current_question = question
                    st.session_state.tree_current_options = options
                    st.session_state.tree_correct_answer = correct_answer_key

                    display_question_text = f"**Nivel {int(st.session_state.tree_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.tree_correct_streak}**\n\n**Pregunta de Arbolín:** {question}\n\n"
                    for key in sorted(options.keys()):
                        display_question_text += f"{key}) {options[key]}\n"

                    st.session_state.tree_game_messages.append({"role": "assistant", "content": display_question_text})
                    st.session_state.tree_game_needs_new_question = False
                    st.rerun()
                else:
                    st.session_state.tree_game_messages.append({"role": "assistant", "content": "¡Lo siento! Arbolín no pudo preparar la pregunta en el formato correcto. ¿Puedes pulsar 'VAMOS A JUGAR' de nuevo?"})
                    st.session_state.tree_game_active = False
                    st.rerun()

            except Exception as e:
                st.error(f"¡Oops! Arbolín no pudo hacer la pregunta. Error: {e}")
                st.session_state.tree_game_messages.append({"role": "assistant", "content": "¡Lo siento! Arbolín tiene un pequeño problema para hacer preguntas ahora. ¿Puedes intentarlo de nuevo?"})
                st.session_state.tree_game_active = False
                st.rerun()


    if st.session_state.tree_current_question is not None and not st.session_state.tree_awaiting_next_game_decision:
        # Audio de la pregunta
        if st.session_state.get('last_played_question_arbolin_tree') != st.session_state.tree_current_question:
            try:
                tts_text = f"Nivel {int(st.session_state.tree_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.tree_correct_streak}. Pregunta de Arbolín: {st.session_state.tree_current_question}. Opción A: {st.session_state.tree_current_options.get('A', '')}. Opción B: {st.session_state.tree_current_options.get('B', '')}. Opción C: {st.session_state.tree_current_options.get('C', '')}."
                tts = gTTS(text=tts_text, lang='es', slow=False)
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                st.session_state.last_played_question_arbolin_tree = st.session_state.tree_current_question
            except Exception as e:
                st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")


        with st.form("arbolin_tree_game_form", clear_on_submit=True):
            radio_placeholder = st.empty()
            with radio_placeholder.container():
                st.markdown("Elige tu respuesta:")
                user_choice = st.radio(
                    "Elige tu respuesta:",
                    options=list(st.session_state.tree_current_options.keys()),
                    format_func=lambda x: f"{x}) {st.session_state.tree_current_options[x]}",
                    key="tree_answer_radio_buttons",
                    label_visibility="collapsed"
                )

            submit_button = st.form_submit_button("Enviar Respuesta")

        if submit_button:
            st.session_state.tree_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_choice}) {st.session_state.tree_current_options[user_choice]}"})
            prev_streak = st.session_state.tree_correct_streak

            # Lógica para actualizar el contador de respuestas correctas
            if user_choice == st.session_state.tree_correct_answer:
                st.session_state.tree_correct_streak += 1
            else:
                st.session_state.tree_correct_streak = 0

            radio_placeholder.empty()

            # --- Lógica de subida de nivel ---
            if st.session_state.tree_correct_streak > 0 and \
               st.session_state.tree_correct_streak % 3 == 0 and \
               st.session_state.tree_correct_streak > prev_streak:

                if st.session_state.tree_correct_streak < 9: # Niveles Básico, Medio, Avanzado
                    current_level_text = ""
                    if st.session_state.tree_correct_streak == 3:
                        current_level_text = "Medio (como un adolescente que ya entiende de lógica de decisiones)"
                    elif st.session_state.tree_correct_streak == 6:
                        current_level_text = "Avanzado (como un Data Scientist junior)"

                    level_up_message = f"🎉 ¡Increíble! ¡Has respondido {st.session_state.tree_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de Árboles de Decisión. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a explorador/a de decisiones! 🚀"
                    st.session_state.tree_game_messages.append({"role": "assistant", "content": level_up_message})
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
                elif st.session_state.tree_correct_streak >= 9:
                    medals_earned = (st.session_state.tree_correct_streak - 6) // 3
                    medal_message = f"🏅 ¡FELICITACIONES, MAESTRO DE DECISIONES! ¡Has ganado tu {medals_earned}ª Medalla del Árbol! ¡Tu habilidad para seguir los caminos correctos es asombrosa y digna de un verdadero EXPERTO en Árboles de Decisión! ¡Sigue así! 🌟"
                    st.session_state.tree_game_messages.append({"role": "assistant", "content": medal_message})
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
                        level_up_message_champion = f"¡Has desbloqueado el **Nivel Campeón (Maestro de Decisiones)**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros científicos de datos que entienden los secretos de las decisiones algorítmicas! ¡Adelante!"
                        st.session_state.tree_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                        try:
                            tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                            audio_fp_level_up_champion = io.BytesIO()
                            tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                            audio_fp_level_up_champion.seek(0)
                            st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0, autoplay=True)
                            time.sleep(2)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de campeón: {e}")


            # Generar feedback de Arbolín
            with st.spinner("Arbolín está revisando tu respuesta..."):
                try:
                    # Usa st.session_state.openai_client
                    if st.session_state.openai_client is None:
                        st.error("Error: El cliente de OpenAI no está inicializado. No se puede generar feedback.")
                        st.session_state.tree_game_active = False
                        st.rerun()

                    feedback_prompt = f"""
                    El usuario respondió '{user_choice}'. La pregunta era: '{st.session_state.tree_current_question}'.
                    La respuesta correcta era '{st.session_state.tree_correct_answer}'.
                    Da feedback como Arbolín.
                    Si es CORRECTO, el mensaje es "¡Decisión acertada! ¡Has elegido bien el camino!" o similar.
                    Si es INCORRECTO, el mensaje es "¡Oh, ese camino no era el correcto! ¡No te preocupes, hay más ramas!" o similar.
                    Luego, una explicación sencilla para niños y adolescentes.
                    Finalmente, pregunta: "¿Quieres seguir explorando el bosque de decisiones?".
                    **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                    """
                    feedback_response = st.session_state.openai_client.chat.completions.create( # Referencia directa a st.session_state.openai_client
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": arbolin_tree_game_system_prompt},
                            {"role": "user", "content": feedback_prompt}
                        ],
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_arbolin_tree_feedback_text = feedback_response.choices[0].message.content

                    feedback_msg, explanation_msg, next_question_prompt = parse_tree_feedback_response(raw_arbolin_tree_feedback_text)

                    st.session_state.tree_game_messages.append({"role": "assistant", "content": feedback_msg})
                    st.session_state.tree_game_messages.append({"role": "assistant", "content": explanation_msg})
                    st.session_state.tree_game_messages.append({"role": "assistant", "content": next_question_prompt})

                    try:
                        tts = gTTS(text=f"{feedback_msg}. {explanation_msg}. {next_question_prompt}", lang='es', slow=False)
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        audio_fp.seek(0)
                        st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                    except Exception as e:
                        st.warning(f"No se pudo reproducir el audio de feedback: {e}")


                    st.session_state.tree_current_question = None
                    st.session_state.tree_current_options = {}
                    st.session_state.tree_correct_answer = None
                    st.session_state.tree_game_needs_new_question = False
                    st.session_state.tree_awaiting_next_game_decision = True

                    st.rerun()

                except Exception as e:
                    st.error(f"Ups, Arbolín no pudo procesar tu respuesta. Error: {e}")
                    st.session_state.tree_game_messages.append({"role": "assistant", "content": "Lo siento, Arbolín tiene un pequeño problema técnico ahora mismo. ¡Pero me encantaría ver tu respuesta!"})


    if st.session_state.tree_awaiting_next_game_decision:
        st.markdown("---")
        st.markdown("¿Qué quieres hacer ahora?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Sí, quiero jugar más preguntas", key="play_more_questions_tree"):
                st.session_state.tree_game_needs_new_question = True
                st.session_state.tree_awaiting_next_game_decision = False
                st.session_state.tree_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío en el bosque de decisiones!"})
                st.rerun()
        with col2:
            if st.button("👎 No, ya no quiero jugar más", key="stop_playing_tree"):
                st.session_state.tree_game_active = False
                st.session_state.tree_awaiting_next_game_decision = False
                st.session_state.tree_game_messages.append({"role": "assistant", "content": "¡De acuerdo! ¡Gracias por explorar el bosque de decisiones conmigo! Espero que hayas aprendido mucho. ¡Hasta la próxima decisión!"})
                st.rerun()

else:
    # Usa st.session_state.openai_client
    if st.session_state.openai_client is None:
        st.info("Para usar la sección de preguntas de Arbolín, necesitas configurar tu clave de API de OpenAI en `secrets.toml`.")

