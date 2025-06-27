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

st.set_page_config(
    page_title="¿Qué es K-Nearest Neighbors (KNN)?",
    layout="wide"
)

# --- Función para cargar animación Lottie desde un archivo local ----
def load_lottiefile(filepath: str):
    """Carga un archivo JSON de animación Lottie desde una ruta local."""
    try:
        with open(filepath, "r", encoding="utf8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo Lottie en la ruta: {filepath}. Por favor, verifica la ruta y asegúrate de que el archivo existe.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: El archivo Lottie '{filepath}' no es un JSON válido. Revisa su contenido.")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar el archivo Lottie '{filepath}': {e}. Asegúrate de que el archivo no esté corrupto y sea un JSON válido.")
        return None

# --- Rutas a Lottie ---
LOTTIE_KNN_PATH = os.path.join("assets", "lottie_animations", "Group.json")

# --- Configuración de la API de OpenAI ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except AttributeError:
    client = None
    st.error("¡Advertencia! No se encontró la clave de API de OpenAI. Algunas funcionalidades (como el juego) no estarán disponibles. Por favor, configura 'OPENAI_API_KEY' en tu archivo .streamlit/secrets.toml")

# --- Sidebar (Menú de navegación) ---
st.sidebar.title("Home")
st.sidebar.markdown("""
- Que es Inteligencia Artificial
- Que son Modelos Predictivos
- Que es EDA
- Que es Regresion Lineal
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

st.title("¿Qué es K-Nearest Neighbors (KNN)?")

st.markdown("""
¡Hola, pequeño clasificador! Hoy vamos a descubrir un método de Inteligencia Artificial que es como un **"club de amigos"** que te ayuda a decidir a qué grupo pertenece algo nuevo. ¡Se llama **K-Nearest Neighbors (KNN)**!
""")

# --- Carga y muestra la animación Lottie para KNN ---
lottie_knn = load_lottiefile(LOTTIE_KNN_PATH)

if lottie_knn:
    st_lottie(
        lottie_knn,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        height=300,
        width=None,
        key="knn_intro_lottie",
    )
else:
    st.image("https://www.statsoft.com/wp-content/uploads/2021/01/K-nearest-neighbors-diagram.png",
             caption="KNN: Clasificando por los vecinos más cercanos",
             use_column_width=True)
    st.warning("No se pudo cargar la animación Lottie. Asegúrate de que la ruta del archivo es correcta y que el archivo JSON es válido.")

st.markdown("""
Imagina que eres un detective de animales y tienes un animal nuevo que nunca habías visto antes.
No sabes si es un perro o un gato. ¿Qué haces?

Miras a los animales que están más **cerca** de él. Si la mayoría de sus **vecinos más cercanos** son perros,
¡entonces lo más probable es que sea un perro! Si la mayoría son gatos, ¡entonces es un gato!

**KNN** hace exactamente eso. Cuando le das un "dato nuevo" que no sabe qué es,
lo compara con todos los "datos viejos" que ya conoce.
Encuentra a los **"K"** datos más cercanos (tú decides cuántos 'K' vecinos mirar)
y el dato nuevo se une al grupo al que pertenecen la mayoría de sus vecinos.

Es una forma sencilla pero poderosa de clasificar cosas, como decidir si un email es spam,
o si una fruta es una manzana o una pera, solo mirando a sus amigos más cercanos.
""")

st.subheader("¡Vamos a jugar a un juego de Clasificación de Mascotas con KNN!")

st.markdown("""
¡Ayuda a nuestro amigo robot Etiquetín a clasificar nuevas mascotas!

1.  **Añade "Mascotas Conocidas"** (perros 🐶 o gatos 🐱) al jardín. Estos son los animales que Etiquetín ya conoce.
2.  **Añade una "Nueva Mascota Misteriosa"** (en gris 🐾). Esta es la que queremos clasificar.
3.  Elige cuántos **"Amigos Cercanos (K)"** quieres que Etiquetín examine.
4.  Pulsa "¡Clasificar Nueva Mascota!" y observa cómo Etiquetín la clasifica.
""")

# --- Juego Interactivo de Clasificación KNN ---
st.write("---")
st.subheader("🐾 El Juego del Clasificador de Mascotas 🐾")

# Inicializar los datos del juego en session_state
if 'knn_game_data' not in st.session_state:
    st.session_state.knn_game_data = [] # Lista de diccionarios: [{'x': x, 'y': y, 'class': 'Perro' or 'Gato'}]
if 'knn_new_point' not in st.session_state:
    st.session_state.knn_new_point = {'x': None, 'y': None}
if 'knn_k_neighbors' not in st.session_state:
    st.session_state.knn_k_neighbors = 3 # Número de vecinos por defecto
if 'knn_predicted_class' not in st.session_state:
    st.session_state.knn_predicted_class = None
if 'knn_nearest_indices' not in st.session_state:
    st.session_state.knn_nearest_indices = []
if 'knn_game_messages' not in st.session_state:
    st.session_state.knn_game_messages = []

# Inicialización de valores para sliders con session_state
if 'point_x_game_value' not in st.session_state:
    st.session_state.point_x_game_value = float(random.randint(0,10))
if 'point_y_game_value' not in st.session_state:
    st.session_state.point_y_game_value = float(random.randint(0,10))

# Crear el gráfico para el juego de KNN
fig_knn, ax_knn = plt.subplots(figsize=(9, 7))
ax_knn.set_xlabel("Fuerza (e.g., ¿Cuánto tira de la correa?)")
ax_knn.set_ylabel("Tamaño (0=pequeño, 10=grande)")
ax_knn.set_title("Juego de Clasificación de Mascotas con KNN")
ax_knn.set_xlim(0, 10)
ax_knn.set_ylim(0, 10)
ax_knn.grid(True, linestyle='--', alpha=0.6)

# Colores y marcadores para las clases
colors = {'Perro': 'brown', 'Gato': 'purple'}
markers = {'Perro': 'P', 'Gato': 'X'}

# Plotear los puntos existentes
if st.session_state.knn_game_data:
    X_train_game = np.array([[d['x'], d['y']] for d in st.session_state.knn_game_data])
    y_train_game = np.array([d['class'] for d in st.session_state.knn_game_data])

    for i, (x, y) in enumerate(X_train_game):
        ax_knn.scatter(x, y, color=colors[y_train_game[i]], s=200, edgecolor='black', zorder=2,
                       marker=markers[y_train_game[i]], label=f'Mascota: {y_train_game[i]}' if i==0 or (i > 0 and y_train_game[i] != y_train_game[i-1]) else "")

# Plotear el nuevo punto si existe
if st.session_state.knn_new_point['x'] is not None and st.session_state.knn_new_point['y'] is not None:
    new_x, new_y = st.session_state.knn_new_point['x'], st.session_state.knn_new_point['y']
    plot_color = 'gray'
    if st.session_state.knn_predicted_class:
        plot_color = colors[st.session_state.knn_predicted_class]
    ax_knn.scatter(new_x, new_y, color=plot_color, s=300, edgecolor='black', marker='*', zorder=3, label='Nueva Mascota')
    ax_knn.text(new_x + 0.2, new_y, '¡Soy nuevo!', fontsize=10, color='dimgray', ha='left', va='center')

    if len(st.session_state.knn_nearest_indices) > 0 and st.session_state.knn_predicted_class:
        for idx in st.session_state.knn_nearest_indices:
            ax_knn.plot([new_x, X_train_game[idx, 0]], [new_y, X_train_game[idx, 1]],
                        '--', color='red', linewidth=1.5, alpha=0.7)

if not st.session_state.knn_game_data and st.session_state.knn_new_point['x'] is None:
    ax_knn.text((ax_knn.get_xlim()[0] + ax_knn.get_xlim()[1]) / 2,
                (ax_knn.get_ylim()[0] + ax_knn.get_ylim()[1]) / 2,
                "¡Añade mascotas conocidas y una nueva mascota misteriosa!",
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='gray', alpha=0.6)

handles, labels = ax_knn.get_legend_handles_labels()
unique_labels = list(set(labels))
unique_handles = [handles[labels.index(l)] for l in unique_labels]
ax_knn.legend(unique_handles, unique_labels)

st.pyplot(fig_knn, use_container_width=True)

st.markdown("---")
st.subheader("¡Ayuda a Etiquetín a clasificar!")

col_add_point_game1, col_add_point_game2, col_add_point_game3 = st.columns(3)

with col_add_point_game1:
    st.session_state.point_x_game_value = st.slider("¿Cuánta Fuerza tiene (0=muy poca, 10=mucha)?",
                                                     min_value=0.0, max_value=10.0,
                                                     value=st.session_state.point_x_game_value,
                                                     step=0.5, key="point_x_game")
with col_add_point_game2:
    st.session_state.point_y_game_value = st.slider("¿Qué Tamaño tiene (0=pequeño, 10=grande)?",
                                                     min_value=0.0, max_value=10.0,
                                                     value=st.session_state.point_y_game_value,
                                                     step=0.5, key="point_y_game")
with col_add_point_game3:
    st.markdown(" ")
    add_point_type_game = st.radio("¿Qué tipo de mascota vas a añadir?", ["🐶 Mascota Conocida (Perro)", "🐱 Mascota Conocida (Gato)", "🐾 Nueva Mascota Misteriosa"], key="add_point_type_game")

    if st.button("➕ Añadir Mascota al jardín", key="add_knn_game_point_button"):
        if "Mascota Conocida" in add_point_type_game:
            point_class_game = 'Perro' if "Perro" in add_point_type_game else 'Gato'
            st.session_state.knn_game_data.append({'x': st.session_state.point_x_game_value,
                                                   'y': st.session_state.point_y_game_value,
                                                   'class': point_class_game})
        else: # Nueva Mascota Misteriosa
            st.session_state.knn_new_point = {'x': st.session_state.point_x_game_value,
                                              'y': st.session_state.point_y_game_value}
            # Al añadir una nueva mascota misteriosa, limpiamos la clasificación anterior
            st.session_state.knn_predicted_class = None
            st.session_state.knn_nearest_indices = []

        # Reinicializar los valores de los sliders para el siguiente punto
        st.session_state.point_x_game_value = float(random.randint(0,10))
        st.session_state.point_y_game_value = float(random.randint(0,10))
        st.rerun() # Forzamos un rerun para actualizar el gráfico y la posible clasificación.

# --- Función para el callback del slider K ---
# ***** CORRECCIÓN DE SYNTAXERROR: Se define una función normal para el callback *****
def update_k_value():
    # Esta función se llama cuando el slider de K cambia.
    # No necesitamos hacer nada aquí aparte de lo que ya hace el slider
    # (que es actualizar st.session_state.knn_k_neighbors).
    # La bandera que habíamos puesto era redundante si la clasificación se ejecuta condicionalmente.
    pass # No es necesario hacer nada explícito aquí, el slider ya actualiza su valor en session_state

# Control de K (Amigos Cercanos)
num_known_data = len(st.session_state.knn_game_data)
min_k_slider = 1
max_k_slider_display = max(2, num_known_data)

# Ajustar K si excede el número de datos o es menor que 1
if st.session_state.knn_k_neighbors > max_k_slider_display and num_known_data > 0:
    st.session_state.knn_k_neighbors = max_k_slider_display
if st.session_state.knn_k_neighbors < 1:
    st.session_state.knn_k_neighbors = 1
# Si no hay datos conocidos, K debe ser 1 (o el slider deshabilitado)
if num_known_data == 0:
    st.session_state.knn_k_neighbors = 1

slider_disabled = False
if num_known_data < 2:
    slider_disabled = True
    st.info("Añade al menos **dos** mascotas conocidas para poder ajustar 'K' y clasificar de forma significativa. Por ahora, K es 1.")
    st.session_state.knn_k_neighbors = 1 # Forzar K a 1 si no hay suficientes datos

st.session_state.knn_k_neighbors = st.slider(
    "¿Cuántos 'Amigos Cercanos (K)' debe mirar Etiquetín?",
    min_value=min_k_slider,
    max_value=max_k_slider_display,
    value=st.session_state.knn_k_neighbors,
    step=1,
    key="k_neighbors_game_slider",
    disabled=slider_disabled,
    on_change=update_k_value # <--- AHORA SE LLAMA A LA FUNCIÓN DEFINIDA ARRIBA
)

# Lógica de Clasificación que se ejecuta si se cumplen las condiciones
# (Ya no necesita un botón dedicado, se activa por cambios relevantes)

# Condición para clasificar: hay una mascota misteriosa Y hay mascotas conocidas
can_classify = (st.session_state.knn_new_point['x'] is not None and
                len(st.session_state.knn_game_data) >= 1 and
                st.session_state.knn_k_neighbors <= len(st.session_state.knn_game_data))

# Validaciones previas a la clasificación
if can_classify:
    # Esta validación de K ya la hicimos antes del slider, pero la mantengo aquí por robustez
    if len(st.session_state.knn_game_data) < st.session_state.knn_k_neighbors:
        st.warning(f"K ({st.session_state.knn_k_neighbors}) es demasiado grande para el número actual de mascotas conocidas ({len(st.session_state.knn_game_data)}). Ajusta K a un valor menor o añade más mascotas conocidas.")
        # Resetear la clasificación si K es inválido para el cálculo actual
        st.session_state.knn_predicted_class = None
        st.session_state.knn_nearest_indices = []
    else:
        X_train_game_np = np.array([[d['x'], d['y']] for d in st.session_state.knn_game_data])
        y_train_game_np = np.array([d['class'] for d in st.session_state.knn_game_data])
        X_new_game_np = np.array([[st.session_state.knn_new_point['x'], st.session_state.knn_new_point['y']]])

        try:
            knn_model_game = KNeighborsClassifier(n_neighbors=st.session_state.knn_k_neighbors)
            knn_model_game.fit(X_train_game_np, y_train_game_np)
            
            distances, indices = knn_model_game.kneighbors(X_new_game_np)
            
            nearest_neighbor_classes = y_train_game_np[indices[0]]
            
            predicted_class_game = knn_model_game.predict(X_new_game_np)[0]
            st.session_state.knn_predicted_class = predicted_class_game
            st.session_state.knn_nearest_indices = indices[0]

            # Mostrar el resultado de la clasificación
            st.success(f"🎉 ¡Etiquetín ha clasificado la nueva mascota como **{predicted_class_game}**! 🎉")
            st.info(f"Lo ha hecho porque sus **{st.session_state.knn_k_neighbors} amigos cercanos** (los puntos unidos con líneas rojas) son: {', '.join(nearest_neighbor_classes)}. La mayoría de ellos son de la clase **{predicted_class_game}**.")

        except Exception as e:
            st.error(f"¡Oops! Etiquetín tuvo un problema al clasificar. Error: {e}")
            st.info("Asegúrate de que tienes suficientes mascotas conocidas para el valor de K seleccionado.")
elif st.session_state.knn_new_point['x'] is None:
    st.info("Añade una 'Nueva Mascota Misteriosa' y algunas 'Mascotas Conocidas' para ver la clasificación.")
elif not st.session_state.knn_game_data:
    st.info("Necesitas añadir algunas 'Mascotas Conocidas' (Perro o Gato) para que Etiquetín aprenda de ellas.")
# El botón "¡Clasificar Nueva Mascota con Etiquetín!" se puede eliminar o hacer invisible si la clasificación es automática
# st.button("¡Clasificar Nueva Mascota con Etiquetín!", key="run_knn_game_button")
st.markdown("---")

if st.button("Borrar todas las mascotas del jardín", key="clear_knn_game_points"):
    st.session_state.knn_game_data = []
    st.session_state.knn_new_point = {'x': None, 'y': None}
    st.session_state.knn_predicted_class = None
    st.session_state.knn_nearest_indices = []
    st.session_state.knn_game_messages = []
    st.session_state.point_x_game_value = float(random.randint(0,10))
    st.session_state.point_y_game_value = float(random.randint(0,10))
    st.rerun()

st.write("---")

st.header("¿Cómo encuentra Etiquetín los grupos sin saber los nombres?")
st.markdown("""
¡Es fascinante cómo Etiquetín clasifica un punto nuevo sin necesidad de "entrenar" mucho! Así es como funciona KNN:

1.  **Guarda todos los datos:** A diferencia de otros métodos, KNN es un poco "vago" al principio. Simplemente **guarda** todos los puntos de datos que ya conoces (los que tú pones de "Perro" y "Gato"). ¡No hace cálculos complicados al inicio!

2.  **¡Llega un punto nuevo!:** Cuando le pones un punto gris (el que quieres clasificar), Etiquetín se pone a trabajar.

3.  **Mide distancias:** Etiquetín calcula lo **cerca** que está el nuevo punto de *todos* los puntos que ya conoce. Piensa en la distancia como si fuera un mapa: ¿cuántos pasos hay desde el punto nuevo hasta cada uno de los puntos viejos?

4.  **Encuentra a los "K" vecinos más cercanos:** Después de medir todas las distancias, Etiquetín selecciona a los **K** puntos (el número que tú eliges, por ejemplo, 3 o 5) que están más cerca del punto nuevo. ¡Estos son sus "amigos más cercanos"!

5.  **Vota por la clase:** Finalmente, Etiquetín mira a esos "K" amigos cercanos y les pregunta: "¿De qué clase sois vosotros?". La clase que tenga la **mayoría de votos** entre esos "K" amigos es la clase que Etiquetín le asigna al nuevo punto.

¡Y así, Etiquetín clasifica cosas por "amistad" o cercanía, sin tener que aprender reglas complejas antes!
""")
st.write("---")


# --- Sección de Chatbot de Juego con Vecinín para "Qué es KNN" ---
st.header("¡Juega y Aprende con Vecinín sobre K-Nearest Neighbors (KNN)!")
st.markdown("¡Hola! Soy Vecinín, el robot que ama encontrar a tus vecinos más cercanos en los datos. ¿Listo para aprender a clasificar y predecir usando la amistad de los datos?")

if client:
    # Inicializa el estado del juego y los mensajes del chat para Vecinín
    if "knn_game_active" not in st.session_state:
        st.session_state.knn_game_active = False
    if "knn_game_messages" not in st.session_state:
        st.session_state.knn_game_messages = []
    if "knn_current_question" not in st.session_state:
        st.session_state.knn_current_question = None
    if "knn_current_options" not in st.session_state:
        st.session_state.knn_current_options = {}
    if "knn_correct_answer" not in st.session_state:
        st.session_state.knn_correct_answer = None
    if "knn_awaiting_next_game_decision" not in st.session_state:
        st.session_state.knn_awaiting_next_game_decision = False
    if "knn_game_needs_new_question" not in st.session_state:
        st.session_state.knn_game_needs_new_question = False
    if "knn_correct_streak" not in st.session_state:
        st.session_state.knn_correct_streak = 0
    if "last_played_question_vecinin_knn" not in st.session_state:
        st.session_state.last_played_question_vecinin_knn = None

    # System prompt para el juego de preguntas de Vecinín
    vecinin_knn_game_system_prompt = f"""
    Eres un **experto consumado en Machine Learning y Clasificación/Regresión**, con una especialización profunda en el algoritmo **K-Nearest Neighbors (KNN)**. Comprendes a fondo sus fundamentos teóricos, la intuición detrás de su funcionamiento, métricas de distancia, la elección óptima de 'K', sus aplicaciones (clasificación y regresión), ventajas y limitaciones. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de KNN mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Clasificación perfecta! Has encontrado los vecinos correctos." o "Esa clasificación no fue la más cercana. Repasemos cómo Vecinín toma decisiones."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "KNN es un algoritmo de aprendizaje supervisado que clasifica nuevos puntos basándose en la mayoría de sus 'K' vecinos más cercanos..."]
    [Pregunta para continuar, ej: "¿Listo para encontrar más vecinos en los datos?" o "¿Quieres explorar cómo K afecta la decisión de Vecinín?"]

    **Reglas adicionales para el Experto en KNN:**
    * **Enfoque Riguroso en K-Nearest Neighbors (KNN):** Todas tus preguntas y explicaciones deben girar en torno a KNN. Cubre sus fundamentos (aprendizaje supervisado, clasificación/regresión basada en la proximidad), el concepto de 'K', métricas de distancia (euclidiana, Manhattan), el impacto de la escala de los datos, la elección del valor óptimo de 'K', las ventajas (simplicidad, no paramétrico) y limitaciones (costo computacional en grandes datasets, sensibilidad a la escala y outliers, maldición de la dimensionalidad) de KNN, y aplicaciones prácticas.
    * **¡VARIEDAD, VARIEDAD, VARIEDAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de KNN que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General de KNN:** ¿Qué es? ¿Por qué es supervisado? Clasificación vs. Regresión.
        * **Funcionamiento de KNN:** Asignación de clases/valores basándose en los 'K' vecinos más cercanos.
        * **Importancia de 'K':** Cómo afecta la complejidad del modelo y el sesgo/varianza.
        * **Métricas de Distancia:** Distancia Euclidiana, Distancia Manhattan.
        * **Preprocesamiento de Datos:** Importancia del escalado/normalización.
        * **Manejo de Outliers:** Sensibilidad de KNN.
        * **Ventajas y Limitaciones de KNN:** Simplicidad, no paramétrico vs. costo computacional, sensibilidad a la escala, maldición de la dimensionalidad.
        * **Aplicaciones Prácticas:** Reconocimiento de patrones, sistemas de recomendación, diagnóstico médico.
        * **Comparación:** Diferencias fundamentales con otros algoritmos de clasificación (árboles de decisión, SVM) a nivel conceptual.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.knn_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Aprendiz de Vecino – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de encontrar cosas similares y ejemplos cotidianos.
            * *Tono:* "Estás empezando a entender cómo los datos encuentran a sus mejores amigos."
        * **Nivel 2 (Buscador de Vecinos – 3-5 respuestas correctas):** Tono más técnico. Introduce el concepto de 'K' y cómo se usan las distancias para tomar decisiones intuitivamente. Preguntas sobre los pasos fundamentales del algoritmo.
            * *Tono:* "Tu habilidad para clasificar datos basándote en la proximidad es cada vez más aguda."
        * **Nivel 3 (Cartógrafo de Datos – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en la importancia del escalado de datos, la elección del 'K' óptimo, y cómo las diferentes métricas de distancia influyen en los resultados.
            * *Tono:* "Tu comprensión de KNN te permite mapear las relaciones en los datos con gran detalle."
        * **Nivel Maestro (Gurú de la Proximidad – 9+ respuestas correctas):** Tono de **especialista en Machine Learning Supervisado y optimización de KNN**. Preguntas sobre la maldición de la dimensionalidad, cómo ponderar los vecinos, o cuándo KNN es (o no es) la mejor elección frente a otros algoritmos para tipos de datos o problemas específicos. Se esperan respuestas que demuestren una comprensión teórica y práctica profunda, incluyendo sus limitaciones y cómo mitigarlas.
            * *Tono:* "Tu maestría en la toma de decisiones basada en la proximidad te convierte en un verdadero arquitecto del aprendizaje supervisado."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Clasificar un nuevo animal en "mamífero" o "ave" según si sus "vecinos" (otros animales conocidos) tienen pelo o plumas.
        * **Nivel 2:** Predecir si un email es "spam" o "no spam" basándose en emails similares que ya han sido clasificados, o recomendar una película a alguien basándose en las películas que les gustaron a sus "vecinos" con gustos parecidos.
        * **Nivel 3:** Usar KNN para diagnosticar una enfermedad observando los síntomas de pacientes con diagnósticos conocidos (los "vecinos"), o predecir el precio de una casa basándose en casas similares vendidas recientemente.
        * **Nivel Maestro:** Implementar KNN para el reconocimiento de escritura a mano, considerando la eficiencia computacional con grandes conjuntos de datos, o ajustar KNN para un sistema de detección de intrusiones en redes, evaluando la robustez del modelo ante datos ruidosos y de alta dimensionalidad.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre KNN, y asegúrate de que no se parezca a las anteriores.
    """

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_knn_question_response(raw_text):
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
    def parse_knn_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"

    # --- Funciones para subir de nivel directamente ---
    def set_vecinin_knn_level(target_streak, level_name):
        st.session_state.knn_correct_streak = target_streak
        st.session_state.knn_game_active = True
        st.session_state.knn_game_messages = []
        st.session_state.knn_current_question = None
        st.session_state.knn_current_options = {}
        st.session_state.knn_correct_answer = None
        st.session_state.knn_game_needs_new_question = True
        st.session_state.knn_awaiting_next_game_decision = False
        st.session_state.knn_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}** de Vecinín! Prepárate para preguntas más desafiantes sobre KNN. ¡Aquí va tu primera!"})
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons_vecinin_knn, col_level_up_buttons_vecinin_knn = st.columns([1, 2])

    with col_game_buttons_vecinin_knn:
        if st.button("¡Vamos a jugar con Vecinín!", key="start_vecinin_knn_game_button"):
            st.session_state.knn_game_active = True
            st.session_state.knn_game_messages = []
            st.session_state.knn_current_question = None
            st.session_state.knn_current_options = {}
            st.session_state.knn_correct_answer = None
            st.session_state.knn_game_needs_new_question = True
            st.session_state.knn_awaiting_next_game_decision = False
            st.session_state.knn_correct_streak = 0
            st.session_state.last_played_question_vecinin_knn = None
            st.rerun()
    
    with col_level_up_buttons_vecinin_knn:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un experto en vecinos? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_vecinin_knn, col_lvl2_vecinin_knn, col_lvl3_vecinin_knn = st.columns(3) # Tres columnas para los botones de nivel
        with col_lvl1_vecinin_knn:
            if st.button("Subir a Nivel Medio (Vecinín - Buscador)", key="level_up_medium_vecinin_knn"):
                set_vecinin_knn_level(3, "Medio") # 3 respuestas correctas para Nivel Medio
        with col_lvl2_vecinin_knn:
            if st.button("Subir a Nivel Avanzado (Vecinín - Cartógrafo)", key="level_up_advanced_vecinin_knn"):
                set_vecinin_knn_level(6, "Avanzado") # 6 respuestas correctas para Nivel Avanzado
        with col_lvl3_vecinin_knn:
            if st.button("👑 ¡Gurú de la Proximidad! (Vecinín)", key="level_up_champion_vecinin_knn"):
                set_vecinin_knn_level(9, "Campeón") # 9 respuestas correctas para Nivel Campeón

    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.knn_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica del juego del chatbot si está activo
    if st.session_state.knn_game_active:
        if st.session_state.knn_current_question is None and st.session_state.knn_game_needs_new_question and not st.session_state.knn_awaiting_next_game_decision:
            with st.spinner("Vecinín está preparando una pregunta sobre KNN..."):
                try:
                    # Incluimos el prompt del sistema actualizado con el nivel de dificultad
                    game_messages_for_api = [{"role": "system", "content": vecinin_knn_game_system_prompt}]
                    # Limita el historial para evitar prompts demasiado largos, tomando las últimas interacciones relevantes
                    if st.session_state.knn_game_messages:
                        last_message = st.session_state.knn_game_messages[-1]
                        if last_message["role"] == "user":
                            game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {last_message['content']}"})
                        elif last_message["role"] == "assistant":
                            # Si el último mensaje fue del asistente (feedback), lo añadimos para que sepa dónde se quedó
                            game_messages_for_api.append({"role": "assistant", "content": last_message['content']})

                    game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre KNN siguiendo el formato exacto. ¡Recuerda, la pregunta debe ser muy VARIADA y CREATIVA, y no se debe parecer a las anteriores!"})

                    game_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=game_messages_for_api,
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_knn_question_text = game_response.choices[0].message.content
                    question, options, correct_answer_key = parse_knn_question_response(raw_knn_question_text)

                    if question:
                        st.session_state.knn_current_question = question
                        st.session_state.knn_current_options = options
                        st.session_state.knn_correct_answer = correct_answer_key

                        display_question_text = f"**Nivel {int(st.session_state.knn_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.knn_correct_streak}**\n\n**Pregunta de Vecinín:** {question}\n\n"
                        for key in sorted(options.keys()):
                            display_question_text += f"{key}) {options[key]}\n"

                        st.session_state.knn_game_messages.append({"role": "assistant", "content": display_question_text})
                        st.session_state.knn_game_needs_new_question = False
                        st.rerun()
                    else:
                        st.session_state.knn_game_messages.append({"role": "assistant", "content": "¡Lo siento! Vecinín no pudo preparar la pregunta en el formato correcto. ¿Puedes pulsar '¡Vamos a jugar!' de nuevo?"})
                        st.session_state.knn_game_active = False
                        st.rerun()

                except Exception as e:
                    st.error(f"¡Oops! Vecinín no pudo hacer la pregunta. Error: {e}")
                    st.session_state.knn_game_messages.append({"role": "assistant", "content": "¡Lo siento! Vecinín tiene un pequeño problema para hacer preguntas ahora. ¿Puedes intentarlo de nuevo?"})
                    st.session_state.knn_game_active = False
                    st.rerun()

        if st.session_state.knn_current_question is not None and not st.session_state.knn_awaiting_next_game_decision:
            # Audio de la pregunta
            if st.session_state.get('last_played_question_vecinin_knn') != st.session_state.knn_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.knn_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.knn_correct_streak}. Pregunta de Vecinín: {st.session_state.knn_current_question}. Opción A: {st.session_state.knn_current_options.get('A', '')}. Opción B: {st.session_state.knn_current_options.get('B', '')}. Opción C: {st.session_state.knn_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                    st.session_state.last_played_question_vecinin_knn = st.session_state.knn_current_question
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")

            with st.form("vecinin_knn_game_form", clear_on_submit=True):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_choice = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.knn_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.knn_current_options[x]}",
                        key="knn_answer_radio_buttons",
                        label_visibility="collapsed"
                    )

                submit_button = st.form_submit_button("Enviar Respuesta")

            if submit_button:
                st.session_state.knn_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_choice}) {st.session_state.knn_current_options[user_choice]}"})
                prev_streak = st.session_state.knn_correct_streak

                # Lógica para actualizar el contador de respuestas correctas
                if user_choice == st.session_state.knn_correct_answer:
                    st.session_state.knn_correct_streak += 1
                else:
                    st.session_state.knn_correct_streak = 0

                radio_placeholder.empty()

                # --- Lógica de subida de nivel ---
                if st.session_state.knn_correct_streak > 0 and \
                   st.session_state.knn_correct_streak % 3 == 0 and \
                   st.session_state.knn_correct_streak > prev_streak:
                    
                    if st.session_state.knn_correct_streak < 9: # Niveles Básico, Medio, Avanzado
                        current_level_text = ""
                        if st.session_state.knn_correct_streak == 3:
                            current_level_text = "Medio (como un joven que ya elige a sus vecinos sabiamente)"
                        elif st.session_state.knn_correct_streak == 6:
                            current_level_text = "Avanzado (como un Data Scientist que encuentra patrones con precisión)"
                        
                        level_up_message = f"🎉 ¡Increíble! ¡Has respondido {st.session_state.knn_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de KNN. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a buscador/a de proximidad!"
                        st.session_state.knn_game_messages.append({"role": "assistant", "content": level_up_message})
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
                    elif st.session_state.knn_correct_streak >= 9:
                        medals_earned = (st.session_state.knn_correct_streak - 6) // 3
                        medal_message = f"🏅 ¡FELICITACIONES, GURÚ DE LA PROXIMIDAD! ¡Has ganado tu {medals_earned}ª Medalla de Vecinos Cercanos! ¡Tu habilidad para clasificar datos es asombrosa y digna de un verdadero EXPERTO en KNN! ¡Sigue así!"
                        st.session_state.knn_game_messages.append({"role": "assistant", "content": medal_message})
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
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Campeón (Gurú de la Proximidad)**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros científicos de datos que entienden la esencia de la similitud! ¡Adelante!"
                            st.session_state.knn_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                audio_fp_level_up_champion = io.BytesIO()
                                tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                                audio_fp_level_up_champion.seek(0)
                                st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0, autoplay=True)
                                time.sleep(2)
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón: {e}")

                # Generar feedback de Vecinín
                with st.spinner("Vecinín está revisando tu respuesta..."):
                    try:
                        feedback_prompt = f"""
                        El usuario respondió '{user_choice}'. La pregunta era: '{st.session_state.knn_current_question}'.
                        La respuesta correcta era '{st.session_state.knn_correct_answer}'.
                        Da feedback como Vecinín.
                        Si es CORRECTO, el mensaje es "¡Clasificación perfecta! Has encontrado a los vecinos correctos." o similar.
                        Si es INCORRECTO, el mensaje es "¡Esa predicción no fue la más cercana. Revisa tus vecinos!" o similar.
                        Luego, una explicación concisa y clara.
                        Finalmente, pregunta: "¿Quieres seguir encontrando vecinos en los datos?".
                        **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                        """
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": vecinin_knn_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.8,
                            max_tokens=300
                        )
                        raw_vecinin_knn_feedback_text = feedback_response.choices[0].message.content

                        feedback_msg, explanation_msg, next_question_prompt = parse_knn_feedback_response(raw_vecinin_knn_feedback_text)

                        st.session_state.knn_game_messages.append({"role": "assistant", "content": feedback_msg})
                        st.session_state.knn_game_messages.append({"role": "assistant", "content": explanation_msg})
                        st.session_state.knn_game_messages.append({"role": "assistant", "content": next_question_prompt})

                        try:
                            tts = gTTS(text=f"{feedback_msg}. {explanation_msg}. {next_question_prompt}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")

                        st.session_state.knn_current_question = None
                        st.session_state.knn_current_options = {}
                        st.session_state.knn_correct_answer = None
                        st.session_state.knn_game_needs_new_question = False
                        st.session_state.knn_awaiting_next_game_decision = True

                        st.rerun()

                    except Exception as e:
                        st.error(f"Ups, Vecinín no pudo procesar tu respuesta. Error: {e}")
                        st.session_state.knn_game_messages.append({"role": "assistant", "content": "Lo siento, Vecinín tiene un pequeño problema técnico ahora mismo. ¡Pero me encantaría ver tu respuesta!"})

        if st.session_state.knn_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Sí, quiero jugar más preguntas", key="play_more_questions_knn"):
                    st.session_state.knn_game_needs_new_question = True
                    st.session_state.knn_awaiting_next_game_decision = False
                    st.session_state.knn_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío!"})
                    st.rerun()
            with col2:
                if st.button("👎 No, ya no quiero jugar más", key="stop_playing_knn"):
                    st.session_state.knn_game_active = False
                    st.session_state.knn_awaiting_next_game_decision = False
                    st.session_state.knn_game_messages.append({"role": "assistant", "content": "¡De acuerdo! ¡Gracias por jugar conmigo! Espero que hayas aprendido mucho sobre KNN. ¡Nos vemos pronto!"})
                    st.rerun()

else:
    st.info("Para usar la sección de preguntas de Vecinín, necesitas configurar tu clave de API de OpenAI en `secrets.toml`.")