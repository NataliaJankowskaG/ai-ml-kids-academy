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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="¿Qué es el Clustering (K-Means)?",
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
LOTTIE_CLUSTER_PATH = os.path.join("assets", "lottie_animations", "Group.json")

# --- Configuración de la API de OpenAI ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai_api_key = None
    st.error("Error: La clave de API de OpenAI no está configurada en `secrets.toml`.")
    st.info("Para configurarla, crea un archivo `.streamlit/secrets.toml` en la raíz de tu proyecto y añade: `OPENAI_API_KEY = 'tu_clave_aqui'`")

client = OpenAI(api_key=openai_api_key) if openai_api_key else None


st.subheader("¡Descubre patrones ocultos en tus datos!")

st.write("---")

# Sección 1: ¿Qué es el Clustering?
st.header("¿Qué es el Clustering?")
st.markdown("""
Imagina que tienes una caja llena de juguetes mezclados, ¡sin etiquetas ni nada!
El **Clustering** es como tener un robot súper inteligente que, sin que le digas nada,
mira los juguetes y dice: "¡Ah, estos son coches! ¡Estos son bloques! ¡Y estos son muñecos!".

A diferencia de la **Clasificación** (donde ya sabíamos los nombres de los grupos),
en Clustering la máquina **descubre los grupos por sí misma**. ¡Es como si encontrara
"familias" escondidas entre los datos!

Usamos el Clustering para organizar información sin tener etiquetas previas,
como agrupar clientes con gustos parecidos, o encontrar diferentes tipos de estrellas en el espacio.
""")

# Pequeña animación para la introducción
col_intro_left, col_intro_right = st.columns([1, 1])
with col_intro_right:
    lottie_cluster = load_lottiefile(LOTTIE_CLUSTER_PATH)
    if lottie_cluster:
        st_lottie(lottie_cluster, height=200, width=200, key="cluster_intro")
    else:
        st.info("No encontrado")

st.write("---")

# Sección 2: ¿Cómo Funciona K-Means?
st.header("¿Cómo Funciona K-Means?")
st.markdown("""
**K-Means** es como una de las formas más famosas que tiene la IA para hacer Clustering.
¡Vamos a ver cómo funciona con un juego interactivo!

1.  **Añade puntos** al gráfico (representan tus datos).
2.  Elige cuántos **grupos (K)** quieres encontrar.
3.  Observa cómo K-Means los organiza automáticamente o pulsa el botón para recalcular.
""")

# Inicializar los datos del juego en session_state
if 'cluster_data' not in st.session_state:
    st.session_state.cluster_data = [] # Lista de diccionarios: [{'x': x, 'y': y}]
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3 # 3 por defecto

# --- Inicializar los valores de los sliders en session_state ---
if 'current_point_x' not in st.session_state:
    st.session_state.current_point_x = random.uniform(0, 10)
if 'current_point_y' not in st.session_state:
    st.session_state.current_point_y = random.uniform(0, 10)


# Crear el gráfico para el clustering
fig_cls, ax_cls = plt.subplots(figsize=(9, 7))
ax_cls.set_xlabel("Característica 1 (X)")
ax_cls.set_ylabel("Característica 2 (Y)")
ax_cls.set_title("Agrupando Puntos con K-Means")
ax_cls.set_xlim(0, 10)
ax_cls.set_ylim(0, 10)
ax_cls.grid(True, linestyle='--', alpha=0.6)

if not st.session_state.cluster_data:
    ax_cls.text((ax_cls.get_xlim()[0] + ax_cls.get_xlim()[1]) / 2,
                (ax_cls.get_ylim()[0] + ax_cls.get_ylim()[1]) / 2,
                "¡Haz clic en el gráfico o usa los sliders para añadir puntos!",
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='gray', alpha=0.6)

# Obtener los puntos existentes
if st.session_state.cluster_data:
    X_cluster = np.array([[d['x'], d['y']] for d in st.session_state.cluster_data])
else:
    X_cluster = np.empty((0, 2))

# Plotear los puntos existentes
if X_cluster.size > 0:
    ax_cls.scatter(X_cluster[:, 0], X_cluster[:, 1], color='gray', s=100, edgecolor='black', zorder=2)

cluster_colors = plt.cm.get_cmap('viridis', max(st.session_state.n_clusters, 1))

# Ejecutar K-Means
kmeans_model = None
if len(X_cluster) >= st.session_state.n_clusters and st.session_state.n_clusters > 0 and X_cluster.size > 0:
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        kmeans_model = KMeans(n_clusters=st.session_state.n_clusters, random_state=42, n_init='auto')
        kmeans_model.fit(X_scaled)
        
        cluster_labels = kmeans_model.labels_
        cluster_centers_scaled = kmeans_model.cluster_centers_
        cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

        ax_cls.clear()
        ax_cls.set_xlabel("Característica 1 (X)")
        ax_cls.set_ylabel("Característica 2 (Y)")
        ax_cls.set_title("Agrupando Puntos con K-Means")
        ax_cls.set_xlim(0, 10)
        ax_cls.set_ylim(0, 10)
        ax_cls.grid(True, linestyle='--', alpha=0.6)

        for i in range(st.session_state.n_clusters):
            cluster_points_indices = np.where(cluster_labels == i)
            ax_cls.scatter(X_cluster[cluster_points_indices, 0], X_cluster[cluster_points_indices, 1],
                            color=cluster_colors(i), s=100, edgecolor='black', zorder=2, label=f'Grupo {i+1}')
            
            ax_cls.scatter(cluster_centers[i, 0], cluster_centers[i, 1], marker='X', s=300,
                            color='white', edgecolor='black', linewidth=2, zorder=3,
                            label=f'Centroide {i+1}' if i==0 else "")

        ax_cls.legend()
        st.markdown(f"La IA ha agrupado los puntos en **{st.session_state.n_clusters}** grupos diferentes.")

    except Exception as e:
        if "n_clusters should be less than or equal to n_samples" in str(e):
             st.warning(f"¡Oops! Para tener {st.session_state.n_clusters} grupos, necesitas al menos {st.session_state.n_clusters} puntos. ¡Añade más!")
        else:
            st.info(f"Necesitas añadir al menos {st.session_state.n_clusters} puntos para que K-Means pueda funcionar. (Error: {e})")
        kmeans_model = None

# Mostrar el gráfico
st.pyplot(fig_cls, use_container_width=True)

st.markdown("---")
st.subheader("¡Añade tus propios puntos y elige cuántos grupos quieres encontrar!")

col_add_point1, col_add_point2, col_add_point3 = st.columns(3)

with col_add_point1:
    point_x = st.slider("Posición X del punto:", min_value=0.0, max_value=10.0,
                        value=st.session_state.current_point_x,
                        step=0.1, key="point_x_cluster")
    st.session_state.current_point_x = point_x

with col_add_point2:
    point_y = st.slider("Posición Y del punto:", min_value=0.0, max_value=10.0,
                        value=st.session_state.current_point_y,
                        step=0.1, key="point_y_cluster")
    st.session_state.current_point_y = point_y

with col_add_point3:
    st.markdown(" ")
    st.markdown(" ")
    add_point_button = st.button("➕ Añadir Punto al gráfico", key="add_cluster_point")
    if add_point_button:
        st.session_state.cluster_data.append({'x': point_x, 'y': point_y})
        st.rerun()

col_k_slider, col_k_button = st.columns([0.7, 0.3])

with col_k_slider:
    st.session_state.n_clusters = st.slider("Número de grupos (K) que quieres que la IA encuentre:",
                                             min_value=1,
                                             max_value=max(1, len(st.session_state.cluster_data)),
                                             value=st.session_state.n_clusters,
                                             step=1,
                                             key="num_clusters_slider"
                                            )
with col_k_button:
    st.markdown(" ")
    st.markdown(" ")
    if st.button("¡Agrupar con este K!", key="recalculate_kmeans_button"):
        st.info("¡Etiquetín está reorganizando los grupos con tu nuevo K!")
        st.rerun()

st.markdown("---")

if st.button("Borrar todos los puntos del gráfico", key="clear_cluster_points"):
    st.session_state.cluster_data = []
    st.session_state.current_point_x = random.uniform(0, 10)
    st.session_state.current_point_y = random.uniform(0, 10)
    st.session_state.n_clusters = 3
    st.rerun()

st.write("---")

# Explicación sencilla del modelo
st.header("¿Cómo encuentra Etiquetín los grupos sin saber los nombres?")
st.markdown("""
¡Es fascinante cómo Etiquetín encuentra las "familias" de puntos sin que le digas cuáles son! Así es como funciona K-Means, el método que usa Etiquetín:

1.  **Imagina Puntos al Azar (Centroides):** Primero, Etiquetín "imagina" unos pocos puntos especiales en el gráfico (tantos como grupos le hayas dicho que busque, por ejemplo, 3 puntos si elegiste K=3). A estos puntos los llamamos **centroides**. Son como los "líderes" provisionales de cada grupo.

2.  **Cada Punto Elige un Líder:** Luego, Etiquetín hace que cada uno de tus puntos (los que añadiste) mire a todos esos centroides y elija al que esté **más cerca**. ¡Ese será su líder y su grupo!

3.  **Los Líderes se Mueven (y se Hacen Mejores):** Una vez que todos los puntos han elegido un líder, Etiquetín calcula un nuevo lugar para cada líder. ¡El nuevo lugar es justo en el **centro** de todos los puntos que lo eligieron! Esto hace que los líderes se acerquen más a los puntos de su grupo.

4.  **¡Repetir, Repetir, Repetir!:** Etiquetín repite los pasos 2 y 3 una y otra vez. Los puntos eligen nuevos líderes, y los líderes se mueven de nuevo. ¿Sabes cuándo para? Cuando los líderes ya no se mueven mucho, ¡significa que han encontrado el mejor lugar para ser el centro de sus grupos!

¡Y así es como Etiquetín organiza tus puntos en grupos, como un detective que descubre las familias de juguetes sin que le digas sus nombres!
""")
st.write("---")

# --- Sección de Chatbot de Juego con Etiquetín para "Qué es el Clustering" ---
st.header("¡Juega y Aprende con Etiquetín sobre el Clustering!")
st.markdown("¡Hola! Soy Etiquetín, la robot que ama descubrir familias escondidas en los datos. ¿Listo para aprender a agrupar cosas con la IA?")

if client:
    # Inicializa el estado del juego y los mensajes del chat
    if "cluster_game_active" not in st.session_state:
        st.session_state.cluster_game_active = False
    if "cluster_game_messages" not in st.session_state:
        st.session_state.cluster_game_messages = []
    if "cluster_current_question" not in st.session_state:
        st.session_state.cluster_current_question = None
    if "cluster_current_options" not in st.session_state:
        st.session_state.cluster_current_options = {}
    if "cluster_correct_answer" not in st.session_state:
        st.session_state.cluster_correct_answer = None
    if "cluster_awaiting_next_game_decision" not in st.session_state:
        st.session_state.cluster_awaiting_next_game_decision = False
    if "cluster_game_needs_new_question" not in st.session_state:
        st.session_state.cluster_game_needs_new_question = False
    if "cluster_correct_streak" not in st.session_state: 
        st.session_state.cluster_correct_streak = 0
    if "last_played_question_etiquetin_cluster" not in st.session_state:
        st.session_state.last_played_question_etiquetin_cluster = None


    # System prompt para el juego de preguntas de Etiquetín
    etiquetin_cluster_game_system_prompt = f"""
    Eres un **experto consumado en Machine Learning No Supervisado y Análisis de Datos Avanzado**, con una especialización profunda en los **Algoritmos de Clustering, particularmente K-Means**. Comprendes a fondo sus fundamentos teóricos, la intuición detrás de su funcionamiento, métricas de evaluación, aplicaciones prácticas y limitaciones. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio del Clustering (K-Means) mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Agrupación óptima! Has descubierto los patrones subyacentes." o "Esa asignación de grupo no fue la más coherente. Repasemos la distancia."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "El Clustering es una técnica de aprendizaje no supervisado que agrupa puntos de datos similares sin etiquetas previas..."]
    [Pregunta para continuar, ej: "¿Listo para afinar tus técnicas de segmentación?" o "¿Quieres explorar cómo K-Means maneja diferentes distribuciones de datos?"]

    **Reglas adicionales para el Experto en Clustering (K-Means):**
    * **Enfoque Riguroso en Clustering (K-Means):** Todas tus preguntas y explicaciones deben girar en torno al Clustering, con énfasis particular en el algoritmo K-Means. Cubre sus fundamentos (aprendizaje no supervisado, agrupación por similitud), el algoritmo K-Means paso a paso (inicialización de centroides, asignación, actualización), la elección del número óptimo de clusters (método del codo, coeficiente de silueta), métricas de evaluación (inercia/suma de cuadrados intra-cluster, coeficiente de silueta), preprocesamiento (escalado), manejo de outliers, ventajas y limitaciones de K-Means, y aplicaciones prácticas.
    * **¡VARIEDAD, VARIEDAD, VARIADAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de Clustering/K-Means que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General de Clustering:** ¿Qué es? ¿Por qué es no supervisado? Objetivos (segmentación, descubrimiento de patrones).
        * **Algoritmo K-Means Paso a Paso:**
            * **Inicialización:** Elección aleatoria o K-Means++.
            * **Asignación de Puntos:** Basado en la distancia (euclidiana).
            * **Actualización de Centroides:** Cálculo de la media de los puntos asignados.
            * **Convergencia:** Criterios de parada.
        * **Elección del Número de Clusters (K):**
            * **Método del Codo (Elbow Method):** Interpretación de la inercia.
            * **Coeficiente de Silueta:** Interpretación y rango de valores.
        * **Métricas de Evaluación Internas:** Inercia (Within-Cluster Sum of Squares - WCSS), Coeficiente de Silueta.
        * **Preprocesamiento de Datos:** Importancia del escalado/normalización para K-Means.
        * **Manejo de Outliers:** Sensibilidad de K-Means a los valores atípicos.
        * **Ventajas y Limitaciones de K-Means:** Simplicidad, eficiencia computacional vs. sensibilidad a la inicialización, forma de los clusters, necesidad de predefinir K.
        * **Distancia:** Rol de la distancia euclidiana.
        * **Aplicaciones Prácticas:** Segmentación de clientes, compresión de imágenes, detección de anomalías (intuitivo).
        * **Comparación:** Diferencias fundamentales con otros algoritmos de clustering (jerárquico, DBSCAN) a nivel conceptual.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.adivino_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Aprendiz de Agrupador – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de agrupar cosas similares y ejemplos cotidianos.
            * *Tono:* "Estás descubriendo el arte de encontrar similitudes y formar grupos."
        * **Nivel 2 (Analista de Segmentación – 3-5 respuestas correctas):** Tono más técnico. Introduce el concepto de centroides, distancia, y el ciclo iterativo de K-Means de forma intuitiva. Preguntas sobre los pasos fundamentales del algoritmo.
            * *Tono:* "Tu habilidad para identificar grupos coherentes en los datos es cada vez más refinada."
        * **Nivel 3 (Ingeniero de Clustering – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en la elección del K óptimo (método del codo, silueta), la importancia del escalado, la interpretación de la inercia, y la sensibilidad de K-Means a la inicialización o a los outliers.
            * *Tono:* "Tu comprensión de los algoritmos de clustering te permite segmentar datos complejos con precisión y conocimiento."
        * **Nivel Maestro (Científico de Datos de Clustering – 9+ respuestas correctas):** Tono de **especialista en Machine Learning No Supervisado y optimización de K-Means**. Preguntas sobre la robustez de K-Means++, la interpretación avanzada del coeficiente de silueta, el impacto de diferentes métricas de distancia, o la comparación estratégica de K-Means con otros algoritmos de clustering para tipos de datos o estructuras específicas. Se esperan respuestas que demuestren una comprensión teórica y práctica profunda, incluyendo sus limitaciones y cómo mitigarlas.
            * *Tono:* "Tu maestría en el descubrimiento de patrones ocultos a través del clustering te posiciona como un verdadero arquitecto del conocimiento no supervisado."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Agrupar diferentes tipos de juguetes en cajas según su categoría.
        * **Nivel 2:** Segmentar clientes de una tienda online basándose en sus hábitos de compra, o agrupar canciones por género musical de forma automática.
        * **Nivel 3:** Aplicar K-Means para la compresión de imágenes (cuantización de color) o para la identificación de diferentes tipos de ataques cibernéticos en datos de red.
        * **Nivel Maestro:** Implementar K-Means para segmentar poblaciones de células en bioinformática, evaluando la estabilidad de los clusters con diferentes inicializaciones y métricas de distancia, o diseñar un sistema de recomendación que use K-Means para agrupar usuarios con gustos similares.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre CLUSTERING (K-MEANS), y asegúrate de que no se parezca a las anteriores.
    """

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_cluster_question_response(raw_text):
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
    def parse_cluster_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"

    # --- Funciones para subir de nivel directamente ---
    def set_etiquetin_cluster_level(target_streak, level_name):
        st.session_state.cluster_correct_streak = target_streak
        st.session_state.cluster_game_active = True
        st.session_state.cluster_game_messages = []
        st.session_state.cluster_current_question = None
        st.session_state.cluster_current_options = {}
        st.session_state.cluster_correct_answer = None
        st.session_state.cluster_game_needs_new_question = True
        st.session_state.cluster_awaiting_next_game_decision = False
        st.session_state.cluster_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}** de Etiquetín! Prepárate para preguntas más desafiantes sobre el Clustering. ¡Aquí va tu primera!"})
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons_etiquetin_cluster, col_level_up_buttons_etiquetin_cluster = st.columns([1, 2])

    with col_game_buttons_etiquetin_cluster:
        if st.button("¡Vamos a jugar con Etiquetín!", key="start_etiquetin_cluster_game_button"):
            st.session_state.cluster_game_active = True
            st.session_state.cluster_game_messages = []
            st.session_state.cluster_current_question = None
            st.session_state.cluster_current_options = {}
            st.session_state.cluster_correct_answer = None
            st.session_state.cluster_game_needs_new_question = True
            st.session_state.cluster_awaiting_next_game_decision = False
            st.session_state.cluster_correct_streak = 0
            st.session_state.last_played_question_etiquetin_cluster = None
            st.rerun()
    
    with col_level_up_buttons_etiquetin_cluster:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un experto en agrupar? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_etiquetin_cluster, col_lvl2_etiquetin_cluster, col_lvl3_etiquetin_cluster = st.columns(3) # Tres columnas para los botones de nivel
        with col_lvl1_etiquetin_cluster:
            if st.button("Subir a Nivel Medio (Etiquetín - Agrupador)", key="level_up_medium_etiquetin_cluster"):
                set_etiquetin_cluster_level(3, "Medio") # 3 respuestas correctas para Nivel Medio
        with col_lvl2_etiquetin_cluster:
            if st.button("Subir a Nivel Avanzado (Etiquetín - Agrupador)", key="level_up_advanced_etiquetin_cluster"):
                set_etiquetin_cluster_level(6, "Avanzado") # 6 respuestas correctas para Nivel Avanzado
        with col_lvl3_etiquetin_cluster:
            if st.button("👑 ¡Maestro del Clustering! (Etiquetín)", key="level_up_champion_etiquetin_cluster"):
                set_etiquetin_cluster_level(9, "Campeón") # 9 respuestas correctas para Nivel Campeón


    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.cluster_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica del juego del chatbot si está activo
    if st.session_state.cluster_game_active:
        if st.session_state.cluster_current_question is None and st.session_state.cluster_game_needs_new_question and not st.session_state.cluster_awaiting_next_game_decision:
            with st.spinner("Etiquetín está preparando una pregunta sobre clustering..."):
                try:
                    # Incluimos el prompt del sistema actualizado con el nivel de dificultad
                    game_messages_for_api = [{"role": "system", "content": etiquetin_cluster_game_system_prompt}]
                    # Limita el historial para evitar prompts demasiado largos, tomando las últimas interacciones relevantes
                    if st.session_state.cluster_game_messages:
                        last_message = st.session_state.cluster_game_messages[-1]
                        if last_message["role"] == "user":
                            game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {last_message['content']}"})
                        elif last_message["role"] == "assistant":
                            # Si el último mensaje fue del asistente (feedback), lo añadimos para que sepa dónde se quedó
                            game_messages_for_api.append({"role": "assistant", "content": last_message['content']})

                    game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre QUÉ ES EL CLUSTERING o K-MEANS siguiendo el formato exacto. ¡Recuerda, la pregunta debe ser muy VARIADA y CREATIVA, y no se debe parecer a las anteriores!"})

                    game_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=game_messages_for_api,
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_cluster_question_text = game_response.choices[0].message.content
                    question, options, correct_answer_key = parse_cluster_question_response(raw_cluster_question_text)

                    if question:
                        st.session_state.cluster_current_question = question
                        st.session_state.cluster_current_options = options
                        st.session_state.cluster_correct_answer = correct_answer_key

                        display_question_text = f"**Nivel {int(st.session_state.cluster_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.cluster_correct_streak}**\n\n**Pregunta de Etiquetín:** {question}\n\n"
                        for key in sorted(options.keys()):
                            display_question_text += f"{key}) {options[key]}\n"

                        st.session_state.cluster_game_messages.append({"role": "assistant", "content": display_question_text})
                        st.session_state.cluster_game_needs_new_question = False
                        st.rerun()
                    else:
                        st.session_state.cluster_game_messages.append({"role": "assistant", "content": "¡Lo siento! Etiquetín no pudo preparar la pregunta en el formato correcto. ¿Puedes pulsar 'VAMOS A JUGAR' de nuevo?"})
                        st.session_state.cluster_game_active = False
                        st.rerun()

                except Exception as e:
                    st.error(f"¡Oops! Etiquetín no pudo hacer la pregunta. Error: {e}")
                    st.session_state.cluster_game_messages.append({"role": "assistant", "content": "¡Lo siento! Etiquetín tiene un pequeño problema para hacer preguntas ahora. ¿Puedes intentarlo de nuevo?"})
                    st.session_state.cluster_game_active = False
                    st.rerun()


        if st.session_state.cluster_current_question is not None and not st.session_state.cluster_awaiting_next_game_decision:
            # Audio de la pregunta
            if st.session_state.get('last_played_question_etiquetin_cluster') != st.session_state.cluster_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.cluster_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.cluster_correct_streak}. Pregunta de Etiquetín: {st.session_state.cluster_current_question}. Opción A: {st.session_state.cluster_current_options.get('A', '')}. Opción B: {st.session_state.cluster_current_options.get('B', '')}. Opción C: {st.session_state.cluster_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                    st.session_state.last_played_question_etiquetin_cluster = st.session_state.cluster_current_question
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")


            with st.form("etiquetin_cluster_game_form", clear_on_submit=True):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_choice = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.cluster_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.cluster_current_options[x]}",
                        key="cluster_answer_radio_buttons",
                        label_visibility="collapsed"
                    )

                submit_button = st.form_submit_button("Enviar Respuesta")

            if submit_button:
                st.session_state.cluster_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_choice}) {st.session_state.cluster_current_options[user_choice]}"})
                prev_streak = st.session_state.cluster_correct_streak

                # Lógica para actualizar el contador de respuestas correctas
                if user_choice == st.session_state.cluster_correct_answer:
                    st.session_state.cluster_correct_streak += 1
                else:
                    st.session_state.cluster_correct_streak = 0

                radio_placeholder.empty()

                # --- Lógica de subida de nivel ---
                if st.session_state.cluster_correct_streak > 0 and \
                   st.session_state.cluster_correct_streak % 3 == 0 and \
                   st.session_state.cluster_correct_streak > prev_streak:
                    
                    if st.session_state.cluster_correct_streak < 9: # Niveles Básico, Medio, Avanzado
                        current_level_text = ""
                        if st.session_state.cluster_correct_streak == 3:
                            current_level_text = "Medio (como un adolescente que ya entiende de agrupar)"
                        elif st.session_state.cluster_correct_streak == 6:
                            current_level_text = "Avanzado (como un Data Scientist junior)"
                        
                        level_up_message = f"🎉 ¡Increíble! ¡Has respondido {st.session_state.cluster_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de Clustering. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a descubridor/a de patrones!"
                        st.session_state.cluster_game_messages.append({"role": "assistant", "content": level_up_message})
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
                    elif st.session_state.cluster_correct_streak >= 9:
                        medals_earned = (st.session_state.cluster_correct_streak - 6) // 3 
                        medal_message = f"🏅 ¡FELICITACIONES, MAESTRO DEL CLUSTERING! ¡Has ganado tu {medals_earned}ª Medalla de Agrupación! ¡Tu habilidad para encontrar familias ocultas es asombrosa y digna de un verdadero EXPERTO en Clustering! ¡Sigue así!"
                        st.session_state.cluster_game_messages.append({"role": "assistant", "content": medal_message})
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
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Campeón (Maestro del Clustering)**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros científicos de datos que descubren los secretos de los datos! ¡Adelante!"
                            st.session_state.cluster_game_messages.append({"role": "assistant", "content": level_up_message_champion})
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
                        El usuario respondió '{user_choice}'. La pregunta era: '{st.session_state.cluster_current_question}'.
                        La respuesta correcta era '{st.session_state.cluster_correct_answer}'.
                        Da feedback como Etiquetín.
                        Si es CORRECTO, el mensaje es "¡Agrupación perfecta! ¡Lo has encontrado bien!" o similar.
                        Si es INCORRECTO, el mensaje es "¡Revisa tus grupos!" o similar.
                        Luego, una explicación sencilla para niños y adolescentes.
                        Finalmente, pregunta: "¿Quieres seguir agrupando cosas?".
                        **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                        """
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": etiquetin_cluster_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.8,
                            max_tokens=300
                        )
                        raw_etiquetin_cluster_feedback_text = feedback_response.choices[0].message.content

                        feedback_msg, explanation_msg, next_question_prompt = parse_cluster_feedback_response(raw_etiquetin_cluster_feedback_text)

                        st.session_state.cluster_game_messages.append({"role": "assistant", "content": feedback_msg})
                        st.session_state.cluster_game_messages.append({"role": "assistant", "content": explanation_msg})
                        st.session_state.cluster_game_messages.append({"role": "assistant", "content": next_question_prompt})

                        try:
                            tts = gTTS(text=f"{feedback_msg}. {explanation_msg}. {next_question_prompt}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")


                        st.session_state.cluster_current_question = None
                        st.session_state.cluster_current_options = {}
                        st.session_state.cluster_correct_answer = None
                        st.session_state.cluster_game_needs_new_question = False
                        st.session_state.cluster_awaiting_next_game_decision = True

                        st.rerun()

                    except Exception as e:
                        st.error(f"Ups, Etiquetín no pudo procesar tu respuesta. Error: {e}")
                        st.session_state.cluster_game_messages.append({"role": "assistant", "content": "Lo siento, Etiquetín tiene un pequeño problema técnico ahora mismo. ¡Pero me encantaría ver tu respuesta!"})


        if st.session_state.cluster_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Sí, quiero jugar más preguntas", key="play_more_questions_cluster"):
                    st.session_state.cluster_game_needs_new_question = True
                    st.session_state.cluster_awaiting_next_game_decision = False
                    st.session_state.cluster_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío!"})
                    st.rerun()
            with col2:
                if st.button("👎 No, ya no quiero jugar más", key="stop_playing_cluster"):
                    st.session_state.cluster_game_active = False
                    st.session_state.cluster_awaiting_next_game_decision = False
                    st.session_state.cluster_game_messages.append({"role": "assistant", "content": "¡De acuerdo! ¡Gracias por jugar conmigo! Espero que hayas aprendido mucho sobre el Clustering. ¡Nos vemos pronto!"})
                    st.rerun()

else:
    st.info("Para usar la sección de preguntas de Etiquetín, necesitas configurar tu clave de API de OpenAI en `secrets.toml`.")