import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import io
import time
from gtts import gTTS
try:
    from sklearn import svm
    from sklearn.datasets import make_blobs, make_circles
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    st.error("Las librerías 'scikit-learn' no están instaladas. Por favor, instálalas usando: pip install scikit-learn")
    svm = None
    make_blobs = None
    make_circles = None
    train_test_split = None
    accuracy_score = None

try:
    from openai import OpenAI
except ImportError:
    st.error("La librería 'openai' no está instalada. Por favor, instálala usando: pip install openai")
    OpenAI = None

# --- Configuración de la página ---
st.set_page_config(
    page_title="¿Qué son Máquinas de Vectores de Soporte (SVM)?",
    layout="wide"
)

# --- Funciones auxiliares para SVM ---

@st.cache_resource
def train_simple_svm(X, y, kernel='linear', C=1.0):
    """Entrena un SVM simple y devuelve el modelo."""
    if svm is None:
        return None
    try:
        model = svm.SVC(kernel=kernel, C=C, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error al entrenar el modelo SVM: {e}")
        return None

# Función para dibujar la frontera de decisión de un SVM
# Esta función es para las secciones de EXPLICACIÓN, no para el juego.
def plot_svm_decision_boundary(model, X, y, ax, title, show_support_vectors=True):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Asegurarse de que el modelo está entrenado antes de intentar predecir
    if model:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Dibuja los puntos para cada clase para una mejor leyenda
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', s=50, edgecolors='k', label='Clase 0 (Rojo)')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=50, edgecolors='k', label='Clase 1 (Azul)')
    
    # Dibujar vectores de soporte si existen y el modelo lo permite
    if show_support_vectors and model and hasattr(model, 'support_vectors_'):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                   linewidth=1, facecolors='none', edgecolors='green', label='Vectores de Soporte')
    
    ax.set_title(title)
    ax.set_xlabel("Característica 1")
    ax.set_ylabel("Característica 2")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.legend() # Asegúrate de que la leyenda se muestre


# --- Inicialización robusta de session_state para el módulo SVM ---
if 'svm_module_config' not in st.session_state:
    st.session_state.svm_module_config = {
        'svm_model_linear': None,
        'svm_model_rbf': None,
        'svm_data_blobs': None,
        'svm_data_circles': None,
        'svm_game_correct_count': 0,
        'svm_game_total_count': 0,
        'current_game_point': None,
        'current_game_label': None,
        'game_awaiting_guess': False,
        'show_svm_explanation': False # Nuevo estado para controlar la explicación de SVM
    }
else: # Asegurarse de que el juego también inicialice sus estados dentro del config si ya existe svm_module_config
    if 'svm_game_correct_count' not in st.session_state.svm_module_config:
        st.session_state.svm_module_config['svm_game_correct_count'] = 0
    if 'svm_game_total_count' not in st.session_state.svm_module_config:
        st.session_state.svm_module_config['svm_game_total_count'] = 0
    if 'current_game_point' not in st.session_state.svm_module_config:
        st.session_state.svm_module_config['current_game_point'] = None
    if 'current_game_label' not in st.session_state.svm_module_config:
        st.session_state.svm_module_config['current_game_label'] = None
    if 'game_awaiting_guess' not in st.session_state.svm_module_config:
        st.session_state.svm_module_config['game_awaiting_guess'] = False
    if 'show_svm_explanation' not in st.session_state.svm_module_config: # Inicializa si no existe
        st.session_state.svm_module_config['show_svm_explanation'] = False


# Generar datos de ejemplo para SVMs al inicio
if make_blobs and make_circles:
    if st.session_state.svm_module_config['svm_data_blobs'] is None:
        X_blobs, y_blobs = make_blobs(n_samples=100, centers=2, random_state=60, cluster_std=1.0) # Asegúrate de que los clusters no sean demasiado separados si quieres un juego desafiante
        st.session_state.svm_module_config['svm_data_blobs'] = (X_blobs, y_blobs)
    
    if st.session_state.svm_module_config['svm_data_circles'] is None:
        X_circles, y_circles = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
        st.session_state.svm_module_config['svm_data_circles'] = (X_circles, y_circles)

    # Entrenar modelos SVM si no están cargados
    if st.session_state.svm_module_config['svm_model_linear'] is None and st.session_state.svm_module_config['svm_data_blobs'] is not None:
        X_blobs, y_blobs = st.session_state.svm_module_config['svm_data_blobs']
        st.session_state.svm_module_config['svm_model_linear'] = train_simple_svm(X_blobs, y_blobs, kernel='linear')
        if st.session_state.svm_module_config['svm_model_linear']:
            st.success("Modelo SVM Lineal entrenado con éxito para ejemplo de blobs!")

    if st.session_state.svm_module_config['svm_model_rbf'] is None and st.session_state.svm_module_config['svm_data_circles'] is not None:
        X_circles, y_circles = st.session_state.svm_module_config['svm_data_circles']
        st.session_state.svm_module_config['svm_model_rbf'] = train_simple_svm(X_circles, y_circles, kernel='rbf')
        if st.session_state.svm_module_config['svm_model_rbf']:
            st.success("Modelo SVM RBF entrenado con éxito para ejemplo de círculos!")

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
        # Store client in session_state for access in other parts of the app
        st.session_state.client = client 
    except Exception as e:
        st.error(f"Error al inicializar cliente OpenAI con la clave proporcionada: {e}")
        client = None
        st.session_state.client = None # Ensure it's None in session_state too
else:
    st.warning("¡ATENCIÓN! La clave de la API de OpenAI no se ha encontrado en `secrets.toml`.")
    st.info("""
    Para usar el chatbot del Profesor IA (si lo implementas aquí), necesitas añadir tu clave de la API de OpenAI a tu archivo `secrets.toml`.
    """)
    OpenAI = None
    st.session_state.client = None # Ensure it's None if API key is missing

# --- Título y Explicación del Módulo ---
st.title("Laboratorio Interactivo: ¿Qué son las Máquinas de Vectores de Soporte (SVM)?")

st.markdown("""
¡Bienvenido al laboratorio donde aprenderemos sobre las **Máquinas de Vectores de Soporte (SVM)**!

---

### ¿Cómo funciona una SVM? ¡Es como encontrar la mejor línea divisoria!

Imagina que tienes un montón de juguetes de dos colores diferentes (por ejemplo, azules y rojos) mezclados en el suelo. Tu misión es dibujar una línea en el suelo para separarlos perfectamente en dos grupos, ¡uno a cada lado de la línea!

* **Encontrando la mejor separación:** Una SVM es un "cerebro" matemático que busca la mejor manera de dibujar esa línea (o una forma más complicada si los juguetes no están en línea recta). No solo busca *cualquier* línea, sino la que deje la mayor "zona de seguridad" o espacio entre los dos grupos de juguetes.
* **Los "juguetes clave" (Vectores de Soporte):** Para dibujar esa línea, la SVM solo necesita fijarse en algunos juguetes muy específicos: los que están más cerca de la línea divisoria. ¡Esos son los **Vectores de Soporte**! Son los "puntos de apoyo" que ayudan a la SVM a trazar la línea perfecta.
* **Clasificando nuevos juguetes:** Una vez que la SVM ha dibujado su línea, si le das un juguete nuevo, simplemente mira de qué lado de la línea cae y te dice si es azul o rojo.

¡Es como un clasificador experto que siempre busca la manera más clara de separar cosas!
""")

st.write("---")

# --- Sección de Explicación de SVM con Visualizaciones ---
st.header("El 'Clasificador Experto' (SVM) - ¡En Acción!")
st.markdown("Las Máquinas de Vectores de Soporte son fantásticas para encontrar la mejor "
             "manera de dividir datos en diferentes categorías. ¡Vamos a verlas en acción!")

if svm is not None and make_blobs and make_circles:
    st.subheader("1. Separando grupos claros con una línea recta (Kernel Lineal)")
    st.write("""
    Cuando los datos (o juguetes) están claramente separados y puedes dibujar una línea recta entre ellos,
    una SVM con un **Kernel Lineal** es perfecta. Busca la línea que maximice el espacio entre los grupos.
    """)

    if st.session_state.svm_module_config['svm_model_linear'] and st.session_state.svm_module_config['svm_data_blobs']:
        X_blobs, y_blobs = st.session_state.svm_module_config['svm_data_blobs']
        model_linear = st.session_state.svm_module_config['svm_model_linear']

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_svm_decision_boundary(model_linear, X_blobs, y_blobs, ax, "Separación Lineal de Datos", show_support_vectors=True)
        st.pyplot(fig)
        st.markdown("""
        Observa cómo la línea (la **frontera de decisión**) divide los dos grupos de puntos.
        Los puntos rodeados de verde son los **Vectores de Soporte**, ¡son los que el SVM usó para definir la línea!
        """)
    else:
        st.warning("No se pudo cargar o entrenar el modelo SVM Lineal. Asegúrate de que `scikit-learn` esté instalado.")

    st.subheader("2. Separando grupos con formas curvas (Kernel RBF - 'truco del Kernel')")
    st.write("""
    ¿Qué pasa si los juguetes no se pueden separar con una línea recta? ¡Imagina que los juguetes rojos están en el centro y los azules los rodean!
    Aquí es donde la SVM usa un truco mágico llamado **Kernel RBF (Radial Basis Function)**.
    Es como si la SVM mirara los juguetes en un espacio diferente donde sí puede dibujar una línea recta,
    ¡y luego esa línea se convierte en una forma curva cuando la vemos de vuelta en nuestro espacio normal!
    """)

    if st.session_state.svm_module_config['svm_model_rbf'] and st.session_state.svm_module_config['svm_data_circles']:
        X_circles, y_circles = st.session_state.svm_module_config['svm_data_circles']
        model_rbf = st.session_state.svm_module_config['svm_model_rbf']

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_svm_decision_boundary(model_rbf, X_circles, y_circles, ax, "Separación No Lineal con Kernel RBF", show_support_vectors=True)
        st.pyplot(fig)
        st.markdown("""
        ¡Mira cómo el SVM ha dibujado una **línea curva** para separar los círculos!
        Los **Vectores de Soporte** (puntos verdes) son cruciales para definir esta frontera.
        """)
    else:
        st.warning("No se pudo cargar o entrenar el modelo SVM RBF. Asegúrate de que `scikit-learn` esté instalado.")

else:
    st.error("No se pueden mostrar los ejemplos de SVM. Asegúrate de que `scikit-learn` esté correctamente instalado.")

st.write("---")

# --- Sección de Juego Interactivo: El Clasificador Humano (Juego 1 de tu CNN) ---
st.header("¡Juego: Tu Turno de Clasificar Datos!")
st.markdown(f"""
¡Ahora es tu oportunidad de ser un clasificador! Te mostraremos un punto y tendrás que adivinar a qué categoría pertenece, usando la frontera de decisión y los vectores de soporte para guiarte.
**Aciertos: {st.session_state.svm_module_config['svm_game_correct_count']} / {st.session_state.svm_module_config['svm_game_total_count']}**
""")

if svm is None or make_blobs is None or st.session_state.svm_module_config['svm_model_linear'] is None:
    st.warning("El juego no está disponible. Asegúrate de que `scikit-learn` esté instalado y el modelo SVM Lineal esté entrenado.")
else:
    def generate_new_game_point():
        X_blobs, y_blobs = st.session_state.svm_module_config['svm_data_blobs']
        model_linear = st.session_state.svm_module_config['svm_model_linear']
        
        # Estrategia para generar un punto cerca de la frontera de decisión
        n_candidates = 500 # Número de puntos candidatos a probar
        x_min, x_max = X_blobs[:, 0].min() - 0.5, X_blobs[:, 0].max() + 0.5
        y_min, y_max = X_blobs[:, 1].min() - 0.5, X_blobs[:, 1].max() + 0.5

        candidate_points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_candidates, 2))
        
        if hasattr(model_linear, 'decision_function'):
            distances = np.abs(model_linear.decision_function(candidate_points))
            closest_idx = np.argmin(distances)
            new_point_raw = candidate_points[closest_idx]
            predicted_true_label = model_linear.predict(new_point_raw.reshape(1, -1))[0]
        else: 
            idx = random.randint(0, n_candidates - 1)
            new_point_raw = candidate_points[idx]
            predicted_true_label = model_linear.predict(new_point_raw.reshape(1, -1))[0]


        st.session_state.svm_module_config['current_game_point'] = new_point_raw
        st.session_state.svm_module_config['current_game_label'] = predicted_true_label
        st.session_state.svm_module_config['game_awaiting_guess'] = True
        st.session_state.svm_module_config['show_svm_explanation'] = False # Ocultar la explicación al generar nuevo punto


    if not st.session_state.svm_module_config['game_awaiting_guess']:
        if st.button("¡Empezar una nueva ronda del juego!", key="start_svm_game_button"):
            generate_new_game_point()
            st.rerun()

    if st.session_state.svm_module_config['current_game_point'] is not None:
        st.subheader("Observa el punto y adivina:")
        
        fig, ax = plt.subplots(figsize=(8, 6)) # Aumentar un poco el tamaño para mayor claridad
        X_blobs, y_blobs = st.session_state.svm_module_config['svm_data_blobs']
        model_linear = st.session_state.svm_module_config['svm_model_linear']
        
        # Dibuja la frontera de decisión completa y los vectores de soporte
        plot_svm_decision_boundary(model_linear, X_blobs, y_blobs, ax, "Clasifica el Punto Morado", show_support_vectors=True)
        
        # Plot the new point
        point_color = 'purple'
        ax.scatter(st.session_state.svm_module_config['current_game_point'][0], st.session_state.svm_module_config['current_game_point'][1], 
                   s=250, color=point_color, edgecolors='black', linewidth=3, zorder=5, label='Nuevo Punto a Clasificar') # Tamaño y borde más grandes
        
        ax.set_title("¿A qué grupo pertenece el punto morado?", fontsize=16) # Título más prominente
        ax.set_xlabel("Característica X", fontsize=12)
        ax.set_ylabel("Característica Y", fontsize=12)
        ax.legend(fontsize=10) # Ajustar tamaño de leyenda si es necesario
        st.pyplot(fig)

        if st.session_state.svm_module_config['game_awaiting_guess']:
            user_guess = st.radio(
                "El punto morado parece ser del grupo...",
                ("Clase 0 (Rojo)", "Clase 1 (Azul)"),
                key="svm_user_guess"
            )

            if st.button("¡Verificar mi adivinanza!", key="check_svm_guess_button"):
                st.session_state.svm_module_config['svm_game_total_count'] += 1
                predicted_label_int = 0 if user_guess == "Clase 0 (Rojo)" else 1

                if predicted_label_int == st.session_state.svm_module_config['current_game_label']:
                    st.session_state.svm_module_config['svm_game_correct_count'] += 1
                    st.success(f"¡Correcto! El punto era de la **Clase {st.session_state.svm_module_config['current_game_label']}**.")
                else:
                    st.error(f"¡Incorrecto! El punto era de la **Clase {st.session_state.svm_module_config['current_game_label']}**.")
                
                st.session_state.svm_module_config['game_awaiting_guess'] = False
                st.session_state.svm_module_config['show_svm_explanation'] = True # Mostrar la explicación después de adivinar
                st.markdown(f"**Resultado actual del juego: {st.session_state.svm_module_config['svm_game_correct_count']} aciertos de {st.session_state.svm_module_config['svm_game_total_count']} intentos.**")
                st.button("¡Siguiente punto!", key="next_svm_point_button", on_click=generate_new_game_point)
                st.rerun() # Rerun to show the next button immediately
        else:
            st.write("Haz clic en '¡Siguiente punto!' para una nueva ronda.")
            if st.button("¡Siguiente punto!", key="next_svm_point_after_reveal", on_click=generate_new_game_point):
                st.rerun() # Ensure next button works after a guess

# --- Nueva Sección: ¿Qué son los Vectores de Soporte? (Explicación Post-Juego) ---
if st.session_state.svm_module_config['show_svm_explanation']:
    st.write("---")
    st.header("¡Has visto los Vectores de Soporte en Acción!")
    st.markdown("""
    En el juego, los puntos rodeados de **verde** son los **Vectores de Soporte**.
    ¿Recuerdas nuestra analogía de los juguetes en el suelo y la línea divisoria?

    Imagina que estás tratando de dibujar la línea más ancha posible para separar tus juguetes rojos de los azules.
    
    * **Los Vectores de Soporte son los "guardias fronterizos"**: No todos los juguetes son igual de importantes para decidir dónde va la línea. Solo los juguetes que están **más cerca** de la línea divisoria son realmente importantes. ¡Esos son los Vectores de Soporte! Son como los "guardias fronterizos" que definen exactamente dónde debe pasar la línea para mantener la separación más grande.
    * **Definen el "margen de seguridad"**: La línea que dibuja el SVM se coloca de tal manera que haya la mayor distancia posible a los Vectores de Soporte de cada clase. Esta distancia se llama el "margen". Los Vectores de Soporte son los puntos que se encuentran justo en los bordes de este margen.
    * **Solo importan unos pocos**: Lo más genial es que una SVM solo necesita estos pocos puntos (los vectores de soporte) para dibujar la línea perfecta. Si quitas o mueves otros juguetes que no son vectores de soporte, ¡la línea no cambia! Solo si mueves o añades un nuevo juguete que se convierte en un vector de soporte, la línea podría ajustarse.

    ¡Son los héroes silenciosos que le dicen a la SVM dónde trazar la línea para una clasificación súper precisa!
    """)
    st.write("---")


# --- Sección de Chatbot de Juego con Vectorín ---
st.header("¡Juega y Aprende con Vectorín sobre las Máquinas de Vectores de Soporte (SVM)!")
st.markdown("¡Hola! Soy **Vectorín**, el explorador que encuentra los vectores de soporte y traza los hiperplanos perfectos en los datos. ¿Listo para sumergirte en el espacio de características y optimizar los márgenes?")


# Inicializa el estado del juego y los mensajes del chat para Vectorín
if "vectorin_game_active" not in st.session_state:
    st.session_state.vectorin_game_active = False
if "vectorin_game_messages" not in st.session_state:
    st.session_state.vectorin_game_messages = []
if "vectorin_current_question" not in st.session_state:
    st.session_state.vectorin_current_question = None
if "vectorin_current_options" not in st.session_state:
    st.session_state.vectorin_current_options = {}
if "vectorin_correct_answer" not in st.session_state:
    st.session_state.vectorin_correct_answer = None
if "vectorin_awaiting_next_game_decision" not in st.session_state:
    st.session_state.vectorin_awaiting_next_game_decision = False
if "vectorin_game_needs_new_question" not in st.session_state:
    st.session_state.vectorin_game_needs_new_question = False
if "vectorin_correct_streak" not in st.session_state:
    st.session_state.vectorin_correct_streak = 0
if "last_played_question_vectorin" not in st.session_state:
    st.session_state.last_played_question_vectorin = None

# System prompt para el juego de preguntas de Vectorín
vectorin_game_system_prompt = f"""
Eres un **experto magistral en Máquinas de Vectores de Soporte (SVM)**. Tu conocimiento abarca desde los fundamentos de la **clasificación lineal y no lineal** hasta los detalles intrínsecos de los **hiperplanos, vectores de soporte, márgenes y el truco del kernel**. Comprendes a fondo cómo las SVM manejan datos separables e inseparables, la importancia del parámetro de regularización (C) y el impacto de los diferentes tipos de kernels (lineal, polinómico, RBF). Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de las SVM mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/posgrado** para los usuarios más avanzados.

**TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
**¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

**Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
Pregunta: [Tu pregunta aquí]
A) [Opción A]
B) [Opción B]
C) [Opción C]
RespuestaCorrecta: [A, B o C]

**Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
[Mensaje de Correcto/Incorrecto, ej: "¡Has encontrado el vector de soporte clave! Tu comprensión es nítida." o "Esa clasificación no fue la más clara. Repasemos cómo Vectorín encuentra el margen óptimo."]
[Breve explicación del concepto, adecuada al nivel del usuario, ej: "Las Máquinas de Vectores de Soporte buscan el hiperplano óptimo que maximiza el margen entre las clases..."]
[Pregunta para continuar, ej: "¿Listo para trazar más hiperplanos?" o "¿Quieres explorar más el espacio de características?"]

**Reglas adicionales para el Experto en SVM (Vectorín):**
* **Enfoque Riguroso en SVM:** Todas tus preguntas y explicaciones deben girar en torno a las Máquinas de Vectores de Soporte. Cubre sus fundamentos (clasificación, regresión si aplica, pero prioriza clasificación), la idea de margen y vectores de soporte, el hiperplano óptimo, la separación lineal y no lineal.
* **Conceptos Clave a Cubrir (¡VARIEDAD!):**
    * **Idea Fundamental de SVM:** Qué es, su objetivo (maximizar el margen).
    * **Hiperplano:** Definición, su rol.
    * **Vectores de Soporte:** Qué son, por qué son cruciales.
    * **Margen:** Definición, margen duro vs. margen suave.
    * **Problemas Linealmente Separables:** Cómo las SVM los resuelven.
    * **Problemas No Linealmente Separables:** Cómo las SVM los resuelven (truco del kernel).
    * **Truco del Kernel:** Su propósito, cómo funciona (mapeo a dimensiones superiores).
    * **Tipos de Funciones Kernel:**
        * **Lineal:** Cuándo se usa.
        * **Polinómico:** Propósito, grado.
        * **Radial Basis Function (RBF) / Gaussiano:** Parámetro gamma, flexibilidad.
    * **Parámetro de Regularización (C):** Su impacto en el margen y el número de errores de clasificación (trade-off entre margen y error).
    * **Ventajas y Desventajas de las SVM:** Casos de uso, cuándo funcionan bien.
    * **SVM para Clasificación Multiclase:** Estrategias (One-vs-One, One-vs-Rest).

* **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.vectorin_correct_streak} preguntas correctas consecutivas.
    * **Nivel 1 (Aprendiz de Hiperplano – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de SVM y cómo separan datos simples.
        * *Tono:* "Estás explorando tus primeras fronteras de decisión. ¡Vectorín te guiará en este camino de aprendizaje!"
    * **Nivel 2 (Navegante de Vectores – 3-5 respuestas correctas):** Tono más técnico. Introduce conceptos como hiperplanos, margen y vectores de soporte. Preguntas sobre los componentes básicos y el proceso fundamental.
        * *Tono:* "¡Tus algoritmos de clasificación son más precisos! Estás aprendiendo a identificar los vectores clave."
    * **Nivel 3 (Maestro del Kernel – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en el truco del kernel, los diferentes tipos de funciones kernel (RBF, Polinómico) y el parámetro de regularización (C).
        * *Tono:* "¡Estás transformando datos en dimensiones superiores! Tu dominio de los kernels te permite clasificar los datos más complejos."
    * **Nivel Virtuoso (Ingeniero de Clasificadores – 9+ respuestas correctas):** Tono de **especialista en optimización y aplicación de SVM**. Preguntas sobre la optimización de hiperparámetros, la robustez de las SVM ante el ruido, o la selección del kernel adecuado para diferentes escenarios del mundo real. Se esperan respuestas que demuestren una comprensión teórica y práctica profunda.
        * *Tono:* "Tu maestría en SVM te permite diseñar clasificadores que son verdaderas obras de arte de la ingeniería de datos. ¡Un auténtico Virtuoso de la Clasificación!"
    * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
    * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
    * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

* **Ejemplos y Casos de Uso (Adaptados al Nivel):**
    * **Nivel 1:** Cómo una SVM puede separar manzanas de naranjas, o correos electrónicos spam de no-spam usando características simples.
    * **Nivel 2:** Explicar cómo una SVM distingue entre diferentes tipos de datos, mostrando la importancia del margen.
    * **Nivel 3:** Un modelo para clasificar tipos de tumores (benignos/malignos) a partir de características de imagen, destacando la necesidad de kernels no lineales.
    * **Nivel Virtuoso:** La aplicación de SVM en bioinformática para la clasificación de secuencias de ADN o proteínas, discutiendo la selección de kernel y la interpretación de los vectores de soporte en espacios de alta dimensión.

* **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
* **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
* **Siempre responde en español de España.**
* **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre las MÁQUINAS DE VECTORES DE SOPORTE, y asegúrate de que no se parezca a las anteriores.
"""

# Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
def parse_vectorin_question_response(raw_text):
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
def parse_vectorin_feedback_response(raw_text):
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    if len(lines) >= 3:
        return lines[0], lines[1], lines[2]
    st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
    return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"

# --- Funciones para subir de nivel directamente ---
def set_vectorin_level(target_streak, level_name):
    st.session_state.vectorin_correct_streak = target_streak
    st.session_state.vectorin_game_active = True
    st.session_state.vectorin_game_messages = []
    st.session_state.vectorin_current_question = None
    st.session_state.vectorin_current_options = {}
    st.session_state.vectorin_correct_answer = None
    st.session_state.vectorin_game_needs_new_question = True
    st.session_state.vectorin_awaiting_next_game_decision = False
    st.session_state.vectorin_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}** de Vectorín! Prepárate para preguntas más desafiantes sobre las Máquinas de Vectores de Soporte. ¡Aquí va tu primera!"})
    st.rerun()

# Botones para iniciar o reiniciar el juego y subir de nivel
col_game_buttons_vectorin, col_level_up_buttons_vectorin = st.columns([1, 2])

with col_game_buttons_vectorin:
    if st.button("¡Vamos a jugar con Vectorín!", key="start_vectorin_game_button"):
        st.session_state.vectorin_game_active = True
        st.session_state.vectorin_game_messages = []
        st.session_state.vectorin_current_question = None
        st.session_state.vectorin_current_options = {}
        st.session_state.vectorin_correct_answer = None
        st.session_state.vectorin_game_needs_new_question = True
        st.session_state.vectorin_awaiting_next_game_decision = False
        st.session_state.vectorin_correct_streak = 0
        st.session_state.last_played_question_vectorin = None
        st.rerun()
        
with col_level_up_buttons_vectorin:
    st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un experto en SVM? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
    col_lvl1_vectorin, col_lvl2_vectorin, col_lvl3_vectorin = st.columns(3) # Tres columnas para los botones de nivel
    with col_lvl1_vectorin:
        if st.button("Subir a Nivel Medio (Vectorín - Navegante)", key="level_up_medium_vectorin"):
            set_vectorin_level(3, "Medio") # 3 respuestas correctas para Nivel Medio
    with col_lvl2_vectorin:
        if st.button("Subir a Nivel Avanzado (Vectorín - Maestro)", key="level_up_advanced_vectorin"):
            set_vectorin_level(6, "Avanzado") # 6 respuestas correctas para Nivel Avanzado
    with col_lvl3_vectorin:
        if st.button("👑 ¡Virtuoso del Clasificador! (Vectorín)", key="level_up_champion_vectorin"):
            set_vectorin_level(9, "Virtuoso") # 9 respuestas correctas para Nivel Virtuoso


# Mostrar mensajes del juego del chatbot
for message in st.session_state.vectorin_game_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Lógica del juego del chatbot si está activo
if st.session_state.vectorin_game_active:
    if st.session_state.vectorin_current_question is None and st.session_state.vectorin_game_needs_new_question and not st.session_state.vectorin_awaiting_next_game_decision:
        with st.spinner("Vectorín está preparando una pregunta sobre Máquinas de Vectores de Soporte..."):
            try:
                # Ensure 'client' is defined
                if st.session_state.client is None:
                    st.error("Error: OpenAI client not initialized. Please ensure your API key is set.")
                    st.session_state.vectorin_game_active = False
                    st.rerun()
                    
                client = st.session_state.client 

                game_messages_for_api = [{"role": "system", "content": vectorin_game_system_prompt}]
                if st.session_state.vectorin_game_messages:
                    last_message = st.session_state.vectorin_game_messages[-1]
                    # Solo añadir el último mensaje si no es el mensaje inicial del salto de nivel
                    if last_message["role"] == "user" or "¡Has saltado directamente al" not in last_message["content"]:
                        game_messages_for_api.append({"role": last_message['role'], "content": last_message['content']})


                game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre QUÉ SON LAS MÁQUINAS DE VECTORES DE SOPORTE siguiendo el formato exacto. ¡Recuerda, la pregunta debe ser muy VARIADA y CREATIVA, y no se debe parecer a las anteriores!"})

                game_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=game_messages_for_api,
                    temperature=0.8,
                    max_tokens=300
                )
                raw_vectorin_question_text = game_response.choices[0].message.content
                question, options, correct_answer_key = parse_vectorin_question_response(raw_vectorin_question_text)

                if question:
                    st.session_state.vectorin_current_question = question
                    st.session_state.vectorin_current_options = options
                    st.session_state.vectorin_correct_answer = correct_answer_key

                    display_question_text = f"**Nivel {int(st.session_state.vectorin_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.vectorin_correct_streak}**\n\n**Pregunta de Vectorín:** {question}\n\n"
                    for key in sorted(options.keys()):
                        display_question_text += f"{key}) {options[key]}\n"

                    st.session_state.vectorin_game_messages.append({"role": "assistant", "content": display_question_text})
                    st.session_state.vectorin_game_needs_new_question = False
                    st.rerun()
                else:
                    st.session_state.vectorin_game_messages.append({"role": "assistant", "content": "¡Lo siento! Vectorín no pudo preparar la pregunta en el formato correcto. ¿Puedes pulsar 'VAMOS A JUGAR' de nuevo?"})
                    st.session_state.vectorin_game_active = False
                    st.rerun()

            except Exception as e:
                st.error(f"¡Oops! Vectorín no pudo hacer la pregunta. Error: {e}")
                st.session_state.vectorin_game_messages.append({"role": "assistant", "content": "¡Lo siento! Vectorín tiene un pequeño problema para hacer preguntas ahora. ¿Puedes intentarlo de nuevo?"})
                st.session_state.vectorin_game_active = False
                st.rerun()


    if st.session_state.vectorin_current_question is not None and not st.session_state.vectorin_awaiting_next_game_decision:
        # Audio de la pregunta
        if st.session_state.get('last_played_question_vectorin') != st.session_state.vectorin_current_question:
            try:
                tts_text = f"Nivel {int(st.session_state.vectorin_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.vectorin_correct_streak}. Pregunta de Vectorín: {st.session_state.vectorin_current_question}. Opción A: {st.session_state.vectorin_current_options.get('A', '')}. Opción B: {st.session_state.vectorin_current_options.get('B', '')}. Opción C: {st.session_state.vectorin_current_options.get('C', '')}."
                tts = gTTS(text=tts_text, lang='es', slow=False)
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                st.session_state.last_played_question_vectorin = st.session_state.vectorin_current_question
            except Exception as e:
                st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")


        with st.form("vectorin_game_form", clear_on_submit=True):
            radio_placeholder = st.empty()
            with radio_placeholder.container():
                st.markdown("Elige tu respuesta:")
                user_choice = st.radio(
                    "Elige tu respuesta:",
                    options=list(st.session_state.vectorin_current_options.keys()),
                    format_func=lambda x: f"{x}) {st.session_state.vectorin_current_options[x]}",
                    key="vectorin_answer_radio_buttons",
                    label_visibility="collapsed"
                )

            submit_button = st.form_submit_button("Enviar Respuesta")

        if submit_button:
            st.session_state.vectorin_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_choice}) {st.session_state.vectorin_current_options[user_choice]}"})
            prev_streak = st.session_state.vectorin_correct_streak

            # Lógica para actualizar el contador de respuestas correctas
            if user_choice == st.session_state.vectorin_correct_answer:
                st.session_state.vectorin_correct_streak += 1
            else:
                st.session_state.vectorin_correct_streak = 0

            radio_placeholder.empty()

            # --- Lógica de subida de nivel ---
            if st.session_state.vectorin_correct_streak > 0 and \
               st.session_state.vectorin_correct_streak % 3 == 0 and \
               st.session_state.vectorin_correct_streak > prev_streak:
                
                if st.session_state.vectorin_correct_streak < 9: # Niveles Aprendiz, Navegante, Maestro
                    current_level_text = ""
                    if st.session_state.vectorin_correct_streak == 3:
                        current_level_text = "Navegante de Vectores"
                    elif st.session_state.vectorin_correct_streak == 6:
                        current_level_text = "Maestro del Kernel"
                    
                    level_up_message = f"🎉 ¡Increíble! ¡Has respondido {st.session_state.vectorin_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de Máquinas de Vectores de Soporte. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a explorador/a de datos! 🚀"
                    st.session_state.vectorin_game_messages.append({"role": "assistant", "content": level_up_message})
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
                elif st.session_state.vectorin_correct_streak >= 9:
                    medals_earned = (st.session_state.vectorin_correct_streak - 6) // 3 
                    medal_message = f"🏅 ¡FELICITACIONES, VIRTUOSO DEL CLASIFICADOR! ¡Has ganado tu {medals_earned}ª Medalla de Hiperplano! ¡Tu habilidad para encontrar el margen óptimo es asombrosa y digna de un verdadero EXPERTO en SVM! ¡Sigue así! 🌟"
                    st.session_state.vectorin_game_messages.append({"role": "assistant", "content": medal_message})
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
                        level_up_message_champion = f"¡Has desbloqueado el **Nivel Virtuoso (Ingeniero de Clasificadores)**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros científicos de datos que entienden los secretos de las SVM! ¡Adelante!"
                        st.session_state.vectorin_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                        try:
                            tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                            audio_fp_level_up_champion = io.BytesIO()
                            tts_level_up_champion.write_to_fp(audio_fp_level_up_champion) 
                            audio_fp_level_up_champion.seek(0)
                            st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0, autoplay=True)
                            time.sleep(2)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de campeón: {e}")


            # Generar feedback de Vectorín
            with st.spinner("Vectorín está revisando tu respuesta..."):
                try:
                    # Ensure 'client' is defined
                    if st.session_state.client is None:
                        st.error("Error: OpenAI client not initialized. Cannot generate feedback.")
                        st.session_state.vectorin_game_active = False
                        st.rerun()
                        
                    client = st.session_state.client 

                    feedback_prompt = f"""
                    El usuario respondió '{user_choice}'. La pregunta era: '{st.session_state.vectorin_current_question}'.
                    La respuesta correcta era '{st.session_state.vectorin_correct_answer}'.
                    Da feedback como Vectorín.
                    Si es CORRECTO, el mensaje es "¡Has encontrado el vector de soporte clave! Tu comprensión es nítida." o similar.
                    Si es INCORRECTO, el mensaje es "Esa clasificación no fue la más clara. Repasemos cómo Vectorín encuentra el margen óptimo." o similar.
                    Luego, una explicación breve del concepto, adecuada al nivel del usuario.
                    Finalmente, pregunta: "¿Listo para trazar más hiperplanos?".
                    **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                    """
                    feedback_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": vectorin_game_system_prompt},
                            {"role": "user", "content": feedback_prompt}
                        ],
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_vectorin_feedback_text = feedback_response.choices[0].message.content

                    feedback_msg, explanation_msg, next_question_prompt = parse_vectorin_feedback_response(raw_vectorin_feedback_text)

                    st.session_state.vectorin_game_messages.append({"role": "assistant", "content": feedback_msg})
                    st.session_state.vectorin_game_messages.append({"role": "assistant", "content": explanation_msg})
                    st.session_state.vectorin_game_messages.append({"role": "assistant", "content": next_question_prompt})

                    try:
                        tts = gTTS(text=f"{feedback_msg}. {explanation_msg}. {next_question_prompt}", lang='es', slow=False)
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        audio_fp.seek(0)
                        st.audio(audio_fp, format="audio/mp3", start_time=0, autoplay=True)
                    except Exception as e:
                        st.warning(f"No se pudo reproducir el audio de feedback: {e}")


                    st.session_state.vectorin_current_question = None
                    st.session_state.vectorin_current_options = {}
                    st.session_state.vectorin_correct_answer = None
                    st.session_state.vectorin_game_needs_new_question = False
                    st.session_state.vectorin_awaiting_next_game_decision = True

                    st.rerun()

                except Exception as e:
                    st.error(f"Ups, Vectorín no pudo procesar tu respuesta. Error: {e}")
                    st.session_state.vectorin_game_messages.append({"role": "assistant", "content": "Lo siento, Vectorín tiene un pequeño problema técnico ahora mismo. ¡Pero me encantaría ver tu respuesta!"})


    if st.session_state.vectorin_awaiting_next_game_decision:
        st.markdown("---")
        st.markdown("¿Qué quieres hacer ahora?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Sí, quiero jugar más preguntas", key="play_more_questions_vectorin"):
                st.session_state.vectorin_game_needs_new_question = True
                st.session_state.vectorin_awaiting_next_game_decision = False
                st.session_state.vectorin_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío con las Máquinas de Vectores de Soporte!"})
                st.rerun()
        with col2:
            if st.button("👎 No, ya no quiero jugar más", key="stop_playing_vectorin"):
                st.session_state.vectorin_game_active = False
                st.session_state.vectorin_awaiting_next_game_decision = False
                st.session_state.vectorin_game_messages.append({"role": "assistant", "content": "¡De acuerdo! ¡Gracias por explorar el espacio de características conmigo! Espero que hayas aprendido mucho. ¡Hasta la próxima clasificación!"})
                st.rerun()

else: 
    if "client" not in st.session_state or st.session_state.client is None: # Asegúrate de que client esté en session_state
        st.info("Para usar la sección de preguntas de Vectorín, necesitas configurar tu clave de API de OpenAI en `secrets.toml`.")