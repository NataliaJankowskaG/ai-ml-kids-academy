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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import plot_tree # Para visualizar árboles individuales (opcional)
    from sklearn.datasets import make_classification # Para generar datos de ejemplo
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    st.error("Las librerías 'scikit-learn' no están instaladas. Por favor, instálalas usando: pip install scikit-learn")
    RandomForestClassifier = None
    GradientBoostingClassifier = None
    make_classification = None
    train_test_split = None
    accuracy_score = None

# Importar OpenAI si se usa el chatbot
try:
    from openai import OpenAI
except ImportError:
    st.error("La librería 'openai' no está instalada. Por favor, instálala usando: pip install openai")
    OpenAI = None

# --- Configuración de la página ---
st.set_page_config(
    page_title="¿Qué son los Métodos Ensemble (Random Forest, Boosting)?",
    layout="wide"
)

# --- Funciones auxiliares para Métodos Ensemble ---

@st.cache_resource
def train_ensemble_models(X, y):
    """Entrena un modelo Random Forest y un modelo Gradient Boosting."""
    if RandomForestClassifier is None or GradientBoostingClassifier is None:
        return None, None
    try:
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf_model.fit(X, y)

        # Gradient Boosting
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
        gb_model.fit(X, y)
        return rf_model, gb_model
    except Exception as e:
        st.error(f"Error al entrenar los modelos ensemble: {e}")
        return None, None

def plot_ensemble_decision_boundary(model, X, y, ax, title):
    """Dibuja la frontera de decisión para un modelo ensemble."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
    
    if model:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', s=50, edgecolors='k', label='Clase 0 (Rojo)')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=50, edgecolors='k', label='Clase 1 (Azul)')
    
    ax.set_title(title)
    ax.set_xlabel("Característica 1")
    ax.set_ylabel("Característica 2")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.legend()

# --- Inicialización robusta de session_state ---

# Inicialización para el módulo Ensemble
if 'ensemble_module_config' not in st.session_state:
    st.session_state.ensemble_module_config = {
        'rf_model': None,
        'gb_model': None,
        'ensemble_data': None,
        'ensemble_game_correct_count': 0,
        'ensemble_game_total_count': 0,
        'current_game_point_ensemble': None,
        'current_game_label_ensemble': None,
        'game_awaiting_guess_ensemble': False,
        'show_ensemble_explanation': False,
        'chatbot_messages': [{"role": "assistant", "content": "¡Hola! Soy el **Oráculo Ensemble**. ¿Listo para explorar el mundo de los métodos de conjunto? Pregúntame lo que quieras."}]
    }

# Inicialización para el chatbot de juego "Bosco el Entrenador de Equipos"
# Asegúrate de que TODAS estas variables estén inicializadas
if "ensemble_chatbot_game_active" not in st.session_state:
    st.session_state.ensemble_chatbot_game_active = False
if "ensemble_chatbot_game_messages" not in st.session_state:
    st.session_state.ensemble_chatbot_game_messages = []
if "ensemble_chatbot_current_question" not in st.session_state:
    st.session_state.ensemble_chatbot_current_question = None
if "ensemble_chatbot_current_options" not in st.session_state:
    st.session_state.ensemble_chatbot_current_options = {}
if "ensemble_chatbot_correct_answer" not in st.session_state:
    st.session_state.ensemble_chatbot_correct_answer = None
if "ensemble_chatbot_awaiting_next_game_decision" not in st.session_state:
    st.session_state.ensemble_chatbot_awaiting_next_game_decision = False
if "ensemble_chatbot_game_needs_new_question" not in st.session_state:
    st.session_state.ensemble_chatbot_game_needs_new_question = False
if "ensemble_chatbot_correct_streak" not in st.session_state:
    st.session_state.ensemble_chatbot_correct_streak = 0 # Esta es la variable que faltaba inicializar explícitamente al principio
if "ensemble_chatbot_last_played_question" not in st.session_state:
    st.session_state.ensemble_chatbot_last_played_question = None


# Generar datos de ejemplo para modelos Ensemble al inicio
if make_classification:
    if st.session_state.ensemble_module_config['ensemble_data'] is None:
        X_ensemble, y_ensemble = make_classification(n_samples=150, n_features=2, n_redundant=0, n_informative=2,
                                                     n_clusters_per_class=1, random_state=42, class_sep=1.5)
        st.session_state.ensemble_module_config['ensemble_data'] = (X_ensemble, y_ensemble)

    # Entrenar modelos ensemble si no están cargados
    if st.session_state.ensemble_module_config['rf_model'] is None and st.session_state.ensemble_module_config['ensemble_data'] is not None:
        X_ensemble, y_ensemble = st.session_state.ensemble_module_config['ensemble_data']
        rf_model_temp, gb_model_temp = train_ensemble_models(X_ensemble, y_ensemble)
        st.session_state.ensemble_module_config['rf_model'] = rf_model_temp
        st.session_state.ensemble_module_config['gb_model'] = gb_model_temp
        if st.session_state.ensemble_module_config['rf_model'] and st.session_state.ensemble_module_config['gb_model']:
            st.success("Modelos Ensemble (Random Forest, Gradient Boosting) entrenados con éxito!")

# Resto de tu código... (sin cambios para las siguientes secciones)

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

# --- Configuración de la API de OpenAI ---
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
    Para usar el chatbot del Oráculo Ensemble o el juego de preguntas, necesitas añadir tu clave de la API de OpenAI a tu archivo `secrets.toml`.
    """)
    OpenAI = None

# --- Título y Explicación del Módulo ---
st.title("Laboratorio Interactivo: ¿Qué son los Métodos Ensemble (Random Forest, Boosting)?")

st.markdown("""
¡Bienvenido al laboratorio donde exploraremos el fascinante mundo de los **Métodos Ensemble**!

---

### ¿Qué es un Método Ensemble? ¡Es como un equipo de expertos tomando una decisión!

Imagina que tienes que resolver un problema importante, como adivinar el resultado de un partido. En lugar de preguntarle a una sola persona, decides preguntarle a un **equipo** de expertos. Cada experto da su opinión, y luego combinas todas esas opiniones para llegar a la mejor conclusión.

¡Eso es exactamente lo que hace un método Ensemble en Machine Learning! En lugar de usar un solo modelo (como un solo "cerebro" o algoritmo) para predecir algo, usa **múltiples modelos** (llamados "estimadores base") y combina sus predicciones para obtener un resultado más preciso y robusto.

#### ¿Por qué usar un equipo en lugar de un solo experto?
* **Más preciso:** Un equipo de expertos suele cometer menos errores que un solo experto, porque los errores individuales se compensan.
* **Más robusto:** Si un experto se equivoca mucho en algo, los otros expertos del equipo pueden corregirlo. El modelo es menos sensible a ruidos o datos atípicos.
* **Mejor generalización:** El equipo aprende de manera más diversa y puede aplicar su conocimiento a nuevos datos de forma más efectiva.

Hay dos estrategias principales para formar estos "equipos":

1.  **"Pedir opiniones a muchos y promediar": Random Forest (Bosque Aleatorio)**
2.  **"Aprender de los errores del equipo": Boosting (Refuerzo)**
""")

st.write("---")

# --- Sección de Explicación de Random Forest ---
st.header("1. Random Forest: El 'Bosque' de Decisiones Aleatorias")
st.markdown("""
Imagina un **bosque lleno de árboles**. Cada árbol es un pequeño "experto" en tomar decisiones (un árbol de decisión).

* **Muchos árboles, muchas opiniones:** Un **Random Forest** crea **muchos árboles de decisión independientes**. Pero aquí está el truco: cada árbol se entrena con una porción **aleatoria** diferente de los datos y considera solo un subconjunto **aleatorio** de características. Esto asegura que los árboles sean diversos y no "vean" exactamente lo mismo.
* **Voto Mayoritario:** Cuando quieres clasificar un nuevo dato, cada árbol del bosque "vota" por una clase. La clase que recibe la mayoría de los votos es la predicción final del Random Forest.

**Beneficios:** Es muy potente, reduce el sobreajuste (overfitting) y es bueno con datos complejos.
""")

if RandomForestClassifier is not None and st.session_state.ensemble_module_config['rf_model'] and st.session_state.ensemble_module_config['ensemble_data']:
    X_ensemble, y_ensemble = st.session_state.ensemble_module_config['ensemble_data']
    rf_model = st.session_state.ensemble_module_config['rf_model']

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_ensemble_decision_boundary(rf_model, X_ensemble, y_ensemble, ax, "Frontera de Decisión de Random Forest")
    st.pyplot(fig)
    st.markdown("""
    Observa cómo el **Random Forest** crea una frontera de decisión compleja pero suave.
    Es el resultado del "consenso" de muchos árboles de decisión individuales.
    """)
else:
    st.warning("No se pudo cargar o entrenar el modelo Random Forest. Asegúrate de que `scikit-learn` esté instalado.")

st.write("---")

# --- Sección de Explicación de Boosting ---
st.header("2. Boosting: Aprendiendo de los Errores del Pasado")
st.markdown("""
Ahora, imagina un equipo de "estudiantes" que aprenden uno tras otro, corrigiendo los errores del anterior.

* **Aprendizaje Secuencial:** El **Boosting** no entrena a todos los modelos a la vez. En cambio, entrena a los modelos de forma **secuencial**.
* **Foco en los Errores:** El primer modelo se entrena con los datos. Luego, el segundo modelo se entrena dándole más "peso" o atención a los datos que el primer modelo clasificó incorrectamente. El tercer modelo hace lo mismo, y así sucesivamente.
* **Combinación Ponderada:** Al final, las predicciones de todos los modelos se combinan, pero los modelos que fueron "mejores" o se centraron en errores más difíciles tienen más influencia.

**Dos tipos populares de Boosting:**
* **AdaBoost:** Cada nuevo modelo se centra en los ejemplos que los modelos anteriores clasificaron mal.
* **Gradient Boosting (como XGBoost, LightGBM):** Construye modelos predictivos de forma secuencial, donde cada modelo corrige los errores (residuos) del modelo anterior.

**Beneficios:** Muy potente para alcanzar alta precisión, a menudo uno de los mejores algoritmos en muchos problemas.
""")

if GradientBoostingClassifier is not None and st.session_state.ensemble_module_config['gb_model'] and st.session_state.ensemble_module_config['ensemble_data']:
    X_ensemble, y_ensemble = st.session_state.ensemble_module_config['ensemble_data']
    gb_model = st.session_state.ensemble_module_config['gb_model']

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_ensemble_decision_boundary(gb_model, X_ensemble, y_ensemble, ax, "Frontera de Decisión de Gradient Boosting")
    st.pyplot(fig)
    st.markdown("""
    Aquí vemos la frontera de decisión creada por un modelo de **Boosting**.
    Nota cómo también es capaz de manejar patrones complejos, aprendiendo de forma incremental.
    """)
else:
    st.warning("No se pudo cargar o entrenar el modelo Gradient Boosting. Asegúrate de que `scikit-learn` esté instalado.")

st.write("---")

# --- Sección de Juego Interactivo: El Juego del Oráculo Ensemble ---
st.header("¡Juego Interactivo: Tu Oráculo Ensemble Personal!")
st.markdown(f"""
¡Es hora de poner a prueba tu intuición como un "Oráculo Ensemble"! Te mostraremos un punto y, basándote en cómo los modelos ensemble dividen el espacio, tendrás que adivinar a qué categoría pertenece.
**Aciertos: {st.session_state.ensemble_module_config['ensemble_game_correct_count']} / {st.session_state.ensemble_module_config['ensemble_game_total_count']}**
""")

if (RandomForestClassifier is None or make_classification is None or
    st.session_state.ensemble_module_config['rf_model'] is None):
    st.warning("El juego no está disponible. Asegúrate de que `scikit-learn` esté instalado y los modelos Ensemble estén entrenados.")
else:
    def generate_new_game_point_ensemble():
        X_ensemble, y_ensemble = st.session_state.ensemble_module_config['ensemble_data']
        rf_model = st.session_state.ensemble_module_config['rf_model']

        n_candidates = 1000 # Increased candidates for better variety near boundary
        x_min, x_max = X_ensemble[:, 0].min() - 0.5, X_ensemble[:, 0].max() + 0.5
        y_min, y_max = X_ensemble[:, 1].min() - 0.5, X_ensemble[:, 1].max() + 0.5

        candidate_points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_candidates, 2))

        # Decide whether to pick a point near the boundary or a completely random point
        if random.random() < 0.2: # 20% chance for a completely random point
            idx = random.randint(0, n_candidates - 1)
            new_point_raw = candidate_points[idx]
        elif hasattr(rf_model, 'predict_proba'):
            probabilities = rf_model.predict_proba(candidate_points)
            distances_from_boundary = np.abs(probabilities - 0.5).sum(axis=1)

            # Get indices of points sorted by distance from boundary (closest first)
            sorted_indices = np.argsort(distances_from_boundary)

            # Select from the top N% of points closest to the boundary
            top_n_percent = int(n_candidates * 0.1) # Top 10%
            if top_n_percent == 0: top_n_percent = 1 # Ensure at least one point is selected
            
            # Randomly pick one index from these top N% closest points
            chosen_idx_in_sorted_list = random.randint(0, top_n_percent - 1)
            actual_chosen_idx = sorted_indices[chosen_idx_in_sorted_list]

            new_point_raw = candidate_points[actual_chosen_idx]
        else: # Fallback if no predict_proba and not the 20% random chance
            # Keep it truly random for variety if predict_proba is not available
            idx = random.randint(0, n_candidates - 1)
            new_point_raw = candidate_points[idx]

        predicted_true_label = rf_model.predict(new_point_raw.reshape(1, -1))[0]

        st.session_state.ensemble_module_config['current_game_point_ensemble'] = new_point_raw
        st.session_state.ensemble_module_config['current_game_label_ensemble'] = predicted_true_label
        st.session_state.ensemble_module_config['game_awaiting_guess_ensemble'] = True
        st.session_state.ensemble_module_config['show_ensemble_explanation'] = False

    if not st.session_state.ensemble_module_config['game_awaiting_guess_ensemble']:
        if st.button("¡Empezar una nueva ronda del juego Ensemble!", key="start_ensemble_game_button"):
            generate_new_game_point_ensemble()
            st.rerun()

    if st.session_state.ensemble_module_config['current_game_point_ensemble'] is not None:
        st.subheader("Observa el punto y adivina:")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        X_ensemble, y_ensemble = st.session_state.ensemble_module_config['ensemble_data']
        
        # Usar Random Forest para la visualización del juego
        ensemble_model_for_game = st.session_state.ensemble_module_config['rf_model']
        plot_ensemble_decision_boundary(ensemble_model_for_game, X_ensemble, y_ensemble, ax, "Clasifica el Punto Morado (Random Forest)")
        
        point_color = 'purple'
        ax.scatter(st.session_state.ensemble_module_config['current_game_point_ensemble'][0], st.session_state.ensemble_module_config['current_game_point_ensemble'][1], 
                   s=250, color=point_color, edgecolors='black', linewidth=3, zorder=5, label='Nuevo Punto a Clasificar')
        
        ax.set_title("¿A qué grupo pertenece el punto morado?", fontsize=16)
        ax.set_xlabel("Característica X", fontsize=12)
        ax.set_ylabel("Característica Y", fontsize=12)
        ax.legend(fontsize=10)
        st.pyplot(fig)

        if st.session_state.ensemble_module_config['game_awaiting_guess_ensemble']:
            user_guess = st.radio(
                "El punto morado parece ser del grupo...",
                ("Clase 0 (Rojo)", "Clase 1 (Azul)"),
                key="ensemble_user_guess"
            )

            if st.button("¡Verificar mi adivinanza!", key="check_ensemble_guess_button"):
                st.session_state.ensemble_module_config['ensemble_game_total_count'] += 1
                predicted_label_int = 0 if user_guess == "Clase 0 (Rojo)" else 1

                if predicted_label_int == st.session_state.ensemble_module_config['current_game_label_ensemble']:
                    st.session_state.ensemble_module_config['ensemble_game_correct_count'] += 1
                    st.success(f"¡Correcto! El punto era de la **Clase {st.session_state.ensemble_module_config['current_game_label_ensemble']}**.")
                else:
                    st.error(f"¡Incorrecto! El punto era de la **Clase {st.session_state.ensemble_module_config['current_game_label_ensemble']}**.")
                
                st.session_state.ensemble_module_config['game_awaiting_guess_ensemble'] = False
                st.session_state.ensemble_module_config['show_ensemble_explanation'] = True
                st.markdown(f"**Resultado actual del juego: {st.session_state.ensemble_module_config['ensemble_game_correct_count']} aciertos de {st.session_state.ensemble_module_config['ensemble_game_total_count']} intentos.**")
                st.button("¡Siguiente punto!", key="next_ensemble_point_button", on_click=generate_new_game_point_ensemble)
                st.rerun()
        else:
            st.write("Haz clic en '¡Siguiente punto!' para una nueva ronda.")
            if st.button("¡Siguiente punto!", key="next_ensemble_point_after_reveal", on_click=generate_new_game_point_ensemble):
                st.rerun()

# --- Nueva Sección: ¿Por qué los Métodos Ensemble? (Explicación Post-Juego) ---
if st.session_state.ensemble_module_config['show_ensemble_explanation']:
    st.write("---")
    st.header("¿Por qué los Métodos Ensemble son tan potentes?")
    st.markdown("""
    En el juego, te habrás dado cuenta de que clasificar puntos cerca de la "frontera" puede ser complicado.
    Aquí es donde la inteligencia de un equipo, como la de un **Random Forest** o **Boosting**, brilla.

    * **Diversidad para la Robustez:** Cada "experto" individual (árbol de decisión) en un **Random Forest** puede tener sus propias fortalezas y debilidades. Al combinar muchos de ellos, los errores de un árbol son compensados por los aciertos de otros, lo que lleva a una predicción más robusta. ¡Es como si diferentes detectives vieran pistas distintas y luego juntaran todas para resolver el misterio!
    * **Corrección de Errores con Boosting:** En **Boosting**, los "estudiantes" aprenden de los errores de sus predecesores. Esto significa que el modelo final es muy bueno en las áreas donde los modelos simples fallarían. ¡Es como tener un equipo que aprende y se perfecciona continuamente!
    * **Reducción del Sobreajuste (Overfitting):** Un solo modelo (como un árbol de decisión muy profundo) puede aprenderse los datos de entrenamiento "de memoria" y luego fallar con datos nuevos que nunca ha visto (sobreajuste). Los métodos Ensemble, al promediar o combinar inteligentemente, son mucho menos propensos a este problema, generalizando mejor a datos no vistos.

    En resumen, los métodos **Ensemble** son como tener un **super-modelo** construido a partir de la sabiduría colectiva de muchos modelos más simples. ¡Son una de las herramientas más poderosas en el arsenal de Machine Learning!
    """)
    st.write("---")

st.write("---")

# --- Sección de Chatbot de Juego con Bosco el Entrenador de Equipos ---
st.header("¡Juega y Aprende con Bosco el Entrenador de Equipos sobre Métodos Ensemble!")
st.markdown("¡Hola! Soy Bosco, tu entrenador personal para formar los mejores equipos de Machine Learning. ¿Listo para entrenar?")

if client:
    # System prompt para el juego de preguntas de Bosco el Entrenador de Equipos
    ensemble_game_system_prompt = f"""
    Eres un **experto consumado en Machine Learning Avanzado y Modelado de Conjuntos (Ensemble Methods)**, con una especialización profunda en **Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost)** y otras técnicas de ensemble. Comprendes a fondo sus fundamentos teóricos, cómo combinan múltiples modelos para mejorar el rendimiento, sus fortalezas, debilidades y aplicaciones prácticas en clasificación y regresión. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de los Métodos Ensemble mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Predicción consolidada! Tu ensemble de conocimientos ha rendido frutos." o "Esa combinación no fue la óptima. Repasemos la estrategia de votación."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "Los métodos ensemble combinan las predicciones de varios modelos base para obtener un rendimiento superior al de un solo modelo..."]
    [Pregunta para continuar, ej: "¿Listo para afinar tus ensambles?" o "¿Quieres explorar las diferencias entre bagging y boosting con más detalle?"]

    **Reglas adicionales para el Experto en Métodos Ensemble:**
    * **Enfoque Riguroso en Métodos Ensemble:** Todas tus preguntas y explicaciones deben girar en torno a los Métodos Ensemble. Cubre sus fundamentos (sabiduría de las multitudes), principales categorías (Bagging, Boosting, Stacking), el funcionamiento detallado de **Random Forest** (construcción de árboles, aleatoriedad, votación/promedio), el funcionamiento de **Boosting** (modelos débiles secuenciales, corrección de errores, Gradient Boosting, XGBoost, LightGBM, CatBoost), comparación entre Bagging y Boosting, métricas de evaluación, sobreajuste, interpretabilidad (importancia de características) y aplicaciones.
    * **¡VARIEDAD, VARIADA!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de Métodos Ensemble que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General:** ¿Qué son los métodos ensemble? ¿Por qué funcionan? (reducción de varianza, sesgo, mejora de la robustez).
        * **Tipos de Métodos Ensemble:**
            * **Bagging (Bootstrap Aggregating):** Concepto, cómo reduce la varianza.
            * **Random Forest:**
                * Funcionamiento: Múltiples árboles de decisión.
                * Aleatoriedad: Submuestreo de datos (bootstrap) y submuestreo de características.
                * Predicción: Votación (clasificación) o promedio (regresión).
                * Importancia de características.
            * **Boosting:**
                * Concepto: Modelos secuenciales que corrigen errores del anterior.
                * AdaBoost (conceptual).
                * **Gradient Boosting Machines (GBM):** Función de pérdida, gradientes.
                * **XGBoost, LightGBM, CatBoost:** Diferencias clave (optimización, manejo de categóricas, paralelización).
            * **Stacking:** Concepto de meta-modelo.
        * **Comparación Bagging vs. Boosting:** Diferencias fundamentales en cómo combinan modelos, objetivos (reducir varianza vs. sesgo).
        * **Ventajas de los Métodos Ensemble:** Alto rendimiento, robustez, manejo de overfitting (Random Forest).
        * **Desventajas/Desafíos:** Mayor complejidad computacional, menor interpretabilidad (especialmente Boosting), sobreajuste (en Boosting si no se controla).
        * **Optimización de Hiperparámetros:** `n_estimators`, `max_depth`, `learning_rate` (para Boosting), `subsample`, `colsample_bytree`.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.ensemble_chatbot_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Aprendiz de Colaboración – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de que "varios cerebros piensan mejor que uno" y ejemplos simples de decisiones tomadas en grupo.
            * *Tono:* "Estás descubriendo el poder de la toma de decisiones en equipo."
        * **Nivel 2 (Analista de Conjuntos – 3-5 respuestas correctas):** Tono más técnico. Introduce los conceptos de **Bagging** y **Boosting** de forma intuitiva. Preguntas sobre la diferencia fundamental entre Random Forest y una técnica de Boosting.
            * *Tono:* "Tu comprensión de cómo combinar modelos para obtener mejores resultados es prometedora."
        * **Nivel 3 (Ingeniero de Ensembles – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en los mecanismos específicos de **Random Forest** (aleatoriedad en datos y características) y **Gradient Boosting** (aprendizaje secuencial de residuos). Preguntas sobre la importancia de características o la optimización de hiperparámetros básicos.
            * *Tono:* "Tu habilidad para implementar y ajustar modelos ensemble es crucial para el rendimiento de vanguardia en Machine Learning."
        * **Nivel Maestro (Científico de Datos de Alto Rendimiento – 9+ respuestas correctas):** Tono de **especialista en la optimización y despliegue de modelos de alto rendimiento**. Preguntas sobre las diferencias matizadas entre XGBoost, LightGBM y CatBoost, la interpretabilidad de los métodos ensemble complejos, estrategias avanzadas de ajuste de hiperparámetros, o el manejo de escenarios donde un método es superior al otro. Se esperan respuestas que demuestren una comprensión teórica y práctica robusta, incluyendo sus limitaciones y cómo explotar al máximo su potencial.
            * *Tono:* "Tu maestría en los métodos ensemble te permite superar los límites del rendimiento predictivo y liderar la creación de soluciones de ML robustas."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Pedir a varios amigos su opinión antes de tomar una decisión importante.
        * **Nivel 2:** Predecir el riesgo de impago de un préstamo utilizando un Random Forest, o mejorar la precisión de un clasificador de spam con un modelo de Boosting.
        * **Nivel 3:** Implementar un modelo de Gradient Boosting para predecir el precio de la vivienda con alta precisión, utilizando características de ingeniería y ajustando los parámetros de aprendizaje.
        * **Nivel Maestro:** Optimizar un modelo XGBoost para un concurso de Kaggle con un dataset ruidoso y desbalanceado, explorando técnicas de regularización y validación cruzada avanzada para maximizar la generalización.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre MÉTODOS ENSEMBLE (RANDOM FOREST, BOOSTING), y asegúrate de que no se parezca a las anteriores.
    """

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_ensemble_question_response(raw_text):
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
            print(f"DEBUG: RAW API QUESTION RESPONSE (UNEXPECTED FORMAT):\n{raw_text}") # Added for console debugging
            return None, {}, ""
        return question, options, correct_answer_key

    # Función para parsear la respuesta de feedback de la IA
    def parse_ensemble_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        print(f"DEBUG: RAW API FEEDBACK RESPONSE (UNEXPECTED FORMAT):\n{raw_text}") # Added for console debugging
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"
        
    # --- Funciones para subir de nivel directamente ---
    def set_ensemble_level(target_streak, level_name):
        st.session_state.ensemble_chatbot_correct_streak = target_streak
        st.session_state.ensemble_chatbot_game_active = True
        st.session_state.ensemble_chatbot_game_messages = []
        st.session_state.ensemble_chatbot_current_question = None
        st.session_state.ensemble_chatbot_current_options = {}
        st.session_state.ensemble_chatbot_correct_answer = None
        st.session_state.ensemble_chatbot_game_needs_new_question = True
        st.session_state.ensemble_chatbot_awaiting_next_game_decision = False
        st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}**! Prepárate para preguntas más desafiantes. ¡Aquí va tu primera!"})
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons_ensemble, col_level_up_buttons_ensemble = st.columns([1, 2])

    with col_game_buttons_ensemble:
        if st.button("¡Vamos a entrenar con Bosco el Entrenador!", key="start_bosco_game_button"):
            st.session_state.ensemble_chatbot_game_active = True
            st.session_state.ensemble_chatbot_game_messages = []
            st.session_state.ensemble_chatbot_current_question = None
            st.session_state.ensemble_chatbot_current_options = {}
            st.session_state.ensemble_chatbot_correct_answer = None
            st.session_state.ensemble_chatbot_game_needs_new_question = True
            st.session_state.ensemble_chatbot_awaiting_next_game_decision = False
            st.session_state.ensemble_chatbot_correct_streak = 0
            st.session_state.ensemble_chatbot_last_played_question = None # Resetear audio
            st.rerun()
        
    with col_level_up_buttons_ensemble:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un entrenador experto? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_ensemble, col_lvl2_ensemble, col_lvl3_ensemble = st.columns(3)
        with col_lvl1_ensemble:
            if st.button("Subir a Nivel Medio (Ensemble)", key="level_up_medium_ensemble"):
                set_ensemble_level(3, "Medio")
        with col_lvl2_ensemble:
            if st.button("Subir a Nivel Avanzado (Ensemble)", key="level_up_advanced_ensemble"):
                set_ensemble_level(6, "Avanzado")
        with col_lvl3_ensemble:
            if st.button("👑 ¡Maestro Entrenador! (Ensemble)", key="level_up_champion_ensemble"):
                set_ensemble_level(9, "Campeón")

    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.ensemble_chatbot_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica del juego del chatbot si está activo
    if st.session_state.ensemble_chatbot_game_active:
        if st.session_state.ensemble_chatbot_current_question is None and st.session_state.ensemble_chatbot_game_needs_new_question and not st.session_state.ensemble_chatbot_awaiting_next_game_decision:
            with st.spinner("Bosco está preparando una pregunta de entrenamiento..."):
                try:
                    ensemble_game_messages_for_api = [{"role": "system", "content": ensemble_game_system_prompt}]
                    # Historial de mensajes para el contexto del LLM
                    for msg in st.session_state.ensemble_chatbot_game_messages[-6:]: # Limitar historial para eficiencia
                        if msg["role"] == "assistant" and msg["content"].startswith("**"):
                            ensemble_game_messages_for_api.append({"role": "assistant", "content": f"PREGUNTA ANTERIOR: {msg['content'].splitlines()[0].replace('**Pregunta de Bosco:** ', '')}"})
                        elif msg["role"] == "user" and "MI RESPUESTA:" not in msg["content"]:
                            ensemble_game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {msg['content']}"})

                    ensemble_game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre MÉTODOS ENSEMBLE (Random Forest, Boosting). **Responde EXCLUSIVAMENTE con el formato exacto: Pregunta: [Tu pregunta aquí]\nA) [Opción A]\nB) [Opción B]\nC) [Opción C]\nRespuestaCorrecta: [A, B o C]**"})

                    ensemble_response = client.chat.completions.create(
                        model="gpt-4o-mini", # Considera usar gpt-4 si tienes acceso para mejor razonamiento
                        messages=ensemble_game_messages_for_api,
                        temperature=0.5, # Reduced temperature for more deterministic output
                        max_tokens=250
                    )
                    raw_ensemble_question_text = ensemble_response.choices[0].message.content
                    question, options, correct_answer_key = parse_ensemble_question_response(raw_ensemble_question_text)

                    if question:
                        st.session_state.ensemble_chatbot_current_question = question
                        st.session_state.ensemble_chatbot_current_options = options
                        st.session_state.ensemble_chatbot_correct_answer = correct_answer_key
                        st.session_state.ensemble_chatbot_game_needs_new_question = False
                        
                        question_content = f"**Nivel {int(st.session_state.ensemble_chatbot_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.ensemble_chatbot_correct_streak}**\n\n**Pregunta de Bosco:** {question}\n\n"
                        for k, v in options.items():
                            question_content += f"**{k})** {v}\n"
                        
                        st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": question_content})
                        st.rerun()
                    else:
                        st.error("Bosco no pudo generar una pregunta válida. Intenta de nuevo.")
                        st.session_state.ensemble_chatbot_game_active = False # Detener el juego para evitar bucle de errores
                        st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": "Bosco no pudo generar una pregunta válida. Parece que hay un problema. Por favor, reinicia el juego."})

                except Exception as e:
                    st.error(f"Error al comunicarse con la API de OpenAI para la pregunta: {e}")
                    st.session_state.ensemble_chatbot_game_active = False
                    st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": "Lo siento, tengo un problema para conectar con mi cerebro (la API). ¡Por favor, reinicia el juego!"})
                    st.rerun()

        # Mostrar la pregunta actual y el formulario de respuesta si existe una pregunta
        if st.session_state.ensemble_chatbot_current_question and not st.session_state.ensemble_chatbot_awaiting_next_game_decision:
            # Audio de la pregunta (si se acaba de generar)
            if st.session_state.ensemble_chatbot_last_played_question != st.session_state.ensemble_chatbot_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.ensemble_chatbot_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.ensemble_chatbot_correct_streak}. Pregunta de Bosco: {st.session_state.ensemble_chatbot_current_question}. Opción A: {st.session_state.ensemble_chatbot_current_options.get('A', '')}. Opción B: {st.session_state.ensemble_chatbot_current_options.get('B', '')}. Opción C: {st.session_state.ensemble_chatbot_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.audio(fp, format='audio/mp3', start_time=0)
                    st.session_state.ensemble_chatbot_last_played_question = st.session_state.ensemble_chatbot_current_question
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")

            # Envuelve el radio button y el submit button en un st.form
            with st.form(key="ensemble_game_form"):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_answer = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.ensemble_chatbot_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.ensemble_chatbot_current_options[x]}",
                        key="ensemble_answer_radio",
                        label_visibility="collapsed"
                    )
                submit_button = st.form_submit_button("¡Enviar Respuesta!")

            if submit_button:
                st.session_state.ensemble_chatbot_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_answer}) {st.session_state.ensemble_chatbot_current_options[user_answer]}"})
                prev_streak = st.session_state.ensemble_chatbot_correct_streak
                is_correct = (user_answer == st.session_state.ensemble_chatbot_correct_answer)

                if is_correct:
                    st.session_state.ensemble_chatbot_correct_streak += 1
                else:
                    st.session_state.ensemble_chatbot_correct_streak = 0

                radio_placeholder.empty()

                # --- Lógica de subida de nivel y confeti ---
                if st.session_state.ensemble_chatbot_correct_streak > 0 and \
                   st.session_state.ensemble_chatbot_correct_streak % 3 == 0 and \
                   st.session_state.ensemble_chatbot_correct_streak > prev_streak:
                    
                    if st.session_state.ensemble_chatbot_correct_streak < 9: # Niveles Básico, Medio, Avanzado
                        current_level_text = ""
                        if st.session_state.ensemble_chatbot_correct_streak == 3:
                            current_level_text = "Medio (como un entrenador que ya domina los conceptos básicos)"
                        elif st.session_state.ensemble_chatbot_correct_streak == 6:
                            current_level_text = "Avanzado (como un excelente estratega de equipos de ML)"
                        
                        level_up_message = f"¡Increíble! ¡Has respondido {st.session_state.ensemble_chatbot_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de Métodos Ensemble. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a entrenador/a de Machine Learning! 🚀"
                        st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": level_up_message})
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
                    elif st.session_state.ensemble_chatbot_correct_streak >= 9: # Nivel Campeón o superior
                        medals_earned = (st.session_state.ensemble_chatbot_correct_streak - 6) // 3
                        medal_message = f"🏅 ¡FELICITACIONES, MAESTRO/A ENTRENADOR/A! ¡Has ganado tu {medals_earned}ª Medalla de Entrenamiento de Equipos! ¡Tu habilidad es asombrosa y digna de un verdadero EXPERTO en Métodos Ensemble! ¡Sigue así! 🌟"
                        st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": medal_message})
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
                        
                        if prev_streak < 9: # Solo si acaba de entrar en el nivel campeón
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Campeón (Maestro Entrenador de Equipos de ML)**! ¡Las preguntas ahora son solo para los verdaderos genios de la combinatoria de modelos! ¡Adelante!"
                            st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                audio_fp_level_up_champion = io.BytesIO()
                                tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                                audio_fp_level_up_champion.seek(0)
                                st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0)
                                time.sleep(2)
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón: {e}")

                # Generar feedback de Bosco
                with st.spinner("Bosco está pensando su respuesta..."):
                    try:
                        feedback_prompt = f"""
                        El usuario respondió '{user_answer}'. La pregunta era: '{st.session_state.ensemble_chatbot_current_question}'.
                        La respuesta correcta era '{st.session_state.ensemble_chatbot_correct_answer}'.
                        Da feedback como Bosco el Entrenador de Equipos.
                        Si es CORRECTO, el mensaje es "¡Equipo perfectamente coordinado!" o similar.
                        Si es INCORRECTO, el mensaje es "¡Necesitamos más entrenamiento!" o similar.
                        Luego, una explicación sencilla para el usuario.
                        Finalmente, pregunta: "¿Quieres seguir entrenando a tu equipo?".
                        **Responde EXCLUSIVAMENTE con el formato estricto de feedback:
                        [Mensaje de Correcto/Incorrecto]
                        [Breve explicación del concepto]
                        [Pregunta para continuar]**
                        """
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": ensemble_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.5, # Reduced temperature for more deterministic output
                            max_tokens=300
                        )
                        raw_ensemble_feedback_text = feedback_response.choices[0].message.content
                        feedback_message, explanation_message, continue_question = parse_ensemble_feedback_response(raw_ensemble_feedback_text)
                        
                        st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": feedback_message})
                        st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": explanation_message})
                        st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": continue_question})

                        try:
                            tts = gTTS(text=f"{feedback_message}. {explanation_message}. {continue_question}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp) 
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")

                        st.session_state.ensemble_chatbot_current_question = None # Limpiar pregunta actual
                        st.session_state.ensemble_chatbot_current_options = {}
                        st.session_state.ensemble_chatbot_correct_answer = None
                        st.session_state.ensemble_chatbot_game_needs_new_question = False
                        st.session_state.ensemble_chatbot_awaiting_next_game_decision = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error al comunicarse con la API de OpenAI para el feedback: {e}")
                        st.session_state.ensemble_chatbot_game_active = False
                        st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": "Lo siento, no puedo darte feedback ahora mismo. ¡Por favor, reinicia el juego!"})
                        st.rerun()

        # Botones para continuar o terminar el juego
        if st.session_state.ensemble_chatbot_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col_continue, col_end = st.columns(2)
            with col_continue:
                if st.button("👍 Sí, ¡quiero seguir entrenando!", key="continue_ensemble_game"):
                    st.session_state.ensemble_chatbot_awaiting_next_game_decision = False
                    st.session_state.ensemble_chatbot_game_needs_new_question = True
                    st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío!"})
                    st.rerun()
            with col_end:
                if st.button("👎 No, ¡quiero descansar del entrenamiento!", key="end_ensemble_game"):
                    st.session_state.ensemble_chatbot_game_active = False
                    st.session_state.ensemble_chatbot_awaiting_next_game_decision = False
                    st.session_state.ensemble_chatbot_game_messages.append({"role": "assistant", "content": "¡Gracias por entrenar con Bosco! ¡Vuelve pronto para seguir formando equipos invencibles de Machine Learning!"})
                    st.rerun()

else:
    st.info("El chatbot Bosco el Entrenador no está disponible porque la clave de la API de OpenAI no está configurada.")

st.write("---")