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

tf = None
OpenAI = None
client = None
openai_api_key_value = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, UpSampling2D, LeakyReLU, Dropout
except ImportError:
    st.error("Las librerías 'tensorflow' y 'keras' no están instaladas. Por favor, instálalas usando: pip install tensorflow")
    tf = None

try:
    from openai import OpenAI
except ImportError:
    st.error("La librería 'openai' no está instalada. Por favor, instálala usando: pip install pip install openai")
    OpenAI = None # Asegurarse de que OpenAI sea None si no se puede importar


# --- Configuración de la página ---
st.set_page_config(
    page_title="Laboratorio de Redes Generativas Antagónicas (GANs)",
    layout="wide"
)

# --- Rutas de los modelos GAN reales (¡Asegúrate de que coincidan con donde los guardaste!) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_project_dir = os.path.dirname(current_dir)
MODEL_DIR = os.path.join(base_project_dir, 'assets', 'models')
GALLERY_REAL_IMAGES_DIR = os.path.join(base_project_dir, 'datasets', 'piece_defects', 'test')

GENERATOR_PATH = os.path.join(MODEL_DIR, 'fashion_mnist_generator.h5')
DISCRIMINATOR_PATH = os.path.join(MODEL_DIR, 'fashion_mnist_discriminator.h5')


current_dir = os.path.dirname(os.path.abspath(__file__))
base_project_dir = os.path.dirname(current_dir)
MODEL_DIR = os.path.join(base_project_dir, 'assets', 'models')
GALLERY_REAL_IMAGES_DIR = os.path.join(base_project_dir, 'datasets', 'piece_defects', 'test')

GENERATOR_PATH = os.path.join(MODEL_DIR, 'fashion_mnist_generator.h5')
DISCRIMINATOR_PATH = os.path.join(MODEL_DIR, 'fashion_mnist_discriminator.h5')

# --- DEBUGGING DE RUTAS: AÑADE ESTO ---
st.sidebar.subheader("Verificación de Rutas")
st.sidebar.write(f"current_dir: `{current_dir}`")
st.sidebar.write(f"base_project_dir: `{base_project_dir}`")
st.sidebar.write(f"MODEL_DIR: `{MODEL_DIR}`")
st.sidebar.write(f"GALLERY_REAL_IMAGES_DIR: `{GALLERY_REAL_IMAGES_DIR}`")

# Verificar existencia de directorios clave
st.sidebar.write(f"¿Existe MODEL_DIR?: `{os.path.exists(MODEL_DIR)}`")
st.sidebar.write(f"¿Existe GALLERY_REAL_IMAGES_DIR?: `{os.path.exists(GALLERY_REAL_IMAGES_DIR)}`")

good_path_check = os.path.join(GALLERY_REAL_IMAGES_DIR, 'good')
defect_path_check = os.path.join(GALLERY_REAL_IMAGES_DIR, 'defect')

st.sidebar.write(f"Ruta 'good': `{good_path_check}`")
st.sidebar.write(f"¿Existe 'good'?: `{os.path.exists(good_path_check)}`")
if os.path.exists(good_path_check):
    st.sidebar.write(f"Archivos en 'good': {os.listdir(good_path_check)[:5]}...") # Mostrar solo los primeros 5

st.sidebar.write(f"Ruta 'defect': `{defect_path_check}`")
st.sidebar.write(f"¿Existe 'defect'?: `{os.path.exists(defect_path_check)}`")
if os.path.exists(defect_path_check):
    st.sidebar.write(f"Archivos en 'defect': {os.listdir(defect_path_check)[:5]}...") # Mostrar solo los primeros 5
# --- FIN DEBUGGING DE RUTAS ---

# ==============================================================================
# Define la subclase FashionGAN para que Keras pueda cargar el modelo correctamente
# Es CRÍTICO que esta definición sea idéntica a la usada al guardar el modelo
# ==============================================================================
class FashionGAN(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss):
        super().compile()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        return {"d_loss": 0.0, "g_loss": 0.0}


# --- Carga de Modelos GAN reales ---
generator_model = None
discriminator_model = None
latent_dim = 128 # Dimensión del vector de ruido, debe coincidir con el entrenamiento

if tf is not None:
    try:
        # Cargar el generador
        generator_model = load_model(GENERATOR_PATH, compile=False)
        st.success("¡Generador de Fashion MNIST cargado con éxito!")

        # Cargar el discriminador
        discriminator_model = load_model(DISCRIMINATOR_PATH, compile=False)
        st.success("¡Discriminador de Fashion MNIST cargado con éxito!")

        generator_model.trainable = False
        discriminator_model.trainable = False

    except Exception as e:
        st.error(f"Error al cargar los modelos GAN. Asegúrate de que los archivos '{os.path.basename(GENERATOR_PATH)}' y '{os.path.basename(DISCRIMINATOR_PATH)}' existen en la ruta '{MODEL_DIR}'. Error: {e}")
        st.warning("Se usará una simulación de GAN en su lugar. Por favor, entrena y guarda tus modelos primero.")
        generator_model = None
        discriminator_model = None


# --- Inicialización de session_state ---
if 'gan_lab_config' not in st.session_state:
    st.session_state.gan_lab_config = {
        'generator_layers': "Real (basado en h5)",
        'discriminator_layers': "Real (basado en h5)",
        'noise_dim': latent_dim,
        'training_epochs': "Variable (ver tu script)",
        'generator_loss': 0.0,
        'discriminator_loss': 0.0,
        'image_quality_progress': 0.0,
        'generated_image': None,
        'discriminator_prediction': None
    }
if 'last_generated_noise_input' not in st.session_state:
    st.session_state['last_generated_noise_input'] = None

# --- Estado para el juego de preguntas de Ganesh ---
if "gan_game_active" not in st.session_state:
    st.session_state.gan_game_active = False
if "gan_game_messages" not in st.session_state:
    st.session_state.gan_game_messages = []
if "gan_current_question" not in st.session_state:
    st.session_state.gan_current_question = None
if "gan_current_options" not in st.session_state:
    st.session_state.gan_current_options = {}
if "gan_correct_answer" not in st.session_state:
    st.session_state.gan_correct_answer = None
if "gan_awaiting_next_game_decision" not in st.session_state:
    st.session_state.gan_awaiting_next_game_decision = False
if "gan_game_needs_new_question" not in st.session_state:
    st.session_state.gan_game_needs_new_question = False
if "gan_correct_streak" not in st.session_state:
    st.session_state.gan_correct_streak = 0


# --- Estado para el Desafío de la Galería ---
if 'gallery_images' not in st.session_state:
    st.session_state['gallery_images'] = [] # Lista de (PIL_Image, is_real, probability)
if 'gallery_revealed' not in st.session_state:
    st.session_state['gallery_revealed'] = False
if 'selected_gallery_image_index' not in st.session_state:
    st.session_state['selected_gallery_image_index'] = None
if 'gallery_explanation_given' not in st.session_state:
    st.session_state['gallery_explanation_given'] = False
if 'gallery_feedback_given' not in st.session_state:
    st.session_state['gallery_feedback_given'] = False
if 'real_fashion_mnist_images' not in st.session_state:
    st.session_state['real_fashion_mnist_images'] = []
if 'gallery_initialized' not in st.session_state:
    st.session_state['gallery_initialized'] = False

st.title("Laboratorio Interactivo de Redes Generativas Antagónicas (GANs) con Fashion MNIST")

# AQUI SE INSERTA LA EXPLICACIÓN PARA NIÑOS
st.markdown("""
¡Bienvenido al laboratorio donde dos redes neuronales, el **Generador** y el **Discriminador**, compiten para crear y detectar falsificaciones!
Estamos utilizando un modelo GAN real entrenado en el conjunto de datos **Fashion MNIST** (imágenes de ropa y accesorios).

---

### ¿Cómo funcionan las Redes GANs? ¡Es como un juego!

Imagina que hay dos cerebros artificiales:

* **El Generador**: Intenta dibujar imágenes de ropa (camisetas, zapatos, etc.) que se vean reales.

* **El Discriminador**: Mira las imágenes y dice si son reales (de verdad del dataset) o falsas (dibujadas por el generador).

Ellos juegan un juego:

* **El generador trata de engañar al discriminador**.

* **El discriminador trata de descubrir cuál es falsa**.

Con el tiempo, el generador se vuelve tan bueno que ¡hace imágenes que parecen reales!

#### **¿Cómo compiten y aprenden? Es como un juego de "pilla-pilla" o "policías y ladrones":**

* **Ronda del Detective**: El Detective (Discriminador) mira un montón de dibujos. Algunos son **reales** (fotos de ropa de verdad) y otros son los **falsos** que hizo el Falsificador (Generador). El Detective intenta adivinar cuáles son cuáles. Si adivina bien, ¡se vuelve más listo!

* **Ronda del Falsificador**: Ahora es el turno del Falsificador (Generador). Él mira cómo le fue al Detective. Si el Detective fue muy bueno detectando sus dibujos falsos, el Falsificador piensa: "¡Vaya! Necesito hacer mis dibujos mucho, mucho más creíbles para engañarlo." Así que se esfuerza en dibujar ropa aún mejor.

#### **El secreto de las GANs es que se ayudan a mejorar mutuamente:**

* El Falsificador (Generador) **mejora creando ropa cada vez más realista** porque sabe que el Detective (Discriminador) se está volviendo mejor.
* El Detective (Discriminador) **mejora detectando las falsificaciones** porque el Falsificador le da dibujos cada vez más difíciles de distinguir.

Al final, si compiten lo suficiente, ¡el Falsificador (Generador) puede llegar a hacer dibujos de ropa que son **casi imposibles de distinguir de la ropa real**! Y eso es lo que hace que las GANs sean tan geniales para crear cosas nuevas y que parecen de verdad, ¡como diseños de ropa fantásticos!
""")

st.write("---")

# =================================================================================================
# --- SECCIÓN: CONOCE A NUESTROS AMIGOS GAN (¡Antes "Configura tu Par de Redes GAN"!) ---
# =================================================================================================
st.markdown("---")
st.header("¡Conoce a nuestros Amigos GAN! 🎨🕵️‍♂️")
st.markdown("""
¡Aquí te presentamos a los dos cerebritos que hacen magia con las imágenes!
Son como superhéroes de la Inteligencia Artificial que trabajan juntos para crear cosas nuevas y sorprendentes.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("El Artista Mágico (Generador) 🎨")
    st.markdown("""
    Este amigo es un pintor muy especial. Su trabajo es **crear nuevas imágenes de ropa** ¡como si las soñara!
    Empieza con una pequeña "idea secreta" (que llamamos ruido) y la transforma en una prenda de vestir.
    """)
    # Puedes añadir un pequeño expander para los curiosos
    with st.expander("¿Cómo es su cerebro de Artista?"):
        st.markdown(f"""
        El cerebro de nuestro Artista Mágico es muy complejo, como un laberinto de ideas.
        Está hecho de muchas "capas" o "pasos" que aprendió durante su entrenamiento.
        Tiene la habilidad de convertir una "chispita creativa" en una imagen real.
        """)
        # Si quieres mostrar un detalle técnico muy simplificado, aquí iría.
        # Por ejemplo: st.write(f"Tiene muchas 'capas' mágicas: {st.session_state.gan_lab_config['generator_layers']}")
        st.write(f"Su chispa creativa inicial es de {st.session_state.gan_lab_config['noise_dim']} 'secretos'.") # Usar latent_dim de st.session_state

with col2:
    st.subheader("El Detective Astuto (Discriminador) 🕵️‍♂️")
    st.markdown("""
    ¡Este es el ojo más agudo del equipo! Su misión es **descubrir si una imagen es real** (una foto de verdad)
    **o si fue creada por nuestro Artista Mágico**. Es como un experto en adivinar trucos.
    """)
    with st.expander("¿Cómo funciona su ojo de Detective?"):
        st.markdown("""
        El Detective Astuto también tiene un "cerebro" lleno de trucos que aprendió.
        Examina cada detalle de la imagen para ver si parece "auténtica" o si tiene algún "fallito" de falsificación.
        """)
        # st.write(f"Sus 'capas' de detective: {st.session_state.gan_lab_config['discriminator_layers']}")

with col3:
    st.subheader("La Gran Competencia (Entrenamiento) 🏆")
    st.markdown("""
    El Artista Mágico y el Detective Astuto no siempre fueron tan buenos. ¡Aprendieron jugando!
    El Artista intentaba engañar al Detective, y el Detective intentaba no dejarse engañar.
    Cada vez que jugaban, ¡se hacían un poquito mejores!
    """)
    st.info(f"""
    ¡Nuestros amigos han jugado **miles de rondas** de este juego! 🎮
    Así es como el Artista aprende a hacer imágenes más reales y el Detective aprende a ser un mejor descubridor de falsificaciones.
    """)

st.markdown("---") # Separador para la siguiente sección


# --- Sección de Simulación de Entrenamiento GAN (Ahora solo información) ---
st.header("Entrena tus GANs y ve su progreso (¡Modelo Real Pre-entrenado!)")
st.markdown("""
Dado que estamos usando un modelo GAN pre-entrenado de Fashion MNIST, no necesitas entrenarlo aquí en tiempo real.
¡Ya ha pasado por miles de rondas de competición!
""")

st.info("El entrenamiento ya se ha realizado previamente. Ahora puedes ir directamente a generar imágenes.")

st.write("---")

# --- Sección de Visualización de GAN y Métricas ---
col_gan_viz, col_gan_metrics = st.columns([2, 1])

with col_gan_viz:
    st.markdown("### ¡Visualización de tus GANs y Métricas!")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generador
    ax.add_patch(plt.Rectangle((0.1, 0.4), 0.3, 0.2, color='purple', ec='black', alpha=0.7))
    ax.text(0.25, 0.5, 'Generador\n(Modelo Real)', ha='center', va='center', color='white', fontsize=12)
    ax.text(0.25, 0.35, f'Entrada: Ruido ({latent_dim}D)\nSalida: Imagen Falsa (28x28)', ha='center', va='top', fontsize=10)

    # Conexión Generador -> Discriminador
    ax.plot([0.4, 0.6], [0.5, 0.5], 'k-', alpha=0.5, linewidth=2)
    ax.text(0.5, 0.52, 'Imágenes Falsas', ha='center', va='bottom', fontsize=9)

    # Discriminador
    ax.add_patch(plt.Rectangle((0.6, 0.4), 0.3, 0.2, color='darkgreen', ec='black', alpha=0.7))
    ax.text(0.75, 0.5, 'Discriminador\n(Modelo Real)', ha='center', va='center', color='white', fontsize=12)
    ax.text(0.75, 0.35, 'Entrada: Imagen (28x28)\nSalida: ¿Real o Falsa?', ha='center', va='top', fontsize=10)

    # Flujo de Imágenes Reales al Discriminador
    ax.plot([0.7, 0.7], [0.7, 0.6], 'k--', alpha=0.5, linewidth=2)
    ax.text(0.7, 0.72, 'Imágenes Reales\n(Fashion MNIST)', ha='center', va='bottom', fontsize=9, color='blue')

    # Resultado del Discriminador
    ax.text(0.75, 0.25, 'Decisión del Detective:\n(Probabilidad de ser Real)', ha='center', va='top', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"Arquitectura del Par de Redes Generativas Antagónicas (GANs)", fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

with col_gan_metrics:
    st.markdown("---")
    st.markdown("### Rendimiento de las GANs (¡Modelo Real!):")
    st.metric(label="Calidad de Imagen Generada (Estimada)", value=f"{st.session_state.gan_lab_config['image_quality_progress']*100:.1f}%",
              help="Estimación de qué tan realistas son las imágenes que crea el Generador, basada en la predicción del Discriminador. ¡Más alto es mejor!")

    st.markdown("*(Las pérdidas exactas del Generador y Discriminador son el resultado del entrenamiento previo. Aquí solo se muestra una estimación del realismo de la imagen generada.)*")

    st.markdown("""
    **Observa:**
    * El **Generador** crea imágenes de ropa que intentan parecer reales.
    * El **Discriminador** evalúa si una imagen es ropa real de Fashion MNIST o una "falsificación" del Generador.
    * Un buen **equilibrio** en las GANs se da cuando el Generador es tan bueno que el Discriminador apenas puede distinguir entre imágenes reales y falsas.
    """)

st.write("---")

# --- Sección de Generación de Imágenes (con "Estudio de Diseño de Ruido" y "Medidor de Realismo") ---
st.subheader("¡El Generador en Acción! Crea tus propias imágenes de Fashion MNIST")
st.markdown("""
Ahora que nuestras GANs están 'entrenadas', puedes pedirle al **Generador** que cree nuevas imágenes de ropa.
¡Verás cómo el Generador, que antes solo producía ruido, ahora puede crear cosas sorprendentes!
""")

col_generate_image, col_gan_insight = st.columns([1, 1])

with col_generate_image:
    st.markdown("#### Genera una imagen nueva:")

    # --- INICIO del "Estudio de Diseño de Ruido" ---
    st.markdown("#### Estudio de Diseño de Ruido (Control Creativo):")
    st.markdown("¡Cambia estos números para ver cómo el Generador crea diferentes diseños de ropa!")

    # Define cuántos componentes del ruido quieres que el niño manipule directamente
    num_interactive_noise_components = 5 # Puedes ajustar este número
    noise_components = []

    # Crear sliders individuales para los primeros N componentes del ruido
    for i in range(num_interactive_noise_components):
        if f'noise_component_{i}' not in st.session_state:
            st.session_state[f'noise_component_{i}'] = 0.0 # Valor inicial

        st.session_state[f'noise_component_{i}'] = st.slider(
            f"Componente de Ruido {i+1}",
            min_value=-2.0, max_value=2.0, value=st.session_state[f'noise_component_{i}'], step=0.1,
            key=f"noise_slider_{i}"
        )
        noise_components.append(st.session_state[f'noise_component_{i}'])

    if st.button("Reiniciar Componentes", key="reset_noise_components_button"):
        for i in range(num_interactive_noise_components):
            st.session_state[f'noise_component_{i}'] = 0.0
        st.rerun() # Usar st.rerun()

    # --- FIN del "Estudio de Diseño de Ruido" ---


    # Aseguramos que los modelos existan para las operaciones de predicción
    if generator_model and discriminator_model:
        # Generación del vector de ruido completo (con una nueva semilla aleatoria para el "fondo")
        np.random.seed(random.randint(0, 10000))
        noise_input = np.random.normal(0, 1, size=(1, latent_dim))

        # Sobrescribir los primeros componentes con los valores de los sliders
        for i in range(num_interactive_noise_components):
            noise_input[0, i] = noise_components[i]

        # ¡IMPORTANTE! Guardar el noise_input final para la visualización en la otra columna
        st.session_state['last_generated_noise_input'] = noise_input

        # Generar imagen con el modelo real
        generated_images = generator_model.predict(noise_input)
        generated_image_array = generated_images[0, :, :, 0]

        # Escalar a 0-255 y convertir a PIL Image
        generated_image_pil = Image.fromarray((generated_image_array * 255).astype(np.uint8))
        st.session_state.gan_lab_config['generated_image'] = generated_image_pil

        # Pedir al Discriminador real que clasifique la imagen generada
        discriminator_prob = discriminator_model.predict(generated_images)[0][0]
        # ¡IMPORTANTE! Convertir a float nativo de Python para st.progress
        st.session_state.gan_lab_config['discriminator_prediction'] = float(discriminator_prob)

        # Estimar la calidad de la imagen generada
        st.session_state.gan_lab_config['image_quality_progress'] = float(discriminator_prob)

        # El botón solo necesita forzar un rerun para que la lógica de arriba se ejecute de nuevo
        if st.button("¡Generar Nueva Imagen de Moda!", key="generate_new_fashion_image_button_final"): # Clave única
            st.info("El Generador está creando una nueva imagen de moda con la configuración actual del ruido...")
            # Resetear la galería al generar una sola imagen nueva
            st.session_state['gallery_images'] = []
            st.session_state['gallery_revealed'] = False
            st.session_state['selected_gallery_image_index'] = None
            st.session_state['gallery_explanation_given'] = False # Reiniciar la explicación
            st.session_state['gallery_feedback_given'] = False # Resetear feedback de galería
            pass # Continúa la ejecución para actualizar la imagen

    else:
        st.warning("Los modelos GAN no se han cargado correctamente. No se pueden generar imágenes reales.")
        # Mostrar mensaje solo si aún no hay imagen generada (al inicio)
        if st.session_state.gan_lab_config['generated_image'] is None:
            st.info("Genera una imagen para ver cómo el Generador la crea y cómo el Discriminador la evalúa.")


    # --- Mostrar la imagen generada y el "Medidor de Realismo" ---
    if st.session_state.gan_lab_config['generated_image']:
        st.markdown("##### Imagen de Moda Generada por tu GAN:")
        st.image(st.session_state.gan_lab_config['generated_image'], caption="Imagen Generada de Fashion MNIST", use_container_width=True, channels="GRAY")

        if st.session_state.gan_lab_config['discriminator_prediction'] is not None:
            disc_pred_prob = st.session_state.gan_lab_config['discriminator_prediction']

            st.markdown("##### Evaluación del Detective de la Moda:")
            # Medidor de Realismo
            st.progress(disc_pred_prob) # Ya es float nativo

            disc_pred_text = f"Probabilidad de ser REAL: **{disc_pred_prob*100:.1f}%**"
            st.markdown(f"**El Detective dice:** {disc_pred_text}")


            # Zonas "Falsa" y "Real" con mensajes
            # disc_pred_prob es la probabilidad de que sea "Real" (la barra verde)
            # La probabilidad de ser "Falsa" es 1 - disc_pred_prob (la barra roja)

# Caso 1: El Discriminador predice que es MÁS FALSA que Real
if disc_pred_prob < 0.5:
    # Cuanto más baja sea disc_pred_prob, más seguro está de que es falsa.
    if disc_pred_prob <= 0.2: # Muy seguro de que es falsa (ej. 0.0-0.2 para Real, 0.8-1.0 para Falsa)
        st.error(f"¡Es una falsificación CLARÍSIMA! El Detective lo ha descubierto fácilmente. ¡El Generador debe mejorar! (Prob. Real: {disc_pred_prob*100:.1f}%)")
    elif disc_pred_prob <= 0.4: # Moderadamente seguro de que es falsa (ej. 0.2-0.4 para Real, 0.6-0.8 para Falsa)
        st.warning(f"El Detective sospecha fuertemente... ¡es una falsificación! El Generador necesita más práctica. (Prob. Real: {disc_pred_prob*100:.1f}%)")
    else: # Ligeramente más falsa que real (ej. 0.4-0.5 para Real, 0.5-0.6 para Falsa)
        st.info(f"El Detective se inclina por que es una falsificación, aunque por poco. ¡El Generador casi lo consigue! (Prob. Real: {disc_pred_prob*100:.1f}%)")
# Caso 2: El Discriminador predice que es MÁS REAL que Falsa
else: # disc_pred_prob >= 0.5
    # Cuanto más alta sea disc_pred_prob, más seguro está de que es real.
    if disc_pred_prob >= 0.8: # Muy seguro de que es real (ej. 0.8-1.0 para Real, 0.0-0.2 para Falsa)
        st.success(f"¡GUAU! ¡Parece MUY real! ¡El Generador ha hecho un trabajo excelente engañando al Detective! (Prob. Real: {disc_pred_prob*100:.1f}%)")
    elif disc_pred_prob >= 0.6: # Moderadamente seguro de que es real (ej. 0.6-0.8 para Real, 0.2-0.4 para Falsa)
        st.info(f"¡Buena! El Detective cree que es bastante real. ¡El Generador lo está haciendo bien! (Prob. Real: {disc_pred_prob*100:.1f}%)")
    else: # Ligeramente más real que falsa (ej. 0.5-0.6 para Real, 0.4-0.5 para Falsa)
        st.warning(f"El Detective se inclina por que es real, aunque por poco. ¡El Generador lo está haciendo bien! (Prob. Real: {disc_pred_prob*100:.1f}%)")

with col_gan_insight:
    st.markdown("#### ¿Cómo 'piensa' el Generador y el Discriminador?")
    st.info("Estas visualizaciones son conceptuales y reflejan el proceso interno del modelo real.")

    if st.session_state.gan_lab_config['generated_image']:
        st.markdown("##### Proceso de Generación (Ruido a Imagen):")

        # Usar el noise_input guardado en session_state
        if 'last_generated_noise_input' in st.session_state and st.session_state['last_generated_noise_input'] is not None:
            noise_input_viz = st.session_state['last_generated_noise_input']

            fig_noise, ax_noise = plt.subplots(figsize=(6, 1))
            # Muestra los primeros 20 valores, o num_interactive_noise_components si quieres ser más específico
            display_length = min(20, latent_dim) # Mostrar 20 o menos si latent_dim es menor
            sns.heatmap(noise_input_viz[:, :display_length], cmap='magma', cbar=False, ax=ax_noise, yticklabels=False)
            ax_noise.set_title(f"Vector de Ruido Inicial (primeros {display_length} de {latent_dim} valores)")
            ax_noise.axis('off')
            st.pyplot(fig_noise)
            plt.close(fig_noise)
            st.markdown("*(Así es como el Generador recibe sus 'instrucciones' aleatorias)*")
            st.markdown(f"*(Los primeros {num_interactive_noise_components} valores son controlados por los sliders)*")
        else:
            st.info("Genera una imagen para ver la visualización del ruido.")

        st.markdown("##### Decisión del Discriminador:")
        disc_prob = st.session_state.gan_lab_config['discriminator_prediction']
        if disc_prob is not None:
            fig_disc_pred, ax_disc_pred = plt.subplots(figsize=(6, 3))
            categories = ['Falsa', 'Real']
            probabilities = [1 - disc_prob, disc_prob]

            bars = ax_disc_pred.bar(categories, probabilities, color=['lightcoral', 'lightgreen'])
            ax_disc_pred.set_ylim(0, 1)
            ax_disc_pred.set_ylabel("Probabilidad")
            ax_disc_pred.set_title("El Discriminador dice...")

            for bar in bars:
                yval = bar.get_height()
                ax_disc_pred.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval*100:.1f}%', ha='center', va='bottom', fontsize=10)

            st.pyplot(fig_disc_pred)
            plt.close(fig_disc_pred)
            st.markdown("*(El Discriminador intenta adivinar si la imagen es de la 'vida real' o es una 'falsificación' del Generador)*")
        else:
            st.info("Genera una imagen para ver cómo el Generador la crea y cómo el Discriminador la evalúa.")

st.write("---")

# --- Configuración de la API de OpenAI ---
# Ya están inicializados como None al principio del script.
# Solo necesitamos poblar openai_api_key_value si se encuentra.

# Comprobar la clave de la API de OpenAI en los secretos de Streamlit
if "openai_api_key" in st.secrets:
    openai_api_key_value = st.secrets['openai_api_key']
elif "OPENAI_API_KEY" in st.secrets: # Comprobar también en mayúsculas
    openai_api_key_value = st.secrets['OPENAI_API_KEY']

# Solo intentar inicializar el cliente de OpenAI si se encuentra una clave API
if openai_api_key_value:
    try:
        # Comprobar si la clase OpenAI se importó correctamente (es decir, no es None)
        if OpenAI is not None:
            client = OpenAI(api_key=openai_api_key_value)
        else:
            st.warning("La librería 'openai' no se pudo cargar. El asistente Ganesh no estará disponible.")
    except Exception as e:
        st.error(f"Error al inicializar cliente OpenAI con la clave proporcionada: {e}")
        client = None # Establecer explícitamente en None si falla la inicialización
else:
    st.warning("¡ATENCIÓN! La clave de la API de OpenAI no se ha encontrado en `secrets.toml`.")
    st.info("""
    Para usar el chatbot de Ganesh, necesitas añadir tu clave de la API de OpenAI a tu archivo `secrets.toml`.

    **Pasos:**
    1.  Crea una carpeta llamada `.streamlit` en la misma carpeta donde está este script.
    2.  Dentro de `.streamlit`, crea un archivo llamado `secrets.toml`.
    3.  Abre `secrets.toml` y añade una de estas líneas (¡solo una, según cómo quieras llamarla!):
        ```toml
        openai_api_key = "sk-TU_CLAVE_API_AQUI"
        # O si prefieres usar el nombre en mayúsculas:
        OPENAI_API_KEY = "sk-TU_CLAVE_API_AQUI"
        ```
        **Recuerda reemplazar `sk-TU_CLAVE_API_AQUI` con tu clave API real, incluyendo las comillas.**
    """)


# --- Sección del Asistente Virtual (Ganesh) ---
# Esta sección debe estar después de la inicialización de `client`
st.sidebar.header("Asistente Virtual - Ganesh")
st.sidebar.markdown("""
¡Hola! Soy Ganesh, tu asistente virtual en este Laboratorio GAN.
Pregúntame cualquier cosa sobre Redes Neuronales, Inteligencia Artificial,
o cómo funcionan el Generador y el Discriminador.
""")

# Ahora, 'client' está garantizado para estar definido (ya sea como un objeto OpenAI o None)
if client:
    # Inicializar historial de chat si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes del historial de chat al volver a ejecutar la aplicación
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Aceptar la entrada del usuario
    if prompt := st.chat_input("Pregúntale algo a Ganesh..."):
        # Añadir mensaje del usuario al historial de chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Mostrar mensaje del usuario en el chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Preparar el contexto de la conversación
            messages_for_api = [
                {"role": "system", "content": "Eres un asistente de IA llamado Ganesh, especializado en explicar sobre Redes Generativas Antagónicas (GANs), Redes Neuronales e Inteligencia Artificial de manera sencilla y didáctica, especialmente para niños. Usa un tono amigable y accesible. No respondas preguntas fuera de este dominio."}
            ]
            for message in st.session_state.messages:
                messages_for_api.append({"role": message["role"], "content": message["content"]})

            try:
                for chunk in client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages_for_api,
                    stream=True,
                ):
                    full_response += chunk.choices[0].delta.content or ""
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error al comunicarse con la API de OpenAI: {e}. Asegúrate de que tu clave es válida y tienes conexión a internet.")
                full_response = "Lo siento, no pude conectar con la IA en este momento."

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Opcional: gTTS para leer la respuesta de Ganesh
        if full_response and client: # Solo leer si hay respuesta y el cliente está configurado
            try:
                tts = gTTS(full_response, lang='es')
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                st.audio(fp, format='audio/mp3', start_time=0)
            except Exception as e:
                st.warning(f"No se pudo generar el audio de Ganesh: {e}")
else:
    st.sidebar.warning("El asistente Ganesh no está disponible. Configura tu clave de la API de OpenAI y asegúrate de que la librería 'openai' esté instalada.")



# --- Sección de Chatbot de Juego con Ganesh ---

st.header("¡Juega y Aprende con Maestro Ganesh sobre Redes Generativas Antagónicas!")
st.markdown("¡Hola! Soy **Maestro Ganesh**, tu guía en el fascinante mundo de la creación de imágenes y datos sintéticos. ¿Listo para desentrañar los secretos de las GANs y cómo generan maravillas a partir de la nada?")

if client:
    if "ganesh_game_active" not in st.session_state:
        st.session_state.ganesh_game_active = False
    if "ganesh_game_messages" not in st.session_state:
        st.session_state.ganesh_game_messages = []
    if "ganesh_current_question" not in st.session_state:
        st.session_state.ganesh_current_question = None
    if "ganesh_current_options" not in st.session_state:
        st.session_state.ganesh_current_options = {}
    if "ganesh_correct_answer" not in st.session_state:
        st.session_state.ganesh_correct_answer = None
    if "ganesh_awaiting_next_game_decision" not in st.session_state:
        st.session_state.ganesh_awaiting_next_game_decision = False
    if "ganesh_game_needs_new_question" not in st.session_state:
        st.session_state.ganesh_game_needs_new_question = False
    if "ganesh_correct_streak" not in st.session_state:
        st.session_state.ganesh_correct_streak = 0

    ganesh_game_system_prompt = f"""
    Eres un **experto consumado en Redes Generativas Antagónicas (GANs)** y Deep Learning, con una especialización profunda en su arquitectura, principios de entrenamiento y diversas aplicaciones. Comprendes a fondo cómo el Generador y el Discriminador compiten y colaboran para crear datos sintéticos realistas. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio de las GANs mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Generación impecable! Tu GAN ha creado una obra maestra." o "Esa discriminación necesita ajuste. Revisemos los datos falsos."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "Una GAN se compone de dos redes neuronales que compiten entre sí: un Generador que crea datos y un Discriminador que evalúa si son reales o falsos..."]
    [Pregunta para continuar, ej: "¿Listo para optimizar tus arquitecturas GAN?" o "¿Quieres profundizar en las aplicaciones de las GANs?"]

    **Reglas adicionales para el Experto en Redes Generativas Antagónicas (Maestro Ganesh):**
    * **Enfoque Riguroso en GANs:** Todas tus preguntas y explicaciones deben girar en torno a las Redes Generativas Antagónicas. Cubre sus fundamentos (el juego de suma cero), los componentes clave (Generador, Discriminador), el proceso de entrenamiento antagónico, el ruido latente, la función de pérdida, los desafíos del entrenamiento (colapso de modo, inestabilidad), y sus diversas aplicaciones.
    * **¡VARIEDAD, VARIADAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de GAN que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General y Funcionamiento:** ¿Qué es una GAN? ¿Cómo funciona el juego entre Generador y Discriminador?
        * **El Generador:** Su propósito (crear datos falsos), entrada de ruido latente.
        * **El Discriminador:** Su propósito (distinguir reales de falsos), salida binaria.
        * **Función de Pérdida:** Pérdida del Generador, pérdida del Discriminador, optimización del juego.
        * **Entrenamiento Antagónico:** Proceso iterativo, roles cambiantes de las redes.
        * **Ruido Latente:** Qué es, su importancia para la diversidad de la generación.
        * **Tipos de GANs:** (conceptual) DCGAN, CycleGAN, StyleGAN, Conditional GANs.
        * **Métricas de Evaluación:** FID, Inception Score (conceptual).
        * **Desafíos del Entrenamiento:** Colapso de modo (mode collapse), inestabilidad, vanishing gradients.
        * **Aplicaciones Principales:** Generación de imágenes realistas, transferencia de estilo, super-resolución, aumento de datos.
        * **Ética y Sesgos:** Consideraciones éticas en la generación de contenido.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.ganesh_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Aprendiz Generativo – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la idea básica de qué es una GAN y para qué sirve (ej., crear caras de personas que no existen).
            * *Tono:* "Estás dando tus primeros pasos en el arte de la creación con inteligencia artificial. ¡Maestro Ganesh te guiará!"
        * **Nivel 2 (Constructor de Realidades – 3-5 respuestas correctas):** Tono más técnico. Introduce los conceptos de **Generador** y **Discriminador** y su rol. Preguntas sobre cómo interactúan o qué hace cada componente.
            * *Tono:* "Tu comprensión de los componentes básicos de la creación de IA está tomando forma. ¡Maestro Ganesh está impresionado!"
        * **Nivel 3 (Arquitecto de Ilusiones – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en los detalles del **ruido latente**, las **funciones de pérdida**, los conceptos de **colapso de modo** o las bases de las **Conditional GANs**.
            * *Tono:* "Tu maestría en el diseño de arquitecturas GAN te permite construir mundos sintéticos cada vez más convincentes. ¡Digno de un Arquitecto de Ilusiones para Maestro Ganesh!"
        * **Nivel Maestro (Artista de IA – 9+ respuestas correctas):** Tono de **especialista en la vanguardia de la IA generativa**. Preguntas sobre el diseño de arquitecturas GAN avanzadas (ej. StyleGAN conceptualmente), técnicas para mitigar el colapso de modo, evaluación de la calidad de las GANs (FID), o las implicaciones éticas y futuras de las GANs. Se esperan respuestas que demuestren una comprensión teórica y práctica robusta, incluyendo sus limitaciones y el estado del arte.
            * *Tono:* "Tu genialidad en las GANs te posiciona como un verdadero artista de la inteligencia artificial, capaz de dar forma a la realidad digital. ¡Maestro Ganesh se inclina ante ti, Artista de IA!"
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Generar fotos de gatos realistas o crear canciones nuevas.
        * **Nivel 2:** Cambiar el estilo de una foto (ej., de verano a invierno), o crear caras de personas que no existen.
        * **Nivel 3:** Aumentar la resolución de imágenes antiguas (super-resolución), o generar datos sintéticos para entrenar otros modelos de IA.
        * **Nivel Maestro:** Crear avatares digitales personalizables, diseñar nuevos materiales con propiedades deseadas, o generar contenido artístico hiperrealista para videojuegos o películas.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre REDES GENERATIVAS ANTAGÓNICAS (GANs), y asegúrate de que no se parezca a las anteriores.
    """

    def parse_ganesh_question_response(raw_text):
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

    def parse_ganesh_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"
    
    def set_ganesh_level(target_streak, level_name):
        st.session_state.ganesh_correct_streak = target_streak
        st.session_state.ganesh_game_active = True
        st.session_state.ganesh_game_messages = []
        st.session_state.ganesh_current_question = None
        st.session_state.ganesh_current_options = {}
        st.session_state.ganesh_correct_answer = None
        st.session_state.ganesh_game_needs_new_question = True
        st.session_state.ganesh_awaiting_next_game_decision = False
        st.session_state.ganesh_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}**! Prepárate para preguntas más desafiantes. ¡Aquí va tu primera!"})
        st.rerun()

    col_game_buttons_ganesh, col_level_up_buttons_ganesh = st.columns([1, 2])

    with col_game_buttons_ganesh:
        if st.button("¡Vamos a jugar con Maestro Ganesh!", key="start_ganesh_game_button"):
            st.session_state.ganesh_game_active = True
            st.session_state.ganesh_game_messages = []
            st.session_state.ganesh_current_question = None
            st.session_state.ganesh_current_options = {}
            st.session_state.ganesh_correct_answer = None
            st.session_state.ganesh_game_needs_new_question = True
            st.session_state.ganesh_awaiting_next_game_decision = False
            st.session_state.ganesh_correct_streak = 0
            st.rerun()
    
    with col_level_up_buttons_ganesh:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un creador experto? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_ganesh, col_lvl2_ganesh, col_lvl3_ganesh = st.columns(3)
        with col_lvl1_ganesh:
            if st.button("Subir a Nivel Medio (GAN)", key="level_up_medium_ganesh"):
                set_ganesh_level(3, "Constructor de Realidades")
        with col_lvl2_ganesh:
            if st.button("Subir a Nivel Avanzado (GAN)", key="level_up_advanced_ganesh"):
                set_ganesh_level(6, "Arquitecto de Ilusiones")
        with col_lvl3_ganesh:
            if st.button("👑 ¡Maestro GANesh! (GAN)", key="level_up_champion_ganesh"):
                set_ganesh_level(9, "Artista de IA")


    for message in st.session_state.ganesh_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if st.session_state.ganesh_game_active:
        if st.session_state.ganesh_current_question is None and st.session_state.ganesh_game_needs_new_question and not st.session_state.ganesh_awaiting_next_game_decision:
            with st.spinner("Maestro Ganesh está preparando una pregunta..."):
                try:
                    ganesh_game_messages_for_api = [{"role": "system", "content": ganesh_game_system_prompt}]
                    for msg in st.session_state.ganesh_game_messages[-6:]:
                        if msg["role"] == "assistant" and msg["content"].startswith("**"):
                            ganesh_game_messages_for_api.append({"role": "assistant", "content": f"PREGUNTA ANTERIOR: {msg['content'].splitlines()[0]}"})
                        elif msg["role"] == "user" and "MI RESPUESTA:" not in msg["content"]:
                            ganesh_game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {msg['content']}"})

                    ganesh_game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre QUÉ SON LAS GANs siguiendo el formato exacto."})

                    # Replace with actual API call if client is configured
                    if client:
                        # Assuming client is an object with a chat.completions.create method
                        ganesh_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=ganesh_game_messages_for_api,
                            temperature=0.7,
                            max_tokens=250
                        )
                        raw_ganesh_question_text = ganesh_response.choices[0].message.content
                    else:
                        # Dummy response for testing without API key
                        raw_ganesh_question_text = "Pregunta: ¿Qué significa 'GAN' en el contexto de la IA?\nA) Redes Generativas Avanzadas\nB) Gráficos Algorítmicos Neuronales\nC) Redes Generativas Antagónicas\nRespuestaCorrecta: C"
                        
                    question, options, correct_answer_key = parse_ganesh_question_response(raw_ganesh_question_text)

                    if question:
                        st.session_state.ganesh_current_question = question
                        st.session_state.ganesh_current_options = options
                        st.session_state.ganesh_correct_answer = correct_answer_key
                        st.session_state.ganesh_game_needs_new_question = False
                        
                        question_content = f"**Nivel {int(st.session_state.ganesh_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.ganesh_correct_streak}**\n\n**Pregunta de Maestro Ganesh:** {question}\n\n"
                        for k, v in options.items():
                            question_content += f"**{k})** {v}\n"
                        
                        st.session_state.ganesh_game_messages.append({"role": "assistant", "content": question_content})
                        st.rerun()
                    else:
                        st.error("Maestro Ganesh no pudo generar una pregunta válida. Intenta de nuevo.")
                        st.session_state.ganesh_game_active = False
                        st.session_state.ganesh_game_messages.append({"role": "assistant", "content": "Maestro Ganesh no pudo generar una pregunta válida. Parece que hay un problema. Por favor, reinicia el juego."})

                except Exception as e:
                    st.error(f"Error al comunicarse con la API de OpenAI para la pregunta: {e}")
                    st.session_state.ganesh_game_active = False
                    st.session_state.ganesh_game_messages.append({"role": "assistant", "content": "Lo siento, tengo un problema para conectar con mi cerebro (la API). ¡Por favor, reinicia el juego!"})
                    st.rerun()

        if st.session_state.ganesh_current_question and not st.session_state.ganesh_awaiting_next_game_decision:
            if st.session_state.get('last_played_ganesh_question') != st.session_state.ganesh_current_question:
                try:
                    # tts_text = f"Nivel {int(st.session_state.ganesh_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.ganesh_correct_streak}. Pregunta de Maestro Ganesh: {st.session_state.ganesh_current_question}. Opción A: {st.session_state.ganesh_current_options.get('A', '')}. Opción B: {st.session_state.ganesh_current_options.get('B', '')}. Opción C: {st.session_state.ganesh_current_options.get('C', '')}."
                    # tts = gTTS(text=tts_text, lang='es', slow=False)
                    # fp = io.BytesIO()
                    # tts.write_to_fp(fp)
                    # fp.seek(0)
                    # st.audio(fp, format='audio/mp3', start_time=0)
                    # st.session_state.last_played_ganesh_question = st.session_state.ganesh_current_question
                    pass # Desactivado temporalmente si no se ha configurado gTTS
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")

            with st.form(key="ganesh_game_form"):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_answer = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.ganesh_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.ganesh_current_options[x]}",
                        key="ganesh_answer_radio",
                        label_visibility="collapsed"
                    )
                submit_button = st.form_submit_button("¡Enviar Respuesta!")

            if submit_button:
                st.session_state.ganesh_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_answer}) {st.session_state.ganesh_current_options[user_answer]}"})
                prev_streak = st.session_state.ganesh_correct_streak
                is_correct = (user_answer == st.session_state.ganesh_correct_answer)

                if is_correct:
                    st.session_state.ganesh_correct_streak += 1
                else:
                    st.session_state.ganesh_correct_streak = 0

                radio_placeholder.empty()

                if st.session_state.ganesh_correct_streak > 0 and \
                   st.session_state.ganesh_correct_streak % 3 == 0 and \
                   st.session_state.ganesh_correct_streak > prev_streak:
                    
                    if st.session_state.ganesh_correct_streak < 9:
                        current_level_text = ""
                        if st.session_state.ganesh_correct_streak == 3:
                            current_level_text = "Constructor de Realidades (¡ya distingues lo real de lo generado!)"
                        elif st.session_state.ganesh_correct_streak == 6:
                            current_level_text = "Arquitecto de Ilusiones (¡tus GANs son cada vez más convincentes!)"
                        
                        level_up_message = f"¡Increíble! ¡Has respondido {st.session_state.ganesh_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de GANs. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a creador/a de IA! 🚀"
                        st.session_state.ganesh_game_messages.append({"role": "assistant", "content": level_up_message})
                        st.balloons()
                        try:
                            # tts_level_up = gTTS(text=level_up_message, lang='es', slow=False)
                            # audio_fp_level_up = io.BytesIO()
                            # tts_level_up.write_to_fp(audio_fp_level_up)
                            # audio_fp_level_up.seek(0)
                            # st.audio(audio_fp_level_up, format="audio/mp3", start_time=0)
                            # time.sleep(2)
                            pass # Desactivado temporalmente
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de subida de nivel: {e}")
                    elif st.session_state.ganesh_correct_streak >= 9:
                        medals_earned = (st.session_state.ganesh_correct_streak - 6) // 3 
                        medal_message = f"🏅 ¡FELICITACIONES, MAESTRO/A GANESH! ¡Has ganado tu {medals_earned}ª Medalla de Creación Generativa! ¡Tu habilidad es asombrosa y digna de un verdadero EXPERTO en GANs! ¡Sigue así! 🌟"
                        st.session_state.ganesh_game_messages.append({"role": "assistant", "content": medal_message})
                        st.balloons()
                        st.snow()
                        try:
                            # tts_medal = gTTS(text=medal_message, lang='es', slow=False)
                            # audio_fp_medal = io.BytesIO()
                            # tts_medal.write_to_fp(audio_fp_medal)
                            # audio_fp_medal.seek(0)
                            # st.audio(audio_fp_medal, format="audio/mp3", start_time=0)
                            # time.sleep(3)
                            pass # Desactivado temporalmente
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de medalla: {e}")
                        
                        if prev_streak < 9:
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Maestro GANesh**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros ingenieros de IA generativa! ¡Adelante!"
                            st.session_state.ganesh_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                # tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                # audio_fp_level_up_champion = io.BytesIO()
                                # tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                                # audio_fp_level_up_champion.seek(0)
                                # st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0)
                                # time.sleep(2)
                                pass # Desactivado temporalmente
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón: {e}")


                with st.spinner("Maestro Ganesh está pensando su respuesta..."):
                    try:
                        feedback_prompt = f"""
                        El usuario respondió '{user_answer}'. La pregunta era: '{st.session_state.ganesh_current_question}'.
                        La respuesta correcta era '{st.session_state.ganesh_correct_answer}'.
                        Da feedback como Maestro Ganesh.
                        Si es CORRECTO, el mensaje es "¡Generación impecable! ¡Acertaste como un Maestro de las GANs!" o similar.
                        Si es INCORRECTO, el mensaje es "¡Necesitas más entrenamiento para generar la respuesta correcta!" o similar.
                        Luego, una explicación sencilla para el usuario.
                        Finalmente, pregunta: "¿Quieres seguir creando maravillas con Maestro Ganesh?".
                        **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                        """
                        # Replace with actual API call if client is configured
                        if client:
                            feedback_response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": ganesh_game_system_prompt},
                                    {"role": "user", "content": feedback_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=300
                            )
                            raw_ganesh_feedback_text = feedback_response.choices[0].message.content
                        else:
                            # Dummy response for testing without API key
                            if is_correct:
                                raw_ganesh_feedback_text = "¡Generación impecable! ¡Acertaste como un Maestro de las GANs!\nUna GAN (Generative Adversarial Network) es un tipo de modelo de IA que puede generar datos nuevos y realistas, como imágenes o audio.\n¿Quieres seguir creando maravillas con Maestro Ganesh?"
                            else:
                                raw_ganesh_feedback_text = "¡Necesitas más entrenamiento para generar la respuesta correcta!\nUna GAN (Generative Adversarial Network) es un tipo de modelo de IA que puede generar datos nuevos y realistas, como imágenes o audio.\n¿Quieres seguir creando maravillas con Maestro Ganesh?"

                        feedback_message, explanation_message, continue_question = parse_ganesh_feedback_response(raw_ganesh_feedback_text)
                        
                        st.session_state.ganesh_game_messages.append({"role": "assistant", "content": feedback_message})
                        st.session_state.ganesh_game_messages.append({"role": "assistant", "content": explanation_message})
                        st.session_state.ganesh_game_messages.append({"role": "assistant", "content": continue_question})

                        try:
                            # tts = gTTS(text=f"{feedback_message}. {explanation_message}. {continue_question}", lang='es', slow=False)
                            # audio_fp = io.BytesIO()
                            # tts.write_to_fp(audio_fp)
                            # audio_fp.seek(0)
                            # st.audio(audio_fp, format="audio/mp3", start_time=0)
                            pass # Desactivado temporalmente
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")


                        st.session_state.ganesh_current_question = None
                        st.session_state.ganesh_current_options = {}
                        st.session_state.ganesh_correct_answer = None
                        st.session_state.ganesh_game_needs_new_question = False
                        st.session_state.ganesh_awaiting_next_game_decision = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error al comunicarse con la API de OpenAI para el feedback: {e}")
                        st.session_state.ganesh_game_active = False
                        st.session_state.ganesh_game_messages.append({"role": "assistant", "content": "Lo siento, no puedo darte feedback ahora mismo. ¡Por favor, reinicia el juego!"})
                        st.rerun()

        if st.session_state.ganesh_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col_continue, col_end = st.columns(2)
            with col_continue:
                if st.button("👍 Sí, quiero seguir creando!", key="continue_ganesh_game"):
                    st.session_state.ganesh_awaiting_next_game_decision = False
                    st.session_state.ganesh_game_needs_new_question = True
                    st.session_state.ganesh_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío de Maestro Ganesh!"})
                    st.rerun()
            with col_end:
                if st.button("👎 No, gracias! Necesito un descanso creativo.", key="end_ganesh_game"):
                    st.session_state.ganesh_game_active = False
                    st.session_state.ganesh_awaiting_next_game_decision = False
                    st.session_state.ganesh_game_messages.append({"role": "assistant", "content": "¡Gracias por jugar! ¡Vuelve pronto para seguir creando maravillas con Maestro Ganesh!"})
                    st.rerun()

else:
    st.info("El chatbot Maestro Ganesh no está disponible porque la clave de la API de OpenAI no está configurada.")

st.write("---")