import streamlit as st
import time
import json # Necesario para cargar el archivo .json
import os
from streamlit_lottie import st_lottie # Importa la librería

# Función para cargar la animación Lottie
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def run_game():
    st.subheader("🕵️‍♀️ El Gran Recolector de Datos: ¿Quién se comió el pastel?")
    st.write("¡Bienvenido, pequeño detective! Tu misión es recolectar pistas para resolver un misterio delicioso.")
    st.info("El misterio: ¡Alguien se comió el último trozo de pastel de chocolate de la nevera! 🍰")

    st.markdown("---")

    # --- Pista 1: Ubicación ---
    st.markdown("### Pista 1: El Lugar del Crimen")
    st.write("¿Dónde estaba el pastel justo antes de desaparecer?")
    ubicacion = st.radio(
        "Elige una opción:",
        ["En la cocina, en la nevera", "En el salón, en la mesa", "En el jardín, bajo un árbol"],
        key="pista1_ubicacion"
    )

    # --- Pista 2: Hora del suceso ---
    st.markdown("### Pista 2: La Hora del Incidente")
    st.write("Según el reloj de la cocina, ¿a qué hora ocurrió el 'ataque' al pastel?")
    hora_str = st.select_slider(
        "Arrastra para seleccionar la hora:",
        options=["10:00 AM", "12:00 PM", "03:00 PM", "06:00 PM", "09:00 PM"],
        value="03:00 PM",
        key="pista2_hora"
    )
    # Convertir la hora a un formato numérico para la lógica
    hora_map = {"10:00 AM": 10, "12:00 PM": 12, "03:00 PM": 15, "06:00 PM": 18, "09:00 PM": 21}
    hora_num = hora_map[hora_str]

    # --- Pista 3: Evidencia dejada ---
    st.markdown("### Pista 3: Evidencia en el Suelo")
    st.write("Encontraste una pequeña pista cerca de la nevera. ¿Qué era?")
    evidencia = st.selectbox(
        "¿Qué evidencia encontraste?",
        ["Huellas de pata", "Migas de pan", "Un lápiz de color"],
        key="pista3_evidencia"
    )
    
    st.markdown("---")

    # --- Ruta a tu animación Lottie ---
    # Asumiendo que tu estructura es Proyecto_final/juegos/juego_recolector_datos.py
    # y la animación está en Proyecto_final/assets/lottie_animations/detective.json
    # Necesitamos una ruta relativa desde el script que se está ejecutando.
    # El archivo juego_recolector_datos.py está en 'juegos/', y 'assets/' está al nivel de 'juegos/'.
    # Entonces, subimos un nivel (..) y luego entramos en 'assets/lottie_animations/'

    # Para ser más robustos, obtenemos la ruta absoluta de la raíz del proyecto.
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

    lottie_detective_path = os.path.join(project_root, "assets", "lottie_animations", "detective.json")

    # Carga la animación Lottie una vez al inicio del script para eficiencia
    lottie_detective = load_lottiefile(lottie_detective_path)

    # --- Botón para analizar las pistas ---
    if st.button("¡Analizar Pistas y Resolver el Misterio! 🔍", key="btn_resolver_misterio"):
        st.write("Analizando todas tus pistas con cuidado...")
        with st.spinner("Un momento, detective..."):
            time.sleep(1.5)

        culpable = "desconocido"
        explicacion = ""
        
        # Caso principal: El perro Max
        if (ubicacion == "En la cocina, en la nevera" and
            hora_num >= 18 and
            evidencia == "Huellas de pata"):
            
            culpable = "Max, el perro travieso"
            explicacion = "¡Todas las pistas apuntan a él! Max adora el chocolate y sabe abrir la nevera por la tarde cuando nadie lo ve. ¡Buen trabajo, detective!"
            # Aquí usaremos la animación Lottie cargada
            st_lottie(lottie_detective, height=200, key="lottie_max_culpable")

        # Caso alternativo 1: El fantasma goloso (para fomentar la imaginación)
        elif (ubicacion == "En el salón, en la mesa" and
              hora_num < 12 and
              evidencia == "Migas de pan"):
            
            culpable = "¡El Fantasma de la Hora del Desayuno!"
            explicacion = "Hmm, estas pistas son curiosas... ¡Parece que alguien desayunó en el salón muy temprano y dejó migas! Tal vez un fantasma goloso que solo come pastel por las mañanas... ¡Tendremos que investigarlo más!"
            # Puedes usar la misma animación de detective o buscar otra Lottie de un fantasma
            st_lottie(lottie_detective, height=200, key="lottie_fantasma")


        # Caso alternativo 2: El artista distraído (para el lápiz)
        elif (evidencia == "Un lápiz de color"):
            
            culpable = "El pequeño artista distraído"
            explicacion = "¡Un lápiz de color! Parece que el pastel fue una inspiración para algún artista. Quizás se distrajo dibujando y el pastel desapareció... ¡Hay que buscar al artista y preguntarle!"
            # Puedes usar la misma animación de detective o buscar otra Lottie de un artista
            st_lottie(lottie_detective, height=200, key="lottie_artista")


        # Caso por defecto si ninguna de las anteriores coincide: Necesidad de más datos
        else:
            culpable = "¡Un misterio por resolver!"
            explicacion = "¡Interesante! Tus pistas nos han dado un poco de información, pero aún no son suficientes para saber quién se comió el pastel. Parece que necesitamos recolectar **más datos** y buscar nuevas pistas para resolver este enigmático caso."
            # Usa la animación de detective aquí también
            st_lottie(lottie_detective, height=200, key="lottie_mas_datos")

        # --- Mostrar el resultado ---
        st.success(f"¡RESULTADO DEL MISTERIO! 🎉")
        st.markdown(f"**El culpable más probable es:** {culpable}")
        st.write(explicacion)
        # st_lottie(lottie_detective, height=200, key="resultado_lottie") # La Lottie ya se llama dentro de cada condición

        st.markdown("**Has aprendido sobre cómo recolectar datos (pistas) y usarlos para resolver un problema.**")


    st.markdown("---")
    st.info("💡 **Concepto aprendido:** Este juego te ayuda a entender que los **datos** son como pistas. Al recolectar y analizar estas pistas, podemos **resolver problemas** y entender mejor lo que pasa a nuestro alrededor. ¡Justo como hacen los científicos de datos!")
    st.markdown("Pulsa los botones de arriba para cambiar de juego.")