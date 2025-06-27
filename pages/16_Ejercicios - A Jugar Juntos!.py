import streamlit as st
import sys
import os


current_dir = os.path.dirname(__file__)
# Subir un nivel para llegar a Proyecto_final/
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from juegos import juego_recolector_datos
from juegos import juego_cerebro_virtual 
from juegos import juego_creador_historias
from juegos import juego_patrones_ocultos


# --- Configuración de la página ---
st.set_page_config(
    page_title="19_Ejercicios - ¡A Jugar Juntos!",
    layout="wide", # Usa un layout amplio para más espacio
    initial_sidebar_state="collapsed" # Puedes ajustar esto según prefieras
)

# --- Título y descripción de la página principal ---
st.title("Ejercicios - ¡A Jugar Juntos!")
st.write("¡Elige un juego para aprender y divertirte con la Inteligencia Artificial y los Datos!")

# --- Lógica de navegación con st.session_state ---
# Inicializa el estado de la sesión si no existe
# Por defecto, mostraremos el juego "recolector"
if 'current_game' not in st.session_state:
    st.session_state.current_game = "recolector"

# --- Botones para seleccionar el juego ---
# Usaremos columnas para que los botones se vean mejor organizados
# Añadiremos más columnas a medida que agreguemos más juegos
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10) # 10 columnas para 10 juegos

with col1:
    if st.button("1. Detective de Datos 🕵️‍♀️", key="btn_recolector"):
        st.session_state.current_game = "recolector"
with col2:
    if st.button("2. Cerebro Virtual 🧠", key="btn_cerebro"):
        st.session_state.current_game = "cerebro"
with col3:
    if st.button("3. Creador de Historias ✍️", key="btn_historias"):
        st.session_state.current_game = "historias"
with col4:
    if st.button("4. Patrones Ocultos ✨", key="btn_patrones"):
        st.session_state.current_game = "patrones"
with col5:
    if st.button("5. Biblioteca Digital 📚", key="btn_biblioteca"):
        st.session_state.current_game = "biblioteca"
with col6:
    if st.button("6. Tesoro de Datos 🗺️", key="btn_tesoro"):
        st.session_state.current_game = "tesoro"
with col7:
    if st.button("7. Gráfico Mágico 📊", key="btn_grafico"):
        st.session_state.current_game = "grafico"
with col8:
    if st.button("8. Conexión Neuronas ⚡", key="btn_neuronas"):
        st.session_state.current_game = "neuronas"
with col9:
    if st.button("9. Robot Traje 👗", key="btn_robot"):
        st.session_state.current_game = "robot"
with col10:
    if st.button("10. Sentimientos Secretos 🧐", key="btn_sentimientos"):
        st.session_state.current_game = "sentimientos"

st.markdown("---") # Un separador visual para delimitar los botones del contenido del juego

# --- Contenido del juego seleccionado ---
# Aquí es donde llamamos a la función run_game() del módulo de cada juego
if st.session_state.current_game == "recolector":
    juego_recolector_datos.run_game()
elif st.session_state.current_game == "cerebro":
    juego_cerebro_virtual.run_game()
elif st.session_state.current_game == "historias":
    juego_creador_historias.run_game()
elif st.session_state.current_game == "patrones":
    juego_patrones_ocultos.run_game()
elif st.session_state.current_game == "biblioteca":
    st.info("Cargando: **Ordena la Biblioteca Digital** (próximamente)")
    # juego_biblioteca_digital.run_game() # Descomentar cuando crees este archivo
elif st.session_state.current_game == "tesoro":
    st.info("Cargando: **El Tesoro de los Datos Escondidos** (próximamente)")
    # juego_tesoro_datos.run_game() # Descomentar cuando crees este archivo
elif st.session_state.current_game == "grafico":
    st.info("Cargando: **Construye tu Gráfico Mágico** (próximamente)")
    # juego_grafico_magico.run_game() # Descomentar cuando crees este archivo
elif st.session_state.current_game == "neuronas":
    st.info("Cargando: **La Conexión de Neuronas** (próximamente)")
    # juego_conexion_neuronas.run_game() # Descomentar cuando crees este archivo
elif st.session_state.current_game == "robot":
    st.info("Cargando: **Entrena a tu Robot Traje** (próximamente)")
    # juego_robot_traje.run_game() # Descomentar cuando crees este archivo
elif st.session_state.current_game == "sentimientos":
    st.info("Cargando: **El Detector de Sentimientos Secretos** (próximamente)")
    # juego_sentimientos_secretos.run_game() # Descomentar cuando crees este archivo
else:
    st.info("Por favor, selecciona un juego de la lista de arriba.")

