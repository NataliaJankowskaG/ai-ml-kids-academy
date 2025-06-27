import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from streamlit_lottie import st_lottie
from openai import OpenAI
from gtts import gTTS
import io
import random
import time

st.set_page_config(
    page_title="¿Qué es EDA?",
    layout="wide"
)

# ---- Función para cargar animación Lottie desde un archivo local ----
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo Lottie en la ruta: {filepath}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: El archivo Lottie '{filepath}' no es un JSON válido o está corrupto.")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar el archivo Lottie '{filepath}': {e}. Asegúrate de que el archivo no esté corrupto y sea un JSON válido.")
        return None

# --- Rutas a Lottie ---
LOTTIE_SHIP_PATH = os.path.join("assets", "lottie_animations", "ship.json")
LOTTIE_MAGNIFY_PATH = os.path.join("assets", "lottie_animations", "magnifying_glass.json")
LOTTIE_DATA_ORG_PATH = os.path.join("assets", "lottie_animations", "data_organization.json")

# --- Ruta a la imagen del Titanic local ---
TITANIC_IMAGE_PATH = os.path.join("assets", "imagenes", "Titanic.jpg")


# --- Configuración de la API de OpenAI ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai_api_key = None
    st.warning("Advertencia: La clave de la API de OpenAI no está configurada en `secrets.toml`. El chatbot Eddy no funcionará.")

client = OpenAI(api_key=openai_api_key) if openai_api_key else None


# Inicializar o cargar el dataset del Titanic en session_state
# SIN RENOMBRAR COLUMNAS
if 'original_titanic_df' not in st.session_state:
    df_raw = sns.load_dataset('titanic').copy()
    st.session_state.original_titanic_df = df_raw.copy()
    st.session_state.titanic_df = st.session_state.original_titanic_df.copy()
    st.session_state.eda_steps_applied = [] # Para registrar los pasos aplicados

st.subheader("¡Prepárate para la aventura de los datos!")

st.write("---")

# Sección 1: Introducción: La Aventura del Titanic y los Datos
st.header("🛳️ La Gran Aventura del Titanic: ¡Un Tesoro de Datos Espera!")
st.markdown("""
Imagina un barco gigante, el **Titanic**, que zarpó hace mucho tiempo.
A bordo iban muchas personas: familias, amigos, aventureros...
Cada persona en el barco es como una pieza de información, un **dato**.
¡Y todos esos datos juntos nos cuentan una historia!

Pero, ¡cuidado! Los datos pueden estar un poco desordenados, como un mapa del tesoro al que le faltan pedazos o está un poco borroso.
Nuestra misión hoy es ser **detectives de datos** para explorar, limpiar y entender ese tesoro.
""")

col_intro_left, col_intro_right = st.columns([1, 1])
with col_intro_left:
    lottie_ship = load_lottiefile(LOTTIE_SHIP_PATH)
    if lottie_ship:
        st_lottie(lottie_ship, height=200, width=200, key="titanic_intro")
    else:
        st.info("Consejo: Descarga una animación de barco (ej. 'ship.json') de LottieFiles.com y ponla en 'assets/lottie_animations/'.")
with col_intro_right:
    st.image(TITANIC_IMAGE_PATH, caption="El famoso barco Titanic", width=300)
    st.markdown("¡Nuestro objetivo no es el viaje, sino los **datos** que nos dejó!")

st.write("---")

# Sección 2: ¿Qué es EDA? Explorar, Limpiar, Entender para el Titanic
st.header("🕵️‍♂️ Eddy el Explorador: ¡Nuestra Misión de Datos!")
st.markdown("""
**EDA** significa **Análisis de Datos Exploratorio** (o como Eddy lo llama: **E**xplorar **D**atos **A** fondo).
Es como ponerse las gafas de detective y una lupa para entender bien la información.

Eddy nos enseñará tres cosas súper importantes:
1.  **Explorar:** Mirar los datos por primera vez. ¿Cuántas personas había? ¿De qué edad?
2.  **Limpiar:** ¡A veces los datos están sucios! Faltan datos o tienen errores. Tenemos que arreglarlos.
3.  **Entender:** ¿Qué descubrimos después de limpiar? ¿Hay patrones escondidos?

¡Es una parte crucial para que los modelos de IA puedan predecir el futuro con la información del pasado!
""")

col_eda_concept_left, col_eda_concept_right = st.columns([1, 1])
with col_eda_concept_left:
    lottie_magnify = load_lottiefile(LOTTIE_MAGNIFY_PATH)
    if lottie_magnify:
        st_lottie(lottie_magnify, height=180, width=180, key="eda_magnify")
    
with col_eda_concept_right:
    lottie_data_org = load_lottiefile(LOTTIE_DATA_ORG_PATH)
    if lottie_data_org:
        st_lottie(lottie_data_org, height=180, width=180, key="eda_organization")
    
st.write("---")

# --- Sección 3: ¡Manos a la Obra! El Laboratorio de Datos del Titanic ---
st.header("Tu Laboratorio a Bordo del Titanic: ¡A Limpiar y Descubrir!")
st.markdown("""
¡Vamos a trabajar con los datos reales del Titanic! Pero no te preocupes, nos centraremos en los números
y categorías para aprender sobre las personas a bordo y cómo vivían esa experiencia.
""")

st.markdown("### Datos Crudos del Titanic: ¡Mira el Tesoro sin Limpiar!")
st.markdown("""
Aquí tienes las primeras filas de los datos del Titanic.
¿Ves algo raro? Algunas columnas tienen información, otras no...
""")
st.dataframe(st.session_state.titanic_df.head())

st.markdown("---")

st.subheader("Desafío 1: ¡Datos Faltantes! ¿Quién no tiene edad registrada?")
st.markdown("""
¡Uff! Parece que a algunas personas les falta la edad en nuestros datos.
Un **dato faltante** es como una pieza que falta en nuestro rompecabezas.
Si no lo arreglamos, el modelo predictivo podría confundirse.

Mira cuántos datos faltan en cada columna:
""")
missing_data_info = st.session_state.titanic_df.isnull().sum()
# Filtrar solo columnas con datos faltantes y mostrarlas
missing_df = missing_data_info[missing_data_info > 0].reset_index().rename(columns={'index': 'Columna', 0: 'Datos Faltantes'})
if not missing_df.empty:
    st.dataframe(missing_df)
else:
    st.info("¡Enhorabuena! No hay datos faltantes en este momento.")


if 'age' in st.session_state.titanic_df.columns and st.session_state.titanic_df['age'].isnull().sum() > 0:
    st.markdown("""
    La columna 'age' tiene muchos datos faltantes. ¿Qué hacemos con ellos?
    """)
    age_missing_option = st.radio(
        "Elige una opción para la Edad:",
        ("Dejar los datos faltantes tal cual (¡no recomendado!)",
         "Eliminar las filas donde falta la Edad",
         "Rellenar las Edades que faltan con la edad promedio",
         "Rellenar las Edades que faltan con la edad más común (moda)"),
        key="age_missing_radio"
    )

    if st.button("Aplicar Opción para la Edad", key="apply_age_missing"):
        df_copy = st.session_state.titanic_df.copy()
        if age_missing_option == "Eliminar las filas donde falta la Edad":
            original_rows = len(df_copy)
            df_copy.dropna(subset=['age'], inplace=True)
            removed_rows = original_rows - len(df_copy)
            st.session_state.eda_steps_applied.append(f"Eliminadas {removed_rows} filas con Edad faltante.")
            st.success(f"¡Filas con Edad faltante eliminadas! ({removed_rows} filas menos)")
        elif age_missing_option == "Rellenar las Edades que faltan con la edad promedio":
            mean_age = df_copy['age'].mean()
            df_copy['age'].fillna(mean_age, inplace=True)
            st.session_state.eda_steps_applied.append(f"Edades faltantes rellenadas con el promedio ({mean_age:.2f}).")
            st.success(f"¡Edades faltantes rellenadas con el promedio: {mean_age:.2f}!")
        elif age_missing_option == "Rellenar las Edades que faltan con la edad más común (moda)":
            mode_age = df_copy['age'].mode()[0]
            df_copy['age'].fillna(mode_age, inplace=True)
            st.session_state.eda_steps_applied.append(f"Edades faltantes rellenadas con la edad más común ({mode_age}).")
            st.success(f"¡Edades faltantes rellenadas con la edad más común: {mode_age}!")
        else:
            st.info("No se realizó ningún cambio en los datos de Edad faltantes.")
            st.session_state.eda_steps_applied.append("Edad faltante no manipulada.")
        
        st.session_state.titanic_df = df_copy
        st.markdown("#### ¡Así se ven los datos de Edad después de tu elección!:")
        fig_age, ax_age = plt.subplots()
        sns.histplot(st.session_state.titanic_df['age'].dropna(), kde=True, ax=ax_age)
        ax_age.set_title('Distribución de la Edad')
        ax_age.set_xlabel('Edad')
        ax_age.set_ylabel('Frecuencia')
        st.pyplot(fig_age)
        plt.close(fig_age)
        st.rerun()
else:
    st.info("¡Bien! No hay datos faltantes en la columna 'age'. ¡Puedes pasar al siguiente desafío!")

st.markdown("---")

st.subheader("Desafío 2: ¡Columnas Innecesarias! ¿Qué información nos distrae?")
st.markdown("""
Algunas columnas tienen información que no ayuda a entender los patrones o a hacer predicciones,
¡son como adornos que nos distraen! Por ejemplo, el nombre de cada pasajero ('name')
o su número de identificación ('passengerId') no nos dicen si sobrevivieron o no en general.
También 'deck' tiene demasiados datos faltantes y 'embark_town' es redundante con 'embarked'.
""")

# Crear una lista de columnas que se pueden eliminar, filtrando las que ya no existen
available_cols_to_drop = [
    col for col in ['deck', 'embark_town', 'passengerId', 'name', 
                    'who', 'adult_male', 'alone', 'alive', 'class'] 
    if col in st.session_state.titanic_df.columns
]

if available_cols_to_drop:
    cols_to_drop = st.multiselect(
        "Elige las columnas que crees que no son útiles para analizar o predecir:",
        available_cols_to_drop,
        default=[col for col in ['deck', 'embark_town', 'passengerId', 'name', 
                                  'who', 'adult_male', 'alone', 'alive', 'class']
                  if col in available_cols_to_drop], 
        key="cols_to_drop_multiselect"
    )

    if st.button("Eliminar Columnas Seleccionadas", key="apply_drop_cols"):
        df_copy = st.session_state.titanic_df.copy()
        initial_cols = set(df_copy.columns)
        
        cols_to_drop_existing = [col for col in cols_to_drop if col in df_copy.columns]

        if cols_to_drop_existing:
            df_copy.drop(columns=cols_to_drop_existing, inplace=True, errors='ignore')
            final_cols = set(df_copy.columns)
            dropped_actual = initial_cols - final_cols
            
            if dropped_actual:
                st.session_state.eda_steps_applied.append(f"Columnas eliminadas: {', '.join(dropped_actual)}.")
                st.success(f"¡Columnas {', '.join(dropped_actual)} eliminadas con éxito!")
            else:
                st.info("No se eliminó ninguna columna nueva o las seleccionadas ya no existían.")
        else:
            st.info("Por favor, selecciona al menos una columna para eliminar.")

        st.session_state.titanic_df = df_copy
        st.rerun()

    st.markdown("#### ¡Así se ven los datos sin las columnas que has elegido!:")
    st.dataframe(st.session_state.titanic_df.head())
else:
    st.info("¡Ya no quedan columnas obvias para eliminar! ¡Buen trabajo!")

st.markdown("---")

st.subheader("Desafío 3: ¡Datos Categóricos! ¿Cómo hablamos con las máquinas?")
st.markdown("""
Nuestros modelos de IA son muy listos, ¡pero les encantan los números!
Algunas columnas como 'sex' (género: male/female) o 'embarked' (puerto de embarque) tienen palabras.
Necesitamos convertirlas a números para que el modelo las entienda.
""")

if 'sex' in st.session_state.titanic_df.columns and st.session_state.titanic_df['sex'].dtype == 'object':
    st.markdown("""
    La columna de género tiene 'male' - hombre y 'female'' - mujer. ¿Qué número le asignamos a cada uno?
    """)
    if st.button("Convertir género a números (0 y 1)", key="convert_gender_to_numbers"):
        df_copy = st.session_state.titanic_df.copy()
        df_copy['sex'] = df_copy['sex'].map({'male': 0, 'female': 1})
        st.session_state.titanic_df = df_copy
        st.session_state.eda_steps_applied.append("Columna de género convertida a números (0='male', 1='female').")
        st.success("¡Columna de género convertida a números! (male=0, female=1)")
        st.rerun()
else:
    if 'sex' in st.session_state.titanic_df.columns:
        st.info("La columna de género ya está en formato numérico.")
    else:
        st.info("La columna de género no está presente en tus datos.")

# Columna 'embarked'
if 'embarked' in st.session_state.titanic_df.columns and st.session_state.titanic_df['embarked'].dtype == 'object':
    st.markdown("""
    La columna 'embarked' (puerto de embarque) tiene las iniciales de los puertos de embarque. ¿La convertimos a números?
    (Necesitamos rellenar los pocos datos faltantes primero, para no perderlos en la conversión).
    """)
    if st.button("Rellenar y Convertir 'embarked' a números", key="convert_embarked_to_numbers"):
        df_copy = st.session_state.titanic_df.copy()
        most_common_embark = df_copy['embarked'].mode()[0]
        df_copy['embarked'].fillna(most_common_embark, inplace=True)
        df_copy['embarked'] = df_copy['embarked'].astype('category').cat.codes
        st.session_state.titanic_df = df_copy
        st.session_state.eda_steps_applied.append("Columna 'embarked' rellenada y convertida a números.")
        st.success("¡Columna 'embarked' rellenada y convertida a números!")
        st.rerun()
else:
    if 'embarked' in st.session_state.titanic_df.columns:
        st.info("La columna 'embarked' ya está en formato numérico.")
    else:
        st.info("La columna 'embarked' no está presente en tus datos.")

st.markdown("#### ¡Así se ven los datos con tus conversiones!:")
st.dataframe(st.session_state.titanic_df.head())

st.markdown("---")

st.subheader("Desafío 4: ¡Visualización para Entender! ¿Qué nos cuentan los gráficos?")
st.markdown("""
Ahora que nuestros datos están más limpios, podemos hacer gráficos para descubrir patrones.
¡Los gráficos son como las fotos del tesoro que estamos buscando!
""")

# Gráfico 1: Supervivencia por Género
if 'sex' in st.session_state.titanic_df.columns and 'survived' in st.session_state.titanic_df.columns:
    st.markdown("**1. Supervivencia por Género:**")
    df_plot_gender = st.session_state.titanic_df.copy()
    if df_plot_gender['sex'].dtype in ['int64', 'float64', 'int8']:
        df_plot_gender['sex_label'] = df_plot_gender['sex'].map({0: 'male', 1: 'female'})
    else:
        df_plot_gender['sex_label'] = df_plot_gender['sex']

    fig_gender_survived, ax_gender_survived = plt.subplots()
    sns.countplot(data=df_plot_gender, x='sex_label', hue='survived', ax=ax_gender_survived, palette='viridis')
    ax_gender_survived.set_title('Supervivencia por Género')
    ax_gender_survived.set_xlabel('Género')
    ax_gender_survived.set_ylabel('Número de Pasajeros')
    ax_gender_survived.legend(title='Sobrevivió', labels=['No', 'Sí'])
    st.pyplot(fig_gender_survived)
    plt.close(fig_gender_survived)
    st.markdown("¿Qué observas? ¿Parece que un género tuvo más probabilidades de sobrevivir que el otro?")
else:
    st.info("Las columnas 'sex' o 'survived' no están disponibles para el gráfico de Supervivencia por Sexo.")

# Gráfico 2: Supervivencia por Clase de Billete
if 'pclass' in st.session_state.titanic_df.columns and 'survived' in st.session_state.titanic_df.columns:
    st.markdown("**2. Supervivencia por Clase de Billete:**")
    fig_pclass_survived, ax_pclass_survived = plt.subplots()
    sns.countplot(data=st.session_state.titanic_df, x='pclass', hue='survived', ax=ax_pclass_survived, palette='magma')
    ax_pclass_survived.set_title('Supervivencia por Clase de Billete (1=Primera, 2=Segunda, 3=Tercera)')
    ax_pclass_survived.set_xlabel('Clase de Billete')
    ax_pclass_survived.set_ylabel('Número de Pasajeros')
    ax_pclass_survived.legend(title='Sobrevivió', labels=['No', 'Sí'])
    st.pyplot(fig_pclass_survived)
    plt.close(fig_pclass_survived)
    st.markdown("¿Qué clase de billete parece que tenía más probabilidad de sobrevivir?")
else:
    st.info("Las columnas 'pclass' o 'survived' no están disponibles para el gráfico de Supervivencia por Clase de Billete.")

# Gráfico 3: Edad vs. Tarifa del Billete (Supervivencia)
if all(col in st.session_state.titanic_df.columns for col in ['age', 'fare', 'survived']):
    st.markdown("**3. Edad vs. Tarifa del Billete (Color por Supervivencia):**")

    # Crear una copia para mapear 'survived' a texto para la visualización
    df_plot = st.session_state.titanic_df.copy()
    df_plot['survived_label'] = df_plot['survived'].map({0: 'No', 1: 'Sí'})

    fig_age_fare_survived, ax_age_fare_survived = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=df_plot,
        x='age',
        y='fare',
        hue='survived_label',
        palette={'No': 'coral', 'Sí': 'cornflowerblue'},
        alpha=0.7,
        s=80,
        ax=ax_age_fare_survived
    )

    ax_age_fare_survived.set_title('Edad vs. Tarifa del Billete (Color por Supervivencia)', fontsize=16)
    ax_age_fare_survived.set_xlabel('Edad', fontsize=12)
    ax_age_fare_survived.set_ylabel('Tarifa del Billete', fontsize=12)

    handles, labels = ax_age_fare_survived.get_legend_handles_labels()

    if handles and labels:
        if ax_age_fare_survived.legend_ is not None:
             ax_age_fare_survived.legend_.remove()

        legend = ax_age_fare_survived.legend(
            handles=handles,
            labels=labels,
            title='Sobrevivió',
            loc='upper right',
            markerscale=2,
            handletextpad=1.0,
            labelspacing=0.7,
            frameon=True,
            edgecolor='black',
            facecolor='white',
            borderpad=0.5
        )

        plt.setp(legend.get_title(), fontsize='12')
        for text in legend.get_texts():
            text.set_fontsize('12')
    else:
        st.warning("No se pudieron obtener los handles o labels para la leyenda.")

    st.pyplot(fig_age_fare_survived)
    plt.close(fig_age_fare_survived)
    st.markdown("¿Hay alguna relación entre la Edad, el precio del billete y la supervivencia? Por ejemplo, ¿los más jóvenes o los que pagaron más tuvieron más suerte?")
else:
    st.info("Algunas de las columnas 'age', 'fare' o 'survived' no están disponibles para este gráfico (quizás las eliminaste).")

st.markdown("---")

st.subheader("¡El Estado Actual de Tus Datos del Titanic Después de tu EDA!")
st.markdown("Has explorado y limpiado los datos. ¡Así es como se ven ahora las primeras filas y su información general!")
st.dataframe(st.session_state.titanic_df.head())
st.markdown("**Información de las columnas (Tipos de datos y conteo de no-nulos):**")
buffer = io.StringIO()
st.session_state.titanic_df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# --- INICIO DE LA EXPLICACIÓN PARA NIÑOS de df.info---
st.markdown("---")

st.markdown("### ¿Qué significan estos datos tan 'raros'?")
st.markdown("""
¡No te preocupes si la tabla de arriba parece un poco técnica! Te voy a explicar qué significa
cada parte de forma sencilla, ¡como si fuera una ficha de tus personajes favoritos!

Imagina que estos datos son como una lista súper detallada de todos los pasajeros del Titanic.
Cada fila es una persona, y cada **'columna'** (o dato) nos cuenta algo sobre ella.

Aquí te explico lo más importante:
""")

with st.expander("Haz clic aquí para entender las columnas (¡las fichas de cada pasajero!)"):
    st.markdown("""
    * **`<class 'pandas.core.frame.DataFrame'>`**:
        * Piensa que es como si te dijeran: "¡Eh, esto es una tabla de datos muy organizada!". Es el tipo de 'cuaderno' donde guardamos toda la información.

    * **`RangeIndex: 891 entries, 0 to 890`**:
        * Esto significa que tienes **891 pasajeros** en total en esta lista (desde el número 0 hasta el 890). ¡Como tener 891 cromos diferentes!

    * **`Data columns (total 10 columns):`**:
        * Nos dice que en tu lista hay **10 tipos diferentes de información** sobre cada pasajero (o sea, 10 columnas). Por ejemplo, una columna para su edad, otra para si sobrevivió, etc.

    * **# Column (Nombre del Dato)**:
        * Esta es la etiqueta que le hemos puesto a cada tipo de información. ¡Es el nombre de la característica que estamos viendo!

    * **Non-Null Count (Cuántos Datos Completos)**:
        * Esto es súper importante. Significa cuántos pasajeros tienen esa información **completa y sin errores**. Si ves "891 non-null", ¡significa que tenemos el dato para TODOS los 891 pasajeros! Si hay menos, es que a algunos les falta ese dato.

    * **Dtype (Tipo de Dato)**:
        * Esto nos dice qué tipo de información es:
            * **`int64`**: ¡Son **números enteros**! Como tu edad, o el número de hermanos que tienes. Son números que no tienen decimales (1, 2, 3...).
            * **`float64`**: ¡Son **números con decimales**! Como el precio de un billete (25.50 euros) o tu altura (1.45 metros).
            * **`category`**: Son datos que entran en **categorías fijas**. Por ejemplo, para la columna 'sexo' sería 'hombre' o 'mujer'. O para la 'clase' del billete, sería 'primera', 'segunda' o 'tercera'. ¡No son números que podamos sumar, sino etiquetas!
            * **`object`**: Puede ser **texto** (como nombres o descripciones) o datos más complejos. Aquí para 'alive' sería 'yes' o 'no'.

    * **`memory usage: 57.7+ KB`**:
        * Esto es como el 'tamaño' del cuaderno de datos en la memoria del ordenador. Nos dice cuánto espacio está usando.

    **¡En resumen, esta información nos ayuda a saber qué tipo de datos tenemos en cada columna y si están todos los datos completos para que podamos usarlos bien!**
    """)

if st.button("Reiniciar Laboratorio de Datos del Titanic", key="reset_eda_lab_final"):
    df_raw = sns.load_dataset('titanic').copy()
    st.session_state.original_titanic_df = df_raw.copy()
    st.session_state.titanic_df = st.session_state.original_titanic_df.copy()
    st.session_state.eda_steps_applied = []
    st.success("¡Laboratorio de datos reiniciado a su estado original!")
    st.rerun()

st.write("---")

# --- Sección de Chatbot de Juego con Eddy el Explorador ---
st.header("¡Juega y Aprende con Eddy el Explorador sobre EDA!")
st.markdown("¡Hola! Soy Eddy, tu guía en el mundo de los datos. ¿Listo para probar tus habilidades como explorador?")

if client:
    # Inicializa el estado del juego y los mensajes del chat
    if "eda_game_active" not in st.session_state:
        st.session_state.eda_game_active = False
    if "eda_game_messages" not in st.session_state:
        st.session_state.eda_game_messages = []
    if "eda_current_question" not in st.session_state:
        st.session_state.eda_current_question = None
    if "eda_current_options" not in st.session_state:
        st.session_state.eda_current_options = {}
    if "eda_correct_answer" not in st.session_state:
        st.session_state.eda_correct_answer = None
    if "eda_awaiting_next_game_decision" not in st.session_state:
        st.session_state.eda_awaiting_next_game_decision = False
    if "eda_game_needs_new_question" not in st.session_state:
        st.session_state.eda_game_needs_new_question = False
    if "eda_correct_streak" not in st.session_state:
        st.session_state.eda_correct_streak = 0


    # System prompt para el juego de preguntas de Eddy el Explorador
    eda_game_system_prompt = f"""
    Eres un **experto consumado en Análisis Exploratorio de Datos (EDA) y Visualización**, con una profunda comprensión de las metodologías, herramientas y mejores prácticas para extraer insights de los datos. Tu misión es actuar como un **tutor interactivo y desafiante**, guiando a los usuarios a través del dominio del EDA mediante un **juego de preguntas adaptativo**. Tu lenguaje y la complejidad de las preguntas deben ajustarse rigurosamente al nivel actual del usuario, alcanzando un tono y contenido de **nivel universitario/bootcamp** para los usuarios más avanzados.

    **TU ÚNICO TRABAJO es generar preguntas y respuestas en un formato específico y estricto, y NADA MÁS.**
    **¡Es CRÍTICO que tus preguntas sean MUY VARIADAS, CREATIVAS Y NO REPETITIVAS! Evita patrones de preguntas obvios o que sigan la misma estructura.**

    **Cuando te pida una pregunta, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    Pregunta: [Tu pregunta aquí]
    A) [Opción A]
    B) [Opción B]
    C) [Opción C]
    RespuestaCorrecta: [A, B o C]

    **Cuando te pida feedback, responde EXCLUSIVAMENTE con el siguiente formato, y NADA MÁS:**
    [Mensaje de Correcto/Incorrecto, ej: "¡Excelente insight! Has interpretado los datos con precisión." o "Revisa tu exploración. Esa conclusión no se deriva de los datos."]
    [Breve explicación del concepto, adecuada al nivel del usuario, ej: "El EDA es el paso inicial para comprender las características principales de un conjunto de datos..."]
    [Pregunta para continuar, ej: "¿Listo para descubrir más patrones ocultos en los datos?" o "¿Quieres profundizar en las técnicas avanzadas de visualización?"]

    **Reglas adicionales para el Experto en Análisis Exploratorio de Datos:**
    * **Enfoque Riguroso en EDA:** Todas tus preguntas y explicaciones deben girar en torno al Análisis Exploratorio de Datos. Cubre sus fundamentos (definición, objetivos, importancia), técnicas para datos univariados y multivariados, manejo de datos faltantes y atípicos, visualización de datos (tipos de gráficos, cuándo usar cada uno), estadísticas descriptivas, identificación de patrones, relaciones y anomalías, y la preparación de datos para el modelado.
    * **¡VARIEDAD, VARIEDAD, VARIEDAD!** Asegúrate de que cada pregunta sea diferente en su formulación, el ejemplo que utiliza y el concepto específico de EDA que evalúa. Rota entre los siguientes subtemas, asegurando una cobertura amplia y equilibrada:
        * **Concepto General y Objetivos del EDA:** ¿Qué es EDA? Por qué es crucial antes del modelado, objetivos principales (entender datos, identificar problemas, generar hipótesis).
        * **Estadísticas Descriptivas:** Medidas de tendencia central (media, mediana, moda), dispersión (varianza, desviación estándar, rango intercuartílico), asimetría y curtosis.
        * **Tipos de Datos y Escalas:** Cuantitativos (continuos, discretos), cualitativos (nominales, ordinales) y su impacto en el análisis.
        * **Manejo de Datos Faltantes (Missing Values):** Detección, estrategias de imputación (media, mediana, moda, regresión), impacto en el análisis.
        * **Detección y Tratamiento de Outliers (Valores Atípicos):** Métodos de detección (IQR, Z-score), estrategias de tratamiento (eliminación, transformación, capping).
        * **Visualización de Datos:**
            * **Univariada:** Histogramas, box plots, gráficos de densidad, gráficos de barras.
            * **Bivariada:** Scatter plots, heatmaps, pair plots, gráficos de líneas (para series temporales).
            * **Multivariada:** Gráficos 3D, facetado, uso de colores/tamaños.
            * **Principios de buena visualización:** Claridad, elección del gráfico adecuado, evitar gráficos engañosos.
        * **Identificación de Patrones y Relaciones:** Correlación (Pearson, Spearman), causalidad vs. correlación, clustering (como técnica exploratoria).
        * **Preparación de Datos para el Modelado:** Imputación, escalado/normalización, codificación de variables categóricas (One-Hot, Label Encoding).
        * **Herramientas para EDA:** Librerías comunes (Pandas, Matplotlib, Seaborn) y su aplicación.

    * **Progreso de Dificultad y Tono (Crucial):** El usuario ha respondido {st.session_state.adivino_correct_streak} preguntas correctas consecutivas.
        * **Nivel 1 (Explorador de Datos Principiante – 0-2 respuestas correctas):** Tono introductorio y conceptual. Preguntas sobre la importancia de mirar los datos y ejemplos sencillos de lo que se busca (valores faltantes, errores).
            * *Tono:* "Estás dando tus primeros pasos en el apasionante mundo de la exploración de datos."
        * **Nivel 2 (Analista de Datos – 3-5 respuestas correctas):** Tono más técnico. Introduce estadísticas descriptivas básicas, tipos de datos y gráficos univariados/bivariados comunes (histogramas, scatter plots). Preguntas sobre la interpretación básica de gráficos.
            * *Tono:* "Tu habilidad para desentrañar las características de los datos está mejorando notablemente."
        * **Nivel 3 (Científico de Datos – 6-8 respuestas correctas):** Tono de **nivel universitario/bootcamp**. Profundiza en el manejo de outliers, estrategias de imputación, métricas de correlación, y principios de visualización avanzada. Preguntas que requieren una comprensión de cómo el EDA informa el modelado predictivo.
            * *Tono:* "Tu maestría en el análisis exploratorio de datos es impecable, preparando el terreno para modelos robustos."
        * **Nivel Maestro (Especialista en Datos e Insights – 9+ respuestas correctas):** Tono de **especialista en análisis y descubrimiento de insights**. Preguntas sobre la elección de técnicas de visualización para datos complejos, EDA para series temporales o datos espaciales, la interpretación de patrones sutiles, o la justificación de decisiones de preprocesamiento basadas en el EDA. Se esperan respuestas que demuestren una comprensión crítica y la capacidad de comunicar insights complejos.
            * *Tono:* "Tu visión para transformar datos crudos en conocimiento accionable te posiciona como un verdadero experto en la ciencia de los datos."
        * Si el usuario responde 3 preguntas bien consecutivas, la dificultad sube GRADUALMENTE.
        * Si falla una pregunta, el contador se resetea a 0 y la dificultad vuelve al Nivel 1.
        * Si subes de nivel, ¡asegúrate de felicitar al usuario de forma entusiasta y explicando a qué tipo de nivel ha llegado!

    * **Ejemplos y Casos de Uso (Adaptados al Nivel):**
        * **Nivel 1:** Examinar una lista de edades de estudiantes para ver la edad más común.
        * **Nivel 2:** Visualizar la relación entre las horas de estudio y las notas de un examen, o identificar patrones de compra en un dataset de ventas.
        * **Nivel 3:** Realizar un EDA completo en un conjunto de datos financieros para identificar transacciones anómalas o preparar datos para un modelo de predicción de la demanda.
        * **Nivel Maestro:** Conducir un EDA para comprender las causas raíz de la baja retención de clientes, identificando segmentos clave y factores influyentes, o desarrollar visualizaciones interactivas para explorar datos genómicos complejos.

    * **Un Turno a la Vez:** Haz solo una pregunta a la vez y espera la respuesta del usuario antes de hacer la siguiente.
    * **Sé motivador y profesional:** Usa un tono que incite al aprendizaje y al rigor técnico, adaptado al nivel de cada etapa.
    * **Siempre responde en español de España.**
    * **La pregunta debe ser MUY VARIADA Y CREATIVA** sobre ANÁLISIS EXPLORATORIO DE DATOS (EDA), y asegúrate de que no se parezca a las anteriores.
    """

    # Función para parsear la respuesta de la IA (extraer pregunta, opciones y respuesta correcta)
    def parse_eda_question_response(raw_text):
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
    def parse_eda_feedback_response(raw_text):
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        st.warning(f"DEBUG: Formato de feedback inesperado de la API. Texto recibido:\n{raw_text}")
        return "Respuesta procesada.", "Aquí tienes la explicación.", "¿Quieres otra pregunta?"
    
    # --- Funciones para subir de nivel directamente ---
    def set_eda_level(target_streak, level_name):
        st.session_state.eda_correct_streak = target_streak
        st.session_state.eda_game_active = True
        st.session_state.eda_game_messages = []
        st.session_state.eda_current_question = None
        st.session_state.eda_current_options = {}
        st.session_state.eda_correct_answer = None
        st.session_state.eda_game_needs_new_question = True
        st.session_state.eda_awaiting_next_game_decision = False
        st.session_state.eda_game_messages.append({"role": "assistant", "content": f"¡Hola! ¡Has saltado directamente al **Nivel {level_name}**! Prepárate para preguntas más desafiantes. ¡Aquí va tu primera!"})
        st.rerun()

    # Botones para iniciar o reiniciar el juego y subir de nivel
    col_game_buttons_eda, col_level_up_buttons_eda = st.columns([1, 2])

    with col_game_buttons_eda:
        if st.button("¡Vamos a jugar con Eddy el Explorador!", key="start_eddy_game_button"):
            st.session_state.eda_game_active = True
            st.session_state.eda_game_messages = []
            st.session_state.eda_current_question = None
            st.session_state.eda_current_options = {}
            st.session_state.eda_correct_answer = None
            st.session_state.eda_game_needs_new_question = True
            st.session_state.eda_awaiting_next_game_decision = False
            st.session_state.eda_correct_streak = 0
            st.rerun()
    
    with col_level_up_buttons_eda:
        st.markdown("<p style='font-size: 1.1em; font-weight: bold;'>¿Ya eres un experto explorador? ¡Salta de nivel! 👇</p>", unsafe_allow_html=True)
        col_lvl1_eda, col_lvl2_eda, col_lvl3_eda = st.columns(3)
        with col_lvl1_eda:
            if st.button("Subir a Nivel Medio (EDA)", key="level_up_medium_eda"):
                set_eda_level(3, "Medio")
        with col_lvl2_eda:
            if st.button("Subir a Nivel Avanzado (EDA)", key="level_up_advanced_eda"):
                set_eda_level(6, "Avanzado")
        with col_lvl3_eda:
            if st.button("👑 ¡Maestro Explorador! (EDA)", key="level_up_champion_eda"):
                set_eda_level(9, "Campeón")


    # Mostrar mensajes del juego del chatbot
    for message in st.session_state.eda_game_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Lógica del juego del chatbot si está activo
    if st.session_state.eda_game_active:
        if st.session_state.eda_current_question is None and st.session_state.eda_game_needs_new_question and not st.session_state.eda_awaiting_next_game_decision:
            with st.spinner("Eddy está preparando una pregunta..."):
                try:
                    eda_game_messages_for_api = [{"role": "system", "content": eda_game_system_prompt}]
                    for msg in st.session_state.eda_game_messages[-6:]: 
                        if msg["role"] == "assistant" and msg["content"].startswith("**"):
                            eda_game_messages_for_api.append({"role": "assistant", "content": f"PREGUNTA ANTERIOR: {msg['content'].splitlines()[0]}"})
                        elif msg["role"] == "user" and "MI RESPUESTA:" not in msg["content"]:
                            eda_game_messages_for_api.append({"role": "user", "content": f"MI RESPUESTA: {msg['content']}"})

                    eda_game_messages_for_api.append({"role": "user", "content": "Genera una **nueva pregunta** sobre QUÉ ES EDA siguiendo el formato exacto."})

                    eda_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=eda_game_messages_for_api,
                        temperature=0.8,
                        max_tokens=300
                    )
                    raw_eda_question_text = eda_response.choices[0].message.content
                    question, options, correct_answer_key = parse_eda_question_response(raw_eda_question_text)

                    if question:
                        st.session_state.eda_current_question = question
                        st.session_state.eda_current_options = options
                        st.session_state.eda_correct_answer = correct_answer_key
                        st.session_state.eda_game_needs_new_question = False
                        
                        question_content = f"**Nivel {int(st.session_state.eda_correct_streak / 3) + 1} - Aciertos consecutivos: {st.session_state.eda_correct_streak}**\n\n**Pregunta de Eddy:** {question}\n\n"
                        for k, v in options.items():
                            question_content += f"**{k})** {v}\n"
                        
                        st.session_state.eda_game_messages.append({"role": "assistant", "content": question_content})
                        st.rerun()
                    else:
                        st.error("Eddy no pudo generar una pregunta válida. Intenta de nuevo.")
                        st.session_state.eda_game_active = False # Detener el juego para evitar bucle de errores
                        st.session_state.eda_game_messages.append({"role": "assistant", "content": "Eddy no pudo generar una pregunta válida. Parece que hay un problema. Por favor, reinicia el juego."})

                except Exception as e:
                    st.error(f"Error al comunicarse con la API de OpenAI para la pregunta: {e}")
                    st.session_state.eda_game_active = False
                    st.session_state.eda_game_messages.append({"role": "assistant", "content": "Lo siento, tengo un problema para conectar con mi cerebro (la API). ¡Por favor, reinicia el juego!"})
                    st.rerun()

        if st.session_state.eda_current_question and not st.session_state.eda_awaiting_next_game_decision:
            if st.session_state.get('last_played_question') != st.session_state.eda_current_question:
                try:
                    tts_text = f"Nivel {int(st.session_state.eda_correct_streak / 3) + 1}. Aciertos consecutivos: {st.session_state.eda_correct_streak}. Pregunta de Eddy: {st.session_state.eda_current_question}. Opción A: {st.session_state.eda_current_options.get('A', '')}. Opción B: {st.session_state.eda_current_options.get('B', '')}. Opción C: {st.session_state.eda_current_options.get('C', '')}."
                    tts = gTTS(text=tts_text, lang='es', slow=False)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.audio(fp, format='audio/mp3', start_time=0)
                    st.session_state.last_played_question = st.session_state.eda_current_question
                except Exception as e:
                    st.error(f"Error al generar o reproducir el audio de la pregunta: {e}")

            with st.form(key="eda_game_form"):
                radio_placeholder = st.empty()
                with radio_placeholder.container():
                    st.markdown("Elige tu respuesta:")
                    user_answer = st.radio(
                        "Elige tu respuesta:",
                        options=list(st.session_state.eda_current_options.keys()),
                        format_func=lambda x: f"{x}) {st.session_state.eda_current_options[x]}",
                        key="eda_answer_radio",
                        label_visibility="collapsed"
                    )
                submit_button = st.form_submit_button("¡Enviar Respuesta!")

            if submit_button:
                st.session_state.eda_game_messages.append({"role": "user", "content": f"MI RESPUESTA: {user_answer}) {st.session_state.eda_current_options[user_answer]}"})
                prev_streak = st.session_state.eda_correct_streak
                is_correct = (user_answer == st.session_state.eda_correct_answer)

                if is_correct:
                    st.session_state.eda_correct_streak += 1
                else:
                    st.session_state.eda_correct_streak = 0

                radio_placeholder.empty()

                if st.session_state.eda_correct_streak > 0 and \
                   st.session_state.eda_correct_streak % 3 == 0 and \
                   st.session_state.eda_correct_streak > prev_streak:
                    
                    if st.session_state.eda_correct_streak < 9:
                        current_level_text = ""
                        if st.session_state.eda_correct_streak == 3:
                            current_level_text = "Medio (como un adolescente que ya entiende la base de los datos)"
                        elif st.session_state.eda_correct_streak == 6:
                            current_level_text = "Avanzado (como un buen explorador de datos)"
                        
                        level_up_message = f"¡Increíble! ¡Has respondido {st.session_state.eda_correct_streak} preguntas seguidas correctamente! ¡Felicidades! Has subido al **Nivel {current_level_text}** de EDA. ¡Las preguntas serán un poco más desafiantes ahora! ¡Eres un/a verdadero/a detective de datos! 🚀"
                        st.session_state.eda_game_messages.append({"role": "assistant", "content": level_up_message})
                        st.balloons()
                        # Generar audio
                        try:
                            tts_level_up = gTTS(text=level_up_message, lang='es', slow=False)
                            audio_fp_level_up = io.BytesIO()
                            tts_level_up.write_to_fp(audio_fp_level_up)
                            audio_fp_level_up.seek(0)
                            st.audio(audio_fp_level_up, format="audio/mp3", start_time=0)
                            time.sleep(2)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de subida de nivel: {e}")
                    elif st.session_state.eda_correct_streak >= 9:
                        medals_earned = (st.session_state.eda_correct_streak - 6) // 3 
                        medal_message = f"🏅 ¡FELICITACIONES, MAESTRO/A EXPLORADOR/A! ¡Has ganado tu {medals_earned}ª Medalla de Exploración de Datos! ¡Tu habilidad es asombrosa y digna de un verdadero EXPERTO en EDA! ¡Sigue así! 🌟"
                        st.session_state.eda_game_messages.append({"role": "assistant", "content": medal_message})
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
                            level_up_message_champion = f"¡Has desbloqueado el **Nivel Campeón (Maestro Explorador de Datos)**! ¡Las preguntas ahora son solo para los verdaderos genios y futuros científicos de datos! ¡Adelante!"
                            st.session_state.eda_game_messages.append({"role": "assistant", "content": level_up_message_champion})
                            try:
                                tts_level_up_champion = gTTS(text=level_up_message_champion, lang='es', slow=False)
                                audio_fp_level_up_champion = io.BytesIO()
                                tts_level_up_champion.write_to_fp(audio_fp_level_up_champion)
                                audio_fp_level_up_champion.seek(0)
                                st.audio(audio_fp_level_up_champion, format="audio/mp3", start_time=0)
                                time.sleep(2)
                            except Exception as e:
                                st.warning(f"No se pudo reproducir el audio de campeón: {e}")

                with st.spinner("Eddy está pensando su respuesta..."):
                    try:
                        feedback_prompt = f"""
                        El usuario respondió '{user_answer}'. La pregunta era: '{st.session_state.eda_current_question}'.
                        La respuesta correcta era '{st.session_state.eda_correct_answer}'.
                        Da feedback como Eddy el Explorador.
                        Si es CORRECTO, el mensaje es "¡Exploración exitosa!" o similar.
                        Si es INCORRECTO, el mensaje es "¡Revisa tu brújula!" o similar.
                        Luego, una explicación sencilla para el usuario.
                        Finalmente, pregunta: "¿Quieres seguir explorando?".
                        **Sigue el formato estricto de feedback que tienes en tus instrucciones de sistema.**
                        """
                        feedback_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": eda_game_system_prompt},
                                {"role": "user", "content": feedback_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=300
                        )
                        raw_eda_feedback_text = feedback_response.choices[0].message.content
                        feedback_message, explanation_message, continue_question = parse_eda_feedback_response(raw_eda_feedback_text)
                        
                        st.session_state.eda_game_messages.append({"role": "assistant", "content": feedback_message})
                        st.session_state.eda_game_messages.append({"role": "assistant", "content": explanation_message})
                        st.session_state.eda_game_messages.append({"role": "assistant", "content": continue_question})

                        try:
                            tts = gTTS(text=f"{feedback_message}. {explanation_message}. {continue_question}", lang='es', slow=False)
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.audio(audio_fp, format="audio/mp3", start_time=0)
                        except Exception as e:
                            st.warning(f"No se pudo reproducir el audio de feedback: {e}")


                        st.session_state.eda_current_question = None # Limpiar pregunta actual
                        st.session_state.eda_current_options = {}
                        st.session_state.eda_correct_answer = None
                        st.session_state.eda_game_needs_new_question = False
                        st.session_state.eda_awaiting_next_game_decision = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error al comunicarse con la API de OpenAI para el feedback: {e}")
                        st.session_state.eda_game_active = False
                        st.session_state.eda_game_messages.append({"role": "assistant", "content": "Lo siento, no puedo darte feedback ahora mismo. ¡Por favor, reinicia el juego!"})
                        st.rerun()

        # Botones para continuar o terminar el juego
        if st.session_state.eda_awaiting_next_game_decision:
            st.markdown("---")
            st.markdown("¿Qué quieres hacer ahora?")
            col_continue, col_end = st.columns(2)
            with col_continue:
                if st.button("👍 Sí, quiero seguir explorando!", key="continue_eda_game"):
                    st.session_state.eda_awaiting_next_game_decision = False
                    st.session_state.eda_game_needs_new_question = True
                    st.session_state.eda_game_messages.append({"role": "assistant", "content": "¡Genial! ¡Aquí va tu siguiente desafío!"})
                    st.rerun()
            with col_end:
                if st.button("👎 No, gracias! Quiero descansar.", key="end_eda_game"):
                    st.session_state.eda_game_active = False
                    st.session_state.eda_awaiting_next_game_decision = False
                    st.session_state.eda_game_messages.append({"role": "assistant", "content": "¡Gracias por jugar! ¡Vuelve pronto para seguir explorando el mundo de los datos!"})
                    st.rerun()

else:
    st.info("El chatbot Eddy no está disponible porque la clave de la API de OpenAI no está configurada.")

st.write("---")