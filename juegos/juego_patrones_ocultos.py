import streamlit as st
import random
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# --- Configuración de Elementos y sus Propiedades ---
ELEMENTOS_DISPONIBLES = {
    "manzana_roja": {"tipo": "fruta", "color": "rojo", "forma": "redonda", "habilidad": "n/a", "lados": -1, "img_url": "assets/imagenes/manzana_roja.png"}, 
    "platano_amarillo": {"tipo": "fruta", "color": "amarillo", "forma": "curva", "habilidad": "n/a", "lados": -1, "img_url": "assets/imagenes/platano_amarillo.png"}, 
    "uva_verde": {"tipo": "fruta", "color": "verde", "forma": "redonda", "habilidad": "n/a", "lados": -1, "img_url": "assets/imagenes/uva_verde.png"}, 
    "fresa_roja": {"tipo": "fruta", "color": "rojo", "forma": "corazon", "habilidad": "n/a", "lados": -1, "img_url": "assets/imagenes/fresa_roja.png"}, 
    
    "pato_amarillo": {"tipo": "animal", "color": "amarillo", "forma": "irregular", "habilidad": "volar", "lados": -1, "img_url": "assets/imagenes/pato_amarillo.png"}, 
    "gato_marron": {"tipo": "animal", "color": "marron", "forma": "irregular", "habilidad": "correr", "lados": -1, "img_url": "assets/imagenes/gato_marron.png"}, 
    "perro_negro": {"tipo": "animal", "color": "negro", "forma": "irregular", "habilidad": "correr", "lados": -1, "img_url": "assets/imagenes/perro_negro.png"}, 
    "pajaro_azul": {"tipo": "animal", "color": "azul", "forma": "irregular", "habilidad": "volar", "lados": -1, "img_url": "assets/imagenes/pajaro_azul.png"}, 
    
    "triangulo_azul": {"tipo": "forma", "color": "azul", "forma": "triangular", "habilidad": "n/a", "lados": 3, "img_url": "assets/imagenes/triangulo_azul.png"}, 
    "cuadrado_verde": {"tipo": "forma", "color": "verde", "forma": "cuadrada", "habilidad": "n/a", "lados": 4, "img_url": "assets/imagenes/cuadrado_verde.png"}, 
    "circulo_rojo": {"tipo": "forma", "color": "rojo", "forma": "redonda", "habilidad": "n/a", "lados": 0, "img_url": "assets/imagenes/circulo_rojo.png"}, 
    "estrella_amarilla": {"tipo": "forma", "color": "amarillo", "forma": "estrella", "habilidad": "n/a", "lados": 5, "img_url": "assets/imagenes/estrella_amarilla.png"}, 
}

# Definiciones de patrones secretos (con un ID para el modelo)
PATRONES_SECRETOS = [
    {"id": 0, "criterio": lambda e: e["tipo"] == "fruta", "nombre": "Todas las frutas", "explicacion_ia": "La IA aprendería que estos objetos tienen la característica 'tipo: fruta'."},
    {"id": 1, "criterio": lambda e: e["color"] == "rojo", "nombre": "Todo lo que sea rojo", "explicacion_ia": "La IA se fijaría en el color rojo de cada objeto."},
    {"id": 2, "criterio": lambda e: e["tipo"] == "animal" and e["habilidad"] == "volar", "nombre": "Animales que vuelan", "explicacion_ia": "La IA buscaría objetos que son 'tipo: animal' y que además tienen 'habilidad: volar'."},
    {"id": 3, "criterio": lambda e: e["forma"] == "redonda", "nombre": "Objetos redondos", "explicacion_ia": "La IA identificaría los objetos por su forma redonda, sea fruta o forma geométrica."},
    {"id": 4, "criterio": lambda e: e["tipo"] == "forma" and e["color"] == "azul", "nombre": "Formas geométricas azules", "explicacion_ia": "La IA reconocería objetos que son 'tipo: forma' y 'color: azul'."}
]

# --- Función de codificación (mapea propiedades a números o one-hot encoding) ---
def encode_elements(elements_data):
    all_types = sorted(list(set(e["tipo"] for e in elements_data.values())))
    all_colors = sorted(list(set(e["color"] for e in elements_data.values())))
    all_formas = sorted(list(set(e["forma"] for e in elements_data.values())))
    all_habilidades = sorted(list(set(e["habilidad"] for e in elements_data.values())))

    type_map = {val: i for i, val in enumerate(all_types)}
    color_map = {val: i for i, val in enumerate(all_colors)}
    forma_map = {val: i for i, val in enumerate(all_formas)}
    habilidad_map = {val: i for i, val in enumerate(all_habilidades)}

    encoded_features = []
    element_names = []

    for name, props in elements_data.items():
        features = [
            type_map[props["tipo"]],
            color_map[props["color"]],
            forma_map[props["forma"]],
            habilidad_map[props["habilidad"]],
            props["lados"]
        ]
        encoded_features.append(features)
        element_names.append(name)
    
    feature_names = ["tipo_encoded", "color_encoded", "forma_encoded", "habilidad_encoded", "lados"]
    
    df = pd.DataFrame(encoded_features, columns=feature_names, index=element_names)
    return df, type_map, color_map, forma_map, habilidad_map

ENCODED_DF, TYPE_MAP, COLOR_MAP, FORMA_MAP, HABILIDAD_MAP = encode_elements(ELEMENTOS_DISPONIBLES)


# --- Función para entrenar un modelo simple ---
def train_pattern_model(pattern_id):
    pattern_info = next(p for p in PATRONES_SECRETOS if p["id"] == pattern_id)
    
    labels = []
    for element_name, element_props in ELEMENTOS_DISPONIBLES.items():
        labels.append(1 if pattern_info["criterio"](element_props) else 0)
    
    X = ENCODED_DF
    y = pd.Series(labels, index=ENCODED_DF.index)

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()


# --- Función principal del juego ---
def run_game():
    st.subheader("🕵️‍♀️ El Buscador de Patrones Ocultos")
    st.write("¡Bienvenido, detective de patrones! Hay un criterio secreto para agrupar algunos de estos objetos. Tu misión es encontrar qué objetos cumplen ese criterio. ¡Cada vez que juegues, el patrón será diferente!")
    st.info("Objetivo: Entender cómo las computadoras y la IA 'aprenden' a reconocer características y clasificar cosas, igual que tú buscarás el patrón secreto.")

    st.markdown("---")

    # Inicialización del estado de la sesión para el juego de patrones
    # Asegúrate de que el estado inicial de current_pattern sea None y que se actualice al iniciar ronda.
    if "patrones_step" not in st.session_state:
        st.session_state.patrones_step = "inicio"
        st.session_state.current_pattern = None # Se establecerá en start_new_round
        st.session_state.displayed_elements = []
        st.session_state.player_selections = {}
        st.session_state.feedback_message = ""
        st.session_state.correct_guesses = 0
        st.session_state.total_correct_in_pattern = 0
        st.session_state.ml_model = None
        st.session_state.model_features = []

    if st.session_state.patrones_step == "inicio":
        st.write("¿Estás listo para el desafío?")
        if st.button("¡Empezar a buscar patrones!", key="start_patterns_game"):
            start_new_round() # Esta función inicializa current_pattern
            st.rerun()

    elif st.session_state.patrones_step == "jugando":
        st.write("🔍 **Criterio Secreto:** ¡Encuentra los objetos que cumplen el patrón oculto!")
        
        cols = st.columns(4)
        for i, (element_name, element_props) in enumerate(st.session_state.displayed_elements):
            with cols[i % 4]:
                st.image(element_props["img_url"], caption=element_name.replace('_', ' ').title(), use_container_width=True)
                
                if element_name not in st.session_state.player_selections:
                    st.session_state.player_selections[element_name] = False

                st.session_state.player_selections[element_name] = st.checkbox(
                    "Seleccionar",
                    value=st.session_state.player_selections[element_name],
                    key=f"checkbox_{element_name}"
                )

        st.markdown("---")
        col_check, col_reveal = st.columns([0.7, 0.3])
        with col_check:
            if st.button("¡Revisar mis selecciones!", key="check_selections"):
                check_player_selections()
                st.rerun()
        with col_reveal:
            if st.button("¡Revelar el patrón!", key="reveal_pattern_btn"):
                # Aquí es donde se cambia el estado a "revelado"
                # Aseguramos que current_pattern ya está seteado desde start_new_round()
                # Si el juego se "recargara" de alguna forma directamente en esta fase sin pasar por "jugando",
                # current_pattern podría ser None. La siguiente verificación lo previene.
                st.session_state.patrones_step = "revelado"
                st.rerun()

    elif st.session_state.patrones_step == "revelado":
        # === SOLUCIÓN: Verificar si current_pattern no es None antes de usarlo ===
        if st.session_state.current_pattern is not None:
            st.success(f"🎉 ¡El patrón secreto era: **'{st.session_state.current_pattern['nombre']}'**!")
        else:
            # En caso de que se llegue aquí sin un patrón definido (raro, pero para evitar el error)
            st.error("¡Oops! Parece que el patrón no se cargó correctamente. ¡Intenta jugar de nuevo!")
            # Y forzamos un reinicio o vuelta al inicio para que el estado se limpie
            st.session_state.patrones_step = "inicio"
            st.rerun()
            return # Salir de la función para evitar más errores

        st.write("Aquí están los objetos que realmente cumplían el patrón:")
        
        cols = st.columns(4)
        correct_elements_found = []
        for i, (element_name, element_props) in enumerate(st.session_state.displayed_elements):
            is_correct = st.session_state.current_pattern["criterio"](element_props)
            with cols[i % 4]:
                st.image(element_props["img_url"], caption=element_name.replace('_', ' ').title(), use_container_width=True)
                if is_correct:
                    st.success("✅ ¡Correcto!")
                    correct_elements_found.append(element_name)
                else:
                    st.error("❌ Incorrecto")
        
        st.write(f"De los {st.session_state.total_correct_in_pattern} elementos que cumplían el patrón, seleccionaste correctamente {st.session_state.correct_guesses}.")

        st.markdown("### 🤖 Cómo lo 'aprendería' una Inteligencia Artificial:")
        st.write("Imagina que la IA es un detective que mira muchas, muchas fotos y busca pistas. Cuando la IA 'aprende' un patrón, en realidad está creando reglas, ¡como esta!")
        
        if st.session_state.ml_model and st.session_state.model_features:
            st.write(f"Para encontrar '{st.session_state.current_pattern['nombre']}', una IA podría usar estas 'reglas' que aprendió:")
            st.code(st.session_state.current_pattern['explicacion_ia'], language='text')
            
            st.write("Por ejemplo, si la IA ve un objeto...")
            
            example_correct_name = None
            for name, props in st.session_state.displayed_elements:
                if st.session_state.current_pattern["criterio"](props):
                    example_correct_name = name
                    break
            
            if example_correct_name:
                example_props = ELEMENTOS_DISPONIBLES[example_correct_name]
                st.write(f"- Si ve un **{example_correct_name.replace('_', ' ').title()}**: La IA miraría sus características y seguiría sus reglas internas. ¡Su regla le diría que **SÍ** cumple el patrón!")
                
            example_incorrect_name = None
            for name, props in st.session_state.displayed_elements:
                if not st.session_state.current_pattern["criterio"](props):
                    example_incorrect_name = name
                    break
            
            if example_incorrect_name:
                example_props_incorrect = ELEMENTOS_DISPONIBLES[example_incorrect_name]
                st.write(f"- Si ve un **{example_incorrect_name.replace('_', ' ').title()}**: La IA miraría sus características y seguiría otras reglas. ¡Su regla le diría que **NO** cumple el patrón!")
            
            st.markdown(f"La clave es que la IA usa las **características** de los objetos (como su tipo, color, forma) para decidir a qué grupo pertenecen, ¡igual que tú lo hiciste!")
        else:
            st.warning("No se pudo cargar el modelo de IA para una explicación detallada.")

        st.markdown("---")
        if st.button("Jugar otra vez", key="play_again_patterns"):
            start_new_round()
            st.rerun()

    st.markdown("---")
    st.info("💡 **Concepto aprendido:** ¡Acabas de hacer lo mismo que una Inteligencia Artificial! Cuando una IA 'aprende' a clasificar cosas, busca **patrones** y **características** en los datos. Por ejemplo, si le mostramos muchas fotos de perros y gatos, con el tiempo aprende a identificar las características únicas de cada uno para clasificarlos correctamente. ¡Es como ser un detective de datos!")
    st.markdown("Pulsa los botones de arriba para cambiar de juego.")

# --- Funciones de lógica del juego ---

def start_new_round():
    st.session_state.patrones_step = "jugando"
    
    selected_pattern = random.choice(PATRONES_SECRETOS)
    st.session_state.current_pattern = selected_pattern # <--- Asegura que se establezca aquí
    
    st.session_state.ml_model, st.session_state.model_features = train_pattern_model(selected_pattern["id"])

    all_elements_list = list(ELEMENTOS_DISPONIBLES.items())
    random.shuffle(all_elements_list)
    
    displayed_elements_temp = []
    correct_elements_in_display = 0
    
    for name, props in all_elements_list:
        displayed_elements_temp.append((name, props))
        if st.session_state.current_pattern["criterio"](props):
            correct_elements_in_display += 1
        if len(displayed_elements_temp) >= 8 and correct_elements_in_display >= 2:
            break
            
    if correct_elements_in_display < 2 or len(displayed_elements_temp) < 8:
        for _ in range(20):
            if len(displayed_elements_temp) >= 12 and correct_elements_in_display >= 2:
                break
            
            rand_name, rand_props = random.choice(list(ELEMENTOS_DISPONIBLES.items()))
            if (rand_name, rand_props) not in displayed_elements_temp:
                displayed_elements_temp.append((rand_name, rand_props))
                if st.session_state.current_pattern["criterio"](rand_props):
                    correct_elements_in_display += 1


    random.shuffle(displayed_elements_temp)

    st.session_state.displayed_elements = displayed_elements_temp
    st.session_state.player_selections = {name: False for name, _ in st.session_state.displayed_elements}
    st.session_state.feedback_message = ""
    st.session_state.correct_guesses = 0
    st.session_state.total_correct_in_pattern = sum(1 for name, props in st.session_state.displayed_elements if st.session_state.current_pattern["criterio"](props))


def check_player_selections():
    correct_selected = 0
    incorrect_selected = 0
    correct_total = 0
    
    for element_name, element_props in st.session_state.displayed_elements:
        is_truly_correct = st.session_state.current_pattern["criterio"](element_props)
        player_selected = st.session_state.player_selections[element_name]

        if is_truly_correct:
            correct_total += 1
            if player_selected:
                correct_selected += 1
        else:
            if player_selected:
                incorrect_selected += 1

    st.session_state.correct_guesses = correct_selected
    st.session_state.total_correct_in_pattern = correct_total

    if correct_selected == correct_total and incorrect_selected == 0:
        st.session_state.feedback_message = f"¡Excelente! ¡Has encontrado todos los {correct_selected} objetos que cumplen el patrón y no te equivocaste en ninguno! 🎉"
        st.session_state.patrones_step = "revelado"
    elif correct_selected > 0 or incorrect_selected > 0:
        st.session_state.feedback_message = f"¡Buen intento! Encontraste {correct_selected} objetos correctos. Cuidado, seleccionaste {incorrect_selected} que no cumplen el patrón. Sigue buscando... 👀"
    else:
        st.session_state.feedback_message = "Aún no has seleccionado ningún objeto que cumpla el patrón. ¡No te rindas!"