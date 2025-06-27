import streamlit as st
import random
import time
import os
from openai import OpenAI
from gtts import gTTS
from io import BytesIO
# from PIL import Image # No es estrictamente necesario si solo usas st.image con URL

# --- Configuración de la API de OpenAI ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai_api_key = None
    st.warning("Advertencia: La clave de API de OpenAI no está configurada en `secrets.toml`. Algunas funcionalidades de IA no estarán disponibles.")


client = OpenAI(api_key=openai_api_key) if openai_api_key else None


# --- Función para generar sugerencias de elementos con OpenAI ---
def generate_openai_suggestions(element_type, count=3, current_options=None):
    if not client:
        return ["Error: API de OpenAI no configurada."]

    system_prompt = f"""
    Eres un asistente creativo de historias para niños. Tu tarea es generar ideas originales y divertidas para {element_type}s.
    Proporciona {count} ideas únicas, separadas por un salto de línea.
    No añadas numeración ni ninguna otra frase, solo las ideas.
    Asegúrate de que sean adecuadas para niños de 6 a 10 años.
    """
    if current_options:
        # Pasa las opciones actuales de forma que la IA las entienda como "evitar"
        system_prompt += f"\nAdemás, no sugieras estos elementos que ya existen: {', '.join(current_options)}."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Dame {count} ideas de {element_type}s."}
            ],
            temperature=0.9, # Alta temperatura para ideas creativas
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        suggestions_raw = response.choices[0].message.content
        # Limpiar y dividir las sugerencias
        suggestions = [s.strip() for s in suggestions_raw.split('\n') if s.strip()]
        return suggestions[:count] # Asegurarse de devolver el número correcto
    except Exception as e:
        st.error(f"Error al generar sugerencias de IA para {element_type}: {e}")
        return [f"Error al cargar sugerencias. ({e})"]


# --- Función para generar historia con OpenAI ---
def generate_openai_story_segment(base_story_context, character, place, magic_item, additional_ideas, story_genre, segment_type="inicio"):
    if not client:
        return "Lo siento, la API de OpenAI no está configurada. No puedo generar historias mágicas."

    # Definir el rol principal de la IA
    system_role = f"""
    Eres un narrador mágico para niños de 6 a 10 años, experto en crear historias divertidas, emocionantes y muy originales.
    Tu objetivo es ser súper creativo y mantener un tono {story_genre} en toda la historia.
    Siempre sé positivo, nunca aburrido, y evita finales tristes o complejos.
    """

    if segment_type == "inicio":
        user_prompt = f"""
        Quiero una historia nueva y {story_genre}.
        Crea el inicio de una aventura con estos elementos:
        - Personaje principal: {character}
        - Lugar de la aventura: {place}
        - Objeto mágico: {magic_item}
        Ideas adicionales del niño (si las hay): "{additional_ideas if additional_ideas else 'Ninguna idea adicional.'}"

        Empieza la historia, presenta los elementos y un pequeño problema o misterio.
        Termina con una pregunta que ofrezca DOS OPCIONES claras para que el niño elija cómo sigue la historia.
        Formato de las opciones:
        Opción A: [Breve descripción de la opción A]
        Opción B: [Breve descripción de la opción B]

        La historia debe ser muy atractiva para niños.
        """
    elif segment_type == "continuacion":
        user_prompt = f"""
        La historia hasta ahora es:
        "{base_story_context}"

        Ahora, el niño ha elegido continuar la historia de esta manera: "{additional_ideas}"
        Continúa la historia de forma {story_genre}, incorporando la elección del niño de manera emocionante.
        **Es crucial que escribas un párrafo o dos de la historia antes de presentar las nuevas opciones.**
        Asegúrate de mantener la coherencia con el personaje principal ({character}), el lugar ({place}) y el objeto mágico ({magic_item}).
        La historia debe avanzar y terminar con un nuevo dilema y DOS OPCIONES claras para que el niño decida el siguiente paso.
        Formato de las opciones:
        Opción A: [Breve descripción de la opción A]
        Opción B: [Breve descripción de la opción B]
        """
    elif segment_type == "final":
        user_prompt = f"""
        La historia hasta ahora es:
        "{base_story_context}"

        El niño ha elegido el camino final: "{additional_ideas}"
        Escribe un final divertido, positivo y memorable para esta aventura {story_genre}.
        Asegúrate de que el personaje principal ({character}) y el objeto mágico ({magic_item}) tengan un papel importante.
        La historia debe terminar con una conclusión feliz o que invite a la imaginación, como "¡Y la aventura continuó!" o "Vivieron muchas más aventuras...".
        """
    else:
        return "Tipo de segmento no reconocido."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=400,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"¡Oh no! Hubo un error al crear la historia mágica: {e}. Asegúrate de que tu clave de API sea válida y tengas créditos.")
        return f"Lo siento, no pude generar esta parte de la historia. Error: {e}"


# --- Función para convertir texto a audio ---
def text_to_audio(text, lang='es'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Error al convertir texto a audio: {e}. Revisa tu conexión a internet o el texto.")
        return None


# --- Función principal del juego ---
def run_game():
    st.subheader("✍️ El Creador de Historias Mágico con IA")
    st.write("¡Desata tu imaginación! Elige un personaje, un lugar, un objeto, un estilo... ¡y nuestra IA mágica creará una historia contigo! ¡Podrás decidir cómo sigue la aventura y escucharla!")
    st.info("Objetivo: Experimentar cómo la Inteligencia Artificial puede escribir historias de forma creativa y cómo tus decisiones cambian el rumbo de la aventura.")

    st.markdown("---")

    # --- Elementos predefinidos (referencia para selectbox) ---
    personajes_base = [
        "un valiente caballero", "una intrépida exploradora", "un pequeño dragón amistoso",
        "una sabia bruja del bosque", "un robot curioso", "una princesa que le gustaba programar",
        "un mago despistado", "un elfo con poderes especiales"
    ]

    lugares_base = [
        "en un castillo encantado", "en el fondo del mar, en una ciudad secreta",
        "en el espacio exterior, en un planeta de caramelos", "en un bosque susurrante",
        "en una cueva brillante llena de cristales", "en lo alto de una montaña nevada",
        "en una escuela de magia y tecnología", "en un pueblo donde los animales hablaban"
    ]

    objetos_magicos_base = [
        "una brújula que señala la verdad", "un mapa que cambia solo", "una espada de luz",
        "un espejo que muestra el futuro", "un libro que habla", "una mochila que saca cualquier cosa",
        "un anillo que te hace invisible", "un amuleto que te da súper fuerza"
    ]

    generos_historia = [
        "aventurera", "misteriosa", "graciosa", "de fantasía", "de ciencia ficción", "de amistad"
    ]

    # --- Inicialización del estado de la sesión ---
    # Esto asegura que el estado se inicialice solo una vez al inicio.
    if 'story_parts' not in st.session_state:
        st.session_state.story_parts = []
        st.session_state.current_story_full_text = ""
        st.session_state.story_step = "inicio"
        st.session_state.choices_available = []
        st.session_state.generated_audio = None
        
        # Estado para la Opción C
        st.session_state.show_option_c_input = False
        st.session_state.option_c_text = ""
        
        # Inicializar las últimas selecciones del usuario a los primeros elementos de las listas base
        st.session_state.last_character_choice = personajes_base[0]
        st.session_state.last_place_choice = lugares_base[0]
        st.session_state.last_item_choice = objetos_magicos_base[0]
        st.session_state.last_ideas_choice = ""
        st.session_state.last_genre_choice = generos_historia[0]
        
        # Inicializar las listas de sugerencias de la IA para evitar el AttributeError
        st.session_state.suggested_characters = []
        st.session_state.suggested_places = []
        st.session_state.suggested_items = []
        st.session_state.generated_image_url = None # Inicializamos la URL de la imagen
        
    # --- Widgets de selección inicial ---
    st.markdown("### Elige los ingredientes principales de tu historia:")
    
    # Personaje
    char_col1, char_col2 = st.columns([0.7, 0.3])
    with char_col1:
        current_char_options = personajes_base + st.session_state.suggested_characters
        # Asegurarse de que la opción seleccionada previamente esté en la lista actual, si no, usa el primer elemento
        try:
            default_index_char = current_char_options.index(st.session_state.last_character_choice)
        except ValueError:
            default_index_char = 0 # Si la última opción no está, usa la primera
        selected_character = st.selectbox("1. Personaje:", current_char_options, key="story_char",
                                        index=default_index_char)
    with char_col2:
        st.markdown("<br>", unsafe_allow_html=True) # Espacio para alinear
        if st.button("🧙‍♀️ Sugerir Personaje IA", key="suggest_char_btn"):
            if not client:
                st.error("Por favor, configura tu clave de API de OpenAI en `.streamlit/secrets.toml` para usar las sugerencias.")
            else:
                with st.spinner("La IA piensa en personajes..."):
                    st.session_state.suggested_characters = generate_openai_suggestions("personaje", 3, personajes_base + st.session_state.suggested_characters)
                st.rerun()

    # Lugar
    place_col1, place_col2 = st.columns([0.7, 0.3])
    with place_col1:
        current_place_options = lugares_base + st.session_state.suggested_places
        try:
            default_index_place = current_place_options.index(st.session_state.last_place_choice)
        except ValueError:
            default_index_place = 0
        selected_place = st.selectbox("2. Lugar:", current_place_options, key="story_place",
                                    index=default_index_place)
    with place_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Sugerir Lugar IA", key="suggest_place_btn"):
            if not client:
                st.error("Por favor, configura tu clave de API de OpenAI en `.streamlit/secrets.toml` para usar las sugerencias.")
            else:
                with st.spinner("La IA piensa en lugares..."):
                    st.session_state.suggested_places = generate_openai_suggestions("lugar", 3, lugares_base + st.session_state.suggested_places)
                st.rerun()

    # Objeto Mágico
    item_col1, item_col2 = st.columns([0.7, 0.3])
    with item_col1:
        current_item_options = objetos_magicos_base + st.session_state.suggested_items
        try:
            default_index_item = current_item_options.index(st.session_state.last_item_choice)
        except ValueError:
            default_index_item = 0
        selected_magic_item = st.selectbox("3. Objeto Mágico:", current_item_options, key="story_item",
                                        index=default_index_item)
    with item_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Sugerir Objeto IA", key="suggest_item_btn"):
            if not client:
                st.error("Por favor, configura tu clave de API de OpenAI en `.streamlit/secrets.toml` para usar las sugerencias.")
            else:
                with st.spinner("La IA piensa en objetos mágicos..."):
                    st.session_state.suggested_items = generate_openai_suggestions("objeto mágico", 3, objetos_magicos_base + st.session_state.suggested_items)
                st.rerun()


    st.markdown("---")
    
    col_genre, col_reroll = st.columns([0.7, 0.3])
    with col_genre:
        selected_genre = st.selectbox("4. ¿De qué tipo de historia quieres que sea?", generos_historia, key="story_genre",
                                    index=generos_historia.index(st.session_state.last_genre_choice) if st.session_state.last_genre_choice in generos_historia else 0)
    with col_reroll:
        st.markdown("<br>", unsafe_allow_html=True) # Espacio para alinear
        if st.button("🔄 Volver a Empezar Historia", key="reset_story_btn"):
            # Reiniciar todo el estado
            st.session_state.story_parts = []
            st.session_state.current_story_full_text = ""
            st.session_state.story_step = "inicio"
            st.session_state.choices_available = []
            st.session_state.generated_audio = None
            
            # Reiniciar estado de Opción C
            st.session_state.show_option_c_input = False
            st.session_state.option_c_text = ""
            
            # Reiniciar selecciones a valores predeterminados o vacíos
            st.session_state.last_character_choice = personajes_base[0]
            st.session_state.last_place_choice = lugares_base[0]
            st.session_state.last_item_choice = objetos_magicos_base[0]
            st.session_state.last_ideas_choice = ""
            st.session_state.last_genre_choice = generos_historia[0] # Reinicia al primer género
            
            # Limpiar sugerencias de IA
            st.session_state.suggested_characters = []
            st.session_state.suggested_places = []
            st.session_state.suggested_items = []
            st.session_state.generated_image_url = None # Limpiar la URL de la imagen
            st.rerun()


    # --- Input adicional del niño para ideas ---
    st.markdown("### ¿Tienes alguna idea extra para inspirar a la IA?")
    additional_ideas_input = st.text_area(
        "Por ejemplo: 'que aparezca un unicornio', 'que vuele por el cielo', 'que resuelvan un misterio con un dragón'.",
        value=st.session_state.last_ideas_choice,
        key="additional_ideas_input",
        height=70
    )

    st.markdown("---")

    # --- Generación inicial de la historia o continuación ---
    if st.session_state.story_step == "inicio":
        if st.button("✨ ¡Crea el INICIO de mi Historia Mágica! ✨", key="generate_first_part_button"):
            if not client:
                st.error("Por favor, configura tu clave de API de OpenAI en `.streamlit/secrets.toml` para usar este juego.")
            else:
                # Guardar las selecciones actuales
                st.session_state.last_character_choice = selected_character
                st.session_state.last_place_choice = selected_place
                st.session_state.last_item_choice = selected_magic_item
                st.session_state.last_ideas_choice = additional_ideas_input
                st.session_state.last_genre_choice = selected_genre

                st.write("La IA está desatando su magia... ¡Un momento por favor!")
                with st.spinner("La IA está escribiendo el inicio de tu aventura..."):
                    story_response = generate_openai_story_segment(
                        "", # No hay contexto previo
                        selected_character,
                        selected_place,
                        selected_magic_item,
                        additional_ideas_input,
                        selected_genre,
                        segment_type="inicio"
                    )
                
                # --- Eliminadas las líneas de depuración para el segmento inicial ---
                
                # Separar la historia de las opciones
                if "Opción A:" in story_response and "Opción B:" in story_response:
                    parts = story_response.split("Opción A:", 1)
                    story_text_part_temp = parts[0].strip()
                    options_part_temp = "Opción A:" + parts[1].strip()

                    # Extraer opciones para los botones
                    option_a_text_temp = options_part_temp.split("Opción A:", 1)[1].split("Opción B:", 1)[0].strip()
                    option_b_text_temp = options_part_temp.split("Opción B:", 1)[1].strip()
                    
                    # Actualizar estado de la sesión (aquí es donde se persisten los datos)
                    st.session_state.story_parts = [story_text_part_temp] # Reinicia para la nueva historia
                    st.session_state.current_story_full_text = story_text_part_temp
                    # AÑADIR Opción C aquí
                    st.session_state.choices_available = [
                        f"Opción A: {option_a_text_temp}",
                        f"Opción B: {option_b_text_temp}",
                        "Opción C: ¡Inventa tú el siguiente paso!" # Nuestra nueva opción
                    ]
                    st.session_state.story_step = "continuacion"
                else:
                    st.write("La historia ha llegado a un punto de finalización.")
                    st.session_state.story_parts = [story_response] # Reinicia y asume que es el final si no hay opciones
                    st.session_state.current_story_full_text = story_response
                    st.session_state.story_step = "finalizada"
                    st.session_state.choices_available = [] # No hay opciones
                    st.session_state.show_option_c_input = False # Asegurarse de que no se muestre el input de la Opción C

                # --- Eliminadas las líneas de depuración para el segmento inicial ---

                st.session_state.generated_audio = text_to_audio(st.session_state.current_story_full_text) # Audio de todo el segmento
                st.rerun()
    
    # --- Mostrar historia actual y opciones ---
    # Esta sección se ejecuta en cada rerun.
    # Si st.session_state.current_story_full_text tiene contenido, se mostrará.
    if st.session_state.current_story_full_text: # La condición aquí es mejor con current_story_full_text
        st.markdown("### 📖 Tu Historia hasta ahora:")
        st.text_area(
            "Lee aquí tu aventura:",
            value=st.session_state.current_story_full_text,
            height=300,
            key="current_story_display"
        )
        if st.session_state.generated_audio:
            st.markdown("### 🎧 Escucha este segmento:")
            st.audio(st.session_state.generated_audio, format="audio/mp3")

        st.markdown("---")

        if st.session_state.story_step == "continuacion":
            st.markdown("### ¿Cómo quieres que continúe la historia?")
            
            # Usar columnas para las opciones A, B y C (Opción C en una columna propia o repartida)
            choice_col1, choice_col2, choice_col3 = st.columns(3) # Ahora 3 columnas
            
            with choice_col1:
                if st.button(st.session_state.choices_available[0], key="choice_A_button"):
                    # Restablecer estado de Opción C si se elige A o B
                    st.session_state.show_option_c_input = False
                    st.session_state.option_c_text = ""
                    process_choice(st.session_state.choices_available[0], "Opción A")

            with choice_col2:
                if st.button(st.session_state.choices_available[1], key="choice_B_button"):
                    # Restablecer estado de Opción C si se elige A o B
                    st.session_state.show_option_c_input = False
                    st.session_state.option_c_text = ""
                    process_choice(st.session_state.choices_available[1], "Opción B")

            with choice_col3:
                # Botón para activar el input de la Opción C
                if st.button(st.session_state.choices_available[2], key="choice_C_button"):
                    st.session_state.show_option_c_input = True
                    st.rerun() # Necesitamos un rerun para que el input de texto aparezca/desaparezca

            # Mostrar el input de texto para la Opción C si el botón fue presionado
            if st.session_state.show_option_c_input:
                st.markdown("---")
                st.write("### ¡Cuéntanos tu idea para el siguiente paso!")
                st.session_state.option_c_text = st.text_input(
                    "Escribe aquí tu idea (Ej: 'Quiero que encuentren un mapa del tesoro', 'Que aparezca un duende travieso')",
                    value=st.session_state.option_c_text,
                    key="option_c_text_input"
                )
                if st.button("🚀 Continuar con mi idea", key="process_option_c_button"):
                    if st.session_state.option_c_text.strip():
                        # Procesar la idea del usuario como la elección
                        process_choice(st.session_state.option_c_text, "Opción C")
                    else:
                        st.warning("Por favor, escribe tu idea antes de continuar.")
                        
        elif st.session_state.story_step == "finalizada":
            st.success("¡Tu historia mágica ha llegado a un final increíble!")
            st.balloons()
            
            # Generar la imagen al final de la historia
            if st.session_state.generated_image_url is None: # Solo generar si no se ha generado ya
                if client:
                    st.toast("La IA está creando una imagen de tu historia...", icon="🖼️")
                    with st.spinner("Generando imagen..."):
                        # Crear un prompt más conciso para la imagen
                        image_prompt = (
                            f"Una ilustración colorida y caprichosa para niños, estilo libro de cuentos, "
                            f"con {st.session_state.last_character_choice} en {st.session_state.last_place_choice}, "
                            f"usando {st.session_state.last_item_choice}, en una historia de tipo {st.session_state.last_genre_choice}. "
                            f"Representa un momento clave del final de la historia. Asegúrate de que sea muy positiva y atractiva para niños."
                        )
                        
                        try:
                            image_results = client.images.generate(
                                model="dall-e-3", # o "dall-e-2" si prefieres y tienes acceso
                                prompt=image_prompt,
                                n=1,
                                size="512x512" # Puedes ajustar el tamaño: "256x256", "512x512", "1024x1024"
                            )
                            st.session_state.generated_image_url = image_results.data[0].url
                            st.toast("¡Imagen de la historia creada!", icon="✅")
                            st.rerun() # Volver a ejecutar para mostrar la imagen inmediatamente
                        except Exception as e:
                            st.error(f"Error al generar la imagen: {e}. Asegúrate de que tu clave de API sea válida y tengas créditos para DALL-E.")
                            st.session_state.generated_image_url = None
                            st.toast("No se pudo crear la imagen.", icon="❌")
                
                if st.session_state.generated_image_url:
                    st.markdown("### 🖼️ ¡Mira la imagen de tu historia!")
                    st.image(st.session_state.generated_image_url, caption="Imagen generada por IA", use_column_width=True)
                else:
                    st.write("No se pudo generar una imagen para esta historia.")

            st.write("Puedes presionar '🔄 Volver a Empezar Historia' para crear una nueva aventura.")


    st.markdown("---")
    st.info("💡 **Concepto aprendido:** ¡Acabas de ver cómo la Inteligencia Artificial puede escribir historias interactivas! Esto se llama **Generación de Lenguaje Natural (NLG)** y también la **toma de decisiones basada en tus inputs**. La IA no 'piensa' como nosotros, pero aprende patrones y estructuras del lenguaje para crear texto que suena muy natural y creativo. ¡Cuanto mejor le des las ideas y elijas, mejor será la historia!")
    st.markdown("Pulsa los botones de arriba para cambiar de juego.")


# --- Nueva función para procesar la elección (A, B o C) y generar el siguiente segmento ---
def process_choice(selected_choice_text, choice_type):
    # Desactivar el input de la Opción C si no se eligió C
    if choice_type != "Opción C":
        st.session_state.show_option_c_input = False
        st.session_state.option_c_text = ""

    st.write(f"La IA está creando la siguiente parte...")
    with st.spinner("Continuando tu aventura..."):
        next_part_raw = generate_openai_story_segment(
            st.session_state.current_story_full_text,
            st.session_state.last_character_choice,
            st.session_state.last_place_choice,
            st.session_state.last_item_choice,
            selected_choice_text, # La elección del niño es ahora una "idea adicional" para la IA
            st.session_state.last_genre_choice,
            segment_type="continuacion"
        )
        
    # --- Eliminadas las líneas de depuración para cualquier opción ---
    
    # Procesar la respuesta para separar historia de opciones (si las hay)
    if "Opción A:" in next_part_raw and "Opción B:" in next_part_raw:
        parts = next_part_raw.split("Opción A:", 1)
        story_text_part = parts[0].strip()
        options_part = "Opción A:" + parts[1].strip()

        # Actualizar la historia completa y las partes
        st.session_state.story_parts.append(story_text_part)
        st.session_state.current_story_full_text += "\n\n" + story_text_part

        option_a_text = options_part.split("Opción A:", 1)[1].split("Opción B:", 1)[0].strip()
        option_b_text = options_part.split("Opción B:", 1)[1].strip()
        # Añadir Opción C al final de la lista de opciones
        st.session_state.choices_available = [
            f"Opción A: {option_a_text}",
            f"Opción B: {option_b_text}",
            "Opción C: ¡Inventa tú el siguiente paso!"
        ]
        st.session_state.story_step = "continuacion" # Seguimos en continuación
    else: # Es el final si la IA no da más opciones
        st.write("La historia ha llegado a un punto de finalización.")
        st.session_state.story_parts.append(next_part_raw)
        st.session_state.current_story_full_text += "\n\n" + next_part_raw
        st.session_state.story_step = "finalizada"
        st.session_state.choices_available = [] # No hay más opciones
        st.session_state.show_option_c_input = False # Asegurarse de que no se muestre el input de la Opción C

    st.session_state.generated_audio = text_to_audio(next_part_raw) # Genera audio para el NUEVO segmento
    
    # --- Eliminadas las líneas de depuración para cualquier opción ---

    st.rerun() # Siempre hacer rerun después de procesar una elección


# Llama a la función principal para ejecutar el juego
if __name__ == "__main__":
    run_game()