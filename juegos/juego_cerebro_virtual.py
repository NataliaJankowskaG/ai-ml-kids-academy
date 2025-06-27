import streamlit as st
import random
import os
import time

def run_game():
    st.subheader("🧠 Construye tu Cerebro Virtual: ¡Aprende Formas!")
    st.write("¡Ayuda a este cerebro virtual a aprender a reconocer formas geométricas! Cada vez que le digas si acierta o se equivoca, ¡aprenderá un poquito más!")
    st.info("Objetivo: Enseña a tu Cerebro Virtual a distinguir entre diferentes formas.")

    st.markdown("---")

    # --- Definir la ruta base para las imágenes de formas geométricas ---
    current_dir = os.path.dirname(__file__)
    # project_root es 'C:\Users\iblan\Desktop\Proyecto_final'
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    # La ruta base para las imágenes del juego (AJUSTADO A TU RUTA ESPECÍFICA)
    image_base_path = os.path.join(project_root, "assets", "imagenes") # <--- CAMBIO AQUÍ: "imagenes" en lugar de "images" o "formas_geometricas"


    # --- Datos del juego (objetos e imágenes) - ¡Ahora con formas geométricas! ---
    objetos_entrenamiento = [
        {"nombre": "Círculo", "tipo": "Forma Redonda", "imagen_path": os.path.join(image_base_path, "circulo.png")},
        {"nombre": "Cuadrado", "tipo": "Forma Recta", "imagen_path": os.path.join(image_base_path, "cuadrado.png")},
        {"nombre": "Triángulo", "tipo": "Forma Recta", "imagen_path": os.path.join(image_base_path, "triangulo.png")},
        {"nombre": "Rectangulo", "tipo": "Forma Recta", "imagen_path": os.path.join(image_base_path, "rectangulo.png")},
        {"nombre": "Estrella", "tipo": "Forma Compleja", "imagen_path": os.path.join(image_base_path, "estrella.png")},
        {"nombre": "Hexágono", "tipo": "Forma Recta", "imagen_path": os.path.join(image_base_path, "hexagono.png")},
        {"nombre": "Óvalo", "tipo": "Forma Redonda", "imagen_path": os.path.join(image_base_path, "ovalo.png")},
        {"nombre": "Gota", "tipo": "Forma Redonda", "imagen_path": os.path.join(image_base_path, "gota.png")},
        {"nombre": "Diamante", "tipo": "Forma Compleja", "imagen_path": os.path.join(image_base_path, "diamante.png")},
        {"nombre": "ZigZag", "tipo": "Forma Compleja", "imagen_path": os.path.join(image_base_path, "zigzag.png")}
        # Asegúrate de que los nombres de los archivos aquí (ej. "circulo.png")
        # coincidan exactamente con los nombres de tus archivos descargados.
    ]

    # --- Resto del código del juego (se mantiene igual que la última versión que te di) ---
    # ... (Inicialización del estado, lógica de predicción, botones, estadísticas, etc.)
    tipos_formas = ["Forma Redonda", "Forma Recta", "Forma Compleja"]

    if 'brain_knowledge_formas' not in st.session_state:
        st.session_state.brain_knowledge_formas = {
            tipo: 0 for tipo in tipos_formas
        }
        st.session_state.brain_knowledge_formas["total_intentos"] = 0
        st.session_state.brain_knowledge_formas["aciertos_cerebro"] = 0
        st.session_state.current_object_index_formas = random.randint(0, len(objetos_entrenamiento) - 1)
        st.session_state.last_prediction_correct_formas = None

    current_object = objetos_entrenamiento[st.session_state.current_object_index_formas]

    st.write(f"Aquí tienes una forma. ¡Ayuda a tu cerebro a aprender!")
    st.image(current_object["imagen_path"], caption=f"¿Qué es esto? (Pista: Es una {current_object['tipo']})", width=300)
    
    knowledge_display = "Tu Cerebro Virtual ha 'visto': "
    for tipo in tipos_formas:
        knowledge_display += f"{st.session_state.brain_knowledge_formas[tipo]} {tipo}s, "
    st.markdown(knowledge_display[:-2] + ".")

    st.markdown("---")

    brain_prediction = "Desconocido"
    if st.session_state.brain_knowledge_formas["total_intentos"] < 5:
        brain_prediction = random.choice(tipos_formas)
    else:
        max_knowledge_type = ""
        max_knowledge_count = -1
        
        for tipo in tipos_formas:
            if st.session_state.brain_knowledge_formas[tipo] > max_knowledge_count:
                max_knowledge_count = st.session_state.brain_knowledge_formas[tipo]
                max_knowledge_type = tipo
            elif st.session_state.brain_knowledge_formas[tipo] == max_knowledge_count and max_knowledge_type:
                if random.random() > 0.5:
                    max_knowledge_type = tipo

        if max_knowledge_type:
            brain_prediction = max_knowledge_type
        else:
            brain_prediction = random.choice(tipos_formas)

    st.write(f"Tu Cerebro Virtual piensa que esto es... ¡una **{brain_prediction}**!")

    col_pred_1, col_pred_2 = st.columns(2)
    with col_pred_1:
        if st.button("¡Sí, acertó!", key="brain_correct_formas"):
            st.session_state.brain_knowledge_formas["total_intentos"] += 1
            if brain_prediction == current_object["tipo"]:
                st.session_state.brain_knowledge_formas["aciertos_cerebro"] += 1
                st.session_state.brain_knowledge_formas[current_object["tipo"]] += 1
                st.success("¡Excelente! Tu Cerebro Virtual está aprendiendo muy bien.")
                st.session_state.last_prediction_correct_formas = True
            else:
                st.error("¡Ups! Parece que el cerebro se equivocó, pero tú pensaste que acertó. Revisa la pista.")
                st.session_state.last_prediction_correct_formas = False
            st.session_state.current_object_index_formas = random.randint(0, len(objetos_entrenamiento) - 1)
            st.rerun()

    with col_pred_2:
        if st.button("¡No, se equivocó!", key="brain_incorrect_formas"):
            st.session_state.brain_knowledge_formas["total_intentos"] += 1
            if brain_prediction != current_object["tipo"]:
                st.session_state.brain_knowledge_formas[current_object["tipo"]] += 1
                st.success("¡Bien hecho! Le has enseñado al Cerebro Virtual la respuesta correcta. ¡Ahora es más listo!")
                st.session_state.last_prediction_correct_formas = False
            else:
                st.error("¡Oh no! El cerebro había acertado, pero tú le dijiste que no. ¡Cuidado, maestro de formas!")
                st.session_state.last_prediction_correct_formas = True
            st.session_state.current_object_index_formas = random.randint(0, len(objetos_entrenamiento) - 1)
            st.rerun()

    st.markdown("---")

    if st.session_state.brain_knowledge_formas["total_intentos"] > 0:
        precision = (st.session_state.brain_knowledge_formas["aciertos_cerebro"] / st.session_state.brain_knowledge_formas["total_intentos"]) * 100
        st.write(f"Tu Cerebro Virtual ha hecho **{st.session_state.brain_knowledge_formas['total_intentos']} intentos**.")
        st.write(f"Ha acertado **{st.session_state.brain_knowledge_formas['aciertos_cerebro']} veces**.")
        st.write(f"Su precisión actual es del **{precision:.2f}%**.")
        
        if precision >= 80:
            st.balloons()
            st.success("¡Tu Cerebro Virtual es un experto en formas! ¡Sigue entrenándolo!")
        elif precision >= 50:
            st.info("Tu Cerebro Virtual está mejorando en el reconocimiento de formas. ¡Sigue enseñándole!")
        else:
            st.warning("Tu Cerebro Virtual aún necesita más entrenamiento para reconocer formas. ¡No te rindas!")


    st.markdown("---")
    st.info("💡 **Concepto aprendido:** Este juego te muestra cómo los sistemas de IA aprenden a **clasificar** objetos. Al darle ejemplos de formas y decir si acierta, ayudas al 'cerebro' a reconocer patrones. ¡Esto es el **reconocimiento de patrones** y la base de muchas cosas que hace la IA!")
    st.markdown("Pulsa los botones de arriba para cambiar de juego.")