import streamlit as st
import pandas as pd
import faiss
import joblib
import os
import socket
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- CONSTANTES Y RUTAS ---
FAISS_FILE = "movie_embeddings.faiss"
DATA_FILE = "movies_preprocessed.pkl"
SCALER_FILE = "scaler.joblib"

# --- DETECCIÓN DE ENTORNO ---
def es_local():
    is_cloud = os.environ.get("STREAMLIT_RUNTIME_ENV", False)
    return not is_cloud and "streamlit" not in socket.gethostname().lower()

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Movie AI Recommender", 
    page_icon=":material/movie:", 
    layout="centered"
)

# --- CARGA DE RECURSOS (CACHED) ---
@st.cache_resource
def load_resources():
    movies_df = pd.read_pickle(DATA_FILE)
    index = faiss.read_index(FAISS_FILE)
    scaler = joblib.load(SCALER_FILE)
    model_st = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return movies_df, index, scaler, model_st

try:
    movies_df, index, scaler, model_st = load_resources()
except Exception as e:
    st.error(f"Error cargando datos: {e}", icon=":material/database_alert:")
    st.stop()

# --- ESTADO DE SESIÓN ---
if "api_autenticada" not in st.session_state:
    st.session_state.api_autenticada = False
if "ultima_respuesta" not in st.session_state:
    st.session_state.ultima_respuesta = None

def configurar_api():
    st.sidebar.header("Configuración de Sistema", divider="gray")
    
    st.sidebar.markdown("### Backend Engine")
    key_input = st.sidebar.text_input(
        "Ingresar API Key Personal", 
        type="password", 
        placeholder="API Key...",
        help="Si el sistema no responde, puedes usar tu propia clave de Google AI Studio."
    )

    if st.sidebar.button("Validar API Key", use_container_width=True):
        if key_input:
            st.session_state.api_autenticada = True
            st.toast("Usando API Key personal", icon=":material/check_circle:")
            st.rerun()

    selected_model = "gemma-3-27b-it" 
    
    if st.session_state.api_autenticada and key_input:
        genai.configure(api_key=key_input)
        st.sidebar.success("Modo: API Key Personal", icon=":material/shield_person:")
        selected_model = st.sidebar.selectbox(
            "Seleccionar Modelo", 
            [ "gemini-2.5-flash", "gemini-3-flash", "gemini-3-pro", "gemma-3-12b-it", "gemma-3-27b-it"]
        )
        st.sidebar.markdown(f"""
            <div style="font-size: 0.8rem; color: gray; margin-top: 15px;">
                Límites sujetos a tu cuota en <a href="https://aistudio.google.com/app/plan_management" target="_blank" style="color: #007BFF; text-decoration: none;">Google AI Studio</a>.<br>
                Consulta tu consumo <a href="https://aistudio.google.com/app/usage" target="_blank" style="color: #007BFF; text-decoration: none;">aquí</a>.
            </div>
            """, unsafe_allow_html=True)
        return genai.GenerativeModel(selected_model)
    else:
        # Intenta usar la Key global de los secrets sin límites de software
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            modo_label = "Modo: API de Cortesía" if es_local() else "Modo: Soporte de Sistema"
            st.sidebar.info(modo_label, icon=":material/settings_suggest:")
            return genai.GenerativeModel(selected_model)
        else:
            st.sidebar.error("Configura una API Key para comenzar.", icon=":material/key_off:")
            return None

# --- LÓGICA DE BÚSQUEDA ---
def buscar_peliculas(query, top_k=5):
    query_embedding = model_st.encode([query])
    query_scaled = scaler.transform(query_embedding)
    distances, indices = index.search(query_scaled, top_k)
    return movies_df.iloc[indices[0]]

# --- INTERFAZ PRINCIPAL ---
st.title(":material/smart_toy: Movie AI Recommender")
st.subheader("Búsqueda Semántica de Cine", divider="blue")

model_gemini = configurar_api()

user_query = st.text_input(
    "Busca películas describiendo lo que sientes o quieres ver", 
    placeholder="Ejemplo: Peliculas que transcurran en la selva",
    max_chars=150
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    btn_search = st.button("Analizar", type="primary", use_container_width=True)

if btn_search:
    if not model_gemini:
        st.error("No hay una API Key configurada.", icon=":material/lock:")
    elif user_query:
        with st.spinner("Analizando preferencias..."):
            try:
                recs = buscar_peliculas(user_query)
                contexto = "\n".join([
                    f"- {row['CleanTitle']} ({row['Year']}): {row['Genres']}." 
                    for _, row in recs.iterrows()
                ])
                
                prompt = f"Usuario busca: {user_query}\nCandidatos:\n{contexto}\nRecomienda brevemente las mejores opciones en español."
                response = model_gemini.generate_content(prompt)
                
                st.session_state.ultima_respuesta = response.text
                st.rerun()
                    
            except Exception as e:
                st.error(f"Error en la consulta: {e}", icon=":material/emergency_home:")
                st.info("Si el error persiste, es posible que la cuota de la API se haya agotado. Intenta cargando tu propia API Key en el menú lateral.")

if st.session_state.ultima_respuesta:
    st.markdown("---")
    st.markdown("### :material/recommend: Recomendación")
    st.success(st.session_state.ultima_respuesta)

st.divider()
# --- SECCIÓN EXPLICATIVA (METODOLOGÍA) ---
with st.expander("📂 Documentación Técnica: ¿Cómo funciona este Recomendador?", expanded=False):
    st.markdown("### Arquitectura del Sistema (Pipeline RAG)")
    st.info("Este proyecto implementa una arquitectura de **Generación Aumentada por Recuperación (RAG)**. A continuación, el detalle del flujo de datos:")

    tab1, tab2, tab3 = st.tabs(["1. Entrada & Preproceso", "2. Vectorización (Embeddings)", "3. Inferencia AI"])

    with tab1:
        st.markdown("#### 📥 Entrada y Preparación de Datos")
        st.write("""
        1. **Dataset Original:** Partimos de un archivo CSV con miles de películas (títulos, años, géneros y sinopsis).
        2. **Limpieza (Preprocessing):** - Eliminamos registros duplicados o con valores nulos.
            - Normalizamos el texto (pasamos a minúsculas, quitamos caracteres especiales).
            - **Feature Engineering:** Creamos una columna 'Sopa de Metadatos' que combina Género + Sinopsis + Director para darle contexto semántico al modelo.
        """)
        st.code("# Ejemplo de preprocesamiento\ndf['metadata'] = df['genres'] + ' ' + df['overview']", language="python")

    with tab2:
        st.markdown("#### 🧠 Procesamiento: Embeddings & FAISS")
        st.write("""
        En lugar de buscar palabras exactas (Keyword Search), usamos **Búsqueda Semántica**:
        1. **Embeddings:** Convertimos las descripciones de las películas en vectores numéricos de alta dimensión (384 dimensiones) usando el modelo `SentenceTransformer`.
        2. **Indexación con FAISS:** - Utilizamos la librería **FAISS (Facebook AI Similarity Search)** para organizar estos vectores.
            - Esto permite realizar búsquedas de similitud de coseno en milisegundos, encontrando las películas 'más cercanas' al deseo del usuario en un espacio vectorial.
        """)
        

    with tab3:
        st.markdown("#### 🚀 Salida: Prompt Engineering & Gemini")
        st.write("""
        El paso final une la base de datos con la Inteligencia Artificial Generativa:
        1. **Recuperación:** FAISS nos devuelve las 5 películas más parecidas a la consulta.
        2. **Contextualización:** Creamos un **Prompt Dinámico** que incluye:
            - La consulta original del usuario.
            - La lista de las 5 películas encontradas en nuestro índice.
        3. **Generación:** Le pedimos a **Gemini/Gemma** que analice ese contexto y redacte una recomendación personalizada, explicando *por qué* esas películas encajan con lo que el usuario busca.
        """)
        st.success("**Resultado final:** Una respuesta coherente, basada en datos reales y procesada por IA.")

st.divider()

st.caption("Ingeniería en Sistemas - UTN FRLP - Ciencia de Datos | v4.0 2026")

# Footer con colaboradores
st.markdown(
    """
    <style>
    .link-colaborador {
        text-decoration: none;
        color: #007bff;
        font-weight: bold;
    }
    .link-colaborador:hover {
        text-decoration: underline;
    }
    </style>
    <p style="font-size: 0.85rem; color: gray;">
        <strong>Desarrollado por: </strong>
        <a href="https://www.linkedin.com/in/gabriel-scarafia/" target="_blank" class="link-colaborador">Gabriel Scarafia</a>, 
        <a href="https://www.linkedin.com/in/matias-russo22/" target="_blank" class="link-colaborador">Matias Russo</a>,
        <a href="https://www.linkedin.com/in/manuel-morullo-161677289/" target="_blank" class="link-colaborador">Manuel Morullo</a>, 
        <a href="https://www.linkedin.com/in/joaquin-munoz-dev/" target="_blank" class="link-colaborador">Joaquin Muñoz</a>, 
        <a href="#" target="_blank" class="link-colaborador">Tomas Sbert</a>
    </p>
    """,
    unsafe_allow_html=True
)
