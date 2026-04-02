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

# --- GESTIÓN DE ESTADO ---
if "rate_limit" not in st.session_state:
    st.session_state.rate_limit = 0
if "api_autenticada" not in st.session_state:
    st.session_state.api_autenticada = False

def configurar_api():
    st.sidebar.header("Configuración de Sistema", divider="gray")
    st.sidebar.markdown("### Backend Engine")

    key_input = st.sidebar.text_input(
        "Ingresar API Key Personal", 
        type="password", 
        placeholder="AIza...",
        help="Obtén tu clave en Google AI Studio."
    )

    if st.sidebar.button("Validar API Key", use_container_width=True):
        if key_input:
            st.session_state.api_autenticada = True
            st.toast("API Key validada correctamente", icon=":material/check_circle:")
        else:
            st.sidebar.warning("Ingresa una clave válida.", icon=":material/warning:")

    selected_model = "gemma-3-27b-it" 
    
    if st.session_state.api_autenticada and key_input:
        genai.configure(api_key=key_input)
        st.sidebar.success("Modo: API Key Personal", icon=":material/shield_person:")
        
        selected_model = st.sidebar.selectbox(
            "Seleccionar Modelo", 
            ["gemini-3-flash", "gemini-3-pro", "gemini-2.5-flash", "gemma-3-27b-it", "gemma-3-12b-it"]
        )

        st.sidebar.markdown(f"""
            <div style="font-size: 0.8rem; color: gray; margin-top: 15px;">
                Límites sujetos a tu cuota en <a href="https://aistudio.google.com/app/plan_management" target="_blank" style="color: #007BFF; text-decoration: none;">Google AI Studio</a>.<br>
                Consulta tu consumo <a href="https://aistudio.google.com/app/usage" target="_blank" style="color: #007BFF; text-decoration: none;">aquí</a>.
            </div>
            """, unsafe_allow_html=True)
        
        return genai.GenerativeModel(selected_model), True

    else:
        if "GEMINI_API_KEY" in st.secrets:
            my_key = st.secrets["GEMINI_API_KEY"]
            
            if st.session_state.rate_limit < 5:
                genai.configure(api_key=my_key)
                restantes = 5 - st.session_state.rate_limit
                
                # CORRECCIÓN AQUÍ: settings_suggest en minúsculas
                modo_label = "Modo: Local (Config)" if es_local() else "Modo: Soporte de Sistema"
                st.sidebar.info(modo_label, icon=":material/settings_suggest:")
                st.sidebar.progress(restantes / 5, text=f"{restantes} créditos restantes")
                st.sidebar.caption("Recursos de infraestructura provistos por el desarrollador.")
                
                return genai.GenerativeModel(selected_model), False
            else:
                st.sidebar.warning("Soporte de sistema agotado", icon=":material/lock_clock:")
                st.sidebar.markdown("Obtén una clave en [Google AI Studio](https://aistudio.google.com/app/apikey).")
                return None, False
        else:
            st.sidebar.error("Archivo de secretos detectado pero 'GEMINI_API_KEY' no encontrada.", icon=":material/key_off:")
            return None, False

# --- LÓGICA DE BÚSQUEDA ---
def buscar_peliculas(query, top_k=5):
    query_embedding = model_st.encode([query])
    query_scaled = scaler.transform(query_embedding)
    distances, indices = index.search(query_scaled, top_k)
    return movies_df.iloc[indices[0]]

# --- INTERFAZ PRINCIPAL ---
st.title(":material/smart_toy: Movie AI Recommender")
st.subheader("Búsqueda Semántica de Cine", divider="blue")
st.markdown("Utiliza Inteligencia Artificial para encontrar películas por su significado.")

res_config = configurar_api()
if res_config:
    model_gemini, usando_personal = res_config
else:
    model_gemini, usando_personal = None, False

user_query = st.text_input(
    "Busca películas describiendo lo que sientes o quieres ver", 
    placeholder="Ejemplo: Una película de acción de los 2000 con mucha comedia",
    max_chars=150
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    btn_search = st.button("Analizar", type="primary", use_container_width=True)

if btn_search:
    if not model_gemini:
        st.error("Se requiere una API Key válida.", icon=":material/lock:")
    elif user_query:
        # Usamos spinner para una carga limpia sin contenedores colapsables
        with st.spinner("Analizando preferencias y consultando base de datos..."):
            # 1. Búsqueda Vectorial
            recs = buscar_peliculas(user_query)
            
            # 2. Preparación de contexto
            contexto = "\n".join([
                f"- {row['CleanTitle']} ({row['Year']}): {row['Genres']}." 
                for _, row in recs.iterrows()
            ])
            
            prompt = f"Usuario busca: {user_query}\nCandidatos:\n{contexto}\nRecomienda brevemente las mejores opciones en español."
            
            # 3. Generación con LLM
            try:
                response = model_gemini.generate_content(prompt)
                
                # Mostramos los resultados directamente
                st.markdown("---")
                st.markdown("### :material/recommend: Recomendación")
                st.success(response.text)
                
                # Actualizar cuota si corresponde
                if not usando_personal:
                    st.session_state.rate_limit += 1
                    
            except Exception as e:
                st.error(f"Fallo en la inferencia: {e}", icon=":material/emergency_home:")
    else:
        st.warning("Ingresa una descripción para iniciar la búsqueda.", icon=":material/chat_error:")

st.divider()
st.caption("Ingeniería en Sistemas | v3.3 2026")