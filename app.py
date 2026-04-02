import streamlit as st
import pandas as pd
import faiss
import joblib
import os
import socket
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from streamlit_local_storage import LocalStorage

# --- CONSTANTES Y RUTAS ---
FAISS_FILE = "movie_embeddings.faiss"
DATA_FILE = "movies_preprocessed.pkl"
SCALER_FILE = "scaler.joblib"
LS_KEY = "movie_recommender_rate_limit"

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

# --- INICIALIZACIÓN DE LOCAL STORAGE ---
local_storage = LocalStorage()

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

# --- GESTIÓN DE ESTADO PERSISTENTE ---
# Intentamos obtener el valor del navegador
val_ls = local_storage.getItem(LS_KEY)

# Sincronizamos el session_state con el LocalStorage
if "rate_limit" not in st.session_state:
    if val_ls is not None:
        st.session_state.rate_limit = int(val_ls)
    else:
        st.session_state.rate_limit = 0

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
        return genai.GenerativeModel(selected_model), True

    else:
        if "GEMINI_API_KEY" in st.secrets:
            my_key = st.secrets["GEMINI_API_KEY"]
            
            # Validación de límite con el estado persistente
            if st.session_state.rate_limit < 5:
                genai.configure(api_key=my_key)
                restantes = 5 - st.session_state.rate_limit
                
                modo_label = "Modo: Local (Config)" if es_local() else "Modo: Soporte de Sistema"
                st.sidebar.info(modo_label, icon=":material/settings_suggest:")
                st.sidebar.progress(restantes / 5, text=f"{restantes} créditos restantes")
                st.sidebar.caption("Recursos provistos por el desarrollador (Persistente).")
                
                return genai.GenerativeModel(selected_model), False
            else:
                st.sidebar.warning("Créditos de sistema agotados", icon=":material/lock_clock:")
                st.sidebar.markdown("Usa tu propia API Key para continuar.")
                return None, False
        else:
            st.sidebar.error("Secrets no configurados.", icon=":material/key_off:")
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
        st.error("Límite alcanzado o API Key no válida.", icon=":material/lock:")
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
                
                # Guardamos la respuesta en el estado
                st.session_state.ultima_respuesta = response.text
                
                # Actualización de créditos
                if not usando_personal:
                    # Incrementamos en la sesión
                    st.session_state.rate_limit += 1
                    # Guardamos en el navegador (LocalStorage)
                    local_storage.setItem(LS_KEY, st.session_state.rate_limit)
                
                st.rerun() 
                    
            except Exception as e:
                st.error(f"Fallo en la inferencia: {e}", icon=":material/emergency_home:")

# Renderizado de resultados (persiste tras el rerun)
if st.session_state.ultima_respuesta:
    st.markdown("---")
    st.markdown("### :material/recommend: Recomendación")
    st.success(st.session_state.ultima_respuesta)

st.divider()
st.caption("Ingeniería en Sistemas | v3.3 2026")