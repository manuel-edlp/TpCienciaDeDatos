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

# --- GESTIÓN DE ESTADO PERSISTENTE---
# 1. Leemos del navegador (asíncrono)
val_ls = local_storage.getItem(LS_KEY)

# 2. Sincronizamos Session State como Fuente de Verdad
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
        placeholder="API Key...",
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

        # Información de cuotas debajo del selector
        st.sidebar.markdown(f"""
            <div style="font-size: 0.8rem; color: gray; margin-top: 10px;">
                Límites sujetos a tu cuota en <a href="https://aistudio.google.com/app/plan_management" target="_blank" style="color: #007BFF; text-decoration: none;">Google AI Studio</a>.<br>
                Consulta tu consumo <a href="https://aistudio.google.com/app/usage" target="_blank" style="color: #007BFF; text-decoration: none;">aquí</a>.
            </div>
            """, unsafe_allow_html=True)

        return genai.GenerativeModel(selected_model), True



    else:
        if "GEMINI_API_KEY" in st.secrets:
            my_key = st.secrets["GEMINI_API_KEY"]
            
            # Usamos siempre el valor de session_state para la UI
            if st.session_state.rate_limit < 5:
                genai.configure(api_key=my_key)
                restantes = 5 - st.session_state.rate_limit
                
                modo_label = "Modo: Local (Config)" if es_local() else "Modo: Soporte de Sistema"
                st.sidebar.info(modo_label, icon=":material/settings_suggest:")
                st.sidebar.progress(restantes / 5, text=f"{restantes} créditos restantes")
                st.sidebar.caption("Recursos provistos por el desarrollador.")
                
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
                
                # 1. Actualizar Respuesta
                st.session_state.ultima_respuesta = response.text
                
                # 2. Actualizar Créditos (RAM + Browser)
                if not usando_personal:
                    st.session_state.rate_limit += 1
                    local_storage.setItem(LS_KEY, st.session_state.rate_limit)
                    st.toast(f"Crédito usado: {st.session_state.rate_limit}/5", icon=":material/analytics:")
                
                # 3. Rerun para refrescar la barra lateral inmediatamente
                st.rerun()
                    
            except Exception as e:
                st.error(f"Fallo en la inferencia: {e}", icon=":material/emergency_home:")

# Renderizado de resultados
if st.session_state.ultima_respuesta:
    st.markdown("---")
    st.markdown("### :material/recommend: Recomendación")
    st.success(st.session_state.ultima_respuesta)

st.divider()
st.caption("Ingeniería en Sistemas - Ciencia de Datos")

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