import streamlit as st
import pandas as pd
import faiss
import joblib
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Movie AI Recommender", 
    page_icon=":material/movie:", 
    layout="centered"
)

# Rutas de artefactos
FAISS_FILE = "movie_embeddings.faiss"
DATA_FILE = "movies_preprocessed.pkl"
SCALER_FILE = "scaler.joblib"

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
except Exception:
    st.error("Error: No se detectaron los archivos de datos.", icon=":material/database_alert:")
    st.stop()

# --- GESTIÓN DE ESTADO ---
if "rate_limit" not in st.session_state:
    st.session_state.rate_limit = 0
if "api_autenticada" not in st.session_state:
    st.session_state.api_autenticada = False

def configurar_api():
    st.sidebar.header("Configuración de Sistema", divider="gray")
    
    st.sidebar.markdown("""
        ### Backend Engine
        Esta aplicación utiliza modelos generativos de Google.
    """)

    # Campo de texto para API Key
    key_input = st.sidebar.text_input(
        "Ingresar API Key Personal", 
        type="password", 
        placeholder="AIza...",
        help="Obtén tu clave en Google AI Studio."
    )

    # Botón para confirmar la clave
    if st.sidebar.button("Validar API Key", use_container_width=True):
        if key_input:
            st.session_state.api_autenticada = True
            st.toast("API Key validada correctamente", icon=":material/check_circle:")
        else:
            st.sidebar.warning("Ingresa una clave válida.", icon=":material/warning:")

    selected_model = "gemini-3-flash" 
    
    # ESCENARIO A: USANDO CLAVE PERSONAL DEL USUARIO
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
    
    # ESCENARIO B: USANDO CLAVE POR DEFECTO (SOPORTE DE SISTEMA)
    else:
        if st.session_state.rate_limit < 5:
            try:
                my_key = st.secrets["GEMINI_API_KEY"]
                genai.configure(api_key=my_key)
                restantes = 5 - st.session_state.rate_limit
                
                st.sidebar.info("Modo: Soporte de Sistema", icon=":material/Settings_Suggest:")
                st.sidebar.progress(restantes / 5, text=f"{restantes} créditos restantes")
                st.sidebar.caption("Recursos de infraestructura provistos por el desarrollador.")
            except Exception:
                st.sidebar.error("Error de configuración de servidor (Secrets).", icon=":material/error:")
                return None, None
        else:
            # ESCENARIO C: CUOTA AGOTADA
            st.sidebar.warning("Soporte de sistema agotado", icon=":material/lock_clock:")
            st.sidebar.markdown("""
                Para continuar con las pruebas:
                1. Obtén una clave gratuita en [Google AI Studio](https://aistudio.google.com/app/apikey).
                2. Pégala arriba y presiona 'Validar API Key'.
            """)
            return None, None
            
    return genai.GenerativeModel(selected_model), (st.session_state.api_autenticada and key_input)

# --- LÓGICA DE BÚSQUEDA ---
def buscar_peliculas(query, top_k=5):
    query_embedding = model_st.encode([query])
    query_scaled = scaler.transform(query_embedding)
    distances, indices = index.search(query_scaled, top_k)
    return movies_df.iloc[indices[0]]

# --- INTERFAZ PRINCIPAL ---
st.title(":material/smart_toy: Movie AI Recommender")
st.subheader("Búsqueda Semántica de Cine", divider="blue")
st.markdown("Utiliza Inteligencia Artificial para encontrar películas por su significado, no solo por palabras clave.")

model_gemini, usando_llave_personal = configurar_api()

user_query = st.text_input(
    "Consulta",
    placeholder="Un thriller psicológico oscuro de los 90 con final inesperado",
    max_chars=150,
    label_visibility="collapsed"
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    btn_search = st.button("Analizar", type="primary", use_container_width=True)

if btn_search:
    if not model_gemini:
        st.error("Se requiere autenticación para procesar la consulta.", icon=":material/lock:")
    elif user_query:
        with st.status("Ejecutando pipeline de IA...", expanded=True) as status:
            st.write("Recuperando información vectorial (FAISS)...")
            recs = buscar_peliculas(user_query)
            
            st.write("Generando análisis semántico con LLM...")
            contexto = "\n".join([
                f"- {row['CleanTitle']} ({row['Year']}): {row['Genres']}." 
                for _, row in recs.iterrows()
            ])
            
            prompt = f"Usuario busca: {user_query}\nCandidatos encontrados en base de datos:\n{contexto}\nRecomienda brevemente las mejores opciones en español."
            
            try:
                response = model_gemini.generate_content(prompt)
                status.update(label="Análisis finalizado", state="complete", expanded=False)
                
                st.markdown("---")
                st.markdown("### :material/recommend: Recomendación")
                st.success(response.text)
                
                if not usando_llave_personal:
                    st.session_state.rate_limit += 1
                    
            except Exception as e:
                st.error(f"Fallo en la inferencia del modelo: {e}", icon=":material/emergency_home:")
    else:
        st.warning("Ingresa una descripción para iniciar la búsqueda.", icon=":material/chat_error:")

st.divider()
st.caption("Ingeniería en Sistemas | v3.2 2026")