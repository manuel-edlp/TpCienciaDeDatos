import streamlit as st
import pandas as pd
import faiss
import joblib
import os
import socket
import google.generativeai as genai
import requests
from sentence_transformers import SentenceTransformer

# --- CONSTANTES Y RUTAS ---
FAISS_FILE = "movie_embeddings.faiss"
DATA_FILE = "movies_preprocessed.pkl"
SCALER_FILE = "scaler.joblib"

def es_local():
    is_cloud = os.environ.get("STREAMLIT_RUNTIME_ENV", False)
    return not is_cloud and "streamlit" not in socket.gethostname().lower()

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Movie AI Recommender", page_icon="🎬", layout="centered")

# --- MANEJO DE SECRETOS (LOCAL Y CLOUD) ---
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY") or os.environ.get("TMDB_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

# --- FUNCIÓN DE POSTERS OPTIMIZADA ---
def obtener_url_poster(movie_id):
    # Intentamos sacar la key de donde sea que esté
    api_key_tmdb = st.secrets.get("TMDB_API_KEY") or os.environ.get("TMDB_API_KEY")
    
    if not api_key_tmdb:
        return "https://placehold.co/500x750?text=Falta+API+Key"

    if not movie_id or pd.isna(movie_id):
        return "https://placehold.co/500x750?text=Sin+ID"

    # Forzamos a que sea un entero
    id_limpio = int(movie_id)
    url = f"https://api.themoviedb.org/3/movie/{id_limpio}?api_key={api_key_tmdb}&language=es-ES"
    
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            # Si la API responde pero no 200 (ej: 401 por key inválida)
            return f"https://placehold.co/500x750?text=Error+API+{response.status_code}"
    except Exception:
        return "https://placehold.co/500x750?text=Error+Red"
    
    return "https://placehold.co/500x750?text=No+Poster"
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
if "query_sugerida" not in st.session_state:
    st.session_state.query_sugerida = ""

def configurar_api():
    st.sidebar.header("Configuración de Sistema", divider="gray")
    
    st.sidebar.markdown("### Cambiar Api Key")
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



# --- SECCIÓN DE INSPIRACIÓN Y TEST RAG ---
with st.container():
    st.markdown("### 💡 ¿Falta de inspiración?")
    st.markdown("""
        Si no sabes por dónde empezar, usá este **Laboratorio de Ideas**. 
        Estos ejemplos están diseñados para desafiar la capacidad de la IA de entender conceptos abstractos y sentimientos, 
        algo que un buscador común no podría resolver.
    """)

    # --- SECCIÓN DE PRUEBAS DE ESTRÉS RAG ---
    with st.expander("✨ Ver categorías de búsqueda avanzada", expanded=True):
        modo = st.radio(
            "Seleccioná qué aspecto de la IA querés poner a prueba:",
            ["Vibe & Sentimiento", "Conceptos Abstractos", "Tramas Específicas", "Híbridos"],
            horizontal=True,
            help="Cada categoría desafía a los vectores de forma diferente."
        )

        opciones_rag = {
            "Vibe & Sentimiento": [
                "Películas que te dejan con una sensación de vacío existencial pero son visualmente hermosas.",
                "Algo para ver un domingo de lluvia que me haga sentir esperanza en la humanidad.",
                "Historias melancólicas sobre amores que nunca pudieron ser.",
                "Una trama que me mantenga en tensión constante y me haga dudar de lo que veo.",
                "Cine contemplativo para desconectar del ruido de la ciudad."
            ],
            "Conceptos Abstractos": [
                "El efecto mariposa y cómo pequeñas decisiones cambian la vida para siempre.",
                "La delgada línea entre la genialidad y la locura en el arte o la ciencia.",
                "Películas que exploren la soledad en ciudades futuristas y altamente tecnológicas.",
                "Crítica social disfrazada de sátira o humor negro muy ácido.",
                "La superación del duelo a través de viajes inesperados."
            ],
            "Tramas Específicas": [
                "Un grupo de personas atrapadas en un solo lugar que deben sobrevivir a un juego macabro.",
                "Un detective retirado que tiene que resolver un último caso que involucra su propio pasado.",
                "Viajes en el tiempo donde las reglas son confusas y hay paradojas temporales.",
                "Inteligencia artificial que cobra conciencia y se enamora o se rebela.",
                "Un náufrago que debe aprender a comunicarse con la naturaleza para no volverse loco."
            ],
            "Híbridos": [
                "Un western pero que transcurra en el espacio exterior con alienígenas.",
                "Terror psicológico mezclado con musical o elementos de danza.",
                "Comedia romántica que tenga un giro de guion (twist) oscuro y violento al final.",
                "Documental ficcionado sobre una civilización que nunca existió.",
                "Animación para adultos con una carga política y filosófica pesada."
            ]
        }

        seleccion = st.selectbox("Elegí una idea para cargar en el buscador:", ["Selecciona una opción..."] + opciones_rag[modo])

        if seleccion != "Selecciona una opción...":
            if st.button("🚀 Cargar esta idea en el buscador", use_container_width=True):
                st.session_state.query_sugerida = seleccion
                st.rerun()

st.divider()

# --- INPUT PRINCIPAL ---
st.markdown("### 🔍 Tu búsqueda personalizada")
st.markdown("_Animate a pedir algo complejo. Hablale a la IA sobre lo que sentís o lo que tenés ganas de experimentar._")

user_query = st.text_input(
    "¿Qué tenés ganas de ver hoy?", 
    value=st.session_state.query_sugerida,
    placeholder="Ej: Una película que me haga sentir que estoy en un sueño...",
    max_chars=150,
    label_visibility="collapsed"
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    btn_search = st.button("Analizar", type="primary", use_container_width=True)

# --- 1. PROMPT CON FILTRO POR TÍTULO ---
if btn_search:
    if user_query:
        with st.spinner("El Sommelier está analizando..."):
            recs = buscar_peliculas(user_query)
            st.session_state.last_recs = recs 
            
            contexto_detallado = "\n".join([
                f"- {row['CleanTitle']} ({row['Year']}): {row['Genres']}." 
                for _, row in recs.iterrows()
            ])
            
            prompt = f"""
            Actúa como un experto cinéfilo. El usuario busca: "{user_query}"
            Candidatos:
            {contexto_detallado}

            Instrucciones:
            1. Selecciona solo las películas que realmente encajen.
            2. No empieces con frases genéricas como 'Basado en tu búsqueda'. 
            3. Usá un lenguaje cálido, entusiasta y humano. 
            4. Imagina que sos un amigo cinéfilo recomendando algo en una charla de café.
            5. Podés usar expresiones como '¡Tengo las opciones perfectas!', 'Si buscás eso, esta te va a volar la cabeza', o 'Prepará los pochoclos para estas joyas'.
            6. Para las seleccionadas, explica por qué las recomiendas.
            7. Si alguna NO encaja, NO la menciones.
            8. AL FINAL, escribí los títulos exactos aprobados así:
               TITULOS_OK: Título 1, Título 2
            """
            
            response = model_gemini.generate_content(prompt)
            texto_completo = response.text
            
            # Extraer títulos para el filtro visual
            titulos_aprobados = []
            if "TITULOS_OK:" in texto_completo:
                partes = texto_completo.split("TITULOS_OK:")
                st.session_state.ultima_respuesta = partes[0].strip()
                titulos_raw = partes[1].strip().split(",")
                titulos_aprobados = [t.strip().lower() for t in titulos_raw]
            else:
                st.session_state.ultima_respuesta = texto_completo

            st.session_state.titulos_filtro = titulos_aprobados
            st.rerun()

# --- 2. RENDERIZADO INTELIGENTE ---
if st.session_state.get("last_recs") is not None:
    # Si tenemos títulos aprobados, filtramos. Si no, por seguridad mostramos todos.
    titulos_ok = st.session_state.get("titulos_filtro", [])
    
    if titulos_ok:
        # Filtramos el DataFrame: el título debe estar en la lista de aprobados
        mask = st.session_state.last_recs['CleanTitle'].str.lower().isin(titulos_ok)
        recs_a_mostrar = st.session_state.last_recs[mask]
    else:
        recs_a_mostrar = st.session_state.last_recs

    # Solo mostramos la sección si hay algo que mostrar
    if not recs_a_mostrar.empty:
        st.markdown("### 🍿 Recomendaciones seleccionadas")
        cols = st.columns(len(recs_a_mostrar))
        for i, (_, row) in enumerate(recs_a_mostrar.iterrows()):
            with cols[i]:
                url = obtener_url_poster(row['id'])
                st.image(url, width='stretch')
                st.caption(f"**{row['CleanTitle']}**")

if st.session_state.get("ultima_respuesta"):
    st.info(st.session_state.ultima_respuesta)

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
