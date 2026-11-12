"""
ACA N.° 2 – Computación en la Nube (CUN)
Chatbot RAG sobre cáncer de mama + gestión documental de la base vectorial (Qdrant Cloud) – App Streamlit
"""
import os
import time

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_core import SIN_INFO, Config, SistemaRAG

st.set_page_config(page_title="Chatbot RAG · Cáncer de mama", page_icon="🩺", layout="wide")


def secreto(nombre, defecto=None):
    """Lee una credencial desde variables de entorno (.env) o desde los secrets de Streamlit."""
    valor = os.getenv(nombre)
    if valor:
        return valor
    try:
        return st.secrets[nombre]
    except Exception:
        return defecto


QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY = secreto("QDRANT_URL"), secreto("QDRANT_API_KEY"), secreto("GOOGLE_API_KEY")
ADMIN_PASSWORD = secreto("ADMIN_PASSWORD")
MAX_PREGUNTAS = int(secreto("MAX_PREGUNTAS_SESION", 20))
if not (QDRANT_URL and QDRANT_API_KEY and GOOGLE_API_KEY):
    st.error("Faltan credenciales. Define QDRANT_URL, QDRANT_API_KEY y GOOGLE_API_KEY en el archivo `.env` (local) "
             "o en los *Secrets* de Streamlit Cloud. Nunca las escribas dentro del código.")
    st.stop()


@st.cache_resource(show_spinner="Conectando con Qdrant Cloud…")
def obtener_sistema():
    return SistemaRAG(Config(qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_API_KEY, google_api_key=GOOGLE_API_KEY,
                             llm_model=secreto("LLM_MODEL")))


@st.cache_data(ttl=30, show_spinner=False)
def cargar_estadisticas(_s):
    return _s.estadisticas()


@st.cache_data(ttl=30, show_spinner=False)
def cargar_documentos(_s):
    return _s.documentos()


try:
    rag = obtener_sistema()
    stats = cargar_estadisticas(rag)
except Exception as e:
    st.error(f"No fue posible conectar con la base vectorial: {e}")
    st.stop()

TIPOS = {"Todos los documentos": None, "Guía MinSalud (pacientes)": "guia_pacientes", "Nota OMS": "nota_oms",
         "Guía clínica (profesionales)": "guia_profesional", "Dataset UCI WDBC (fichas)": "dataset",
         "Modelo SVM (ACA Aprendizaje Automático)": "dataset_modelo", "Documentos cargados por el usuario": "usuario"}
EJEMPLOS = ["¿Cuáles son los factores de riesgo del cáncer de mama?",
            "¿Cada cuánto debe hacerse una mamografía una mujer sin síntomas?",
            "¿Qué efectos secundarios tiene la quimioterapia?",
            "¿Qué características del núcleo celular distinguen un tumor maligno de uno benigno?"]

st.session_state.setdefault("mensajes", [])
st.session_state.setdefault("n_preguntas", 0)

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("🩺 Configuración")
    st.caption(f"Base vectorial: {stats['dimension']} dim. · {stats['distancia']} · fragmentos indexados: **{stats['puntos']}**")
    st.caption(f"LLM: **{rag.modelo_activo or 'se detecta en la primera consulta'}**")
    st.divider()
    filtro_nombre = st.selectbox("Buscar en", list(TIPOS), help="Filtra la recuperación por tipo de fuente (metadato `tipo`).")
    k = st.slider("Fragmentos a recuperar (Top-k)", 1, 10, rag.cfg.k)
    umbral = st.slider("Umbral de similitud coseno", 0.30, 0.90, float(rag.cfg.umbral), 0.01,
                       help="Si ningún fragmento lo supera, el chatbot responde que no tiene información y no consulta al LLM.")
    temperatura = st.slider("Temperatura del LLM", 0.0, 1.0, float(rag.cfg.temperatura), 0.05,
                            help="Valores bajos dan respuestas más estables y apegadas al contexto.")
    reformular = st.checkbox("Reformular preguntas de seguimiento", value=True)
    if st.button("🗑️ Nueva conversación", width="stretch"):
        st.session_state.mensajes = []
        st.rerun()
    st.divider()
    st.caption(f"Preguntas en esta sesión: {st.session_state.n_preguntas}/{MAX_PREGUNTAS}")
    st.warning("Prototipo académico de consulta informativa. **No es un diagnóstico médico.** No escribas datos personales: "
               "las preguntas se envían a la API de Google Gemini.")

st.title("🩺 Chatbot RAG sobre cáncer de mama")
st.markdown("Responde **solo con base en los documentos indexados** (guías clínicas y dataset UCI WDBC) y cita sus fuentes.")
tab_chat, tab_docs, tab_info = st.tabs(["💬 Chat", "🗂️ Gestión documental", "ℹ️ Acerca del proyecto"])


# ------------------------------------------------------------------ utilidades de presentación
def mostrar_fuentes(meta):
    usados, citadas = meta["usados"], set(meta["citadas"])
    if meta["rechazada"] and not usados:
        mejor = max((f["score"] for f in meta["recuperados"]), default=0)
        st.info(f"Ningún fragmento superó el umbral de similitud (el mejor fue {mejor:.3f}), por lo que no se consultó al LLM.")
        return
    with st.expander(f"📚 Fuentes: {len(citadas)} citadas · {len(usados)} fragmentos usados"):
        for i, f in enumerate(usados, start=1):
            marca = "✅ citado" if i in citadas else "• recuperado"
            pag = f" · pág. {f['pagina']}" if f["pagina"] else ""
            st.markdown(f"**[{i}] {f['titulo']}**{pag} · similitud **{f['score']:.3f}** · {marca}")
            st.caption(f["texto"][:600] + ("…" if len(f["texto"]) > 600 else ""))
    if meta.get("reformulada") and meta["reformulada"] != meta["pregunta"]:
        st.caption(f"🔁 Pregunta usada para buscar: _{meta['reformulada']}_")
    st.caption(f"⏱️ {meta['t_total']:.1f} s en total (recuperación {meta['t_recuperacion']:.1f} s · LLM {meta['t_llm']:.1f} s) · modelo: {meta['modelo'] or '—'}")


with tab_chat:
    if not st.session_state.mensajes:
        st.markdown("**Prueba con una pregunta de ejemplo:**")
        cols = st.columns(2)
        for i, ej in enumerate(EJEMPLOS):
            if cols[i % 2].button(ej, key=f"ej{i}", width="stretch"):
                st.session_state.pendiente = ej
    for m in st.session_state.mensajes:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("meta"):
                mostrar_fuentes(m["meta"])

    pregunta = st.chat_input("Escribe tu pregunta sobre cáncer de mama…") or st.session_state.pop("pendiente", None)
    if pregunta:
        pregunta = pregunta.strip()[:500]
        with st.chat_message("user"):
            st.markdown(pregunta)
        if st.session_state.n_preguntas >= MAX_PREGUNTAS:
            st.warning("Alcanzaste el límite de preguntas de esta sesión (protege la cuota gratuita de la API). Recarga la página para continuar.")
        else:
            historial = [(m["role"], m["content"]) for m in st.session_state.mensajes]
            with st.chat_message("assistant"):
                with st.spinner("Buscando en los documentos y generando la respuesta…"):
                    try:
                        r = rag.responder(pregunta, historial=historial, k=k, umbral=umbral, tipo=TIPOS[filtro_nombre],
                                          temperatura=temperatura, reformulacion=reformular)
                    except Exception as e:
                        st.error(f"Ocurrió un error al responder: {e}")
                        st.stop()
                meta = dict(pregunta=pregunta, reformulada=r.pregunta_busqueda, rechazada=r.rechazada, recuperados=r.recuperados,
                            usados=r.usados, citadas=r.citadas, modelo=r.modelo, t_recuperacion=r.t_recuperacion,
                            t_llm=r.t_llm, t_total=r.t_total)
                st.markdown(r.texto)
                mostrar_fuentes(meta)
            st.session_state.mensajes += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": r.texto, "meta": meta}]
            st.session_state.n_preguntas += 1
            st.rerun()

# ------------------------------------------------------------------ gestión documental
with tab_docs:
    st.subheader("Estado del índice vectorial")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fragmentos indexados", stats["puntos"])
    c2.metric("Dimensión", stats["dimension"])
    c3.metric("Distancia", stats["distancia"])
    c4.metric("Estado del clúster", stats["estado"])

    docs = cargar_documentos(rag)
    st.markdown("#### Documentos en la base")
    st.dataframe(docs, width="stretch", hide_index=True)
    if not docs.empty:
        st.bar_chart(docs.set_index("fuente")["fragmentos"])

        st.markdown("#### Explorar los fragmentos de un documento")
        fuente = st.selectbox("Documento", docs["fuente"].tolist())
        frs = rag.fragmentos_de(fuente)
        st.caption(f"{len(frs)} fragmentos")
        vista = frs.assign(texto=frs["texto"].str.replace(r"\s+", " ", regex=True).str[:160] + "…")
        st.dataframe(vista, width="stretch", hide_index=True)
        if len(frs):
            i = st.number_input("Ver el texto completo del fragmento n.°", 0, len(frs) - 1, 0)
            st.text_area("Contenido", frs.iloc[int(i)]["texto"], height=180, disabled=True)

    st.markdown("#### Probador de búsqueda semántica (sin LLM)")
    consulta = st.text_input("Consulta", placeholder="p. ej. ¿qué es el carcinoma ductal in situ?")
    if consulta:
        res = rag.recuperar(consulta, k, TIPOS[filtro_nombre])
        st.dataframe(pd.DataFrame(res).assign(texto=lambda d: d["texto"].str.replace(r"\s+", " ", regex=True).str[:200] + "…")
                     [["rank", "score", "fuente", "pagina", "tipo", "texto"]].round({"score": 4}), width="stretch", hide_index=True)

    st.markdown("#### Administración (requiere contraseña)")
    if not ADMIN_PASSWORD:
        st.caption("Modo solo lectura: no hay contraseña de administración configurada (ADMIN_PASSWORD).")
    elif st.text_input("Contraseña de administrador", type="password") == ADMIN_PASSWORD:
        archivo = st.file_uploader("Agregar un documento (PDF, TXT o MD)", type=["pdf", "txt", "md"])
        if archivo and st.button("Indexar documento"):
            with st.spinner("Fragmentando, vectorizando e insertando en Qdrant…"):
                n = rag.indexar_archivo(archivo.name, archivo.getvalue())
            cargar_documentos.clear(); cargar_estadisticas.clear()
            st.success(f"Se indexaron {n} fragmentos de «{archivo.name}».")
        propios = docs[docs["tipo"] == "usuario"]["fuente"].tolist() if not docs.empty else []
        if propios:
            borrar = st.selectbox("Eliminar un documento cargado por el usuario", propios)
            if st.checkbox("Confirmo que quiero eliminarlo") and st.button("Eliminar documento"):
                rag.eliminar_documento(borrar)
                cargar_documentos.clear(); cargar_estadisticas.clear()
                st.success(f"Se eliminó «{borrar}».")
        st.caption("Por seguridad solo se pueden eliminar los documentos cargados desde esta interfaz, no el corpus base del ACA 1.")

# ------------------------------------------------------------------ información
with tab_info:
    st.markdown("""
**Asignatura:** Computación en la Nube · Especialización en Inteligencia Artificial · CUN  
**Proyecto:** ACA N.° 2 – Integración de LLM, chatbot RAG e interfaz gráfica para gestión documental y vectorial  
**Pilar:** Salud · **Dataset base:** UCI Breast Cancer Wisconsin (Diagnostic)

**Cómo responde el sistema**
1. La pregunta se vectoriza (Gemini Embedding, `RETRIEVAL_QUERY`) y se buscan los Top-k fragmentos más cercanos en Qdrant Cloud (distancia coseno).
2. Si ningún fragmento supera el umbral de similitud, responde que no tiene información y **no consulta al LLM**.
3. Si hay evidencia, los fragmentos se numeran e inyectan en el prompt; el LLM (Gemini) debe responder solo con ese contexto y **citar cada afirmación** con [n].
4. La interfaz muestra las fuentes, páginas, similitudes y tiempos de respuesta.

**Medidas contra las alucinaciones:** instrucciones estrictas de usar solo el contexto, citas obligatorias, umbral de similitud, temperatura baja y rechazo explícito cuando no hay información.

**Privacidad:** las consultas se envían a la API de Gemini (capa gratuita). No escribas datos personales ni clínicos propios.
""")
    st.warning("Esta aplicación es un prototipo académico de recuperación de información. No ofrece diagnóstico médico y sus respuestas "
               "deben ser valoradas por un profesional de la salud.")
