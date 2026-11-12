# Chatbot RAG sobre cáncer de mama: LLM, base vectorial en la nube e interfaz de gestión documental

Proyecto ACA N.° 2 – **Computación en la Nube** · CUN · Especialización en Inteligencia Artificial
**Autor:** Cristian Fernando Marín Caro · **Docente:** Hamilton Rivera Flor · **Pilar:** Salud

Completa el ciclo end-to-end del sistema RAG iniciado en el [ACA 1](https://github.com/CristianMarinC/rag-cancer-mama): un chatbot que recupera fragmentos de la base vectorial (**Qdrant Cloud**) y los entrega a un **LLM (Gemini por API)** para responder solo con evidencia documental y citando sus fuentes, con una interfaz en **Streamlit** para conversar y administrar los documentos.

## Arquitectura

```
Usuario ─► Streamlit (chat + gestión documental)
              │ pregunta
              ▼
  Embedding de la consulta (Gemini, RETRIEVAL_QUERY, 768 dim.)
              ▼
  Qdrant Cloud (ACA 1) ─ Top-k por similitud coseno ─► ¿similitud ≥ umbral?
              │                                          │ no ► "No encontré información…" (no se llama al LLM)
              │ sí
              ▼
  Prompt = reglas (solo contexto + citas [n]) + fragmentos numerados + historial + pregunta
              ▼
  LLM (Gemini, temperatura 0,2) ─► respuesta con citas [n] ─► fuentes, páginas, similitudes y tiempos en la interfaz
```

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `ACA2_Computacion_Nube_Chatbot_RAG.ipynb` | Notebook reproducible: conexión, LLM, pipeline RAG, pruebas de diálogo y evaluación |
| `app.py` | Interfaz Streamlit: chat con historial y fuentes, gestión documental |
| `rag_core.py` | Lógica RAG compartida por el notebook y la app |
| `requirements.txt` | Dependencias |
| `env.example` | Plantilla de credenciales (sin valores reales) |

## Cómo ejecutarlo

**Requisito:** la colección `rag_cancer_mama` ya indexada en Qdrant Cloud con el [notebook del ACA 1](https://github.com/CristianMarinC/rag-cancer-mama).

1. Crea un clúster gratuito en [Qdrant Cloud](https://cloud.qdrant.io) y una API key en [Google AI Studio](https://aistudio.google.com/apikey).
2. Copia `env.example` como `.env` y completa tus claves. **Nunca subas el archivo `.env` al repositorio.**
3. Instala y ejecuta:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. O abre el notebook en Google Colab y ejecútalo con *Entorno de ejecución → Ejecutar todo*.

### Despliegue en Streamlit Community Cloud
En *Advanced settings* elige **Python 3.12** y pega las credenciales en *Secrets* (formato TOML):
```toml
QDRANT_URL = "https://TU-CLUSTER.cloud.qdrant.io:6333"
QDRANT_API_KEY = "..."
GOOGLE_API_KEY = "..."
ADMIN_PASSWORD = "..."
```

## Qué incluye la interfaz

* **Chat:** historial conversacional, preguntas de ejemplo, fuentes citadas (documento, página, similitud y texto), pregunta reformulada en preguntas de seguimiento y tiempos de respuesta.
* **Gestión documental:** estado del índice, tabla de documentos, explorador de fragmentos con metadatos, probador de búsqueda semántica sin LLM y administración con contraseña (agregar o eliminar documentos).
* **Barra lateral:** filtro por tipo de fuente, Top-k, umbral de similitud, temperatura y reformulación.

## Medidas contra las alucinaciones

Instrucciones de responder solo con el contexto y citar con `[n]`; umbral de similitud (0,62, calibrado en el ACA 1) que evita llamar al LLM sin evidencia; temperatura baja (0,2); el contexto se trata como datos y no como instrucciones; sin diagnósticos ni recomendaciones médicas personales.

## Seguridad

Las credenciales se leen solo desde variables de entorno (`.env`) o *Secrets*; no hay claves en el código. La app limita las preguntas por sesión y su longitud para proteger la cuota gratuita, y las funciones de administración requieren contraseña. Las consultas se envían a la API de Gemini (capa gratuita): no se deben escribir datos personales.

## Aviso

Prototipo académico de consulta informativa. No ofrece diagnóstico médico ni reemplaza el criterio de un profesional de la salud.

## Referencias

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474. https://arxiv.org/abs/2005.11401

Streamlit Inc. (2024). *Streamlit documentation: Build and share data apps*. https://docs.streamlit.io/

Wolberg, W., Street, N., & Mangasarian, O. (1995). *UCI Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
