import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import warnings

# Silenciamos advertencias en la interfaz
warnings.filterwarnings("ignore")

# Cargar variables de entorno (.env)
load_dotenv()

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Agente IA - RAG Corporativo",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agente de Atención al Cliente (RAG)")
st.caption("Consulta información en tiempo real basada en la documentación corporativa.")

# --- INICIALIZACIÓN DEL MOTOR RAG ---
@st.cache_resource
def iniciar_agente():
    # 1. Cargar embeddings y base vectorial
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. Configurar modelo de lenguaje (LLM)
    llm = ChatCohere(model="command-r-plus-08-2024")

    # 3. Prompt
    template = """Eres un asistente experto y profesional corporativo. 
Tu tarea es responder a la pregunta del usuario utilizando ÚNICAMENTE 
los siguientes fragmentos de contexto recuperados del documento adjunto. 
Si la respuesta no está en el contexto, di educadamente que no tienes esa información. 
No inventes datos.

Contexto recuperado:
{context}

Pregunta: {input}
Respuesta:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 4. Cadena RAG
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Cargar la cadena de ejecución
rag_chain = iniciar_agente()

# --- HISTORIAL DE CHAT EN STREAMLIT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu agente virtual. ¿En qué te puedo ayudar hoy sobre las citas médicas u otra gestión?"}
    ]

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- ENTRADA DEL USUARIO ---
if prompt_user := st.chat_input("Escribe tu pregunta aquí..."):
    # Guardar y mostrar pregunta del usuario
    st.session_state.messages.append({"role": "user", "content": prompt_user})
    st.chat_message("user").write(prompt_user)

    # Generar respuesta con el Agente RAG
    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de conocimiento..."):
            try:
                respuesta = rag_chain.invoke(prompt_user)
                st.write(respuesta)
                # Guardar respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la respuesta: {e}")
                