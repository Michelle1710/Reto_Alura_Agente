from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
import importlib

try:
    st = importlib.import_module("streamlit")
except ImportError as exc:
    raise ImportError(
        "The 'streamlit' package is required to run this app. Install it with 'pip install streamlit'."
    ) from exc

from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import warnings
import json  
import os   
import pandas as pd
from langchain_core.tools import tool


def create_retriever_tool(retriever, name: str, description: str):
    """Crea una herramienta simple a partir de un retriever compatible."""

    @tool(name=name, description=description)
    def retriever_tool(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return retriever_tool

# Silenciamos advertencias en la interfaz
warnings.filterwarnings("ignore")

# Cargar variables de entorno (.env)
load_dotenv()

# --- PASO 1 Y 2: HERRAMIENTA PARA AGENDAR EN EXCEL ---
@tool
def guardar_cita_en_archivo(nombre: str, fecha: str, especialidad: str) -> str:
    """
    Útil para agendar, registrar o crear una nueva cita médica para un paciente.
    Activa esta herramienta ÚNICAMENTE cuando el usuario confirme que quiere agendar una cita 
    y te haya proporcionado su nombre, la fecha deseada y la especialidad.
    """
    archivo_citas = "citas_agendadas.xlsx"
    
    # Creamos una tabla temporal (DataFrame) con la nueva cita
    nueva_cita = pd.DataFrame([{
        "Nombre": nombre,
        "Fecha": fecha,
        "Especialidad": especialidad
    }])
    
    # Verificamos si el archivo Excel ya existe de una cita anterior
    if os.path.exists(archivo_citas):
        df_existente = pd.read_excel(archivo_citas)
        # Unimos la información antigua con la nueva cita
        df_final = pd.concat([df_existente, nueva_cita], ignore_index=True)
    else:
        # Si es la primera cita, la tabla final es solo esta nueva cita
        df_final = nueva_cita
        
    # Guardamos todo de vuelta en el archivo Excel
    df_final.to_excel(archivo_citas, index=False)
        
    return f"Éxito: La cita de {nombre} para {especialidad} el día {fecha} ha sido guardada."

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Agente IA - RAG Corporativo",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agente de Atención al Cliente (RAG)")
st.caption("Consulta información en tiempo real basada en la documentación corporativa.")

# --- INICIALIZACIÓN DEL AGENTE INTELIGENTE ---
@st.cache_resource
def iniciar_agente():
    # 1. Cargar base vectorial
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. Convertir el PDF en una Herramienta (MÉTODO MANUAL A PRUEBA DE FALLOS)
    from langchain_core.tools import Tool
    
    def buscar_en_pdf(query):
        documentos = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in documentos])

    herramienta_pdf = Tool(
        name="buscar_info_citas",
        description="Busca y devuelve información sobre las políticas de la clínica, horarios, preguntas frecuentes y cómo agendar.",
        func=buscar_en_pdf
    )

    # 3. Empaquetar las herramientas disponibles
    tools = [herramienta_pdf, guardar_cita_en_archivo]

    # 4. Configurar modelo de lenguaje (LLM)
    llm = ChatCohere(model="command-r-plus-08-2024")

    # 5. Crear el Prompt del Agente Autónomo
    template = """Eres un asistente virtual experto y amable de una clínica médica. 
Tienes acceso a dos herramientas: una para buscar información y otra para agendar citas.
Si el usuario te hace una pregunta general, usa la herramienta de buscar información.
Si el usuario quiere agendar una cita, DEBES pedirle su Nombre, Fecha deseada y Especialidad ANTES de usar la herramienta de agendar. 
Nunca inventes datos. Si una cita se guarda con éxito, confírmaselo al usuario de manera cordial."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 6. Construir el Agente y el Ejecutor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

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
                # El agente recibe un diccionario y nos devuelve otro
                resultado = rag_chain.invoke({"input": prompt_user})
                respuesta = resultado["output"]
                
                st.write(respuesta)
                # Guardar respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la respuesta: {e}")
