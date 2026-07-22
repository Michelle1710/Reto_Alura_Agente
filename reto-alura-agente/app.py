from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
import importlib
import time # Añadimos time para medir los 40 segundos

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
from langchain.tools import tool
import pandas as pd
import os

@tool
def agendar_cita(nombre: str, fecha: str, especialidad: str, hora: str) -> str:
    """Útil para agendar una cita médica. Requiere el nombre del paciente, la fecha, la especialidad y la hora."""
    
    archivo = "citas_agendadas.xlsx"
    # Añadimos la columna 'Hora' al diccionario
    nueva_cita = pd.DataFrame([{"Nombre": nombre, "Fecha": fecha, "Hora": hora, "Especialidad": especialidad}])

    if os.path.exists(archivo):
        df = pd.read_excel(archivo)
        df = pd.concat([df, nueva_cita], ignore_index=True)
    else:
        df = nueva_cita

    df.to_excel(archivo, index=False)
    # El mensaje de retorno ahora incluye la hora
    return f"Tu cita para {especialidad} el {fecha} a las {hora} ha sido agendada con éxito, te esperamos."

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
    tools = [herramienta_pdf, agendar_cita]

    # 4. Configurar modelo de lenguaje (LLM)
    llm = ChatCohere(model="command-r-plus-08-2024")

    # 5. Crear el Prompt del Agente Autónomo
    template = """Eres un asistente virtual experto y amable de una clínica médica. 
Tienes acceso a dos herramientas: una para buscar información y otra para agendar citas.
Si el usuario te hace una pregunta general, usa la herramienta de buscar información.
Si el usuario quiere agendar una cita, DEBES pedirle su Nombre, Fecha deseada y Especialidad ANTES de usar la herramienta de agendar. 
Nunca inventes datos. Si una cita se guarda con éxito, confírmaselo al usuario de manera cordial.

IMPORTANTE: NUNCA reveles tus pensamientos internos, tus planes, ni narres lo que vas a hacer. 
Responde ÚNICA Y EXCLUSIVAMENTE con el mensaje final dirigido al usuario en un tono amable."""

    # MODIFICACIÓN: Añadimos el chat_history al prompt para evitar la amnesia
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 6. Construir el Agente y el Ejecutor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

# Cargar la cadena de ejecución
rag_chain = iniciar_agente()

# --- CONFIGURACIÓN DE MEMORIA Y TIEMPO EN STREAMLIT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu agente virtual. ¿En qué te puedo ayudar hoy sobre las citas médicas u otra gestión?"}
    ]
if "last_time" not in st.session_state:
    st.session_state.last_time = time.time()
if "chat_activo" not in st.session_state:
    st.session_state.chat_activo = True

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Si el chat se cerró, mostramos un mensaje y bloqueamos el script
if not st.session_state.chat_activo:
    st.warning("El chat ha finalizado. Por favor, recarga la página web para iniciar una nueva conversación.")
    st.stop()

# --- ENTRADA DEL USUARIO ---
if prompt_user := st.chat_input("Escribe tu pregunta aquí..."):
    tiempo_actual = time.time()

    # 1. Comprobar inactividad (40 segundos)
    if (tiempo_actual - st.session_state.last_time) > 40:
        st.warning("⏱️ La sesión ha expirado por inactividad (más de 40 segundos).")
        st.session_state.chat_activo = False
        st.rerun()
    
    # Actualizamos la marca de tiempo para la siguiente iteración
    st.session_state.last_time = tiempo_actual

    # 2. Mostrar pregunta del usuario
    st.session_state.messages.append({"role": "user", "content": prompt_user})
    st.chat_message("user").write(prompt_user)

    # 3. Comprobar palabras clave de cierre ("chao" o "gracias por tu ayuda")
    texto_minusculas = prompt_user.lower()
    if "chao" in texto_minusculas or "gracias por tu ayuda" in texto_minusculas:
        mensaje_despedida = "¡Ha sido un placer ayudarte! Que tengas un excelente día. Hasta luego. 👋"
        st.chat_message("assistant").write(mensaje_despedida)
        st.session_state.messages.append({"role": "assistant", "content": mensaje_despedida})
        st.session_state.chat_activo = False
        st.stop() # Detenemos la ejecución aquí para que no busque en LangChain

    # 4. Traducir el historial de Streamlit al formato de LangChain (Evita la amnesia)
    chat_history_formateado = []
    for msg in st.session_state.messages:
        # LangChain necesita ignorar el saludo inicial si queremos ser estrictos, pero le pasaremos todo
        if msg["role"] == "user":
            chat_history_formateado.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            chat_history_formateado.append(("assistant", msg["content"]))

    # 5. Generar respuesta con el Agente RAG
    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            try:
                # MODIFICACIÓN: Pasamos la memoria en la invocación
                resultado = rag_chain.invoke({
                    "input": prompt_user,
                    "chat_history": chat_history_formateado
                })
                respuesta = resultado["output"]
                
                st.write(respuesta)
                # Guardar respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la respuesta: {e}")