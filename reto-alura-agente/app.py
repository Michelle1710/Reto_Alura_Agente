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
from typing import Optional

class template:
    """Wrapper for agent prompt templates with a concrete implementation."""

    def __init__(self, system_message: str):
        self.system_message = system_message

    def build_messages(
        self,
        input_text: str = "{input}",
        chat_history: Optional[list] = None,
        agent_scratchpad: Optional[str] = None,
    ):
        messages = [("system", self.system_message)]
        if chat_history:
            messages.extend(chat_history)
        messages.append(("human", input_text))
        if agent_scratchpad is not None:
            messages.append(("assistant", agent_scratchpad))
        return messages

    def to_chat_prompt_template(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])


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
# 1. Cargar PDF y crear base vectorial EN VIVO (MÉTODO INFALIBLE)
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import os

    # Buscamos la ruta exacta del PDF guiándonos por tu estructura de carpetas
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_pdf = os.path.abspath(os.path.join(ruta_actual, "..", "Documento", "Información clinica.pdf"))
    
    # Cargamos y dividimos el documento
    loader = PyPDFLoader(ruta_pdf)
    documentos_pdf = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos_divididos = text_splitter.split_documents(documentos_pdf)
    
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    
    # Creamos la base de datos directamente en la memoria temporal
    vectorstore = Chroma.from_documents(documents=textos_divididos, embedding=embeddings)
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

# 5. Crear el Prompt del Agente Autónomo
    template_text = """Eres un recepcionista virtual de una clínica médica. Hablas directamente con el paciente de forma amable y conversacional.

    REGLAS ESTRICTAS PARA AGENDAR CITAS:
    Para agendar, necesitas 4 datos, pero DEBES PEDIRLOS UNO POR UNO, NUNCA todos a la vez. Sigue exactamente este orden:
    1. Si no tienes la especialidad, pregunta SOLO: "¿Para qué especialidad médica necesitas la cita?" y detente.
    2. Si ya tienes la especialidad pero no la fecha, pregunta SOLO: "¿Para qué fecha te gustaría venir?" y detente.
    3. Si ya tienes especialidad y fecha, pregunta SOLO: "¿A qué hora te viene mejor?" y detente.
    4. Si ya tienes especialidad, fecha y hora, pregunta SOLO: "¿Me podrías indicar tu nombre completo?" y detente.

    REGLA DE FORMATO:
    NUNCA pienses en voz alta. NUNCA escribas "Le preguntaré al...", "Voy a pedirle..." o "El paciente no ha...". Escribe ÚNICAMENTE la pregunta final."""

    prompt_template = template(template_text)
    prompt = prompt_template.to_chat_prompt_template()
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
    if (tiempo_actual - st.session_state.last_time) > 60:
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
                # Invocamos al agente
                resultado = rag_chain.invoke({
                    "input": prompt_user,
                    "chat_history": chat_history_formateado
                })
                respuesta = resultado["output"]
                
                # --- FILTRO LIMPIADOR DEFINITIVO ---
                texto_min = respuesta.lower()
                # Si detectamos intenciones del agente de pensar en voz alta
                if "preguntaré" in texto_min or "pediré" in texto_min or "voy a" in texto_min:
                    # Cortamos la basura y nos quedamos desde el signo de exclamación o interrogación
                    if "¡" in respuesta:
                        respuesta = "¡" + respuesta.split("¡", 1)[1]
                    elif "¿" in respuesta:
                        respuesta = "¿" + respuesta.split("¿", 1)[1]
                    else:
                        # Fallback por si acaso no usó signos
                        respuesta = respuesta.split(".", 1)[-1].strip()
                # ----------------------------------------
                
                st.write(respuesta)
                # Guardar respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la respuesta: {e}")