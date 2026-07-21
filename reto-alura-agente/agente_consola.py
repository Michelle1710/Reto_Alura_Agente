from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import warnings

# Silenciamos advertencias para una consola limpia
warnings.filterwarnings("ignore")

# 1. Cargar las variables de entorno
load_dotenv()

print("🧠 Despertando al agente...")

# 2. Conectar a la base de conocimiento vectorial
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Configuramos el recuperador para traer los 2 fragmentos más relevantes
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Configurar el "Cerebro" (LLM de Cohere)
llm = ChatCohere(model="command-r-plus-08-2024")

# 4. Crear el Prompt corporativo
template = """Eres un asistente de una clínica médica. Tienes dos herramientas:
1. Buscar información: Úsala para dudas generales.
2. Agendar cita: Úsala PARA GUARDAR LA CITA.

REGLA DE ORO: Si el usuario te pide agendar una cita y en ese mismo mensaje te da su Nombre, Fecha y Especialidad, USA LA HERRAMIENTA 'guardar_cita_en_archivo' INMEDIATAMENTE. 
¡No le vuelvas a preguntar los datos si ya los tienes! Solo ejecuta la herramienta y confírmale que fue exitoso. 
Responde siempre de forma directa y natural, sin explicar tu proceso de pensamiento interno.

Contexto recuperado:
{context}

Pregunta: {input}
Respuesta:"""

prompt = ChatPromptTemplate.from_template(template)

# Función auxiliar para unir los documentos recuperados en un solo texto de contexto
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Construir la cadena de ejecución directa (Evita problemas de dependencias con chains)
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- PRUEBA EN CONSOLA ---
if __name__ == "__main__":
    print("✅ Agente listo.")
    print("-" * 50)
    
    pregunta_usuario = "¿Cómo puedo agendar una cita médica?"
    
    print(f"👤 Pregunta: {pregunta_usuario}\n")
    print("⏳ Buscando en los documentos e interpretando...")
    
    # Ejecutamos la consulta directamente
    respuesta = rag_chain.invoke(pregunta_usuario)
    
    print(f"\n🤖 Respuesta del Agente:\n{respuesta}")
    print("-" * 50)