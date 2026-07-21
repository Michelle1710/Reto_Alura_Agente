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