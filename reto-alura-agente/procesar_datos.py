import importlib
import os
import pandas as pd


def _obtener_pdf_reader():
    """Devuelve PdfReader de PyPDF2 o pypdf, según esté instalado."""
    for modulo in ("PyPDF2", "pypdf"):
        try:
            modulo_importado = importlib.import_module(modulo)
            reader_class = getattr(modulo_importado, "PdfReader", None)
            if reader_class is not None:
                return reader_class
        except Exception:
            continue

    raise ImportError(
        "PdfReader not found. Install 'PyPDF2' or 'pypdf' (pip install PyPDF2 or pypdf)."
    )


PdfReader = _obtener_pdf_reader()
from dotenv import load_dotenv

# --- IMPORTACIONES ACTUALIZADAS DE LANGCHAIN ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings 

# Cargar la clave de API desde el archivo .env
load_dotenv()

def extraer_texto_pdf(ruta_archivo):
    """Extrae el texto de un archivo PDF."""
    print(f"📄 Leyendo el PDF: {ruta_archivo}")
    texto = ""
    try:
        lector = PdfReader(ruta_archivo)
        for pagina in lector.pages:
            if pagina.extract_text():
                texto += pagina.extract_text() + "\n"
        print("✅ PDF procesado con éxito.")
        return texto
    except Exception as e:
        print(f"❌ Error al leer el PDF: {e}")
        return None

def crear_base_conocimiento(texto):
    """Divide el texto y lo guarda en una base de datos vectorial Chroma."""
    print("🪓 Dividiendo el texto en fragmentos...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    fragmentos = text_splitter.split_text(texto)
    print(f"✅ Texto dividido en {len(fragmentos)} fragmentos.")

    print("🧠 Generando embeddings con COHERE y guardando en ChromaDB...")
    
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    
    vectorstore = Chroma.from_texts(
        texts=fragmentos, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("🚀 ¡Base de conocimiento vectorial creada con éxito en './chroma_db'!")
    return vectorstore

# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    ruta_documento = "C:\\Users\\Michelle\\Documents\\Proyectos Alura\\Documento\\Información clinica.pdf"
    
    if os.path.exists(ruta_documento):
        contenido = extraer_texto_pdf(ruta_documento)
        
        if contenido:
            vectorstore = crear_base_conocimiento(contenido)
    else:
        print(f"⚠️ No se encontró el archivo en: {ruta_documento}")