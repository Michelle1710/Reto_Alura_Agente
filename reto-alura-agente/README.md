
## 📌 Descripción General del Proyecto
🤖 Agente IA - RAG Corporativo (Reto Alura)

Este proyecto es un Asistente Virtual Inteligente creado para una clínica médica. Utiliza Arquitectura RAG (Generación Aumentada por Recuperación) para responder preguntas basadas en un PDF corporativo y tiene la capacidad de agendar citas médicas guardándolas automáticamente en un archivo de Excel.

## 🛠️ Tecnologías utilizadas
* Python

## 🏗️ Arquitectura de la Solución
La solución sigue un flujo modular de Recuperación y Generación Aumentada:
1. **Ingesta y Procesamiento (`procesar_datos.py`):** Lee el documento oficial en formato PDF, lo divide en fragmentos manejables mediante un `RecursiveCharacterTextSplitter`, genera los embeddings vectoriales con Cohere y los almacena localmente en una base de datos vectorial (**ChromaDB**).
2. **Motor de Búsqueda y Herramientas (`app.py`):** Utiliza un recuperador (`retriever`) conectado a ChromaDB y una herramienta de persistencia en archivos (Excel/CSV) para el agendamiento.
3. **Agente Autónomo y UI (`Streamlit`):** Un modelo de lenguaje avanzado (**Cohere Command-R**) orquesta las herramientas mediante un marco de agentes autónomos, expuesto a través de una interfaz gráfica web interactiva desarrollada en **Streamlit**.

---

## 🛠️ Tecnologías utilizadas
* Python
* LangChain & LangGraph
* Cohere (LLM y Embeddings)
* Streamlit (Interfaz gráfica y despliegue)
* Pandas (Manejo de datos en Excel)
* PyPDF (Lectura de documentos)

## 🚀 Despliegue en Vivo
Puedes probar el agente funcionando en la nube aquí: 
**[(https://retoaluraagente-53ppujagipbrqndbyh37ja.streamlit.app/)]**

## 📂 Características Principales
1. **Consulta de información:** Responde dudas sobre horarios y políticas leyendo directamente el documento clínico.
2. **Uso de Herramientas (Tool Calling):** Recopila paso a paso el nombre, fecha, hora y especialidad del paciente para agendar una cita.
3. **Memoria Conversacional:** Recuerda el hilo de la charla y filtra los pensamientos internos del modelo para dar una respuesta humana.
---

## 🚀 Instrucciones para Ejecutar el Proyecto

Sigue estos pasos en tu terminal para poner en marcha la aplicación localmente:

1. **Clona el repositorio e ingresa al directorio:**
   ```bash
   git clone <https://github.com/Michelle1710/Reto_Alura_Agente>
   cd reto-alura-agente

Crea y activa tu entorno virtual
   python -m venv env
# En Windows:
.\env\Scripts\activate
# En Mac/Linux:
source env/bin/activate

Instala las dependencias necesarias:

Bash
pip install -r requirements.txt

Configura tus credenciales:
Crea un archivo llamado .env en la raíz del proyecto y añade tu API Key de Cohere:

Fragmento de código
COHERE_API_KEY="tu_clave_de_api_aqui

Procesa el documento fuente (Genera la base vectorial):

Bash
python procesar_datos.py
Ejecuta la aplicación web:

Bash
streamlit run app.py

Ejemplos de Preguntas que el Agente Puede Responder
"¿Cuáles son los horarios de atención y días laborables de la clínica?"

"¿Qué dicen las políticas de privacidad sobre los datos personales de los pacientes?"

"¿Cuáles son las instrucciones a seguir para las preconsultas?"

📝 Ejemplos de Respuestas Generadas por el Agente
Consulta de Información (Vía RAG):

Usuario: ¿Cuáles son los horarios de atención?
Agente: La clínica atiende de lunes a viernes de 08:00 a.m. a 08:00 p.m. y los sábados de 08:00 a.m. a 02:00 p.m. Los domingos y días festivos la clínica permanece cerrada para consultas externas.

Gestión de Citas (Vía Tool Calling):

Usuario: Quiero agendar una cita médica para Michelle el 24 de julio para la especialidad de traumatología.
Agente: ¡Perfecto, Michelle! Tu cita para la especialidad de Traumatología ha sido agendada con éxito para el día 24 de julio y registrada en nuestro sistema.

Evidencia de funcionamiento 
https://docs.google.com/document/d/1HmKM-1VpnxuaOfgeHSYm_ylJUkNWpn1p9Pp4WLeEBOM/edit?tab=t.0