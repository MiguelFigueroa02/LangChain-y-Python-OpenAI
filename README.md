# LangChain-y-Python-OpenAI

⚙️ Guía de Configuración

Siga los pasos a continuación para configurar su entorno y utilizar los scripts del proyecto.

1. Crear y Activar un Entorno Virtual

Windows:

python -m venv langchain
langchain\Scripts\activate

Mac/Linux:

python3 -m venv langchain
source langchain/bin/activate

2. Instalar Dependencias

Utilice el siguiente comando para instalar las bibliotecas necesarias:

pip install -r requirements.txt

3. Configurar la Clave de OpenAI

Cree o edite el archivo .env agregando su clave de API de OpenAI:

OPENAI_API_KEY="SU_CLAVE_DE_API"
