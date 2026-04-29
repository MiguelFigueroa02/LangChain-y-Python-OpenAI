import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
clave_api = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=clave_api
)

prompt_sugerencia = ChatPromptTemplate.from_messages([
    ("system", "Eres un guía de viajes especializado en destinos latinos. Preséntate como sr. Paseos."),
    ("placeholder", "{historial}"),
    ("human", "{consulta}")
])

cadena = prompt_sugerencia | modelo | StrOutputParser()

memoria = {}

sesion = "aula_langchain_chat"

def historial_por_sesion(sesion: str):
    if sesion not in memoria:
        memoria[sesion] = InMemoryChatMessageHistory()
    return memoria[sesion]

lista_preguntas = [
    "Quiero visitar un lugar en latinoamerica famoso por playas y cultura. Puedes sugerir uno?",
    "Cuál es la mejor epoca del año para visitarlo?"
]

cadena_con_memoria = RunnableWithMessageHistory(
    runnable=cadena,
    get_session_history=historial_por_sesion,
    input_messages_key="consulta",
    history_messages_key="historial"
)

for pregunta in lista_preguntas:

    respuesta = cadena_con_memoria.invoke(
        {"consulta": pregunta},
        config={"configurable": {"session_id": sesion}}
    )

    print("Usuario: ", pregunta)
    print("IA: ", respuesta)
    print()