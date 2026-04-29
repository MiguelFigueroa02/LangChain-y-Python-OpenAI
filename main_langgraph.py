import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)

prompt_consultor_playa = ChatPromptTemplate.from_messages([
    ("system", "Preséntate como Sra. Playa. Eres una especialista de viajes con destinos playa."),
    ("human", "{query}")
])

prompt_consultor_montana = ChatPromptTemplate.from_messages([
    ("system", "Preséntate como Sr. Montaña. Eres una especialista de viajes a montañas y actividades de aventura."),
    ("human", "{query}")
])

cadena_playa = prompt_consultor_playa | modelo | StrOutputParser()

cadena_montana = prompt_consultor_montana | modelo | StrOutputParser()

prompt_ruteador = ChatPromptTemplate.from_messages([
    ("system", "Responde únicamente con 'playa' o 'montaña'."),
    ("human", "{query}")
])

class Ruta(TypedDict):
    destino: Literal["playa", "montaña"]

ruteador = prompt_ruteador | modelo.with_structured_output(Ruta)

class Estado(TypedDict):
    query: str
    destino: dict
    respuesta: str

async def nodo_roteador(estado: Estado, config: RunnableConfig):
    return {
        "destino": await ruteador.ainvoke(
            {"query": estado["query"]},
            config
        )
    }

async def nodo_playa(estado: Estado, config: RunnableConfig):
    return {
        "respuesta": await cadena_playa.ainvoke(
            {"query": estado["query"]},
            config
        )
    }

async def nodo_montana(estado: Estado, config: RunnableConfig):
    return {
        "respuesta": await cadena_montana.ainvoke(
            {"query": estado["query"]},
            config
        )
    }

def elegir_nodo(estado: Estado) -> Literal["playa", "montaña"]:
    return "playa" if estado["destino"]["destino"] == "playa" else "montaña"

grafo = StateGraph(Estado)

grafo.add_node("rotear", nodo_roteador)
grafo.add_node("playa", nodo_playa)
grafo.add_node("montaña", nodo_montana)

grafo.add_edge(START, "rotear")

grafo.add_conditional_edges("rotear", elegir_nodo)

grafo.add_edge("playa", END)
grafo.add_edge("montaña", END)

app = grafo.compile()

async def main():
    respuesta = await app.ainvoke({
        "query": "Quiero hacer deporte de aventura en la Cordillera de los Andes"
    })

    print(respuesta["respuesta"])

asyncio.run(main())
