from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destino(BaseModel):
    ciudad: str = Field(description="La ciudad recomendada para visitar")
    motivo: str = Field(description= "Razón por la que es interesante visitar esta ciudad")

class Restaurantes(BaseModel):
    ciudad: str = Field(description="Ciudad recomendada")
    restaurantes: str = Field(description="Restaurantes populares en la ciudad")

parser_destino = JsonOutputParser(
    pydantic_object=Destino
)

parser_restaurantes = JsonOutputParser(
    pydantic_object=Restaurantes
)

modelo=ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0.5,
    api_key=api_key
)

plantilla_ciudad=PromptTemplate(
    template="""
         Sugiere una ciudad en Latinoamérica para viajar
         basada en mi interés {interes}.

         # Formato de Salida
         {instrucciones_formato}
        """,
        input_variables=["interes"],
        partial_variables={
            "instrucciones_formato":parser_destino.get_format_instructions()
        }
)

plantilla_restaurantes=PromptTemplate(
    template="""
         Sugiere restaurantes entre los locales en {ciudad}.

         # Formato de Salida
         {instrucciones_formato}
        """,
        input_variables=["ciudad"],
        partial_variables={
            "instrucciones_formato":parser_restaurantes.get_format_instructions()
        }
)

plantilla_cultural=PromptTemplate(
    template="""
         Sugiere actividades y lugares culturales para visitar en {ciudad}.
        """,
        input_variables=["ciudad"]
)


cadena1 = plantilla_ciudad | modelo | parser_destino
cadena2 = plantilla_restaurantes | modelo | parser_restaurantes
cadena3 = plantilla_cultural | modelo | StrOutputParser()

cadena = (cadena1 | cadena2 | cadena3)

respuesta= cadena.invoke(
    {"interes":"ciudades históricas "}
)
print(respuesta)
