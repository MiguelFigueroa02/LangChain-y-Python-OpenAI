from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader,PyMuPDFLoader
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0.5,
    api_key=api_key
)

embeddings = OpenAIEmbeddings()

archivos = [
    "./documentos/GTB_gold.pdf",
    "./documentos/GTB_platinum.pdf",
    "./documentos/GTB_standard.pdf"
]

documentos = sum(
    [PyMuPDFLoader(archivo).load() for archivo in archivos],
    []
)

fragmentos=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documentos)

recuperador=FAISS.from_documents(
    fragmentos,
    embeddings
).as_retriever(search_kwargs={"k":2})

prompt_consulta = ChatPromptTemplate.from_messages([
    ("system","Responda únicamente usando el contexto proporcionado"),
    ("human","{query}\n\nContexto:\n{contexto}\n\nRespuesta:")
])

cadena = prompt_consulta | modelo | StrOutputParser()

def responder(pregunta:str):
    fragmentos_recuperados = recuperador.invoke(pregunta)

    contexto = "\n\n".join(
        un_fragmento.page_content for un_fragmento in fragmentos_recuperados
    )

    return cadena.invoke({
        "query":pregunta,
        "contexto":contexto
    })

print(responder("¿Qué hacer si me roban un objeto y tengo tarjeta Standard?"))

