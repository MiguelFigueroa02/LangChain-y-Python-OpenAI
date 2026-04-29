from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pais="Perú"
numero_dias= 6
perfil= "pareja joven"
intereses="cultura, gastronomía e historia"

prompt=(
    f"Crea un itinerario de viajes de {numero_dias} días en {pais}"
    f"Para una {perfil}, con foco en {intereses}."
    f"Incluye ciudades, actividades y experiencias culturales."
)

cliente = OpenAI(api_key=api_key)

