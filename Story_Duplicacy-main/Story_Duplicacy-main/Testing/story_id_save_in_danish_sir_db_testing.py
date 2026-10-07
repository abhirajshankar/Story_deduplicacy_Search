from zeep import Client
import os
from dotenv import load_dotenv

load_dotenv()


wsdl = os.getenv("DATA_SAVE_API_URL")

print(f"API Key loaded: {wsdl}")

client = Client(wsdl)

print(client.wsdl.services)