"""Conexión con el modelo de lenguaje (LLM).

Usa el mismo proveedor y modelo que el Reto 1 (catálogo de NVIDIA).
"""
import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]  # carpeta "Reto 2"
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / "Reto 1" / ".env")  # si no hay .env propio, usa el del Reto 1

MODELO = "mistralai/mistral-large-4-1t-a52b"  # Cambie el modelo según su proveedor


def llamar_llm(sistema: str, mensajes: list[dict[str, Any]], max_tokens: int = 1024) -> str:
    """Envía un prompt de sistema y una lista de mensajes al LLM y devuelve el texto."""
    if not mensajes:
        raise ValueError("La lista de mensajes no puede estar vacía.")
    if os.environ.get("NVIDIA_API_KEY"):
        # La librería aún no reconoce este modelo y avisa en cada llamada
        warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        cliente = ChatNVIDIA(model=MODELO, temperature=0, max_completion_tokens=max_tokens, timeout=300)
        respuesta = cliente.invoke([{"role": "system", "content": sistema}, *mensajes])
        return str(respuesta.text)
    # Modo de prueba: sin clave de API devuelve una respuesta simulada
    ultimo_mensaje = mensajes[-1]
    contenido = ultimo_mensaje.get("content", "")
    if isinstance(contenido, list):
        piezas = []
        for item in contenido:
            if isinstance(item, dict):
                pieza = item.get("text", str(item))
                piezas.append(str(pieza))
            else:
                piezas.append(str(item))
        contenido = " ".join(piezas)
    return "[respuesta simulada] " + str(contenido)[:200]
