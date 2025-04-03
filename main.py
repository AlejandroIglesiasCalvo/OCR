import sys
from pathlib import Path
import time
import os
from dotenv import load_dotenv
from gemini_ocr import GeminiOCRProcessor

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_RETRIES = 5  # Número máximo de reintentos para errores de cuota
MAX_WORKERS = 2  # Reducido para evitar sobrecargar la API
MAX_REQUESTS_PER_DAY = 25  # Límite de solicitudes por día según la documentación de Google
INTERACTIVE = True  # Permite al usuario decidir si continuar o no cuando se alcanza el límite diario
PROMPT = (
    "Extrae únicamente el texto e imágenes visibles en el documento. "
    "El resultado debe estar completamente en español y en formato Markdown, "
    "preservando exactamente el formato original, incluyendo encabezados, listas, tablas, negrita, cursiva y cualquier otro estilo presente. "
    "Si alguna parte es ilegible, márcala como 'texto ilegible'. "
    "No incluyas ningún contenido adicional ni información sobre herramientas externas o detalles que no estén presentes en el documento."
)

def process_pdf(pdf_path):
    """
    Procesa un archivo PDF completo directamente con la API de Google Gemini.
    
    Args:
        pdf_path (Path): Ruta al archivo PDF a procesar
        
    Returns:
        str: Contenido extraído en formato Markdown
    """
    print(f"Procesando PDF: {pdf_path.name}")
    
    # Inicializar GeminiOCRProcessor
    ocr = GeminiOCRProcessor(
        api_key=API_KEY, 
        max_workers=MAX_WORKERS, 
        max_retries=MAX_RETRIES,
        max_requests_per_day=MAX_REQUESTS_PER_DAY,
        interactive=INTERACTIVE
    )
    
    print(f"Iniciando procesamiento OCR con límite de {MAX_REQUESTS_PER_DAY} solicitudes por día.")
    print(f"El proceso puede tomar tiempo si se alcanzan los límites de la API.")
    if INTERACTIVE:
        print("Modo interactivo activado: se te preguntará si deseas continuar cuando se alcance el límite diario.")
    
    # Procesar el PDF completo directamente
    result = ocr.process_pdf_direct(
        pdf_path=str(pdf_path),
        custom_prompt=PROMPT,
        language="Spanish"
    )
    
    # Verificar si el proceso fue cancelado por el usuario
    if result.startswith("Proceso cancelado por el usuario"):
        print(f"Procesamiento cancelado por el usuario.")
        raise Exception("Procesamiento cancelado por el usuario.")
        
    # Verificar si hay errores en el resultado
    if result.startswith("Error:"):
        print(f"Advertencia: {result}")
    
    return result

def main(pdf_folder):
    pdf_dir = Path(pdf_folder)
    for pdf_path in pdf_dir.glob('*.pdf'):
        print(f"Procesando: {pdf_path.name}")
        output_path = pdf_path.with_suffix('.md')
        if output_path.exists():
            print(f"El archivo {output_path.name} ya existe. Saltando...")
            continue

        try:
            md_content = process_pdf(pdf_path)
            
            # Guardar el contenido
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            print(f"Guardado: {output_path}")
            
        except KeyboardInterrupt:
            print("Interrupción por teclado. Finalizando procesamiento de PDFs.")
            sys.exit(0)
        except Exception as e:
            print(f"Error procesando {pdf_path.name}: {str(e)}")
            if "Procesamiento cancelado por el usuario" in str(e):
                print("Finalizando procesamiento de PDFs por solicitud del usuario.")
                sys.exit(0)
            print("Continuando con el siguiente archivo...")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python main.py [ruta_carpeta_pdfs]")
        sys.exit(1)

    pdf_folder = sys.argv[1]
    try:
        main(pdf_folder)
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
        sys.exit(0)
