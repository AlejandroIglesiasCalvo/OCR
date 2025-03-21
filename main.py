import sys
import fitz
import cv2
import numpy as np
import base64
import ollama
from pathlib import Path
import threading
from tqdm import tqdm
import time
import multiprocessing  # ***CAMBIO*** para subprocesos

MODEL_NAME = "llama3.2-vision:11b"
PROMPT = (
    "Extrae únicamente el texto e imágenes visibles en la imagen. "
    "El resultado debe estar completamente en español y en formato Markdown, "
    "preservando exactamente el formato original, incluyendo encabezados, listas, tablas, negrita, cursiva y cualquier otro estilo presente. "
    "Si alguna parte es ilegible, márcala como 'texto ilegible'. "
    "No incluyas ningún contenido adicional ni información sobre herramientas externas o detalles que no estén presentes en la imagen."
)
TIMEOUT = 900  # segundos

def preprocess_image(page):
    """
    Preprocesamiento menos agresivo para preservar más detalles,
    inspirado en la versión antigua que realizaba un mejor OCR.
    """
    # Renderizamos la página a alta resolución
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pix.tobytes("png")
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convertimos a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicamos un filtro de mediana para reducir el ruido sin perder detalles finos
    processed = cv2.medianBlur(gray, 3)

    # Mejoramos el contraste con ecualización del histograma
    processed = cv2.equalizeHist(processed)

    return processed

# Función auxiliar que se ejecutará en un subproceso
def run_ollama_subprocess(model_name, prompt, b64_img, return_dict):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt, 'images': [b64_img]}]
        )
        return_dict["content"] = response['message']['content']
    except Exception as e:
        return_dict["error"] = str(e)

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    md_output = []
    total_pages = len(doc)

    for i, page in enumerate(doc):
        print(f"Procesando página {i + 1}/{total_pages}")

        stop_event = threading.Event()
        p = None  # Inicializamos la variable para el proceso hijo

        def progress_bar():
            with tqdm(total=TIMEOUT, desc="Tiempo restante", unit="s") as pbar:
                for _ in range(TIMEOUT):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                    pbar.update(1)

        progress_thread = threading.Thread(target=progress_bar)
        progress_thread.start()

        try:
            clean_image = preprocess_image(page)
            _, png_bytes = cv2.imencode(".png", clean_image)
            b64_img = base64.b64encode(png_bytes).decode('utf-8')

            # Usamos multiprocessing para ejecutar ollama.chat
            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            p = multiprocessing.Process(
                target=run_ollama_subprocess,
                args=(MODEL_NAME, PROMPT, b64_img, return_dict)
            )
            p.start()

            # Esperamos hasta TIMEOUT segundos
            p.join(TIMEOUT)

            # Si el proceso sigue vivo, lo terminamos
            if p.is_alive():
                print(f"Tiempo excedido para la página {i + 1}, terminando proceso.")
                p.terminate()
                p.join()
            else:
                # Si no hay error, agregamos el contenido devuelto
                if "error" in return_dict:
                    print(f"Error en la página {i + 1}: {return_dict['error']}")
                else:
                    md_output.append(return_dict["content"])

        except KeyboardInterrupt:
            print("\nInterrupción por teclado detectada. Terminando proceso actual...")
            if p is not None and p.is_alive():
                p.terminate()
                p.join()
            raise  # Propagamos la excepción para finalizar todo el programa
        except Exception as e:
            print(f"Error en la página {i + 1}: {e}")
        finally:
            stop_event.set()
            progress_thread.join()

    return "\n\n".join(md_output)

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
        except KeyboardInterrupt:
            print("Interrupción por teclado. Finalizando procesamiento de PDFs.")
            sys.exit(0)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"Guardado: {output_path}")

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
