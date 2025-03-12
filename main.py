import sys
import fitz, cv2, numpy as np, base64, ollama
from pathlib import Path
import threading
from tqdm import tqdm
import time

MODEL_NAME = "llama3.2-vision:11b"
PROMPT = ("Extrae todo el texto de la imagen en español y conviértelo a formato Markdown, "
          "preservando el formato original, incluyendo encabezados, listas, tablas, "
          "negritas, cursivas y cualquier otro estilo de formato presente en el documento.")
TIMEOUT = 300  # segundos

def preprocess_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pix.tobytes("png")
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=15)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > 0.1:
        h, w = thresh.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_LINEAR)

    return thresh

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    md_output = []
    total_pages = len(doc)

    for i, page in enumerate(doc):
        print(f"Procesando página {i + 1}/{total_pages}")

        timeout_occurred = False
        stop_event = threading.Event()

        def timeout_handler():
            nonlocal timeout_occurred
            timeout_occurred = True
            stop_event.set()

        timer = threading.Timer(TIMEOUT, timeout_handler)
        timer.start()

        # Barra de progreso que se detiene inmediatamente al terminar el OCR o timeout
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

            response = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'user', 'content': PROMPT, 'images': [b64_img]}
            ])

            if not timeout_occurred:
                md_output.append(response['message']['content'])
            else:
                print(f"Tiempo excedido para la página {i + 1}, saltando a la siguiente.")

        except Exception as e:
            print(f"Error en la página {i + 1}: {e}")
        finally:
            timer.cancel()
            stop_event.set()  # Asegurar que la barra se detenga inmediatamente
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

        md_content = process_pdf(pdf_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"Guardado: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python main.py [ruta_carpeta_pdfs]")
        sys.exit(1)

    pdf_folder = sys.argv[1]
    main(pdf_folder)
