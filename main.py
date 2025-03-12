import sys
import fitz
import cv2
import numpy as np
import base64
from pathlib import Path
import threading
from tqdm import tqdm
import time
import multiprocessing
from ollama_ocr import OCRProcessor  # Nuevo import

MODEL_NAME = "granite3.2-vision"
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

# Se elimina la función run_ollama_subprocess ya no necesaria

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    # Crear carpeta temporal para las imágenes extraídas
    temp_dir = pdf_path.parent / (pdf_path.stem + "_temp")
    temp_dir.mkdir(exist_ok=True)
    image_paths = []
    total_pages = len(doc)
    for i, page in enumerate(doc):
        print(f"Procesando página {i + 1}/{total_pages}")
        clean_image = preprocess_image(page)
        _, png_bytes = cv2.imencode(".png", clean_image)
        img_path = temp_dir / f"page_{i+1:03d}.png"
        with open(img_path, "wb") as f:
            f.write(png_bytes.tobytes())
        image_paths.append(str(img_path))
    
    # Inicializar OCRProcessor y procesar en lote
    ocr = OCRProcessor(model_name=MODEL_NAME, max_workers=4)
    batch_results = ocr.process_batch(
        input_path=str(temp_dir),
        format_type="markdown",
        recursive=False,
        preprocess=False,
        custom_prompt=PROMPT,
        language="Spanish"
    )
    
    # Recopilar resultados ordenados por nombre (asumiendo numeración en el nombre del archivo)
    md_output = []
    for img_file in sorted(batch_results['results'].keys()):
        md_output.append(batch_results['results'][img_file])
    
    # ...opcional: código para eliminar la carpeta temporal si se desea...
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
