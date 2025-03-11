# OCR Project

Este proyecto extrae texto de archivos PDF y lo convierte en formato Markdown, preservando el formato original (encabezados, listas, tablas, etc.).

## Requisitos

- Python 3.x
- Librerías: `fitz` (PyMuPDF), `cv2` (OpenCV), `numpy`, `base64`, `ollama`

## Instalación

1. Clona este repositorio:
    ```bash
    git clone https://github.com/tu_usuario/OCR.git
    cd OCR
    ```

2. Instala las dependencias:
    ```bash
    pip install pymupdf opencv-python numpy
    ```

## Uso

Ejecuta el script `main.py` proporcionando la ruta a la carpeta que contiene los archivos PDF:

```bash
python main.py [ruta_carpeta_pdfs]
```

Por ejemplo:

```bash
python main.py ./pdfs
```

## Salida

El script generará archivos `.md` en la misma carpeta que los archivos PDF originales, con el contenido extraído en formato Markdown.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para discutir cualquier cambio que desees realizar.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT.
