# OCR Project

Este proyecto extrae texto de archivos PDF y lo convierte en formato Markdown, preservando el formato original (encabezados, listas, tablas, etc.) utilizando Google Gemini 2.5 Pro.

## Requisitos

- Python 3.x
- Librerías: `google-generativeai`, `fitz` (PyMuPDF), `cv2` (OpenCV), `numpy`, `tqdm`

## Instalación

1. Clona este repositorio:
    ```bash
    git clone https://github.com/tu_usuario/OCR.git
    cd OCR
    ```

2. Crea un entorno virtual e instala las dependencias:
    ```bash
    python -m venv .cline_venv
    # En Windows
    .\.cline_venv\Scripts\activate
    # En Linux/Mac
    source .cline_venv/bin/activate
    
    pip install -r requirements.txt
    ```

## Configuración

El proyecto utiliza la API de Google Gemini. La clave API ya está configurada en el código, pero si necesitas cambiarla, puedes modificar la variable `API_KEY` en el archivo `main.py`.

## Uso

Ejecuta el script `main.py` proporcionando la ruta a la carpeta que contiene los archivos PDF:

```bash
python main.py [ruta_carpeta_pdfs]
```

Por ejemplo:

```bash
python main.py ./pdfs
```

## Funcionamiento

1. El script procesa cada archivo PDF en la carpeta especificada.
2. Cada página del PDF se convierte en una imagen PNG.
3. Google Gemini 2.5 Pro analiza cada imagen y extrae el texto con formato.
4. El resultado se guarda en un archivo Markdown con el mismo nombre que el PDF original.

## Salida

El script generará archivos `.md` en la misma carpeta que los archivos PDF originales, con el contenido extraído en formato Markdown.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para discutir cualquier cambio que desees realizar.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT.
