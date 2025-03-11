# Instalar uv si no existe
pip install uv

# Comprobar si el entorno virtual ya existe
if (Test-Path -Path .\ollama_env) {
    # Activar entorno virtual
    .\ollama_env\Scripts\Activate.ps1
} else {
    # Crear entorno virtual con uv (sintaxis corregida)
    uv venv ollama_env
    # Activar entorno virtual
    .\ollama_env\Scripts\Activate.ps1
}

# Instalar dependencias usando uv
uv pip install -r requirements.txt

# Ejecutar script principal con la ruta proporcionada
python main.py $args[0]
