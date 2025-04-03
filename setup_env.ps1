# Comprobar si el entorno virtual ya existe
if (-not (Test-Path -Path .\.cline_venv)) {
    Write-Host "Creando entorno virtual .cline_venv..." -ForegroundColor Green
    python -m venv .cline_venv
}

# Comprobar si el entorno virtual está activado
if (-not ($env:VIRTUAL_ENV -like "*cline_venv*")) {
    Write-Host "Activando entorno virtual .cline_venv..." -ForegroundColor Green
    .\.cline_venv\Scripts\Activate.ps1
}

# Instalar dependencias
Write-Host "Instalando dependencias..." -ForegroundColor Green
pip install -r requirements.txt

# Verificar que se proporcionó una ruta como parámetro
if (-not $args[0]) {
    Write-Host "Error: Debes proporcionar la ruta al directorio que contiene los archivos PDF." -ForegroundColor Red
    Write-Host "Uso: .\setup_env.ps1 [ruta_carpeta_pdfs]" -ForegroundColor Yellow
    exit 1
}

# Ejecutar script principal con la ruta proporcionada
Write-Host "Ejecutando OCR en los archivos PDF de $($args[0])..." -ForegroundColor Green
python main.py $args[0]
