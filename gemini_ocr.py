import os
import glob
import base64
import time
import random
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from tqdm import tqdm

class RateLimiter:
    """
    Clase para controlar la velocidad de las solicitudes a la API.
    Implementa límites basados en:
    - RPD (Requests per Day): 25
    """
    
    def __init__(self, max_requests_per_day=25):
        self.max_requests_per_day = max_requests_per_day
        self.request_timestamps = []
        self.lock = __import__('threading').Lock()
    
    def _clean_old_timestamps(self):
        """Elimina timestamps más antiguos que 24 horas"""
        now = datetime.datetime.now()
        one_day_ago = now - datetime.timedelta(days=1)
        with self.lock:
            self.request_timestamps = [ts for ts in self.request_timestamps 
                                     if ts > one_day_ago]
    
    def wait_if_needed(self, interactive=False):
        """
        Espera si es necesario para respetar los límites de la API.
        
        Args:
            interactive (bool): Si es True, pregunta al usuario si desea continuar esperando
                               o cancelar el proceso cuando se alcanza el límite.
        
        Returns:
            tuple: (tiempo de espera en segundos, True si debe continuar, False si debe cancelar)
        """
        self._clean_old_timestamps()
        
        with self.lock:
            # Si no hemos alcanzado el límite diario, no es necesario esperar
            if len(self.request_timestamps) < self.max_requests_per_day:
                self.request_timestamps.append(datetime.datetime.now())
                return 0, True
            
            # Calcular cuánto tiempo debemos esperar
            oldest_timestamp = min(self.request_timestamps)
            now = datetime.datetime.now()
            one_day_from_oldest = oldest_timestamp + datetime.timedelta(days=1)
            
            if now < one_day_from_oldest:
                # Necesitamos esperar hasta que el timestamp más antiguo tenga más de 24 horas
                wait_seconds = (one_day_from_oldest - now).total_seconds()
                
                # Convertir segundos a un formato más legible
                hours, remainder = divmod(wait_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                wait_time_str = f"{int(hours)} horas, {int(minutes)} minutos y {int(seconds)} segundos"
                
                print("\n" + "="*80)
                print(f"⚠️  LÍMITE DIARIO ALCANZADO: Has llegado al límite de {self.max_requests_per_day} solicitudes por día.")
                print(f"⏱️  Tiempo de espera necesario: {wait_time_str}")
                print(f"📅  Podrás realizar más solicitudes a partir de: {one_day_from_oldest.strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80 + "\n")
                
                if interactive:
                    try:
                        response = input("¿Deseas esperar y continuar automáticamente? (s/n): ").strip().lower()
                        if response != 's' and response != 'si' and response != 'sí':
                            print("Proceso cancelado por el usuario.")
                            return wait_seconds, False
                    except KeyboardInterrupt:
                        print("\nProceso cancelado por el usuario.")
                        return wait_seconds, False
                
                print(f"Esperando {wait_time_str} para respetar los límites de la API...")
                time.sleep(wait_seconds)
                
                # Después de esperar, actualizamos y añadimos el nuevo timestamp
                self._clean_old_timestamps()
                self.request_timestamps.append(datetime.datetime.now())
                return wait_seconds, True
            else:
                # Ya ha pasado más de 24 horas desde la solicitud más antigua
                self.request_timestamps.pop(0)  # Eliminar el más antiguo
                self.request_timestamps.append(datetime.datetime.now())
                return 0, True

class GeminiOCRProcessor:
    """
    Procesador OCR que utiliza Google Gemini 2.5 Pro para extraer texto de imágenes y PDFs.
    """
    
    def __init__(self, api_key, max_workers=2, max_retries=5, max_requests_per_day=25, interactive=False):
        """
        Inicializa el procesador OCR con la API de Google Gemini.
        
        Args:
            api_key (str): API key de Google Gemini
            max_workers (int): Número máximo de trabajadores para procesamiento paralelo
            max_retries (int): Número máximo de reintentos para errores de cuota
            max_requests_per_day (int): Número máximo de solicitudes por día
            interactive (bool): Si es True, pregunta al usuario si desea continuar esperando
                               o cancelar el proceso cuando se alcanza el límite.
        """
        self.api_key = api_key
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.interactive = interactive
        genai.configure(api_key=self.api_key)
        
        # Configurar el modelo Gemini 2.5 Pro
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Inicializar el rate limiter
        self.rate_limiter = RateLimiter(max_requests_per_day=max_requests_per_day)
    
    def _encode_file(self, file_path):
        """
        Codifica un archivo para enviarlo a la API.
        
        Args:
            file_path (str): Ruta al archivo (imagen o PDF)
            
        Returns:
            dict: Objeto para la API de Gemini
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        mime_type = "application/pdf" if file_extension == ".pdf" else "image/png"
        
        with open(file_path, "rb") as file:
            return {"mime_type": mime_type, "data": file.read()}
    
    def process_pdf_direct(self, pdf_path, custom_prompt, language="Spanish"):
        """
        Procesa un archivo PDF completo directamente con Gemini.
        
        Args:
            pdf_path (str): Ruta al archivo PDF
            custom_prompt (str): Prompt personalizado para la extracción
            language (str): Idioma del texto a extraer
            
        Returns:
            str: Texto extraído del PDF
        """
        retry_count = 0
        base_wait_time = 2  # Tiempo base de espera en segundos
        
        while retry_count <= self.max_retries:
            try:
                # Esperar si es necesario para respetar los límites de la API
                wait_time, should_continue = self.rate_limiter.wait_if_needed(self.interactive)
                if not should_continue:
                    return f"Proceso cancelado por el usuario para {pdf_path}"
                
                # Preparar el PDF
                pdf_data = self._encode_file(pdf_path)
                
                # Crear el prompt completo
                full_prompt = f"{custom_prompt} El texto está en {language}."
                
                print(f"Procesando PDF completo: {os.path.basename(pdf_path)}")
                
                # Realizar la solicitud a la API
                response = self.model.generate_content(
                    contents=[
                        full_prompt,
                        pdf_data
                    ]
                )
                
                # Extraer el texto de la respuesta
                extracted_text = response.text
                return extracted_text
            
            except Exception as e:
                error_message = str(e)
                
                # Si es un error de cuota (429), implementar reintento con espera
                if "429" in error_message and retry_count < self.max_retries:
                    retry_count += 1
                    
                    # Extraer el tiempo de espera recomendado si está disponible
                    wait_time = None
                    if "retry_delay" in error_message and "seconds:" in error_message:
                        try:
                            wait_time_str = error_message.split("seconds:")[1].split("}")[0].strip()
                            wait_time = int(wait_time_str)
                        except:
                            pass
                    
                    # Si no se pudo extraer el tiempo de espera, usar espera exponencial
                    if not wait_time:
                        wait_time = base_wait_time * (2 ** retry_count) + random.uniform(0, 1)
                    
                    print(f"Error de cuota en {pdf_path}. Reintentando en {wait_time:.1f} segundos (intento {retry_count}/{self.max_retries})...")
                    time.sleep(wait_time)
                else:
                    # Si es otro tipo de error o se agotaron los reintentos, propagar el error
                    print(f"Error procesando {pdf_path}: {error_message}")
                    return f"Error procesando PDF: {error_message}"
        
        return f"Error: Se agotaron los reintentos para procesar {pdf_path}"
    
    def process_image(self, image_path, custom_prompt, language="Spanish"):
        """
        Procesa una imagen con Gemini para extraer texto.
        
        Args:
            image_path (str): Ruta a la imagen
            custom_prompt (str): Prompt personalizado para la extracción
            language (str): Idioma del texto a extraer
            
        Returns:
            str: Texto extraído de la imagen
        """
        retry_count = 0
        base_wait_time = 2  # Tiempo base de espera en segundos
        
        while retry_count <= self.max_retries:
            try:
                # Esperar si es necesario para respetar los límites de la API
                wait_time, should_continue = self.rate_limiter.wait_if_needed(self.interactive)
                if not should_continue:
                    return f"Proceso cancelado por el usuario para {image_path}"
                
                # Preparar la imagen
                image_data = self._encode_file(image_path)
                
                # Crear el prompt completo
                full_prompt = f"{custom_prompt} El texto está en {language}."
                
                # Realizar la solicitud a la API
                response = self.model.generate_content(
                    contents=[
                        full_prompt,
                        image_data
                    ]
                )
                
                # Extraer el texto de la respuesta
                extracted_text = response.text
                return extracted_text
            
            except Exception as e:
                error_message = str(e)
                
                # Si es un error de cuota (429), implementar reintento con espera
                if "429" in error_message and retry_count < self.max_retries:
                    retry_count += 1
                    
                    # Extraer el tiempo de espera recomendado si está disponible
                    wait_time = None
                    if "retry_delay" in error_message and "seconds:" in error_message:
                        try:
                            wait_time_str = error_message.split("seconds:")[1].split("}")[0].strip()
                            wait_time = int(wait_time_str)
                        except:
                            pass
                    
                    # Si no se pudo extraer el tiempo de espera, usar espera exponencial
                    if not wait_time:
                        wait_time = base_wait_time * (2 ** retry_count) + random.uniform(0, 1)
                    
                    print(f"Error de cuota en {image_path}. Reintentando en {wait_time:.1f} segundos (intento {retry_count}/{self.max_retries})...")
                    time.sleep(wait_time)
                else:
                    # Si es otro tipo de error o se agotaron los reintentos, propagar el error
                    print(f"Error procesando {image_path}: {error_message}")
                    return f"Error procesando imagen: {error_message}"
        
        return f"Error: Se agotaron los reintentos para procesar {image_path}"
    
    def process_batch(self, input_path, format_type="markdown", recursive=False, 
                     preprocess=False, custom_prompt="", language="Spanish"):
        """
        Procesa un lote de imágenes en una carpeta.
        
        Args:
            input_path (str): Ruta a la carpeta con imágenes
            format_type (str): Formato de salida (markdown por defecto)
            recursive (bool): Si se deben buscar imágenes en subcarpetas
            preprocess (bool): Si se debe preprocesar las imágenes
            custom_prompt (str): Prompt personalizado para la extracción
            language (str): Idioma del texto a extraer
            
        Returns:
            dict: Resultados del procesamiento por archivo
        """
        # Encontrar todas las imágenes
        pattern = os.path.join(input_path, "**/*.png" if recursive else "*.png")
        image_files = glob.glob(pattern, recursive=recursive)
        
        results = {"results": {}}
        
        # Procesar imágenes en paralelo
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Crear una lista de futuros
            future_to_file = {
                executor.submit(
                    self.process_image, 
                    image_file, 
                    custom_prompt,
                    language
                ): image_file for image_file in image_files
            }
            
            # Procesar resultados a medida que se completan
            for future in tqdm(future_to_file, desc="Procesando imágenes"):
                image_file = future_to_file[future]
                try:
                    result = future.result()
                    results["results"][image_file] = result
                    
                    # Verificar si el proceso fue cancelado por el usuario
                    if result.startswith("Proceso cancelado por el usuario"):
                        print("\nProcesamiento interrumpido por el usuario.")
                        break
                        
                except Exception as e:
                    print(f"Error procesando {image_file}: {str(e)}")
                    results["results"][image_file] = f"Error: {str(e)}"
        
        return results
