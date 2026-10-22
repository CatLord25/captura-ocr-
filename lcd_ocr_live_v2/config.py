"""Configuración central de LCD OCR Live v2 (Raspberry Pi 4B + OV5647 / PC).

Todas las dimensiones de imagen se expresan como ``(ancho, alto)``.
Los colores del HUD se expresan en BGR porque los dibuja OpenCV.
"""

from pathlib import Path
import unicodedata

RAIZ_PROYECTO = Path(__file__).resolve().parent
DIRECTORIO_CAPTURAS = RAIZ_PROYECTO / "assets" / "capturas"
CAPTURA_ACTUAL = DIRECTORIO_CAPTURAS / "captura_actual.jpg"
ARCHIVO_ROIS = RAIZ_PROYECTO / "data" / "rois.json"
ARCHIVO_LECTURAS = RAIZ_PROYECTO / "data" / "lecturas.csv"
ARCHIVO_PERSPECTIVA = RAIZ_PROYECTO / "data" / "perspectiva.json"

# Rutas relativas al proyecto: el dataset viaja con el código.
DATASET_BALANZA = "datasets/Balanza"
DATASET_TERMOHIGRO = "datasets/termohiglobarometro"

# ---------------------------------------------------------------- instrumentos
# Perfil de preprocesado/reconocimiento por instrumento.
PERFIL_POR_INSTRUMENTO = {
    "Balanza": "lcd",
    "termohigrobarometro": "7seg",
}
INSTRUMENTOS = ("Balanza", "Termohigrobarómetro")

# Roles de medición reconocidos por la capa de instrumentos.
ROLES = {
    "peso": "Peso",
    "temperatura": "Temperatura",
    "humedad": "Humedad",
    "presion": "Presión",
    "otro": "Otro",
}
# Unidades por defecto cuando el OCR no detecta el símbolo en la ROI.
UNIDAD_POR_ROL = {
    "peso": "kg",
    "temperatura": "°C",
    "humedad": "%",
    "presion": "hPa",
    "otro": "",
}
# Heurística por nombre de ROI para asignar el rol automáticamente.
CLAVES_ROL = (
    ("temperatura", ("temp", "t_", "temperatura", "celsius", "fahrenheit")),
    ("humedad", ("hum", "rh", "humedad", "relative")),
    ("presion", ("pres", "baro", "presion", "pressure", "hpa")),
    ("peso", ("bal", "peso", "masa", "weight", "kg", "gram")),
)

# -------------------------------------------------------------------- cámara
CAMARA_INDICE = 0
# "auto" | "v4l2" | "gstreamer" | "picamera2" | "dshow" | "msmf" | "any"
CAMARA_BACKEND = "auto"
# Resolución de trabajo: 720p a 30 FPS mantiene la CPU de la RPi 4B holgada
# y conserva nitidez suficiente en los segmentos del display.
CAMARA_RESOLUCION = (1280, 720)
CAMARA_RESOLUCION_ALTERNATIVA = (960, 720)
CAMARA_FPS = 30
CAMARA_BUFFERSIZE = 1
CAMARA_CUATROCC = "MJPG"
CAMARA_ESTABILIZACION_SEGUNDOS = 0.8
CAMARA_REINTENTOS = 2
CAMARA_TIMEOUT_LECTURA_S = 2.0

# Órdenes de intento por plataforma (el primero que abra gana).
CAMARA_BACKENDS_LINUX = ("picamera2", "gstreamer", "v4l2")
CAMARA_BACKENDS_WINDOWS = ("dshow", "msmf", "any")
PIPELINE_GSTREAMER = (
    "libcamerasrc ! video/x-raw,width={ancho},height={alto},"
    "framerate={fps}/1 ! videoconvert ! appsink drop=true max-buffers=1 sync=false"
)

# Presets de sensor para evitar que la iluminación LED/IR "queme" los dígitos.
AJUSTES_SENSOR_DIA = {
    "auto_exposicion": True,
    "brillo": 128,
    "contraste": 128,
    "ganancia": 20,
}
AJUSTES_SENSOR_NOCHE = {
    "auto_exposicion": False,
    "exposicion": 250,       # V4L2: 100 µs por unidad -> 25 ms
    "brillo": 96,
    "contraste": 160,
    "ganancia": 90,
}
# En picamera2 la exposición se expresa en microsegundos.
EXPOSICION_US_NOCHE = 25_000
GANANCIA_ANALOGICA_NOCHE = 4.0

# --------------------------------------------------------- plantillas / OCR
TAMANO_PLANTILLA = (32, 48)
NORMALIZACION_AREA_RELATIVA_MINIMA = 0.03
NORMALIZACION_MARGEN_RELATIVO = 1.0 / 14.0
AR_ESTRECHO = 0.38
AR_ESTRECHO_OBJETIVO = 0.46        # relleno negro lateral mínimo para un "1"
K_VECINOS = 3
ESCALA_DISTANCIA_KNN = 0.15
DESPLAZAMIENTO_PLANTILLA = 2
AUMENTO_DESPLAZAMIENTOS = (-2, 0, 2)
UMBRAL_CONFIANZA = 0.70
UMBRAL_CONFIANZA_VERDE = 0.80
UMBRAL_CONFIANZA_AMARILLO = 0.60
PESOS_FUSION = {"template": 0.50, "knn": 0.30, "topologia": 0.20}
PESOS_FUSION_UNIDAD = {"template": 0.65, "knn": 0.35}
BONIFICACION_TOPOLOGIA = 0.10
PENALIZACION_TOPOLOGIA = 0.08

# ------------------------------------------------------- preprocesado / ROI
THRESHOLD_BINARIO = 127
KERNEL_MORFOLOGICO = 3
INVERTIR_BINARIO = False
BLOQUE_ADAPTATIVO = 31
CONSTANTE_ADAPTATIVA = 7
CLAHE_ACTIVO = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
DESENFOQUE_KERNEL = 3
PUENTE_7SEG = (3, 5)              # kernel elíptico (ancho, alto)
PUENTE_LCD = (2, 3)
SEGMENTACION_AREA_MINIMA = 6
SEGMENTACION_SEPARACION_MAXIMA = 3
SEGMENTACION_SOLAPE_MINIMO = 0.45
SEGMENTACION_ANCHO_MAXIMO_DIGITO = 1.10   # veces la altura del dígito
SEGMENTACION_VALLE_VENTANA = 0.22
GEOM_PUNTO_Y_MIN = 0.55
GEOM_PUNTO_ALTO_MAX = 0.35
GEOM_PUNTO_AR_MAX = 1.30
GEOM_MENOS_Y_MIN = 0.30
GEOM_MENOS_Y_MAX = 0.70
GEOM_MENOS_AR_MIN = 1.70
GEOM_MENOS_ALTO_MAX = 0.50
GEOM_DIGITO_ALTO_MIN = 0.55
GEOM_DIGITO_AR_MAX = 0.95

# Ventanas normalizadas de los siete segmentos (x0, y0, x1, y1) sobre 32×48.
SEGMENTOS_7 = {
    "A": (0.16, 0.03, 0.84, 0.15),
    "B": (0.70, 0.10, 0.94, 0.47),
    "C": (0.70, 0.53, 0.94, 0.90),
    "D": (0.16, 0.85, 0.84, 0.97),
    "E": (0.06, 0.53, 0.30, 0.90),
    "F": (0.06, 0.10, 0.30, 0.47),
    "G": (0.16, 0.44, 0.84, 0.56),
}
PATRONES_7 = {
    "0": "ABCDEF", "1": "BC", "2": "ABGED", "3": "ABGCD", "4": "FGBC",
    "5": "AFGCD", "6": "AFGEDC", "7": "ABC", "8": "ABCDEFG", "9": "ABCDFG",
}
SEGMENTO_DENSIDAD_MIN = 0.16
SEGMENTO_TOLERANCIA = 0.30
CONFUSIONES_7 = (("8", "0"), ("8", "6"), ("6", "5"), ("9", "5"),
                 ("9", "8"), ("3", "2"), ("8", "9"), ("0", "6"))

# ------------------------------------------------------------- temporal/CSV
ESTABILIZACION_MUESTRAS = 5
ESTABILIZACION_MINIMO_ACUERDOS = 3
INTERVALOS_GUARDADO = (
    ("Tiempo real", 0.0),
    ("1 s", 1.0),
    ("5 s", 5.0),
    ("10 s", 10.0),
    ("60 s", 60.0),
)
INTERVALO_GUARDADO_POR_DEFECTO = 1.0
CSV_CABECERA = ("timestamp", "instrumento", "roi", "valor", "confianzas")

# ----------------------------------------------------------------------- HUD
HUD_ANCHO_PANEL = 470
HUD_ALTO_MINIMO = 480
HUD_TITULO = "LCD OCR LIVE v2 — Balanza + Termohigrobarómetro"
HUD_COLOR_FONDO = (18, 14, 12)          # BGR casi negro azulado
HUD_COLOR_PANEL = (34, 26, 22)
HUD_COLOR_TARJETA = (48, 38, 32)
HUD_COLOR_TEXTO = (238, 240, 242)
HUD_COLOR_TEXTO_TENUE = (150, 156, 162)
HUD_COLOR_VERDE = (120, 220, 90)
HUD_COLOR_AMARILLO = (60, 200, 250)
HUD_COLOR_ROJO = (80, 80, 240)
HUD_COLOR_ACENTO = (255, 190, 70)
HUD_BANNER_ALFA = 0.62
HUD_ESCALA_MINIMA = 0.62
HUD_ANCHO_TARJETA_PREVIEW = 402
HUD_ALTO_TARJETA_PREVIEW = 96

# ------------------------------------------------------------------ live/UI
LIVE_FPS_MAX_OCR = 12.0           # OCR limitado para dejar CPU al render
LIVE_ESCALA_VENTANA = 1.0
LIVE_GUARDAR_POR_DEFECTO = True
LIVE_FILTRO_TEMPORAL_POR_DEFECTO = True
LIVE_MOSTRAR_DEPURACION = False

# --------------------------------------------------------------- ventana Tk
VENTANA_TITULO = "LCD OCR LIVE v2 — Balanza y Termohigrobarómetro"
VENTANA_GEOMETRIA = "1280x720"
VENTANA_MINIMA = (1100, 640)
NIVEL_LOG = "INFO"
NUM_HILOS_OPENCV = 2
EXTENSIONES_IMAGEN = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
EXTENSIONES_VIDEO = frozenset({".mp4", ".avi", ".mkv", ".mov", ".mjpg"})


def normalizar_nombre(valor: str) -> str:
    """Minúsculas sin acentos, para comparar nombres de forma tolerante."""
    if not isinstance(valor, str):
        raise ValueError("El nombre debe ser texto.")
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFKD", valor.strip().casefold())
        if not unicodedata.combining(caracter)
    )


def instrumento_canonico(instrumento: str) -> str:
    """Devuelva 'Balanza' o 'termohigrobarometro' a partir de un alias."""
    nombre = normalizar_nombre(instrumento)
    if nombre == "balanza":
        return "Balanza"
    if nombre in {"termohigro", "termohigrobarometro", "termohiglobarometro",
                  "termohigrobarometro "}:
        return "termohigrobarometro"
    raise ValueError(f"Instrumento desconocido: {instrumento!r}. Use Balanza o Termohigrobarómetro.")


def perfil_de_instrumento(instrumento: str) -> str:
    """Devuelva el perfil de imagen ('lcd' o '7seg') del instrumento."""
    return PERFIL_POR_INSTRUMENTO[instrumento_canonico(instrumento)]


def resolver_ruta_dataset(instrumento: str) -> Path:
    """Localice la carpeta de recortes del instrumento dentro del proyecto.

    Acepta alias con o sin acentos, rutas absolutas configuradas y ``~``.
    Ignora carpetas sin imágenes y lanza un error con instrucciones claras.
    """
    canonico = instrumento_canonico(instrumento)
    carpeta = "Balanza" if canonico == "Balanza" else "termohiglobarometro"
    configurada = Path(
        DATASET_BALANZA if canonico == "Balanza" else DATASET_TERMOHIGRO
    ).expanduser()
    if not configurada.is_absolute():
        configurada = RAIZ_PROYECTO / configurada
    destinos = (
        configurada,
        RAIZ_PROYECTO / "datasets" / carpeta,
        Path.home() / "lcd_ocr" / "datasets" / carpeta,
    )
    intentadas: list[Path] = []
    vacias: list[Path] = []
    for candidata in destinos:
        candidata = candidata.resolve()
        if candidata in intentadas:
            continue
        intentadas.append(candidata)
        if candidata.is_dir():
            if any(
                archivo.is_file() and archivo.suffix.casefold() in EXTENSIONES_IMAGEN
                for archivo in candidata.rglob("*")
            ):
                return candidata
            vacias.append(candidata)
    detalle = "\n".join(f"  - {ruta}" for ruta in intentadas)
    extra = ""
    if vacias:
        extra = "Carpetas existentes sin imágenes:\n" + "\n".join(
            f"  - {ruta}" for ruta in vacias
        ) + "\n"
    raise FileNotFoundError(
        f"No se encontró el dataset de {instrumento}.\n"
        f"Rutas intentadas:\n{detalle}\n{extra}"
        f"Copie la carpeta '{carpeta}' con sus PNG en:\n  {destinos[1]}"
    )


def intervalo_guardado(nombre: str) -> float:
    """Traduzca la etiqueta del selector de intervalo a segundos."""
    for etiqueta, segundos in INTERVALOS_GUARDADO:
        if etiqueta == nombre:
            return segundos
    raise ValueError(f"Intervalo de guardado desconocido: {nombre!r}.")
