"""Configuración central de la aplicación y resolución de datasets locales."""

from pathlib import Path
import unicodedata


RAIZ_PROYECTO = Path(__file__).resolve().parent
DIRECTORIO_CAPTURAS = RAIZ_PROYECTO / "assets" / "capturas"
CAPTURA_ACTUAL = DIRECTORIO_CAPTURAS / "captura_actual.jpg"
ARCHIVO_ROIS = RAIZ_PROYECTO / "data" / "rois.json"
ARCHIVO_LECTURAS = RAIZ_PROYECTO / "data" / "lecturas.csv"

# Editar estas dos rutas si se trasladan los recortes a otra ubicación.
DATASET_BALANZA = (
    r"C:\Users\CATLORD\Documents\Codex\2026-10-09"
    r"\c-users-catlord-downloads-recortes\outputs\recortes_lcd\Balanza"
)
DATASET_TERMOHIGRO = (
    r"C:\Users\CATLORD\Documents\Codex\2026-10-09"
    r"\c-users-catlord-downloads-recortes\outputs\recortes_lcd\termohiglobarometro"
)

# Todas las dimensiones se expresan como (ancho, alto).
CAPTURA_RESOLUCION = (2592, 1944)
PREVIEW_RESOLUCION = (640, 480)
INTERVALO_PREVIEW_MS = 100
CAMARA_INDICE = 0
CAMARA_ESTABILIZACION_SEGUNDOS = 0.7
TAMANO_PLANTILLA = (28, 28)
THRESHOLD_BINARIO = 127
KERNEL_MORFOLOGICO = 3
INVERTIR_BINARIO = False
BLOQUE_ADAPTATIVO = 31
CONSTANTE_ADAPTATIVA = 7
UMBRAL_CONFIANZA = 0.75
K_VECINOS = 3
DESPLAZAMIENTO_OCR_MAXIMO = 2
ESCALA_DISTANCIA_KNN = 0.18
NORMALIZACION_AREA_RELATIVA_MINIMA = 0.02
SEGMENTACION_AREA_MINIMA = 2
SEGMENTACION_SEPARACION_MAXIMA = 2
RETARDO_FILTRO_MS = 180
NUM_HILOS_OPENCV = 2
EXTENSIONES_IMAGEN = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})

VENTANA_TITULO = "LCD OCR — Balanza y Termohigrobarómetro"
VENTANA_GEOMETRIA = "1100x700"
VENTANA_MINIMA = (900, 500)
NIVEL_LOG = "INFO"


def resolver_ruta_dataset(instrumento: str) -> Path:
    """Busque los recortes y devuelva la primera carpeta con archivos de imagen.

    Acepte Balanza, Termohigrobarómetro, termohiglobarometro o termohigro,
    sin distinguir mayúsculas ni acentos. Resuelva rutas relativas respecto
    al proyecto y expanda ``~``. Lance ValueError para un instrumento
    desconocido y FileNotFoundError con instrucciones si no hay carpeta.
    Ignore carpetas vacías para que los destinos del Setup no oculten otros
    fallbacks. La carga y validación de imágenes corresponden a core/dataset.py.
    """
    if not isinstance(instrumento, str):
        raise ValueError("El instrumento debe ser Balanza o Termohigrobarómetro.")
    nombre = "".join(
        caracter
        for caracter in unicodedata.normalize("NFKD", instrumento.strip())
        if not unicodedata.combining(caracter)
    ).casefold()
    if nombre == "balanza":
        ruta_configurada = DATASET_BALANZA
        carpeta = "Balanza"
    elif nombre in {"termohigro", "termohigrobarometro", "termohiglobarometro"}:
        ruta_configurada = DATASET_TERMOHIGRO
        carpeta = "termohiglobarometro"
    else:
        raise ValueError(
            f"Instrumento desconocido: {instrumento!r}. "
            "Use Balanza o Termohigrobarómetro."
        )

    configurada = Path(ruta_configurada).expanduser()
    if not configurada.is_absolute():
        configurada = RAIZ_PROYECTO / configurada
    destinos = (
        RAIZ_PROYECTO / "datasets" / carpeta,
        Path.home() / "lcd_ocr" / "datasets" / carpeta,
    )
    intentadas: list[Path] = []
    vacias: list[Path] = []
    for candidata in (configurada, *destinos):
        if candidata in intentadas:
            continue
        intentadas.append(candidata)
        if candidata.is_dir():
            if any(
                archivo.is_file() and archivo.suffix.casefold() in EXTENSIONES_IMAGEN
                for archivo in candidata.rglob("*")
            ):
                return candidata.resolve()
            vacias.append(candidata)

    detalle = "\n".join(f"  - {ruta}" for ruta in intentadas)
    detalle_vacias = ""
    if vacias:
        detalle_vacias = (
            "Carpetas existentes sin archivos de imagen:\n"
            + "\n".join(f"  - {ruta}" for ruta in vacias)
            + "\n"
        )
    raise FileNotFoundError(
        f"No se encontró un dataset con recortes de {instrumento}.\n"
        f"Rutas intentadas:\n{detalle}\n"
        f"{detalle_vacias}"
        f"Copie la carpeta '{carpeta}', con sus recortes, en:\n"
        f"  {destinos[0]}\n"
        f"o en:\n  {destinos[1]}\n"
        "También puede editar DATASET_BALANZA / DATASET_TERMOHIGRO en config.py."
    )
