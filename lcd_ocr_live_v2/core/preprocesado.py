"""Preprocesado adaptativo dual: LCD (reflejos) y siete segmentos (puenteo).

El objetivo es entregar a la segmentación una máscara estable aunque cambien
la iluminación, el reflejo del acrílico o el ángulo de la cámara OV5647.
"""

from dataclasses import dataclass
import logging
import unicodedata

import cv2
import numpy as np

import config
from core.utils import a_grises, normalizar_iluminacion, validar_imagen

LOG = logging.getLogger(__name__)
PERFILES = ("lcd", "7seg")


@dataclass(frozen=True)
class PreprocesoROI:
    """Resultado intermedio listo para segmentar y reconocer."""

    gris: np.ndarray
    binaria: np.ndarray
    puente: np.ndarray
    perfil: str
    metodo: str
    usando_clahe: bool


def _nombre(valor: str) -> str:
    if not isinstance(valor, str):
        raise ValueError("El nombre debe ser texto.")
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFKD", valor.strip().casefold())
        if not unicodedata.combining(caracter)
    )


def perfil_valido(perfil: str) -> str:
    """Valide y normalice el perfil de imagen ('lcd' o '7seg')."""
    nombre = _nombre(perfil)
    if nombre in {"7seg", "7 segmentos", "siete segmentos", "seven"}:
        return "7seg"
    if nombre in {"lcd", "display lcd", "lcd_reflectivo"}:
        return "lcd"
    raise ValueError("Perfil desconocido: use 'lcd' o '7seg'.")


def inferir_metodo(gris: np.ndarray) -> str:
    """Elija Otsu o umbral adaptativo según contraste y bimodalidad.

    Un histograma con demasiados píxeles cerca del umbral Otsu (reflejo
    difuso, fondo con gradiente) se trata con umbral adaptativo gaussiano.
    """
    if float(np.ptp(gris)) < 12 or float(np.std(gris)) < 9:
        return "adaptativo"
    umbral, _ = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cercanos = float(np.count_nonzero(np.abs(gris.astype(np.int16) - umbral) <= 6))
    return "adaptativo" if cercanos / gris.size > 0.32 else "otsu"


def binarizar(
    gris: np.ndarray,
    metodo: str = "auto",
    invertir: bool | None = None,
    threshold: int = config.THRESHOLD_BINARIO,
    bloque: int = config.BLOQUE_ADAPTATIVO,
    constante: float = config.CONSTANTE_ADAPTATIVA,
) -> np.ndarray:
    """Binarice con tinta blanca sobre fondo negro.

    ``invertir=None`` deduce la polaridad del borde de la ROI; ``True`` fuerza
    tinta oscura (``THRESH_BINARY_INV``) y ``False`` tinta clara.
    """
    validar_imagen(gris)
    if gris.ndim != 2:
        raise ValueError("La binarización espera una imagen en escala de grises.")
    nombre = _nombre(metodo)
    if nombre == "auto":
        nombre = inferir_metodo(gris)
    if nombre not in {"manual", "otsu", "adaptativo"}:
        raise ValueError("Método desconocido: use auto, manual, otsu o adaptativo.")
    if not isinstance(threshold, (int, np.integer)) or not 0 <= int(threshold) <= 255:
        raise ValueError("El threshold debe ser un entero entre 0 y 255.")
    if invertir is None:
        # Decisión provisional con Otsu; la polaridad se corrige tras el umbral.
        _, provisional = cv2.threshold(
            gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        invertir = _borde_blanco(provisional)
    tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
    if nombre == "manual":
        _, binaria = cv2.threshold(gris, int(threshold), 255, tipo)
    elif nombre == "otsu":
        _, binaria = cv2.threshold(gris, 0, 255, tipo | cv2.THRESH_OTSU)
    else:
        tamano = max(3, int(bloque)) | 1
        binaria = cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, tipo, tamano, float(constante)
        )
    return binaria


def _borde_blanco(binaria: np.ndarray) -> bool:
    """True si el borde de la máscara es mayoritariamente blanco."""
    borde = np.concatenate((binaria[0], binaria[-1], binaria[:, 0], binaria[:, -1]))
    return float(np.count_nonzero(borde)) / max(1, borde.size) > 0.5


def limpiar_componentes(binaria: np.ndarray, area_minima: int | None = None) -> np.ndarray:
    """Elimine motas de ruido menores al área mínima indicada."""
    minimo = int(config.SEGMENTACION_AREA_MINIMA if area_minima is None else area_minima)
    if minimo < 1:
        raise ValueError("El área mínima debe ser positiva.")
    cantidad, componentes, estadisticas, _ = cv2.connectedComponentsWithStats(binaria, 8)
    if cantidad <= 1:
        return binaria
    areas = estadisticas[1:, cv2.CC_STAT_AREA]
    conservar = np.concatenate(([False], areas >= minimo))
    return np.where(conservar[componentes], 255, 0).astype(np.uint8)


def puentear(binaria: np.ndarray, perfil: str = "7seg") -> np.ndarray:
    """Funda los siete segmentos de cada dígito sin pegar dígitos vecinos.

    Kernel elíptico vertical (3×5) para 7 segmentos: cierra los huecos
    horizontales internos entre segmentos apilados y conserva la separación
    horizontal con el dígito siguiente. En LCD se usa (2×3) para unir trazos
    punteados sin engrosar el glifo.
    """
    perfil = perfil_valido(perfil)
    ancho, alto = config.PUENTE_7SEG if perfil == "7seg" else config.PUENTE_LCD
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ancho), int(alto)))
    return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, nucleo)


def preprocesar(
    imagen: np.ndarray,
    perfil: str = "7seg",
    usar_clahe: bool = config.CLAHE_ACTIVO,
    metodo: str = "auto",
    invertir: bool | None = None,
    threshold: int = config.THRESHOLD_BINARIO,
    bloque: int = config.BLOQUE_ADAPTATIVO,
    constante: float = config.CONSTANTE_ADAPTATIVA,
    area_minima: int | None = None,
) -> PreprocesoROI:
    """Normalice iluminación, binarice, limpie ruido y puentee segmentos."""
    perfil = perfil_valido(perfil)
    validar_imagen(imagen)
    gris = normalizar_iluminacion(imagen, usar_clahe, config.CLAHE_CLIP, config.CLAHE_TILE)
    if gris.ndim == 3:
        gris = a_grises(gris)
    binaria = binarizar(gris, metodo, invertir, threshold, bloque, constante)
    binaria = limpiar_componentes(binaria, area_minima)
    puente = puentear(binaria, perfil)
    metodo_usado = inferir_metodo(gris) if _nombre(metodo) == "auto" else _nombre(metodo)
    return PreprocesoROI(gris, binaria, puente, perfil, metodo_usado, bool(usar_clahe))


def filtrar_imagen(
    imagen: np.ndarray,
    threshold: int = config.THRESHOLD_BINARIO,
    kernel: int = config.KERNEL_MORFOLOGICO,
    operacion: str = "ninguna",
    invertir: bool = config.INVERTIR_BINARIO,
    metodo: str = "manual",
    iteraciones: int = 1,
    usar_clahe: bool = False,
    perfil: str = "lcd",
) -> np.ndarray:
    """Binarizado manual para la pestaña de filtrado, con CLAHE opcional.

    Mantiene la semántica del proyecto original (operaciones morfológicas
    sobre regiones blancas) y añade la normalización de iluminación.
    """
    if not isinstance(kernel, (int, np.integer)) or not 1 <= int(kernel) <= 51:
        raise ValueError("El tamaño de kernel debe ser un entero entre 1 y 51.")
    if not isinstance(iteraciones, (int, np.integer)) or not 1 <= int(iteraciones) <= 20:
        raise ValueError("Las iteraciones deben ser un entero entre 1 y 20.")
    perfil = perfil_valido(perfil)
    gris = normalizar_iluminacion(imagen, usar_clahe, config.CLAHE_CLIP, config.CLAHE_TILE)
    if gris.ndim == 3:
        gris = a_grises(gris)
    binaria = binarizar(gris, metodo, bool(invertir), threshold)
    nombre = _nombre(operacion)
    operaciones = {
        "erosion": cv2.MORPH_ERODE,
        "dilatacion": cv2.MORPH_DILATE,
        "apertura": cv2.MORPH_OPEN,
        "cierre": cv2.MORPH_CLOSE,
    }
    if nombre in {"", "ninguna"}:
        return binaria
    if nombre not in operaciones:
        raise ValueError("Operación desconocida: use ninguna, erosion, dilatacion, apertura o cierre.")
    nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kernel), int(kernel)))
    return cv2.morphologyEx(binaria, operaciones[nombre], nucleo,
                            iterations=int(iteraciones))
