"""Template matching y K-NN entrenados exclusivamente con MIS recortes."""

from collections import Counter
from dataclasses import dataclass
import logging
from threading import RLock

import cv2
import numpy as np

import config
from core.dataset import cargar_plantillas, normalizar_caracter


LOG = logging.getLogger(__name__)
_MODELOS: dict[tuple, tuple[object, list[str], int]] = {}
_LOCK = RLock()


@dataclass(frozen=True)
class ResultadoCaracter:
    """Contenga la etiqueta aceptada, similitud, candidato y método empleado.

    ``confianza`` es un score de similitud entre 0 y 1, no una probabilidad
    calibrada. ``etiqueta`` vale '?' bajo el umbral; ``candidato`` conserva
    la clase que obtuvo el mejor resultado para inspección y ajuste.
    """

    etiqueta: str
    confianza: float
    candidato: str
    metodo: str


def _metodo_canonico(metodo: str) -> str:
    nombre = str(metodo).strip().casefold().replace("-", "")
    if nombre in {"template", "template matching", "plantillas"}:
        return "template"
    if nombre in {"knn", "k vecinos"}:
        return "knn"
    raise ValueError("Método OCR desconocido; use 'template' o 'knn'.")


def _margen_busqueda() -> int:
    margen = int(getattr(config, "DESPLAZAMIENTO_OCR_MAXIMO", 2))
    if not 0 <= margen <= min(config.TAMANO_PLANTILLA) // 4:
        raise ValueError("DESPLAZAMIENTO_OCR_MAXIMO debe estar entre 0 y 1/4 de la plantilla.")
    return margen


def _mejor_plantilla(
    candidato: np.ndarray, plantillas: list[tuple[str, np.ndarray]]
) -> tuple[str, float]:
    if np.std(candidato) < 1:
        return "", 0.0
    margen = _margen_busqueda()
    ampliada = cv2.copyMakeBorder(candidato, margen, margen, margen, margen,
                                 cv2.BORDER_CONSTANT, value=0)
    etiqueta_mejor, score_mejor = "", -1.0
    for etiqueta, plantilla in plantillas:
        if np.std(plantilla) < 1:
            continue
        respuesta = cv2.matchTemplate(ampliada, plantilla, cv2.TM_CCOEFF_NORMED)
        score = float(np.nanmax(respuesta))
        if np.isfinite(score) and score > score_mejor:
            etiqueta_mejor, score_mejor = etiqueta, score
    return etiqueta_mejor, float(np.clip(max(0.0, score_mejor), 0.0, 1.0))


def _modelo_knn(plantillas: list[tuple[str, np.ndarray]]) -> tuple[object, list[str], int]:
    if not hasattr(cv2, "ml") or not hasattr(cv2.ml, "KNearest_create"):
        raise RuntimeError(
            "Este OpenCV no incluye cv2.ml.KNearest. Instale python3-opencv "
            "en Raspberry Pi o una versión opencv-python 4.x según requirements.txt."
        )
    etiquetas = sorted({etiqueta for etiqueta, _ in plantillas})
    solicitado = int(config.K_VECINOS)
    if solicitado < 1:
        raise ValueError("K_VECINOS debe ser un entero positivo.")
    # Con una sola muestra por clase K=3 produce empates arbitrarios.
    # Limitar K por la clase menos representada evita ese problema.
    k = min(solicitado, min(Counter(etiqueta for etiqueta, _ in plantillas).values()))
    clave = (tuple((etiqueta, id(imagen)) for etiqueta, imagen in plantillas), k)
    with _LOCK:
        if clave not in _MODELOS:
            clases = {etiqueta: indice for indice, etiqueta in enumerate(etiquetas)}
            muestras = np.stack([imagen.ravel() for _, imagen in plantillas]).astype(np.float32) / 255
            respuestas = np.array([clases[etiqueta] for etiqueta, _ in plantillas],
                                  dtype=np.float32).reshape(-1, 1)
            modelo = cv2.ml.KNearest_create()
            modelo.setIsClassifier(True)
            modelo.train(muestras, cv2.ml.ROW_SAMPLE, respuestas)
            _MODELOS[clave] = (modelo, etiquetas, k)
            LOG.info("K-NN: %s muestras, %s clases, K efectivo=%s (configurado=%s)",
                     len(plantillas), len(etiquetas), k, solicitado)
        return _MODELOS[clave]


def _mejor_knn(
    candidato: np.ndarray, plantillas: list[tuple[str, np.ndarray]]
) -> tuple[str, float]:
    modelo, etiquetas, k = _modelo_knn(plantillas)
    margen = _margen_busqueda()
    alto, ancho = candidato.shape
    ampliada = cv2.copyMakeBorder(candidato, margen, margen, margen, margen,
                                 cv2.BORDER_CONSTANT, value=0)
    # Una consulta por desplazamiento, en lote; conserva tolerancia al centrado
    # cuando la segmentación elimina una mota del borde del recorte original.
    muestras = np.stack([ampliada[y:y + alto, x:x + ancho].ravel()
                         for y in range(2 * margen + 1)
                         for x in range(2 * margen + 1)]).astype(np.float32) / 255
    _, resultados, vecinos, distancias = modelo.findNearest(muestras, k)
    clases = resultados[:, 0].astype(int)
    relevantes = np.where(vecinos.astype(int) == clases[:, None], distancias, np.inf)
    mejores = np.min(relevantes, axis=1)
    indice = int(np.argmin(mejores))
    clase, distancia = int(clases[indice]), float(mejores[indice])
    escala = float(getattr(config, "ESCALA_DISTANCIA_KNN", 0.18))
    if not np.isfinite(escala) or escala <= 0:
        raise ValueError("ESCALA_DISTANCIA_KNN debe ser positiva y finita.")
    # Distancia media de píxeles en [0,1]; no equivale a una probabilidad.
    score = float(np.exp(-distancia / (candidato.size * escala)))
    return etiquetas[clase], float(np.clip(score, 0.0, 1.0))


def reconocer_caracter(
    imagen: np.ndarray,
    instrumento: str,
    metodo: str = "template",
    umbral: float | None = None,
) -> ResultadoCaracter:
    """Compare un recorte con el dataset del instrumento y aplique el umbral.

    Acepte un carácter o una unidad/texto completos. Para los textos de varias
    letras compare la ROI completa, sin partirla en caracteres. Devuelva '?'
    en un recorte vacío de contraste o con score inferior al umbral. Lance
    errores claros ante configuración, imagen o dataset inválidos.
    """
    nombre_metodo = _metodo_canonico(metodo)
    limite = float(config.UMBRAL_CONFIANZA if umbral is None else umbral)
    if not np.isfinite(limite) or not 0 <= limite <= 1:
        raise ValueError("El umbral de confianza debe estar entre 0 y 1.")
    plantillas = cargar_plantillas(instrumento)
    if not isinstance(imagen, np.ndarray) or imagen.size == 0:
        raise ValueError("El recorte para reconocer debe ser una matriz no vacía.")
    try:
        normalizada = normalizar_caracter(imagen)
    except ValueError as error:
        if "contraste" in str(error) or "primer plano" in str(error):
            return ResultadoCaracter("?", 0.0, "", nombre_metodo)
        raise
    funcion = _mejor_plantilla if nombre_metodo == "template" else _mejor_knn
    candidato, score = funcion(normalizada, plantillas)
    aceptada = candidato if candidato and score >= limite else "?"
    return ResultadoCaracter(aceptada, score, candidato, nombre_metodo)


def reconocer_segmentos(
    segmentos: list,
    instrumento: str,
    metodo: str = "template",
    umbral: float | None = None,
) -> list[ResultadoCaracter]:
    """Reconozca, en orden, matrices o segmentos con atributo ``imagen``."""
    if not segmentos:
        raise ValueError("No hay caracteres segmentados para reconocer.")
    return [reconocer_caracter(getattr(segmento, "imagen", segmento), instrumento,
                              metodo, umbral) for segmento in segmentos]


def limpiar_cache_reconocimiento() -> None:
    """Vacíe los modelos K-NN cacheados; no modifique las plantillas locales."""
    with _LOCK:
        _MODELOS.clear()
