"""Rectificación por homografía de cuatro puntos, reutilizable en vídeo.

La matriz se calcula una vez por resolución y se cachea: en vivo solo queda
un ``cv2.warpPerspective`` por frame, no una descomposición por fotograma.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

import config
from core.utils import validar_imagen

LOG = logging.getLogger(__name__)


def ordenar_puntos(puntos) -> np.ndarray:
    """Ordene cuatro esquinas convexas como SI, SD, ID, II.

    Rechace puntos repetidos, colineales o donde una esquina quede dentro del
    triángulo de las otras tres.
    """
    try:
        esquinas = np.asarray(puntos, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Las esquinas deben ser cuatro pares de coordenadas.") from error
    if esquinas.shape != (4, 2) or not np.isfinite(esquinas).all():
        raise ValueError("Seleccione exactamente cuatro esquinas válidas del LCD.")
    if len(np.unique(esquinas, axis=0)) != 4:
        raise ValueError("Las cuatro esquinas deben ser distintas.")
    envolvente = cv2.convexHull(esquinas)
    if len(envolvente) != 4 or cv2.contourArea(envolvente) < 4:
        raise ValueError("Las esquinas deben formar un cuadrilátero convexo con área.")
    centro = esquinas.mean(axis=0)
    angulos = np.arctan2(esquinas[:, 1] - centro[1], esquinas[:, 0] - centro[0])
    ordenadas = esquinas[np.argsort(angulos)]
    inicio = min(range(4), key=lambda indice: (ordenadas[indice].sum(), ordenadas[indice, 1]))
    return np.roll(ordenadas, -inicio, axis=0).astype(np.float32)


def tamano_por_defecto(esquinas: np.ndarray) -> tuple[int, int]:
    """Deduzca el tamaño rectificado de las distancias entre lados opuestos."""
    superior_izq, superior_der, inferior_der, inferior_izq = esquinas
    ancho = round(max(np.linalg.norm(superior_der - superior_izq),
                      np.linalg.norm(inferior_der - inferior_izq)))
    alto = round(max(np.linalg.norm(inferior_izq - superior_izq),
                     np.linalg.norm(inferior_der - superior_der)))
    return max(2, int(ancho)), max(2, int(alto))


def corregir_perspectiva(imagen: np.ndarray, puntos, tamano_salida=None) -> np.ndarray:
    """Enderece una imagen mediante homografía conservando su tipo de color."""
    validar_imagen(imagen)
    rectificador = Rectificador.desde_puntos(puntos, tamano_salida)
    return rectificador.aplicar(imagen)


@dataclass(frozen=True)
class Rectificador:
    """Homografía persistible: puntos de origen, tamaño de salida y resolución."""

    puntos: tuple[tuple[float, float], ...]
    tamano: tuple[int, int]
    resolucion_origen: tuple[int, int]
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Valide puntos, tamaño y resolución de origen."""
        esquinas = ordenar_puntos(self.puntos)
        object.__setattr__(self, "puntos", tuple((float(x), float(y)) for x, y in esquinas))
        ancho, alto = (int(valor) for valor in self.tamano)
        if ancho < 2 or alto < 2 or ancho * alto > 25_000_000:
            raise ValueError("El LCD rectificado debe medir entre 2×2 y 25 MP.")
        object.__setattr__(self, "tamano", (ancho, alto))
        origen_ancho, origen_alto = (int(valor) for valor in self.resolucion_origen)
        if origen_ancho < 2 or origen_alto < 2:
            raise ValueError("La resolución de origen debe ser positiva.")
        object.__setattr__(self, "resolucion_origen", (origen_ancho, origen_alto))
        for x, y in self.puntos:
            if not (0 <= x <= origen_ancho - 1 and 0 <= y <= origen_alto - 1):
                raise ValueError("Las esquinas deben quedar dentro de la imagen original.")

    @classmethod
    def desde_puntos(
        cls, puntos, tamano_salida=None, resolucion_origen=None
    ) -> "Rectificador":
        """Cree un rectificador a partir de cuatro puntos en cualquier orden."""
        esquinas = ordenar_puntos(puntos)
        if tamano_salida is None:
            tamano = tamano_por_defecto(esquinas)
        else:
            if len(tamano_salida) != 2 or any(
                not isinstance(valor, (int, np.integer)) for valor in tamano_salida
            ):
                raise ValueError("El tamaño de salida debe ser (ancho, alto) enteros.")
            tamano = (int(tamano_salida[0]), int(tamano_salida[1]))
        if resolucion_origen is None:
            origen = (
                max(2, int(np.ceil(esquinas[:, 0].max())) + 1),
                max(2, int(np.ceil(esquinas[:, 1].max())) + 1),
            )
        else:
            origen = (int(resolucion_origen[0]), int(resolucion_origen[1]))
        return cls(
            puntos=tuple((float(x), float(y)) for x, y in esquinas),
            tamano=tamano,
            resolucion_origen=origen,
        )

    def matriz(self, ancho_frame: int, alto_frame: int) -> np.ndarray:
        """Matriz 3×3 para el tamaño de frame actual (con caché)."""
        if ancho_frame < 2 or alto_frame < 2:
            raise ValueError("El frame debe medir al menos 2×2.")
        clave = (int(ancho_frame), int(alto_frame))
        with self._lock:
            if clave in self._cache:
                return self._cache[clave]
            origen_ancho, origen_alto = self.resolucion_origen
            escala_x = ancho_frame / origen_ancho
            escala_y = alto_frame / origen_alto
            puntos = np.asarray(self.puntos, dtype=np.float32) * (escala_x, escala_y)
            ancho, alto = self.tamano
            destino = np.array(
                [[0, 0], [ancho - 1, 0], [ancho - 1, alto - 1], [0, alto - 1]],
                dtype=np.float32,
            )
            matriz = cv2.getPerspectiveTransform(puntos.astype(np.float32), destino)
            self._cache[clave] = matriz
            return matriz

    def aplicar(self, imagen: np.ndarray) -> np.ndarray:
        """Rectifique el frame (BGR o gris) al tamaño de salida configurado."""
        validar_imagen(imagen)
        alto_frame, ancho_frame = imagen.shape[:2]
        matriz = self.matriz(ancho_frame, alto_frame)
        return cv2.warpPerspective(
            imagen, matriz, self.tamano, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

    def para_resolucion(self, ancho: int, alto: int) -> "Rectificador":
        """Devuelva una copia con otra resolución de origen (misma ROI física)."""
        origen_ancho, origen_alto = self.resolucion_origen
        escala = (ancho / origen_ancho, alto / origen_alto)
        puntos = tuple((x * escala[0], y * escala[1]) for x, y in self.puntos)
        return Rectificador(
            puntos=puntos, tamano=self.tamano, resolucion_origen=(ancho, alto)
        )

    def guardar(self, ruta: str | Path | None = None) -> Path:
        """Escriba la homografía en JSON (puntos, tamaño y resolución)."""
        destino = Path(ruta if ruta is not None else config.ARCHIVO_PERSPECTIVA).expanduser()
        destino.parent.mkdir(parents=True, exist_ok=True)
        datos = {
            "puntos": [[round(x, 2), round(y, 2)] for x, y in self.puntos],
            "tamano": list(self.tamano),
            "resolucion_origen": list(self.resolucion_origen),
        }
        destino.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
        LOG.info("Perspectiva guardada en %s", destino)
        return destino.resolve()

    @classmethod
    def cargar(cls, ruta: str | Path | None = None) -> "Rectificador | None":
        """Cargue la homografía; None si el archivo no existe, error si es inválido."""
        origen = Path(ruta if ruta is not None else config.ARCHIVO_PERSPECTIVA).expanduser()
        try:
            contenido = origen.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return None
        except OSError as error:
            raise OSError(f"No se pudo leer la perspectiva de {origen}: {error}") from error
        if not contenido.strip():
            return None
        try:
            datos = json.loads(contenido)
            return cls(
                puntos=tuple(tuple(punto) for punto in datos["puntos"]),
                tamano=tuple(datos["tamano"]),
                resolucion_origen=tuple(datos["resolucion_origen"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Archivo de perspectiva inválido en {origen}: {error}") from error
