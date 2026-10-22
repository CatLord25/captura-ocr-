"""Desempate topológico de siete segmentos (A–G) aprendido del dataset.

En lugar de umbrales absolutos —que dependen del grosor del trazo, del reflejo
y del estilo de la fuente—, se mide la firma de densidad de las siete regiones
en cada plantilla del instrumento. El desempate compara la firma en vivo con
las dos firmas candidatas en los segmentos que realmente las diferencian:

* ``G`` separa ``0`` de ``8``.
* ``B`` separa ``6`` de ``8`` y ``5`` de ``9``.
* ``E`` separa ``9`` de ``8`` y ``3`` de ``2`` (junto con ``C`` si hace falta).
"""

from dataclasses import dataclass
import logging
from threading import RLock

import cv2
import numpy as np

import config

LOG = logging.getLogger(__name__)
SEGMENTOS = tuple("ABCDEFG")

# Reglas explícitas del enunciado; el resto de pares usa los segmentos que más
# difieren entre las firmas reales del dataset.
REGLAS_DESEMPATE = {
    frozenset(("0", "8")): ("G",),
    frozenset(("6", "8")): ("B",),
    frozenset(("5", "9")): ("B",),
    frozenset(("9", "8")): ("E",),
    frozenset(("2", "3")): ("E", "C"),
    frozenset(("5", "6")): ("E",),
    frozenset(("3", "9")): ("F",),
}


@dataclass(frozen=True)
class FirmaSegmentos:
    """Densidad de tinta (0..1) en cada una de las siete regiones."""

    densidades: tuple[float, ...]

    def como_dict(self) -> dict[str, float]:
        """Devuelva la firma con nombres A..G."""
        return dict(zip(SEGMENTOS, self.densidades))

    def vector(self) -> np.ndarray:
        """Vector float32 ordenado A..G."""
        return np.asarray(self.densidades, dtype=np.float32)


def medir_firma(binaria_canonica: np.ndarray) -> FirmaSegmentos:
    """Mida las densidades A..G sobre una plantilla canónica (32×48)."""
    if binaria_canonica is None or binaria_canonica.size == 0:
        raise ValueError("La imagen canónica está vacía.")
    if binaria_canonica.ndim == 3:
        binaria_canonica = cv2.cvtColor(binaria_canonica, cv2.COLOR_BGR2GRAY)
    alto, ancho = binaria_canonica.shape
    densidades = []
    for x0, y0, x1, y1 in config.SEGMENTOS_7.values():
        region = binaria_canonica[
            int(y0 * alto):max(int(y0 * alto) + 1, int(y1 * alto)),
            int(x0 * ancho):max(int(x0 * ancho) + 1, int(x1 * ancho)),
        ]
        densidades.append(float(np.count_nonzero(region)) / max(1, region.size))
    return FirmaSegmentos(tuple(densidades))


def medir_presencia(binaria_canonica: np.ndarray) -> dict[str, bool]:
    """Compruebe si cada segmento tiene tinta en su centro (sonda 5×5)."""
    alto, ancho = binaria_canonica.shape
    presencia = {}
    for nombre, (x0, y0, x1, y1) in config.SEGMENTOS_7.items():
        cx = int((x0 + x1) / 2 * ancho)
        cy = int((y0 + y1) / 2 * alto)
        region = binaria_canonica[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
        presencia[nombre] = bool(region.size and np.count_nonzero(region) > 0)
    return presencia


def distancia_firmas(a: FirmaSegmentos, b: FirmaSegmentos) -> float:
    """Distancia L1 media entre dos firmas (0..1)."""
    return float(np.mean(np.abs(a.vector() - b.vector())))


class DecodificadorSieteSegmentos:
    """Firmas de referencia por dígito y desempate entre candidatos."""

    def __init__(self, firmas_por_digito: dict[str, FirmaSegmentos]) -> None:
        """Guarde las firmas medidas en las plantillas del instrumento."""
        if not firmas_por_digito:
            raise ValueError("Se necesita al menos una firma de siete segmentos.")
        self._firmas = dict(firmas_por_digito)
        self._lock = RLock()

    @classmethod
    def desde_banco(cls, banco) -> "DecodificadorSieteSegmentos":
        """Construya las firmas a partir de las variantes aumentadas del banco."""
        acumuladas: dict[str, list[np.ndarray]] = {}
        for plantilla in banco.plantillas:
            if plantilla.etiqueta not in config.PATRONES_7:
                continue
            variantes = plantilla.aumentadas or (plantilla.imagen,)
            for variante in variantes:
                acumuladas.setdefault(plantilla.etiqueta, []).append(
                    medir_firma(variante).vector()
                )
        if not acumuladas:
            raise ValueError(f"El banco {banco.instrumento} no contiene dígitos.")
        firmas = {
            etiqueta: FirmaSegmentos(tuple(np.median(np.stack(vectores), axis=0)))
            for etiqueta, vectores in acumuladas.items()
        }
        LOG.info("%s: firmas 7seg de %s", banco.instrumento, ", ".join(sorted(firmas)))
        return cls(firmas)

    @property
    def etiquetas(self) -> tuple[str, ...]:
        """Dígitos con firma disponible."""
        return tuple(sorted(self._firmas))

    def puntajes(self, firma: FirmaSegmentos) -> dict[str, float]:
        """Puntaje 0..1 por dígito según cercanía de su firma."""
        with self._lock:
            puntajes = {}
            for etiqueta, referencia in self._firmas.items():
                distancia = distancia_firmas(firma, referencia)
                puntajes[etiqueta] = float(np.clip(1.0 - distancia / 0.42, 0.0, 1.0))
            return puntajes

    def segmentos_discriminantes(self, a: str, b: str) -> tuple[str, ...]:
        """Segmentos donde las firmas de ``a`` y ``b`` más difieren."""
        regla = REGLAS_DESEMPATE.get(frozenset((a, b)))
        if regla:
            return regla
        if a not in self._firmas or b not in self._firmas:
            return ()
        diferencia = np.abs(self._firmas[a].vector() - self._firmas[b].vector())
        orden = np.argsort(diferencia)[::-1]
        elegidos = [SEGMENTOS[indice] for indice in orden[:2]
                    if diferencia[indice] > 0.02]
        return tuple(elegidos)

    def desempatar(self, firma: FirmaSegmentos, a: str, b: str) -> tuple[str, float, tuple[str, ...]]:
        """Elija entre ``a`` y ``b`` comparando los segmentos que los separan.

        Devuelve (ganador, margen 0..1, segmentos usados). Un margen alto
        significa que la medición en vivo cae claramente del lado del ganador.
        """
        if a == b:
            return a, 1.0, ()
        if a not in self._firmas or b not in self._firmas:
            return a, 0.0, ()
        segmentos = self.segmentos_discriminantes(a, b)
        if not segmentos:
            return a, 0.0, ()
        indices = [SEGMENTOS.index(segmento) for segmento in segmentos]
        medido = firma.vector()[indices]
        ref_a = self._firmas[a].vector()[indices]
        ref_b = self._firmas[b].vector()[indices]
        distancia_a = float(np.mean(np.abs(medido - ref_a)))
        distancia_b = float(np.mean(np.abs(medido - ref_b)))
        separacion = float(np.mean(np.abs(ref_a - ref_b))) or 1e-6
        ganador = a if distancia_a <= distancia_b else b
        margen = float(np.clip((distancia_b - distancia_a) / separacion, 0.0, 1.0))
        return ganador, margen, segmentos
