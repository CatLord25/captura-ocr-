"""Filtro temporal anti-parpadeo por votación de moda sobre los últimos N frames.

Cada ROI mantiene una cola circular con las lecturas válidas recientes. El
valor publicado (HUD y CSV) es la moda, no el último frame: así se descartan
los estados transitorios mientras el display cambia de dígito.
"""

from collections import Counter, deque
from dataclasses import dataclass
import logging

import numpy as np

import config

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Voto:
    """Resultado de la votación temporal de una ROI."""

    valor: str
    confianza: float
    estable: bool
    muestras: int
    acuerdos: int


VACIO = Voto("", 0.0, False, 0, 0)


class Estabilizador:
    """Cola circular y moda estadística para un único valor de ROI."""

    def __init__(
        self,
        tamano: int = config.ESTABILIZACION_MUESTRAS,
        minimo_acuerdos: int = config.ESTABILIZACION_MINIMO_ACUERDOS,
    ) -> None:
        """Cree la cola con los últimos ``tamano`` valores válidos."""
        if tamano < 1:
            raise ValueError("El tamaño de estabilización debe ser positivo.")
        if minimo_acuerdos < 1 or minimo_acuerdos > tamano:
            raise ValueError("El mínimo de acuerdos debe estar entre 1 y el tamaño.")
        self._valores: deque[tuple[str, float]] = deque(maxlen=int(tamano))
        self._minimo = int(minimo_acuerdos)

    def agregar(self, valor: str, confianza: float) -> Voto:
        """Añada un valor válido y devuelva la votación actualizada."""
        if not isinstance(valor, str):
            raise ValueError("El valor temporal debe ser texto.")
        limpio = valor.strip()
        if not limpio or "?" in limpio:
            return self.ultimo
        confianza = float(confianza) if np.isfinite(confianza) else 0.0
        self._valores.append((limpio, max(0.0, min(1.0, confianza))))
        return self._moda()

    def _moda(self) -> Voto:
        """Calcule la moda con desempate por el valor más reciente."""
        if not self._valores:
            return VACIO
        conteo = Counter(valor for valor, _ in self._valores)
        maximo = max(conteo.values())
        orden = [valor for valor, cantidad in conteo.items() if cantidad == maximo]
        # Desempate: el candidato que apareció más recientemente.
        for valor, _ in reversed(self._valores):
            if valor in orden:
                ganador = valor
                break
        confianzas = [c for valor, c in self._valores if valor == ganador]
        return Voto(
            valor=ganador,
            confianza=float(np.mean(confianzas)) if confianzas else 0.0,
            estable=maximo >= self._minimo,
            muestras=len(self._valores),
            acuerdos=int(maximo),
        )

    @property
    def ultimo(self) -> Voto:
        """Última votación sin añadir muestras."""
        return self._moda()

    def reiniciar(self) -> None:
        """Vacíe la cola (cambio de ROI, de instrumento o de encuadre)."""
        self._valores.clear()


class EstabilizadorROIs:
    """Conjunto de estabilizadores indexados por (instrumento, ROI)."""

    def __init__(
        self,
        tamano: int = config.ESTABILIZACION_MUESTRAS,
        minimo_acuerdos: int = config.ESTABILIZACION_MINIMO_ACUERDOS,
    ) -> None:
        """Guarde la configuración y cree el diccionario de colas."""
        self._tamano = int(tamano)
        self._minimo = int(minimo_acuerdos)
        self._estabilizadores: dict[tuple[str, str], Estabilizador] = {}

    def votar(self, clave: tuple[str, str], valor: str, confianza: float) -> Voto:
        """Añada una lectura a la cola de esa ROI y devuelva su moda."""
        estabilizador = self._estabilizadores.get(clave)
        if estabilizador is None:
            estabilizador = Estabilizador(self._tamano, self._minimo)
            self._estabilizadores[clave] = estabilizador
        return estabilizador.agregar(valor, confianza)

    def reiniciar(self, clave: tuple[str, str] | None = None) -> None:
        """Vacíe una ROI o todas las colas."""
        if clave is None:
            self._estabilizadores.clear()
        elif clave in self._estabilizadores:
            self._estabilizadores[clave].reiniciar()

    def instantanea(self) -> dict[tuple[str, str], Voto]:
        """Devuelva la votación actual de cada ROI."""
        return {clave: estabilizador.ultimo for clave, estabilizador in self._estabilizadores.items()}
