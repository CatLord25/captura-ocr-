"""Orquestación por ROI: preprocesado → segmentación → OCR → filtro temporal."""

from dataclasses import dataclass, replace
import logging
from threading import RLock

import cv2
import numpy as np

import config
from core.dataset import normalizar_caracter
from core.filtro_temporal import EstabilizadorROIs, Voto
from core.instrumentos import Lectura, componer_lectura, parsear_numero
from core.preprocesado import preprocesar
from core.reconocimiento import ResultadoCaracter, obtener_motor
from core.roi import ROI, recortar_roi_seguro
from core.segmentacion import Segmento, segmentar_preproceso
from core.utils import validar_imagen

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpcionesProceso:
    """Parámetros de una pasada de OCR sobre un frame."""

    usar_clahe: bool = config.CLAHE_ACTIVO
    metodo_binarizado: str = "auto"
    filtro_temporal: bool = config.LIVE_FILTRO_TEMPORAL_POR_DEFECTO
    umbral: float = config.UMBRAL_CONFIANZA
    roi_completa: bool = False


@dataclass(frozen=True)
class ResultadoROI:
    """Salida de una ROI, con trazabilidad para la interfaz."""

    roi: ROI
    lectura: Lectura
    segmentos: tuple[Segmento, ...] = ()
    caracteres: tuple[ResultadoCaracter, ...] = ()
    binaria: np.ndarray | None = None
    error: str = ""


class PipelineOCR:
    """Ejecute el pipeline completo manteniendo el filtro temporal entre frames."""

    def __init__(self, estabilizadores: EstabilizadorROIs | None = None) -> None:
        """Cree el estado temporal compartido entre pasadas."""
        self._estabilizadores = estabilizadores or EstabilizadorROIs()
        self._lock = RLock()

    def reiniciar_temporal(self) -> None:
        """Vacíe las colas de votación (cambio de fuente, ROIs o calibración)."""
        with self._lock:
            self._estabilizadores.reiniciar()

    def procesar(
        self,
        imagen: np.ndarray,
        rois: list[ROI],
        opciones: OpcionesProceso | None = None,
    ) -> list[ResultadoROI]:
        """Procese todas las ROIs y devuelva una lectura por cada una."""
        validar_imagen(imagen)
        opciones = opciones or OpcionesProceso()
        resultados: list[ResultadoROI] = []
        contador: dict[str, int] = {}
        for roi in rois:
            indice = contador.get(roi.instrumento, 0)
            contador[roi.instrumento] = indice + 1
            resultados.append(self._procesar_roi(imagen, roi, indice, opciones))
        return resultados

    def _procesar_roi(
        self, imagen: np.ndarray, roi: ROI, indice: int, opciones: OpcionesProceso
    ) -> ResultadoROI:
        """Procese una ROI aislando errores para no tumbar el frame completo."""
        motor = obtener_motor(roi.instrumento)
        perfil = config.perfil_de_instrumento(roi.instrumento)
        recorte, caja = recortar_roi_seguro(imagen, roi)
        if recorte is None:
            lectura = componer_lectura(
                roi, (), roi.instrumento, indice, error="ROI fuera del frame"
            )
            return ResultadoROI(roi, lectura, error="ROI fuera del frame")
        if getattr(roi, "rol", "") in {"unidad", "texto"} or opciones.roi_completa:
            return self._procesar_completa(recorte, roi, indice, opciones, motor, perfil, caja)
        try:
            pre = preprocesar(
                recorte, perfil=perfil, usar_clahe=opciones.usar_clahe,
                metodo=opciones.metodo_binarizado,
            )
            segmentos = segmentar_preproceso(pre)
            caracteres = tuple(motor.reconocer_segmentos(segmentos, opciones.umbral)) \
                if segmentos else ()
            cajas = tuple(
                (segmento.x, segmento.y, segmento.ancho, segmento.alto, caracter.confianza)
                for segmento, caracter in zip(segmentos, caracteres)
            )
            lectura = componer_lectura(
                roi, caracteres, roi.instrumento, indice, cajas,
                vista=pre.binaria,
                error="" if segmentos else "Sin caracteres en la ROI",
            )
            lectura = self._aplicar_temporal(lectura, opciones)
            return ResultadoROI(roi, lectura, tuple(segmentos), caracteres, pre.binaria)
        except (ValueError, cv2.error) as error:
            LOG.warning("ROI %s: %s", roi.nombre, error)
            lectura = componer_lectura(
                roi, (), roi.instrumento, indice, error=str(error)
            )
            return ResultadoROI(roi, lectura, error=str(error))

    def _procesar_completa(
        self, recorte, roi, indice, opciones, motor, perfil, caja
    ) -> ResultadoROI:
        """Reconozca la ROI entera (unidad, texto o valor sin segmentar)."""
        try:
            pre = preprocesar(
                recorte, perfil=perfil, usar_clahe=opciones.usar_clahe,
                metodo=opciones.metodo_binarizado,
            )
            grupo = roi.rol if roi.rol in {"unidad", "texto"} else "auto"
            caracter = motor.reconocer(pre.gris, grupo, opciones.umbral)
            lectura = componer_lectura(
                roi, (caracter,), roi.instrumento, indice,
                ((0, 0, recorte.shape[1], recorte.shape[0], caracter.confianza),),
                vista=pre.binaria,
            )
            return ResultadoROI(roi, lectura, (), (caracter,), pre.binaria)
        except (ValueError, cv2.error) as error:
            LOG.warning("ROI completa %s: %s", roi.nombre, error)
            lectura = componer_lectura(roi, (), roi.instrumento, indice, error=str(error))
            return ResultadoROI(roi, lectura, error=str(error))

    def _aplicar_temporal(self, lectura: Lectura, opciones: OpcionesProceso) -> Lectura:
        """Sustituya el valor por la moda de los últimos frames válidos."""
        clave = (lectura.instrumento, lectura.roi)
        if not opciones.filtro_temporal:
            return replace(lectura, estable=True, muestras=1)
        with self._lock:
            voto: Voto = self._estabilizadores.votar(clave, lectura.texto, lectura.confianza)
        if not voto.valor:
            return replace(lectura, confianza=0.0, estable=False, muestras=voto.muestras)
        return replace(
            lectura,
            texto=voto.valor,
            numero=parsear_numero(voto.valor),
            confianza=voto.confianza,
            estable=voto.estable,
            muestras=voto.muestras,
        )

    def reconocer_recorte(
        self,
        imagen: np.ndarray,
        instrumento: str,
        grupo: str = "auto",
        umbral: float | None = None,
        perfil: str | None = None,
    ) -> ResultadoCaracter:
        """Reconozca un recorte suelto (usado por la pestaña de reconocimiento)."""
        motor = obtener_motor(instrumento)
        perfil_usado = perfil or config.perfil_de_instrumento(instrumento)
        try:
            normalizada = normalizar_caracter(imagen, config.TAMANO_PLANTILLA, perfil_usado)
        except ValueError as error:
            return ResultadoCaracter("?", 0.0, "", "sin_tinta", grupo)
        return motor.reconocer(normalizada, grupo, umbral)

    def miniaturas(self, resultados: list[ResultadoROI]) -> dict[str, np.ndarray]:
        """Diccionario ROI→máscara binaria para el panel de diagnóstico."""
        return {
            resultado.roi.nombre: resultado.binaria
            for resultado in resultados
            if resultado.binaria is not None
        }
