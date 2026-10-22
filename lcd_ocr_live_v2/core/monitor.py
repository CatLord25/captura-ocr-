"""Monitor en vivo: hilo de adquisición + OCR + HUD + registro CSV.

Tanto ``ui/tab6_live_hud.py`` como ``main_live_hud.py`` consumen el mismo
``MonitorEnVivo``: el vídeo se captura en un hilo y la interfaz solo copia el
último estado compuesto, de modo que la UI nunca se bloquea.
"""

from dataclasses import dataclass
import logging
from threading import Event, RLock, Thread
import time

import numpy as np

import config
from core.hud import componer_vista, dibujar_dashboard, dibujar_hud
from core.instrumentos import Lectura
from core.log_lecturas import contar_registros, registrar_lecturas
from core.perspectiva import Rectificador
from core.pipeline import OpcionesProceso, PipelineOCR
from core.roi import ROI
from core.utils import MedidorFPS

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class EstadoMonitor:
    """Instantánea lista para pintar en la interfaz."""

    vista: np.ndarray | None = None
    lecturas: tuple[Lectura, ...] = ()
    fps: float = 0.0
    latencia_ms: float = 0.0
    ocr_ms: float = 0.0
    muestras: int = 0
    estado_csv: str = ""
    pausado: bool = False
    mensaje: str = ""
    error: str = ""
    marca: float = 0.0


class MonitorEnVivo:
    """Coordine fuente de vídeo, pipeline OCR, HUD y CSV en un hilo propio."""

    def __init__(
        self,
        fuente,
        rois: list[ROI],
        rectificador: Rectificador | None = None,
        pipeline: PipelineOCR | None = None,
        intervalo: float = config.INTERVALO_GUARDADO_POR_DEFECTO,
        guardar: bool = config.LIVE_GUARDAR_POR_DEFECTO,
        filtro_temporal: bool = config.LIVE_FILTRO_TEMPORAL_POR_DEFECTO,
        umbral: float = config.UMBRAL_CONFIANZA,
        ruta_csv=None,
        roi_completa: bool = False,
        componer_hud: bool = True,
    ) -> None:
        """Configure el monitor sin iniciar hilos."""
        self.fuente = fuente
        self._rois = list(rois)
        self.rectificador = rectificador
        self.pipeline = pipeline or PipelineOCR()
        self._intervalo = float(intervalo)
        self._guardar = bool(guardar)
        self._filtro_temporal = bool(filtro_temporal)
        self._umbral = float(umbral)
        self._ruta_csv = ruta_csv
        self._roi_completa = bool(roi_completa)
        self._componer_hud = bool(componer_hud)
        self._lock = RLock()
        self._fin = Event()
        self._hilo: Thread | None = None
        self._pausado = False
        self._error = ""
        self._muestras = contar_registros(ruta_csv)
        self._ocr_ms = 0.0
        self._estado = EstadoMonitor(estado_csv=self._texto_csv())
        self._fps = MedidorFPS(60)
        self._ultimo_envio = 0.0
        self._ultimo_ocr = 0.0

    # -------------------------------------------------------------- configuración
    @property
    def rois(self) -> list[ROI]:
        """Copia de las ROIs activas."""
        with self._lock:
            return list(self._rois)

    def actualizar_rois(self, rois: list[ROI]) -> None:
        """Reemplace las ROIs y reinicie la votación temporal."""
        with self._lock:
            self._rois = list(rois)
        self.pipeline.reiniciar_temporal()

    def configurar(
        self,
        intervalo: float | None = None,
        guardar: bool | None = None,
        filtro_temporal: bool | None = None,
        umbral: float | None = None,
        roi_completa: bool | None = None,
    ) -> None:
        """Cambie en caliente intervalo, CSV, filtro temporal o umbral."""
        with self._lock:
            if intervalo is not None:
                if intervalo < 0:
                    raise ValueError("El intervalo de guardado no puede ser negativo.")
                self._intervalo = float(intervalo)
            if guardar is not None:
                self._guardar = bool(guardar)
            if filtro_temporal is not None:
                self._filtro_temporal = bool(filtro_temporal)
            if umbral is not None:
                if not 0 <= umbral <= 1:
                    raise ValueError("El umbral debe estar entre 0 y 1.")
                self._umbral = float(umbral)
            if roi_completa is not None:
                self._roi_completa = bool(roi_completa)
            if filtro_temporal:
                self.pipeline.reiniciar_temporal()
            self._estado = self._con_estado(
                estado_csv=self._texto_csv(), pausado=self._pausado
            )

    @property
    def pausado(self) -> bool:
        """Estado de pausa del monitoreo."""
        return self._pausado

    @property
    def guardando(self) -> bool:
        """Indica si el CSV está activo."""
        return self._guardar

    @property
    def filtro_temporal(self) -> bool:
        """Indica si el filtro de moda está activo."""
        return self._filtro_temporal

    def alternar_guardado(self) -> bool:
        """Conmute el registro CSV y devuelva el nuevo estado."""
        self.configurar(guardar=not self._guardar)
        return self._guardar

    def alternar_filtro(self) -> bool:
        """Conmute el filtro temporal y devuelva el nuevo estado."""
        self.configurar(filtro_temporal=not self._filtro_temporal)
        return self._filtro_temporal

    @property
    def estado(self) -> EstadoMonitor:
        """Última instantánea compuesta."""
        with self._lock:
            return self._estado

    def _texto_csv(self) -> str:
        """Etiqueta corta del estado de grabación."""
        if self._error.startswith("CSV"):
            return "CSV error"
        if not self._guardar:
            return "CSV off"
        if self._pausado:
            return "CSV pausado"
        return "Grabando CSV"

    # ------------------------------------------------------------------ control
    def iniciar(self) -> None:
        """Arranque el hilo de monitoreo (idempotente)."""
        if self._hilo is not None and self._hilo.is_alive():
            return
        self._fin.clear()
        self._hilo = Thread(target=self._bucle, name="monitor-lcd", daemon=True)
        self._hilo.start()

    def pausar(self) -> None:
        """Detenga OCR y registro CSV conservando el vídeo."""
        self._pausado = True
        with self._lock:
            self._estado = self._con_estado(pausado=True, estado_csv=self._texto_csv())

    def reanudar(self) -> None:
        """Reanude OCR y registro CSV."""
        self._pausado = False
        self._ultimo_envio = 0.0
        with self._lock:
            self._estado = self._con_estado(pausado=False, estado_csv=self._texto_csv())

    def alternar(self) -> None:
        """Conmute pausa/reanudación."""
        if self._pausado:
            self.reanudar()
        else:
            self.pausar()

    def detener(self) -> None:
        """Detenga el hilo sin cerrar la fuente."""
        self._fin.set()
        hilo = self._hilo
        self._hilo = None
        if hilo is not None and hilo.is_alive():
            hilo.join(timeout=2.0)

    def cerrar(self) -> None:
        """Detenga el hilo y libere la fuente de vídeo."""
        self.detener()
        try:
            self.fuente.cerrar()
        except Exception:  # pragma: no cover - cierre defensivo
            LOG.exception("Error al cerrar la fuente de vídeo.")

    def _con_estado(self, **cambios) -> EstadoMonitor:
        """Reemplace campos del estado actual de forma inmutable."""
        from dataclasses import replace

        with self._lock:
            base = self._estado
        return replace(base, marca=time.time(), **cambios)

    # -------------------------------------------------------------------- hilo
    def _bucle(self) -> None:
        """Bucle de captura, OCR limitado, HUD y CSV."""
        try:
            self._precargar()
        except Exception as error:  # datasets ausentes, etc.
            LOG.exception("No se pudieron cargar las plantillas")
            with self._lock:
                self._estado = self._con_estado(error=str(error), mensaje=str(error))
            return
        try:
            self.fuente.iniciar()
        except Exception as error:
            LOG.exception("No se pudo iniciar la fuente de vídeo")
            with self._lock:
                self._estado = self._con_estado(error=str(error), mensaje=str(error))
            return
        with self._lock:
            self._estado = self._con_estado(mensaje="Monitoreo en vivo")
        ultimo_hud = 0.0
        while not self._fin.is_set():
            try:
                inicio = time.perf_counter()
                frame = self.fuente.leer()
                if frame is None:
                    time.sleep(0.01)
                    continue
                if self.rectificador is not None:
                    frame = self.rectificador.aplicar(frame)
                latencia = self.fuente.latencia_ms
                with self._lock:
                    rois = list(self._rois)
                    filtro = self._filtro_temporal
                    umbral = self._umbral
                    completa = self._roi_completa
                    pausado = self._pausado
                    guardar = self._guardar
                    intervalo = self._intervalo
                ahora = time.monotonic()
                lecturas = self._estado.lecturas
                if not pausado and (ahora - self._ultimo_ocr) >= 1.0 / max(1.0, config.LIVE_FPS_MAX_OCR):
                    t_ocr = time.perf_counter()
                    resultados = self.pipeline.procesar(
                        frame, rois,
                        OpcionesProceso(
                            usar_clahe=config.CLAHE_ACTIVO,
                            filtro_temporal=filtro,
                            umbral=umbral,
                            roi_completa=completa,
                        ),
                    )
                    self._ocr_ms = (time.perf_counter() - t_ocr) * 1000.0
                    lecturas = tuple(resultado.lectura for resultado in resultados)
                    self._ultimo_ocr = ahora
                    errores = [resultado.error for resultado in resultados if resultado.error]
                    if guardar and self._debe_guardar(ahora, intervalo):
                        try:
                            self._muestras += registrar_lecturas(lecturas, self._ruta_csv)
                            self._error = ""
                        except Exception as error:  # CSV bloqueado o cabecera inválida
                            self._error = f"CSV: {error}"
                            LOG.exception("No se pudo registrar el CSV")
                    with self._lock:
                        self._estado = self._con_estado(
                            lecturas=lecturas,
                            estado_csv=self._texto_csv(),
                            mensaje="; ".join(errores[:2]) if errores else "Monitoreo en vivo",
                        )
                # HUD a ~30 FPS como máximo para no saturar la CPU de la RPi.
                if (ahora - ultimo_hud) >= 1.0 / 30.0:
                    ultimo_hud = ahora
                    fps = self._fps.marcar()
                    estado_csv = self._texto_csv()
                    latencia_total = latencia + (time.perf_counter() - inicio) * 1000.0
                    vista = None
                    if self._componer_hud:
                        hud = dibujar_hud(
                            frame, list(lecturas), fps, latencia_total, estado_csv,
                            self._muestras, pausado, self._ocr_ms,
                        )
                        panel = dibujar_dashboard(
                            frame.shape[0], list(lecturas), estado_csv,
                            self._muestras, pausado,
                        )
                        vista = componer_vista(hud, panel)
                    with self._lock:
                        self._estado = EstadoMonitor(
                            vista=vista,
                            lecturas=tuple(lecturas),
                            fps=fps,
                            latencia_ms=latencia_total,
                            ocr_ms=self._ocr_ms,
                            muestras=self._muestras,
                            estado_csv=estado_csv,
                            pausado=pausado,
                            mensaje=self._estado.mensaje,
                            error=self._error,
                            marca=time.time(),
                        )
                resto = 1.0 / 30.0 - (time.perf_counter() - inicio)
                if resto > 0:
                    time.sleep(min(resto, 0.05))
            except Exception as error:  # pragma: no cover - hardware
                LOG.exception("Fallo en el bucle de monitoreo")
                self._error = str(error)
                with self._lock:
                    self._estado = self._con_estado(error=str(error), estado_csv="CSV error")
                time.sleep(0.2)

    def _precargar(self) -> None:
        """Cargue bancos, entrene modelos y caliente el motor antes del HUD."""
        from core.dataset import cargar_banco

        generador = np.random.default_rng(0)
        calentamiento = generador.integers(30, 225, (48, 32), dtype=np.uint8)
        instrumentos = {roi.instrumento for roi in self.rois} or {"Balanza"}
        for instrumento in instrumentos:
            banco = cargar_banco(instrumento)
            self.pipeline.reconocer_recorte(calentamiento, instrumento, grupo="digito")
            LOG.info("Plantillas listas para %s: %d recortes.",
                     banco.instrumento, len(banco.plantillas))

    def _debe_guardar(self, ahora: float, intervalo: float) -> bool:
        """Cadencia de guardado: 0 s = cada pasada de OCR."""
        if intervalo <= 0:
            self._ultimo_envio = ahora
            return True
        if ahora - self._ultimo_envio >= intervalo:
            self._ultimo_envio = ahora
            return True
        return False


def crear_monitor(
    fuente_especificacion: str = "camara",
    rois: list[ROI] | None = None,
    **opciones,
) -> MonitorEnVivo:
    """Cree fuente + monitor con la configuración del proyecto."""
    from core.camara import abrir_fuente

    fuente = abrir_fuente(fuente_especificacion)
    rectificador = Rectificador.cargar()
    if rois is None:
        from core.roi import cargar_rois

        rois = cargar_rois()
    return MonitorEnVivo(fuente, rois, rectificador=rectificador, **opciones)
