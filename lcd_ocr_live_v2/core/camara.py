"""Captura en vivo multi-backend (Raspberry Pi 4B + OV5647 / Windows).

- Linux: Picamera2 (libcamera), GStreamer ``libcamerasrc`` o V4L2.
- Windows: DirectShow, Media Foundation o backend automático.
- Hilo dedicado que vacía el buffer del driver y entrega siempre el frame más
  reciente (latencia mínima).
- Control de exposición, brillo, contraste y ganancia para evitar que el
  OV5647 sature los dígitos con iluminación LED o IR.
"""

from dataclasses import dataclass
import logging
import platform
from threading import Condition, Event, RLock, Thread
import time

import cv2
import numpy as np

import config
from core.utils import MedidorFPS, cargar_imagen, guardar_imagen

LOG = logging.getLogger(__name__)


class ErrorCamara(RuntimeError):
    """Error de disponibilidad, configuración o captura de vídeo."""


def _es_windows() -> bool:
    return platform.system().casefold() == "windows"


def _backend_id(nombre: str) -> int | None:
    """Traduzca el nombre de backend a la constante de OpenCV."""
    mapa = {
        "v4l2": getattr(cv2, "CAP_V4L2", None),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", None),
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "any": getattr(cv2, "CAP_ANY", None),
    }
    return mapa.get(nombre.casefold())


def _candidatos(backend: str) -> list[str]:
    """Lista ordenada de backends a intentar."""
    nombre = (backend or "auto").casefold()
    if nombre != "auto":
        if nombre == "picamera2":
            return ["picamera2"]
        if _backend_id(nombre) is None:
            raise ErrorCamara(
                f"Backend desconocido: {backend!r}. Use auto, v4l2, gstreamer, "
                "picamera2, dshow, msmf o any."
            )
        return [nombre]
    preferidos = list(config.CAMARA_BACKENDS_WINDOWS if _es_windows()
                      else config.CAMARA_BACKENDS_LINUX)
    # Descartar backends no compilados en esta build de OpenCV.
    return [item for item in preferidos
            if item == "picamera2" or _backend_id(item) is not None]


@dataclass(frozen=True)
class EstadoSensor:
    """Valores solicitados y leídos del sensor, para diagnóstico."""

    solicitado: dict
    leido: dict
    backend: str
    soportado: bool


class CameraStream:
    """Fuente de vídeo con hilo de captura y buffer de un solo frame."""

    def __init__(
        self,
        indice: int = config.CAMARA_INDICE,
        resolucion: tuple[int, int] = config.CAMARA_RESOLUCION,
        fps: int = config.CAMARA_FPS,
        backend: str = config.CAMARA_BACKEND,
        cuatrocc: str = config.CAMARA_CUATROCC,
        usar_picamera2: bool = True,
    ) -> None:
        """Configure la fuente sin abrir hardware todavía."""
        self.indice = int(indice)
        self.resolucion = (int(resolucion[0]), int(resolucion[1]))
        self.fps_objetivo = int(fps)
        self.backend_solicitado = backend
        self.cuatrocc = cuatrocc
        self.usar_picamera2 = bool(usar_picamera2)
        self._camara = None
        self._origen = ""
        self._backend_usado = ""
        self._hilo: Thread | None = None
        self._fin = Event()
        self._lock = RLock()
        self._condicion = Condition(self._lock)
        self._frame: np.ndarray | None = None
        self._secuencia = 0
        self._t_frame = 0.0
        self._error = ""
        self._fps = MedidorFPS()
        self._picamera_rgb = False
        self._activa = False

    # ------------------------------------------------------------- propiedades
    @property
    def activa(self) -> bool:
        """True si hay un frame disponible y la captura sigue viva."""
        return self._activa

    @property
    def backend_usado(self) -> str:
        """Backend que finalmente abrió la cámara."""
        return self._backend_usado

    @property
    def resolucion_real(self) -> tuple[int, int]:
        """Resolución efectiva comunicada por el driver."""
        with self._lock:
            if self._frame is None:
                return self.resolucion
            return (int(self._frame.shape[1]), int(self._frame.shape[0]))

    @property
    def fps(self) -> float:
        """FPS medidos en el hilo de captura."""
        return self._fps.fps

    @property
    def latencia_ms(self) -> float:
        """Edad del último frame al momento de consultar."""
        with self._lock:
            if self._t_frame <= 0:
                return 0.0
            return (time.perf_counter() - self._t_frame) * 1000.0

    @property
    def error(self) -> str:
        """Último error del hilo de captura (cadena vacía si no hubo)."""
        return self._error

    @property
    def es_estatica(self) -> bool:
        """Indique si la fuente repite un frame estático."""
        return False

    # ---------------------------------------------------------------- apertura
    def iniciar(self) -> None:
        """Abra la cámara con el primer backend disponible e inicie el hilo."""
        if self._activa:
            return
        # Liberar restos de una sesión anterior sin retener el lock durante join.
        self.cerrar()
        errores: list[str] = []
        for candidato in _candidatos(self.backend_solicitado):
            try:
                if candidato == "picamera2":
                    if not self.usar_picamera2:
                        continue
                    self._abrir_picamera2()
                elif candidato == "gstreamer":
                    self._abrir_gstreamer()
                else:
                    self._abrir_opencv(candidato)
            except ErrorCamara as error:
                errores.append(f"{candidato}: {error}")
                self._liberar_dispositivo()
                continue
            self._backend_usado = candidato
            self._fin.clear()
            self._error = ""
            with self._lock:
                self._activa = True
            self._hilo = Thread(target=self._bucle, name="camara-lcd", daemon=True)
            self._hilo.start()
            ancho, alto = self.resolucion
            LOG.info("Cámara iniciada con %s a %d×%d (objetivo %d FPS).",
                     candidato, ancho, alto, self.fps_objetivo)
            return
        detalle = "\n".join(f"  - {linea}" for linea in errores) or "  - sin backends"
        raise ErrorCamara(
            "No se pudo abrir ninguna cámara.\n"
            f"Rutas/backends intentados:\n{detalle}\n"
            "En Raspberry Pi compruebe el cable CSI y «rpicam-hello --list-cameras»; "
            "en Windows revise la cámara y sus permisos."
        )

    def _configurar_opencv(self, captura: cv2.VideoCapture) -> None:
        """Aplique buffersize 1, formato, resolución y FPS."""
        captura.set(cv2.CAP_PROP_BUFFERSIZE, int(config.CAMARA_BUFFERSIZE))
        if self.cuatrocc:
            cuatrocc = cv2.VideoWriter_fourcc(*self.cuatrocc)
            if not captura.set(cv2.CAP_PROP_FOURCC, cuatrocc):
                LOG.debug("El driver no aceptó FOURCC %s.", self.cuatrocc)
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolucion[0])
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolucion[1])
        captura.set(cv2.CAP_PROP_FPS, self.fps_objetivo)

    def _abrir_opencv(self, backend: str) -> None:
        """Abra un ``cv2.VideoCapture`` y valide que entrega frames."""
        identificador = _backend_id(backend)
        captura = cv2.VideoCapture(self.indice, identificador)
        if not captura.isOpened():
            captura.release()
            raise ErrorCamara("el dispositivo no se abrió")
        self._configurar_opencv(captura)
        correcto, frame = captura.read()
        if not correcto or frame is None or frame.size == 0:
            captura.release()
            raise ErrorCamara("abrió pero no entregó ningún frame")
        self._camara = captura
        self._origen = "opencv"
        with self._lock:
            self._frame = frame
            self._t_frame = time.perf_counter()
            self._secuencia += 1
        ancho, alto = captura.get(cv2.CAP_PROP_FRAME_WIDTH), captura.get(cv2.CAP_PROP_FRAME_HEIGHT)
        LOG.info("Backend %s: resolución efectiva %.0f×%.0f.", backend, ancho, alto)
        if (round(ancho), round(alto)) != self.resolucion:
            LOG.warning(
                "El driver entregó %.0f×%.0f en lugar de %s; las ROIs deben "
                "definirse sobre la resolución real.", ancho, alto, self.resolucion,
            )

    def _abrir_gstreamer(self) -> None:
        """Abra el pipeline ``libcamerasrc ! appsink`` si GStreamer está disponible."""
        if getattr(cv2, "CAP_GSTREAMER", 0) == 0:
            raise ErrorCamara("esta build de OpenCV no incluye GStreamer")
        pipeline = config.PIPELINE_GSTREAMER.format(
            ancho=self.resolucion[0], alto=self.resolucion[1], fps=self.fps_objetivo
        )
        captura = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not captura.isOpened():
            captura.release()
            raise ErrorCamara("el pipeline GStreamer no se abrió")
        correcto, frame = captura.read()
        if not correcto or frame is None:
            captura.release()
            raise ErrorCamara("GStreamer no entregó ningún frame")
        self._camara = captura
        self._origen = "opencv"
        with self._lock:
            self._frame = frame
            self._t_frame = time.perf_counter()
            self._secuencia += 1
        LOG.info("GStreamer: %s", pipeline)

    def _abrir_picamera2(self) -> None:
        """Abra la cámara CSI con Picamera2 (libcamera) y formato RGB888."""
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise ErrorCamara(
                "Picamera2 no está instalado (python3-picamera2 + venv con "
                "--system-site-packages)"
            ) from error
        try:
            if not Picamera2.global_camera_info():
                raise ErrorCamara("no se detectó cámara CSI")
            camara = Picamera2(self.indice)
            configuracion = camara.create_video_configuration(
                main={"size": self.resolucion, "format": "RGB888"}, buffer_count=2
            )
            camara.configure(configuracion)
            camara.start()
            time.sleep(config.CAMARA_ESTABILIZACION_SEGUNDOS)
            frame = camara.capture_array("main")
        except ErrorCamara:
            raise
        except Exception as error:  # pragma: no cover - depende del hardware
            raise ErrorCamara(f"Picamera2 falló: {error}") from error
        if frame is None or frame.size == 0:
            camara.stop()
            camara.close()
            raise ErrorCamara("Picamera2 no entregó ningún frame")
        self._camara = camara
        self._origen = "picamera2"
        self._picamera_rgb = True
        with self._lock:
            self._frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._t_frame = time.perf_counter()
            self._secuencia += 1

    def _liberar_dispositivo(self) -> None:
        """Libere el dispositivo actual sin tocar el hilo."""
        camara, self._camara = self._camara, None
        self._origen = ""
        if camara is None:
            return
        try:
            if isinstance(camara, cv2.VideoCapture):
                camara.release()
            else:
                camara.stop()
                camara.close()
        except Exception:  # pragma: no cover - cierre defensivo
            LOG.exception("Error al liberar la cámara.")

    # -------------------------------------------------------------- hilo loop
    def _bucle(self) -> None:
        """Lea continuamente y conserve solo el frame más reciente."""
        fallos = 0
        while not self._fin.is_set():
            camara = self._camara
            if camara is None:
                break
            try:
                if self._origen == "picamera2":
                    frame = camara.capture_array("main")
                    if self._picamera_rgb:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    correcto = frame is not None and frame.size > 0
                else:
                    correcto = bool(camara.grab())
                    if correcto:
                        correcto, frame = camara.retrieve()
                if not correcto or frame is None:
                    fallos += 1
                    if fallos > 150:
                        self._error = "La cámara dejó de entregar frames."
                        LOG.error(self._error)
                        break
                    time.sleep(0.01)
                    continue
                fallos = 0
                with self._condicion:
                    self._frame = frame
                    self._secuencia += 1
                    self._t_frame = time.perf_counter()
                    self._condicion.notify_all()
                self._fps.marcar()
            except Exception as error:  # pragma: no cover - hardware
                self._error = f"Fallo de captura: {error}"
                LOG.exception("Fallo de captura")
                time.sleep(0.05)
        self._activa = False

    # ---------------------------------------------------------------- lectura
    def leer(self, esperar_nuevo: bool = False, timeout: float | None = None) -> np.ndarray | None:
        """Devuelva una copia del frame más reciente (o espere uno nuevo)."""
        limite = config.CAMARA_TIMEOUT_LECTURA_S if timeout is None else float(timeout)
        with self._condicion:
            if esperar_nuevo:
                secuencia = self._secuencia
                self._condicion.wait_for(lambda: self._secuencia != secuencia, limite)
            frame = self._frame
        if frame is None:
            return None
        return frame.copy()

    def capturar(self, ruta=None) -> np.ndarray:
        """Guarde el frame actual en disco y devuélvalo (BGR)."""
        frame = self.leer()
        if frame is None:
            raise ErrorCamara("No hay frame disponible para capturar.")
        destino = guardar_imagen(frame, ruta or config.CAPTURA_ACTUAL)
        LOG.info("Captura guardada en %s (%s×%s).", destino, frame.shape[1], frame.shape[0])
        return frame

    # ------------------------------------------------------- control de sensor
    def configurar_sensor(
        self,
        auto_exposicion: bool | None = None,
        exposicion: int | float | None = None,
        brillo: int | float | None = None,
        contraste: int | float | None = None,
        ganancia: int | float | None = None,
    ) -> EstadoSensor:
        """Ajuste exposición, brillo, contraste y ganancia si el driver lo permite.

        Los valores se leen de vuelta para informar si el driver los aceptó.
        En V4L2 la exposición se expresa en unidades de 100 µs.
        """
        with self._lock:
            if not self._activa or self._camara is None:
                raise ErrorCamara("Inicie la cámara antes de ajustar el sensor.")
            if self._origen == "picamera2":
                return self._configurar_picamera2(
                    auto_exposicion, exposicion, brillo, contraste, ganancia
                )
            captura = self._camara
            if not isinstance(captura, cv2.VideoCapture):
                raise ErrorCamara("La fuente actual no admite control de sensor.")
            solicitado: dict[str, float | None] = {}
            if auto_exposicion is not None:
                if _es_windows():
                    valor = 0.75 if auto_exposicion else 0.25
                else:
                    valor = 3.0 if auto_exposicion else 1.0
                solicitado["auto_exposicion"] = valor
                captura.set(cv2.CAP_PROP_AUTO_EXPOSURE, valor)
            for nombre, propuesta, propiedad in (
                ("exposicion", exposicion, cv2.CAP_PROP_EXPOSURE),
                ("brillo", brillo, cv2.CAP_PROP_BRIGHTNESS),
                ("contraste", contraste, cv2.CAP_PROP_CONTRAST),
                ("ganancia", ganancia, cv2.CAP_PROP_GAIN),
            ):
                if propuesta is None:
                    continue
                valor = float(np.clip(propuesta, 0, 255))
                solicitado[nombre] = valor
                captura.set(propiedad, valor)
            leido = {
                nombre: float(captura.get(propiedad))
                for nombre, propiedad in (
                    ("auto_exposicion", cv2.CAP_PROP_AUTO_EXPOSURE),
                    ("exposicion", cv2.CAP_PROP_EXPOSURE),
                    ("brillo", cv2.CAP_PROP_BRIGHTNESS),
                    ("contraste", cv2.CAP_PROP_CONTRAST),
                    ("ganancia", cv2.CAP_PROP_GAIN),
                )
            }
        soportado = all(
            abs(leido[nombre] - valor) <= max(2.0, abs(valor) * 0.25)
            for nombre, valor in solicitado.items() if nombre in leido
        ) if solicitado else False
        if not soportado:
            LOG.warning("El driver no aplicó todos los controles: %s", solicitado)
        return EstadoSensor(dict(solicitado), leido, self._backend_usado, soportado)

    def _configurar_picamera2(
        self, auto_exposicion, exposicion, brillo, contraste, ganancia
    ) -> EstadoSensor:
        """Ajuste los controles de libcamera con rangos propios de Picamera2."""
        controles: dict[str, float | int | bool] = {}
        if auto_exposicion is not None:
            controles["AeEnable"] = bool(auto_exposicion)
        if exposicion is not None and not auto_exposicion:
            controles["ExposureTime"] = int(max(1, float(exposicion)) * 100)
        if brillo is not None:
            controles["Brightness"] = float(np.clip(float(brillo) / 128.0 - 1.0, -1, 1))
        if contraste is not None:
            controles["Contrast"] = float(np.clip(float(contraste) / 128.0, 0.2, 4.0))
        if ganancia is not None:
            controles["AnalogueGain"] = float(max(0.5, min(16.0, float(ganancia) / 20.0)))
        try:
            self._camara.set_controls(controles)
        except Exception as error:  # pragma: no cover - hardware
            raise ErrorCamara(f"libcamera rechazó los controles: {error}") from error
        LOG.info("Controles Picamera2 aplicados: %s", controles)
        return EstadoSensor(dict(controles), dict(controles), "picamera2", True)

    def modo_nocturno(self, activar: bool = True) -> EstadoSensor:
        """Aplique el preset día/noche (IR) para no saturar los dígitos."""
        ajustes = config.AJUSTES_SENSOR_NOCHE if activar else config.AJUSTES_SENSOR_DIA
        LOG.info("Modo %s activado.", "nocturno (IR)" if activar else "diurno")
        if self._origen == "picamera2" and activar:
            return self.configurar_sensor(
                auto_exposicion=False, exposicion=config.EXPOSICION_US_NOCHE / 100,
                brillo=ajustes["brillo"], contraste=ajustes["contraste"],
                ganancia=config.GANANCIA_ANALOGICA_NOCHE * 20,
            )
        return self.configurar_sensor(
            auto_exposicion=ajustes.get("auto_exposicion"),
            exposicion=ajustes.get("exposicion"),
            brillo=ajustes["brillo"], contraste=ajustes["contraste"],
            ganancia=ajustes["ganancia"],
        )

    # ------------------------------------------------------------------ cierre
    def cerrar(self) -> None:
        """Detenga el hilo, libere el dispositivo y deje la fuente inactiva."""
        with self._lock:
            activa = self._activa
            self._fin.set()
            self._activa = False
        hilo = self._hilo
        self._hilo = None
        if hilo is not None and hilo.is_alive():
            hilo.join(timeout=2.0)
        with self._condicion:
            self._liberar_dispositivo()
            self._frame = None
            self._secuencia += 1
            self._condicion.notify_all()
        if activa:
            LOG.info("Cámara liberada.")


class FuenteImagen:
    """Fuente estática para probar el HUD sin cámara (imagen local)."""

    def __init__(self, ruta) -> None:
        """Cargue la imagen una sola vez."""
        self._imagen = cargar_imagen(ruta)
        self._activa = True
        self.backend_usado = f"imagen:{ruta}"

    @property
    def activa(self) -> bool:
        """La imagen siempre está disponible."""
        return self._activa

    @property
    def fps(self) -> float:
        """Una imagen estática no mide FPS."""
        return 0.0

    @property
    def latencia_ms(self) -> float:
        """Sin latencia de sensor."""
        return 0.0

    @property
    def es_estatica(self) -> bool:
        """Indica que la fuente repite el mismo frame."""
        return True

    @property
    def resolucion_real(self) -> tuple[int, int]:
        """Resolución de la imagen."""
        return (self._imagen.shape[1], self._imagen.shape[0])

    def iniciar(self) -> None:
        """Sin hardware que abrir."""

    def leer(self, esperar_nuevo: bool = False, timeout: float | None = None):
        """Devuelva una copia de la imagen."""
        return self._imagen.copy() if self._activa else None

    def capturar(self, ruta=None):
        """Guarde la imagen cargada."""
        destino = guardar_imagen(self._imagen, ruta or config.CAPTURA_ACTUAL)
        LOG.info("Imagen guardada en %s", destino)
        return self._imagen.copy()

    def cerrar(self) -> None:
        """Marque la fuente como inactiva."""
        self._activa = False


class FuenteVideoArchivo(FuenteImagen):
    """Fuente de vídeo en bucle para probar el pipeline sin cámara."""

    def __init__(self, ruta) -> None:
        """Abra el archivo de vídeo."""
        self._captura = cv2.VideoCapture(str(ruta))
        if not self._captura.isOpened():
            raise ErrorCamara(f"No se pudo abrir el vídeo: {ruta}")
        self._ruta = ruta
        correcto, frame = self._captura.read()
        if not correcto:
            raise ErrorCamara(f"El vídeo no contiene frames: {ruta}")
        self._imagen = frame
        self._activa = True
        self._fps = MedidorFPS()
        self.backend_usado = f"video:{ruta}"

    @property
    def es_estatica(self) -> bool:
        """El vídeo entrega frames nuevos."""
        return False

    def leer(self, esperar_nuevo: bool = False, timeout: float | None = None):
        """Lea el siguiente frame, reiniciando al llegar al final."""
        if not self._activa:
            return None
        correcto, frame = self._captura.read()
        if not correcto or frame is None:
            self._captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
            correcto, frame = self._captura.read()
        if not correcto:
            return None
        self._fps.marcar()
        return frame.copy()

    def cerrar(self) -> None:
        """Libere el archivo."""
        self._activa = False
        self._captura.release()


class FuenteSintetica:
    """Generador de escenas LCD/7seg para demos y pruebas sin hardware."""

    def __init__(self, instrumento: str = "termohigrobarometro", **opciones) -> None:
        """Cree el generador de escenas del instrumento."""
        from core.simulador import GeneradorEscenas

        self._generador = GeneradorEscenas(instrumento, **opciones)
        self._activa = True
        self._fps = MedidorFPS()
        self._indice = 0
        self.backend_usado = "sintetica"
        self._t_frame = time.perf_counter()

    @property
    def activa(self) -> bool:
        """La fuente sintética está siempre lista."""
        return self._activa

    @property
    def fps(self) -> float:
        """FPS de generación."""
        return self._fps.fps

    @property
    def latencia_ms(self) -> float:
        """Tiempo desde el último frame generado."""
        return (time.perf_counter() - self._t_frame) * 1000.0

    @property
    def es_estatica(self) -> bool:
        """Genera un frame nuevo en cada lectura."""
        return False

    @property
    def resolucion_real(self) -> tuple[int, int]:
        """Tamaño del frame sintético."""
        return (self._generador.ancho, self._generador.alto)

    def rois(self):
        """ROIs que corresponden a los campos dibujados en la escena."""
        return self._generador.rois()

    def iniciar(self) -> None:
        """Sin hardware que abrir."""

    def leer(self, esperar_nuevo: bool = False, timeout: float | None = None):
        """Genere el siguiente frame (los valores cambian con el tiempo)."""
        if not self._activa:
            return None
        self._indice += 1
        self._t_frame = time.perf_counter()
        self._fps.marcar()
        return self._generador.frame()

    def capturar(self, ruta=None):
        """Guarde el frame sintético actual."""
        frame = self.leer()
        destino = guardar_imagen(frame, ruta or config.CAPTURA_ACTUAL)
        LOG.info("Escena sintética guardada en %s", destino)
        return frame

    def cerrar(self) -> None:
        """Marque la fuente como inactiva."""
        self._activa = False


def abrir_fuente(
    especificacion: str = "camara",
    indice: int = config.CAMARA_INDICE,
    resolucion: tuple[int, int] = config.CAMARA_RESOLUCION,
    fps: int = config.CAMARA_FPS,
    backend: str = config.CAMARA_BACKEND,
):
    """Fábrica: 'camara', 'imagen:ruta', 'video:ruta' o 'sintetica'."""
    texto = str(especificacion).strip()
    if texto.casefold() in {"camara", "cámara", "camera", "0"}:
        return CameraStream(indice, resolucion, fps, backend)
    if texto.casefold().startswith("imagen:"):
        return FuenteImagen(texto.split(":", 1)[1])
    if texto.casefold().startswith("video:"):
        return FuenteVideoArchivo(texto.split(":", 1)[1])
    if texto.casefold().startswith("sintetica"):
        partes = texto.split(":", 1)
        instrumento = partes[1] if len(partes) == 2 else "termohigrobarometro"
        return FuenteSintetica(instrumento, resolucion=resolucion)
    raise ErrorCamara(
        f"Fuente desconocida: {especificacion!r}. Use camara, imagen:ruta, "
        "video:ruta o sintetica[:instrumento]."
    )
