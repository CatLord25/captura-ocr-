"""Captura CSI con Picamera2, sin dependencia de cámara al importar el módulo."""

import logging
from pathlib import Path
from threading import RLock
import time

import cv2
import numpy as np

import config
from core.utils import guardar_imagen


LOGGER = logging.getLogger(__name__)


class ErrorCamara(RuntimeError):
    """Error de disponibilidad, configuración o captura de la cámara CSI."""


class CamaraLCD:
    """Encapsule la cámara y serialice todas las operaciones de Picamera2."""

    def __init__(self) -> None:
        """Prepare el controlador sin importar Picamera2 ni abrir hardware."""
        self._camara = None
        self._config_captura = None
        self._lock = RLock()

    @property
    def activa(self) -> bool:
        """Indique si la cámara está iniciada."""
        return self._camara is not None

    def iniciar(self) -> None:
        """Abra la cámara CSI y configure un preview de resolución moderada."""
        with self._lock:
            if self.activa:
                return
            try:
                from picamera2 import Picamera2
            except ImportError as error:
                raise ErrorCamara(
                    "Picamera2 no está disponible. En Raspberry Pi OS instale "
                    "python3-picamera2 y use un venv con --system-site-packages. "
                    "En Windows puede usar «Abrir imagen»."
                ) from error
            camara = None
            try:
                if not Picamera2.global_camera_info():
                    raise ErrorCamara(
                        "No se detectó una cámara CSI. Revise el cable OV5647 "
                        "con la Pi apagada y compruebe rpicam-hello --list-cameras."
                    )
                camara = Picamera2(config.CAMARA_INDICE)
                # RGB888 de libcamera entrega bytes BGR: convertir al contrato RGB.
                preview = camara.create_preview_configuration(
                    main={"size": config.PREVIEW_RESOLUCION, "format": "RGB888"},
                    buffer_count=4,
                )
                self._config_captura = camara.create_still_configuration(
                    main={"size": config.CAPTURA_RESOLUCION, "format": "RGB888"},
                    buffer_count=1,
                )
                camara.configure(preview)
                camara.start()
                time.sleep(config.CAMARA_ESTABILIZACION_SEGUNDOS)
                self._camara = camara
                LOGGER.info("Cámara CSI iniciada; preview %s.", config.PREVIEW_RESOLUCION)
            except Exception as error:
                if camara is not None:
                    camara.close()
                self._config_captura = None
                if isinstance(error, ErrorCamara):
                    raise
                raise ErrorCamara(f"No se pudo iniciar la cámara CSI: {error}") from error

    def obtener_preview(self) -> np.ndarray:
        """Capture un frame de preview RGB sin guardar a disco."""
        with self._lock:
            if not self.activa:
                raise ErrorCamara("Inicie la cámara antes de solicitar un preview.")
            try:
                return cv2.cvtColor(self._camara.capture_array("main"), cv2.COLOR_BGR2RGB)
            except Exception as error:
                raise ErrorCamara(f"Falló la captura del preview: {error}") from error

    def capturar(self, ruta: str | Path | None = None) -> np.ndarray:
        """Capture en alta resolución, guarde JPEG y devuelva una imagen RGB."""
        with self._lock:
            if not self.activa:
                raise ErrorCamara("Inicie la cámara antes de capturar.")
            try:
                datos = self._camara.switch_mode_and_capture_array(
                    self._config_captura, "main"
                )
                imagen = cv2.cvtColor(datos, cv2.COLOR_BGR2RGB)
                destino = guardar_imagen(imagen, ruta or config.CAPTURA_ACTUAL)
                LOGGER.info("Captura guardada: %s (%s).", destino, imagen.shape)
                return imagen
            except Exception as error:
                raise ErrorCamara(f"No se pudo capturar o guardar la imagen: {error}") from error

    def cerrar(self) -> None:
        """Detenga y libere el dispositivo, también después de un fallo."""
        with self._lock:
            camara, self._camara = self._camara, None
            self._config_captura = None
            if camara is not None:
                try:
                    camara.stop()
                finally:
                    camara.close()
                LOGGER.info("Cámara CSI liberada.")
