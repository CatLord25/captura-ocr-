"""Visor de imágenes y utilidades de interfaz (Tkinter + OpenCV).

El motor de visión es 100 % ``cv2``/``numpy``. Para pintar en Tk se usa Pillow
si está disponible (más rápido) y, si no, se codifica el frame con
``cv2.imencode`` y se entrega a ``tk.PhotoImage``; así el proyecto funciona con
solo OpenCV y numpy instalados.
"""

import base64
import logging
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np


LOG = logging.getLogger(__name__)

try:  # Pillow es opcional: solo acelera el volcado del frame a Tk.
    from PIL import Image, ImageTk

    _PIL_DISPONIBLE = True
except ImportError:  # pragma: no cover - depende del entorno
    Image = ImageTk = None
    _PIL_DISPONIBLE = False


def _a_bgr(imagen: np.ndarray) -> np.ndarray:
    """Convierta gris/BGRA a BGR de 3 canales."""
    if imagen.ndim == 2:
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    if imagen.shape[2] == 4:
        return cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
    return imagen


class VisorImagen(ttk.Frame):
    """Muestre BGR/gris con escala proporcional y convierta coordenadas."""

    def __init__(self, parent, ancho=640, alto=480):
        """Construya un Canvas redimensionable con la imagen centrada."""
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=ancho, height=alto,
                                background="#20262e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.imagen = None
        self.al_redibujar = None
        self._foto = None
        self._rectangulo = (0.0, 0.0, 0.0, 0.0)
        self._pendiente = None
        self.canvas.bind("<Configure>", self._programar_redibujo)

    def mostrar_imagen(self, imagen: np.ndarray | None) -> None:
        """Muestre un ndarray BGR/BGRA/gris; None vacía el visor."""
        if imagen is not None:
            if not isinstance(imagen, np.ndarray) or imagen.ndim not in (2, 3):
                raise ValueError("El visor necesita un ndarray BGR o de grises.")
            if imagen.ndim == 3 and imagen.shape[2] not in (3, 4):
                raise ValueError("La imagen debe tener 3 canales BGR o 4 BGRA.")
            if imagen.size == 0:
                raise ValueError("No se puede mostrar una imagen vacía.")
        self.imagen = imagen
        self._redibujar()

    def limpiar(self) -> None:
        """Borre la imagen y muestre el texto de espera."""
        self.imagen = None
        self._redibujar()

    def punto_imagen(self, x_canvas: float, y_canvas: float):
        """Devuelva coordenadas originales o None si el clic cae en márgenes."""
        if self.imagen is None:
            return None
        x0, y0, ancho, alto = self._rectangulo
        if ancho <= 0 or alto <= 0:
            return None
        if not (x0 <= x_canvas <= x0 + ancho and y0 <= y_canvas <= y0 + alto):
            return None
        alto_img, ancho_img = self.imagen.shape[:2]
        return (
            min(ancho_img - 1, (x_canvas - x0) * ancho_img / ancho),
            min(alto_img - 1, (y_canvas - y0) * alto_img / alto),
        )

    def punto_canvas(self, x_imagen: float, y_imagen: float):
        """Convierta una coordenada de la imagen en coordenadas del Canvas."""
        if self.imagen is None:
            return (0.0, 0.0)
        x0, y0, ancho, alto = self._rectangulo
        alto_img, ancho_img = self.imagen.shape[:2]
        return x0 + x_imagen * ancho / ancho_img, y0 + y_imagen * alto / alto_img

    def _programar_redibujo(self, _evento=None) -> None:
        if self._pendiente is not None:
            self.after_cancel(self._pendiente)
        self._pendiente = self.after(60, self._redibujar)

    def _redibujar(self) -> None:
        self._pendiente = None
        self.canvas.delete("all")
        ancho = max(1, self.canvas.winfo_width())
        alto = max(1, self.canvas.winfo_height())
        if self.imagen is None:
            self._foto = None
            self._rectangulo = (0.0, 0.0, 0.0, 0.0)
            self.canvas.create_text(ancho / 2, alto / 2, text="Sin imagen", fill="white")
            return
        alto_img, ancho_img = self.imagen.shape[:2]
        escala = min(ancho / ancho_img, alto / alto_img)
        nuevo_w = max(1, round(ancho_img * escala))
        nuevo_h = max(1, round(alto_img * escala))
        x0, y0 = (ancho - nuevo_w) / 2, (alto - nuevo_h) / 2
        self._rectangulo = (x0, y0, nuevo_w, nuevo_h)
        try:
            self._foto = _photo_desde_bgr(self.imagen, nuevo_w, nuevo_h)
        except (cv2.error, tk.TclError, OSError) as error:  # pragma: no cover
            LOG.exception("No se pudo convertir el frame para Tk: %s", error)
            self._foto = None
            self.canvas.create_text(ancho / 2, alto / 2,
                                    text=f"No se pudo mostrar la imagen\n{error}",
                                    fill="#ffb0b0")
            return
        self.canvas.create_image(x0, y0, image=self._foto, anchor="nw")
        if self.al_redibujar is not None:
            self.al_redibujar()


def _photo_desde_bgr(imagen: np.ndarray, ancho: int, alto: int):
    """Convierta BGR/gris al tamaño dado y devuelva un ``PhotoImage``."""
    bgr = _a_bgr(imagen)
    interpolacion = cv2.INTER_AREA if (bgr.shape[1] > ancho or bgr.shape[0] > alto) \
        else cv2.INTER_LINEAR
    if imagen.ndim == 2:
        interpolacion = cv2.INTER_NEAREST
    reducida = cv2.resize(bgr, (ancho, alto), interpolation=interpolacion)
    if _PIL_DISPONIBLE:
        rgb = cv2.cvtColor(reducida, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))
    correcto, codificada = cv2.imencode(
        ".png", reducida, [cv2.IMWRITE_PNG_COMPRESSION, 1]
    )
    if not correcto:  # pragma: no cover
        raise OSError("OpenCV no pudo codificar el frame para Tk.")
    datos = base64.b64encode(codificada.tobytes())
    return tk.PhotoImage(data=datos)


def mostrar_error(titulo: str, error: Exception | str, parent=None) -> None:
    """Registre un error y muestre al usuario su mensaje completo."""
    LOG.error("%s: %s", titulo, error)
    messagebox.showerror(titulo, str(error), parent=parent)
