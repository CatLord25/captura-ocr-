"""Visor de imágenes y coordenadas compartido por las cinco pestañas."""

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


class VisorImagen(ttk.Frame):
    """Muestre RGB/gris con escala proporcional y convierta coordenadas."""

    def __init__(self, parent, ancho=640, alto=480):
        """Construya un Canvas redimensionable con una imagen centrada."""
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=ancho, height=alto,
                                background="#20262e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.imagen = None
        self.al_redibujar = None
        self._foto = None
        self._rectangulo = (0, 0, 0, 0)
        self._pendiente = None
        self.canvas.bind("<Configure>", self._programar_redibujo)

    def mostrar_imagen(self, imagen):
        """Muestre un ndarray RGB/gris; None vacía el visor."""
        if imagen is not None:
            if not isinstance(imagen, np.ndarray) or imagen.ndim not in (2, 3):
                raise ValueError("El visor necesita un ndarray RGB o de grises.")
            if imagen.ndim == 3 and imagen.shape[2] != 3:
                raise ValueError("La imagen debe tener tres canales RGB.")
            if imagen.size == 0:
                raise ValueError("No se puede mostrar una imagen vacía.")
        self.imagen = imagen
        self._redibujar()

    def punto_imagen(self, x_canvas, y_canvas):
        """Devuelva coordenadas originales o None si el clic cae en márgenes."""
        if self.imagen is None:
            return None
        x0, y0, ancho, alto = self._rectangulo
        if ancho <= 0 or alto <= 0:
            return None
        if not (x0 <= x_canvas <= x0 + ancho and y0 <= y_canvas <= y0 + alto):
            return None
        h, w = self.imagen.shape[:2]
        return (min(w - 1, (x_canvas - x0) * w / ancho),
                min(h - 1, (y_canvas - y0) * h / alto))

    def punto_canvas(self, x_imagen, y_imagen):
        """Convierta una coordenada de la imagen en coordenadas del Canvas."""
        if self.imagen is None:
            return None
        x0, y0, ancho, alto = self._rectangulo
        h, w = self.imagen.shape[:2]
        return x0 + x_imagen * ancho / w, y0 + y_imagen * alto / h

    def _programar_redibujo(self, _evento=None):
        if self._pendiente is not None:
            self.after_cancel(self._pendiente)
        self._pendiente = self.after(60, self._redibujar)

    def _redibujar(self):
        self._pendiente = None
        self.canvas.delete("all")
        ancho = max(1, self.canvas.winfo_width())
        alto = max(1, self.canvas.winfo_height())
        if self.imagen is None:
            self._foto = None
            self._rectangulo = (0, 0, 0, 0)
            self.canvas.create_text(ancho / 2, alto / 2, text="Sin imagen",
                                    fill="white")
            return
        h, w = self.imagen.shape[:2]
        escala = min(ancho / w, alto / h)
        nuevo_w = max(1, round(w * escala))
        nuevo_h = max(1, round(h * escala))
        x0, y0 = (ancho - nuevo_w) / 2, (alto - nuevo_h) / 2
        self._rectangulo = (x0, y0, nuevo_w, nuevo_h)
        pil = Image.fromarray(self.imagen.astype(np.uint8, copy=False))
        remuestreo = Image.Resampling.NEAREST if self.imagen.ndim == 2 else Image.Resampling.BILINEAR
        pil = pil.resize((nuevo_w, nuevo_h), remuestreo)
        self._foto = ImageTk.PhotoImage(pil)
        self.canvas.create_image(x0, y0, image=self._foto, anchor="nw")
        if self.al_redibujar is not None:
            self.al_redibujar()


def mostrar_error(titulo, error):
    """Registre un error y muestre al usuario su mensaje completo."""
    import logging
    from tkinter import messagebox

    logging.getLogger(__name__).error("%s: %s", titulo, error)
    messagebox.showerror(titulo, str(error))
