"""Selección de esquinas y vista de la corrección de perspectiva."""

import tkinter as tk
from tkinter import filedialog, ttk

import config
from core.perspectiva import corregir_perspectiva
from core.utils import guardar_imagen
from ui.widgets import VisorImagen, mostrar_error


class PerspectivaTab(ttk.Frame):
    """Permita seleccionar cuatro esquinas sobre los píxeles originales."""

    def __init__(self, parent, app):
        """Cree los controles y dos vistas proporcionales de la imagen."""
        super().__init__(parent, padding=10)
        self.app = app
        self.puntos = []
        self._fuente = None
        barra = ttk.Frame(self)
        barra.pack(fill="x")
        ttk.Button(barra, text="Deshacer punto", command=self._deshacer).pack(side="left")
        ttk.Button(barra, text="Reiniciar puntos", command=self._reiniciar).pack(side="left", padx=5)
        ttk.Button(barra, text="Corregir perspectiva", command=self._corregir).pack(side="left")
        ttk.Button(barra, text="Guardar corregida…", command=self._guardar).pack(side="left", padx=5)
        self.mensaje = tk.StringVar(value="Abra una imagen en Captura de Imagen.")
        ttk.Label(self, textvariable=self.mensaje).pack(fill="x", pady=6)
        ttk.Label(self, text="Clic: superior izquierda → superior derecha → inferior derecha → inferior izquierda.").pack(fill="x")
        vistas = ttk.Panedwindow(self, orient="horizontal")
        vistas.pack(fill="both", expand=True, pady=(8, 0))
        self.original = VisorImagen(vistas, ancho=480, alto=360)
        self.corregida = VisorImagen(vistas, ancho=480, alto=360)
        vistas.add(self.original, weight=1)
        vistas.add(self.corregida, weight=1)
        self.original.canvas.bind("<Button-1>", self._clic)
        self.original.al_redibujar = self._dibujar_puntos

    def actualizar(self):
        """Actualice las vistas y reinicie los clics si cambia la captura."""
        fuente = self.app.imagen_capturada
        if fuente is not self._fuente:
            self._fuente = fuente
            self.puntos.clear()
            self.original.mostrar_imagen(fuente)
            self._mensaje_puntos()
        if self.corregida.imagen is not self.app.imagen_corregida:
            self.corregida.mostrar_imagen(self.app.imagen_corregida)

    def _clic(self, evento):
        punto = self.original.punto_imagen(evento.x, evento.y)
        if punto is None or len(self.puntos) >= 4:
            return
        self.puntos.append(punto)
        self._dibujar_puntos()
        self._mensaje_puntos()

    def _mensaje_puntos(self):
        if self._fuente is None:
            self.mensaje.set("Abra una imagen en Captura de Imagen.")
        elif len(self.puntos) == 4:
            self.mensaje.set("Cuatro esquinas seleccionadas. Pulse Corregir perspectiva.")
        else:
            self.mensaje.set(f"Esquinas seleccionadas: {len(self.puntos)}/4.")

    def _dibujar_puntos(self):
        canvas = self.original.canvas
        canvas.delete("esquinas")
        coords = []
        for indice, punto in enumerate(self.puntos, 1):
            x, y = self.original.punto_canvas(*punto)
            coords.extend((x, y))
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                               fill="#ffcc00", outline="black", tags="esquinas")
            canvas.create_text(x + 12, y - 12, text=str(indice),
                               fill="#ffcc00", tags="esquinas")
        if len(coords) >= 4:
            canvas.create_line(*coords, fill="#ffcc00", width=2, tags="esquinas")
        if len(self.puntos) == 4:
            canvas.create_line(*coords[-2:], *coords[:2], fill="#ffcc00", width=2, tags="esquinas")

    def _deshacer(self):
        if self.puntos:
            self.puntos.pop()
        self._dibujar_puntos()
        self._mensaje_puntos()

    def _reiniciar(self):
        self.puntos.clear()
        self._dibujar_puntos()
        self._mensaje_puntos()

    def _corregir(self):
        try:
            if self._fuente is None:
                raise ValueError("Primero capture o abra una imagen.")
            if len(self.puntos) != 4:
                raise ValueError("Seleccione las cuatro esquinas del LCD.")
            imagen = corregir_perspectiva(self._fuente, self.puntos)
            self.app.establecer_corregida(imagen)
            h, w = imagen.shape[:2]
            self.mensaje.set(f"LCD corregido: {w} × {h} píxeles.")
        except Exception as error:
            mostrar_error("No se pudo corregir la perspectiva", error)

    def _guardar(self):
        if self.app.imagen_corregida is None:
            mostrar_error("Sin imagen corregida", "Primero aplique la perspectiva.")
            return
        ruta = filedialog.asksaveasfilename(
            parent=self, initialdir=config.DIRECTORIO_CAPTURAS,
            initialfile="lcd_corregido.png", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if ruta:
            try:
                guardar_imagen(self.app.imagen_corregida, ruta)
                self.mensaje.set(f"Guardada: {ruta}")
            except Exception as error:
                mostrar_error("No se pudo guardar", error)
