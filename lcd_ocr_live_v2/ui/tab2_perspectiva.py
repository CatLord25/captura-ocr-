"""Pestaña 2: selección de cuatro esquinas y homografía persistida."""

import tkinter as tk
from tkinter import filedialog, ttk

import config
from core.perspectiva import Rectificador
from core.utils import guardar_imagen
from ui.widgets import VisorImagen, mostrar_error


class PerspectivaTab(ttk.Frame):
    """Permita seleccionar cuatro esquinas y guardar la rectificación."""

    def __init__(self, parent, app):
        """Cree controles y dos vistas proporcionales."""
        super().__init__(parent, padding=10)
        self.app = app
        self.puntos = []
        self._fuente = None
        barra = ttk.Frame(self)
        barra.pack(fill="x")
        ttk.Button(barra, text="Deshacer punto", command=self._deshacer).pack(side="left")
        ttk.Button(barra, text="Reiniciar puntos", command=self._reiniciar).pack(side="left", padx=5)
        ttk.Button(barra, text="Aplicar y guardar", command=self._corregir).pack(side="left")
        ttk.Button(barra, text="Guardar corregida…", command=self._guardar).pack(side="left", padx=5)
        ttk.Button(barra, text="Cargar guardada", command=self._cargar).pack(side="left", padx=5)
        self.mensaje = tk.StringVar(value="Abra una imagen en Captura de Imagen.")
        ttk.Label(self, textvariable=self.mensaje).pack(fill="x", pady=6)
        ttk.Label(
            self,
            text="Clic en orden: superior izquierda → superior derecha → inferior "
                 "derecha → inferior izquierda. La homografía se guarda en data/perspectiva.json "
                 "para el monitoreo en vivo.",
        ).pack(fill="x")
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
            self.mensaje.set("Cuatro esquinas seleccionadas. Pulse «Aplicar y guardar».")
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
            canvas.create_line(*coords[-2:], *coords[:2], fill="#ffcc00", width=2,
                               tags="esquinas")

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
            alto, ancho = self._fuente.shape[:2]
            rectificador = Rectificador.desde_puntos(
                self.puntos, resolucion_origen=(ancho, alto)
            )
            imagen = rectificador.aplicar(self._fuente)
            self.app.rectificador = rectificador
            ruta = rectificador.guardar()
            self.app.establecer_corregida(imagen)
            h, w = imagen.shape[:2]
            self.mensaje.set(
                f"LCD corregido {w}×{h}. Homografía guardada en {ruta}. "
                "Defina las ROIs sobre esta vista."
            )
        except Exception as error:
            mostrar_error("No se pudo corregir la perspectiva", error, self)

    def _cargar(self):
        try:
            rectificador = Rectificador.cargar()
            if rectificador is None:
                raise ValueError(f"No existe {config.ARCHIVO_PERSPECTIVA}.")
            if self._fuente is None:
                raise ValueError("Abra una captura para aplicar la homografía guardada.")
            imagen = rectificador.aplicar(self._fuente)
            self.app.rectificador = rectificador
            self.app.establecer_corregida(imagen)
            self.mensaje.set(
                f"Perspectiva cargada: salida {rectificador.tamano[0]}×{rectificador.tamano[1]}."
            )
        except Exception as error:
            mostrar_error("No se pudo cargar la perspectiva", error, self)

    def _guardar(self):
        if self.app.imagen_corregida is None:
            mostrar_error("Sin imagen corregida", "Primero aplique la perspectiva.", self)
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
                mostrar_error("No se pudo guardar", error, self)
