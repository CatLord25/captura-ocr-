"""Pestaña 4: editor de ROIs con rol de medición y persistencia JSON."""

import tkinter as tk
from tkinter import ttk

import config
from core.roi import ROI, cargar_rois, guardar_rois, recortar_roi
from ui.widgets import VisorImagen, mostrar_error

ROLES = ("Auto", "Peso", "Temperatura", "Humedad", "Presión", "Unidad", "Texto")
_CLAVE_ROL = {
    "Auto": "", "Peso": "peso", "Temperatura": "temperatura",
    "Humedad": "humedad", "Presión": "presion", "Unidad": "unidad", "Texto": "texto",
}
_ROLES_INVERSO = {valor: clave for clave, valor in _CLAVE_ROL.items()}


class ROIsTab(ttk.Frame):
    """Dibuje ROIs con coordenadas de la imagen, nombre, rol y unidad."""

    def __init__(self, parent, app):
        """Prepare campos, lista y eventos de arrastre del visor."""
        super().__init__(parent, padding=10)
        self.app = app
        self._inicio = None
        self._fin = None
        self.nombre = tk.StringVar(value="valor_1")
        self.instrumento = tk.StringVar(value="Balanza")
        self.rol = tk.StringVar(value="Auto")
        self.unidad = tk.StringVar(value="")
        barra = ttk.Frame(self)
        barra.pack(fill="x")
        ttk.Label(barra, text="Nombre").pack(side="left")
        ttk.Entry(barra, textvariable=self.nombre, width=16).pack(side="left", padx=5)
        ttk.Combobox(barra, textvariable=self.instrumento, state="readonly",
                     values=config.INSTRUMENTOS, width=20).pack(side="left", padx=5)
        ttk.Label(barra, text="Rol").pack(side="left")
        ttk.Combobox(barra, textvariable=self.rol, state="readonly",
                     values=ROLES, width=13).pack(side="left", padx=5)
        ttk.Label(barra, text="Unidad").pack(side="left")
        ttk.Entry(barra, textvariable=self.unidad, width=7).pack(side="left", padx=5)
        ttk.Button(barra, text="Eliminar seleccionada", command=self._eliminar).pack(side="left", padx=3)
        ttk.Button(barra, text="Guardar ROIs", command=self._guardar).pack(side="left", padx=3)
        ttk.Button(barra, text="Cargar ROIs", command=self._cargar).pack(side="left", padx=3)
        self.mensaje = tk.StringVar(
            value="Arrastre del vértice superior izquierdo al inferior derecho de cada campo."
        )
        ttk.Label(self, textvariable=self.mensaje).pack(fill="x", pady=7)
        panel = ttk.Panedwindow(self, orient="horizontal")
        panel.pack(fill="both", expand=True)
        self.visor = VisorImagen(panel)
        lista = ttk.Frame(panel, width=330)
        panel.add(self.visor, weight=3)
        panel.add(lista, weight=1)
        self.tabla = ttk.Treeview(
            lista, columns=("instrumento", "rol", "rect"),
            show="tree headings", height=10, selectmode="browse",
        )
        self.tabla.heading("#0", text="ROI")
        self.tabla.heading("instrumento", text="Instrumento")
        self.tabla.heading("rol", text="Rol")
        self.tabla.heading("rect", text="x, y, ancho, alto")
        self.tabla.column("#0", width=110, minwidth=70)
        self.tabla.column("instrumento", width=140, minwidth=80)
        self.tabla.column("rol", width=90, minwidth=60)
        self.tabla.column("rect", width=130, minwidth=90)
        self.tabla.pack(fill="both", expand=True)
        self.tabla.bind("<<TreeviewSelect>>", lambda _: self._dibujar())
        self.visor.al_redibujar = self._dibujar
        self.visor.canvas.bind("<ButtonPress-1>", self._iniciar)
        self.visor.canvas.bind("<B1-Motion>", self._arrastrar)
        self.visor.canvas.bind("<ButtonRelease-1>", self._terminar)

    def _imagen(self):
        for candidata in (self.app.imagen_binaria, self.app.imagen_corregida,
                          self.app.imagen_capturada):
            if candidata is not None:
                return candidata
        return None

    def actualizar(self):
        """Sincronice imagen y listado sin cambiar coordenadas guardadas."""
        fuente = self._imagen()
        if fuente is None and "Sin imagen" not in self.mensaje.get():
            self.mensaje.set("Capture o abra una imagen para dibujar ROIs.")
        if fuente is not self.visor.imagen:
            self._inicio = self._fin = None
            self.visor.mostrar_imagen(fuente)
        if getattr(self.app, "rois_pendientes", None):
            self.app.rois = list(self.app.rois_pendientes)
            self.app.rois_pendientes = None
            self.app.resultados = []
        seleccion = self.tabla.selection()
        self.tabla.delete(*self.tabla.get_children())
        for indice, roi in enumerate(self.app.rois):
            self.tabla.insert(
                "", "end", iid=str(indice), text=roi.nombre,
                values=(roi.instrumento, _ROLES_INVERSO.get(roi.rol, roi.rol or "Auto"),
                        f"{roi.x}, {roi.y}, {roi.ancho}, {roi.alto}"),
            )
        if seleccion and self.tabla.exists(seleccion[0]):
            self.tabla.selection_set(seleccion[0])
        self._dibujar()

    def _iniciar(self, evento):
        self._inicio = self.visor.punto_imagen(evento.x, evento.y)
        self._fin = self._inicio

    def _arrastrar(self, evento):
        if self._inicio is None:
            return
        punto = self.visor.punto_imagen(evento.x, evento.y)
        if punto is not None:
            self._fin = punto
        self._dibujar()

    def _terminar(self, evento):
        if self._inicio is None:
            return
        punto = self.visor.punto_imagen(evento.x, evento.y)
        if punto is not None:
            self._fin = punto
        x1, y1 = self._inicio
        x2, y2 = self._fin
        self._inicio = self._fin = None
        x, y = int(min(x1, x2)), int(min(y1, y2))
        ancho, alto = max(0, round(abs(x2 - x1))), max(0, round(abs(y2 - y1)))
        if ancho < 4 or alto < 4:
            self._dibujar()
            self.mensaje.set("La ROI es demasiado pequeña; vuelva a arrastrar.")
            return
        try:
            nombre = self.nombre.get().strip()
            if not nombre:
                raise ValueError("Escriba un nombre para la ROI antes de dibujarla.")
            instrumento = self.instrumento.get()
            if any(r.nombre.casefold() == nombre.casefold()
                   and r.instrumento == config.instrumento_canonico(instrumento)
                   for r in self.app.rois):
                raise ValueError("Ese nombre ya existe para el instrumento seleccionado.")
            roi = ROI(nombre, x, y, ancho, alto, instrumento,
                      _CLAVE_ROL.get(self.rol.get(), ""), self.unidad.get().strip())
            if self.visor.imagen is None:
                raise ValueError("No hay imagen para validar la ROI.")
            recortar_roi(self.visor.imagen, roi)
            self.app.rois.append(roi)
            self.app.resultados = []
            self.nombre.set(f"valor_{len(self.app.rois) + 1}")
            self.app.notificar_cambio()
            self.mensaje.set(f"ROI '{nombre}' agregada. Guarde para reutilizarla.")
        except Exception as error:
            mostrar_error("No se pudo crear la ROI", error, self)
            self._dibujar()

    def _dibujar(self):
        canvas = self.visor.canvas
        canvas.delete("rois")
        if self.visor.imagen is None:
            return
        seleccion = self.tabla.selection()
        colores = {
            "Balanza": "#00e0a0", "termohigrobarometro": "#4fc3ff",
        }
        for indice, roi in enumerate(self.app.rois):
            x1, y1 = self.visor.punto_canvas(roi.x, roi.y)
            x2, y2 = self.visor.punto_canvas(roi.x + roi.ancho, roi.y + roi.alto)
            color = "#ffcc00" if str(indice) in seleccion else colores.get(
                roi.instrumento, "#00e0a0"
            )
            canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="rois")
            etiqueta = roi.nombre
            if roi.rol:
                etiqueta += f" · {roi.rol}"
            canvas.create_text(x1 + 3, y1 + 3, anchor="nw", text=etiqueta,
                               fill=color, tags="rois")
        if self._inicio and self._fin:
            canvas.create_rectangle(*self.visor.punto_canvas(*self._inicio),
                                    *self.visor.punto_canvas(*self._fin),
                                    outline="#ffcc00", width=2, dash=(4, 2), tags="rois")

    def _eliminar(self):
        seleccion = self.tabla.selection()
        if not seleccion:
            self.mensaje.set("Seleccione una ROI en la lista para eliminarla.")
            return
        del self.app.rois[int(seleccion[0])]
        self.app.resultados = []
        self.app.notificar_cambio()

    def _guardar(self):
        try:
            if not self.app.rois:
                raise ValueError("No hay ROIs definidas. Dibuje al menos una.")
            guardar_rois(self.app.rois, config.ARCHIVO_ROIS)
            self.mensaje.set(f"ROIs guardadas: {config.ARCHIVO_ROIS}")
        except Exception as error:
            mostrar_error("No se pudieron guardar las ROIs", error, self)

    def _cargar(self):
        try:
            rois = cargar_rois(config.ARCHIVO_ROIS)
            if not rois:
                self.mensaje.set("data/rois.json está vacío; dibuje ROIs y guárdelas.")
                return
            imagen = self.visor.imagen
            if imagen is None:
                raise ValueError("Primero capture o abra una imagen para verificar las ROIs.")
            for roi in rois:
                recortar_roi(imagen, roi)
            self.app.rois = rois
            self.app.resultados = []
            self.app.notificar_cambio()
            self.mensaje.set(
                f"ROIs cargadas: {len(rois)}. Verifique que coincidan con los campos del LCD."
            )
        except Exception as error:
            mostrar_error("No se pudieron cargar las ROIs", error, self)
