"""Reconocimiento por ROI, confianza por carácter y registro de lecturas."""

import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

import config
from core.log_lecturas import registrar_lectura
from core.preprocesado import filtrar_imagen
from core.reconocimiento import reconocer_segmentos
from core.roi import recortar_roi
from core.segmentacion import segmentar_caracteres
from core.utils import a_grises, cargar_imagen
from ui.widgets import VisorImagen, mostrar_error


class ReconocimientoTab(ttk.Frame):
    """Ejecute OCR con los datasets propios sin bloquear la interfaz."""

    def __init__(self, parent, app):
        """Prepare selector, preview, tabla y cola de resultados del trabajador."""
        super().__init__(parent, padding=10)
        self.app = app
        self.instrumento = tk.StringVar(value="Balanza")
        self.metodo = tk.StringVar(value="template")
        self.segmentacion = tk.StringVar(value="contornos")
        self.roi_elegida = tk.StringVar(value="Todas")
        self.hueco = tk.IntVar(value=getattr(config, "SEGMENTACION_SEPARACION_MAXIMA", 2))
        self.area_minima = tk.IntVar(value=getattr(config, "SEGMENTACION_AREA_MINIMA", 2))
        self._cola = queue.Queue()
        self._ocupado = False
        self._cerrado = False
        self._resultados = []
        self._revision = -1
        barra = ttk.Frame(self)
        barra.pack(fill="x")
        ttk.Label(barra, text="Instrumento").grid(row=0, column=0)
        instrumento = ttk.Combobox(barra, textvariable=self.instrumento,
                                   values=("Balanza", "Termohigrobarómetro"),
                                   state="readonly", width=22)
        instrumento.grid(row=0, column=1, padx=5)
        instrumento.bind("<<ComboboxSelected>>", lambda _: self._cambiar_selector())
        ttk.Label(barra, text="OCR").grid(row=0, column=2)
        ttk.Combobox(barra, textvariable=self.metodo, values=("template", "knn"),
                     state="readonly", width=10).grid(row=0, column=3, padx=5)
        ttk.Label(barra, text="Segmentación").grid(row=0, column=4)
        ttk.Combobox(barra, textvariable=self.segmentacion,
                     values=("contornos", "proyeccion", "ROI completa"),
                     state="readonly", width=14).grid(row=0, column=5, padx=5)
        self.boton = ttk.Button(barra, text="Reconocer y registrar", command=self._reconocer)
        self.boton.grid(row=0, column=6, padx=5)
        ttk.Button(barra, text="Abrir imagen procesada…", command=self._abrir).grid(row=1, column=0, columnspan=2, sticky="w", pady=7)
        ttk.Label(barra, text="ROI").grid(row=1, column=2)
        self.selector_roi = ttk.Combobox(barra, textvariable=self.roi_elegida,
                                         values=("Todas",), state="readonly", width=22)
        self.selector_roi.grid(row=1, column=3, columnspan=2, padx=5)
        ttk.Label(barra, text=f"Umbral: {config.UMBRAL_CONFIANZA:.2f}; score bajo → ?").grid(row=1, column=5, columnspan=2, sticky="w")
        ttk.Label(barra, text="Unir barras (px)").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(barra, from_=0, to=30, textvariable=self.hueco, width=5).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(barra, text="Área mínima").grid(row=2, column=2)
        ttk.Spinbox(barra, from_=1, to=100, textvariable=self.area_minima, width=5).grid(row=2, column=3, sticky="w", padx=5)
        ttk.Label(barra, text="ROI completa: unidades y textos del dataset.").grid(row=2, column=4, columnspan=3, sticky="w")
        self.mensaje = tk.StringVar(value="Defina ROIs en la pestaña anterior; luego seleccione su instrumento.")
        ttk.Label(self, textvariable=self.mensaje, wraplength=1000).pack(fill="x", pady=(0, 5))
        panel = ttk.Panedwindow(self, orient="vertical")
        panel.pack(fill="both", expand=True)
        self.visor = VisorImagen(panel, ancho=640, alto=270)
        self.visor.al_redibujar = self._dibujar
        panel.add(self.visor, weight=2)
        resultado = ttk.Frame(panel)
        panel.add(resultado, weight=1)
        self.tabla = ttk.Treeview(resultado,
                                  columns=("valor", "confianza", "candidato"),
                                  show="tree headings", height=8)
        self.tabla.heading("#0", text="ROI / carácter")
        self.tabla.heading("valor", text="Valor")
        self.tabla.heading("confianza", text="Score")
        self.tabla.heading("candidato", text="Mejor candidato")
        for columna, ancho in (("#0", 240), ("valor", 180), ("confianza", 120), ("candidato", 180)):
            self.tabla.column(columna, width=ancho)
        scroll = ttk.Scrollbar(resultado, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tabla.pack(fill="both", expand=True)
        self._poll = self.after(80, self._recibir)

    def actualizar(self):
        """Actualice la imagen y elimine resultados si cambia el flujo."""
        if self._revision != self.app.revision:
            self._revision = self.app.revision
            self._resultados = []
            self.tabla.delete(*self.tabla.get_children())
        if self.visor.imagen is not self.app.imagen_binaria:
            self.visor.mostrar_imagen(self.app.imagen_binaria)
        self._actualizar_rois()
        self._dibujar()

    def cerrar(self):
        """Cancele el sondeo; ningún trabajador manipula widgets."""
        self._cerrado = True
        self.after_cancel(self._poll)

    def _cambiar_selector(self):
        self._resultados = []
        self.tabla.delete(*self.tabla.get_children())
        self._actualizar_rois()
        self._dibujar()

    def _actualizar_rois(self):
        nombres = [r.nombre for r in self.app.rois if r.instrumento == self.instrumento.get()]
        self.selector_roi.configure(values=("Todas", *nombres))
        if self.roi_elegida.get() not in ("Todas", *nombres):
            self.roi_elegida.set("Todas")

    def _abrir(self):
        ruta = filedialog.askopenfilename(parent=self, title="Abrir imagen procesada",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            rgb = cargar_imagen(ruta)
            gris = a_grises(rgb)
            binaria = gris if np.all((gris == 0) | (gris == 255)) else filtrar_imagen(rgb, metodo="otsu")
            self.app.establecer_procesada(rgb, binaria)
            self.mensaje.set("Procesada cargada. Dibuje o cargue ROIs; si era gris se aplicó Otsu.")
        except Exception as error:
            mostrar_error("No se pudo abrir la imagen", error)

    def _reconocer(self):
        if self._ocupado:
            return
        try:
            if self.app.filtro_pendiente:
                raise ValueError("Espere a que termine el filtrado antes de reconocer.")
            imagen = self.app.imagen_binaria
            if imagen is None:
                raise ValueError("Primero procese una captura o abra una imagen procesada.")
            instrumento = self.instrumento.get()
            rois = [roi for roi in self.app.rois if roi.instrumento == instrumento]
            if self.roi_elegida.get() != "Todas":
                rois = [roi for roi in rois if roi.nombre == self.roi_elegida.get()]
            if not rois:
                raise ValueError(f"No hay ROIs de {instrumento}. Defínalas en la pestaña 4 y asigne el instrumento.")
            revision = self.app.revision
            metodo = self.metodo.get()
            segmentacion = self.segmentacion.get()
            hueco, area = self.hueco.get(), self.area_minima.get()
            if hueco < 0 or hueco > 30 or area < 1 or area > 100:
                raise ValueError("Unir barras debe estar entre 0 y 30; área mínima entre 1 y 100.")
            # Recortes copiados en el hilo Tk para congelar el estado del trabajo.
            entradas = [(roi, recortar_roi(imagen, roi).copy()) for roi in rois]
        except Exception as error:
            mostrar_error("No se puede reconocer", error)
            return
        self._ocupado = True
        self.boton.configure(state="disabled")
        self.mensaje.set("Cargando sus recortes y reconociendo…")
        def trabajo():
            try:
                resultados = []
                for roi, recorte in entradas:
                    segmentos = segmentar_caracteres(recorte, metodo=segmentacion,
                                                     max_hueco=hueco, area_minima=area)
                    if not segmentos:
                        raise ValueError(f"ROI '{roi.nombre}': no se encontraron caracteres. Ajuste el filtrado o la segmentación.")
                    caracteres = reconocer_segmentos(segmentos, instrumento, metodo=metodo)
                    valor = "".join(caracter.etiqueta for caracter in caracteres)
                    resultados.append((roi, segmentos, caracteres, valor))
                self._cola.put((revision, instrumento, resultados, None))
            except Exception as error:
                self._cola.put((revision, instrumento, None, error))
        threading.Thread(target=trabajo, daemon=True).start()

    def _recibir(self):
        if self._cerrado:
            return
        try:
            revision, instrumento, resultados, error = self._cola.get_nowait()
        except queue.Empty:
            pass
        else:
            self._ocupado = False
            self.boton.configure(state="normal")
            if revision != self.app.revision or instrumento != self.instrumento.get():
                self.mensaje.set("La imagen, ROIs o instrumento cambiaron. Vuelva a reconocer.")
            elif error:
                mostrar_error("Error de reconocimiento", error)
                self.mensaje.set(str(error))
            else:
                self._resultados = resultados
                self.app.resultados = resultados
                self._mostrar_resultados()
                try:
                    for roi, _, caracteres, valor in resultados:
                        registrar_lectura(instrumento, roi.nombre, valor,
                                          [c.confianza for c in caracteres],
                                          ruta=config.ARCHIVO_LECTURAS)
                    valores = "; ".join(f"{roi.nombre}: {valor}" for roi, _, _, valor in resultados)
                    self.mensaje.set(f"{valores} · Registro guardado en data/lecturas.csv.")
                    self.app.estado.set("Lecturas reconocidas y registradas con timestamp.")
                except Exception as fallo:
                    mostrar_error("OCR completado; no se pudo registrar", fallo)
        self._poll = self.after(80, self._recibir)

    def _mostrar_resultados(self):
        self.tabla.delete(*self.tabla.get_children())
        for roi, _, caracteres, valor in self._resultados:
            minimo = min(c.confianza for c in caracteres)
            padre = self.tabla.insert("", "end", text=roi.nombre, open=True,
                                     values=(valor, f"mín. {minimo:.3f}", ""))
            for indice, caracter in enumerate(caracteres, 1):
                self.tabla.insert(padre, "end", text=f"Carácter {indice}",
                                  values=(caracter.etiqueta, f"{caracter.confianza:.3f}", caracter.candidato))
        self._dibujar()

    def _dibujar(self):
        canvas = self.visor.canvas
        canvas.delete("ocr")
        if self.visor.imagen is None:
            return
        for roi in self.app.rois:
            if roi.instrumento != self.instrumento.get():
                continue
            canvas.create_rectangle(*self.visor.punto_canvas(roi.x, roi.y),
                                    *self.visor.punto_canvas(roi.x + roi.ancho, roi.y + roi.alto),
                                    outline="#00e0a0", width=2, tags="ocr")
            canvas.create_text(*self.visor.punto_canvas(roi.x + 2, roi.y + 2),
                               text=roi.nombre, anchor="nw", fill="#00e0a0", tags="ocr")
        for roi, segmentos, caracteres, _ in self._resultados:
            for segmento, caracter in zip(segmentos, caracteres):
                x, y = roi.x + segmento.x, roi.y + segmento.y
                color = "#ff6666" if caracter.etiqueta == "?" else "#ffcc00"
                canvas.create_rectangle(*self.visor.punto_canvas(x, y),
                                        *self.visor.punto_canvas(x + segmento.ancho, y + segmento.alto),
                                        outline=color, width=1, tags="ocr")
