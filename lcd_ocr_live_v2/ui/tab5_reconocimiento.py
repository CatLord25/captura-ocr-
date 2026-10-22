"""Pestaña 5: reconocimiento por ROI, confianza por carácter y CSV."""

import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import config
from core.log_lecturas import registrar_lecturas
from core.pipeline import OpcionesProceso, PipelineOCR
from core.utils import cargar_imagen, color_confianza
from ui.widgets import VisorImagen, mostrar_error


class ReconocimientoTab(ttk.Frame):
    """Ejecute el pipeline OCR con los datasets propios sin bloquear Tk."""

    def __init__(self, parent, app):
        """Prepare selector, preview, tabla y cola del trabajador."""
        super().__init__(parent, padding=10)
        self.app = app
        self.instrumento = tk.StringVar(value="Balanza")
        self.modo = tk.StringVar(value="Segmentos")
        self.roi_elegida = tk.StringVar(value="Todas")
        self.umbral = tk.DoubleVar(value=config.UMBRAL_CONFIANZA)
        self.guardar_csv = tk.BooleanVar(value=True)
        self._cola = queue.Queue()
        self._pipeline = PipelineOCR()
        self._ocupado = False
        self._cerrado = False
        self._resultados = []
        self._revision = -1
        self._fuente = None

        barra = ttk.Frame(self)
        barra.pack(fill="x")
        ttk.Label(barra, text="Instrumento").grid(row=0, column=0)
        combo = ttk.Combobox(barra, textvariable=self.instrumento,
                             values=config.INSTRUMENTOS, state="readonly", width=20)
        combo.grid(row=0, column=1, padx=5)
        combo.bind("<<ComboboxSelected>>", lambda _: self._cambiar_selector())
        ttk.Label(barra, text="Modo").grid(row=0, column=2)
        ttk.Combobox(barra, textvariable=self.modo, values=("Segmentos", "ROI completa"),
                     state="readonly", width=14).grid(row=0, column=3, padx=5)
        ttk.Label(barra, text="Umbral").grid(row=0, column=4)
        ttk.Spinbox(barra, from_=0.30, to=0.99, increment=0.05,
                    textvariable=self.umbral, width=6).grid(row=0, column=5, padx=5)
        ttk.Checkbutton(barra, text="Guardar en CSV",
                        variable=self.guardar_csv).grid(row=0, column=6, padx=6)
        self.boton = ttk.Button(barra, text="Reconocer ROIs", command=self._reconocer)
        self.boton.grid(row=0, column=7, padx=5)
        ttk.Button(barra, text="Abrir imagen procesada…", command=self._abrir).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=7)
        ttk.Label(barra, text="ROI").grid(row=1, column=2)
        self.selector_roi = ttk.Combobox(barra, textvariable=self.roi_elegida,
                                         values=("Todas",), state="readonly", width=22)
        self.selector_roi.grid(row=1, column=3, padx=5)
        ttk.Label(barra, text="El CSV se escribe en data/lecturas.csv").grid(
            row=1, column=4, columnspan=2, sticky="w")

        self.mensaje = tk.StringVar(
            value="Defina ROIs en la pestaña anterior y pulse «Reconocer ROIs»."
        )
        ttk.Label(self, textvariable=self.mensaje, wraplength=1050).pack(fill="x", pady=(0, 5))
        panel = ttk.Panedwindow(self, orient="vertical")
        panel.pack(fill="both", expand=True)
        self.visor = VisorImagen(panel, ancho=640, alto=300)
        self.visor.al_redibujar = self._dibujar
        panel.add(self.visor, weight=2)
        resultado = ttk.Frame(panel)
        panel.add(resultado, weight=1)
        self.tabla = ttk.Treeview(resultado,
                                  columns=("valor", "confianza", "candidato"),
                                  show="tree headings", height=8)
        self.tabla.heading("#0", text="ROI / carácter")
        self.tabla.heading("valor", text="Valor")
        self.tabla.heading("confianza", text="Confianza")
        self.tabla.heading("candidato", text="Mejor candidato")
        for columna, ancho in (("#0", 250), ("valor", 200), ("confianza", 130),
                               ("candidato", 200)):
            self.tabla.column(columna, width=ancho)
        scroll = ttk.Scrollbar(resultado, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tabla.pack(fill="both", expand=True)
        self._poll = self.after(80, self._recibir)

    # ------------------------------------------------------------------ estado
    def actualizar(self):
        """Actualice la imagen y descarte resultados si cambia el flujo."""
        if self._revision != self.app.revision:
            self._revision = self.app.revision
            self._resultados = []
            self.tabla.delete(*self.tabla.get_children())
        fuente = self._imagen()
        if fuente is not self._fuente:
            self._fuente = fuente
            self.visor.mostrar_imagen(fuente)
        self._actualizar_rois()
        self._dibujar()

    def _imagen(self):
        for candidata in (self.app.imagen_corregida, self.app.imagen_binaria,
                          self.app.imagen_capturada):
            if candidata is not None:
                return candidata
        return None

    def cerrar(self):
        """Cancele el sondeo; ningún trabajador toca Tk."""
        self._cerrado = True
        self.after_cancel(self._poll)

    def _cambiar_selector(self):
        self._resultados = []
        self.tabla.delete(*self.tabla.get_children())
        self._actualizar_rois()
        self._dibujar()

    def _actualizar_rois(self):
        canonico = config.instrumento_canonico(self.instrumento.get())
        nombres = [r.nombre for r in self.app.rois if r.instrumento == canonico]
        self.selector_roi.configure(values=("Todas", *nombres))
        if self.roi_elegida.get() not in ("Todas", *nombres):
            self.roi_elegida.set("Todas")

    def _abrir(self):
        ruta = filedialog.askopenfilename(
            parent=self, title="Abrir imagen procesada",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            imagen = cargar_imagen(ruta)
            self.app.establecer_procesada(imagen)
            self.mensaje.set("Imagen cargada. Dibuje o cargue ROIs y reconozca.")
        except Exception as error:
            mostrar_error("No se pudo abrir la imagen", error, self)

    # ------------------------------------------------------------- reconocer
    def _reconocer(self):
        if self._ocupado:
            return
        try:
            if self.app.filtro_pendiente:
                raise ValueError("Espere a que termine el filtrado antes de reconocer.")
            imagen = self._imagen()
            if imagen is None:
                raise ValueError("Primero capture, abra o filtre una imagen.")
            canonico = config.instrumento_canonico(self.instrumento.get())
            rois = [roi for roi in self.app.rois if roi.instrumento == canonico]
            if self.roi_elegida.get() != "Todas":
                rois = [roi for roi in rois if roi.nombre == self.roi_elegida.get()]
            if not rois:
                raise ValueError(
                    f"No hay ROIs de {self.instrumento.get()}. Defínalas en la pestaña 4."
                )
            umbral = float(self.umbral.get())
            if not 0.0 <= umbral <= 1.0:
                raise ValueError("El umbral debe estar entre 0 y 1.")
            revision = self.app.revision
            completa = self.modo.get() == "ROI completa"
            copia = imagen.copy()
        except Exception as error:
            mostrar_error("No se puede reconocer", error, self)
            return
        self._ocupado = True
        self.boton.configure(state="disabled")
        self.mensaje.set("Cargando plantillas y reconociendo…")

        def trabajo():
            try:
                resultados = self._pipeline.procesar(
                    copia, rois,
                    OpcionesProceso(filtro_temporal=False, umbral=umbral,
                                    roi_completa=completa),
                )
                self._cola.put((revision, resultados, None))
            except Exception as error:  # pragma: no cover
                self._cola.put((revision, None, error))

        threading.Thread(target=trabajo, daemon=True, name="ocr-rois").start()

    def _recibir(self):
        if self._cerrado:
            return
        try:
            revision, resultados, error = self._cola.get_nowait()
        except queue.Empty:
            pass
        else:
            self._ocupado = False
            self.boton.configure(state="normal")
            if revision != self.app.revision:
                self.mensaje.set("La imagen o las ROIs cambiaron. Vuelva a reconocer.")
            elif error:
                mostrar_error("Error de reconocimiento", error, self)
                self.mensaje.set(str(error))
            else:
                self._resultados = resultados
                self.app.resultados = resultados
                self._mostrar_resultados()
                lecturas = [resultado.lectura for resultado in resultados]
                valores = "; ".join(f"{lectura.roi}: {lectura.valor_mostrado}"
                                    for lectura in lecturas)
                if self.guardar_csv.get():
                    try:
                        guardadas = registrar_lecturas(lecturas)
                        self.mensaje.set(
                            f"{valores} · {guardadas} fila(s) añadidas a data/lecturas.csv."
                        )
                    except Exception as fallo:
                        mostrar_error("OCR completado; no se pudo registrar", fallo, self)
                        self.mensaje.set(valores)
                else:
                    self.mensaje.set(f"{valores} · CSV no modificado.")
                self.app.estado.set("Lecturas reconocidas con su confianza.")
        self._poll = self.after(80, self._recibir)

    def _mostrar_resultados(self):
        self.tabla.delete(*self.tabla.get_children())
        for resultado in self._resultados:
            lectura = resultado.lectura
            minimo = lectura.confianza_minima
            padre = self.tabla.insert(
                "", "end", text=lectura.roi, open=True,
                values=(f"{lectura.valor_mostrado}",
                        f"{lectura.confianza:.0%} (mín. {minimo:.0%})",
                        lectura.etiqueta_rol),
            )
            if resultado.error:
                self.tabla.insert(padre, "end", text="Aviso",
                                  values=(resultado.error, "", ""))
            for indice, caracter in enumerate(resultado.caracteres, 1):
                self.tabla.insert(
                    padre, "end", text=f"Carácter {indice}",
                    values=(caracter.etiqueta, f"{caracter.confianza:.3f}",
                            caracter.candidato),
                )
        self._dibujar()

    def _dibujar(self):
        canvas = self.visor.canvas
        canvas.delete("ocr")
        if self.visor.imagen is None:
            return
        canonico = config.instrumento_canonico(self.instrumento.get())
        for roi in self.app.rois:
            if roi.instrumento != canonico:
                continue
            x1, y1 = self.visor.punto_canvas(roi.x, roi.y)
            x2, y2 = self.visor.punto_canvas(roi.x + roi.ancho, roi.y + roi.alto)
            canvas.create_rectangle(x1, y1, x2, y2, outline="#00e0a0", width=2, tags="ocr")
            canvas.create_text(x1 + 2, y1 + 2, text=roi.nombre, anchor="nw",
                               fill="#00e0a0", tags="ocr")
        for resultado in self._resultados:
            lectura = resultado.lectura
            _, color = color_confianza(lectura.confianza)
            for (dx, dy, dw, dh, confianza) in lectura.cajas:
                _, color_digito = color_confianza(confianza)
                x = lectura.caja_roi[0] + dx
                y = lectura.caja_roi[1] + dy
                canvas.create_rectangle(
                    *self.visor.punto_canvas(x, y),
                    *self.visor.punto_canvas(x + dw, y + dh),
                    outline=color_digito, width=1, tags="ocr",
                )
            if not lectura.estable and lectura.valido:
                canvas.create_text(
                    *self.visor.punto_canvas(lectura.caja_roi[0],
                                             lectura.caja_roi[1] + lectura.caja_roi[3]),
                    text=f"{lectura.valor_mostrado} · {lectura.confianza:.0%}",
                    anchor="nw", fill=color, tags="ocr",
                )
