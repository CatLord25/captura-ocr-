"""Verificación integral del proyecto LCD OCR LIVE v2.

Comprueba imports, datasets, compatibilidad de ``rois.json``, OCR de extremo a
extremo con escenas sintéticas, filtro temporal, CSV y render del HUD.

Uso:
    python tools/verificar_instalacion.py
    python tools/verificar_instalacion.py --camara --original RUTA_AL_PROYECTO_V1
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import cv2
import numpy as np

import config


class Informe:
    """Acumule resultados sin abortar la verificación completa."""

    def __init__(self) -> None:
        """Inicialice contadores y líneas de salida."""
        self.ok = 0
        self.avisos = 0
        self.fallos = 0
        self.lineas: list[str] = []

    def bien(self, mensaje: str) -> None:
        """Registre una comprobación superada."""
        self.ok += 1
        self.lineas.append(f"  [OK]    {mensaje}")

    def aviso(self, mensaje: str) -> None:
        """Registre una advertencia no bloqueante."""
        self.avisos += 1
        self.lineas.append(f"  [AVISO] {mensaje}")

    def fallo(self, mensaje: str) -> None:
        """Registre un fallo bloqueante."""
        self.fallos += 1
        self.lineas.append(f"  [FALLO] {mensaje}")

    def comprobar(self, nombre: str, funcion) -> bool:
        """Ejecute una comprobación y capture cualquier excepción."""
        try:
            detalle = funcion()
        except Exception as error:  # noqa: BLE001 - informe de diagnóstico
            self.fallo(f"{nombre}: {type(error).__name__}: {error}")
            return False
        self.bien(f"{nombre}{f' — {detalle}' if detalle else ''}")
        return True

    def concluir(self) -> int:
        """Imprima el informe y devuelva el código de salida."""
        print("\n".join(self.lineas))
        print(
            f"\nResumen: {self.ok} comprobaciones OK, {self.avisos} avisos, "
            f"{self.fallos} fallos."
        )
        if self.fallos:
            print("RESULTADO: FALLO — revise los puntos marcados.")
            return 1
        print("RESULTADO: OK — el proyecto está listo para operar.")
        return 0


def _importar_modulos(informe: Informe) -> dict:
    """Importe el núcleo y devuelva el diccionario de módulos."""
    modulos = {}
    for nombre in (
        "core.utils", "core.roi", "core.perspectiva", "core.preprocesado",
        "core.segmentacion", "core.dataset", "core.siete_segmentos",
        "core.reconocimiento", "core.filtro_temporal", "core.instrumentos",
        "core.pipeline", "core.hud", "core.camara", "core.simulador",
        "core.monitor", "core.log_lecturas", "ui.widgets",
    ):
        try:
            modulos[nombre] = __import__(nombre, fromlist=["*"])
            informe.bien(f"Import {nombre}")
        except Exception as error:  # noqa: BLE001
            informe.fallo(f"Import {nombre}: {error}")
    return modulos


def _verificar_entorno(informe: Informe) -> None:
    """Versiones de Python, OpenCV y backend de OpenCV ML."""
    def version_python() -> str:
        if sys.version_info < (3, 11):
            raise RuntimeError(f"versión {sys.version.split()[0]} < 3.11")
        return sys.version.split()[0]

    informe.comprobar("Python 3.11+", version_python)
    informe.comprobar("OpenCV", lambda: cv2.__version__)
    if hasattr(cv2, "ml") and hasattr(cv2.ml, "KNearest_create"):
        informe.bien("cv2.ml.KNearest disponible (KNN nativo de OpenCV)")
    else:
        informe.aviso(
            f"OpenCV {cv2.__version__} sin cv2.ml.KNearest; se usará el KNN numpy "
            "equivalente (mismo resultado, portable en OpenCV 5)."
        )
    informe.comprobar("numpy", lambda: np.__version__)
    try:
        import PIL  # noqa: F401

        informe.bien(f"Pillow {PIL.__version__} disponible (volcado rápido a Tk)")
    except ImportError:
        informe.aviso(
            "Pillow no instalado: Tk usará cv2.imencode + PhotoImage (más lento)."
        )
    try:
        import tkinter

        informe.bien(f"Tkinter {tkinter.TkVersion}")
    except ImportError:
        informe.aviso("Tkinter no disponible: use main_live_hud.py en la Pi headless.")


def _verificar_datasets(informe: Informe, modulos: dict) -> dict:
    """Cargue ambos bancos y valide etiquetas y plantillas canónicas."""
    dataset = modulos["core.dataset"]
    bancos = {}
    for instrumento in ("Balanza", "Termohigrobarómetro"):
        def cargar(instrumento=instrumento):
            banco = dataset.cargar_banco(instrumento)
            faltan = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                      ",", ".", "-"} - set(banco.etiquetas())
            if faltan:
                raise RuntimeError(f"faltan etiquetas {sorted(faltan)}")
            for plantilla in banco.plantillas:
                if plantilla.imagen.shape != (config.TAMANO_PLANTILLA[1],
                                              config.TAMANO_PLANTILLA[0]):
                    raise RuntimeError(
                        f"{plantilla.archivo.name}: forma {plantilla.imagen.shape}"
                    )
            if instrumento != "Balanza":
                faltan_unidades = {"%", "%RH", "°C", "°F"} - set(banco.etiquetas())
                if faltan_unidades:
                    raise RuntimeError(f"faltan unidades {sorted(faltan_unidades)}")
            bancos[instrumento] = banco
            return (f"{len(banco.plantillas)} recortes, {len(banco.etiquetas())} clases, "
                    f"perfil {banco.perfil}")
        informe.comprobar(f"Dataset {instrumento}", cargar)
    return bancos


def _verificar_rois(informe: Informe, modulos: dict) -> None:
    """Valide compatibilidad con el JSON original y el formato extendido."""
    roi_mod = modulos["core.roi"]
    with tempfile.TemporaryDirectory() as temporal:
        carpeta = Path(temporal)
        antiguo = carpeta / "rois_v1.json"
        antiguo.write_text(json.dumps([
            {"nombre": "valor_1", "x": 10, "y": 20, "ancho": 100, "alto": 40,
             "instrumento": "Balanza"}
        ]), encoding="utf-8")
        cargadas = roi_mod.cargar_rois(antiguo)
        if len(cargadas) != 1 or cargadas[0].rol != "":
            raise RuntimeError("El formato original no se leyó correctamente.")
        informe.bien("ROIs: formato original de 6 campos compatible")

        nuevo = carpeta / "rois_v2.json"
        roi_mod.guardar_rois([
            roi_mod.ROI("peso", 10, 20, 120, 40, "Balanza", "peso", "kg"),
            roi_mod.ROI("temperatura", 10, 80, 120, 40,
                        "Termohigrobarómetro", "temperatura", "°C"),
        ], nuevo)
        releidas = roi_mod.cargar_rois(nuevo)
        if len(releidas) != 2 or releidas[0].unidad != "kg":
            raise RuntimeError("El formato extendido no sobrevivió el viaje JSON.")
        informe.bien("ROIs: round-trip con rol y unidad correcto")

        rois_actuales = roi_mod.cargar_rois(config.ARCHIVO_ROIS)
        informe.bien(f"ROIs del proyecto: {config.ARCHIVO_ROIS} "
                     f"({len(rois_actuales)} entradas)")


def _procesar_escena(pipeline_mod, generador, indice: int):
    """Procese una muestra sintética y devuelva (esperado, obtenido, ms)."""
    frame = generador.frame_muestra(indice)
    esperado = generador.valores_muestra(indice)
    rois = generador.rois()
    pipeline = pipeline_mod.PipelineOCR()
    opciones = pipeline_mod.OpcionesProceso(filtro_temporal=False)
    inicio = time.perf_counter()
    resultados = pipeline.procesar(frame, rois, opciones)
    milisegundos = (time.perf_counter() - inicio) * 1000.0
    obtenido = {resultado.lectura.roi: resultado.lectura.texto for resultado in resultados}
    return esperado, obtenido, milisegundos, frame, resultados


def _verificar_ocr(informe: Informe, modulos: dict) -> list[np.ndarray]:
    """OCR de extremo a extremo con los valores de las escenas sintéticas."""
    simulador = modulos["core.simulador"]
    pipeline_mod = modulos["core.pipeline"]
    capturas = []
    for instrumento in ("Balanza", "Termohigrobarómetro"):
        generador = simulador.GeneradorEscenas(instrumento)
        aciertos = 0
        total = 0
        tiempos = []
        for indice in range(generador.cantidad_muestras):
            esperado, obtenido, milisegundos, frame, resultados = _procesar_escena(
                pipeline_mod, generador, indice
            )
            total += len(esperado)
            for rol, texto in esperado.items():
                if obtenido.get(rol) == texto:
                    aciertos += 1
                else:
                    informe.aviso(
                        f"{instrumento} muestra {indice}: {rol} esperado {texto!r} "
                        f"obtenido {obtenido.get(rol)!r}"
                    )
            tiempos.append(milisegundos)
            if indice == 0:
                capturas.append((instrumento, frame, resultados))
        if aciertos == total:
            informe.bien(f"OCR sintético {instrumento}: {aciertos}/{total} valores")
        else:
            informe.fallo(f"OCR sintético {instrumento}: {aciertos}/{total} valores")
        informe.bien(
            f"Rendimiento {instrumento}: {statistics.median(tiempos):.1f} ms/frame "
            f"(media {statistics.mean(tiempos):.1f} ms, {len(tiempos)} muestras)"
        )
    # Comparación de todos los dígitos sueltos por instrumento.
    for instrumento in ("Balanza", "Termohigrobarómetro"):
        banco = modulos["core.dataset"].cargar_banco(instrumento)
        motor = modulos["core.reconocimiento"].obtener_motor(instrumento)
        correctos = 0
        for plantilla in banco.plantillas:
            if plantilla.etiqueta in "0123456789":
                crudo = cv2.imdecode(np.fromfile(plantilla.archivo, np.uint8),
                                     cv2.IMREAD_GRAYSCALE)
                resultado = motor.reconocer(crudo, "digito")
                correctos += resultado.etiqueta == plantilla.etiqueta
        informe.bien(f"Dígitos sueltos {instrumento}: {correctos}/10")
    return capturas, tiempos


def _verificar_temporal(informe: Informe, modulos: dict) -> None:
    """Compruebe la moda y el descarte de un frame transitorio."""
    filtro = modulos["core.filtro_temporal"]
    estabilizador = filtro.Estabilizador(tamano=5, minimo_acuerdos=3)
    for valor in ("12.34", "12.34", "12.38", "12.34", "12.34"):
        voto = estabilizador.agregar(valor, 0.9)
    if voto.valor != "12.34" or not voto.estable:
        raise RuntimeError(f"moda inesperada: {voto}")
    estable = filtro.Estabilizador(tamano=5, minimo_acuerdos=3)
    estable.agregar("12.34", 0.9)
    estable.agregar("1?.34", 0.4)
    if estable.ultimo.valor != "12.34" or estable.ultimo.muestras != 1:
        raise RuntimeError("Los valores con '?' no se descartaron.")
    informe.bien("Filtro temporal: moda de 5 y descarte de lecturas dudosas")


def _verificar_csv(informe: Informe, modulos: dict) -> None:
    """Registre lecturas en un CSV temporal y valide cabecera y conteo."""
    log = modulos["core.log_lecturas"]
    with tempfile.TemporaryDirectory() as temporal:
        ruta = Path(temporal) / "lecturas.csv"
        log.registrar_lectura("Balanza", "peso", "12.345", [0.9, 0.8], ruta)
        log.registrar_lectura("Termohigrobarómetro", "temperatura", "23.4", [0.95], ruta)
        if log.contar_registros(ruta) != 2:
            raise RuntimeError("conteo de filas incorrecto")
        primera = ruta.read_text(encoding="utf-8").splitlines()[0]
        if primera != "timestamp,instrumento,roi,valor,confianzas":
            raise RuntimeError(f"cabecera inesperada: {primera}")
        if not log.leer_ultimas(1, ruta):
            raise RuntimeError("leer_ultimas no devolvió filas")
    informe.bien("CSV: cabecera, conteo y lectura de las últimas filas")


def _verificar_hud(informe: Informe, modulos: dict, capturas, escribir: bool = True) -> None:
    """Genere el HUD y el dashboard y guarde una imagen de comprobación."""
    hud = modulos["core.hud"]
    if escribir:
        config.DIRECTORIO_CAPTURAS.mkdir(parents=True, exist_ok=True)
    guardadas = []
    for instrumento, frame, resultados in capturas:
        lecturas = [resultado.lectura for resultado in resultados]
        imagen = hud.dibujar_hud(frame, lecturas, 30.0, 12.0, "Grabando CSV", 42, False, 18.0)
        panel = hud.dibujar_dashboard(frame.shape[0], lecturas, "Grabando CSV", 42)
        vista = hud.componer_vista(imagen, panel)
        if vista.shape[0] != frame.shape[0] or vista.shape[1] <= frame.shape[1]:
            raise RuntimeError(f"composición inválida: {vista.shape}")
        if escribir:
            ruta = config.DIRECTORIO_CAPTURAS / (
                f"verificacion_hud_{instrumento[:6].lower()}.png"
            )
            if not cv2.imwrite(str(ruta), vista):
                raise RuntimeError("no se pudo escribir la imagen del HUD")
            guardadas.append(ruta.name)
    detalle = ", ".join(guardadas) if guardadas else "sin escritura de archivos"
    informe.bien(f"HUD y dashboard renderizados ({detalle})")


def _verificar_camara(modulos: dict) -> str:
    """Intente abrir la cámara y leer un frame (opcional)."""
    camara_mod = modulos["core.camara"]
    camara = camara_mod.CameraStream()
    try:
        camara.iniciar()
        frame = camara.leer(timeout=2.0)
        if frame is None:
            raise RuntimeError("no se recibió ningún frame")
        return (f"{camara.backend_usado} a {frame.shape[1]}×{frame.shape[0]}, "
                f"{camara.fps:.1f} FPS")
    finally:
        camara.cerrar()


def _verificar_original(ruta: Path) -> str:
    """Compruebe con git que el proyecto original sigue sin cambios."""
    if not ruta.is_dir():
        raise RuntimeError(f"no existe la carpeta {ruta}")
    resultado = subprocess.run(
        ["git", "-C", str(ruta), "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    )
    if resultado.returncode != 0:
        raise RuntimeError(f"git falló: {resultado.stderr.strip()}")
    if resultado.stdout.strip():
        raise RuntimeError(
            "el repositorio original tiene cambios:\n" + resultado.stdout.strip()
        )
    return f"git status limpio en {ruta}"


def _verificar_archivos(informe: Informe) -> None:
    """Compruebe la presencia de recursos y documentación."""
    requeridos = [
        config.RAIZ_PROYECTO / "requirements.txt",
        config.RAIZ_PROYECTO / "README.md",
        config.ARCHIVO_ROIS,
        config.ARCHIVO_LECTURAS,
        config.RAIZ_PROYECTO / "main.py",
        config.RAIZ_PROYECTO / "main_live_hud.py",
        config.RAIZ_PROYECTO / "datasets" / "Balanza",
        config.RAIZ_PROYECTO / "datasets" / "termohiglobarometro",
    ]
    faltan = [str(ruta.relative_to(config.RAIZ_PROYECTO))
              for ruta in requeridos if not ruta.exists()]
    if faltan:
        raise RuntimeError(f"faltan recursos: {faltan}")
    informe.bien("Archivos y carpetas requeridos presentes")


def _verificar_ocr_seguro(informe: Informe, modulos: dict, capturas: list) -> bool:
    """Ejecute el OCR de verificación y guarde las capturas para el HUD."""
    try:
        resultado, _ = _verificar_ocr(informe, modulos)
    except Exception as error:  # noqa: BLE001
        informe.fallo(f"OCR de extremo a extremo: {type(error).__name__}: {error}")
        return False
    capturas.extend(resultado)
    return bool(capturas)


def _seccion(informe: Informe, nombre: str, funcion, *args) -> None:
    """Ejecute una sección completa sin abortar el resto del informe."""
    try:
        funcion(*args)
    except Exception as error:  # noqa: BLE001
        informe.fallo(f"{nombre}: {type(error).__name__}: {error}")


def main(argv=None) -> int:
    """Ejecute la verificación completa."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camara", action="store_true",
                        help="Pruebe también la apertura real de la cámara.")
    parser.add_argument("--original", type=Path, default=None,
                        help="Ruta del proyecto v1 para verificar git status limpio.")
    parser.add_argument("--sin-escritura", action="store_true",
                        help="No escriba PNG de verificación (útil en carpetas de solo lectura).")
    argumentos = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    informe = Informe()
    print("=" * 78)
    print(f"LCD OCR LIVE v2 — verificación {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Proyecto: {RAIZ}")
    print("=" * 78)
    _verificar_entorno(informe)
    modulos = _importar_modulos(informe)
    if informe.fallos:
        return informe.concluir()
    _seccion(informe, "Archivos requeridos", _verificar_archivos, informe)
    _seccion(informe, "Datasets", _verificar_datasets, informe, modulos)
    _seccion(informe, "ROIs", _verificar_rois, informe, modulos)
    capturas: list = []
    if _verificar_ocr_seguro(informe, modulos, capturas):
        _seccion(informe, "HUD", _verificar_hud, informe, modulos, capturas,
                 not argumentos.sin_escritura)
    _seccion(informe, "Filtro temporal", _verificar_temporal, informe, modulos)
    _seccion(informe, "CSV", _verificar_csv, informe, modulos)
    if argumentos.camara:
        informe.comprobar("Cámara en vivo", lambda: _verificar_camara(modulos))
    if argumentos.original:
        informe.comprobar("Proyecto original sin cambios",
                          lambda: _verificar_original(argumentos.original))
    return informe.concluir()


if __name__ == "__main__":
    raise SystemExit(main())
