"""Plantillas locales etiquetadas por los nombres reales de los recortes."""

from collections import Counter
import logging
from pathlib import Path
import re
from threading import RLock

import cv2
import numpy as np

import config
from core.utils import a_grises


LOG = logging.getLogger(__name__)
_CACHE: dict[tuple, tuple[tuple[str, np.ndarray], ...]] = {}
_LOCK = RLock()
_ETIQUETAS = {
    "coma": ",",
    "punto": ".",
    "signo_menos": "-",
    "porcentaje": "%",
    "porciento": "%",
    "grados_celsius": "°C",
    "grados_fahrenheit": "°F",
    "humedad_relativa": "%RH",
    "texto_data_logger": "Data Logger",
    "texto_humidity": "Humidity",
    "texto_pressure": "Pressure",
    "texto_temperature": "Temperature",
}


def etiqueta_de_archivo(archivo: str | Path) -> str:
    """Obtenga la clase del nombre real; rechace nombres sin mapeo conocido.

    Acepte ``digito_0.png`` a ``digito_9.png`` y símbolos/textos del dataset.
    Para nuevas muestras de una clase use, por ejemplo, ``digito_3__02.png``:
    el sufijo doble guion bajo evita confundir la muestra con otro dígito.
    También acepte los nombres simples ``0.png`` a ``9.png``.
    """
    nombre = Path(archivo).stem.casefold()
    nombre = re.sub(r"__\d+$", "", nombre)
    coincidencia = re.fullmatch(r"(?:digito_)?([0-9])", nombre)
    if coincidencia:
        return coincidencia.group(1)
    if nombre in _ETIQUETAS:
        return _ETIQUETAS[nombre]
    raise ValueError(
        f"Nombre de recorte sin etiqueta conocida: {Path(archivo).name!r}. "
        "Use digito_0 ... digito_9, coma, punto, signo_menos, porcentaje, "
        "grados_celsius, grados_fahrenheit, humedad_relativa o texto_" 
        "data_logger/humidity/pressure/temperature. "
        "Añada el mapeo explícito en core/dataset.py si incorpora otra clase."
    )


def normalizar_caracter(
    imagen: np.ndarray, tamano: tuple[int, int] | None = None
) -> np.ndarray:
    """Binarice, recorte y centre conservando proporción en fondo negro.

    Use la misma transformación para plantillas y candidatos. Determine la
    polaridad por el borde del recorte, aplique Otsu y quite componentes
    diminutos respecto al mayor. Conserve segmentos desconectados, puntos
    de letras y símbolos; no deforme un signo estrecho para llenar el ancho.
    Devuelva uint8 con primer plano 255 y margen de dos píxeles a 28×28.
    """
    ancho, alto = tuple(tamano or config.TAMANO_PLANTILLA)
    if not isinstance(ancho, int) or not isinstance(alto, int) or min(ancho, alto) < 8:
        raise ValueError("TAMANO_PLANTILLA debe contener dos enteros de al menos 8.")
    gris = a_grises(imagen)
    if float(np.ptp(gris)) < 2:
        raise ValueError("El recorte no tiene contraste suficiente para OCR.")
    _, clara = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    borde = np.concatenate((clara[0], clara[-1], clara[:, 0], clara[:, -1]))
    binaria = cv2.bitwise_not(clara) if np.count_nonzero(borde) >= borde.size / 2 else clara
    cantidad, componentes, estadisticas, _ = cv2.connectedComponentsWithStats(binaria, 8)
    if cantidad <= 1:
        raise ValueError("No se encontró primer plano en el recorte.")
    areas = estadisticas[1:, cv2.CC_STAT_AREA]
    proporcion = float(getattr(config, "NORMALIZACION_AREA_RELATIVA_MINIMA", 0.02))
    if not np.isfinite(proporcion) or not 0 <= proporcion < 0.25:
        raise ValueError("NORMALIZACION_AREA_RELATIVA_MINIMA debe estar entre 0 y 0.25.")
    minimo = max(1, int(np.ceil(areas.max() * proporcion)))
    conservar = np.concatenate(([False], areas >= minimo))
    mascara = np.where(conservar[componentes], 255, 0).astype(np.uint8)
    puntos = cv2.findNonZero(mascara)
    if puntos is None:
        raise ValueError("No se encontró un carácter después de quitar ruido.")
    x, y, w, h = cv2.boundingRect(puntos)
    contenido = mascara[y:y + h, x:x + w]
    margen = max(1, round(min(ancho, alto) / 14))
    escala = min((ancho - 2 * margen) / w, (alto - 2 * margen) / h)
    destino = (max(1, round(w * escala)), max(1, round(h * escala)))
    reducido = cv2.resize(contenido, destino, interpolation=cv2.INTER_AREA)
    _, reducido = cv2.threshold(reducido, 127, 255, cv2.THRESH_BINARY)
    salida = np.zeros((alto, ancho), dtype=np.uint8)
    izquierda, superior = (ancho - destino[0]) // 2, (alto - destino[1]) // 2
    salida[superior:superior + destino[1], izquierda:izquierda + destino[0]] = reducido
    return salida


def cargar_plantillas(instrumento: str) -> list[tuple[str, np.ndarray]]:
    """Cargue MIS recortes, normalizados en gris; reutilice la caché en memoria.

    Resuelva rutas mediante config, lea PNG/JPEG/etc. de forma compatible
    con rutas Unicode y registre clases y cantidad de muestras por clase.
    Las matrices cacheadas son de solo lectura para evitar cambios accidentales.
    """
    ruta = config.resolver_ruta_dataset(instrumento)
    tamano = tuple(config.TAMANO_PLANTILLA)
    proporcion = float(getattr(config, "NORMALIZACION_AREA_RELATIVA_MINIMA", 0.02))
    clave = (str(ruta), tamano, proporcion)
    with _LOCK:
        if clave in _CACHE:
            return list(_CACHE[clave])
        archivos = sorted(
            (f for f in ruta.rglob("*") if f.is_file()
             and f.suffix.casefold() in config.EXTENSIONES_IMAGEN),
            key=lambda f: str(f.relative_to(ruta)).casefold(),
        )
        if not archivos:
            raise FileNotFoundError(f"No hay recortes de imagen en {ruta}.")
        plantillas = []
        for archivo in archivos:
            etiqueta = etiqueta_de_archivo(archivo)
            try:
                datos = np.fromfile(archivo, dtype=np.uint8)
                imagen = cv2.imdecode(datos, cv2.IMREAD_GRAYSCALE)
                if imagen is None:
                    raise ValueError("OpenCV no pudo decodificar la imagen")
                normalizada = normalizar_caracter(imagen, tamano)
            except (OSError, ValueError, cv2.error) as error:
                raise ValueError(f"Recorte inválido {archivo}: {error}") from error
            normalizada.flags.writeable = False
            plantillas.append((etiqueta, normalizada))
        _CACHE[clave] = tuple(plantillas)
        conteos = Counter(etiqueta for etiqueta, _ in plantillas)
        LOG.info("Dataset %s: %s imágenes, %s clases; ruta=%s",
                 instrumento, len(plantillas), len(conteos), ruta)
        for etiqueta, cantidad in sorted(conteos.items()):
            LOG.info("  Clase %r: %s imagen(es)", etiqueta, cantidad)
        return list(_CACHE[clave])


def limpiar_cache() -> None:
    """Vacíe las plantillas cacheadas para recargar archivos modificados."""
    with _LOCK:
        _CACHE.clear()


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verifique MIS recortes locales y sus clases.")
    parser.add_argument("instrumento", nargs="?", help="Balanza o Termohigrobarómetro; ambos por defecto.")
    argumentos = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    instrumentos = [argumentos.instrumento] if argumentos.instrumento else ["Balanza", "Termohigrobarómetro"]
    try:
        for instrumento in instrumentos:
            cargar_plantillas(instrumento)
    except (FileNotFoundError, ValueError, OSError) as error:
        LOG.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    _main()
