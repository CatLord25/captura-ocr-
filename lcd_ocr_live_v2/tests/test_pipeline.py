"""Pruebas del núcleo de visión y de la compatibilidad de datos."""

from pathlib import Path
import sys
import tempfile
import unittest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import cv2
import numpy as np

import config
from core.dataset import cargar_banco, normalizar_caracter
from core.filtro_temporal import Estabilizador
from core.instrumentos import parsear_numero
from core.log_lecturas import contar_registros, leer_ultimas, registrar_lectura
from core.perspectiva import Rectificador
from core.pipeline import OpcionesProceso, PipelineOCR
from core.preprocesado import preprocesar
from core.roi import ROI, cargar_rois, guardar_rois
from core.segmentacion import segmentar_preproceso
from core.simulador import GeneradorEscenas


class PruebasDataset(unittest.TestCase):
    """Plantillas canónicas y conservación de proporción."""

    def test_plantillas_canonicas(self):
        for instrumento in ("Balanza", "Termohigrobarómetro"):
            banco = cargar_banco(instrumento)
            self.assertGreaterEqual(len(banco.plantillas), 13)
            for plantilla in banco.plantillas:
                self.assertEqual(plantilla.imagen.shape,
                                 (config.TAMANO_PLANTILLA[1], config.TAMANO_PLANTILLA[0]))
                self.assertEqual(plantilla.imagen.dtype, np.uint8)
                self.assertFalse(plantilla.imagen.flags.writeable)
                self.assertTrue(len(plantilla.aumentadas) >= 4)

    def test_digito_uno_no_se_estira(self):
        """Un trazo estrecho debe conservar su proporción dentro del lienzo."""
        lienzo = np.full((120, 80), 220, np.uint8)
        cv2.rectangle(lienzo, (36, 12), (44, 108), 20, -1)
        normalizada = normalizar_caracter(lienzo, perfil="7seg")
        columnas = np.flatnonzero(np.any(normalizada > 0, axis=0))
        self.assertTrue(columnas.size > 0)
        ancho_util = columnas[-1] - columnas[0] + 1
        self.assertLess(ancho_util / normalizada.shape[1], 0.60)
        # Las columnas extremas del lienzo deben quedar libres.
        self.assertTrue(np.all(normalizada[:, :2] == 0))
        self.assertTrue(np.all(normalizada[:, -2:] == 0))


class PruebasSegmentacion(unittest.TestCase):
    """Proyección, puenteo y clasificación geométrica."""

    def test_segmenta_seis_caracteres(self):
        generador = GeneradorEscenas("termohigrobarómetro")
        frame = generador.frame_muestra(0)
        roi = generador.rois()[2]
        recorte = frame[roi.y:roi.y + roi.alto, roi.x:roi.x + roi.ancho]
        pre = preprocesar(recorte, perfil="7seg")
        segmentos = segmentar_preproceso(pre)
        self.assertEqual(len(segmentos), 6)  # 1013.2
        self.assertEqual(segmentos[4].clase, "puntuacion")
        # Orden ascendente por x.
        self.assertEqual([s.x for s in segmentos], sorted(s.x for s in segmentos))

    def test_roi_sin_tinta_devuelve_vacio(self):
        pre = preprocesar(np.full((60, 120, 3), 128, np.uint8), perfil="7seg")
        self.assertEqual(segmentar_preproceso(pre), [])


class PruebasOCR(unittest.TestCase):
    """OCR de extremo a extremo con escenas sintéticas reales."""

    def test_valores_sinteticos(self):
        for instrumento in ("Balanza", "Termohigrobarómetro"):
            generador = GeneradorEscenas(instrumento)
            for indice in range(generador.cantidad_muestras):
                with self.subTest(instrumento=instrumento, indice=indice):
                    frame = generador.frame_muestra(indice)
                    esperado = generador.valores_muestra(indice)
                    linea = PipelineOCR()
                    resultados = linea.procesar(
                        frame, generador.rois(),
                        OpcionesProceso(filtro_temporal=False),
                    )
                    obtenido = {r.lectura.roi: r.lectura.texto for r in resultados}
                    self.assertEqual(obtenido, esperado)

    def test_unidad_reconocida(self):
        banco = cargar_banco("Termohigrobarómetro")
        from core.reconocimiento import obtener_motor

        motor = obtener_motor("Termohigrobarómetro")
        for etiqueta in ("°C", "°F", "%RH", "%"):
            plantilla = next(p for p in banco.plantillas if p.etiqueta == etiqueta)
            crudo = cv2.imdecode(np.fromfile(plantilla.archivo, np.uint8),
                                 cv2.IMREAD_GRAYSCALE)
            resultado = motor.reconocer(crudo, "unidad")
            self.assertEqual(resultado.etiqueta, etiqueta)


class PruebasTemporal(unittest.TestCase):
    """Votación por moda y descarte de valores dudosos."""

    def test_moda_y_descarte(self):
        filtro = Estabilizador(5, 3)
        for valor in ("12.34", "12.34", "12.38", "12.34", "12.34"):
            voto = filtro.agregar(valor, 0.9)
        self.assertEqual(voto.valor, "12.34")
        self.assertTrue(voto.estable)
        self.assertEqual(voto.acuerdos, 4)
        filtro.agregar("1?.34", 0.2)
        self.assertEqual(filtro.ultimo.muestras, 5)


class PruebasSemantica(unittest.TestCase):
    """Conversión de texto OCR a número."""

    def test_parsear(self):
        casos = {
            "12,345": 12.345, "-0.125": -0.125, "1013.2": 1013.2,
            "-5,0": -5.0, "0.000": 0.0, "?2.4": None, "": None, "1.234,5": 1234.5,
        }
        for texto, esperado in casos.items():
            with self.subTest(texto=texto):
                self.assertEqual(parsear_numero(texto), esperado)


class PruebasROIs(unittest.TestCase):
    """Compatibilidad del JSON de ROIs."""

    def test_formato_original(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "rois.json"
            ruta.write_text(
                '[{"nombre": "valor_1", "x": 1, "y": 2, "ancho": 30, "alto": 40, '
                '"instrumento": "Balanza"}]', encoding="utf-8")
            rois = cargar_rois(ruta)
            self.assertEqual(len(rois), 1)
            self.assertEqual(rois[0].rol, "")
            self.assertEqual(rois[0].instrumento, "Balanza")

    def test_roundtrip_extendido(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "rois.json"
            original = [ROI("peso", 4, 5, 60, 20, "Balanza", "peso", "kg")]
            guardar_rois(original, ruta)
            recuperadas = cargar_rois(ruta)
            self.assertEqual(recuperadas, original)

    def test_rechaza_campos_desconocidos(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "rois.json"
            ruta.write_text('[{"nombre": "a", "x": 0, "y": 0, "ancho": 4, "alto": 4, '
                            '"instrumento": "Balanza", "extra": 1}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                cargar_rois(ruta)


class PruebasPerspectiva(unittest.TestCase):
    """Homografía, persistencia y reescalado."""

    def test_rectifica_y_guarda(self):
        imagen = np.zeros((300, 400, 3), np.uint8)
        cv2.rectangle(imagen, (50, 60), (350, 240), (0, 200, 0), -1)
        rectificador = Rectificador.desde_puntos(
            [(50, 60), (350, 60), (350, 240), (50, 240)],
            tamano_salida=(300, 180), resolucion_origen=(400, 300),
        )
        salida = rectificador.aplicar(imagen)
        self.assertEqual(salida.shape, (180, 300, 3))
        self.assertGreater(salida.mean(), 0)
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "perspectiva.json"
            rectificador.guardar(ruta)
            recuperado = Rectificador.cargar(ruta)
            self.assertEqual(recuperado, rectificador)
            escalado = recuperado.para_resolucion(800, 600)
            self.assertEqual(escalado.tamano, (300, 180))
            self.assertEqual(escalado.resolucion_origen, (800, 600))


class PruebasCSV(unittest.TestCase):
    """Registro y cabecera compatible."""

    def test_registro(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "lecturas.csv"
            registrar_lectura("Balanza", "peso", "1.234", [0.9], ruta)
            registrar_lectura("Balanza", "peso", "1.235", [0.8], ruta)
            self.assertEqual(contar_registros(ruta), 2)
            self.assertEqual(leer_ultimas(1, ruta)[0]["valor"], "1.235")
            cabecera = ruta.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(cabecera, "timestamp,instrumento,roi,valor,confianzas")


if __name__ == "__main__":
    unittest.main(verbosity=2)
