# LCD OCR

Aplicación modular en Python 3.11+ para Raspberry Pi 4/5 con Raspberry Pi OS
Bookworm, preferiblemente de 64 bits, y cámara CSI OV5647 mediante Picamera2.
También se puede desarrollar y probar la interfaz en Windows.

## Funciones

La aplicación mantiene la lógica de visión en `core/` y una interfaz Tkinter
con cinco pestañas en `ui/`:

1. Captura de Imagen.
2. Corrección de Perspectiva.
3. Filtrado y Binarización.
4. Definición de ROIs.
5. Segmentación y Reconocimiento.

El flujo es captura o archivo → perspectiva → binarización → ROIs → OCR y CSV.
La ventana abre sin cámara conectada; el dataset se carga al reconocer.
Cada archivo Python tiene menos de 400 líneas, con docstrings en español.

La lógica de visión pertenece a `core/`; la interfaz, a `ui/`. Las rutas y los
parámetros se centralizan en `config.py`. No se requieren PyYAML ni CustomTkinter.

## Windows

Usa Python 3.11+ con Tcl/Tk instalado. Abre PowerShell en esta carpeta `lcd_ocr`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

`requirements.txt` instala NumPy, Pillow y OpenCV 4.x para el procesamiento.
Tkinter forma parte de la instalación de Python; Picamera2 se utilizará en la Pi.
Si aparece un error al importar Tkinter, repara la instalación de Python e incluye
el componente Tcl/Tk.

## Raspberry Pi OS Bookworm

Usa una sesión de escritorio para mostrar Tkinter. Copia esta carpeta como
`~/lcd_ocr` y ejecuta:

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-numpy python3-pil python3-pil.imagetk python3-tk python3-venv
cd ~/lcd_ocr
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python main.py
```

El entorno comparte las dependencias instaladas por `apt`, incluidas las
bibliotecas de cámara. Raspberry Pi recomienda instalar Picamera2 mediante
`apt` para mantenerlo compatible con libcamera; consulta su
[documentación oficial](https://github.com/raspberrypi/picamera2/blob/main/README.md#installation).
En la Pi utiliza los paquetes anteriores y omite la instalación por pip de
`requirements.txt`: sus entradas están reservadas a Windows.

Si aparece un error sobre `$DISPLAY`, abre la aplicación desde el escritorio de
la Pi o desde una sesión gráfica remota configurada.

## Dataset propio

El reconocimiento de los próximos módulos utilizará exclusivamente tus recortes.
No se descargan MNIST, EMNIST ni otros datasets.

Las dos claves de `config.py` son `DATASET_BALANZA` y `DATASET_TERMOHIGRO`.
Las rutas originales de Windows son:

```text
C:\Users\CATLORD\Documents\Codex\2026-10-09\c-users-catlord-downloads-recortes\outputs\recortes_lcd\Balanza
C:\Users\CATLORD\Documents\Codex\2026-10-09\c-users-catlord-downloads-recortes\outputs\recortes_lcd\termohiglobarometro
```

Conserva los nombres de las carpetas y archivos al copiarlos a la Pi. El destino
recomendado es:

```text
~/lcd_ocr/datasets/Balanza/
~/lcd_ocr/datasets/termohiglobarometro/
```

El proyecto entregado incluye una copia de tus 34 recortes originales en esas
dos carpetas (13 de Balanza y 21 de Termohigrobarómetro). Al copiar el proyecto
completo como `~/lcd_ocr`, el fallback los encuentra sin editar las rutas.
Si agregas recortes después, cópialos conservando sus nombres. También puedes
definir otra ubicación en las dos claves de `config.py`.

`resolver_ruta_dataset(instrumento)` busca, en este orden:

1. La ruta configurada para el instrumento.
2. `datasets/Balanza` o `datasets/termohiglobarometro` dentro del proyecto.
3. `~/lcd_ocr/datasets/Balanza` o `~/lcd_ocr/datasets/termohiglobarometro`.

Las carpetas sin archivos de imagen se omiten para probar el siguiente fallback.
Si no encuentra los recortes, el error detalla las ubicaciones intentadas y dónde
copiar las carpetas. El cargador valida cada imagen, normaliza a 28×28 conservando
la proporción y mantiene las plantillas en memoria. Las consultas reciben la
misma normalización, con primer plano blanco sobre fondo negro.

| Nombre real | Clase |
|---|---|
| `digito_0.png` … `digito_9.png` | `0` … `9` |
| `coma.png`, `punto.png`, `signo_menos.png` | `,`, `.`, `-` |
| `porcentaje.png` | `%` |
| `grados_celsius.png`, `grados_fahrenheit.png` | `°C`, `°F` |
| `humedad_relativa.png` | `%RH` |
| `texto_data_logger.png`, `texto_humidity.png` | `Data Logger`, `Humidity` |
| `texto_pressure.png`, `texto_temperature.png` | `Pressure`, `Temperature` |

Para varias muestras de una clase, utiliza `digito_3__01.png`,
`digito_3__02.png`, etc. Los nombres desconocidos producen un error explícito;
añade su mapeo en `core/dataset.py` antes de utilizarlos.

Comprueba el dataset desde la carpeta del proyecto, con el entorno activo:

```bash
python -m core.dataset
```

La consola debe mostrar la ruta utilizada, el total de imágenes y clases y una
línea por clase: inicialmente 13/13 para Balanza y 21/21 para Termohigrobarómetro.
Para verificar solamente uno:

```bash
python -m core.dataset Balanza
```

## Inicio y prueba por módulos

Con el entorno virtual activo y desde `lcd_ocr`, ejecuta:

```bash
python main.py
```

Debe abrir una ventana con las cinco pestañas y sus controles. Los mensajes de
diagnóstico se envían a consola mediante `logging`.

### Módulo 1: Captura

En la Pi, comprueba primero que se detecta la cámara:

```bash
rpicam-hello --list-cameras
```

En la pestaña 1 pulsa **Iniciar cámara** y después **Capturar**. Debes ver el
preview y obtener `assets/capturas/captura_actual.jpg` a 2592×1944. El preview usa
640×480, aproximadamente 10 actualizaciones por segundo, y mantiene sólo el
frame más reciente. **Detener cámara** libera el dispositivo. Al cerrar la
ventana, una captura en curso termina y libera la cámara antes de salir.

En Windows o sin cámara, utiliza **Abrir imagen**. Esta opción guarda también
`captura_actual.jpg` y permite continuar con todo el flujo. Se admite PNG/JPEG,
BMP y TIFF. Picamera2 se importa únicamente cuando se solicita iniciar la cámara.

### Módulo 2: Perspectiva

Selecciona las cuatro esquinas sobre la imagen original, preferiblemente en el
orden indicado en pantalla. Usa **Deshacer punto** si te equivocas y pulsa
**Corregir perspectiva**. Debes ver el LCD rectificado a la derecha. La homografía
acepta puntos en cualquier orden y rechaza esquinas repetidas o fuera de la imagen.
**Guardar corregida** permite conservar el resultado, preferiblemente en PNG.

### Módulo 3: Filtrado

Ajusta threshold o selecciona Otsu/adaptativo. Prueba kernel, erosión, dilatación,
apertura, cierre e inversión: el preview cambia en vivo sin bloquear la ventana.
El kernel debe ser positivo e impar. En OpenCV la morfología actúa sobre blanco:
activa **Invertir** para que los segmentos oscuros sean el primer plano al aplicar
esas operaciones. Comienza con **ninguna** para conservar puntos y trazos finos.
El filtrado usa la imagen corregida si existe o la captura original.
**Guardar binaria** genera PNG para evitar artefactos de compresión JPEG.

### Módulo 4: ROIs

Escribe un nombre, selecciona el instrumento y arrastra sobre el campo a leer.
Deja un pequeño margen de fondo. Define una ROI por línea numérica; crea ROIs
separadas para temperatura, humedad y presión. Las coordenadas se refieren a
los píxeles originales, independientemente del tamaño del preview.

Pulsa **Guardar ROIs**: `data/rois.json` debe contener una lista de rectángulos.
**Cargar ROIs** permite reutilizarlos cuando la posición del LCD coincide. Se
rechazan nombres duplicados para un mismo instrumento y rectángulos fuera de
la imagen. Una captura o perspectiva nueva descarta las ROIs en memoria para
evitar usar coordenadas antiguas; el archivo guardado se conserva para recargarlo.

### Módulo 5: Segmentación y reconocimiento

Selecciona instrumento, método OCR y ROI. **Todas** procesa las ROIs asignadas
al instrumento; el selector individual permite reconocer un campo aislado.

- Usa **contornos** o **proyeccion** para una línea numérica. Las barras
  desconectadas que comparten columnas se agrupan; puntos y comas se conservan.
- Ajusta **Unir barras (px)** si un dígito se divide en piezas. Mantén ese valor
  por debajo del espacio entre caracteres para evitar fusionarlos.
- **Área mínima** elimina ruido pequeño. Empieza con 2 para preservar decimales.
- Usa **ROI completa** para un símbolo compuesto, unidad o texto incluido en
  tus recortes, como `%RH`, `°C` o `Humidity`. Reconoce estas ROIs individualmente.

Pulsa **Reconocer y registrar**. Debes ver el valor, score por carácter, mejor
candidato y cajas de segmentación. Se añade una fila por ROI a
`data/lecturas.csv`, con timestamp local ISO y una lista JSON de confianzas.
El OCR no se ejecuta mientras haya un filtrado pendiente y se descartan resultados
si la imagen, ROIs o instrumento cambian durante el cálculo.

**Abrir imagen procesada** permite partir de una binaria ya guardada; si se abre
una imagen de grises, se aplica Otsu. Después dibuja o carga sus ROIs.

### Módulo 6: Flujo completo

Realiza las etapas anteriores, cambia un parámetro de filtrado y reconoce otra
vez. Debes obtener una segunda lectura CSV y resultados correspondientes a la
imagen vigente. Los errores de cámara, dataset, ROIs, geometría o guardado se
muestran con un mensaje y se registran en consola.

## Reconocimiento y ajuste

Template matching utiliza `cv2.matchTemplate` con `TM_CCOEFF_NORMED`; K-NN usa
píxeles normalizados y se entrena únicamente con las plantillas locales. Ambas
opciones comparan recortes normalizados y admiten pequeños desplazamientos.

`UMBRAL_CONFIANZA=0.75` marca como `?` un score inferior al umbral. Los scores
expresan similitud, no una probabilidad calibrada de acierto. El mejor candidato
sigue visible para ayudar a ajustar el recorte y los parámetros.

Actualmente hay una muestra por clase: K-NN usa K efectivo=1 aunque
`K_VECINOS=3`; aumenta el número de muestras por clase si deseas usar más vecinos.
Las palabras completas se normalizan como una clase; no se implementa un OCR
general de letras que no estén en tus recortes.

Los parámetros de cámara, plantillas, filtros, segmentación, tolerancia de
desplazamiento y confianza están centralizados en `config.py`. Modifica ese
archivo y reinicia la aplicación para cambiar sus valores predeterminados.

## APIs del backend

```python
from core.dataset import cargar_plantillas
from core.perspectiva import corregir_perspectiva
from core.preprocesado import filtrar_imagen
from core.roi import ROI, recortar_roi, guardar_rois, cargar_rois
from core.segmentacion import segmentar_caracteres
from core.reconocimiento import reconocer_segmentos
from core.log_lecturas import registrar_lectura
```

Las imágenes internas son `numpy.ndarray` uint8 RGB o de grises; Picamera2 y
los archivos se convierten al mismo contrato. `cargar_plantillas(instrumento)`
devuelve `list[(label, imagen)]` y las matrices cacheadas son de sólo lectura.
Los objetos `Segmento` guardan imagen y caja; `ResultadoCaracter` contiene
`etiqueta`, `confianza`, `candidato` y `metodo`.

## Validación y alcance

Se comprobaron en Windows los recortes originales con ambos métodos, variantes
con ruido leve, segmentación de LCD de siete segmentos, perspectiva, filtrado,
persistencia JSON/CSV y el flujo Tkinter. Entorno de prueba: Python 3.12.14,
OpenCV 4.14.0, NumPy 2.5.3 y Pillow 12.3.0. El código usa APIs de OpenCV 4.x
disponibles en Raspberry Pi OS Bookworm.

Resultados: 80/80 combinaciones de dígitos originales con ambos segmentadores
y ambos métodos OCR; 96/96 lecturas compuestas con signo, punto/coma y escalas
1/2/4; 68/68 recortes tras perspectiva y Otsu; 68/68 recortes originales y 68/68
variantes con ruido leve. Las entradas uniformes y de ruido desconocido se
rechazaron como `?`. Se corrigieron los fragmentos internos contados como
signos y la influencia del marcador verde de esquina presente en tus recortes.

Estas pruebas no miden la precisión sobre capturas nuevas: ajústala con imágenes
reales del montaje. Mantén un margen de fondo y una iluminación uniforme.
La cámara CSI requiere una prueba en la Pi; sus llamadas y liberación se probaron
con un controlador simulado, porque este entorno no tiene una OV5647 conectada.

Fuentes de referencia: [Picamera2](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf),
[template matching](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html) y
[transformaciones de perspectiva](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).

