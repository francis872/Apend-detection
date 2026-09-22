# Detector geoespacial de outliers

Proyecto cooperativo para cargar, concatenar (`append`), depurar y analizar capas Shapefile con un pipeline estadistico-geoespacial.

## Flujo implementado

1. Lectura de uno o varios Shapefiles con GeoPandas.
2. Normalizacion de CRS y concatenacion mediante `pandas.concat` (reemplazo moderno de `DataFrame.append`).
3. Validacion de geometria, eliminacion de duplicados y limpieza de ruido basico.
4. Ajuste de funciones de probabilidad sobre variables numericas, con comparacion por AIC, BIC, Kolmogorov-Smirnov, Bhattacharyya y Fisher-Rao.
5. Analisis multivariado robusto.
6. Geometria SPD sobre matrices de covarianza locales.
7. Distancia Riemanniana afín-invariante para SPD.
8. Procrustes para detectar configuraciones espaciales anomalas.
9. Puntaje combinado de anomalia.
10. Umbrales 90%, 95% y 99%.
11. Exportacion de capa completa marcada y de dataset limpio.

## Datos procesados

- Registros totales combinados: 62,941.
- CRS: EPSG:4326.
- Variables utilizadas: `BRIGHTNESS`, `BRIGHT_T31`, `FRP`, `SCAN`, `TRACK` y geometria.
- Geometrias invalidas detectadas en la corrida: 0.
- Duplicados exactos eliminados en la corrida: 0.

## Resultado de referencia

Los umbrales observados en la corrida actual del `outlier_score` fueron aproximadamente:

- 90%: 0.6890 -> 6,295 observaciones marcadas.
- 95%: 0.7315 -> 3,148 observaciones marcadas.
- 99%: 0.8197 -> 630 observaciones marcadas.

La capa `cleaned.gpkg` elimina por defecto solamente los outliers del nivel 99%, para evitar descartar eventos extremos que pueden ser reales.

## Distribuciones

Se comparan 20 distribuciones candidatas. En la corrida actual la distribucion F obtuvo el mejor desempeno agregado entre las candidatas evaluadas. Esto no implica ajuste perfecto: el test KS muestra que aun existen diferencias estadisticamente significativas entre el modelo y los datos observados.

## Interfaz web

La interfaz esta en `ui/index.html` y muestra histograma, ajuste probabilistico, ranking de distribuciones, umbrales 90/95/99 y mapa.

Para abrirla localmente:

```bash
python -m http.server 8000
```

Luego abre `http://localhost:8000/ui/`.

## Ejecucion

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ejemplo:

```bash
PYTHONPATH=src python -m geo_outliers.cli data/raw --output results --probability-feature FRP
```

## Archivos de datos

Los GeoPackage y Shapefiles binarios no se versionan en Git. Los resultados tabulares reproducibles `summary.json` y `distribution_fits.csv` si se incluyen.
