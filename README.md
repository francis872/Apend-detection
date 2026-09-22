# Detector geoespacial de outliers

Proyecto cooperativo para cargar, concatenar (`append`), depurar y analizar capas Shapefile con un pipeline estadistico-geoespacial.

## Flujo implementado

1. Lectura de uno o varios Shapefiles con GeoPandas.
2. Normalizacion de CRS y concatenacion mediante `pandas.concat`.
3. Validacion de geometria, eliminacion de duplicados y limpieza de ruido basico.
4. Histograma y ajuste de 20 funciones de probabilidad, comparadas por AIC, BIC, Kolmogorov-Smirnov, Bhattacharyya y Fisher-Rao.
5. **Cuadratura Gauss-Legendre:** integra la PDF ajustada para calcular area bajo la curva, intervalos centrales 90/95/99 y probabilidad bilateral de cola.
6. Analisis multivariado robusto.
7. Geometria SPD sobre matrices de covarianza locales.
8. Distancia Riemanniana afin-invariante para SPD.
9. Procrustes para detectar configuraciones espaciales anomalas.
10. Puntaje combinado de anomalia y niveles 90%, 95% y 99%.
11. Exportacion de capa completa marcada y dataset limpio.

## Cuadratura Gaussiana

El modulo `src/geo_outliers/quadrature.py` implementa Gauss-Legendre. Para una PDF `f(x)` calcula numericamente:

`P(a <= X <= b) = integral_a^b f(x) dx`.

El detector usa esa integral para construir `probability_tail_area` y `probability_quadrature_score = -log10(p_tail)`. Los intervalos centrales 90%, 95% y 99% se vuelven a integrar numericamente y el reporte registra el area obtenida y su error frente al area objetivo. Las distribuciones de soporte infinito se truncan solo para la integracion numerica mediante cuantiles extremos de la distribucion ajustada.

## Datos procesados

- Registros combinados de referencia: 62,941.
- CRS: EPSG:4326.
- Variables: `BRIGHTNESS`, `BRIGHT_T31`, `FRP`, `SCAN`, `TRACK` y geometria.

## Distribuciones

Se comparan 20 distribuciones candidatas. En la corrida de referencia, F obtuvo el mejor desempeno agregado entre las candidatas. El KS indica diferencias estadisticamente significativas, por lo que el consenso final no depende de una sola distribucion.

## Interfaz web

`ui/index.html` muestra histograma, PDF, ranking estadistico, niveles 90/95/99 y el bloque de cuadratura gaussiana.

```bash
python -m http.server 8000
```

Abre `http://localhost:8000/ui/`.

## Ejecucion

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python -m geo_outliers.cli data/raw --output results --probability-feature FRP
```

Si no instalas el paquete en modo editable, usa `PYTHONPATH=src` antes del comando.

## Archivos de datos

GeoPackage y Shapefiles binarios no se versionan. Los resultados tabulares reproducibles si se pueden conservar en `results/`.


## Version 0.2 - pipeline completo

Apend Detection ahora integra las diez capas de trabajo:

1. Ejecucion reproducible del pipeline con Gauss-Legendre.
2. API FastAPI: upload de uno o varios ZIP Shapefile y analisis desde la interfaz.
3. Histograma en UI y parametros de la distribucion ganadora disponibles en el resultado para representar la PDF real.
4. Explicabilidad por registro mediante `consensus_methods` y `anomaly_reason`.
5. Separacion explicita entre `anomaly_candidate` y `noise_candidate`; una anomalia valida no se elimina automaticamente.
6. Validacion espacial con kNN Local Moran, residual de vecindario y Local Outlier Factor espacial.
7. Validacion temporal robusta mediante mediana/MAD movil sobre `ACQ_DATE` + `ACQ_TIME`.
8. CI con GitHub Actions y pytest.
9. Exportacion QGIS: `analysis.gpkg`, `cleaned.gpkg`, `outliers_90.gpkg`, `outliers_95.gpkg`, `outliers_99.gpkg`, `analysis.csv`.
10. Reporte automatico `REPORT.md` + `summary.json`.

### Ejecutar la aplicacion

```bash
pip install -e .
uvicorn geo_outliers.api:app --reload
```

Abre `http://127.0.0.1:8000/ui/`. Desde alli selecciona uno o varios ZIP Shapefile y pulsa **Ejecutar analisis completo**.

### Criterio de ruido

`outlier_90/95/99` representa rareza estadistica. `noise_candidate` es deliberadamente mas conservador: requiere pertenecer al 1% superior del score combinado y que al menos tres metodos independientes alcancen score >= 0.95. Por eso `cleaned.gpkg` elimina candidatos a ruido, no todos los eventos extremos.

### Reproducir por CLI

```bash
geo-outliers data/raw --output results --probability-feature FRP --quadrature-order 48
```


## Apend Detection 1.0 - Final

La version 1.0 cierra el proyecto con un pipeline reproducible de analisis estadistico-geoespacial. Ademas de la integral numerica, el sistema calcula la primera y segunda derivada de la PDF ajustada. Los puntos de inflexion se aceptan cuando `f''(x)` cambia de signo; el valor cero aislado no basta.

Cada observacion exportada puede incluir `pdf_value`, `pdf_derivative_1`, `pdf_derivative_2`, `distance_to_inflection` e `is_inflection_zone`. Las derivadas se usan como diagnostico e interpretabilidad de la distribucion y no se agregan artificialmente al score de anomalia: el score conserva sus componentes estadisticos, geometricos, espaciales y temporales.

### Inicio en Windows

Haz doble clic en `start.bat`. El lanzador prepara el entorno, inicia FastAPI y abre:

`http://127.0.0.1:8000/ui/`

La interfaz ejecuta el analisis, representa histograma + PDF + `f'(x)` + `f''(x)`, informa los puntos de inflexion y permite descargar los artefactos QGIS/CSV/JSON/reporte.

### Pipeline 1.0

`Append -> Cleaning -> Probability Fit -> Gaussian Quadrature -> Derivatives/Inflection -> Mahalanobis -> SPD -> Procrustes -> Spatial -> Temporal -> Explainable Consensus -> 90/95/99 -> Noise validation -> Export/Report`


## Apend Detection 1.1

La interfaz localhost incorpora mapa Leaflet, inspeccion previa de Shapefiles, selector de variable, comparacion multivariable, graficos separados de PDF/f'/f'', areas centrales, tabla explicable de anomalias, pesos configurables, jobs con progreso, historial SQLite, bootstrap de estabilidad, ablation sensitivity, validacion de ZIP/CRS/tamano, reporte HTML y metadata reproducible con SHA-256.

Los jobs se persisten localmente en `runtime/apend_detection.sqlite3`. Los resultados de cada corrida permanecen en `runtime/jobs/<job_id>/results`.

Endpoints principales: `/inspect`, `/analyze`, `/jobs/{id}/progress`, `/jobs/{id}/result`, `/history`, `/health`.
