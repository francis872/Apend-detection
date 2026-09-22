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
