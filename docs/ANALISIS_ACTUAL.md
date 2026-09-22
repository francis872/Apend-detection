# Analisis estadistico actual

## Objetivo
Detectar ruido y observaciones atipicas en datos geoespaciales sin reducir la deteccion a una sola regla univariada.

## Metodologia
1. **Append:** GeoPandas + normalizacion de CRS + `pandas.concat`.
2. **Limpieza:** geometria nula/invalida, duplicados y valores numericos no utilizables.
3. **Probabilidad:** histograma y 20 distribuciones candidatas comparadas por AIC, BIC, KS, Bhattacharyya y Fisher-Rao.
4. **Geometria SPD:** covarianzas locales regularizadas comparadas con distancia Riemanniana afin-invariante.
5. **Procrustes:** compara forma de vecindarios reduciendo traslacion, rotacion y escala.
6. **Consenso:** combina probabilidad, Mahalanobis robusto, SPD y Procrustes.
7. **Niveles:** 90, 95 y 99 como cuantiles empiricos del score.

## Corrida de referencia
- Total: 62,941 registros.
- Outliers 90%: 6,295.
- Outliers 95%: 3,148.
- Outliers 99%: 630.
- Umbrales: 0.6890, 0.7315 y 0.8197.

El mejor ajuste agregado entre las candidatas fue F, pero el KS detecta discrepancias estadisticamente significativas; por eso la clasificacion final no depende de una sola distribucion.
