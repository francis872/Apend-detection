# Meridian Core
Motor computacional reusable para Meridian, SINCRONIA y Backstage.

## Engines
- Spatial: encoding esferico, distancia geodesica y grafo de proximidad.
- Baseline: media, volatilidad, delta relativo y z-score.
- Corridor: ranking de conexiones espaciales.
- Risk: riesgo compuesto configurable; los pesos son parametros, no verdad empirica.
- Forecasting: baseline reproducible por regresion lineal con RMSE/confianza.
- GWO: Grey Wolf Optimizer con Alpha/Beta/Delta, limites, semilla y curva de convergencia.

Los motores no fabrican datos territoriales. Reciben observaciones reales de cada producto consumidor.
