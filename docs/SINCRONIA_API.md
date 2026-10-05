# Meridian API for Sincronia

This integration keeps Meridian as an independent analytical engine. Sincronia is a client of Meridian over HTTP; Meridian is not embedded inside the Sincronia application.

## Contract

Base namespace:

```
/v1/integrations/sincronia
```

### GET /capabilities

Returns the Meridian capabilities explicitly exposed to Sincronia.

### POST /property/context/{job_id}

Request:

```json
{
  "external_id": "SIN-PROP-001",
  "latitude": 6.244,
  "longitude": -75.5812,
  "radius_m": 1000,
  "metadata": {
    "property_type": "apartment"
  }
}
```

The `job_id` references a completed Meridian analysis. Meridian selects the evidence inside the requested radius and returns the territorial summary and available signals.

### POST /properties/compare/{job_id}

Request:

```json
{
  "radius_m": 1000,
  "properties": [
    {"external_id":"A","latitude":6.244,"longitude":-75.5812},
    {"external_id":"B","latitude":6.250,"longitude":-75.5900}
  ]
}
```

The comparison ranks only metrics present in Meridian evidence. It does not convert anomaly/risk-like signals into an investment recommendation.

### POST /accessibility

Request:

```json
{
  "graph": {
    "nodes": [{"id":"property"},{"id":"hospital"}],
    "edges": [{"source":"property","target":"hospital","weight":1200}]
  },
  "source_node": "property",
  "targets": [
    {"node_id":"hospital","category":"health","name":"Hospital"}
  ]
}
```

The returned distance is the graph edge weight. A travel-time interpretation requires a graph whose edge weights are calibrated as time.

## Integration rule

Sincronia owns property, user, inventory, commercial and transaction data. Meridian owns geospatial/territorial computation. Sincronia should persist the Meridian response together with `job_id`, Meridian version and timestamp so the analysis remains traceable.
