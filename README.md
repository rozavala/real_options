# Real Options

## Prediction API

Para empezar necesitas `uv`, [homepage](https://github.com/astral-sh/uv), instálalo con `brew install uv` en mac, o como quieras en otras plataformas. (TODO: meterlo en un container de Docker pa no depender de ambiente local).

Para echarlo a andar, `uv run prediction_api.py` y va a escuchar en el puerto 8000
También se puede echar a andar con `uv run uvicon --port 12345 prediction_api:app` para el puerto 12345, y se puede agregar el flag `--reload` para hacer hot-reloading cuando cambias el código

Ahi expone esta api:

#### `POST /predictions`
Encola un nuevo job de predicción, recibe un JSON con datos (se puede editar el modelo `PredictionJobCreateRequest` en `prediction_api.py` para modificar los campos), y retorna el estado del job pendiente, incluyendo su id

```
> curl --json '{"data":"date,price\n2025-01-01,100\n2025-01-02,101"}' localhost:8000/predictions
{"id":"c32de7e5-35eb-44d1-a86c-41df0ca5b873","status":"pending","result":null,"error":null}
```

#### `GET /predictions/{id}`
Obtiene el estado de un job de predicción con el ID dado.

```
> curl localhost:8000/predictions/c32de7e5-35eb-44d1-a86c-41df0ca5b873
{"id":"a3ce44a7-ae79-4255-b5a1-0bbf188c23b1","status":"completed","result":{"predicted_price":100.0},"error":null}
```
