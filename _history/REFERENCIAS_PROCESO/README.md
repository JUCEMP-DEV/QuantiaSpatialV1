# REFERENCIAS_PROCESO

Esta carpeta contiene material retirado del árbol operativo de Quantia SpatialV1 sin eliminarlo.

## Criterio

- `01_DOCUMENTACION_HISTORICA/`: históricos, versionados anteriores, árboles antiguos y auditorías supersedidas.
- `02_CHECKPOINTS_MANIFIESTOS/`: puntos de retorno, manifests SHA256 y checkpoints.
- `03_PRUEBAS_PROBES_HISTORICOS/`: probes fuera del flujo canónico actual.
- `04_OUTPUTS_HISTORICOS/`: outputs anteriores, providers no canónicos y paquetes duplicados.
- `05_EXPERIMENTOS/`: código explícitamente experimental no importado por el orquestador productivo actual.
- `99_CACHE_GENERADO/`: `__pycache__` y `.pyc`; no son fuente.

## Regla

Nada dentro de esta carpeta debe importarse ni utilizarse como fuente productiva. Para restaurar un archivo, consultar `CLEANUP_MANIFEST_2026-09-30.json`, revisar colisiones y validar antes de copiarlo de regreso.
