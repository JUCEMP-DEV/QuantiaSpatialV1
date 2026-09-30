# Call 2 e Interfaz 04: revisión del estado real

Fecha: 2026-09-28. Reporte contrastado: `REPORTE_CALL2_PARAMETRIZACION_INTERFAZ04_2026-09-28.md` de Downloads, SHA256 `c796764482f4acb79b85eec8054cdde595d0b8fdfa1841d6ff948d83eb7b6182`. El original se conserva.

## Dictamen

El reporte describe correctamente la separación entre reconstrucción canónica y parametrización pendiente. No demuestra una integración terminada con Interfaz 04 ni un resultado semántico perfecto. El contrato estructural no prueba por sí solo que cada muro físico esté representado exactamente una vez.

La referencia continúa siendo Adaptive + PostFilter V4 / walls-only, seguida del finalizador canónico V2. Esta tarea no modifica F03, Mask, Topology ni el filtro congelado.

## APIs y selección

- `providers/call2_providers.py`: adaptadores Groq y Mistral, resolución de credenciales desde entorno/archivos de configuración y fábrica `build_call2_provider`.
- `providers/vision.py`: proveedor Gemini de spatialV1; `app/services` contiene proveedores de visión usados por otras capas.
- La fábrica usa `QUANTIA_CALL2_PROVIDER` y, en ausencia de selección, Groq. El modelo de Groq observado es `qwen/qwen3.8-27b`.
- Atención: construir `WallGraphMultimodalReviewer()` directamente usa Gemini. El ejecutor nuevo inyecta explícitamente el proveedor de la fábrica para evitar esa diferencia.
- No se copian ni imprimen credenciales. No hay cambio automático de proveedor. Los historiales nuevos están separados por proveedor y hash del nombre del modelo.

## Diferencias relevantes para Interfaz 04

`04DisenoViviendaView.vue` consume `estructuraEspacial`; `QuantiaPlanEditor.vue` dibuja muros en modo plano desde `muros[].segmentos[].geometria.raster.vertices`. Un DTO con únicamente coordenadas métricas no basta.

El paquete experimental entrega ambos sistemas: raster local y metros locales, con origen superior izquierdo y eje Y hacia abajo. Incluye escala, documento, página, bbox, hash del raster, IDs, espesor, confianza, fuentes, contexto, topología y linaje de cada muro. La referencia `original.png` es relativa al paquete: una integración web deberá resolverla a una URL autenticada o blob.

Una región de Call 2 es un candidato, no una abertura validada. Se conservan regiones aceptadas, rechazadas y evidencia original; no se inventan ancho, altura, antepecho ni posición sobre un muro. El render actual de aberturas usa start/end y widthM: se debe verificar su coherencia de unidades antes de habilitarlas en modo raster.

El grafo entrega conteo de espacios, no polígonos editables completos. No se convierte ese conteo en habitaciones ficticias. Ejes y cotas semánticos sin posición siguen en la evidencia. La exportación marca explícitamente como pendientes espacios, aberturas, ejes/cotas ubicados y alturas. Por tanto `workflowContinuation=false` y `quantification=false`.

## Prueba añadida

Desde `Backend`:

```powershell
.venv/Scripts/python.exe -B -m pytest app/quantia_spatialV1/tests/test_call2_interface04_delivery.py -q -p no:cacheprovider
.venv/Scripts/python.exe -B -m app.quantia_spatialV1.tests.run_call2_interface04 --offline
```

El ejecutor recupera el snapshot canónico y verifica hashes. Prepara PNG A/B, prompt y schema reales. La modalidad offline usa una respuesta explícitamente sintética sin cambios: no mide calidad del modelo. Después recorre el orquestador productivo, validador, correcciones con linaje y finalizador; únicamente sustituye la reconstrucción previa por el snapshot y la respuesta por la captura.

Salida: `tests/output/call2_interface04/offline/401ffb2c754c/casa_viri__02_planta_alta_copia_1/`. Contiene `interface04.json`, `final_graph.json`, `evidence_bundle.json`, `validation.json`, `integrity.json`, `review.json`, `original.png`, `call2_input.png`, prompt, schema y estado de ejecución.

Validación ejecutada: 3 pruebas nuevas aprobadas y 14 pruebas existentes de validador, adaptadores de schema y contrato canónico aprobadas. El caso offline de Casa Viri Planta Alta exportó correctamente. También se prueba el compositor A/B y la petición multimodal con proveedor sintético y red bloqueada. Se comprueba rechazo de un delta de baja confianza, correspondencia raster/métrica, conservación del linaje, separación de candidatos y rechazo de respuestas de otro nivel.

## Ejecución real y límites

```powershell
.venv/Scripts/python.exe -B -m app.quantia_spatialV1.tests.run_call2_interface04 --live
```

La llamada real prepara un smoke visual y después Call 2 con el proveedor configurado. La consulta previa de disponibilidad de Groq fue satisfactoria; esto no equivale a una inferencia ejecutada.

La revisión automática de permisos rechazó enviar el plano y el grafo a Groq por falta de autorización explícita para ese contenido y destino. La inferencia real queda pendiente de esa autorización; no se presenta la salida offline como respuesta del modelo.

Esta entrega es una prueba reproducible de revisión de muros y de información disponible/faltante para la interfaz. No conecta rutas HTTP ni modifica el store de vivienda. Para cerrar la integración completa aún hacen falta parametrización determinista, polígonos de espacios, validación de hosts y unidades de aberturas, adaptación del nivel seleccionado y una prueba de consumo en la interfaz real.


## Actualizaci?n: prueba real autorizada y ejecutada

El usuario autoriz? el env?o a Groq. Smoke visual aprobado; primer intento de Call 2 recibi? HTTP 429 temporal y el reintento concluy? correctamente. Modelo `qwen/qwen3.8-27b`. Respuesta: dos REMOVE_WALL (PW_041, PW_087), ambos rechazados por confianza 0.8/0.7 inferior a 0.82; dos candidatos STAIR/STAIR_HANDRAIL y una regi?n incierta. Resultado: 28 muros, 30 gaps, cero referencias inv?lidas, integridad v?lida. La frase del modelo que afirma haber eliminado muros no describe lo aplicado por el validador.

Vista visual real: `tests/output/call2_interface04/groq/401ffb2c754c/casa_viri__02_planta_alta_copia_1/index.html`. El paquete de Interfaz 04 est? junto a la vista. Persisten los l?mites de parametrizaci?n indicados anteriormente.
