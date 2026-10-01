# PROJECT SITE CONTEXT V1

Fecha: 2026-10-01

## Objetivo

Incorporar al primer llamado semantico de QuantiaSpatial la informacion ya declarada
por el usuario en interfaces anteriores, sin convertirla en verdad geometrica.

## Fuente

El contrato acepta directamente los nombres de `datosGeneralesObra` usados por
QuantiaV2L, entre ellos:

- `anchoTerrenoM`
- `largoTerrenoM`
- `areaTerrenoM2`
- `areaConstruccionPropuestaM2`
- `areaConstruccionM2`
- `niveles`
- alturas por nivel
- sistema estructural
- tipo de cimentacion
- condiciones especiales

## Regla de verdad

Se conservan tres fuentes separadas:

1. USER_DECLARED: datos introducidos por el usuario.
2. DOCUMENT_OBSERVED: datos leidos del plano por Gemini/OCR/PyMuPDF.
3. SPATIAL_DERIVED: geometria calculada posteriormente por QuantiaSpatial.

Ninguna fuente sustituye automaticamente a otra. Coincidencias refuerzan confianza;
contradicciones pasan a REVIEW/conflicto.

## Distinciones obligatorias

- PREDIO / PROPERTY BOUNDARY
- FOOTPRINT por nivel
- areas exteriores dentro del predio
- contexto exterior fuera del predio
- proyecciones de niveles superiores

El area del terreno no se usa como igualdad obligatoria para PB o PA. Balcones,
marquesinas y voladizos pueden producir footprints diferentes entre niveles.

## Alcance V1

V1 agrega el contrato, la inyeccion controlada al primer prompt y la persistencia del
contexto en el resultado del engine. No modifica F02, CandidateGraph, WallGraph,
Call 2, router ni endpoints legacy.
