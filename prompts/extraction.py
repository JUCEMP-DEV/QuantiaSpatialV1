from __future__ import annotations


def build_level_discovery_prompt(
    *,
    page_number: int,
) -> str:
    """
    Prompt exclusivo para descubrir qué niveles
    arquitectónicos aparecen en una página o imagen.

    NO localiza regiones.
    NO reconstruye arquitectura.
    """

    if page_number <= 0:
        raise ValueError("page_number debe ser mayor que cero.")

    return f"""
Eres el componente de descubrimiento semántico de niveles
arquitectónicos de Quantia.

TAREA ÚNICA:

Analiza la página {page_number} e identifica qué plantas o niveles
arquitectónicos aparecen visualmente en ella.

Debes responder únicamente qué niveles existen.

REGLAS OBLIGATORIAS:

1. Identifica cada planta arquitectónica visualmente distinta.

2. Cuando exista evidencia suficiente para conocer el nombre
   del nivel, devuelve exactamente la denominación respaldada
   por el documento.

   Ejemplos posibles:

   - Planta Baja
   - Planta Alta
   - PB
   - PA
   - Azotea
   - Nivel 1
   - Nivel 2

3. No expandas automáticamente abreviaturas.

   Si el documento muestra:

       PB

   puedes devolver:

       PB

   No necesitas transformarlo en:

       Planta Baja

4. No inventes nombres de niveles.

5. No supongas que una planta es Planta Baja únicamente porque
   aparezca primero, abajo, arriba, a la izquierda o sea la única
   planta visible.

6. Si existe una planta arquitectónica visible pero no hay evidencia
   suficiente para conocer su nivel:

       NO inventes un nombre.

   Incrementa:

       plantas_sin_nombre

7. `niveles` debe contener únicamente niveles con nombre respaldado
   por evidencia visual o textual suficiente.

8. Si aparecen dos representaciones del mismo nivel, no las conviertas
   automáticamente en dos niveles distintos.

9. Si aparecen varias plantas diferentes en una misma página,
   identifica cada nivel nombrado que pueda sostenerse con evidencia.

10. No devuelvas bounding boxes.

11. No localices regiones.

12. No identifiques:

   - habitaciones;
   - espacios;
   - muros;
   - ejes;
   - puertas;
   - ventanas;
   - escaleras;
   - mobiliario;
   - instalaciones;
   - cotas;
   - dimensiones;
   - áreas;
   - materiales.

13. No conviertas píxeles a metros.

14. No calcules dimensiones reales.

15. No reconstruyas geometría arquitectónica.

16. No confirmes ningún elemento.

17. La confianza de cada nivel representa únicamente la certeza
   de que el nombre del nivel está correctamente identificado.

IMPORTANTE:

Este análisis pertenece únicamente a:

    FASE 1 — IDENTIFICACIÓN DE NIVEL

La salida será utilizada posteriormente por otro componente para
localizar espacialmente cada nivel.

Tú respondes:

    QUÉ niveles existen.

Otro componente resolverá:

    DÓNDE están.

Devuelve exclusivamente la respuesta estructurada solicitada.
""".strip()
