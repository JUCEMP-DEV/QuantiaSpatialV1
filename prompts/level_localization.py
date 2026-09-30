from __future__ import annotations


def build_level_localization_prompt(
    *,
    page_number: int,
    expected_level_names: list[str],
) -> str:
    """
    Construye el prompt exclusivo de:

        FASE 1 — IDENTIFICACIÓN DE NIVEL

    Objetivo:
        localizar visualmente niveles ya conocidos
        dentro de una sola página.

    Este prompt NO solicita:
        - espacios;
        - muros;
        - puertas;
        - ventanas;
        - escaleras;
        - cotas;
        - áreas;
        - geometría arquitectónica;
        - medidas métricas, salvo transcribir una escala impresa claramente asociada al nivel.
    """

    if page_number <= 0:
        raise ValueError("page_number debe ser mayor que cero.")

    normalized_names = [
        str(name).strip() for name in expected_level_names if str(name).strip()
    ]

    if not normalized_names:
        raise ValueError("expected_level_names no puede estar vacío.")

    level_lines = "\n".join(f"- {name}" for name in normalized_names)

    return f"""
Eres un sistema de localización visual de plantas arquitectónicas.

TAREA ÚNICA:
Localiza dentro de la página {page_number} exclusivamente los niveles
indicados en la lista de niveles esperados.

NIVELES ESPERADOS:
{level_lines}

REGLAS OBLIGATORIAS:

1. No agregues niveles nuevos.
2. No elimines niveles esperados de la respuesta.
3. No renombres niveles.
4. Devuelve exactamente una entrada por cada nivel esperado.
5. Si un nivel puede localizarse visualmente con suficiente evidencia:
   - localizado = true
   - bbox_normalizado debe contener su región visual.
6. Si un nivel no puede localizarse de forma razonable:
   - localizado = false
   - bbox_normalizado = null
7. Las coordenadas del bbox deben estar normalizadas entre 0.0 y 1.0:
   - x_min
   - y_min
   - x_max
   - y_max
8. El bbox debe abarcar la planta arquitectónica correspondiente,
   no únicamente su título o texto identificador.
9. No incluyas espacios, habitaciones, mobiliario, muros, puertas,
   ventanas, escaleras, cotas ni ningún otro elemento arquitectónico.
10. No calcules medidas reales.
11. No conviertas píxeles a metros.
12. No inventes regiones cuando la evidencia sea insuficiente.
13. Si dos plantas aparecen en una misma página, localiza cada una
    de forma independiente.
14. Si una planta ocupa prácticamente toda la página, puede utilizarse
    un bbox cercano al área total ocupada por esa planta.
15. La confianza debe representar únicamente la certeza de la
    localización visual del nivel.
16. Para cada nivel, si existe una escala IMPRESA claramente asociada a esa
    planta (por ejemplo "1:30" o "Esc. 1:50"), transcríbela literalmente en
    `escala_declarada` y publica `confianza_escala`.
17. Si no existe una escala impresa claramente asociada al nivel, usa
    `escala_declarada = null` y `confianza_escala = null`.
18. No infieras la escala por tamaño aparente, mobiliario, muros, cotas ni
    conocimiento arquitectónico. Esta salida es solo evidencia para un
    reconciliador determinístico posterior.

IMPORTANTE:
Tu salida será utilizada únicamente para separar la página en
LEVEL_VIEW independientes. No estás reconstruyendo arquitectura.

Devuelve exclusivamente la respuesta estructurada solicitada.
""".strip()
