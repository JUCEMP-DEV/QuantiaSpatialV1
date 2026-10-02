from __future__ import annotations

import json

from app.quantia_spatialV1.core.models.project_site_context import ProjectSiteContext


def build_project_site_context_prompt(
    *,
    base_prompt: str,
    project_site_context: ProjectSiteContext | None,
) -> str:
    """
    Anexa contexto declarado por usuario al primer llamado Gemini.

    Si no existe contexto util, devuelve el prompt base byte-for-byte para
    conservar equivalencia de cache/replay.
    """

    if project_site_context is None or not project_site_context.has_meaningful_data:
        return base_prompt

    payload = json.dumps(
        project_site_context.as_prompt_payload(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )

    return (
        base_prompt.rstrip()
        + "\n\n"
        + """
CONTEXTO DECLARADO POR EL USUARIO — EVIDENCIA INDEPENDIENTE

El siguiente objeto proviene de interfaces anteriores de Quantia.
NO es verdad geometrica confirmada y NO sustituye lo visible en el plano.

REGLAS OBLIGATORIAS:
1. Usalo como evidencia independiente para comparar contra el documento.
2. No cambies una observacion visible solo para hacerla coincidir con estos datos.
3. Si documento y usuario coinciden, conserva la coincidencia como soporte.
4. Si existe contradiccion material, conserva ambas evidencias y reporta el conflicto
   usando los campos ya existentes del contrato de salida.
5. ancho/largo/area del terreno describen el PREDIO, no el footprint construido.
6. No asumas que area del terreno = area de Planta Baja = area de Planta Alta.
7. Un nivel superior puede contener balcon, voladizo o marquesina que exceda el
   footprint del nivel inferior e incluso proyectarse hacia el exterior del predio.
   En tal caso no lo conviertas automaticamente en error.
8. Distingue cuando sea visible:
   - limite de propiedad/predio;
   - footprint construido por nivel;
   - area exterior dentro del predio: jardin, patio, cochera, acceso, retiro;
   - contexto exterior fuera del predio: banqueta, calle, vecino;
   - proyecciones: balcon, voladizo, marquesina.
9. No deduzcas el limite legal del predio solamente a partir de ejes o del perimetro
   construido.
10. No inventes campos nuevos fuera del schema solicitado. Expresa la informacion
    mediante los campos existentes y usa conflictos/datos_no_identificados/
    confirmaciones_requeridas cuando corresponda.

DATOS DECLARADOS:
""".strip()
        + "\n"
        + payload
    )
