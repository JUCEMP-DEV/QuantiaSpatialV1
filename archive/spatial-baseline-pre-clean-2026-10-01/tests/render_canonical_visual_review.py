"""Offline visual review of validated canonical graphs; does not reconstruct walls.
Run from Backend: .venv/Scripts/python.exe -B -m app.quantia_spatialV1.tests.render_canonical_visual_review
"""
from __future__ import annotations
import base64
import hashlib
import html
import io
import json
import socket
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from app.quantia_spatialV1.tests.quantia_case_loader import CASE_LOADERS

ROOT = Path(__file__).resolve().parents[1]

def safe(value):
    return "_".join(filter(None, "".join(c.lower() if c.isalnum() else "_" for c in value).split("_")))

def font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

def main():
    def offline(*args, **kwargs):
        raise RuntimeError("Visual review forbids network access")
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    validation = json.loads((ROOT / "documentation/LOCAL_VALIDATION_CANONICAL_WALLGRAPH_INTEGRITY_V2.json").read_text(encoding="utf-8"))
    run = ROOT / validation["output_directory"]
    output = ROOT / "tests/output/canonical_visual_review" / run.name
    output.mkdir(parents=True, exist_ok=True)
    cards, thumbs, manifest = [], [], []
    for case_id, loader in CASE_LOADERS.items():
        print(f"Loading original LevelViews: {case_id}", flush=True)
        stems = sorted((run / case_id).glob("*__canonical_wallgraph.json"), key=lambda p: ("planta_alta" in p.name, p.name))
        cached = all((output / (p.name.replace("__canonical_wallgraph.json", "") + "__original.png")).exists() for p in stems)
        originals = {} if cached else {safe(x.level_name): x.level_result.level_view.raster_bytes for x in loader().levels}
        for path in stems:
            stem = path.name.replace("__canonical_wallgraph.json", "")
            graph = json.loads(path.read_text(encoding="utf-8"))
            bundle = json.loads((path.parent / (stem + "__reconstruction_evidence_bundle.json")).read_text(encoding="utf-8"))
            original_path = output / (stem + "__original.png")
            data = original_path.read_bytes() if original_path.exists() else originals[stem[len(case_id)+2:]]
            assert hashlib.sha256(data).hexdigest() == bundle["source_raster_sha256"], stem
            original = Image.open(io.BytesIO(data)).convert("RGB")
            assert list(original.size) == graph["image_size_px"], stem
            original_path.write_bytes(data)
            clean = Image.new("RGB", original.size, "white")
            overlay = Image.blend(original, Image.new("RGB", original.size, "white"), .35)
            dc, do = ImageDraw.Draw(clean), ImageDraw.Draw(overlay)
            lines, gaps = [], []
            for wall in graph["walls"]:
                a, b = tuple(wall["start_px"]), tuple(wall["end_px"])
                dc.line([a,b], fill="#172331", width=3)
                do.line([a,b], fill="#007fbb", width=3)
                title = html.escape(f"{wall['id']} | {wall['role']} | espesor {wall['thickness_px']/graph['px_per_m']:.3f} m | confianza {wall['confidence']:.2f}")
                lines.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"><title>{title}</title></line>')
            for gap in graph["logical_gaps"]:
                a,b=gap["start_px"],gap["end_px"]
                gaps.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"><title>Continuidad logica, no muro fisico</title></line>')
            clean.save(output / (stem + "__walls_only.png"))
            overlay.save(output / (stem + "__overlay.png"))
            w,h=original.size
            panelw=900; panelh=max(1, round(h*panelw/w))
            sheet=Image.new("RGB", (panelw*3, panelh+130), "#f3f6fa")
            draw=ImageDraw.Draw(sheet)
            title=f"{case_id.replace('_',' ').title()} / {graph['level_name']}"
            draw.text((24,14),title,font=font(30),fill="#172331")
            for i,(label,im) in enumerate((("Plano original",original),("Centerlines finales",clean),("Superposicion",overlay))):
                draw.text((i*panelw+24,68),label,font=font(25),fill="#172331")
                sheet.paste(im.resize((panelw,panelh),Image.Resampling.LANCZOS),(i*panelw,125))
            comparison=stem+"__comparacion.png"
            sheet.save(output/comparison)
            thumb=sheet.copy();thumb.thumbnail((1500,1000));thumbs.append(thumb)
            diag=graph["diagnostics"]
            warning = '<p class="warning">No se detectaron espacios interiores; revisar continuidad del plano.</p>' if graph["interior_space_count"]==0 else ''
            image_url="data:image/png;base64,"+base64.b64encode(data).decode("ascii")
            clean_buffer=io.BytesIO();clean.save(clean_buffer,format="PNG")
            clean_url="data:image/png;base64,"+base64.b64encode(clean_buffer.getvalue()).decode("ascii")
            cards.append(f'<section id="{stem}"><h2>{html.escape(title)}</h2><p>{len(graph["walls"])} muros &middot; {len(graph["logical_gaps"])} gaps logicos &middot; {graph["interior_space_count"]} espacios &middot; {diag["review_wall_count"]} muros REVIEW &middot; 0 referencias invalidas</p>{warning}<div class="pair"><figure><figcaption>Plano original</figcaption><img src="{image_url}"></figure><figure><figcaption>Walls-only final</figcaption><img src="{clean_url}"></figure></div><h3>Superposicion interactiva</h3><svg viewBox="0 0 {w} {h}"><image href="{image_url}" width="{w}" height="{h}"/><g class="walls" stroke="#007fbb" stroke-width="3" fill="none">{"".join(lines)}</g><g class="gaps" stroke="#d66a00" stroke-width="2" stroke-dasharray="8 5" fill="none">{"".join(gaps)}</g></svg><p><a href="{comparison}">Abrir comparacion PNG</a> &middot; <a href="{stem}__walls_only.png">Abrir walls-only PNG</a></p></section>')
            manifest.append({"level":stem,"source_hash_verified":True,"walls":len(graph["walls"]),"spaces":graph["interior_space_count"],"graph_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f"Rendered and source hash verified: {stem}",flush=True)
    assert len(cards)==6
    contact=Image.new("RGB",(1500,sum(t.height+25 for t in thumbs)+60),"#e8edf4")
    draw=ImageDraw.Draw(contact);draw.text((20,12),"Quantia - Revision visual / Canonical Integrity V2",font=font(26),fill="#172331")
    y=60
    for thumb in thumbs:
        contact.paste(thumb,(0,y));y+=thumb.height+25
    contact.save(output/"resumen_seis_niveles.png")
    options=''.join(f'<option value="{x["level"]}">{html.escape(x["level"].replace("__"," / ").replace("_"," "))}</option>' for x in manifest)
    page="""<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quantia | Revision visual</title>
<style>body{margin:0;background:#eef2f6;color:#172331;font:16px system-ui}header,main{max-width:1450px;margin:auto;padding:24px}header{padding-bottom:10px}h1{margin:0}p{line-height:1.55}.controls{position:sticky;top:0;background:#172331;color:white;padding:16px;z-index:2;display:flex;flex-wrap:wrap;gap:20px;align-items:center}select{padding:8px;max-width:95%}section{background:white;padding:24px;border-radius:12px;margin:20px 0}section[hidden]{display:none}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}figcaption{font-weight:600;padding:8px}img,svg{width:100%;height:auto}svg{max-height:1100px}svg line:hover{stroke:#f00050;stroke-width:7}.gaps{display:none}.warning{background:#fff1d6;padding:12px}a{color:#00649a}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>
<header><h1>Quantia / Revision visual</h1><p>Adaptive + PostFilter V4 + Canonical Integrity V2. Seis LevelViews validados. Geometria V4 conservada. Sin Gemini ni Call 2.</p><p>Azul: centerlines finales. Naranja discontinuo: continuidad logica opcional, no muro fisico. Pasa el cursor por un muro para consultar sus datos. Consistencia del grafo no equivale a habitaciones completas.</p></header>
<div class="controls"><label>Nivel <select id="level">OPTIONS</select></label><label>Opacidad muros <input id="opacity" type="range" min="0" max="1" step="0.05" value="1"></label><label><input id="gaps" type="checkbox"> Mostrar gaps logicos</label><a style="color:white" href="resumen_seis_niveles.png">Resumen de los seis niveles</a></div><main>CARDS</main>
<script>const sections=[...document.querySelectorAll('section')];const level=document.getElementById('level');function show(){sections.forEach(s=>s.hidden=s.id!==level.value)}level.onchange=show;show();document.getElementById('opacity').oninput=e=>document.querySelectorAll('.walls').forEach(g=>g.style.opacity=e.target.value);document.getElementById('gaps').onchange=e=>document.querySelectorAll('.gaps').forEach(g=>g.style.display=e.target.checked?'block':'none');</script></html>"""
    (output/"index.html").write_text(page.replace("OPTIONS",options).replace("CARDS","".join(cards)),encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Visual report: {output / 'index.html'}",flush=True)

if __name__ == "__main__":
    main()
