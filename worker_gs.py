#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════
# render-gs-worker: de fotos → malla por 2DGS (Gaussian Splatting)
# ════════════════════════════════════════════════════════════════════════
# Lo corre el pod de RunPod (lo lanza el backend Render-GS automáticamente).
# Hace TODO el camino y le reporta al backend por callbacks firmados (HMAC):
#   1. Descarga el ZIP de fotos (INPUT_URL, ya firmada — sin credenciales).
#   2. COLMAP → genera las poses de cámara.
#   3. Entrena 2DGS sobre fotos + poses.
#   4. Extrae la malla por TSDF.
#   5. Sube el .ply a UPLOAD_URL_PLY (ya firmada) y avisa "completed".
#
# Manda "progress" cada poco (heartbeat) para que el watchdog NO lo mate.
# Si algo falla, manda "error" con el log para poder revisarlo en la página.
# ════════════════════════════════════════════════════════════════════════

import os, sys, zipfile, subprocess, shutil, time, json, hmac, hashlib, threading
from pathlib import Path
import urllib.request

# ── Variables que manda el backend (NO credenciales: URLs ya firmadas) ──
TOUR_ID         = os.environ.get("TOUR_ID", "test")
INPUT_URL       = os.environ.get("INPUT_URL", "")          # descarga del ZIP
UPLOAD_URL_PLY  = os.environ.get("UPLOAD_URL_PLY", "")     # subida del .ply
CALLBACK_URL    = os.environ.get("CALLBACK_URL", "")       # a dónde reportar
CALLBACK_SECRET = os.environ.get("CALLBACK_SECRET", "")    # para firmar HMAC
QUALITY         = os.environ.get("QUALITY", "fast")

# Iteraciones de 2DGS según calidad (rápido para la primera prueba).
ITERS = {"fast": 7000, "balanced": 30000, "quality": 30000}.get(QUALITY, 7000)

WORK = Path("/workspace/job")
WORK.mkdir(parents=True, exist_ok=True)

# Buffer del log completo (se manda al backend en cada heartbeat y al final).
_LOG = []
def log(msg):
    linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linea, flush=True)
    _LOG.append(linea)

def _firmar(body: bytes) -> str:
    return hmac.new(CALLBACK_SECRET.encode(), body, hashlib.sha256).hexdigest()

def callback(tipo, **datos):
    """Manda un callback firmado al backend (progress/completed/error)."""
    if not CALLBACK_URL:
        return
    payload = {"type": tipo, **datos}
    body = json.dumps(payload).encode()
    try:
        req = urllib.request.Request(
            CALLBACK_URL, data=body,
            headers={"Content-Type": "application/json",
                     "X-Signature": _firmar(body)},
            method="POST")
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:
        print(f"[callback] error enviando {tipo}: {e}", flush=True)

def progreso(p, msg):
    """Reporta avance + el log hasta ahora (heartbeat para el watchdog)."""
    callback("progress", progress=p, message=msg, log="\n".join(_LOG))

# ── Heartbeat en hilo aparte: late aunque COLMAP/2DGS bloqueen el proceso ──
_estado = {"p": 0.0, "msg": "iniciando", "vivo": True}
def _latido():
    while _estado["vivo"]:
        progreso(_estado["p"], _estado["msg"])
        time.sleep(30)
def fase(p, msg):
    _estado["p"] = p; _estado["msg"] = msg
    log(msg)

def run(cmd, cwd=None):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    # Capturamos salida para que, si falla, el log muestre el motivo real
    # (antes solo decía el código de error, sin el mensaje de COLMAP).
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    # Mostrar las últimas líneas de salida (útiles para diagnóstico).
    cola = (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:]
    if cola.strip():
        for linea in cola.strip().splitlines()[-15:]:
            log(f"   | {linea}")
    if r.returncode != 0:
        raise RuntimeError(f"Falló (código {r.returncode}): {cmd[0]} {cmd[1] if len(cmd)>1 else ''}")


def main():
    t0 = time.time()
    hb = threading.Thread(target=_latido, daemon=True); hb.start()
    try:
        log(f"═══ render-gs-worker 2DGS · job {TOUR_ID} · calidad {QUALITY} ({ITERS} iter) ═══")

        # ── PASO 1: descargar y descomprimir fotos ──
        fase(0.05, "PASO 1/5 — Descargando fotos")
        zip_local = WORK / "input.zip"
        urllib.request.urlretrieve(INPUT_URL, zip_local)
        log(f"   ZIP: {zip_local.stat().st_size/1e6:.1f} MB")
        raw = WORK / "raw"; raw.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_local, "r") as z:
            z.extractall(raw)
        imgs = (list(raw.rglob("*.jpg")) + list(raw.rglob("*.JPG")) +
                list(raw.rglob("*.png")) + list(raw.rglob("*.jpeg")) +
                list(raw.rglob("*.PNG")))
        if not imgs:
            raise RuntimeError("No se encontraron imágenes en el ZIP")
        images_dir = WORK / "images"; images_dir.mkdir(exist_ok=True)
        for i, img in enumerate(sorted(imgs)):
            shutil.copy(img, images_dir / f"foto_{i:04d}{img.suffix.lower()}")
        n_fotos = len(imgs)
        log(f"   {n_fotos} fotos listas")

        # ── PASO 2: COLMAP (poses) ──
        fase(0.15, "PASO 2/5 — COLMAP (poses de cámara)")
        colmap_dir = WORK / "colmap"; colmap_dir.mkdir(exist_ok=True)
        db = colmap_dir / "database.db"
        sparse = colmap_dir / "sparse"; sparse.mkdir(exist_ok=True)
        run(["colmap", "feature_extractor",
             "--database_path", str(db), "--image_path", str(images_dir),
             "--ImageReader.single_camera", "1", "--SiftExtraction.use_gpu", "0"])
        fase(0.25, "PASO 2/5 — COLMAP emparejando fotos")
        run(["colmap", "exhaustive_matcher",
             "--database_path", str(db), "--SiftMatching.use_gpu", "0"])
        fase(0.35, "PASO 2/5 — COLMAP reconstruyendo (mapper)")
        run(["colmap", "mapper",
             "--database_path", str(db), "--image_path", str(images_dir),
             "--output_path", str(sparse)])
        modelo = sparse / "0"
        if not modelo.exists():
            raise RuntimeError("COLMAP no produjo reconstrucción (sparse/0). "
                               "Posible: fotos con poco solape o borrosas.")
        log("   COLMAP OK")

        # Estructura que 2DGS espera: dataset/images + dataset/sparse/0
        dataset = WORK / "dataset"; dataset.mkdir(exist_ok=True)
        for sub in ("images", "sparse"):
            if (dataset / sub).exists():
                shutil.rmtree(dataset / sub)
        shutil.copytree(images_dir, dataset / "images")
        shutil.copytree(sparse, dataset / "sparse")

        # ── PASO 3: entrenar 2DGS ──
        fase(0.45, f"PASO 3/5 — Entrenando 2DGS ({ITERS} iter)")
        dgs_out = WORK / "output"; dgs_out.mkdir(exist_ok=True)
        run(["python", "/opt/2dgs/train.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--iterations", str(ITERS), "--quiet"])
        log("   2DGS entrenado")

        # ── PASO 4: extraer malla por TSDF ──
        fase(0.80, "PASO 4/5 — Extrayendo malla (TSDF)")
        run(["python", "/opt/2dgs/render.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--skip_train", "--skip_test", "--mesh_res", "1024"])
        candidatos = list(dgs_out.rglob("*.ply"))
        malla = None
        for c in candidatos:
            if "fuse" in c.name.lower() or "mesh" in c.name.lower():
                malla = c; break
        if malla is None and candidatos:
            malla = max(candidatos, key=lambda p: p.stat().st_size)
        if malla is None:
            raise RuntimeError("No se encontró malla .ply de salida")
        ply_mb = malla.stat().st_size / 1e6
        log(f"   malla: {malla.name} ({ply_mb:.1f} MB)")

        # ── PASO 5: subir el .ply ──
        fase(0.92, "PASO 5/5 — Subiendo malla")
        with open(malla, "rb") as f:
            req = urllib.request.Request(UPLOAD_URL_PLY, data=f.read(),
                                         method="PUT")
            urllib.request.urlopen(req, timeout=300).read()
        log("   malla subida")

        # ── Listo ──
        _estado["vivo"] = False
        seconds = time.time() - t0
        log(f"═══ LISTO en {seconds/60:.1f} min ═══")
        callback("completed", frames_used=n_fotos, ply_mb=round(ply_mb, 1),
                 seconds=round(seconds), log="\n".join(_LOG))

    except Exception as e:
        _estado["vivo"] = False
        log(f"✗ ERROR: {e}")
        callback("error", error_message=str(e), log="\n".join(_LOG))
        sys.exit(1)


if __name__ == "__main__":
    main()
