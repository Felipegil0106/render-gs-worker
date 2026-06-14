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

import os, sys, zipfile, subprocess, shutil, time, json, hmac, hashlib, threading, struct
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
# 2DGS aplica el regularizador de "normales" (el que une superficies) A PARTIR
# de la iteración 7000. Con 7000 justas, NUNCA actúa (por eso normal=0.00000 y
# la malla salía mal). Subimos fast a 15000 para que ese regularizador trabaje
# ~8000 iteraciones y la geometría quede sólida y conectada.
ITERS = {"fast": 15000, "balanced": 30000, "quality": 30000}.get(QUALITY, 15000)

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
             "--ImageReader.single_camera", "1",
             "--SiftExtraction.use_gpu", "0",
             # Limitar tamaño de imagen evita picos de RAM; 1600px es de sobra
             # para fotos de 1440x1920 sin perder calidad de poses.
             "--SiftExtraction.max_image_size", "1600",
             "--SiftExtraction.max_num_features", "8192",
             # CLAVE del error -9 (OOM): COLMAP-CPU abre 1 hilo por núcleo y
             # cada uno carga una foto en RAM. En máquinas con muchos vCPU
             # (como la A6000) eso revienta la memoria. Con 4 hilos, máximo
             # 4 fotos en RAM a la vez → estable en cualquier GPU.
             "--SiftExtraction.num_threads", "4"])
        fase(0.25, "PASO 2/5 — COLMAP emparejando fotos")
        run(["colmap", "exhaustive_matcher",
             "--database_path", str(db), "--SiftMatching.use_gpu", "0",
             # Mismo motivo que arriba: limitar hilos evita el OOM (-9).
             "--SiftMatching.num_threads", "4"])
        fase(0.35, "PASO 2/5 — COLMAP reconstruyendo (mapper)")
        # Mapper en ajustes por defecto: en la prueba anterior ya registró 108
        # de 127 fotos (85%), así que COLMAP no era el problema. No bajamos
        # umbrales para no degradar la calidad de las poses.
        run(["colmap", "mapper",
             "--database_path", str(db), "--image_path", str(images_dir),
             "--output_path", str(sparse)])
        # COLMAP puede crear VARIOS modelos (sparse/0, sparse/1, ...). El worker
        # antes tomaba siempre sparse/0, que podía ser un fragmento pequeño.
        # Ahora elegimos el modelo con MÁS fotos registradas (el más completo).
        modelos = [d for d in sparse.iterdir()
                   if d.is_dir() and (d / "images.bin").exists()]
        if not modelos:
            raise RuntimeError("COLMAP no produjo reconstrucción. "
                               "Revisa que las fotos tengan solape suficiente.")
        def _num_imgs_registradas(model_dir):
            # Las primeras 8 bytes de images.bin = nº de imágenes registradas.
            try:
                with open(model_dir / "images.bin", "rb") as f:
                    return struct.unpack("<Q", f.read(8))[0]
            except Exception:
                return 0
        modelo = max(modelos, key=_num_imgs_registradas)
        registradas = _num_imgs_registradas(modelo)
        log(f"   COLMAP produjo {len(modelos)} modelo(s). Uso el más completo "
            f"({modelo.name}): {registradas} de {n_fotos} fotos registradas")
        if registradas < n_fotos * 0.5:
            log(f"   ⚠ OJO: solo se registró {registradas}/{n_fotos} fotos. "
                f"La malla puede salir incompleta (poco solape entre fotos).")
        log("   COLMAP OK")

        # ── Quitar distorsión y convertir a PINHOLE (lo que 2DGS exige) ──
        # COLMAP usa por defecto SIMPLE_RADIAL (con distorsión de lente), pero
        # 2DGS solo acepta PINHOLE. 'image_undistorter' corrige las fotos y
        # convierte el modelo a PINHOLE. Es el paso estándar de Gaussian Splatting.
        fase(0.40, "PASO 2/5 — Quitando distorsión (undistort → PINHOLE)")
        dataset = WORK / "dataset"
        if dataset.exists():
            shutil.rmtree(dataset)
        dataset.mkdir(exist_ok=True)
        run(["colmap", "image_undistorter",
             "--image_path", str(images_dir),
             "--input_path", str(modelo),          # sparse/0 con SIMPLE_RADIAL
             "--output_path", str(dataset),         # genera images/ + sparse/ PINHOLE
             "--output_type", "COLMAP"])
        # image_undistorter deja sparse/*.bin sueltos; 2DGS los quiere en sparse/0/.
        sparse_out = dataset / "sparse"
        sparse_0 = sparse_out / "0"
        if not sparse_0.exists():
            sparse_0.mkdir(parents=True, exist_ok=True)
            for f in list(sparse_out.iterdir()):
                if f.is_file():
                    shutil.move(str(f), str(sparse_0 / f.name))
        log("   undistort OK (cámaras PINHOLE, fotos corregidas)")

        # ── PASO 3: entrenar 2DGS ──
        fase(0.45, f"PASO 3/5 — Entrenando 2DGS ({ITERS} iter)")
        dgs_out = WORK / "output"; dgs_out.mkdir(exist_ok=True)
        # NOTA: probamos --lambda_dist 100 y dejó la malla VACÍA (demasiado
        # agresivo). Lo quitamos. Ahora la unión de superficies la logra el
        # regularizador de NORMALES, que sí actúa gracias a las 15000 iteraciones.
        run(["python", "/opt/2dgs/train.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--iterations", str(ITERS),
             "--quiet"])
        log("   2DGS entrenado")

        # ── PASO 4: extraer malla por TSDF ──
        fase(0.80, "PASO 4/5 — Extrayendo malla (TSDF)")
        # --num_cluster 1: se queda con el pedazo conectado MÁS GRANDE (el cuarto)
        # y descarta trozos flotantes sueltos. Además evita el crash que daba el
        # valor por defecto (50) cuando la malla tiene menos de 50 pedazos.
        # Lo corremos de forma TOLERANTE: si el post-proceso fallara pero ya se
        # generó una malla en disco, la usamos igual (no abortamos el trabajo).
        log("$ python /opt/2dgs/render.py ... --num_cluster 1")
        r_mesh = subprocess.run(
            ["python", "/opt/2dgs/render.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--skip_train", "--skip_test", "--mesh_res", "1024",
             "--num_cluster", "1"],
            capture_output=True, text=True)
        cola = (r_mesh.stdout or "")[-1200:] + (r_mesh.stderr or "")[-1200:]
        if cola.strip():
            for linea in cola.strip().splitlines()[-12:]:
                log(f"   | {linea}")
        # Buscar la malla generada (preferimos la post-procesada; si no, la cruda).
        candidatos = list(dgs_out.rglob("*.ply"))
        def _es_no_vacia(p):
            try:
                return p.stat().st_size > 1000   # >1KB = tiene geometría real
            except Exception:
                return False
        malla = None
        # 1º: fuse_post.ply (limpia). 2º: fuse.ply (cruda). 3º: la más grande.
        for clave in ("fuse_post", "fuse", "mesh"):
            for c in candidatos:
                if clave in c.name.lower() and _es_no_vacia(c):
                    malla = c; break
            if malla:
                break
        if malla is None:
            no_vacias = [c for c in candidatos if _es_no_vacia(c)]
            if no_vacias:
                malla = max(no_vacias, key=lambda p: p.stat().st_size)
        if malla is None:
            # No hubo malla con geometría → ahí sí es error real.
            raise RuntimeError(
                f"La malla salió vacía. Código render={r_mesh.returncode}. "
                f"Posible: pocas iteraciones o poses débiles.")
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
