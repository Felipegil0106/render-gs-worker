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

# Iteraciones de 2DGS según calidad.
# 2DGS aplica el regularizador de "normales" (une superficies) a partir de la
# iteración 7000, y el de "distorsión" (aplana paredes) a partir de la 3000. La
# DENSIFICACIÓN (la que crea gaussianas nuevas para cerrar superficies) corre
# hasta la iteración 15000. Con solo 15000, la geometría NO converge: termina
# justo cuando deja de densificar. La investigación confirmó que 15000 es
# insuficiente para interiores. Subimos a 30000 → la geometría converge,
# las superficies se cierran y se aplanan. Duplica el tiempo (~60 min) pero
# es necesario para que el cuarto no salga a medias.
ITERS = {"fast": 30000, "balanced": 30000, "quality": 30000}.get(QUALITY, 30000)

WORK = Path("/workspace/job")
WORK.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# Script que corre MASt3R-SfM y escribe las poses en formato COLMAP (texto)
# que 2DGS lee. Se escribe a disco y se ejecuta como proceso aparte para
# aislar la memoria del modelo de IA. (raw string: el \n de adentro queda
# literal y Python lo interpreta al ejecutar el script.)
# ════════════════════════════════════════════════════════════════════════
MAST3R_SCRIPT = r'''
import sys, os
sys.path.insert(0, "/opt/mast3r")
sys.path.insert(0, "/opt/mast3r/dust3r")
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

IMAGES_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

CKPT = "/opt/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
RETR = "/opt/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
device = "cuda"

exts = (".jpg", ".jpeg", ".png")
filelist = sorted([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)
                   if os.path.splitext(f)[1].lower() in exts])
print("MAST3R: %d fotos" % len(filelist), flush=True)

model = AsymmetricMASt3R.from_pretrained(CKPT).to(device)
print("MAST3R: modelo cargado", flush=True)

imgs = load_images(filelist, size=512, verbose=False)

# retrieval -> matriz de similitud para elegir que pares de fotos comparar
retriever = Retriever(RETR, backbone=model, device=device)
with torch.no_grad():
    sim_matrix = retriever(filelist)
del retriever
torch.cuda.empty_cache()
print("MAST3R: retrieval OK", flush=True)

# retrieval-Na-k : Na anclas (FPS) + k vecinos mas similares por foto
pairs = make_pairs(imgs, scene_graph="retrieval-20-10", prefilter=None,
                   symmetrize=True, sim_mat=sim_matrix)
print("MAST3R: %d pares" % len(pairs), flush=True)

cache_dir = os.path.join(OUT_DIR, "mast3r_cache")
os.makedirs(cache_dir, exist_ok=True)
scene = sparse_global_alignment(filelist, pairs, cache_dir, model,
                                lr1=0.07, niter1=300, lr2=0.014, niter2=300,
                                device=device, opt_depth=True,
                                shared_intrinsics=True, matching_conf_thr=5.0)
print("MAST3R: alineamiento global OK", flush=True)

cams2world = to_numpy(scene.get_im_poses())
intrinsics = [to_numpy(K) for K in scene.intrinsics]
rgbimgs = scene.imgs
N = len(rgbimgs)
print("MAST3R: %d camaras registradas" % N, flush=True)

img_out = os.path.join(OUT_DIR, "images")
sparse_out = os.path.join(OUT_DIR, "sparse", "0")
os.makedirs(img_out, exist_ok=True)
os.makedirs(sparse_out, exist_ok=True)

fcam = open(os.path.join(sparse_out, "cameras.txt"), "w")
fimg = open(os.path.join(sparse_out, "images.txt"), "w")
fcam.write("# Camera list\n")
fimg.write("# Image list\n")
for i in range(N):
    im = rgbimgs[i]
    H, W = im.shape[:2]
    name = "img_%04d.png" % i
    Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(os.path.join(img_out, name))
    K = intrinsics[i]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    cam_id = i + 1
    fcam.write("%d PINHOLE %d %d %.6f %.6f %.6f %.6f\n" % (cam_id, W, H, fx, fy, cx, cy))
    # COLMAP guarda world->cam = inversa de cam->world
    w2c = np.linalg.inv(cams2world[i])
    q = Rotation.from_matrix(w2c[:3, :3]).as_quat()   # [x,y,z,w]
    t = w2c[:3, 3]
    fimg.write("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %d %s\n" %
               (cam_id, float(q[3]), float(q[0]), float(q[1]), float(q[2]),
                float(t[0]), float(t[1]), float(t[2]), cam_id, name))
    fimg.write("\n")   # linea de puntos 2D (vacia)
fcam.close()
fimg.close()
print("MAST3R: poses escritas", flush=True)

# nube de puntos densa con color, para inicializar 2DGS
pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)
pts3d = to_numpy(pts3d)
confs = to_numpy(confs)
masks = [c > 1.5 for c in confs]
pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, masks)]).reshape(-1, 3)
col = np.concatenate([im[m] for im, m in zip(rgbimgs, masks)]).reshape(-1, 3)
valid = np.isfinite(pts.sum(axis=1))
pts = pts[valid]
col = (np.clip(col[valid], 0, 1) * 255).astype(np.uint8)
if len(pts) > 200000:
    idx = np.random.choice(len(pts), 200000, replace=False)
    pts = pts[idx]; col = col[idx]
print("MAST3R: %d puntos 3D para init" % len(pts), flush=True)

fp = open(os.path.join(sparse_out, "points3D.txt"), "w")
fp.write("# 3D point list\n")
for j in range(len(pts)):
    x, y, z = pts[j]
    r, g, b = col[j]
    fp.write("%d %.6f %.6f %.6f %d %d %d 0\n" % (j + 1, x, y, z, int(r), int(g), int(b)))
fp.close()
print("MAST3R: points3D.txt escrito. LISTO.", flush=True)
'''


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
    # Enviamos solo las últimas 150 líneas para que el payload no crezca de más
    # ahora que transmitimos el progreso en vivo.
    callback("progress", progress=p, message=msg, log="\n".join(_LOG[-150:]))

# ── Heartbeat en hilo aparte: late aunque COLMAP/2DGS bloqueen el proceso ──
_estado = {"p": 0.0, "msg": "iniciando", "vivo": True}
def _latido():
    while _estado["vivo"]:
        progreso(_estado["p"], _estado["msg"])
        time.sleep(30)
def fase(p, msg):
    _estado["p"] = p; _estado["msg"] = msg
    log(msg)

def run(cmd, cwd=None, env=None, fase_label=None, check=True):
    """Ejecuta un comando enviando su salida a un ARCHIVO (no a un pipe).
    Esto evita el deadlock que colgaba el proceso (con la GPU en 0%) cuando la
    salida llenaba el buffer del pipe y nadie lo leía hasta el final.
    Mientras corre, actualiza el mensaje del heartbeat con los minutos que lleva,
    para que la página NO se vea congelada. Devuelve (codigo, salida)."""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    out_path = WORK / "_cmd_out.txt"
    with open(out_path, "w") as outf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                                stdout=outf, stderr=subprocess.STDOUT, text=True)
        t0 = time.time()
        ultima = ""
        while proc.poll() is None:
            time.sleep(10)
            if fase_label:
                mins = int((time.time() - t0) / 60)
                _estado["msg"] = f"{fase_label} · {mins} min trabajando…"
            # Leer la ÚLTIMA línea del archivo de salida y mandarla EN VIVO.
            # Así, si un paso (p.ej. el entrenamiento) se cuelga, queda registrado
            # DÓNDE se quedó, y la página lo muestra gracias al heartbeat. Antes
            # la salida solo se veía al terminar el paso (por eso parecía congelado).
            try:
                with open(out_path, errors="ignore") as f:
                    lineas = [l.rstrip() for l in f if l.strip()]
                if lineas and lineas[-1] != ultima:
                    ultima = lineas[-1]
                    log(f"   · {ultima}")
            except Exception:
                pass
    try:
        salida = open(out_path, errors="ignore").read()
    except Exception:
        salida = ""
    # Mostrar las últimas líneas en nuestro log (diagnóstico).
    cola = salida.strip().splitlines()[-15:] if salida.strip() else []
    for linea in cola:
        log(f"   | {linea}")
    if check and proc.returncode != 0:
        raise RuntimeError(f"Falló (código {proc.returncode}): {cmd[0]} "
                           f"{cmd[1] if len(cmd) > 1 else ''}")
    return proc.returncode, salida


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

        # ── PASO 2: POSES CON MASt3R (reemplaza COLMAP+SIFT+GLOMAP) ──
        # MASt3R es un modelo de IA feed-forward que estima la geometria de cada
        # foto SIN detectar "features" (puntos tipo SIFT). Por eso registra casi
        # todas las camaras incluso en paredes blancas lisas, donde SIFT fallaba
        # (solo 55/127, cuarto fantasma doble). Produce camaras PINHOLE
        # directamente, asi que NO hace falta el paso de undistort.
        fase(0.15, "PASO 2/5 — MASt3R (poses con IA)")
        dataset = WORK / "dataset"
        if dataset.exists():
            shutil.rmtree(dataset)
        dataset.mkdir(exist_ok=True)
        # Escribir el script de MASt3R a disco y ejecutarlo como proceso aparte
        # (aisla la memoria del modelo de IA del resto del worker).
        mast3r_py = WORK / "mast3r_sfm.py"
        mast3r_py.write_text(MAST3R_SCRIPT)
        run(["python", str(mast3r_py), str(images_dir), str(dataset)],
            fase_label="PASO 2/5 — MASt3R calculando poses")
        # MASt3R escribe dataset/images/ + dataset/sparse/0/ (cameras/images/points3D.txt).
        sparse_0 = dataset / "sparse" / "0"
        if not (sparse_0 / "images.txt").exists():
            raise RuntimeError("MASt3R no produjo poses (sparse/0/images.txt). "
                               "Revisa el log de MASt3R arriba.")
        # Contar cuantas camaras registro (lineas de imagen en images.txt).
        try:
            lineas = (sparse_0 / "images.txt").read_text().splitlines()
            n_reg = sum(1 for ln in lineas
                        if ln and not ln.startswith("#") and len(ln.split()) >= 10)
            log(f"   MASt3R registró {n_reg} de {n_fotos} fotos")
            if n_reg < n_fotos * 0.8:
                log(f"   ⚠ OJO: {n_reg}/{n_fotos} registradas. Si es bajo, "
                    f"puede ser la captura (poco solape entre fotos).")
        except Exception as e:
            log(f"   (no se pudo contar cámaras: {e})")
        log("   MASt3R OK (cámaras PINHOLE, sin necesidad de undistort)")

        # ── PASO 3: entrenar 2DGS ──
        fase(0.45, f"PASO 3/5 — Entrenando 2DGS ({ITERS} iter)")
        dgs_out = WORK / "output"; dgs_out.mkdir(exist_ok=True)
        # --lambda_dist 100 : ACTIVA el regularizador de DISTORSIÓN de profundidad.
        # Hasta ahora estaba en 0 (el log mostraba "distort=0.00000" siempre) y por
        # eso las PAREDES SALÍAN ONDULADAS y había floaters: sin este término, las
        # gaussianas no colapsan sobre una superficie fina y plana. El paper de 2DGS
        # lo confirma. Valor 100 (el paper usa 100 para escenas 360°/grandes).
        # ANTES vació la malla porque se probó con solo 7000 iteraciones; ahora con
        # 30000 y depth_ratio=0 (por defecto) es seguro. 2DGS lo activa desde la
        # iteración 3000, así que el log debe mostrar "distort=" CON VALOR (no 0).
        # Sin --quiet: la consola del pod muestra el avance iteración a iteración.
        run(["python", "/opt/2dgs/train.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--iterations", str(ITERS),
             "--lambda_dist", "100"],
            fase_label="PASO 3/5 — Entrenando 2DGS")
        log("   2DGS entrenado")

        # ── PASO 4: extraer malla por TSDF (OPTIMIZADO) ──
        fase(0.80, "PASO 4/5 — Extrayendo malla (TSDF)")
        # GANANCIA GRANDE de velocidad: el TSDF de Open3D corre en CPU y, en pods
        # con muchos vCPU, abre demasiados hilos y se vuelve LENTÍSIMO (28 min en
        # la prueba anterior, ~2.87 s por vista). Limitando OMP a 8 hilos, baja a
        # ~2-3 min sin cambiar el algoritmo.
        env_mesh = dict(os.environ)
        env_mesh["OMP_NUM_THREADS"] = "8"

        # depth_trunc ADAPTATIVO: COLMAP usa una escala ARBITRARIA (no metros),
        # así que un valor fijo (9.0) puede cortar el cuarto en una escena y
        # sobrar en otra. Medimos la escala real con las posiciones de las
        # cámaras y ponemos un depth_trunc generoso para que NUNCA corte las
        # paredes (los floaters lejanos los limpia num_cluster).
        def _centros_camara(images_txt):
            centros = []
            with open(images_txt, "r") as f:
                lineas = [l for l in f if l.strip() and not l.startswith("#")]
            # En images.txt cada cámara son 2 líneas; la de pose tiene >=10
            # columnas (la 2ª, de puntos 2D, va vacía en nuestro caso).
            for l in lineas:
                p = l.split()
                if len(p) < 10:
                    continue
                qw, qx, qy, qz = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                tx, ty, tz = float(p[5]), float(p[6]), float(p[7])
                # centro de cámara C = -R^T t (R desde el cuaternión COLMAP)
                R = [[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]]
                Cx = -(R[0][0]*tx + R[1][0]*ty + R[2][0]*tz)
                Cy = -(R[0][1]*tx + R[1][1]*ty + R[2][1]*tz)
                Cz = -(R[0][2]*tx + R[1][2]*ty + R[2][2]*tz)
                centros.append((Cx, Cy, Cz))
            return centros
        depth_trunc = "8.0"   # fallback sensato si no se puede medir
        try:
            import math
            cps = _centros_camara(dataset / "sparse" / "0" / "images.txt")
            if len(cps) >= 2:
                cx = sum(p[0] for p in cps) / len(cps)
                cy = sum(p[1] for p in cps) / len(cps)
                cz = sum(p[2] for p in cps) / len(cps)
                dists = sorted(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2)
                               for p in cps)
                # CLAVE: antes usaba la distancia MÁXIMA ×3 = 40.99 → integraba
                # ruido MUY detrás de las paredes → 50 millones de triángulos y
                # pedazos flotando lejos. Ahora usamos el percentil 85 (robusto a
                # cámaras atípicas) ×1.8 y lo acotamos a un rango sensato. Así
                # cubre las paredes SIN integrar ruido lejano.
                radio = dists[int(len(dists) * 0.85)]
                depth_trunc = "(no se usa en unbounded)"
                log(f"   escala COLMAP (informativa): radio(p85)={radio:.2f} unidades. "
                    f"Modo unbounded → sin recorte de profundidad.")
        except Exception as e:
            log(f"   (depth_trunc fijo {depth_trunc}; no se midió escala: {e})")

        # ── EXTRACCIÓN EN MODO UNBOUNDED (sin recorte) ──
        # CAMBIO CLAVE (investigación): el modo BOUNDED del TSDF descarta toda la
        # profundidad más lejana que depth_trunc. En un cuarto fotografiado desde
        # adentro, eso RECORTABA paredes lejanas, techo y piso → "cuarto a medias".
        # El modo --unbounded usa contracción de espacio (como Mip-NeRF360): NO
        # recorta nada. Probamos si ESTE era el motivo de las zonas faltantes.
        # En unbounded NO se usan depth_trunc/voxel_size/sdf_trunc (los ignora);
        # solo manda --mesh_res (resolución). num_cluster=100 conserva los 100
        # pedazos conectados más grandes (los diagnósticos de abajo nos dirán
        # cuántos pedazos hay y dónde está la "ventana flotante").
        # mesh_res=512 (NO 1024): con 1024 la malla salía GIGANTESCA y el paso de
        # simplificación se quedaba sin memoria (se colgó en "90%"). 512 sigue
        # siendo unbounded (sin recorte) pero produce una malla manejable.
        log(f"$ python /opt/2dgs/render.py --unbounded --mesh_res 512 --num_cluster 100  (OMP=8)")
        rc_mesh, _salida_mesh = run(
            ["python", "/opt/2dgs/render.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--skip_train", "--skip_test",
             "--depth_ratio", "0",
             "--unbounded",
             "--mesh_res", "512",
             "--num_cluster", "100"],
            env=env_mesh, fase_label="PASO 4/5 — Extrayendo malla", check=False)
        # Buscar la malla generada. En modo unbounded los nombres son
        # fuse_unbounded.ply (cruda) y fuse_unbounded_post.ply (limpia). Preferimos
        # la limpia. Dejamos los nombres bounded como respaldo por si acaso.
        candidatos = list(dgs_out.rglob("*.ply"))
        def _es_no_vacia(p):
            try:
                return p.stat().st_size > 1000   # >1KB = tiene geometría real
            except Exception:
                return False
        malla = None
        for nombre in ("fuse_unbounded_post.ply", "fuse_unbounded.ply",
                       "fuse_post.ply", "fuse.ply"):
            for c in candidatos:
                if c.name.lower() == nombre and _es_no_vacia(c):
                    malla = c; break
            if malla:
                break
        if malla is None:
            for c in candidatos:
                if "mesh" in c.name.lower() and _es_no_vacia(c):
                    malla = c; break
        if malla is None:
            no_vacias = [c for c in candidatos if _es_no_vacia(c)]
            if no_vacias:
                malla = max(no_vacias, key=lambda p: p.stat().st_size)
        if malla is None:
            # No hubo malla con geometría → ahí sí es error real.
            raise RuntimeError(
                f"La malla salió vacía. Código render={rc_mesh}. "
                f"Posible: pocas iteraciones o poses débiles.")
        ply_mb = malla.stat().st_size / 1e6
        log(f"   malla: {malla.name} ({ply_mb:.1f} MB)")

        # ── Simplificar la malla (de cientos de MB a algo liviano) ──
        # La malla TSDF trae MILLONES de triángulos con color por vértice (por
        # eso pesaba 355 MB). La decimamos a ~1 millón de triángulos manteniendo
        # la forma y el color. Usamos open3d (ya instalado para el TSDF de 2DGS).
        fase(0.90, "PASO 4/5 — Simplificando malla")
        decimada = WORK / "mesh_lite.ply"
        script_dec = (
            "import open3d as o3d\n"
            "import numpy as np\n"
            f"m = o3d.io.read_triangle_mesh(r'{malla}')\n"
            "n0 = len(m.triangles)\n"
            "print('DIAG vertices', len(m.vertices), 'triangulos', n0, flush=True)\n"
            # --- DECIMAR PRIMERO (sobre la malla grande) ---
            # ANTES los diagnósticos corrían sobre la malla GIGANTE y se quedaban
            # sin memoria (se colgó en 90%). Ahora decimamos primero y los
            # diagnósticos van después, sobre algo liviano. Si la malla viene
            # enorme (>4M triángulos), decimamos en 2 pasos para no reventar RAM.
            "target = 500000\n"
            "if n0 > 4000000:\n"
            "    print('DECIMATE malla muy grande (%d), simplifico en 2 pasos...' % n0, flush=True)\n"
            "    m = m.simplify_quadric_decimation(target_number_of_triangles=1500000)\n"
            "    print('DECIMATE paso 1 ok:', len(m.triangles), flush=True)\n"
            "if len(m.triangles) > target:\n"
            "    m = m.simplify_quadric_decimation(target_number_of_triangles=target)\n"
            "m.remove_unreferenced_vertices()\n"
            "m.remove_degenerate_triangles()\n"
            "m.remove_duplicated_vertices()\n"
            "m.compute_vertex_normals()\n"
            "m.compute_triangle_normals()\n"
            f"o3d.io.write_triangle_mesh(r'{decimada}', m)\n"
            "print('DECIMATE triangulos', n0, '->', len(m.triangles), flush=True)\n"
            # --- DIAGNÓSTICOS (sobre la malla YA decimada = liviana y segura) ---
            # Nos dicen: tamaño del cuarto (bbox), cuántos pedazos sueltos hay, y
            # de cada pedazo su tamaño y QUÉ TAN LEJOS está del centro.
            # La "ventana flotante" = un pedazo pequeño MUY lejos del centro.
            "try:\n"
            "    aabb = m.get_axis_aligned_bounding_box()\n"
            "    ext = aabb.get_extent(); cg = aabb.get_center()\n"
            "    nt = len(m.triangles)\n"
            "    print('DIAG bbox_global X=%.2f Y=%.2f Z=%.2f (unidades COLMAP)' % (ext[0], ext[1], ext[2]), flush=True)\n"
            "    cl = m.cluster_connected_triangles()\n"
            "    lab = np.asarray(cl[0]); ntri = np.asarray(cl[1])\n"
            "    print('DIAG componentes_conexas', len(ntri), flush=True)\n"
            "    V = np.asarray(m.vertices); T = np.asarray(m.triangles)\n"
            "    order = np.argsort(ntri)[::-1]\n"
            "    for k, i in enumerate(order[:8]):\n"
            "        mask = lab == i\n"
            "        vidx = np.unique(T[mask].reshape(-1)); vv = V[vidx]\n"
            "        cmin = vv.min(0); cmax = vv.max(0); c = (cmin+cmax)/2; sz = cmax-cmin\n"
            "        d = float(np.linalg.norm(c - cg)); pct = 100.0*ntri[i]/max(nt,1)\n"
            "        print('DIAG comp%d: %d tri (%.1f%%) tamano(%.2f,%.2f,%.2f) dist_al_centro=%.2f' % (k, int(ntri[i]), pct, sz[0], sz[1], sz[2], d), flush=True)\n"
            "except Exception as e:\n"
            "    print('DIAG (fallo diagnostico, sigo):', e, flush=True)\n"
        )
        rc_dec, _ = run(["python", "-c", script_dec],
                        fase_label="PASO 4/5 — Simplificando malla", check=False)
        if decimada.exists() and decimada.stat().st_size > 1000:
            nuevo_mb = decimada.stat().st_size / 1e6
            log(f"   malla simplificada: {nuevo_mb:.1f} MB (antes {ply_mb:.1f} MB)")
            malla = decimada
            ply_mb = nuevo_mb
        else:
            log("   (no se pudo simplificar; subo la malla original)")

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
