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
# ID del pod donde estoy corriendo. RunPod lo inyecta solo. Se lo mandamos al
# backend en CADA callback para que SIEMPRE sepa a quién apagar, aunque su base
# de datos se haya borrado (Railway borra /data en cada reinicio/despliegue).
# Sin esto pasó lo siguiente EN PRODUCCIÓN: el render terminó, el aviso rebotó
# con 404, nadie apagó el pod, RunPod lo relanzó y volvió a rendir 35 min. En
# bucle. Cobrando GPU.
POD_ID = (os.environ.get("RUNPOD_POD_ID")
          or os.environ.get("RUNPOD_POD_HOSTNAME", "").split("-")[0]
          or "")

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

# Semilla fija → resultados reproducibles entre corridas (misma entrada = misma salida).
# Antes, dos corridas con las MISMAS fotos podían dar geometrías distintas (a veces
# buena, a veces dañada) por la aleatoriedad interna. Esto lo elimina en gran parte.
import random as _rnd
_rnd.seed(42); np.random.seed(42); torch.manual_seed(42)
try:
    torch.cuda.manual_seed_all(42)
except Exception:
    pass

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
# ===== ENTRENAMIENTO EN ALTA RESOLUCIÓN (CAMBIO CLAVE DE CALIDAD) =====
# MASt3R calcula las POSES a 512px (donde da buenas poses), PERO guardamos las
# imágenes de ENTRENAMIENTO en alta resolución desde las fotos ORIGINALES del
# celular (12MP) para que 2DGS aprenda MÁS DETALLE. Esto NO es "pegar fotos a la
# malla" (eso fue OpenMVS y se desalineaba): aquí la IA aprende el cuarto entero
# con más detalle desde el inicio y el color queda integrado, sin nada que pueda
# desalinearse. El FOV es un ÁNGULO (invariante a la resolución), así que las
# poses de 512px siguen siendo válidas; solo escalamos los intrínsecos al nuevo
# tamaño. Para fotos 4:3 el recorte central es exacto; para otros formatos,
# near-exact. Si una foto original falla, cae a la de 512px (no rompe la corrida).
TRAIN_RES = 1000   # lado mayor de las imágenes de entrenamiento. 1600px resultó
#                    INESTABLE: el entrenamiento colapsaba (PSNR caía a 27.7, malla con
#                    huecos y techo derrumbado). 1000px es ESTABLE (PSNR ~32.5). El techo
#                    real es la captura (poses a 512px, celular sin LiDAR), no la resolución.
print("ENTRENAMIENTO a %dpx (alta resolucion; poses a 512px)" % TRAIN_RES, flush=True)
_n_hi = 0
for i in range(N):
    im = rgbimgs[i]
    H, W = im.shape[:2]              # tamaño a 512px (referencia de aspecto/encuadre)
    aspect = W / float(H)
    name = "img_%04d.png" % i
    K = intrinsics[i]
    try:
        orig = Image.open(filelist[i]).convert("RGB")   # foto original (cam i = filelist[i])
        Wo, Ho = orig.size
        # recorte central al MISMO aspecto que la versión 512px (replica el encuadre)
        if (Wo / float(Ho)) > aspect:
            cw = int(round(Ho * aspect)); ch = Ho
        else:
            cw = Wo; ch = int(round(Wo / aspect))
        left = (Wo - cw) // 2; top = (Ho - ch) // 2
        orig = orig.crop((left, top, left + cw, top + ch))
        # escalar para que el lado mayor sea TRAIN_RES
        if cw >= ch:
            nw = TRAIN_RES; nh = max(1, int(round(ch * TRAIN_RES / float(cw))))
        else:
            nh = TRAIN_RES; nw = max(1, int(round(cw * TRAIN_RES / float(ch))))
        orig.resize((nw, nh), Image.LANCZOS).save(os.path.join(img_out, name))
        scale = nw / float(W)        # factor 512px -> alta resolución
        Wsave, Hsave = nw, nh
        _n_hi += 1
    except Exception as _e:
        # fallback seguro: guardar la imagen de 512px de MASt3R
        print("HIRES fallo en cam %d (%s), uso 512px" % (i, _e), flush=True)
        Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(os.path.join(img_out, name))
        scale = 1.0; Wsave, Hsave = W, H
    # intrínsecos escalados al nuevo tamaño (mismo FOV); cx,cy siguen centrados
    fx = float(K[0, 0]) * scale; fy = float(K[1, 1]) * scale
    cx = float(K[0, 2]) * scale; cy = float(K[1, 2]) * scale
    cam_id = i + 1
    fcam.write("%d PINHOLE %d %d %.6f %.6f %.6f %.6f\n" % (cam_id, Wsave, Hsave, fx, fy, cx, cy))
    # COLMAP guarda world->cam = inversa de cam->world (poses NO cambian con la resolución)
    w2c = np.linalg.inv(cams2world[i])
    q = Rotation.from_matrix(w2c[:3, :3]).as_quat()   # [x,y,z,w]
    t = w2c[:3, 3]
    fimg.write("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %d %s\n" %
               (cam_id, float(q[3]), float(q[0]), float(q[1]), float(q[2]),
                float(t[0]), float(t[1]), float(t[2]), cam_id, name))
    fimg.write("\n")   # linea de puntos 2D (vacia)
print("ENTRENAMIENTO: %d/%d imagenes guardadas en alta resolucion" % (_n_hi, N), flush=True)
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


# ════════════════════════════════════════════════════════════════════════════
# SCRIPT DE TEXTURIZADO CON OpenMVS (corre en el pod como subproceso).
# ----------------------------------------------------------------------------
# Reemplaza al horneado propio (mejor-vista + nivelacion casera), que dejaba
# mosaicos y costuras. OpenMVS TextureMesh (ya viene en la imagen v4.2) hace lo
# mismo que Polycam: elige la mejor foto por cara con graph-cut y nivela el
# color de las costuras de forma GLOBAL y LOCAL (Waechter et al. ECCV 2014).
# Le pasamos NUESTRA malla (-m) y las fotos 12MP con las poses escaladas exactas;
# el atlas y la textura los genera OpenMVS. Calidad ADAPTATIVA: usa tantas
# texturas de 8192 como la escena pida, sin tope fijo.
OPENMVS_TEXTURE_SCRIPT = r'''
import os, sys, time, shutil, subprocess, glob
import numpy as np

def log(s): print("   [omvs] " + s, flush=True)

# Texturiza la malla con OpenMVS (metodo Polycam: mejor-vista por cara con
# graph-cut + nivelado de costuras). Reemplaza al horneado propio.
#
#   python openmvs_texture.py  mesh.ply  ORIG12MP_DIR  SPARSE_DIR  out.glb  [ao.npy]
#
# HISTORIA:
#  v8.0: sparse+poses+coords OK, pero glb 12MP+8192 reventaba la RAM.
#  v8.1: baje fotos/textura -> OpenMVS texturizo OK (23 tex 2048, 11.6 GB),
#        PERO el glb PROPIO de OpenMVS sale mal formado -> se veia BLANCO.
#  v8.2: pido OBJ (que OpenMVS escribe perfecto) y lo convierto a glb
#        limpio con trimesh (texturas bien incrustadas). Ademas: el "nivelado
#        GLOBAL de costuras" de OpenMVS tiene un bug (_Map_base::at) que
#        crashea con esta malla -> lo dejo APAGADO por defecto (queda el local).
#  v8.5 (esta, tras investigacion a fondo del codigo fuente de OpenMVS):
#        el "vitral" = ~54k micro-parches con banda NEGRA interna. Confirmado:
#        (a) cost-smoothness-ratio va AL REVES (hacia 0 = parches grandes) y
#        casi no influye; (b) las bandas negras las escriben el nivelado LOCAL
#        de costuras (Poisson sin base global) y el sharpen (default 0.5);
#        (c) el crash _Map_base::at del nivelado GLOBAL viene de malla
#        NO-manifold (el mantenedor lo confirmo). ARREGLOS v8.5: reparar la
#        malla a manifold + --virtual-face-images 3 (agrupa triangulos
#        coplanares en parches GRANDES: el arreglo real) + nivelado local y
#        sharpen APAGADOS. Respaldo cfg2 con SOLO banderas viejas probadas,
#        por si el binario del pod no conoce las nuevas.
#  v8.6 (esta): el vitral MURIO con v8.5 (confirmado por Felipe); quedaron
#        escalones de tono entre parches (esperado: niveladores apagados).
#        Ahora que la malla va manifold, se REACTIVAN nivelado GLOBAL+LOCAL
#        (la config de diseno del algoritmo) para emparejar el tono. Respaldo
#        cfg2 = la config exacta de v8.5 que acaba de funcionar: si el global
#        crashara, se cae a lo de hoy, nunca peor.
#  v8.7 (esta): el nivelado global crasheo (rc=-6) INCLUSO con la malla ya
#        manifold -> investigacion 2: el crash es un mapa de parches sin fila
#        en GlobalSeamLeveling (probable choque con las caras virtuales, que
#        NO podemos quitar porque matan el vitral). Solucion: NIVELAR LA
#        EXPOSICION NOSOTROS (Plan B1) = resolver una ganancia por canal por
#        foto (minimos cuadrados en log, espacio LINEAL, mediana por par,
#        ancla sum(log g)=0, tope +-1 stop) usando puntos de la malla vistos
#        en varias fotos, y corregir las fotos ANTES de texturizar. Es lo que
#        hacen AliceVision ("correct exposure in linear") y Metashape
#        ("Calibrate colors"). Niveladores de OpenMVS: apagados otra vez.
#  v8.8 (esta, OPCION C): analisis DIRECTO de los .glb reales (descargados y
#        destripados byte a byte) probo: (a) el naranja NO es relleno, es
#        madera real de las fotos; (b) la nivelacion de fotos (B1) EMPEORO el
#        tono -> APAGADA; (c) los 6 materiales YA estan balanceados entre si
#        (nivelar por material = 1% de mejora); (d) el escalon esta entre los
#        ~86.000 PARCHES dentro de cada material. Salto medido en costuras:
#        mediana 15.8, p90 58.7 (de 0-255). ARREGLO: NIVELADO DE TONO POR
#        PARCHE aqui mismo, en el atlas, donde SI conozco las islas: cada isla
#        recibe UNA ganancia por canal (solo tono, jamas toca el detalle),
#        resuelta por minimos cuadrados para que las islas que se TOCAN en 3D
#        tengan el mismo tono. Es el nivelado global que OpenMVS no pudo hacer
#        (crashea), hecho por nosotros y sin riesgo: si algo falla, la textura
#        queda intacta.
MESH   = sys.argv[1]
ORIGD  = sys.argv[2]
SPARSE = sys.argv[3]
OUTGLB = sys.argv[4]

# Perillas (defaults probados en produccion):
TEX_MESH_TRIS = int(os.environ.get("TEX_MESH_TRIS", "1100000"))  # caras del glb final
IMG_MAX       = int(os.environ.get("OMVS_IMG_MAX", "2000"))      # lado mayor de las fotos que ve OpenMVS
MAX_TEX       = int(os.environ.get("OMVS_MAX_TEX", "4096"))      # tam. de textura (agrupa mejor -> menos archivos)
RES_LEVEL     = int(os.environ.get("OMVS_RES_LEVEL", "0"))       # 0 = usa las fotos tal cual se las paso
OUTLIER       = os.environ.get("OMVS_OUTLIER", "0.06")           # descarta fotos inconsistentes
SMOOTH_RATIO  = os.environ.get("OMVS_SMOOTH", "0.02")            # hacia 0 = parches GRANDES (investigacion: la escala va AL REVES; 1=mas fragmentado)
GLOBAL_SEAM   = os.environ.get("OMVS_GLOBAL_SEAM", "0")          # 0 = apagado: crashea (rc=-6) INCLUSO con malla manifold (probable choque con las caras virtuales). La nivelacion la hacemos nosotros (EXPO abajo)
LOCAL_SEAM    = os.environ.get("OMVS_LOCAL_SEAM", "0")           # 0 = apagado: sin base global escribe bandas negras (comprobado byte a byte)
SHARP         = os.environ.get("OMVS_SHARP", "0")                # 0 = apagado: el enfoque (default 0.5) crea halos oscuros en bordes de parches
VFACES        = os.environ.get("OMVS_VFACES", "3")               # caras virtuales coplanares: agrupa triangulos del mismo plano en parches GRANDES (el arreglo real de la fragmentacion)
EXPOCOMP      = os.environ.get("OMVS_EXPOCOMP", "0") == "1"     # 0 = APAGADO: medido sobre el .glb real, EMPEORO el tono (dispersion 21.6 -> 34.0). Se deja por si acaso
TONE_LEVEL    = os.environ.get("OMVS_TONE", "1") == "1"          # NIVELADO DE TONO POR PARCHE (Opcion C): iguala el tono entre islas vecinas del atlas
TONE_CLAMP    = float(os.environ.get("OMVS_TONE_CLAMP", "1.35")) # tope de la correccion por isla (1.35 = +-35%): solo mueve el TONO, nunca el detalle
TONE_MINF     = int(os.environ.get("OMVS_TONE_MINF", "3"))       # caras minimas por costura para creerle
EXPO_SAMPLES  = int(os.environ.get("OMVS_EXPO_SAMPLES", "40000"))# puntos de la malla muestreados para medir las ganancias
OMP_HI        = os.environ.get("OMVS_OMP", "6")                  # hilos del intento bueno

t0 = time.time()
WORK = os.path.dirname(os.path.abspath(OUTGLB))
MVS  = os.path.join(WORK, "mvs")
IMGD = os.path.join(MVS, "images")
SPD  = os.path.join(MVS, "sparse")
if os.path.isdir(MVS):
    shutil.rmtree(MVS, ignore_errors=True)
for d in (MVS, IMGD, SPD):
    os.makedirs(d, exist_ok=True)

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def find_photo(dirpath, name):
    p = os.path.join(dirpath, name)
    if os.path.exists(p): return p
    stem = os.path.splitext(name)[0]
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        q = os.path.join(dirpath, stem + ext)
        if os.path.exists(q): return q
    return None

# ── 1) cameras.txt (a 1000px) ──────────────────────────────────────────────
cams = {}
for line in open(os.path.join(SPARSE, "cameras.txt")):
    if line.startswith("#") or not line.strip(): continue
    e = line.split()
    cams[int(e[0])] = [int(e[2]), int(e[3]), float(e[4]), float(e[5]), float(e[6]), float(e[7])]

# ── 2) sparse a las fotos (recorte al aspecto + bajado a IMG_MAX; escala exacta) ─
fcam = open(os.path.join(SPD, "cameras.txt"), "w"); fcam.write("# Camera list\n")
fimg = open(os.path.join(SPD, "images.txt"), "w"); fimg.write("# Image list\n")
n_ok = 0; n_miss = 0; _res_ej = None
raw = [l for l in open(os.path.join(SPARSE, "images.txt"))]
i = 0
while i < len(raw):
    l = raw[i]
    if l.startswith("#") or not l.strip():
        i += 1; continue
    e = l.split()
    if len(e) < 10:
        i += 2; continue
    cid = int(e[0]); name = e[9]
    path = find_photo(ORIGD, name)
    if path is None or cid not in cams:
        if n_miss < 3: log("FOTO NO ENCONTRADA para %s (probe .jpg/.jpeg/.png)" % name)
        n_miss += 1; i += 2; continue
    W1, H1, fx1, fy1, cx1, cy1 = cams[cid]
    asp = W1 / float(H1)
    im = Image.open(path).convert("RGB")
    Wo, Ho = im.size
    if (Wo/float(Ho)) > asp: cw = int(round(Ho*asp)); ch = Ho
    else:                    cw = Wo; ch = int(round(Wo/asp))
    left = (Wo-cw)//2; top = (Ho-ch)//2
    im = im.crop((left, top, left+cw, top+ch))
    longe = max(cw, ch)
    if longe > IMG_MAX:
        r = IMG_MAX / float(longe)
        rw = max(1, int(round(cw*r))); rh = max(1, int(round(ch*r)))
        im = im.resize((rw, rh), Image.LANCZOS)
    else:
        rw, rh = cw, ch
    s = rw / float(W1)
    fx, fy, cx, cy = fx1*s, fy1*s, cx1*s, cy1*s
    jpg = os.path.splitext(name)[0] + ".jpg"
    im.save(os.path.join(IMGD, jpg), quality=92)
    if _res_ej is None: _res_ej = (rw, rh)
    fcam.write("%d PINHOLE %d %d %.6f %.6f %.6f %.6f\n" % (cid, rw, rh, fx, fy, cx, cy))
    fimg.write("%d %s %s %s %s %s %s %s %d %s\n" %
               (cid, e[1], e[2], e[3], e[4], e[5], e[6], e[7], int(e[8]), jpg))
    fimg.write("\n")
    n_ok += 1
    i += 2
fcam.close(); fimg.close()
try:
    shutil.copy(os.path.join(SPARSE, "points3D.txt"), os.path.join(SPD, "points3D.txt"))
except Exception as _pe:
    open(os.path.join(SPD, "points3D.txt"), "w").write("# 3D point list\n")
    log("(points3D.txt no copiado: %s)" % _pe)
log("sparse listo: %d camaras a %s px (%d fotos no encontradas)"
    % (n_ok, ("%dx%d" % _res_ej if _res_ej else "?"), n_miss))
if n_ok == 0:
    log("ERROR: 0 camaras utilizables; no puedo texturizar"); sys.exit(3)

# ── 3) decimar la malla + REPARARLA A MANIFOLD ─────────────────────────────
#   (Stage 0 de la investigacion: el crash del nivelado global y parte del dano
#   en los parches vienen de aristas/vertices NO-manifold, tipicos de una malla
#   TSDF decimada. Se limpian ANTES de texturizar.)
import open3d as o3d
m = o3d.io.read_triangle_mesh(MESH)
nt0 = len(m.triangles)
if nt0 > TEX_MESH_TRIS:
    m = m.simplify_quadric_decimation(target_number_of_triangles=TEX_MESH_TRIS)
    m.remove_unreferenced_vertices()
m.remove_degenerate_triangles(); m.remove_duplicated_vertices()
m.remove_duplicated_triangles()
_qe = _qv = 0
for _rep in range(4):
    try:
        _e = np.asarray(m.get_non_manifold_edges())
    except Exception:
        _e = np.zeros((0, 2))
    try:
        _v = np.asarray(m.get_non_manifold_vertices())
    except Exception:
        _v = np.zeros(0)
    if len(_e) == 0 and len(_v) == 0:
        break
    if len(_e):
        try:
            m.remove_non_manifold_edges(); _qe += len(_e)
        except Exception:
            pass
    try:
        _v = np.asarray(m.get_non_manifold_vertices())
        if len(_v):
            m.remove_vertices_by_index([int(i) for i in _v]); _qv += len(_v)
    except Exception:
        pass
    m.remove_degenerate_triangles(); m.remove_duplicated_vertices()
    m.remove_duplicated_triangles(); m.remove_unreferenced_vertices()
try:
    _re = len(np.asarray(m.get_non_manifold_edges()))
    _rv = len(np.asarray(m.get_non_manifold_vertices()))
except Exception:
    _re = _rv = -1
log("malla reparada a MANIFOLD: quite %d aristas + %d vertices no-manifold (quedan %d aristas / %d vertices)"
    % (_qe, _qv, _re, _rv))
MFT = os.path.join(WORK, "mesh_for_tex.ply")
m2 = o3d.geometry.TriangleMesh(m.vertices, m.triangles)
o3d.io.write_triangle_mesh(MFT, m2)
log("malla para textura: %d -> %d caras" % (nt0, len(m2.triangles)))

# ── 3b) NIVELACION DE EXPOSICION entre las fotos (Plan B1 de la investigacion) ──
#   El nivelador de OpenMVS crashea (rc=-6), asi que nivelamos NOSOTROS antes de
#   texturizar: puntos de la malla visibles en varias fotos -> una GANANCIA por
#   canal por foto (minimos cuadrados en log, espacio LINEAL, mediana por par =
#   robusto, ancla sum(log g)=0, tope +-1 stop) -> fotos corregidas en disco.
#   Si algo falla, se sigue con las fotos originales: el render nunca se pierde.
if EXPOCOMP:
  try:
    _te = time.time()
    def _s2l(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    def _l2s(c):
        c = np.clip(c, 0.0, 1.0)
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)
    def _q2R(qw, qx, qy, qz):
        n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
        return np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)]])
    _ec = {}
    for _l in open(os.path.join(SPD, "cameras.txt")):
        if _l.startswith("#") or not _l.strip(): continue
        _p = _l.split()
        _ec[int(_p[0])] = (int(_p[2]), int(_p[3]), float(_p[4]), float(_p[5]), float(_p[6]), float(_p[7]))
    _ev = []
    for _l in open(os.path.join(SPD, "images.txt")):
        if _l.startswith("#") or not _l.strip(): continue
        _p = _l.split()
        if len(_p) >= 10 and _p[9].endswith(".jpg"):
            _ev.append((int(_p[0]), _p[9],
                        _q2R(float(_p[1]), float(_p[2]), float(_p[3]), float(_p[4])),
                        np.array([float(_p[5]), float(_p[6]), float(_p[7])])))
    _scn = o3d.t.geometry.RaycastingScene()
    _scn.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m2))
    m.compute_vertex_normals()
    _P = np.asarray(m2.vertices); _NRM = np.asarray(m.vertex_normals)
    np.random.seed(42)
    _S = min(EXPO_SAMPLES, len(_P))
    _sel = np.random.choice(len(_P), _S, replace=False)
    _Ps = _P[_sel]; _Ns = _NRM[_sel]
    _NC = len(_ev)
    _O = np.full((_NC, _S, 3), np.nan, np.float32)
    for _k, (_cid, _nm, _R, _t) in enumerate(_ev):
        if _cid not in _ec: continue
        _W, _H, _fx, _fy, _cx, _cy = _ec[_cid]
        _pth = os.path.join(IMGD, _nm)
        if not os.path.exists(_pth): continue
        _Xc = (_R @ _Ps.T).T + _t
        _z = _Xc[:, 2]
        _u = _fx * _Xc[:, 0] / np.maximum(_z, 1e-9) + _cx
        _v = _fy * _Xc[:, 1] / np.maximum(_z, 1e-9) + _cy
        _in = (_z > 0.05) & (_u >= 2) & (_u <= _W - 3) & (_v >= 2) & (_v <= _H - 3)
        _C = -_R.T @ _t
        _dir = _Ps - _C[None, :]
        _dst = np.linalg.norm(_dir, axis=1)
        _dirn = _dir / np.maximum(_dst[:, None], 1e-9)
        _in &= np.abs((_Ns * (-_dirn)).sum(1)) > 0.25    # sin angulos rasantes
        _idx = np.where(_in)[0]
        if len(_idx) < 50: continue
        _rays = np.concatenate([np.repeat(_C[None, :], len(_idx), 0), _dirn[_idx]], 1).astype(np.float32)
        _th = _scn.cast_rays(o3d.core.Tensor(_rays))["t_hit"].numpy()
        _eps = np.maximum(0.02, 0.01 * _dst[_idx])
        _keep = np.isfinite(_th) & (_th >= _dst[_idx] - _eps)   # sin oclusion
        _vis = _idx[_keep]
        if len(_vis) < 50: continue
        _im = np.asarray(Image.open(_pth).convert("RGB"))
        _xi = np.clip(np.round(_u[_vis]).astype(int), 0, _W - 1)
        _yi = np.clip(np.round(_v[_vis]).astype(int), 0, _H - 1)
        _px = _im[_yi, _xi].astype(np.float32)
        _ok = ((_px > 6) & (_px < 250)).all(1)           # sin pixeles recortados
        _vis = _vis[_ok]; _px = _px[_ok]
        if len(_vis) < 50: continue
        _O[_k, _vis] = _s2l(_px / 255.0)
    _V = np.isfinite(_O[:, :, 0])
    _rows = []; _rhs = [[], [], []]; _w = []
    for _i in range(_NC):
        for _j in range(_i + 1, _NC):
            _mij = _V[_i] & _V[_j]
            _n = int(_mij.sum())
            if _n < 20: continue
            _li = np.log(np.maximum(_O[_i, _mij], 1e-4))
            _lj = np.log(np.maximum(_O[_j, _mij], 1e-4))
            _d = np.median(_lj - _li, axis=0)            # a_i - a_j = log(Ij/Ii)
            _rows.append((_i, _j)); _w.append(min(_n, 500) ** 0.5)
            for _c in range(3): _rhs[_c].append(float(_d[_c]))
    if len(_rows) < _NC:
        log("EXPO: solo %d pares de fotos con puntos comunes; NO nivelo (sigo con las fotos originales)" % len(_rows))
    else:
        _A = np.zeros((len(_rows) + 1, _NC))
        for _r, (_i, _j) in enumerate(_rows):
            _A[_r, _i] = _w[_r]; _A[_r, _j] = -_w[_r]
        _A[-1, :] = (10.0 * max(_w)) / _NC               # ancla: sum(log g) = 0
        _G = np.ones((_NC, 3))
        for _c in range(3):
            _b = np.array([_rhs[_c][_r] * _w[_r] for _r in range(len(_rows))] + [0.0])
            _a = np.linalg.lstsq(_A, _b, rcond=None)[0]
            _G[:, _c] = np.exp(_a)
        _G[_V.sum(1) < 50] = 1.0                         # camaras sin datos: no tocar
        _nclamp = int(((_G < 0.5) | (_G > 2.0)).sum())
        _G = np.clip(_G, 0.5, 2.0)                       # tope +-1 stop
        for _k, (_cid, _nm, _R, _t) in enumerate(_ev):
            _pth = os.path.join(IMGD, _nm)
            if not os.path.exists(_pth): continue
            _im = np.asarray(Image.open(_pth).convert("RGB"), np.float32) / 255.0
            _out = (_l2s(_s2l(_im) * _G[_k][None, None, :]) * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(_out).save(_pth, quality=92)
        log("EXPO: exposicion nivelada en %d fotos (%d pares, %d muestras) en %.1f min"
            % (_NC, len(_rows), _S, (time.time() - _te) / 60.0))
        log("EXPO: ganancias medianas R %.3f G %.3f B %.3f | rango %.2f-%.2f | %d en tope (muchos en tope = correspondencias ruidosas)"
            % (float(np.median(_G[:, 0])), float(np.median(_G[:, 1])), float(np.median(_G[:, 2])),
               float(_G.min()), float(_G.max()), _nclamp))
  except Exception as _ee:
    log("EXPO: nivelacion fallo (%s); sigo con las fotos originales" % _ee)
else:
    log("EXPO: nivelacion de exposicion APAGADA (OMVS_EXPOCOMP=0)")

# ── 4) binarios de OpenMVS ─────────────────────────────────────────────────
def which(nm):
    p = shutil.which(nm)
    if p: return p
    for c in ("/usr/local/bin/OpenMVS/" + nm, "/usr/local/bin/" + nm):
        if os.path.exists(c): return c
    return None
IFACE = which("InterfaceCOLMAP"); TEXM = which("TextureMesh")
if not IFACE or not TEXM:
    log("ERROR: no encuentro InterfaceCOLMAP/TextureMesh en la imagen"); sys.exit(4)

def run(cmd, tag, env=None):
    log("$ " + " ".join([os.path.basename(cmd[0])] + [str(a) for a in cmd[1:]]))
    t = time.time()
    r = subprocess.run(cmd, cwd=MVS, capture_output=True, text=True, env=env)
    for ln in ((r.stdout or "") + "\n" + (r.stderr or "")).strip().splitlines()[-8:]:
        log("  | " + ln[:170])
    log("%s en %.1f min (rc=%d)" % (tag, (time.time()-t)/60.0, r.returncode))
    return r.returncode

# 4a) COLMAP -> .mvs
SCENE = os.path.join(MVS, "scene.mvs")
rc = run([IFACE, "-i", MVS, "-o", SCENE, "--image-folder", IMGD], "InterfaceCOLMAP")
if rc != 0 or not os.path.exists(SCENE):
    log("ERROR: InterfaceCOLMAP no produjo scene.mvs"); sys.exit(5)

# 4b) TextureMesh -> OBJ (el glb propio de OpenMVS sale roto). OBJ = malla+mtl+
#     imagenes, que trimesh convierte a un glb limpio con texturas incrustadas.
BASE = os.path.join(MVS, "textured")
def texcmd(max_tex, extra):
    c = [TEXM, "-i", SCENE, "-m", MFT, "-o", BASE + ".obj",
         "--export-type", "obj",
         "--resolution-level", str(RES_LEVEL),
         "--max-texture-size", str(max_tex),
         "--outlier-threshold", str(OUTLIER),
         "--cost-smoothness-ratio", str(SMOOTH_RATIO)]
    return c + list(extra)

def _patch_unlit_matte(glbpath):
    """Parcha TODOS los materiales del glb a MATE + UNLIT: metallicFactor=0,
    roughnessFactor=1 y KHR_materials_unlit. Sin esto, glTF asume metal=1.0 y el
    visor pinta el cuarto como metal negro facetado -> aspecto de 'vidrio roto'."""
    import json as _json, struct as _st
    try:
        _d = bytearray(open(glbpath, "rb").read())
        _jlen = _st.unpack("<I", _d[12:16])[0]
        _g = _json.loads(_d[20:20 + _jlen].decode("utf-8"))
        _g.setdefault("extensionsUsed", [])
        if "KHR_materials_unlit" not in _g["extensionsUsed"]:
            _g["extensionsUsed"].append("KHR_materials_unlit")
        if not _g.get("materials"):
            _g["materials"] = [{}]
        for _m in _g["materials"]:
            _pbr = _m.setdefault("pbrMetallicRoughness", {})
            _pbr["metallicFactor"] = 0.0
            _pbr["roughnessFactor"] = 1.0
            _m.setdefault("extensions", {})["KHR_materials_unlit"] = {}
        _bin = _d[20 + _jlen:]
        _nj = _json.dumps(_g, separators=(",", ":"), allow_nan=False).encode("utf-8")
        while len(_nj) % 4:
            _nj += b" "
        _out = bytearray(); _out += _d[:12]
        _out += _st.pack("<I", len(_nj)) + b"JSON" + _nj + _bin
        _out[8:12] = _st.pack("<I", len(_out))
        open(glbpath, "wb").write(bytes(_out))
        log("material -> MATE + UNLIT (%d materiales; quita el metal/vidrio roto)" % len(_g["materials"]))
        return True
    except Exception as e:
        log("(no pude parchar material a unlit: %s)" % e); return False


def tone_level(objf, texfiles, mtl2tex):
    """NIVELADO DE TONO POR PARCHE (Opcion C) — v8.9.

    v8.8 FALLO: identificaba los parches por pixeles conectados del atlas, pero
    OpenMVS los empaca PEGADOS -> 69.319 parches se fundian en 8.300 manchones y
    solo aparecian 189 costuras (de decenas de miles). El nivelado tocaba casi
    nada y la metrica del log, medida solo sobre esas 189, enganaba.

    v8.9: los parches se identifican por su GEOMETRIA UV (exacto): dos caras son
    del mismo parche si comparten un indice de UV. Un vertice no puede estar en
    dos parches (tendria dos UV), asi que la conectividad UV ES el parche.
      1) parches = componentes conexas de caras que comparten indice UV;
      2) dos parches son VECINOS si comparten un VERTICE 3D de la malla;
      3) por cada costura mido la diferencia de tono entre los dos lados;
      4) resuelvo UNA ganancia por canal por parche (minimos cuadrados en log,
         luz LINEAL, prior hacia 1) para que los vecinos queden iguales;
      5) pinto: cada pixel del atlas toma la ganancia de su cara mas cercana.
    Como es UNA ganancia por parche, solo mueve el TONO; el detalle fino queda
    intacto. Si algo falla, devuelve False y la textura queda como estaba.
    """
    import numpy as _np
    from PIL import Image
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spl
    from scipy.sparse.csgraph import connected_components as _cc
    from scipy.spatial import cKDTree as _KDT
    _t = time.time()

    def _s2l(c): return _np.where(c <= 0.04045, c/12.92, ((c+0.055)/1.055)**2.4)
    def _l2s(c):
        c = _np.clip(c, 0.0, 1.0)
        return _np.where(c <= 0.0031308, c*12.92, 1.055*(c**(1.0/2.4)) - 0.055)
    def _fillmask(a):
        return (_np.abs(a[:, :, 0].astype(_np.int16) - 128) < 6) & \
               (_np.abs(a[:, :, 1].astype(_np.int16) - 128) < 6) & \
               (_np.abs(a[:, :, 2].astype(_np.int16) - 128) < 6)

    # ── leer el OBJ ──
    Vn = []; Tn = []; F = []; FT = []; FM = []; cur = -1
    with open(objf) as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split(); Vn.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith("vt "):
                p = ln.split(); Tn.append((float(p[1]), float(p[2])))
            elif ln.startswith("usemtl"):
                cur = mtl2tex.get(ln.split(None, 1)[1].strip(), -1)
            elif ln.startswith("f "):
                p = ln.split()
                if len(p) < 4: continue
                a = []; b = []
                for c in p[1:4]:
                    q = c.split("/")
                    a.append(int(q[0]) - 1)
                    b.append(int(q[1]) - 1 if len(q) > 1 and q[1] else -1)
                F.append(a); FT.append(b); FM.append(cur)
    F = _np.asarray(F, _np.int64); FT = _np.asarray(FT, _np.int64)
    FM = _np.asarray(FM, _np.int64); Tn = _np.asarray(Tn, _np.float64)
    NF = len(F)
    if NF < 1000 or len(Tn) == 0 or (FT < 0).any() or (FM < 0).any():
        log("TONO: el OBJ no trae UVs/materiales utilizables; textura sin tocar"); return False

    # ── 1) PARCHES = componentes conexas por indice de UV ──
    vt = FT.reshape(-1); ff = _np.repeat(_np.arange(NF), 3)
    o = _np.argsort(vt, kind="stable"); vt = vt[o]; ff = ff[o]
    st = _np.r_[0, _np.flatnonzero(_np.diff(vt)) + 1]
    msk = _np.ones(len(ff), bool); msk[st] = False
    pos = _np.flatnonzero(msk)
    e0 = ff[pos]; e1 = ff[pos - 1]
    NISL, isl = _cc(_sp.coo_matrix((_np.ones(len(e0), _np.int8), (e0, e1)),
                                   shape=(NF, NF)), directed=False)
    isl = isl.astype(_np.int64)

    # ── 2) color de cada cara en el atlas (una textura a la vez) ──
    cuv = Tn[FT].mean(axis=1)
    col = _np.zeros((NF, 3), _np.float64); okf = _np.zeros(NF, bool)
    votes = [0, 0]
    for ti in range(len(texfiles)):
        m = FM == ti
        if not m.any(): continue
        a = _np.asarray(Image.open(texfiles[ti]).convert("RGB"))
        H, W = a.shape[:2]; fm = _fillmask(a)
        px = _np.clip((cuv[m, 0] * (W - 1)).astype(_np.int64), 0, W - 1)
        for fl in (0, 1):
            vv = (1.0 - cuv[m, 1]) if fl == 0 else cuv[m, 1]
            py = _np.clip((vv * (H - 1)).astype(_np.int64), 0, H - 1)
            votes[fl] += int((~fm[py, px]).sum())
    flip = 0 if votes[0] >= votes[1] else 1
    for ti in range(len(texfiles)):
        m = FM == ti
        if not m.any(): continue
        a = _np.asarray(Image.open(texfiles[ti]).convert("RGB"))
        H, W = a.shape[:2]; fm = _fillmask(a)
        px = _np.clip((cuv[m, 0] * (W - 1)).astype(_np.int64), 0, W - 1)
        vv = (1.0 - cuv[m, 1]) if flip == 0 else cuv[m, 1]
        py = _np.clip((vv * (H - 1)).astype(_np.int64), 0, H - 1)
        col[m] = _s2l(a[py, px].astype(_np.float64) / 255.0)
        okf[m] = ~fm[py, px]
    okf &= col.min(1) > 0.002
    if okf.sum() < 1000:
        log("TONO: no pude leer el color de las caras; textura sin tocar"); return False

    # ── 3) costuras: vertices 3D donde se tocan DOS parches ──
    vi = F.reshape(-1); fi = _np.repeat(_np.arange(NF), 3)
    g = okf[fi]; vi = vi[g]; fi = fi[g]
    o = _np.argsort(vi, kind="stable"); vi = vi[o]; fi = fi[o]
    isf = isl[fi]
    st = _np.r_[0, _np.flatnonzero(_np.diff(vi)) + 1]
    en = _np.r_[st[1:], len(vi)]
    mn = _np.minimum.reduceat(isf, st); mx = _np.maximum.reduceat(isf, st)
    seam = _np.flatnonzero(mn != mx)
    lg = _np.log(_np.maximum(col, 1e-4))
    acc = {}; tocadas = set()
    for gi in seam:
        s0, s1 = st[gi], en[gi]
        ii = isf[s0:s1]; fj = fi[s0:s1]
        uq = _np.unique(ii)
        if len(uq) < 2: continue
        med = {}
        for u in uq:
            sel = fj[ii == u]; med[int(u)] = lg[sel].mean(0); tocadas.update(sel.tolist())
        for x in range(len(uq)):
            for y in range(x + 1, len(uq)):
                A, B = int(uq[x]), int(uq[y])
                d = _np.clip(med[B] - med[A], -0.7, 0.7)
                k = (A, B) if A < B else (B, A)
                if A > B: d = -d
                if k in acc: acc[k][0] += d; acc[k][1] += 1
                else: acc[k] = [d.copy(), 1]
    pairs = [(k, v) for k, v in acc.items() if v[1] >= TONE_MINF]
    cover = 100.0 * len(tocadas) / NF
    log("TONO: %d parches (OpenMVS empaco los suyos), %d costuras, tocan el %.0f%% de las caras"
        % (NISL, len(pairs), cover))
    if len(pairs) < 500 or cover < 5.0:
        log("TONO: muy pocas costuras utiles -> NO nivelo (textura sin tocar)"); return False

    # ── 4) una ganancia por canal por parche ──
    NP = len(pairs)
    ri = _np.repeat(_np.arange(NP), 2)
    ci = _np.empty(NP * 2, _np.int64); dv = _np.empty(NP * 2, _np.float64)
    w = _np.empty(NP); rhs = _np.empty((NP, 3))
    for i, ((A, B), (sd, n)) in enumerate(pairs):
        ww = min(n, 60) ** 0.5
        ci[2*i] = A; ci[2*i+1] = B; dv[2*i] = ww; dv[2*i+1] = -ww
        w[i] = ww; rhs[i] = sd / n
    lam = 0.05
    M = _sp.vstack([_sp.coo_matrix((dv, (ri, ci)), shape=(NP, NISL)),
                    _sp.identity(NISL, format="coo") * lam]).tocsr()
    G = _np.ones((NISL, 3))
    for c in range(3):
        b = _np.r_[rhs[:, c] * w, _np.zeros(NISL)]
        G[:, c] = _np.exp(_spl.lsqr(M, b, atol=1e-7, btol=1e-7, iter_lim=800)[0])
    lo, hi = 1.0 / TONE_CLAMP, TONE_CLAMP
    nclamp = int(((G < lo) | (G > hi)).sum()); G = _np.clip(G, lo, hi)
    lgG = _np.log(G).mean(1)
    d0 = _np.array([abs(float((sd/n).mean())) for (_k, (sd, n)) in pairs])
    d1 = _np.array([abs(float((sd/n).mean()) - (lgG[A]-lgG[B])) for ((A, B), (sd, n)) in pairs])
    red = 100.0 * (1.0 - d1.mean()/max(d0.mean(), 1e-9))

    # ── 5) pintar: cada pixel toma la ganancia de su cara mas cercana ──
    for ti, tf in enumerate(texfiles):
        m = _np.flatnonzero(FM == ti)
        if len(m) == 0: continue
        a = _np.asarray(Image.open(tf).convert("RGB"))
        H, W = a.shape[:2]; fm = _fillmask(a)
        ys, xs = _np.nonzero(~fm)
        if len(ys) == 0: continue
        cx = cuv[m, 0] * (W - 1)
        cy = ((1.0 - cuv[m, 1]) if flip == 0 else cuv[m, 1]) * (H - 1)
        tree = _KDT(_np.c_[cy, cx])
        try: _, nn = tree.query(_np.c_[ys, xs], workers=-1)
        except TypeError: _, nn = tree.query(_np.c_[ys, xs])
        gpx = G[isl[m[nn]]]
        lin = _s2l(a.astype(_np.float64) / 255.0)
        lin[ys, xs] *= gpx
        out = (_np.clip(_l2s(lin), 0, 1) * 255.0 + 0.5).astype(_np.uint8)
        im = Image.fromarray(out)
        if tf.lower().endswith((".jpg", ".jpeg")): im.save(tf, quality=95)
        else: im.save(tf)
    log("TONO: escalon medio en costuras BAJO %.0f%% | ganancias %.2f-%.2f (%d en tope) "
        "en %.1f min" % (red, float(G.min()), float(G.max()), nclamp, (time.time()-_t)/60.0))
    return True


def obj_to_glb(objf, outglb):
    """OBJ texturizado de OpenMVS -> glb. (a) Recolorea el NARANJA de relleno de
    OpenMVS (255,127,39; caras que ninguna foto vio) a gris DIRECTO en los archivos
    de textura, antes de cargar. (b) Carga la Scene (varios materiales) y exporta
    SIN concatenar (concatenar revienta la RAM de trimesh; trimesh.load ya orienta
    bien las UV, verificado byte a byte en el archivo real). (c) Material MATE+UNLIT
    para quitar el aspecto de metal/'vidrio roto'."""
    import trimesh
    import numpy as _np
    from PIL import Image
    objdir = os.path.dirname(objf)
    # (a) recolorear el naranja en los ARCHIVOS de textura (via el .mtl)
    mtlpath = None
    for l in open(objf):
        if l.startswith("mtllib"):
            mtlpath = os.path.join(objdir, l.split(None, 1)[1].strip()); break
    texfiles = []; mtl2tex = {}; _curm = None
    if mtlpath and os.path.exists(mtlpath):
        for l in open(mtlpath):
            p = l.split()
            if p and p[0] == "newmtl":
                _curm = l.split(None, 1)[1].strip()
            elif p and p[0] == "map_Kd":
                tf = os.path.join(objdir, l.split(None, 1)[1].strip())
                if os.path.exists(tf):
                    if _curm is not None: mtl2tex[_curm] = len(texfiles)
                    texfiles.append(tf)
    _norange = 0
    for tf in texfiles:
        try:
            a = _np.asarray(Image.open(tf).convert("RGB")).copy()
            fill = (a[:, :, 0] > 235) & (a[:, :, 1] > 105) & (a[:, :, 1] < 150) & (a[:, :, 2] < 70)
            if fill.any():
                a[fill] = (128, 128, 128); _norange += int(fill.sum())
                Image.fromarray(a).save(tf)
        except Exception as _te:
            log("(no pude recolorear %s: %s)" % (os.path.basename(tf), _te))
    if _norange:
        log("relleno naranja de OpenMVS -> gris: %d pixeles en %d texturas" % (_norange, len(texfiles)))
    log("glb: %d texturas" % len(texfiles))
    # (a2) NIVELADO DE TONO POR PARCHE (Opcion C) — antes de incrustar las texturas
    if TONE_LEVEL and texfiles and mtl2tex:
        try:
            if not tone_level(objf, texfiles, mtl2tex):
                log("TONO: no se aplico (la textura queda igual que antes)")
        except Exception as _tl:
            log("TONO: nivelado fallo (%s); la textura queda igual que antes" % _tl)
    elif not TONE_LEVEL:
        log("TONO: nivelado por parche APAGADO (OMVS_TONE=0)")
    # (b) cargar la Scene (texturas ya corregidas) y exportar SIN concatenar
    obj = trimesh.load(objf, process=False)
    obj.export(outglb)
    # (c) material MATE + UNLIT
    _patch_unlit_matte(outglb)
    return os.path.exists(outglb) and os.path.getsize(outglb) > 200000


# AUTO-SANADOR (2 configs):
#  cfg1 = la de la INVESTIGACION: caras virtuales coplanares (mata la
#         fragmentacion usando las paredes/piso ya aplanados) + nivelado local
#         y sharpen APAGADOS (los que escribian las bandas negras).
#  cfg2 = respaldo: la config EXACTA de v8.5 que ya funciono en produccion
#         (sin niveladores). Si el nivelado global crashara, caes a lo de hoy.
CFG1 = ["--virtual-face-images", str(VFACES),
        "--local-seam-leveling", str(LOCAL_SEAM),
        "--sharpness-weight", str(SHARP),
        "--global-seam-leveling", str(GLOBAL_SEAM)]
CFG2 = ["--virtual-face-images", str(VFACES),
        "--local-seam-leveling", "0",
        "--sharpness-weight", "0",
        "--global-seam-leveling", "0"]
CONFIGS = [(MAX_TEX, CFG1, OMP_HI), (MAX_TEX, CFG2, OMP_HI)]
final = None
for ci, (mt, extra, omp) in enumerate(CONFIGS):
    for f in glob.glob(BASE + ".*"):
        try: os.remove(f)
        except Exception: pass
    envt = dict(os.environ); envt["OMP_NUM_THREADS"] = str(omp)
    tag = "TextureMesh cfg%d (tex=%s %s omp=%s)" % (ci+1, mt, " ".join(extra), omp)
    rc = run(texcmd(mt, extra), tag, env=envt)
    objf = BASE + ".obj"
    if rc == 0 and os.path.exists(objf):
        try:
            if obj_to_glb(objf, OUTGLB):
                final = OUTGLB; break
            else:
                log("conversion OBJ->glb no produjo glb valido")
        except Exception as ce:
            log("conversion OBJ->glb fallo: %s" % ce)
    log("config %d no sirvio (rc=%d); %s"
        % (ci+1, rc, "reintento mas liviano" if ci+1 < len(CONFIGS) else "sin mas intentos"))

if final is None:
    log("ERROR: OpenMVS no produjo una textura utilizable (todas las configs)"); sys.exit(6)

log("TEXTURA OpenMVS lista: %.1f MB en %.1f min"
    % (os.path.getsize(OUTGLB)/1e6, (time.time()-t0)/60.0))
sys.exit(0)
'''


# ═════════════════════════════════════════════════════════════════════════
# BA_SCRIPT — PASO 2b: afinar poses con Bundle Adjustment (pycolmap).
# Las poses de MASt3R traen ~0.1° de error angular; ese error emborrona la
# textura al proyectar las fotos. Aquí: (1) SIFT en las fotos, (2) matching
# secuencial, (3) triangular puntos 3D con las poses MASt3R FIJAS, (4) Bundle
# Adjustment que afina poses+focal (centro óptico fijo: 2DGS lo exige),
# (5) VALIDAR y solo entonces escribir. Si algo no cuadra → exit 2 y el
# worker sigue con las poses MASt3R originales (respaldo en sparse/0_mast3r).
# ═════════════════════════════════════════════════════════════════════════
# VERTEXPAINT_SCRIPT — PASO 4c por defecto: pinta CADA VERTICE proyectando las
# fotos originales (misma matematica validada de la textura: espacio LINEAL,
# peso cos^4 a la mejor vista, oclusion por raycast, gamma 0.8, unlit), pero
# SIN el desdoblado UV de xatlas (que tardaba 25-90 min). Tarda ~1-3 min.
# Vertices que ninguna foto ve conservan su color del entrenamiento (TSDF).
# ═════════════════════════════════════════════════════════════════════════
VERTEXPAINT_SCRIPT = r'''
import sys, os, gc, json, struct
import numpy as np
from PIL import Image
import open3d as o3d
import trimesh

MESH_PLY   = sys.argv[1]
IMAGES_DIR = sys.argv[2]
SPARSE_DIR = sys.argv[3]
OUT_GLB    = sys.argv[4]
AO_PATH    = sys.argv[5] if len(sys.argv) > 5 else ""   # ambient occlusion por vertice

def log(s): print("   [paint] " + s, flush=True)

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def linear_to_srgb(c):
    c = np.maximum(c, 0.0)
    return np.clip(np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055), 0, 1)

# 1) malla (conserva el color TSDF como respaldo para vertices sin foto)
m = o3d.io.read_triangle_mesh(MESH_PLY)
V = np.asarray(m.vertices); F = np.asarray(m.triangles)
if len(V) == 0 or len(F) == 0:
    log("malla vacia, abortando"); sys.exit(1)
orig = np.asarray(m.vertex_colors) if len(m.vertex_colors) == len(V) else None
log("malla %d vert %d caras" % (len(V), len(F)))

# 2) escena de raycasting (visibilidad/oclusion, igual que la textura)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
INVALID = scene.INVALID_ID

# 3) intrinsecos + poses (parseo identico al validado)
cams = {}
for line in open(os.path.join(SPARSE_DIR, "cameras.txt")):
    if line.startswith("#") or not line.strip(): continue
    e = line.split()
    cams[int(e[0])] = (int(e[2]), int(e[3]), float(e[4]), float(e[5]), float(e[6]), float(e[7]))

def q2R(qw, qx, qy, qz):
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)]])

views = []
for line in open(os.path.join(SPARSE_DIR, "images.txt")):
    if line.startswith("#") or not line.strip(): continue
    e = line.split()
    if len(e) >= 10 and (e[9].endswith(".png") or e[9].endswith(".jpg")):
        qw, qx, qy, qz = map(float, e[1:5]); tx, ty, tz = map(float, e[5:8])
        R = q2R(qw, qx, qy, qz); t = np.array([tx, ty, tz])
        E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t
        views.append((int(e[8]), e[9], E, R, t))
log("poses: %d camaras" % len(views))

# 4) PINTAR: por cada foto, raycast y reparto del pixel a los 3 vertices del
#    triangulo golpeado (peso = baricentrica x cos^4 de la mejor vista)
accV  = np.zeros((len(V), 3), np.float64)
wsumV = np.zeros(len(V), np.float64)
Ktens = {}
nuse = 0
for cid, name, E, R, t in views:
    if cid not in cams: continue
    W, H, fx, fy, cx, cy = cams[cid]
    # PINTAR DESDE LAS FOTOS DE 12MP (originales) en vez de las de 1000px.
    # El color se ve borroso de cerca en parte por la resolución de la foto de
    # entrenamiento (1000px). Si existe la original (12MP), la usamos.
    # OJO: MASt3R RECORTA al centro las fotos de entrenamiento al aspecto W/H de
    # cameras.txt antes de escalarlas. Para NO desalinear, aplicamos EXACTAMENTE
    # el mismo recorte central a la foto original; tras el recorte, el aspecto
    # coincide y el escalado de intrínsecos (sx=Wp/W) ya es correcto.
    # PAINT_ORIG_DIR lo pasa el worker; si falta, cae a la de entrenamiento.
    path = os.path.join(IMAGES_DIR, name)
    _origdir = os.environ.get("PAINT_ORIG_DIR", "")
    _usa_orig = False
    if _origdir:
        _cand = os.path.join(_origdir, name)
        if os.path.exists(_cand):
            path = _cand; _usa_orig = True
    if not os.path.exists(path): continue
    _im = Image.open(path).convert("RGB")
    if _usa_orig:
        _Wo, _Ho = _im.size; _asp = W / float(H)
        if (_Wo / float(_Ho)) > _asp:
            _cw = int(round(_Ho * _asp)); _ch = _Ho
        else:
            _cw = _Wo; _ch = int(round(_Wo / _asp))
        _l = (_Wo - _cw) // 2; _tp = (_Ho - _ch) // 2
        _im = _im.crop((_l, _tp, _l + _cw, _tp + _ch))   # MISMO recorte que MASt3R
    photo = np.asarray(_im, np.float32) / 255.0
    photo = srgb_to_linear(photo)
    Hp, Wp = photo.shape[:2]
    sx = Wp / float(W); sy = Hp / float(H)
    key = (cid, Wp, Hp)
    if key not in Ktens:
        Ktens[key] = o3d.core.Tensor(np.array([[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]))
    rays = scene.create_rays_pinhole(Ktens[key], o3d.core.Tensor(E), Wp, Hp)
    ans = scene.cast_rays(rays)
    tri  = ans['primitive_ids'].numpy().astype(np.int64)
    bary = ans['primitive_uvs'].numpy().astype(np.float64)
    nrm  = ans['primitive_normals'].numpy().astype(np.float64)
    thit = ans['t_hit'].numpy()
    del rays, ans
    hit = np.isfinite(thit) & (tri != INVALID)
    if hit.sum() == 0:
        del photo; gc.collect(); continue
    yy, xx = np.where(hit)
    ti = tri[hit]; b1 = bary[hit][:, 0]; b2 = bary[hit][:, 1]; b0 = 1 - b1 - b2
    col = photo[yy, xx]
    P3 = V[F[ti]]
    pos = b0[:, None]*P3[:, 0] + b1[:, None]*P3[:, 1] + b2[:, None]*P3[:, 2]
    Ccam = -R.T @ t
    vd = Ccam[None, :] - pos
    vd /= (np.linalg.norm(vd, axis=1, keepdims=True) + 1e-9)
    nh = nrm[hit]; nh /= (np.linalg.norm(nh, axis=1, keepdims=True) + 1e-9)
    w = np.clip(np.abs((nh*vd).sum(1)), 0.05, 1.0) ** 4
    tris = F[ti]
    for k, bk in ((0, b0), (1, b1), (2, b2)):
        ww = w * bk
        np.add.at(accV,  tris[:, k], col * ww[:, None])
        np.add.at(wsumV, tris[:, k], ww)
    nuse += 1
    del photo, tri, bary, nrm, thit, hit, col; gc.collect()
log("proyectadas %d/%d camaras" % (nuse, len(views)))
if nuse == 0:
    log("ninguna camara proyectada, abortando"); sys.exit(1)

# 5) COLOR FINAL — cadena arreglada (el render salia "blanco"/lavado):
#    (a) color FIEL desde las fotos (promedio en LINEAL -> sRGB). El gamma 0.8 que
#        habia aqui INFLABA el brillo ~15% y empujaba todo hacia el blanco.
#    (b) SATURACION: promediar ~127 vistas LAVA el color (como un desenfoque de
#        color). Se recupera empujando la saturacion.
#    (c) AMBIENT OCCLUSION: sombras de contacto en rincones/juntas. El post-proceso
#        YA lo calculaba, pero este script lo BORRABA al sobrescribir el color ->
#        render plano sin profundidad. Ahora se vuelve a aplicar.
#    (d) gamma final leve, para compensar el oscurecimiento del AO.
#    TODO ajustable por entorno SIN tocar codigo:
#      PAINT_SAT  1.35 (1.0 = fiel a la foto; 1.5 = colores mas vivos)
#      PAINT_AO   0.40 (0 = sin sombras/plano; 0.55 = mas profundidad/mas oscuro)
#      PAINT_GAMMA 1.0 (0.9 = mas claro; 1.1 = mas oscuro). El 0.8 de antes
#                  QUEMABA el 27% de la superficie hacia el blanco = el "velo blanco".
_SAT   = float(os.environ.get("PAINT_SAT", "1.15"))
_AOSTR = float(os.environ.get("PAINT_AO", "0.40"))
_GAM   = float(os.environ.get("PAINT_GAMMA", "1.0"))
# ESPACIO DE COLOR DE SALIDA (arregla el "velo blanco" residual).
# La spec de glTF dice que el color por vértice COLOR_0 es LINEAL, no sRGB. Este
# pintor venía guardándolo en sRGB -> los visores correctos le aplicaban gamma
# OTRA VEZ = doble gamma = lavado. PAINT_STORE=linear (nuevo default) guarda en
# lineal como manda la spec: se ve fiel en gltf-viewer.donmccurdy.com (Linear) y
# en F3D --unlit. Si algún visor viejo lo necesita en sRGB, PAINT_STORE=srgb.
# Nota: como ya no hay doble-gamma que compensar, bajo PAINT_SAT de 1.35 a 1.15.
_STORE = os.environ.get("PAINT_STORE", "linear").lower()
painted = wsumV > 0
cols = np.zeros((len(V), 3), np.float32)
if orig is not None:
    cols[:] = orig.astype(np.float32)
else:
    cols[:] = 0.5
lin = (accV[painted] / wsumV[painted, None]).astype(np.float32)
cols[painted] = np.clip(linear_to_srgb(lin), 0, 1)          # (a) color FIEL
log("pintados %d/%d vertices (%.1f%%) desde las fotos" % (painted.sum(), len(V), 100.0*painted.mean()))
_b0 = float(cols.mean())
# (b) SATURACION alrededor de la luminancia (no cambia el brillo, solo la viveza)
if abs(_SAT - 1.0) > 1e-3:
    _lum = (cols * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(1, keepdims=True)
    cols = np.clip(_lum + (cols - _lum) * _SAT, 0, 1)
# (c) AMBIENT OCCLUSION (el paso que faltaba: devuelve profundidad y quita el "velo blanco")
_ao = None
if AO_PATH and os.path.exists(AO_PATH):
    try:
        _a = np.load(AO_PATH).astype(np.float32)
        if len(_a) == len(V) and np.isfinite(_a).all():
            _ao = np.clip(_a, 0.0, 1.0)
        else:
            log("AO ignorado: %d valores vs %d vertices" % (len(_a), len(V)))
    except Exception as _e:
        log("AO no cargado (%s)" % _e)
if _ao is not None:
    cols = cols * (1.0 - _AOSTR * _ao)[:, None]
    log("AO aplicado al color de las fotos (fuerza %.2f, oclusion media %.3f)" % (_AOSTR, float(_ao.mean())))
else:
    log("AVISO: sin AO -> el render puede verse plano/lavado")
# (d) gamma final
cols = np.clip(cols, 0, 1) ** _GAM
cols = np.nan_to_num(cols, nan=0.5, posinf=1.0, neginf=0.0)
cols = np.clip(cols, 0, 1).astype(np.float32)
log("color: brillo %.3f -> %.3f (sat %.2f, AO %.2f, gamma %.2f)" % (_b0, float(cols.mean()), _SAT, _AOSTR, _GAM))

# Guardamos el color COMO SE VE (sRGB, lo que muestra un visor fiel) para la
# auditoría automática de más abajo. La conversión a lineal es solo para el
# archivo; lo que Felipe ve en F3D --unlit es esta versión sRGB.
_cols_display = cols.copy()

# ESPACIO DE SALIDA: la spec de glTF exige COLOR_0 en LINEAL. Todo el ajuste
# (saturación, AO, gamma) se hizo en sRGB porque es perceptual; ahora, si
# PAINT_STORE=linear (default), convertimos a lineal para guardarlo como manda
# la spec. Así los visores correctos NO le vuelven a aplicar gamma (adiós doble
# gamma / velo blanco). PAINT_STORE=srgb mantiene el comportamiento viejo.
if _STORE == "linear":
    cols = np.clip(srgb_to_linear(cols), 0, 1).astype(np.float32)
    log("COLOR_0 guardado en LINEAL (spec glTF): evaluar en "
        "gltf-viewer.donmccurdy.com [Tone Mapping=Linear] o F3D --unlit")
else:
    log("COLOR_0 guardado en sRGB (modo compatible viejo)")

# 6) exportar .glb: color por vertice + normales suaves + mate + unlit
rgba = np.concatenate([(cols*255).astype(np.uint8),
                       np.full((len(V), 1), 255, np.uint8)], 1)
mesh_out = trimesh.Trimesh(vertices=V, faces=F, process=False)
mesh_out.visual = trimesh.visual.ColorVisuals(mesh_out, vertex_colors=rgba)
# Normales calculadas A MANO: los exportadores viejos de trimesh re-normalizan
# y dividen por cero con normales degeneradas, metiendo NaN literal al JSON del
# .glb (el error "Unexpected token N" del visor). Aqui NINGUNA fila queda en
# cero ni no-finita, en ninguna version de trimesh.
_fv = V[F]
_fn = np.cross(_fv[:, 1] - _fv[:, 0], _fv[:, 2] - _fv[:, 0])
vn = np.zeros((len(V), 3), np.float64)
for _k in range(3):
    np.add.at(vn, F[:, _k], _fn)
vn = np.nan_to_num(vn, nan=0.0, posinf=0.0, neginf=0.0)
_bad = np.linalg.norm(vn, axis=1) < 1e-12
vn[_bad] = (0.0, 0.0, 1.0)
vn /= np.linalg.norm(vn, axis=1, keepdims=True)
if _bad.any():
    log("normales degeneradas corregidas: %d (anti-NaN)" % int(_bad.sum()))
mesh_out.vertex_normals = vn
mesh_out.export(OUT_GLB)
try:
    _d = bytearray(open(OUT_GLB, "rb").read())
    _jlen = struct.unpack("<I", _d[12:16])[0]
    _g = json.loads(_d[20:20+_jlen].decode("utf-8"))
    _g.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in _g["extensionsUsed"]:
        _g["extensionsUsed"].append("KHR_materials_unlit")
    if not _g.get("materials"):
        _g["materials"] = [{}]
        for _mesh in _g.get("meshes", []):
            for _pr in _mesh.get("primitives", []):
                _pr["material"] = 0
    for _m in _g["materials"]:
        _pbr = _m.setdefault("pbrMetallicRoughness", {})
        _pbr["metallicFactor"] = 0.0
        _pbr["roughnessFactor"] = 1.0
        _m.setdefault("extensions", {})["KHR_materials_unlit"] = {}
    # ── SANEADOR ANTI-NaN (a prueba de cualquier version de trimesh) ──
    # Repara valores NaN/inf en los buffers float (NORMAL -> 0,0,1) y recalcula
    # los min/max REALES de cada accessor float. Un solo NaN en el JSON revienta
    # JSON.parse del visor ("Unexpected token N").
    _bin = bytearray(_d[20+_jlen:])   # incluye cabecera del chunk BIN (8 bytes)
    _attr = {}
    for _mesh in _g.get("meshes", []):
        for _pr in _mesh.get("primitives", []):
            for _an, _ai in _pr.get("attributes", {}).items():
                _attr[_ai] = _an
    _rep = 0
    _NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    for _ai, _acc in enumerate(_g.get("accessors", [])):
        _ncomp = _NC.get(_acc.get("type"), 0)
        if _acc.get("componentType") != 5126 or "bufferView" not in _acc or not _ncomp:
            continue
        _bv = _g["bufferViews"][_acc["bufferView"]]
        _off = 8 + _bv.get("byteOffset", 0) + _acc.get("byteOffset", 0)
        _nfl = _acc["count"] * _ncomp
        _arr = np.frombuffer(bytes(_bin[_off:_off + _nfl * 4]), np.float32)
        _arr = _arr.reshape(_acc["count"], _ncomp).copy()
        if not np.isfinite(_arr).all():
            if _attr.get(_ai) == "NORMAL" and _ncomp == 3:
                _mal = ~np.isfinite(_arr).all(axis=1)
                _arr[_mal] = (0.0, 0.0, 1.0)
                _rep += int(_mal.sum())
            else:
                _rep += int((~np.isfinite(_arr)).sum())
                _arr = np.nan_to_num(_arr, nan=0.0, posinf=0.0, neginf=0.0)
            _bin[_off:_off + _nfl * 4] = _arr.astype(np.float32).tobytes()
        if "min" in _acc or "max" in _acc or _attr.get(_ai) == "POSITION":
            _acc["min"] = [float(x) for x in _arr.min(0)]
            _acc["max"] = [float(x) for x in _arr.max(0)]
    if _rep:
        log("saneados %d valores NaN dentro del archivo" % _rep)
    # allow_nan=False = alarma: si algo no-finito sobreviviera, aqui explota
    _nj = json.dumps(_g, separators=(",", ":"), allow_nan=False).encode("utf-8")
    while len(_nj) % 4:
        _nj += b" "
    _out = bytearray()
    _out += _d[:12]
    _out += struct.pack("<I", len(_nj)) + b"JSON" + _nj
    _out += _bin
    _out[8:12] = struct.pack("<I", len(_out))
    open(OUT_GLB, "wb").write(bytes(_out))
    log("material unlit + mate + saneamiento anti-NaN aplicados")
except Exception as e:
    log("(patch/saneamiento fallo: %s)" % e)
# ════════════════════════════════════════════════════════════════════════
# AUDITORÍA AUTOMÁTICA — "ver" la malla con NÚMEROS, sin enviar archivos.
# Re-proyecta la malla YA pintada a las poses REALES de las cámaras y la
# compara contra las FOTOS reales. Dos números clave que van al log:
#   NITIDEZ = detalle(render) / detalle(foto)  -> mide el EFECTO DERRETIDO
#             (100% = igual de nítido que la foto; 30% = perdió 70% del detalle)
#   FIDELIDAD = PSNR de la malla final vs la realidad (distinto del PSNR de
#             entrenamiento, que solo mide las gaussianas, no la malla+color).
# Felipe pega el log y así yo "veo" objetivamente qué tan derretido está.
# (auditoria del log retirada en v8 a pedido de Felipe)

log("color por vertice desde FOTOS exportado a .glb")
sys.exit(0)
'''

# ═════════════════════════════════════════════════════════════════════════
BA_SCRIPT = r'''
import sys, os, shutil, time, subprocess, traceback
def log(s): print("   [ba] " + s, flush=True)

IMAGES = sys.argv[1]   # dataset/images (fotos de entrenamiento)
SPARSE = sys.argv[2]   # dataset/sparse/0 (modelo COLMAP texto de MASt3R)
WORKD  = sys.argv[3]   # carpeta de trabajo

# Usamos el binario colmap CLASICO que ya viene en la imagen (/usr/bin/colmap):
# es pre-"rigs", asi que entiende el modelo texto de MASt3R sin los chequeos
# internos nuevos de pycolmap 4.x que fallaron en el pod (RigId mismatch).
# pycolmap se usa SOLO para leer y validar (eso si funciono siempre).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
COLMAP = shutil.which("colmap")
if not COLMAP:
    log("colmap CLI no esta en la imagen: dejo poses MASt3R"); sys.exit(2)
try:
    import pycolmap
except Exception as e:
    log("pycolmap no disponible (%s): dejo poses MASt3R" % e); sys.exit(2)

def cli(args):
    r = subprocess.run([COLMAP] + args, capture_output=True, text=True)
    if r.returncode != 0:
        lineas = ((r.stderr or "") + "\n" + (r.stdout or "")).strip().splitlines()
        clave = [l for l in lineas if ("Check failed" in l or "ERROR" in l)][-3:]
        cola = clave + lineas[-4:]
        raise RuntimeError("colmap %s rc=%d :: %s" % (args[0], r.returncode, " | ".join(cola)))

os.makedirs(WORKD, exist_ok=True)
db = os.path.join(WORKD, "ba.db")
if os.path.exists(db): os.remove(db)
t0 = time.time()
try:
    rec_in = pycolmap.Reconstruction(SPARSE)
    n_in = rec_in.num_reg_images()
    log("modelo MASt3R: %d camaras, %d puntos" % (n_in, rec_in.num_points3D()))


    # 1) puntos SIFT (CPU, mismo binario que hara todo lo demas)
    cli(["feature_extractor", "--database_path", db, "--image_path", IMAGES,
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE",
         "--SiftExtraction.max_num_features", "4096", "--SiftExtraction.use_gpu", "0"])
    log("SIFT extraido (%.0fs)" % (time.time() - t0))

    # Modelo SOLO-POSES renumerado a los IDs de la BASE DE DATOS. Dos razones,
    # ambas fallos REALES del pod: (a) el colmap clasico aborta si el modelo trae
    # los 200k puntos de MASt3R; (b) tambien aborta si model.image_id no coincide
    # con db.image_id ("Check failed: existing_image.Name() ... img_0125 vs
    # img_0122"). Se mapea por NOMBRE de archivo, que es lo unico estable.
    import sqlite3
    _con = sqlite3.connect(db)
    _dbids = {n: int(i) for i, n in _con.execute("SELECT image_id, name FROM images")}
    _con.close()
    po = os.path.join(WORKD, "pose_only"); os.makedirs(po, exist_ok=True)
    shutil.copy2(os.path.join(SPARSE, "cameras.txt"), os.path.join(po, "cameras.txt"))
    _falt = 0
    with open(os.path.join(SPARSE, "images.txt")) as _f, \
         open(os.path.join(po, "images.txt"), "w") as _g:
        for _l in _f:
            _e = _l.split()
            if len(_e) >= 10 and (_e[9].endswith(".jpg") or _e[9].endswith(".png")):
                _nid = _dbids.get(_e[9])
                if _nid is None:
                    _falt += 1
                    continue
                _g.write(" ".join([str(_nid)] + _e[1:10]) + "\n\n")
    open(os.path.join(po, "points3D.txt"), "w").write("# vacio: solo poses\n")
    if _falt:
        log("VALIDACION FALLO: %d imagenes del modelo no estan en la BD -> dejo poses MASt3R" % _falt)
        sys.exit(2)

    # 2) matching secuencial (video: frames vecinos se solapan)
    cli(["sequential_matcher", "--database_path", db,
         "--SequentialMatching.overlap", "20",
         "--SequentialMatching.loop_detection", "0",
         "--SiftMatching.use_gpu", "0"])
    log("matching secuencial OK (%.0fs)" % (time.time() - t0))

    # 3) triangular con poses MASt3R FIJAS
    tri = os.path.join(WORKD, "tri"); os.makedirs(tri, exist_ok=True)
    cli(["point_triangulator", "--database_path", db, "--image_path", IMAGES,
         "--input_path", po, "--output_path", tri])

    # 4) Bundle Adjustment (afina poses + focal; centro optico FIJO)
    ba = os.path.join(WORKD, "ba_out"); os.makedirs(ba, exist_ok=True)
    # CLAVE DEL ARREGLO: refine_extrinsics=0 CONGELA las poses de MASt3R.
    # El BA solo mueve la focal para reducir el error de reproyeccion; NO puede
    # trasladar/rotar/estirar las camaras -> imposible deformar la escena.
    cli(["bundle_adjuster", "--input_path", tri, "--output_path", ba,
         "--BundleAdjustment.refine_focal_length", "1",
         "--BundleAdjustment.refine_principal_point", "0",
         "--BundleAdjustment.refine_extra_params", "0",
         "--BundleAdjustment.refine_extrinsics", "0"])
    txt = os.path.join(WORKD, "ba_txt"); os.makedirs(txt, exist_ok=True)
    cli(["model_converter", "--input_path", ba, "--output_path", txt,
         "--output_type", "TXT"])

    # 5) VALIDAR antes de tocar nada
    rec = pycolmap.Reconstruction(txt)
    err = rec.compute_mean_reprojection_error()
    npts = rec.num_points3D(); nreg = rec.num_reg_images()
    log("BA hecho: %d camaras, %d puntos, err %.2f px (%.0fs)" % (nreg, npts, err, time.time() - t0))
    if nreg != n_in:
        log("VALIDACION FALLO: %d/%d camaras registradas -> dejo poses MASt3R" % (nreg, n_in)); sys.exit(2)
    if npts < 5000:
        log("VALIDACION FALLO: solo %d puntos (<5000) -> dejo poses MASt3R" % npts); sys.exit(2)
    if err > 2.5:
        log("VALIDACION FALLO: error reproyeccion %.2f px (>2.5) -> dejo poses MASt3R" % err); sys.exit(2)
    # GUARDIAN DE ESCALA: el modo anclado NO deberia mover las camaras, pero por
    # seguridad medimos el bbox de los centros de camara antes/despues. Si la
    # escala cambio > 5%, algo se deformo -> rechazar y quedarse con MASt3R.
    import numpy as _np
    def _span(_rec):
        _c = _np.array([_img.projection_center() if hasattr(_img, "projection_center")
                        else (-_img.cam_from_world.rotation.matrix().T @ _img.cam_from_world.translation)
                        for _img in _rec.images.values()])
        return float(_np.linalg.norm(_c.max(0) - _c.min(0)))
    try:
        _s_in = _span(rec_in); _s_out = _span(rec)
        if _s_in > 1e-6:
            _ratio = _s_out / _s_in
            log("escala camaras: entrada %.3f -> BA %.3f (x%.3f)" % (_s_in, _s_out, _ratio))
            if _ratio > 1.05 or _ratio < 0.95:
                log("VALIDACION FALLO: el BA cambio la escala %.1f%% (>5%%) -> dejo poses MASt3R" % (abs(_ratio-1)*100)); sys.exit(2)
    except SystemExit:
        raise
    except Exception as _se:
        log("(guardian de escala no pudo medir: %s; sigo)" % _se)

    # 6) respaldo y escritura del modelo refinado (SOLO los 3 .txt clasicos,
    #    para que 2DGS y el script de priors lo lean igual que el de MASt3R)
    bak = os.path.join(os.path.dirname(SPARSE), "0_mast3r")
    if os.path.exists(bak): shutil.rmtree(bak)
    shutil.copytree(SPARSE, bak)
    for fn in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy2(os.path.join(txt, fn), os.path.join(SPARSE, fn))
    log("poses REFINADAS escritas en sparse/0 (respaldo: sparse/0_mast3r)")
    sys.exit(0)
except SystemExit:
    raise
except Exception as e:
    log("fallo inesperado: %s" % e)
    traceback.print_exc()
    sys.exit(2)
'''


# ═════════════════════════════════════════════════════════════════════════
# PRIORS_SCRIPT — PASO 2c: priors monoculares por foto.
# PROFUNDIDAD: Depth Anything V2 Metric-Indoor (vitb) → metros.
# NORMALES: DSINE (si su checkpoint está en la imagen); si no, fallback de
# normales-desde-profundidad (unproyectar con K + producto cruz).
# Guarda <foto>.npz con depth (H,W f16) y normal (3,H,W f16, espacio CÁMARA
# OpenCV apuntando HACIA la cámara). El train de 2DGS los usa para anclar
# techos/paredes lisas → menos huecos, techo continuo.
# Rutas sobreescribibles por entorno (para pruebas): DAV2_DIR, DSINE_DIR,
# MODELS_DIR, PRIORS_INPUT_SIZE.
# ═════════════════════════════════════════════════════════════════════════
PRIORS_SCRIPT = r'''
import sys, os, gc, traceback
import numpy as np
def log(s): print("   [priors] " + s, flush=True)

IMAGES = sys.argv[1]; SPARSE = sys.argv[2]; OUT = sys.argv[3]
DAV2_DIR   = os.environ.get("DAV2_DIR", "/opt/depth_anything_v2")
DSINE_DIR  = os.environ.get("DSINE_DIR", "/opt/dsine")
MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/models")
INSZ       = int(os.environ.get("PRIORS_INPUT_SIZE", "518"))
os.makedirs(OUT, exist_ok=True)

import torch
import cv2
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log("dispositivo: %s" % DEV)

# ---- leer camaras e imagenes del modelo COLMAP (texto) ----
cams = {}
with open(os.path.join(SPARSE, "cameras.txt")) as f:
    for ln in f:
        if ln.startswith("#") or not ln.strip(): continue
        e = ln.split(); cid = int(e[0]); mdl = e[1]
        W = int(e[2]); H = int(e[3]); p = [float(x) for x in e[4:]]
        if mdl == "PINHOLE": fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else: fx = fy = p[0]; cx, cy = p[1], p[2]
        cams[cid] = (W, H, fx, fy, cx, cy)
imgs = []
with open(os.path.join(SPARSE, "images.txt")) as f:
    raw = [l for l in f if not l.startswith("#")]
i = 0
while i < len(raw):
    ln = raw[i].strip()
    if ln:
        e = ln.split()
        if len(e) >= 10 and e[8].isdigit():
            imgs.append((e[9], int(e[8]))); i += 2; continue
    i += 1
if not imgs:
    log("no hay imagenes en images.txt"); sys.exit(1)
log("%d imagenes en el modelo" % len(imgs))

# ============ FASE 1: PROFUNDIDAD (Depth Anything V2 Metric-Indoor) ============
ck_d = os.path.join(MODELS_DIR, "depth_anything_v2_metric_hypersim_vitb.pth")
if not os.path.exists(ck_d):
    log("falta el checkpoint de profundidad %s" % ck_d); sys.exit(1)
sys.path.insert(0, os.path.join(DAV2_DIR, "metric_depth"))
from depth_anything_v2.dpt import DepthAnythingV2
md = DepthAnythingV2(encoder="vitb", features=128,
                     out_channels=[96, 192, 384, 768], max_depth=20.0)
md.load_state_dict(torch.load(ck_d, map_location="cpu"))
md = md.to(DEV).eval()
log("Depth Anything V2 (metric indoor) cargado")
depths = {}
with torch.no_grad():
    for k, (name, cid) in enumerate(imgs):
        bgr = cv2.imread(os.path.join(IMAGES, name))
        if bgr is None:
            log("no pude leer %s, la salto" % name); continue
        d = md.infer_image(bgr, input_size=INSZ)   # HxW float32 (metros)
        depths[name] = np.asarray(d, np.float16)
        if (k + 1) % 20 == 0 or (k + 1) == len(imgs):
            log("profundidad %d/%d" % (k + 1, len(imgs)))
if not depths:
    log("ninguna profundidad calculada"); sys.exit(1)
_d0 = depths[next(iter(depths))].astype(np.float32)
log("profundidad img0: %.2f..%.2f m" % (float(_d0.min()), float(_d0.max())))
del md; gc.collect()
if DEV == "cuda": torch.cuda.empty_cache()

# ============ FASE 2: NORMALES (DSINE; fallback desde profundidad) ============
def normal_desde_profundidad(d32, fx, fy, cx, cy):
    H, W = d32.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    X = (xs - cx) / fx * d32; Y = (ys - cy) / fy * d32; Z = d32
    P = np.stack([X, Y, Z], 0)                        # 3,H,W espacio camara OpenCV
    dPy = np.stack([np.gradient(P[c], axis=0) for c in range(3)], 0)
    dPx = np.stack([np.gradient(P[c], axis=1) for c in range(3)], 0)
    n = np.cross(dPy.reshape(3, -1).T, dPx.reshape(3, -1).T).T.reshape(3, H, W)
    nn = np.linalg.norm(n, axis=0, keepdims=True)
    n = n / np.maximum(nn, 1e-8)
    flip = (n * P).sum(0) > 0                         # que apunte HACIA la camara
    n[:, flip] = -n[:, flip]
    return n.astype(np.float16)

dsine = None
ck_n = os.path.join(MODELS_DIR, "dsine.pt")
if os.path.exists(ck_n):
    try:
        sys.path.insert(0, DSINE_DIR)   # PRIMERO: sus 'models'/'utils' ganan a 2DGS
        import geffnet
        _og = geffnet.create_model
        geffnet.create_model = lambda *a, **k: _og(*a, **{**k, "pretrained": False})
        from models.dsine import DSINE as _DSINE
        import utils.utils as _du
        import torch.nn.functional as _F
        from torchvision import transforms as _T
        dsine = _DSINE()
        _sd = torch.load(ck_n, map_location="cpu")
        _sd = _sd.get("model", _sd) if isinstance(_sd, dict) else _sd
        try:
            dsine.load_state_dict(_sd)
        except Exception:
            dsine.load_state_dict(_sd, strict=False)
            log("DSINE: pesos cargados con strict=False")
        dsine = dsine.to(DEV).eval()
        try: dsine.pixel_coords = dsine.pixel_coords.to(DEV)
        except Exception: pass
        _norm = _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        log("DSINE cargado (normales de alta calidad)")
    except Exception as e:
        log("DSINE no cargo (%s): usare normales-desde-profundidad" % e)
        traceback.print_exc()
        dsine = None
else:
    log("dsine.pt no esta en la imagen: usare normales-desde-profundidad")

nok = 0
with torch.no_grad():
    for k, (name, cid) in enumerate(imgs):
        if name not in depths or cid not in cams: continue
        W, H, fx, fy, cx, cy = cams[cid]
        d16 = depths[name]
        normal = None
        if dsine is not None:
            try:
                bgr = cv2.imread(os.path.join(IMAGES, name))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEV)
                _, _, Hi, Wi = t.shape
                pl, pr, pt, pb = _du.pad_input(Hi, Wi)
                t = _F.pad(t, (pl, pr, pt, pb), mode="constant", value=0.0)
                t = _norm(t)
                K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                                 dtype=torch.float32, device=DEV).unsqueeze(0)
                K[:, 0, 2] += pl; K[:, 1, 2] += pt
                out = dsine(t, intrins=K)[-1]
                n = out[:, :3, pt:pt + Hi, pl:pl + Wi]
                n = torch.nn.functional.normalize(n, dim=1)
                normal = n[0].float().cpu().numpy().astype(np.float16)
            except Exception as e:
                log("DSINE fallo en %s (%s): fallback" % (name, e))
                normal = None
        if normal is None:
            normal = normal_desde_profundidad(d16.astype(np.float32), fx, fy, cx, cy)
        base = os.path.splitext(name)[0]
        np.savez_compressed(os.path.join(OUT, base + ".npz"),
                            depth=d16, normal=normal)
        nok += 1
        if (k + 1) % 20 == 0 or (k + 1) == len(imgs):
            log("normales %d/%d" % (k + 1, len(imgs)))
log("LISTO: %d priors guardados" % nok)
sys.exit(0 if nok > 0 else 1)
'''


# ═════════════════════════════════════════════════════════════════════════
# Parche de PRIORS al train.py de 2DGS (validado con test matemático local):
# PRIOR_UTILS se inyecta tras "import uuid"; PRIOR_LOSS reemplaza la línea del
# total_loss. Añade: L_depth (alineación escala+desplaz. por mínimos cuadrados,
# estilo MonoSDF/DN-Splatter, con guardián s>0) y L_normal (1−coseno, con la
# MISMA transformación cámara→mundo del renderer de 2DGS). Todo se controla
# por entorno: MONO_PRIORS_DIR (si está vacío, NO hace nada), MONO_LAMBDA_DEPTH
# (0.2), MONO_LAMBDA_NORMAL (0.1), MONO_FROM_ITER (100).
# ═════════════════════════════════════════════════════════════════════════
TRAIN_ANCHOR = "        total_loss = loss + dist_loss + normal_loss\n"

PRIOR_UTILS = r'''
# ======= PRIORS MONOCULARES (inyectado por el worker; DN-Splatter style) =======
import numpy as _np
_PRIORS_DIR = os.environ.get("MONO_PRIORS_DIR", "")
# Profundidad monocular EN 0.2 (valor del render b02d2d8c que preservaba la
# estructura del cuarto). El experimento de apagarla (0.0) quedó confundido con
# la deformación del BA, así que se vuelve al valor conocido-bueno. Ancla la
# geometría métrica y densifica el techo. Las estrías se atacan por extracción
# (depth_ratio=1), no aquí.
_L_DEPTH = float(os.environ.get("MONO_LAMBDA_DEPTH", "0.2"))
_L_NORM  = float(os.environ.get("MONO_LAMBDA_NORMAL", "0.1"))
_P_FROM  = int(os.environ.get("MONO_FROM_ITER", "100"))
_prior_cache = {}
def _get_prior(name):
    if not _PRIORS_DIR: return None
    if name in _prior_cache: return _prior_cache[name]
    p = os.path.join(_PRIORS_DIR, name + ".npz")
    if not os.path.exists(p):
        _prior_cache[name] = None; return None
    try:
        z = _np.load(p)
        d = torch.from_numpy(z["depth"].astype(_np.float32))
        n = torch.from_numpy(z["normal"].astype(_np.float32))
        _prior_cache[name] = (d, n)
    except Exception:
        _prior_cache[name] = None
    return _prior_cache[name]

def _mono_losses(viewpoint_cam, render_pkg):
    pr = _get_prior(viewpoint_cam.image_name)
    if pr is None:
        return None
    d_mono, n_mono = pr
    sd_full = render_pkg["surf_depth"]
    dev = sd_full.device
    H = viewpoint_cam.image_height; W = viewpoint_cam.image_width
    d_mono = d_mono.to(dev); n_mono = n_mono.to(dev)
    if d_mono.shape[0] != H or d_mono.shape[1] != W:
        d_mono = torch.nn.functional.interpolate(d_mono[None, None], (H, W), mode="bilinear", align_corners=False)[0, 0]
        n_mono = torch.nn.functional.interpolate(n_mono[None], (H, W), mode="bilinear", align_corners=False)[0]
        n_mono = torch.nn.functional.normalize(n_mono, dim=0, eps=1e-6)
    # --- profundidad: alinear escala+desplazamiento (minimos cuadrados, estilo MonoSDF) ---
    sd = sd_full[0]
    alpha = render_pkg["rend_alpha"][0].detach()
    m = (alpha > 0.5) & (d_mono > 1e-4) & torch.isfinite(sd.detach()) & (sd.detach() > 1e-4)
    L_d = sd.new_tensor(0.0)
    if m.sum() > 500:
        x = d_mono[m]; y = sd[m].detach()
        mx = x.mean(); my = y.mean()
        vx = ((x - mx) * (x - mx)).mean()
        cov = ((x - mx) * (y - my)).mean()
        s = cov / (vx + 1e-8); t = my - s * mx
        if torch.isfinite(s) and s > 1e-4:
            d_al = (s * d_mono + t).detach()
            L_d = torch.abs(sd - d_al)[m].mean()
    # --- normales: prior (espacio de camara) -> mundo, coseno vs rend_normal ---
    n_world = (n_mono.permute(1, 2, 0) @ (viewpoint_cam.world_view_transform[:3, :3].T)).permute(2, 0, 1)
    rn = render_pkg["rend_normal"]
    cosine = (rn * n_world).sum(dim=0)
    L_n = ((1.0 - cosine) * m.float()).sum() / (m.float().sum() + 1e-6)
    return L_d, L_n
# ======= fin priors =======
'''

PRIOR_LOSS = r'''        mono_loss = 0.0
        if _PRIORS_DIR and iteration >= _P_FROM:
            _ml = _mono_losses(viewpoint_cam, render_pkg)
            if _ml is not None:
                mono_loss = _L_DEPTH * _ml[0] + _L_NORM * _ml[1]
        total_loss = loss + dist_loss + normal_loss + mono_loss
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
    """Manda un callback firmado al backend (progress/completed/error).
    SIEMPRE incluye pod_id: es lo único que permite al backend apagar este pod
    aunque haya perdido su base de datos."""
    if not CALLBACK_URL:
        return
    payload = {"type": tipo, "pod_id": POD_ID, **datos}
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

def run(cmd, cwd=None, env=None, fase_label=None, check=True, timeout=None):
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
            if timeout and (time.time() - t0) > timeout:
                try:
                    proc.kill(); proc.wait(timeout=15)
                except Exception:
                    pass
                log(f"   ⏱ paso cortado a los {int(timeout/60)} min (limite de seguridad); sigo con el respaldo")
                break
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
    # ═══════════════════════════════════════════════════════════════════════
    # SEGURO ANTI-BUCLE (el que te quemó crédito toda una noche)
    # ───────────────────────────────────────────────────────────────────────
    # Cuando el worker termina, el proceso principal del contenedor se acaba y
    # RUNPOD LO RELANZA AUTOMÁTICAMENTE. Lo normal es que el backend apague el
    # pod al recibir el aviso de "completed"... pero si ese aviso falla (404 por
    # BD borrada, red, Railway reiniciando), NADIE lo apaga: RunPod lo relanza y
    # el worker VUELVE A RENDIR 35 MINUTOS. Y otra vez. Y otra. Pasó de verdad.
    # ARREGLO: al terminar dejamos una MARCA en /workspace (que es el volumen del
    # pod y SOBREVIVE al reinicio del contenedor). Si al arrancar la marca ya
    # existe, este job YA SE HIZO: NO se vuelve a rendir. Solo se reintenta el
    # aviso al backend (por si aquella vez falló) y se sale.
    # ═══════════════════════════════════════════════════════════════════════
    marca = Path("/workspace") / f"HECHO_{TOUR_ID}.txt"
    if marca.exists():
        log("═" * 60)
        log(f"⚠ ESTE JOB YA SE RENDERIZÓ ({TOUR_ID}). NO lo repito.")
        log("  (RunPod relanzó el contenedor porque nadie apagó el pod.)")
        log("  Reintento avisarle al backend para que lo apague...")
        log("═" * 60)
        try:
            info = json.loads(marca.read_text())
        except Exception:
            info = {}
        _estado["vivo"] = False
        callback("completed",
                 frames_used=info.get("frames_used", 0),
                 ply_mb=info.get("ply_mb", 0),
                 seconds=info.get("seconds", 0),
                 log="\n".join(_LOG))
        log("Aviso reenviado. Si el pod sigue encendido, APÁGALO EN runpod.io.")
        time.sleep(20)          # dar tiempo a que el backend lo apague
        sys.exit(0)
    try:
        try:
            _img_tag = Path("/opt/IMAGE_TAG").read_text().strip()
        except Exception:
            _img_tag = "v3-o-v4-vieja (sin marcador)"
        _bn_pr = "priorsOFF" if os.environ.get("MONO_PRIORS", "0") != "1" else "priorsON"
        _bn_sm = os.environ.get("SMOOTH_MODE", "twostep")
        _bn_sn = "snapON" if os.environ.get("PLANE_SNAP", "1") == "1" else "snapOFF"
        _bn_tr = int(os.environ.get("MESH_TRIS", "2500000")) // 1000
        _bn_st = os.environ.get("PAINT_STORE", "linear")
        _bn_au = "audit" if os.environ.get("AUDIT","1")=="1" else "noaudit"
        _bn_uv = "uv" if os.environ.get("UV_TEXTURE","1")=="1" else "noUV"
        log(f"═══ render-gs-worker 2DGS · v8-{_bn_pr}-{_bn_sm}-{_bn_sn}-{_bn_tr}k-{_bn_st}-openmvs"
            f" · imagen {_img_tag} · job {TOUR_ID} · calidad {QUALITY} ({ITERS} iter) ═══")

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
        # CLAVE: lo corremos DESDE /opt/mast3r (cwd) y con PYTHONPATH explícito.
        # En el build, MASt3R se importaba bien porque la carpeta de trabajo era
        # /opt/mast3r; al correr el script desde /workspace/job, el sys.path.insert
        # del script no bastaba (MASt3R reconfigura rutas de forma especial).
        # Replicar cwd=/opt/mast3r + PYTHONPATH garantiza que encuentre el paquete.
        mast3r_py = WORK / "mast3r_sfm.py"
        mast3r_py.write_text(MAST3R_SCRIPT)
        env_mast3r = dict(os.environ)
        env_mast3r["PYTHONPATH"] = "/opt/mast3r:/opt/mast3r/dust3r:/opt/2dgs"
        run(["python", str(mast3r_py), str(images_dir), str(dataset)],
            cwd="/opt/mast3r",
            env=env_mast3r,
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

        # ── PASO 2b: afinar poses con Bundle Adjustment (pycolmap) ──
        # MASt3R deja un error de pose pequeño (~0.1°) que emborrona la textura
        # al promediar vistas. Re-triangulamos puntos SIFT manteniendo las poses
        # de MASt3R fijas y luego un BA clásico las pule. Si pycolmap no está o
        # el resultado no pasa las validaciones, el script sale con código 2 y
        # seguimos con las poses originales (el render NO se pierde por esto).
        # BA APAGADO por defecto (POSE_BA=0). Evidencia de PRODUCCIÓN: el bundle
        # adjustment deformó el cuarto en TODAS las corridas donde llegó a aplicarse
        # (suelto: cuarto estirado 9x6x7m; incluso en modo anclado la estructura se
        # dañó con las 127 cámaras reales). El render b02d2d8c que SÍ preservó la
        # estructura tenía el BA sin aplicar. Para GARANTIZAR que la estructura no se
        # deforme -prioridad #1 de Felipe- se apaga. NO se borra: reactivable con
        # POSE_BA=1 para revisarlo con cuidado cuando el cuarto salga sólido.
        if os.environ.get("POSE_BA", "0") == "1":
            fase(0.40, "PASO 2b/5 — Afinando poses (bundle adjustment)")
            ba_py = WORK / "pose_ba.py"
            ba_py.write_text(BA_SCRIPT)
            _rc_ba, _ = run(["python", str(ba_py), str(dataset / "images"),
                             str(dataset / "sparse" / "0"), str(WORK / "ba_work")],
                            check=False)
            if _rc_ba == 0:
                log("   ✓ poses REFINADAS con BA (respaldo MASt3R en sparse/0_mast3r)")
            else:
                log(f"   BA no aplicado (rc={_rc_ba}): sigo con las poses MASt3R")
        else:
            log("   PASO 2b saltado (POSE_BA=0)")

        # ── PASO 2c: priors monoculares — APAGADOS (causa de las LÁMINAS) ──
        # EVIDENCIA DURA de los logs: al activar los priors, la malla cruda pasó de
        # 17,949 a 116,896 PEDAZOS SUELTOS (6.5x), las gaussianas de 922k a 1.96M
        # (2.1x) y el error de orientación de los surfels de 12° a 45° (13x peor).
        # Y la queja de "láminas/branquias" apareció EXACTAMENTE en ese momento
        # (nunca antes en todo el proyecto).
        # MECANISMO: la profundidad monocular se alinea (escala+desplazamiento) FOTO
        # POR FOTO. Cada foto pide la misma pared a una distancia distinta; para
        # complacerlas a todas, el entrenamiento CONSTRUYE UNA CAPA POR VERSIÓN ->
        # capas apiladas = las estrías. Está documentado (MonoFusion: "duplicated
        # object parts" por la escala-shift por vista).
        # Se APAGAN (MONO_PRIORS=0). Apagar una pérdida NO puede deformar el cuarto
        # (solo quita una restricción; la forma viene de las poses MASt3R + fotos).
        # Riesgo conocido: sin la profundidad, el techo liso puede volver a tener
        # algún hueco -> lo compensa sdf_trunc 5x (banda ancha que rellena).
        # Reactivable con MONO_PRIORS=1.
        if os.environ.get("MONO_PRIORS", "0") == "1":
            fase(0.42, "PASO 2c/5 — Priors monoculares (profundidad+normales)")
            pri_py = WORK / "make_priors.py"
            pri_py.write_text(PRIORS_SCRIPT)
            priors_dir = dataset / "priors"
            _rc_pr, _ = run(["python", str(pri_py), str(dataset / "images"),
                             str(dataset / "sparse" / "0"), str(priors_dir)],
                            check=False)
            _n_npz = len(list(priors_dir.glob("*.npz"))) if priors_dir.exists() else 0
            if _rc_pr == 0 and _n_npz > 0:
                os.environ["MONO_PRIORS_DIR"] = str(priors_dir)
                log(f"   ✓ {_n_npz} priors listos (se usarán en el entrenamiento)")
            else:
                log(f"   priors no disponibles (rc={_rc_pr}, n={_n_npz}): "
                    "entreno sin priors como hasta ahora")
        else:
            log("   PASO 2c saltado (MONO_PRIORS=0)")

        # ── PARCHE matplotlib en 2DGS ──
        # 2DGS usa fig.canvas.tostring_rgb() en su función colormap(), pero
        # matplotlib 3.8+ ELIMINÓ ese método (ahora es buffer_rgba). Esa función
        # solo genera una imagen de diagnóstico para TensorBoard en la iteración
        # de test (7000), pero su ausencia hace CRASHEAR todo el entrenamiento.
        # Parcheamos el archivo de 2DGS en caliente (la imagen trae un matplotlib
        # nuevo). Es un reemplazo de 2 líneas: tostring_rgb()->buffer_rgba() y el
        # reshape a 4 canales (RGBA) recortando el alfa -> RGB.
        try:
            gu_path = Path("/opt/2dgs/utils/general_utils.py")
            txt = gu_path.read_text()
            if "tostring_rgb()" in txt:
                txt = txt.replace("fig.canvas.tostring_rgb()",
                                  "fig.canvas.buffer_rgba()")
                txt = txt.replace("get_width_height()[::-1] + (3,))",
                                  "get_width_height()[::-1] + (4,))[:, :, :3]")
                gu_path.write_text(txt)
                log("   parche matplotlib aplicado a 2DGS (tostring_rgb→buffer_rgba)")
        except Exception as e:
            log(f"   (no se pudo parchear general_utils: {e})")

        # ── PARCHE de SEMILLA en 2DGS (reproducibilidad) ──
        # Fijamos la semilla aleatoria al inicio de train.py para que el entrenamiento
        # sea reproducible (misma entrada → misma malla). Esto, junto con bajar
        # lambda_dist, elimina el problema de que una corrida salía buena y la siguiente
        # fatal. HONESTIDAD: el rasterizador CUDA de 2DGS usa sumas atómicas que no son
        # 100% deterministas, así que reduce MUCHO la varianza pero no del todo; por eso
        # más abajo añadimos un chequeo de PSNR que avisa si la corrida salió mal.
        try:
            tp = Path("/opt/2dgs/train.py")
            tptxt = tp.read_text()
            if "manual_seed(42)" not in tptxt:
                seed_code = (
                    "import random as _sr, numpy as _snp, torch as _st\n"
                    "_sr.seed(42); _snp.random.seed(42); _st.manual_seed(42)\n"
                    "try:\n    _st.cuda.manual_seed_all(42)\nexcept Exception:\n    pass\n")
                if tptxt.lstrip().startswith("from __future__"):
                    _i = tptxt.index("\n") + 1   # 'from __future__' debe ir primero
                    tp.write_text(tptxt[:_i] + seed_code + tptxt[_i:])
                else:
                    tp.write_text(seed_code + tptxt)
                log("   semilla fija inyectada en train.py (reproducibilidad)")
        except Exception as e:
            log(f"   (no se pudo inyectar semilla en train.py: {e})")

        # ── PARCHE de PRIORS MONOCULARES en train.py de 2DGS ──
        # Inyecta (1) las utilidades que cargan los .npz del PASO 2c y (2) las dos
        # pérdidas nuevas (profundidad alineada por escala + normales) justo donde
        # 2DGS suma su pérdida total. El parche solo ACTÚA en runtime si
        # MONO_PRIORS_DIR está definido (o sea, si el PASO 2c dejó priors listos);
        # si no hay priors, train.py se comporta exactamente igual que antes.
        try:
            tp = Path("/opt/2dgs/train.py")
            tptxt = tp.read_text()
            if "_mono_losses" in tptxt:
                log("   parche de priors ya presente en train.py")
            elif "import uuid" in tptxt and TRAIN_ANCHOR in tptxt:
                tptxt = tptxt.replace("import uuid", "import uuid" + PRIOR_UTILS, 1)
                tptxt = tptxt.replace(TRAIN_ANCHOR, PRIOR_LOSS, 1)
                tp.write_text(tptxt)
                log("   parche de priors monoculares inyectado en train.py")
            else:
                log("   AVISO: no encontré las anclas en train.py — entreno SIN priors")
        except Exception as e:
            log(f"   (no se pudo parchear priors en train.py: {e})")

        # ── PASO 3: entrenar 2DGS ──
        fase(0.45, f"PASO 3/5 — Entrenando 2DGS ({ITERS} iter)")
        dgs_out = WORK / "output"; dgs_out.mkdir(exist_ok=True)
        # --lambda_dist : regularizador de DISTORSIÓN. En TEORÍA (paper 2DGS) subirlo
        # de 25 a 100-1000 debería consolidar las láminas en una superficie. Se probó
        # en PRODUCCIÓN y el resultado fue INEQUÍVOCO: con 100 la malla se DEFORMÓ (el
        # cuarto perdió su forma). La escala métrica de MASt3R hace que 100 sea
        # demasiado y colapse la geometría. LECCIÓN: en ESTE pipeline, 25 es el valor
        # que preserva la estructura; subirlo la rompe. Se vuelve a 25 (estructura
        # intacta como en el render b02d2d8c). Las estrías se atacan por la vía SEGURA
        # (extracción de malla: depth_ratio=1), no tocando la geometría entrenada.
        # ── VELOCIDAD: dónde viven las imágenes de entrenamiento ──
        # "cpu" = las 127 fotos viven en RAM y se COPIAN a la GPU en CADA una de las
        # 30.000 iteraciones. "cuda" = viven en la VRAM y no se copian nunca.
        # Es la MISMA matemática y el MISMO resultado (ni un pixel cambia): solo
        # cambia DÓNDE están los datos. Cuesta ~1.2 GB de VRAM (127 fotos de
        # 1000x750). Se activa solo si la GPU tiene ≥20 GB (4090=24, A6000=48), y
        # si el entrenamiento fallara por memoria, REINTENTA solo con cpu (red de
        # seguridad: el render no se pierde). Forzable con DATA_DEVICE=cpu/cuda.
        _dev = os.environ.get("DATA_DEVICE", "")
        if not _dev:
            _dev = "cpu"
            try:
                _r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                                     "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=20)
                _vram = int(_r.stdout.strip().splitlines()[0])
                if _vram >= 20000:
                    _dev = "cuda"
                log(f"   VRAM {_vram/1000:.0f} GB → data_device={_dev}"
                    f"{' (imágenes en VRAM: sin copiar 30.000 veces)' if _dev=='cuda' else ' (VRAM justa: modo seguro)'}")
            except Exception as _e:
                log(f"   (no pude leer la VRAM: {_e}) → data_device=cpu (modo seguro)")
        _LAMBDA_DIST = os.environ.get("LAMBDA_DIST", "25")
        log(f"   lambda_dist = {_LAMBDA_DIST} (25 preserva estructura; 100 deformaba)")
        def _entrenar(_dd):
            return run(["python", "/opt/2dgs/train.py",
                 "-s", str(dataset), "-m", str(dgs_out),
                 "--iterations", str(ITERS),
                 "--lambda_dist", _LAMBDA_DIST,
                 "--lambda_normal", "0.05",
                 "-r", "1",                 # resolución COMPLETA (1000px), no reducir
                 "--data_device", _dd],
                fase_label="PASO 3/5 — Entrenando 2DGS", check=False)
        _t_tr = time.time()
        _rc_tr, _out_tr = _entrenar(_dev)
        if _rc_tr != 0 and _dev == "cuda":
            log("   ⚠ el entrenamiento falló en VRAM (probable falta de memoria); REINTENTO en modo seguro (cpu)")
            _rc_tr, _out_tr = _entrenar("cpu")
        if _rc_tr != 0:
            raise RuntimeError(f"El entrenamiento 2DGS falló (código {_rc_tr}).")
        log(f"   2DGS entrenado en {(time.time()-_t_tr)/60:.1f} min (data_device={_dev})")
        # ── CHEQUEO DE CALIDAD (PSNR) — red de seguridad ──
        # La investigación mostró que el entrenamiento puede salir mal e inestable.
        # Leemos el PSNR final del log de 2DGS y avisamos si salió bajo (< 30): en ese
        # caso la malla probablemente saldrá dañada/incompleta y conviene re-correr.
        psnr_final = None
        try:
            import re as _re
            _psnrs = _re.findall(r'PSNR\s+([0-9]+\.[0-9]+)', _out_tr or "")
            if _psnrs:
                psnr_final = float(_psnrs[-1])
                # Con priors activos el PSNR baja 1-3 puntos y ES NORMAL: se
                # cambia un poco de fidelidad fotografica por geometria solida
                # (paredes/techo). El umbral de alarma baja de 30 a 28.
                _con_priors = bool(os.environ.get("MONO_PRIORS_DIR"))
                _umbral = 28.0 if _con_priors else 30.0
                if psnr_final >= _umbral:
                    _nota = " (con priors; 1-3 pts menos que sin priors es normal)" if _con_priors else " (buena base estable)"
                    log(f"   ✓ CALIDAD OK: PSNR final {psnr_final:.1f}{_nota}")
                else:
                    log(f"   ⚠⚠⚠ CALIDAD BAJA: PSNR final {psnr_final:.1f} (< {_umbral:.0f}). La malla "
                        f"puede salir dañada/incompleta. RECOMIENDO RE-CORRER el render.")
        except Exception as e:
            log(f"   (no se pudo leer el PSNR: {e})")

        # ── PASO 4: extraer malla por TSDF (OPTIMIZADO) ──
        fase(0.80, "PASO 4/5 — Extrayendo malla (TSDF)")
        # GANANCIA GRANDE de velocidad: el TSDF de Open3D corre en CPU y, en pods
        # con muchos vCPU, abre demasiados hilos y se vuelve LENTÍSIMO (28 min en
        # la prueba anterior, ~2.87 s por vista). Limitando OMP a 8 hilos, baja a
        # ~2-3 min sin cambiar el algoritmo.
        env_mesh = dict(os.environ)
        env_mesh["OMP_NUM_THREADS"] = "8"

        # ── ESCALA DE LA ESCENA (robusta a la escala de MASt3R) ──
        # Medimos el tamaño real del cuarto desde la nube de puntos de MASt3R y
        # derivamos los parámetros del TSDF en proporción. Así funcionan igual
        # aunque MASt3R entregue una escala distinta entre escenas.
        import numpy as _np
        _ext = _np.array([8.0, 6.0, 8.0])   # valor por defecto si falla la medición
        try:
            _pts = []
            with open(dataset / "sparse" / "0" / "points3D.txt") as _f:
                for _l in _f:
                    if _l.startswith("#") or not _l.strip():
                        continue
                    _p = _l.split()
                    if len(_p) >= 4:
                        _pts.append((float(_p[1]), float(_p[2]), float(_p[3])))
            _pts = _np.asarray(_pts)
            # percentiles 2-98 -> ignora floaters al medir el tamaño del cuarto
            _lo = _np.percentile(_pts, 2, axis=0)
            _hi = _np.percentile(_pts, 98, axis=0)
            _ext = _hi - _lo
        except Exception as e:
            log(f"   (no se midió la escala, uso valores por defecto: {e})")
        _diag = float(_np.linalg.norm(_ext))
        _maxext = float(_ext.max())
        # ~500 voxeles en la dimensión mayor (≈1cm en este cuarto). PROBADO que la
        # malla resultante SÍ carga en 3dviewer.net. Se intentó /800 (~6mm) para más
        # detalle pero generaba una malla DEMASIADO densa/fragmentada (5.96M triángulos,
        # 2539 pedazos) → la decimación agresiva dejaba triángulos degenerados y valores
        # NaN → el visor se colgaba ("cargando para siempre"). El camino del voxel fino
        # choca con un muro de visualización; el detalle vendrá por TEXTURA UV (no añade
        # triángulos, no rompe el visor). FAIL-SAFE: este /500 es la base que carga.
        voxel = max(_maxext / 500.0, 0.005)
        sdf_trunc = 5.0 * voxel          # banda ~5 voxeles (antes 4): cierra mejor los HUECOS
        #                                  en zonas de poca observación, a cambio de redondear
        #                                  un poquito los detalles finos (compromiso aceptable).
        depth_trunc = _diag * 1.3        # cubre el cuarto + margen; corta agujas lejanas
        log(f"   escala medida: cuarto≈{_ext[0]:.2f}×{_ext[1]:.2f}×{_ext[2]:.2f}, "
            f"voxel={voxel:.4f}, sdf_trunc={sdf_trunc:.4f}, depth_trunc={depth_trunc:.2f}")

        # ── EXTRACCIÓN EN MODO BOUNDED (CORRECTO para un cuarto cerrado) ──
        # CAMBIOS (2ª investigación) para COMPLETITUD sin perder la malla única:
        #   - --depth_ratio 0 (profundidad MEDIA, no mediana): la mediana descartaba
        #     superficies de poca confianza (techo liso, zonas de poco solape) ->
        #     huecos y TECHO FALTANTE. La media integra lo que haya -> más completa.
        #   - --sdf_trunc 4x voxel (antes 2x): banda más gruesa que FUNDE mejor las
        #     superficies entre fotos -> rellena huecos y cierra el techo. (5x es el
        #     default de 2DGS; usamos 4x como equilibrio para no re-fundir doble capa.)
        #   - --voxel más fino (/500): más detalle y mejor relleno.
        #   - --depth_trunc acotado: recorta las AGUJAS de las ventanas (vidrio).
        #   - --num_cluster 50: conserva techo + muebles aunque queden como islas
        #     separadas; los floaters diminutos los quita el post-proceso por TAMAÑO.
        # NOTA: el modo bounded (no unbounded) sigue evitando la doble cáscara que
        # daba z-fighting. Si el DIAG vuelve a mostrar dos componentes ~50%, bajar
        # sdf_trunc a 3x. El suavizado Taubin del post-proceso limpia el ruido de
        # usar media en vez de mediana.
        # ── depth_ratio: VUELTA A 0 (promedio). La mediana (1) se probó y EMPEORÓ:
        # no quitó las estrías, quitó estructura (componente principal 95%->82%,
        # pedazos 1713->3686) y blanqueó el tono. Las estrías NO son un problema de
        # extracción: se fabrican en el entrenamiento (priors). Aquí se restaura el
        # valor de b02d2d8c.
        _DEPTH_RATIO = os.environ.get("DEPTH_RATIO", "0")
        log(f"   depth_ratio = {_DEPTH_RATIO} (0=promedio, el de b02d2d8c; la mediana empeoró)")
        log(f"$ python /opt/2dgs/render.py (BOUNDED) --depth_ratio {_DEPTH_RATIO} "
            f"--voxel_size {voxel:.4f} --sdf_trunc {sdf_trunc:.4f} "
            f"--depth_trunc {depth_trunc:.2f} --num_cluster 50  (OMP=8)")
        rc_mesh, _salida_mesh = run(
            ["python", "/opt/2dgs/render.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--skip_train", "--skip_test",
             "--depth_ratio", _DEPTH_RATIO,
             "--voxel_size", f"{voxel:.6f}",
             "--sdf_trunc", f"{sdf_trunc:.6f}",
             "--depth_trunc", f"{depth_trunc:.6f}",
             "--num_cluster", "50"],
            env=env_mesh, fase_label="PASO 4/5 — Extrayendo malla", check=False)
        # Buscar la malla generada. En modo unbounded los nombres son
        # fuse_unbounded.ply (cruda) y fuse_unbounded_post.ply (limpia). Preferimos
        # la limpia. En modo BOUNDED los nombres son fuse_post.ply (limpia) y
        # fuse.ply (cruda); dejamos los unbounded como respaldo por si acaso.
        candidatos = list(dgs_out.rglob("*.ply"))
        def _es_no_vacia(p):
            try:
                return p.stat().st_size > 1000   # >1KB = tiene geometría real
            except Exception:
                return False
        malla = None
        for nombre in ("fuse_post.ply", "fuse.ply",
                       "fuse_unbounded_post.ply", "fuse_unbounded.ply"):
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
        # ── MEDIDOR DE LÁMINAS (objetivo, no depende del ojo) ──
        # render.py imprime "#clusters=N": los PEDAZOS SUELTOS de la malla cruda.
        # Una superficie sólida = pocos pedazos. Las láminas = muchísimos pedazos.
        # Referencias medidas en este proyecto:
        #   ~18,000  = SIN priors, superficie sana (nunca hubo queja de estrías)
        #  ~117,000  = CON priors (render b02d2d8c): LAMINADO
        #  ~317,000  = CON priors + mediana: PEOR
        try:
            import re as _re2
            _cl = _re2.findall(r'#clusters=(\d+)', _salida_mesh or "")
            if _cl:
                _nc = int(_cl[-1])
                if _nc < 40000:
                    log(f"   ✓ LÁMINAS: {_nc} pedazos sueltos — SANO (ref: 18k sano / 117k laminado)")
                elif _nc < 80000:
                    log(f"   ~ LÁMINAS: {_nc} pedazos sueltos — MEJOR pero no del todo (ref: 18k sano / 117k laminado)")
                else:
                    log(f"   ⚠ LÁMINAS: {_nc} pedazos sueltos — SIGUE LAMINADO (ref: 18k sano / 117k laminado)")
        except Exception as _e:
            log(f"   (no pude medir las láminas: {_e})")

        # ── Limpiar + SUAVIZAR + simplificar la malla ──
        # 3 mejoras (investigación) sobre la malla cruda:
        #  1. FILTRO POR TAMAÑO: conserva pedazos grandes (techo, muebles) y quita
        #     solo floaters diminutos. Mejor que num_cluster=1 (que borraba el techo
        #     si quedaba como isla suelta).
        #  2. SUAVIZADO TAUBIN: quita el "papel arrugado"/facetado SIN encoger el
        #     cuarto (Taubin λ=0.5 μ=-0.53 compensa la contracción). Se hace ANTES
        #     de decimar para no congelar el ruido en la malla final.
        #  3. Decimar a ~500k triángulos manteniendo forma y color.
        fase(0.90, "PASO 4/5 — Suavizando y simplificando malla")
        decimada = WORK / "mesh_lite.ply"
        script_dec = (
            "import open3d as o3d\n"
            "import numpy as np\n"
            "import os\n"
            f"m = o3d.io.read_triangle_mesh(r'{malla}')\n"
            "n0 = len(m.triangles)\n"
            "print('DIAG vertices', len(m.vertices), 'triangulos', n0, flush=True)\n"
            # --- Limpieza básica ---
            "m.remove_unreferenced_vertices()\n"
            "m.remove_degenerate_triangles()\n"
            "m.remove_duplicated_vertices()\n"
            "m.remove_duplicated_triangles()\n"
            # --- 1) FILTRO: quita floaters diminutos Y pedazos disparados FUERA del
            #     cuarto (ventanas explotadas). El filtro por tamaño solo no bastaba:
            #     la ventana explotada era "grande" (4%) pero estaba lejísimos. Ahora
            #     hallamos el componente principal (el cuarto) y quitamos los demás
            #     que: o son diminutos, o su centro cae FUERA de la caja del cuarto
            #     expandida 15% (esos son los pedazos que el vidrio disparó hacia afuera).
            "try:\n"
            "    cl = m.cluster_connected_triangles()\n"
            "    lab = np.asarray(cl[0]); ntri = np.asarray(cl[1])\n"
            "    total = int(ntri.sum())\n"
            "    umbral = max(1000, int(0.002 * total))\n"   # 0.2% de los triángulos
            "    V = np.asarray(m.vertices); T = np.asarray(m.triangles)\n"
            "    main_i = int(np.argmax(ntri))\n"
            "    mv = np.unique(T[lab == main_i].reshape(-1))\n"
            "    bmin = V[mv].min(0); bmax = V[mv].max(0)\n"
            "    bc = (bmin+bmax)/2.0; bh = (bmax-bmin)/2.0 * 1.15 + 1e-6\n"
            "    lo = bc - bh; hi = bc + bh\n"
            "    quitar = np.zeros(len(T), dtype=bool); nq_s=0; nq_f=0\n"
            "    for i in range(len(ntri)):\n"
            "        if i == main_i: continue\n"
            "        cm = lab == i\n"
            "        if ntri[i] < umbral:\n"
            "            quitar[cm] = True; nq_s += 1; continue\n"
            "        cv = np.unique(T[cm].reshape(-1)); cc = (V[cv].min(0)+V[cv].max(0))/2.0\n"
            "        if np.any(cc < lo) or np.any(cc > hi):\n"
            "            quitar[cm] = True; nq_f += 1\n"
            "    m.remove_triangles_by_mask(quitar)\n"
            "    m.remove_unreferenced_vertices()\n"
            "    print('FILTER quito %d diminutos + %d fuera-del-cuarto (de %d comp)' % (nq_s, nq_f, len(ntri)), flush=True)\n"
            "except Exception as e:\n"
            "    print('FILTER (fallo, sigo):', e, flush=True)\n"
            # --- Si viene gigantesca, pre-decimar para no reventar RAM al suavizar ---
            "if len(m.triangles) > 4000000:\n"
            "    print('PRE-DECIMATE malla muy grande (%d)...' % len(m.triangles), flush=True)\n"
            "    m = m.simplify_quadric_decimation(target_number_of_triangles=1500000)\n"
            # --- 2) SUAVIZADO TAUBIN MÍNIMO (1 ITERACIÓN) — PASO DE DETALLE ---
            # La investigación confirmó que Taubin 3× sobre una malla de alta
            # resolución DESPERDICIA los vértices nuevos (los alisa y borra el
            # micro-relieve. CAMBIO: subimos a 5 ITERACIONES (antes 1) para alisar
            # la RUGOSIDAD GEOMÉTRICA real ("braille"/papel de lija) que tiene la malla
            # del TSDF — la realidad empírica es que el "detalle fino" que queríamos
            # conservar NO existe (es ruido del TSDF a 1cm de voxel), así que suavizar
            # fuerte solo quita lo malo. Taubin NO encoge la malla (preserva volumen),
            # solo alisa. FAIL-SAFE: si quita demasiado (bordes muy redondeados), bajar
            # a 3.
            # --- 2) SUAVIZADO BILATERAL "TwoStep" (preserva aristas) ---
            # ANTES: Taubin 8 iter, que alisa TODO POR IGUAL -> quitaba la lija pero
            # también REDONDEABA los muebles (el síntoma "geometría derretida").
            # AHORA: TwoStep Smooth de MeshLab (Belyaev & Ohtake), que es un filtro
            # BILATERAL en dos etapas: (1) promedia solo normales PARECIDAS entre sí,
            # (2) recoloca los vértices para ajustarse a esas normales. Las aristas que
            # forman ángulos MAYORES al umbral (45°) SE PRESERVAN. Resultado: alisa lo
            # plano (paredes/techo/piso) y NO redondea las esquinas ni los muebles.
            # Si pymeshlab no está o falla, cae a Taubin (comportamiento anterior).
            "SMOOTH_MODE = os.environ.get('SMOOTH_MODE', 'twostep')\n"
            "_suavizado = False\n"
            "if SMOOTH_MODE == 'twostep':\n"
            "    try:\n"
            "        import pymeshlab\n"
            "        Vs = np.array(m.vertices, dtype=np.float64)   # COPIA (no vista)\n"
            "        Ts = np.asarray(m.triangles)\n"
            "        ms = pymeshlab.MeshSet()\n"
            "        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=Vs, face_matrix=Ts))\n"
            "        ms.apply_coord_two_steps_smoothing(\n"
            "            stepsmoothnum=4, normalthr=45.0, stepnormalnum=20, stepfitnum=15,\n"
            "            selected=False)\n"
            "        Vn = ms.current_mesh().vertex_matrix()\n"
            "        assert Vn.shape == Vs.shape, 'TwoStep cambio el numero de vertices'\n"
            "        _des = float(np.linalg.norm(Vn - Vs, axis=1).mean())\n"
            "        m.vertices = o3d.utility.Vector3dVector(Vn)\n"
            "        print('SMOOTH TwoStep bilateral OK (umbral 45 grados; preserva aristas); '\n"
            "              'desplazamiento medio %.2f mm' % (_des*1000), flush=True)\n"
            "        _suavizado = True\n"
            "    except Exception as e:\n"
            "        print('SMOOTH TwoStep FALLO (%s) -> caigo a Taubin' % e, flush=True)\n"
            "if not _suavizado:\n"
            "    try:\n"
            "        m = m.filter_smooth_taubin(number_of_iterations=8)\n"
            "        print('SMOOTH Taubin 8 iter (respaldo) OK', flush=True)\n"
            "    except Exception as e:\n"
            "        print('SMOOTH (fallo, sigo):', e, flush=True)\n"
            # --- 2b) SNAP A PLANOS (RANSAC) — paredes/techo/piso PLANOS DE VERDAD ---
            # Detecta los planos dominantes del cuarto y proyecta SOLO sus vértices
            # sobre el plano ideal. Es post-proceso puro: no reescala nada, no toca
            # el entrenamiento -> RIESGO CERO de deformar la estructura.
            # TRES CANDADOS para NO aplanar la cama ni los muebles:
            #   1) TAMAÑO: el plano debe medir >=1.2 m en sus dos ejes (una cama no).
            #   2) PERIFERIA: el plano debe estar en el BORDE del cuarto (paredes,
            #      techo, piso), no flotando en el medio (la cama está en el medio).
            #   3) NORMAL DEL VÉRTICE: solo se mueve el vértice si su normal ya
            #      apunta casi igual que el plano (<35 grados) -> los objetos apoyados
            #      contra la pared no se absorben.
            # Además: FEATHERING en el borde (anillos con peso 0.75/0.5/0.25) para que
            # no quede un escalón entre la zona aplanada y el resto.
            "PLANE_SNAP = os.environ.get('PLANE_SNAP', '1') == '1'\n"
            "if PLANE_SNAP:\n"
            "  try:\n"
            "    import scipy.sparse as _sp\n"
            "    _V = np.asarray(m.vertices); _T = np.asarray(m.triangles)\n"
            "    m.compute_vertex_normals()\n"
            "    _N = np.asarray(m.vertex_normals)\n"
            "    _ab = m.get_axis_aligned_bounding_box()\n"
            "    _ext = _ab.get_extent(); _cg = _ab.get_center()\n"
            "    _half = np.linalg.norm(_ext)/2.0\n"
            "    _diag = float(np.linalg.norm(_ext))\n"
            "    _thr = float(np.clip(0.004*_diag, 0.012, 0.03))   # ~2.5 cm\n"
            "    _minin = max(30000, int(0.03*len(_V)))\n"
            "    # grafo de vecinos (para el feathering)\n"
            "    _e0 = np.concatenate([_T[:,0],_T[:,1],_T[:,2]])\n"
            "    _e1 = np.concatenate([_T[:,1],_T[:,2],_T[:,0]])\n"
            "    _A = _sp.csr_matrix((np.ones(len(_e0),dtype=np.int8), (_e0,_e1)),\n"
            "                        shape=(len(_V),len(_V)))\n"
            "    _A = _A + _A.T\n"
            "    _pcd = o3d.geometry.PointCloud()\n"
            "    _pcd.points = o3d.utility.Vector3dVector(_V)\n"
            "    _rest = np.arange(len(_V))\n"
            "    _nuevo = _V.copy(); _nplanos = 0\n"
            "    for _ronda in range(8):\n"
            "        if len(_rest) < _minin: break\n"
            "        _sub = o3d.geometry.PointCloud()\n"
            "        _sub.points = o3d.utility.Vector3dVector(_V[_rest])\n"
            "        try:\n"
            "            _mod, _inl = _sub.segment_plane(distance_threshold=_thr,\n"
            "                                            ransac_n=3, num_iterations=2000)\n"
            "        except Exception:\n"
            "            break\n"
            "        if len(_inl) < _minin: break\n"
            "        _idx = _rest[np.asarray(_inl)]\n"
            "        _rest = np.setdiff1d(_rest, _idx, assume_unique=False)\n"
            "        _a,_b,_c,_d = _mod; _nrm = np.array([_a,_b,_c],dtype=np.float64)\n"
            "        _ln = np.linalg.norm(_nrm)\n"
            "        if _ln < 1e-9: continue\n"
            "        _nrm = _nrm/_ln; _d = _d/_ln\n"
            "        # CANDADO 1 — tamaño: debe medir >=1.2 m en sus dos ejes\n"
            "        _pv = _V[_idx]\n"
            "        _u = np.array([1.0,0,0]) if abs(_nrm[0])<0.9 else np.array([0,1.0,0])\n"
            "        _e_1 = np.cross(_nrm,_u); _e_1 /= (np.linalg.norm(_e_1)+1e-12)\n"
            "        _e_2 = np.cross(_nrm,_e_1)\n"
            "        _p1 = _pv@_e_1; _p2 = _pv@_e_2\n"
            "        _ex2 = (float(_p1.max()-_p1.min()), float(_p2.max()-_p2.min()))\n"
            "        if min(_ex2) < 1.2:\n"
            "            print('PLANO descartado (pequeno %.2fx%.2f m: es un mueble)'\n"
            "                  % _ex2, flush=True); continue\n"
            "        # CANDADO 2 — periferia por PERCENTIL (robusto a bbox desbalanceado)\n"
            "        # ANTES: distancia del plano al CENTRO del bbox. Fallaba porque\n"
            "        # con muebles/geometria fuera del cuarto el centro se corre y\n"
            "        # TODAS las paredes daban perif baja -> se descartaban (visto en\n"
            "        # produccion: 0.04, 0.28, 0.35...). AHORA: proyectamos TODOS los\n"
            "        # vertices sobre la normal y miramos en que PERCENTIL cae el plano.\n"
            "        # Una pared/piso/techo real esta en el EXTREMO (percentil <=20 o\n"
            "        # >=80); un mueble cae en el medio. No depende del centro del cuarto.\n"
            "        # v7: los limites 12/88 quedaron APRETADOS: en produccion rechazaban\n"
            "        # paredes REALES en percentil 84/88/20 (la geometria flotante fuera\n"
            "        # del cuarto ensancha la distribucion). Con 20/80 esas paredes entran\n"
            "        # y los muebles (percentil 48 y 73) siguen afuera.\n"
            "        _sall = _V @ _nrm; _splane = -_d\n"
            "        _pct = float((_sall < _splane).mean()) * 100.0\n"
            "        if 20.0 < _pct < 80.0:\n"
            "            print('PLANO descartado (percentil %.0f: en el MEDIO del'\n"
            "                  ' cuarto, probablemente un mueble)' % _pct, flush=True); continue\n"
            "        # CANDADO 3 — normal del vertice casi paralela al plano (<35 grados)\n"
            "        _cos = np.abs(_N[_idx] @ _nrm)\n"
            "        _core = _idx[_cos > 0.819]\n"
            "        if len(_core) < _minin//2:\n"
            "            print('PLANO descartado (normales no coinciden)', flush=True); continue\n"
            "        # distancia ANTES (para medir la mejora)\n"
            "        _dist0 = np.abs(_V[_core] @ _nrm + _d)\n"
            "        _rms0 = float(np.sqrt((_dist0**2).mean()))\n"
            "        # FEATHERING: peso 1.0 en el nucleo, y 0.75/0.5/0.25 hacia fuera\n"
            "        _w = np.zeros(len(_V)); _w[_core] = 1.0\n"
            "        _frente = np.zeros(len(_V), dtype=bool); _frente[_core] = True\n"
            "        for _pw in (0.75, 0.5, 0.25):\n"
            "            _vec = np.zeros(len(_V), dtype=np.int8); _vec[_frente] = 1\n"
            "            _nb = (_A @ _vec) > 0\n"
            "            _nuevos = _nb & (_w == 0)\n"
            "            _w[_nuevos] = _pw; _frente = _nuevos\n"
            "        _mask = _w > 0\n"
            "        _dd = (_V[_mask] @ _nrm + _d)[:,None] * _nrm[None,:]\n"
            "        _nuevo[_mask] = _V[_mask] - _w[_mask][:,None] * _dd\n"
            "        _dist1 = np.abs(_nuevo[_core] @ _nrm + _d)\n"
            "        _rms1 = float(np.sqrt((_dist1**2).mean()))\n"
            "        _nplanos += 1\n"
            "        print('PLANO %d: %d vertices, planitud RMS %.1f mm -> %.1f mm '\n"
            "              '(tamano %.1fx%.1f m)' % (_nplanos, len(_core), _rms0*1000,\n"
            "              _rms1*1000, _ex2[0], _ex2[1]), flush=True)\n"
            "    if _nplanos:\n"
            "        m.vertices = o3d.utility.Vector3dVector(_nuevo)\n"
            "        m.compute_vertex_normals()\n"
            "        print('SNAP %d superficies aplanadas (paredes/techo/piso)'\n"
            "              % _nplanos, flush=True)\n"
            "    else:\n"
            "        print('SNAP: no encontre planos grandes que aplanar', flush=True)\n"
            "  except Exception as e:\n"
            "    print('SNAP a planos (fallo, sigo sin el):', e, flush=True)\n"
            # --- 3) DECIMAR a 2.5M — EL DOBLE de densidad de color ---
            #     El color se ve BORROSO de cerca porque la densidad de vertices ES la
            #     resolucion del color. La malla cruda tiene ~2.28M triangulos; ANTES
            #     la bajabamos a 1.2M (perdiendo la mitad del color). AHORA target=2.5M:
            #     como la cruda es menor, en la practica NO se decima (~1.18M vertices
            #     = el DOBLE de colores). Los visores de escritorio manejan 2-3M bien.
            #     FAIL-SAFE: si pesa mucho o el visor va lento, bajar MESH_TRIS a 1500000.
            "target = int(os.environ.get('MESH_TRIS', '2500000'))\n"
            "if len(m.triangles) > target:\n"
            "    m = m.simplify_quadric_decimation(target_number_of_triangles=target)\n"
            # --- LIMPIEZA PROFUNDA tras decimar (CLAVE para que el visor NO se cuelgue)
            #     La decimación puede dejar triángulos degenerados (área ~0), vértices
            #     duplicados y bordes no-manifold. Sobre esos, las normales salen NaN y
            #     el visor se cuelga al calcular el encuadre. Limpiamos TODO antes de
            #     calcular normales para garantizar una malla válida.
            "m.remove_unreferenced_vertices()\n"
            "m.remove_degenerate_triangles()\n"
            "m.remove_duplicated_vertices()\n"
            "m.remove_duplicated_triangles()\n"
            "try:\n"
            "    m.remove_non_manifold_edges()\n"
            "except Exception as _e:\n"
            "    print('non_manifold (fallo, sigo):', _e, flush=True)\n"
            "m.remove_unreferenced_vertices()\n"
            # Quitar triángulos con vértices NaN/Inf (de una decimación problemática):
            # son la causa típica del 'cargando para siempre' en 3dviewer.net.
            "try:\n"
            "    _Vc = np.asarray(m.vertices)\n"
            "    _bad = ~np.isfinite(_Vc).all(axis=1)\n"
            "    if _bad.any():\n"
            "        _keep = np.where(~_bad)[0]\n"
            "        m = m.select_by_index(_keep.tolist())\n"
            "        print('NAN-GUARD: quite %d vertices invalidos' % int(_bad.sum()), flush=True)\n"
            "except Exception as _e:\n"
            "    print('NAN-GUARD (fallo, sigo):', _e, flush=True)\n"
            "m.compute_vertex_normals()\n"
            "m.compute_triangle_normals()\n"
            # --- 4) AMBIENT OCCLUSION por vértice (EL PASO QUE MÁS QUITA EL PLÁSTICO)
            #     Hornea sombras de contacto (rincones, juntas, muebles contra piso/
            #     pared) en el color por vértice → el ojo lo lee como DETALLE y
            #     profundidad. Lanza 64 rayos desde cada vértice (Open3D Raycasting)
            #     y oscurece según cuántos chocan cerca. Si falla, sigue sin AO.
            "try:\n"
            "    scn = o3d.t.geometry.RaycastingScene()\n"
            "    scn.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))\n"
            "    Vv = np.asarray(m.vertices); Nn = np.asarray(m.vertex_normals)\n"
            "    _ext = m.get_axis_aligned_bounding_box().get_extent()\n"
            "    _dg = float(np.linalg.norm(_ext)); _rad = 0.08*_dg; _eps = 0.0015*_dg\n"
            "    K = 64; _gi = (1+5**0.5)/2; _ii = np.arange(K)+0.5\n"
            "    _phi = np.arccos(1-2*_ii/K); _th = 2*np.pi*_ii/_gi\n"
            "    dirs = np.stack([np.sin(_phi)*np.cos(_th), np.sin(_phi)*np.sin(_th), np.cos(_phi)],1)\n"
            "    ao = np.zeros(len(Vv)); _ch = 20000\n"
            "    for s in range(0, len(Vv), _ch):\n"
            "        vs = Vv[s:s+_ch]; ns = Nn[s:s+_ch]; nv = len(vs)\n"
            "        O = np.repeat(vs + ns*_eps, K, axis=0); D = np.tile(dirs,(nv,1))\n"
            "        nd = (D*np.repeat(ns,K,axis=0)).sum(1); up = nd > 0\n"
            "        rays = np.concatenate([O,D],1).astype(np.float32)\n"
            "        rt = scn.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()\n"
            "        hh = ((rt < _rad) & up).reshape(nv,K); u2 = up.reshape(nv,K)\n"
            "        ao[s:s+_ch] = hh.sum(1)/np.maximum(u2.sum(1),1)\n"
            "    C = np.asarray(m.vertex_colors)\n"
            "    ao = np.nan_to_num(ao, nan=0.0, posinf=0.0, neginf=0.0)\n"
            # GUARDAR EL AO A DISCO: el pintor (paso 4c) SOBRESCRIBE el color de los
            # vertices con el de las fotos, y hasta ahora eso BORRABA el AO en el 97%
            # de los vertices -> render plano, lavado, "blanco". Guardandolo aqui, el
            # pintor puede volver a aplicarlo sobre el color de las fotos.
            "    np.save('/workspace/job/ao.npy', ao.astype(np.float32))\n"
            "    print('AO guardado para el pintor (%d vertices)' % len(ao), flush=True)\n"
            "    if len(C) == len(Vv):\n"
            "        C = C * (1 - 0.3*ao)[:,None]\n"
            "        C = np.clip(C, 0, 1) ** 0.85\n"
            "        C = np.nan_to_num(C, nan=0.5, posinf=1.0, neginf=0.0)\n"
            "        m.vertex_colors = o3d.utility.Vector3dVector(np.clip(C,0,1))\n"
            "        print('AO suave 0.3 + realce brillo (gamma 0.85); oclusion media %.3f' % float(ao.mean()), flush=True)\n"
            "    else:\n"
            "        print('AO: malla sin color por vertice, lo salto', flush=True)\n"
            "except Exception as e:\n"
            "    print('AO (fallo, sigo sin AO):', e, flush=True)\n"
            f"o3d.io.write_triangle_mesh(r'{decimada}', m)\n"
            "print('DECIMATE triangulos', n0, '->', len(m.triangles), flush=True)\n"
            # --- DIAGNÓSTICOS (sobre la malla YA decimada = liviana y segura) ---
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
        # pymeshlab lo usa el suavizado TwoStep (preserva aristas). No viene en la
        # imagen; lo instalamos aquí (rápido). Si falla, el postproc cae a Taubin.
        log("   preparando suavizado bilateral (pymeshlab)...")
        os.system(sys.executable + " -m pip install pymeshlab --quiet 2>/dev/null")
        rc_dec, _ = run(["python", "-c", script_dec],
                        fase_label="PASO 4/5 — Simplificando malla", check=False)
        if decimada.exists() and decimada.stat().st_size > 1000:
            nuevo_mb = decimada.stat().st_size / 1e6
            log(f"   malla simplificada: {nuevo_mb:.1f} MB (antes {ply_mb:.1f} MB)")
            malla = decimada
            ply_mb = nuevo_mb
        else:
            log("   (no se pudo simplificar; subo la malla original)")

        # ══════════════════════════════════════════════════════════════════════
        # PASO 4b: exportar la malla (color por vértice + AO) a .glb
        # ══════════════════════════════════════════════════════════════════════
        # DESCARTAMOS el pegado de fotos (OpenMVS TextureMesh): daba mal resultado
        # visual. Volvemos al color por vértice del 2DGS, PERO con las mejoras
        # anti-plástico ya aplicadas en el post-proceso: Taubin mínimo (conserva el
        # micro-relieve) + Ambient Occlusion horneado (da profundidad y sensación de
        # detalle). Al exportar a .glb, trimesh genera NORMALES SUAVES → no se ven
        # triángulos. Sin pasos pesados que puedan fallar.
        fase(0.93, "PASO 4b/5 — Exportando malla a .glb")
        import trimesh
        glb_final = WORK / "mesh_2dgs.glb"
        try:
            sc = trimesh.load(str(malla), process=False)
            # ── ARREGLO DEL FACETADO (probado): forzar el cálculo de NORMALES SUAVES
            #    antes de exportar. Sin esto, trimesh exporta el .glb SIN normales y el
            #    visor calcula normales PLANAS por triángulo → se ven los triángulos
            #    (aspecto áspero/geométrico). Al acceder a vertex_normals, trimesh
            #    promedia las caras por vértice (suave) y SÍ las mete en el .glb.
            #    El color por vértice + AO se conserva intacto.
            try:
                def _sane_normales(_gm):
                    import numpy as _np
                    _vn = _np.asarray(_gm.vertex_normals, dtype=_np.float64)
                    _bad = ~_np.isfinite(_vn).all(axis=1) | (_np.linalg.norm(_vn, axis=1) < 1e-8)
                    if _bad.any():
                        _vn = _vn.copy(); _vn[_bad] = (0.0, 0.0, 1.0)
                        _vn /= (_np.linalg.norm(_vn, axis=1, keepdims=True) + 1e-12)
                        _gm.vertex_normals = _vn
                if isinstance(sc, trimesh.Scene):
                    for _g in sc.geometry.values():
                        _sane_normales(_g)
                else:
                    _sane_normales(sc)
                log("   normales suaves forzadas en el .glb (anti-facetado)")
            except Exception as _ne:
                log(f"   ⚠ no pude forzar normales ({_ne}); el visor podría facetar")
            sc.export(str(glb_final))
            log(f"   .glb (color por vértice + AO): {glb_final.stat().st_size/1e6:.1f} MB")
        except Exception as e:
            log(f"   ⚠ no se pudo exportar .glb ({e}); subo el .ply")

        # ── SANEADOR ANTI-NaN de la vista previa (mismo escudo que el pintor):
        # exportadores viejos de trimesh pueden meter NaN literal al JSON del .glb
        # y eso revienta el JSON.parse del visor ("Unexpected token N"). Se repara
        # el binario y se recalculan los min/max REALES de cada accessor float.
        try:
            if glb_final.exists() and glb_final.stat().st_size > 1000:
                import json as _json, struct as _st
                import numpy as _np
                _d = bytearray(open(glb_final, "rb").read())
                _jlen = _st.unpack("<I", _d[12:16])[0]
                _g = _json.loads(_d[20:20 + _jlen].decode("utf-8"))
                _bin = bytearray(_d[20 + _jlen:])
                _attr = {}
                for _msh in _g.get("meshes", []):
                    for _pr in _msh.get("primitives", []):
                        for _an, _ai in _pr.get("attributes", {}).items():
                            _attr[_ai] = _an
                _rep = 0
                _NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
                for _ai, _acc in enumerate(_g.get("accessors", [])):
                    _ncomp = _NC.get(_acc.get("type"), 0)
                    if _acc.get("componentType") != 5126 or "bufferView" not in _acc or not _ncomp:
                        continue
                    _bv = _g["bufferViews"][_acc["bufferView"]]
                    _off = 8 + _bv.get("byteOffset", 0) + _acc.get("byteOffset", 0)
                    _nfl = _acc["count"] * _ncomp
                    _arr = _np.frombuffer(bytes(_bin[_off:_off + _nfl * 4]), _np.float32)
                    _arr = _arr.reshape(_acc["count"], _ncomp).copy()
                    if not _np.isfinite(_arr).all():
                        if _attr.get(_ai) == "NORMAL" and _ncomp == 3:
                            _mal = ~_np.isfinite(_arr).all(axis=1)
                            _arr[_mal] = (0.0, 0.0, 1.0)
                            _rep += int(_mal.sum())
                        else:
                            _rep += int((~_np.isfinite(_arr)).sum())
                            _arr = _np.nan_to_num(_arr, nan=0.0, posinf=0.0, neginf=0.0)
                        _bin[_off:_off + _nfl * 4] = _arr.astype(_np.float32).tobytes()
                    if "min" in _acc or "max" in _acc or _attr.get(_ai) == "POSITION":
                        _acc["min"] = [float(x) for x in _arr.min(0)]
                        _acc["max"] = [float(x) for x in _arr.max(0)]
                _nj = _json.dumps(_g, separators=(",", ":"), allow_nan=False).encode("utf-8")
                while len(_nj) % 4:
                    _nj += b" "
                _out = bytearray(); _out += _d[:12]
                _out += _st.pack("<I", len(_nj)) + b"JSON" + _nj + _bin
                _out[8:12] = _st.pack("<I", len(_out))
                open(glb_final, "wb").write(bytes(_out))
                if _rep:
                    log(f"   saneados {_rep} valores NaN en la vista previa")
        except Exception as _se:
            log(f"   (saneador de vista previa falló: {_se}; sigo)")

        # Subida ANTICIPADA: el .glb de color por vértice se sube YA, a la misma
        # URL final. Si la textura (xatlas, 25-90 min) se cancela o el pod muere,
        # igual queda un modelo visible para evaluar geometría (huecos/techo).
        # El .glb texturizado lo SOBRESCRIBE al terminar.
        try:
            if glb_final.exists() and glb_final.stat().st_size > 1000:
                with open(glb_final, "rb") as _f:
                    _req = urllib.request.Request(UPLOAD_URL_PLY, data=_f.read(), method="PUT")
                    urllib.request.urlopen(_req, timeout=300).read()
                log("   ⬆ VISTA PREVIA subida (color por vértice): ya se puede abrir el modelo; ahora empieza la textura")
        except Exception as _pe:
            log(f"   (vista previa no subida: {_pe}; sigo)")

        # ══════════════════════════════════════════════════════════════════════
        # ──────────────────────────────────────────────────────────────────
        # PASO 4c: PINTAR VERTICES desde las fotos (SIEMPRE: es el respaldo
        #          seguro y da la linea base de la auditoria en el log).
        # PASO 4d: TEXTURA UV de MEJOR-VISTA — el arreglo REAL del efecto
        #          oleo/acuarela. El color por vertice reparte ~1.1M colores en
        #          el cuarto (texel ~5mm: no puede mostrar letras); la textura
        #          4K da texel ~1mm tomando el color de la MEJOR camara por
        #          punto (promediar 127 vistas emborrona — Waechter ECCV'14,
        #          la base de OpenMVS). Si la textura falla por lo que sea, se
        #          sube el pintado por vertice: el render NUNCA se pierde.
        # ──────────────────────────────────────────────────────────────────
        glb_tex = WORK / "mesh_textured.glb"
        glb_uv  = WORK / "mesh_uv.glb"
        _paint_env = dict(os.environ)
        try:
            fase(0.94, "PASO 4c/5 — Pintando vértices desde las fotos")
            paint_py = WORK / "vertex_paint.py"
            paint_py.write_text(VERTEXPAINT_SCRIPT)
            # Preparar las fotos ORIGINALES de 12MP con el nombre que usa el
            # pintor (img_NNNN.*). MASt3R nombra sus imágenes img_0000, img_0001…
            # en el MISMO orden en que copiamos las originales foto_0000,
            # foto_0001… (ambos son sorted() del ZIP), así que el índice coincide.
            # El pintor recorta cada original al aspecto de cameras.txt, así que
            # solo importa el índice, no el tamaño.
            try:
                _orig_src = sorted((WORK / "images").glob("foto_*"))
                _dst = WORK / "orig12mp"; _dst.mkdir(exist_ok=True)
                for _i, _op in enumerate(_orig_src):
                    _lnk = _dst / ("img_%04d%s" % (_i, _op.suffix.lower()))
                    if not _lnk.exists():
                        shutil.copy(_op, _lnk)
                _paint_env["PAINT_ORIG_DIR"] = str(_dst)
                log(f"   carpeta 12MP lista ({len(_orig_src)} fotos) — la usa la TEXTURA UV (el pintor usa 1000px: es solo el respaldo)")
            except Exception as _pe:
                log(f"   (no pude preparar las 12MP: {_pe}; pinto desde 1000px)")
            run(["python", str(paint_py), str(malla), str(dataset / "images"),
                 str(dataset / "sparse" / "0"), str(glb_tex),
                 str(WORK / "ao.npy")],   # AO -> devuelve profundidad al color
                fase_label="PASO 4c/5 — Pintando vértices", check=False, env=_paint_env)
            if glb_tex.exists() and glb_tex.stat().st_size > 1000:
                log(f"   ✓ vértices pintados desde las FOTOS: {glb_tex.stat().st_size/1e6:.1f} MB")
            else:
                log("   ⚠ el pintado no produjo archivo; uso color por vértice del entrenamiento")
        except Exception as e:
            log(f"   ⚠ pintado falló ({e}); uso color por vértice del entrenamiento")

        # ── PASO 4d: TEXTURA UV (UV_TEXTURE=1 por defecto; =0 la apaga) ──
        if os.environ.get("UV_TEXTURE", "1") == "1":
            try:
                fase(0.945, "PASO 4d/5 - Texturizando con OpenMVS (fotos 12MP, metodo Polycam)")
                omvs_py = WORK / "openmvs_texture.py"
                omvs_py.write_text(OPENMVS_TEXTURE_SCRIPT)
                _dir_uv = _paint_env.get("PAINT_ORIG_DIR", str(dataset / "images"))
                run(["python", str(omvs_py), str(malla), _dir_uv,
                     str(dataset / "sparse" / "0"), str(glb_uv),
                     str(WORK / "ao.npy")],
                    fase_label="PASO 4d/5 - Texturizando con OpenMVS",
                    check=False, env=_paint_env, timeout=2400)  # 40 min max y corta
                if glb_uv.exists() and glb_uv.stat().st_size > 200000:
                    log(f"   OK textura OpenMVS: {glb_uv.stat().st_size/1e6:.1f} MB - se sube ESTA (sin costuras)")
                else:
                    log("   OpenMVS no produjo archivo valido; subo el color por vertice (respaldo)")
            except Exception as e:
                log(f"   textura OpenMVS fallo ({e}); subo el color por vertice (respaldo)")
        else:
            log("   PASO 4d saltado (UV_TEXTURE=0): se sube el color por vertice")

        # Archivo a subir (orden de preferencia):
        #   1º TEXTURA UV (nítida)   2º vértices pintados   3º color del
        #   entrenamiento   4º .ply crudo
        if glb_uv.exists() and glb_uv.stat().st_size > 200000:
            archivo_subir = glb_uv
            ply_mb = glb_uv.stat().st_size / 1e6
        elif glb_tex.exists() and glb_tex.stat().st_size > 1000:
            archivo_subir = glb_tex
            ply_mb = glb_tex.stat().st_size / 1e6
        elif glb_final.exists() and glb_final.stat().st_size > 1000:
            archivo_subir = glb_final
            ply_mb = glb_final.stat().st_size / 1e6
        else:
            archivo_subir = malla

        # ── PASO 5: subir la malla ──
        fase(0.95, "PASO 5/5 — Subiendo malla")
        with open(archivo_subir, "rb") as f:
            req = urllib.request.Request(UPLOAD_URL_PLY, data=f.read(),
                                         method="PUT")
            urllib.request.urlopen(req, timeout=300).read()
        log(f"   malla subida ({archivo_subir.name})")

        # ── Listo ──
        _estado["vivo"] = False
        seconds = time.time() - t0
        log(f"═══ LISTO en {seconds/60:.1f} min ═══")
        # MARCA DE TERMINADO: se escribe ANTES de avisar al backend, a propósito.
        # Si el aviso falla (404, red caída, Railway reiniciando) y RunPod relanza
        # el contenedor, el worker verá esta marca y NO volverá a rendir. Sin ella
        # se repetía el render entero: 35 min de GPU, una y otra vez.
        try:
            marca.write_text(json.dumps({
                "frames_used": n_fotos, "ply_mb": round(ply_mb, 1),
                "seconds": round(seconds), "pod_id": POD_ID}))
            log("   marca de terminado escrita — este job NO se repetirá aunque se reinicie")
        except Exception as _me:
            log(f"   ⚠ no pude escribir la marca ({_me}); si el pod se reinicia PODRÍA repetir")
        callback("completed", frames_used=n_fotos, ply_mb=round(ply_mb, 1),
                 seconds=round(seconds), log="\n".join(_LOG))
        if POD_ID:
            log(f"   backend avisado (pod {POD_ID}); debería apagarse solo")
        else:
            log("   ⚠ no sé mi pod_id: si no se apaga solo, apágalo en runpod.io")

    except Exception as e:
        _estado["vivo"] = False
        log(f"✗ ERROR: {e}")
        # Marca también al fallar: un error determinista (ZIP corrupto, imagen mala)
        # volvería a fallar igual al reiniciar, quemando GPU para nada.
        try:
            marca.write_text(json.dumps({"error": str(e)[:200], "pod_id": POD_ID}))
        except Exception:
            pass
        callback("error", error_message=str(e), log="\n".join(_LOG))
        sys.exit(1)


if __name__ == "__main__":
    main()
