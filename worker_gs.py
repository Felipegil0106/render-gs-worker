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
# ════════════════════════════════════════════════════════════════════════
# DEDUPLICACION DE FOTOS REDUNDANTES (v10) — aportado por Felipe
# ------------------------------------------------------------------------
# Al recorrer la casa el celular saca varias fotos casi identicas desde el
# mismo punto y el mismo angulo. Esas fotos no aportan paralaje (que es de
# donde sale la geometria), sobrerrepresentan una zona en el entrenamiento, y
# SI cuestan GPU en MASt3R (~20-30 pares por foto, cada par un ViT-Large).
# MEDIDO en produccion (job a1161a5d): 185 -> 164 fotos, PSNR 33.5 -> 33.8 y
# el render bajo de 59.7 a 54.8 min. O sea: quitando fotos MEJORO la calidad.
# OJO: no acelera el PASO 3 (ITERS es fijo y 2DGS escoge vista al azar).
# Se hace en dos sitios: PREFILTRO (PASO 1, 0 GPU, con el poses.json de la app)
# y FILTRO FINO (PASO 2, con la pose 6DoF exacta). Todo se apaga con DEDUP=0.
# ════════════════════════════════════════════════════════════════════════
DEDUP           = os.environ.get("DEDUP", "1") == "1"
DEDUP_DIST_M    = float(os.environ.get("DEDUP_DIST_M", "0.08"))   # 8 cm
DEDUP_ROT_DEG   = float(os.environ.get("DEDUP_ROT_DEG", "6.0"))   # 6 grados
DEDUP_MIN_KEEP  = int(os.environ.get("DEDUP_MIN_KEEP", "60"))     # nunca bajar de aqui
PREFILTER        = os.environ.get("PREFILTER", "1") == "1"
PREFILTER_TIME_S = float(os.environ.get("PREFILTER_TIME_S", "20"))
# SEGURO ANTI-TEXTURA-REPETIDA (suelos de baldosa, paredes lisas, escaleras)
# Si dos fotos salen en la MISMA pose segun MASt3R pero se tomaron con mucha
# diferencia de tiempo, puede que MASt3R se haya confundido con baldosas
# iguales y haya puesto dos sitios distintos en la misma coordenada. Ante la
# duda NO se borra. 0 = desactivar el seguro.
DEDUP_TIME_GUARD_S = float(os.environ.get("DEDUP_TIME_GUARD_S", "120"))

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
STAMPS_JSON = sys.argv[3] if len(sys.argv) > 3 else ""

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
# ════════════════════════════════════════════════════════════════════════
# DEDUPLICACION POR POSE 6DoF EXACTA (Felipe)
# Regla: se CONSERVA si te moviste > DIST metros O giraste > ROT grados.
# Si las dos cosas estan por debajo, es la misma pose -> se descarta.
# Solo afecta al ENTRENAMIENTO: la nube densa de mas abajo sigue usando las N
# vistas, asi que la GEOMETRIA no pierde nada.
# De cada grupo sobrevive la MAS NITIDA (varianza del Laplaciano).
# ════════════════════════════════════════════════════════════════════════
DEDUP     = os.environ.get("DEDUP", "1") == "1"
DIST_M    = float(os.environ.get("DEDUP_DIST_M", "0.08"))
ROT_DEG   = float(os.environ.get("DEDUP_ROT_DEG", "6.0"))
MIN_KEEP  = int(os.environ.get("DEDUP_MIN_KEEP", "60"))
GUARD_S   = float(os.environ.get("DEDUP_TIME_GUARD_S", "120"))
_stamps = {}
if STAMPS_JSON and os.path.exists(STAMPS_JSON):
    try:
        import json as _json
        _stamps = _json.load(open(STAMPS_JSON))
        print("DEDUP: seguro anti-textura-repetida ACTIVO (%d timestamps, %.0f s)"
              % (len(_stamps), GUARD_S), flush=True)
    except Exception as _e:
        print("DEDUP: no pude leer timestamps (%s), seguro desactivado" % _e, flush=True)
def _t_de(i):
    """Instante de captura de la camara i, o None si no se sabe."""
    try:
        return _stamps.get(os.path.splitext(os.path.basename(filelist[i]))[0])
    except Exception:
        return None
# En cams2world la traslacion ES el centro optico (cam->world): no hay que
# hacer -R^T*t (eso es solo para el formato de COLMAP).
centros = np.array([c[:3, 3] for c in cams2world])
rots    = Rotation.from_matrix(np.array([c[:3, :3] for c in cams2world]))
_ext = centros.max(axis=0) - centros.min(axis=0)
_diag = float(np.linalg.norm(_ext))
print("ESCALA: recorrido de camaras = %.2f x %.2f x %.2f (diagonal %.2f). "
      "Si esto se parece a metros, DEDUP_DIST_M=%.3f son %.1f cm reales."
      % (_ext[0], _ext[1], _ext[2], _diag, DIST_M, DIST_M * 100), flush=True)
def _nitidez(im):
    """Varianza del Laplaciano sobre la imagen de 512px (float 0..1)."""
    try:
        g = im.mean(axis=2) if im.ndim == 3 else im
        lap = (4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
               - g[1:-1, :-2] - g[1:-1, 2:])
        return float(lap.var())
    except Exception:
        return 0.0
keep = list(range(N))
if DEDUP and N > MIN_KEEP:
    nitidez = np.array([_nitidez(im) for im in rgbimgs])
    orden = list(np.argsort(-nitidez))      # las mas nitidas primero
    lado = max(DIST_M, 1e-4)
    rejilla = {}
    aceptadas = []
    descartadas = []
    for i in orden:
        c = centros[i]
        k = (int(np.floor(c[0]/lado)), int(np.floor(c[1]/lado)), int(np.floor(c[2]/lado)))
        repetida = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in rejilla.get((k[0]+dx, k[1]+dy, k[2]+dz), ()):
                        if float(np.linalg.norm(c - centros[j])) >= DIST_M:
                            continue                      # otro punto -> sirve
                        ang = np.degrees((rots[i].inv() * rots[j]).magnitude())
                        if ang >= ROT_DEG:
                            continue                      # otro angulo -> sirve
                        if GUARD_S > 0:
                            ta, tb = _t_de(i), _t_de(j)
                            if ta is not None and tb is not None and abs(ta - tb) > GUARD_S:
                                continue
                        repetida = True
                        break
                    if repetida: break
                if repetida: break
            if repetida: break
        if repetida:
            descartadas.append(i)
        else:
            rejilla.setdefault(k, []).append(i)
            aceptadas.append(i)
    if len(aceptadas) < MIN_KEEP:
        faltan = MIN_KEEP - len(aceptadas)
        rescatadas = descartadas[:faltan]     # ya vienen ordenadas por nitidez
        print("DEDUP: solo quedaban %d, devuelvo %d para no romper el solape"
              % (len(aceptadas), len(rescatadas)), flush=True)
        aceptadas += rescatadas
    keep = sorted(aceptadas)
    quitadas = N - len(keep)
    print("DEDUP: %d fotos -> %d (%d repetidas quitadas, %.1f%%). "
          "Criterio: misma pose = < %.0f cm Y < %.1f grados"
          % (N, len(keep), quitadas, 100.0*quitadas/max(N,1), DIST_M*100, ROT_DEG),
          flush=True)
    if quitadas > 0.6 * N:
        print("DEDUP: OJO, se quito mas del 60%. Si MASt3R registra pocas "
              "camaras o la malla sale con huecos, sube DEDUP_DIST_M.", flush=True)
else:
    print("DEDUP: desactivado o pocas fotos (%d). Se usan todas." % N, flush=True)

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
# IMPORTANTE: se recorre `keep`, no range(N), y se RENUMERA con out_idx: los IDs
# de camara y los nombres tienen que quedar consecutivos porque algunos lectores
# de 2DGS asumen que no hay huecos.
for out_idx, i in enumerate(keep):
    im = rgbimgs[i]
    H, W = im.shape[:2]              # tamaño a 512px (referencia de aspecto/encuadre)
    aspect = W / float(H)
    name = "img_%04d.png" % out_idx
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
    cam_id = out_idx + 1
    fcam.write("%d PINHOLE %d %d %.6f %.6f %.6f %.6f\n" % (cam_id, Wsave, Hsave, fx, fy, cx, cy))
    # COLMAP guarda world->cam = inversa de cam->world (poses NO cambian con la resolución)
    w2c = np.linalg.inv(cams2world[i])
    q = Rotation.from_matrix(w2c[:3, :3]).as_quat()   # [x,y,z,w]
    t = w2c[:3, 3]
    fimg.write("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %d %s\n" %
               (cam_id, float(q[3]), float(q[0]), float(q[1]), float(q[2]),
                float(t[0]), float(t[1]), float(t[2]), cam_id, name))
    fimg.write("\n")   # linea de puntos 2D (vacia)
print("ENTRENAMIENTO: %d/%d imagenes guardadas en alta resolucion"
      % (_n_hi, len(keep)), flush=True)
fcam.close()
fimg.close()
print("MAST3R: poses escritas", flush=True)

# nube de puntos densa con color, para inicializar 2DGS
# OJO: esto usa las N vistas A PROPOSITO, no solo las de `keep`. La dedup es
# solo para el ENTRENAMIENTO; la geometria se beneficia de TODAS las fotos.
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
# MEDIDO en el render (45) contra la malla de Polycam:
#   Polycam: 20.7 m2 de superficie en 2 atlas 8192 -> 0.50 mm por texel
#   nuestro: 76.4 m2 (¡3.7x mas cuarto!) en 2 atlas 8192 -> 1.49 mm por texel
# Con el mismo presupuesto de textura repartido en 3.7x mas superficie, cada
# texel cubre 3x mas: por eso la etiqueta del frasco no se lee y la cama no
# se define. v9.5: OpenMVS empaqueta a 4096 (necesita mas atlas para la misma
# densidad) y el horneador los sube a 8192 -> 4x texeles = ~0.75 mm/texel.
# Reparto de atlas: se vuelve al del render (45) —probado, 2 atlas de 8192—
# para que ESTE render pruebe UNA sola cosa nueva: la correccion de tono.
# Para subir la densidad a ~0.75 mm/texel (el doble de fino) mas adelante:
#     OMVS_MAX_TEX=4096  OMVS_BAKE_SCALE=2  OMVS_BAKE_MAXATL=8
# Advertencia medida: eso pide ~8 atlas de 8192 -> archivo ~60 MB y ~2 GB de
# memoria de video al abrirlo. En celular puede no cargar. Polycam usa 2.
MAX_TEX       = int(os.environ.get("OMVS_MAX_TEX", "8192"))      # probado en el (45): 2 atlas
RES_LEVEL     = int(os.environ.get("OMVS_RES_LEVEL", "0"))       # 0 = usa las fotos tal cual se las paso
OUTLIER       = os.environ.get("OMVS_OUTLIER", "0.06")           # descarta fotos inconsistentes
SMOOTH_RATIO  = os.environ.get("OMVS_SMOOTH", "0.02")            # hacia 0 = parches GRANDES (investigacion: la escala va AL REVES; 1=mas fragmentado)
GLOBAL_SEAM   = os.environ.get("OMVS_GLOBAL_SEAM", "0")          # 0 = apagado: crashea (rc=-6) INCLUSO con malla manifold (probable choque con las caras virtuales). La nivelacion la hacemos nosotros (EXPO abajo)
LOCAL_SEAM    = os.environ.get("OMVS_LOCAL_SEAM", "0")           # 0 = apagado: sin base global escribe bandas negras (comprobado byte a byte)
SHARP         = os.environ.get("OMVS_SHARP", "0")                # 0 = apagado: el enfoque (default 0.5) crea halos oscuros en bordes de parches
VFACES        = os.environ.get("OMVS_VFACES", "3")               # caras virtuales coplanares: agrupa triangulos del mismo plano en parches GRANDES (el arreglo real de la fragmentacion)
# PACKH=0 ("mejor ajuste") COLGO el render del 23-jul: con decenas de miles de
# parches ese empaque es cuadratico y no termina nunca (22 min sin pasar de
# TextureMesh; el paso se habria cortado a los 40). Vuelve al 3 (el default de
# OpenMVS, el que corrio bien en el render 45). NO subir a 0 sin medir antes
# cuantos parches genera la malla.
PACKH         = os.environ.get("OMVS_PACKH", "3")                # 3 = buena velocidad (probado). 0 = mejor ajuste pero se cuelga con muchos parches
EXPOCOMP      = os.environ.get("OMVS_EXPOCOMP", "0") == "1"     # 0 = APAGADO: medido sobre el .glb real, EMPEORO el tono (dispersion 21.6 -> 34.0). Se deja por si acaso
TONE_LEVEL    = os.environ.get("OMVS_TONE", "0") == "1"          # SUPERADO por el horneador (v9.1): la mezcla multi-vista iguala el tono por construccion
TONE_CLAMP    = float(os.environ.get("OMVS_TONE_CLAMP", "1.35")) # tope de la correccion por isla (1.35 = +-35%): solo mueve el TONO, nunca el detalle
TONE_MINF     = int(os.environ.get("OMVS_TONE_MINF", "3"))       # caras minimas por costura para creerle
EXPO_SAMPLES  = int(os.environ.get("OMVS_EXPO_SAMPLES", "40000"))# puntos de la malla muestreados para medir las ganancias
OMP_HI        = os.environ.get("OMVS_OMP", "6")                  # hilos del intento bueno
# ── HORNEADOR MULTI-VISTA (v9.1; plan P1 de la investigacion, estilo Polycam) ──
BAKE          = os.environ.get("OMVS_BAKE", "1") == "1"          # repinta cada texel MEZCLANDO todas las fotos que lo ven
BAKE_SCALE    = int(os.environ.get("OMVS_BAKE_SCALE", "1"))      # 1 = el atlas de OpenMVS tal cual (probado). Con MAX_TEX=4096 sube esto a 2 para el doble de fino
BAKE_MAXATL   = int(os.environ.get("OMVS_BAKE_MAXATL", "5"))     # tope de atlas para permitir el x2 (5 atlas 8192 ~ 45-55 MB)
BAKE_DS       = int(os.environ.get("OMVS_BAKE_DS", "8"))         # banda baja = foto reducida /8 y devuelta (multiBandDownscale de AliceVision)
BAKE_COSK     = float(os.environ.get("OMVS_BAKE_COSK", "2"))     # peso angular cos^k (k=2 recomendado por la investigacion)
BAKE_TOL      = float(os.environ.get("OMVS_BAKE_TOL", "0.010"))  # tolerancia de visibilidad = 1% de la profundidad (0.6% rechazaba camaras buenas donde la malla erra 1-2 cm -> cobertura 70% y parches leves)
BAKE_BILIN    = os.environ.get("OMVS_BAKE_BILIN", "1") == "1"    # muestreo BILINEAL de las fotos (bordes de letras suaves, sin dientes)
BAKE_JQ       = int(os.environ.get("OMVS_BAKE_JQ", "85"))        # JPEG q85 + croma 4:2:0 (como Polycam): ~3x mas liviano que q92 4:4:4
BAKE_DILA     = int(os.environ.get("OMVS_BAKE_DILA", "6"))       # dilatacion del borde horneado (px) para mipmaps
BAKE_EXPO     = os.environ.get("OMVS_BAKE_EXPO", "1") == "1"     # normaliza la exposicion de cada foto a la mediana global antes de mezclar (mata el "distintos tonos" del auto-exposicion del celular)
BAKE_FIX      = os.environ.get("OMVS_BAKE_FIX", "1") == "1"      # a los texeles que ninguna foto ve les copia el TONO de sus vecinos horneados (mata las islas poligonales de tono ajeno)
BAKE_FIXBLUR  = int(os.environ.get("OMVS_BAKE_FIXBLUR", "9"))    # suavizado del campo de correccion (celdas de la rejilla gruesa)
BAKE_TONO     = os.environ.get("OMVS_BAKE_TONO", "1") == "1"   # APAGADO: fallo con V; se prueba aparte
# MEDIDO en la malla (61), tras el primer nivelado: la banda de 1-3 m bajo de
# 22.3 a 15.9 (-29%), PERO la que manda ahora es la de 5-30 cm, con 16.8. Con la
# escala en 1 metro ese rango quedaba fuera a proposito, para no borrar textura
# fina. Fue un error de diseno: 5-30 cm NO es textura fina, son PARCHES del
# tamano de una mano. Bajando a 0.30 m el filtro los agarra y deja intacto lo de
# menos de 5 cm (11.5), que si es el grano real de la pared.
# Si aplana de mas y la pared queda artificial: OMVS_TONO_ESC=0.50
BAKE_TONO_ESC = float(os.environ.get("OMVS_TONO_ESC", "0.30"))
BAKE_TONO_FZA = float(os.environ.get("OMVS_TONO_FZA", "1.0"))  # 1.0 = correccion completa
BAKE_TONO_TOPE= float(os.environ.get("OMVS_TONO_TOPE","1.6"))  # tope de la ganancia
# APAGADO por defecto. El intento del job d87eae06 EMPEORO el render: la
# compensacion se SATURO en el tope (+-40 niveles, o sea el sistema no
# convergio) y pinto la pared de retazos -> los cuadraditos.
# Aqui queda con tres arreglos, para encenderlo con OMVS_BAKE_SEAM=1:
#   1) deteccion EXACTA de costura (por UV en la arista compartida)
#   2) candado que aborta si >12% de aristas salen costura
#   3) anclaje 10x mas fuerte y tope 4x mas bajo, para que NO se sature
BAKE_SEAM     = os.environ.get("OMVS_BAKE_SEAM", "0") == "1"
BAKE_SEAM_MU  = float(os.environ.get("OMVS_SEAM_MU",  "0.05"))  # rigidez dentro del parche
BAKE_SEAM_LAM = float(os.environ.get("OMVS_SEAM_LAM", "0.20"))  # anclaje del offset
BAKE_SEAM_TOPE= float(os.environ.get("OMVS_SEAM_TOPE","10"))    # tope del offset en niveles
# MEDIDO: en TODOS los logs las ganancias salian 0.70-1.40, o sea LOS DOS
# extremos del recorte a la vez. Eso significa que a las fotos mas claras y
# mas oscuras se les queda diferencia SIN corregir, y como cada zona de la
# pared se hornea desde fotos distintas, esa diferencia aparece como parches
# de tono. Se abre el rango y ahora el log dice CUANTAS fotos tocan el limite.
# CORRECCION FOTOMETRICA (investigacion, prioridad 1)
# MEDIDO: las "figuras difusas de distinto tono" NO vienen de la geometria
# (correlacion brillo-vs-relieve r=+0.02..+0.10, o sea cero). Quedan dos causas
# fotometricas que la exposicion NO corrige, porque la exposicion es UNA ganancia
# escalar por foto:
#   1) BALANCE DE BLANCOS: el auto-WB del celular cambia foto a foto y deja
#      tintes distintos. Al mezclar 185 fotos por texel, esos tintes aparecen
#      como manchas suaves del mismo color pero distinto tono.
#   2) VINETEADO: el lente oscurece los bordes del cuadro. Cada zona de pared se
#      hornea desde fotos distintas y cae en zonas distintas del cuadro -> parches.
# Se corrigen aqui, ANTES de mezclar. Ninguna toca la geometria.
BAKE_WB       = os.environ.get("OMVS_BAKE_WB", "1") == "1"      # igualar el color entre fotos
BAKE_WB_TOPE  = float(os.environ.get("OMVS_WB_TOPE", "1.30"))   # tope de la correccion por canal
BAKE_VIG      = os.environ.get("OMVS_BAKE_VIG", "1") == "1"     # corregir el vineteado del lente
BAKE_VIG_TOPE = float(os.environ.get("OMVS_VIG_TOPE", "1.60"))  # tope de la correccion de borde
_EXPLO        = float(os.environ.get("OMVS_EXPO_MIN", "0.45"))
_EXPHI        = float(os.environ.get("OMVS_EXPO_MAX", "2.20"))
BAKE_VDOT     = int(os.environ.get("OMVS_BAKE_VDOT", "2"))       # radio del puntito para las caras de area cero (texeles)
BAKE_VFILL    = os.environ.get("OMVS_BAKE_VFILL", "1") == "1"     # tapa los parches vacios del atlas con el color por vertice de la malla (medido: 23% de las caras salian en gris plano)
BAKE_FIXDS    = int(os.environ.get("OMVS_BAKE_FIXDS", "16"))     # la correccion se calcula a 1/16 de resolucion (es de baja frecuencia): megas en vez de gigas
POSE_OPT      = os.environ.get("OMVS_POSEOPT", "0") == "1"       # Zhou-Koltun rigido (P2): OFF hasta medir su costo en el pod
POSE_ITERS    = int(os.environ.get("OMVS_POSE_ITERS", "60"))     # iteraciones si se enciende
POSE_VERTS    = int(os.environ.get("OMVS_POSE_VERTS", "400000")) # malla reducida para el optimizador (CPU)

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
CROPS = {}   # nombre.jpg -> (ruta_original_12MP, left, top, cw, ch, rw, rh): mismo recorte del paso 2, SIN reducir
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
    CROPS[jpg] = (path, left, top, cw, ch, rw, rh)
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
def _leer_sparse(spd):
    """Lee el sparse REESCALADO (cameras.txt PINHOLE + images.txt) -> camaras listas."""
    import numpy as _np
    cams2 = {}
    for l in open(os.path.join(spd, "cameras.txt")):
        if l.startswith("#") or not l.strip(): continue
        p = l.split()
        cams2[int(p[0])] = (int(p[2]), int(p[3]), float(p[4]), float(p[5]), float(p[6]), float(p[7]))
    vistas = []
    for l in open(os.path.join(spd, "images.txt")):
        if l.startswith("#") or len(l.split()) < 10: continue
        p = l.split()
        qw, qx, qy, qz = [float(x) for x in p[1:5]]
        t = _np.array([float(p[5]), float(p[6]), float(p[7])], _np.float64)
        cid = int(p[8]); nom = p[9]
        R = _np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]], _np.float64)
        vistas.append([nom, cid, R, t])
    return cams2, vistas

def _pose_opt_zhou():
    """P2 (Zhou-Koltun rigido, Open3D color_map): afina las poses ANTES de
    texturizar para enderezar las juntas onduladas. Reescribe images.txt con
    las poses refinadas. CPU; costo sin medir en el pod -> OFF por defecto."""
    import numpy as _np, open3d as _o3
    t0 = time.time()
    cams2, vistas = _leer_sparse(SPD)
    mm = _o3.io.read_triangle_mesh(MESH)
    if len(mm.triangles) > POSE_VERTS:
        mm = mm.simplify_quadric_decimation(target_number_of_triangles=POSE_VERTS)
    mm.compute_vertex_normals()
    scn = _o3.t.geometry.RaycastingScene()
    scn.add_triangles(_o3.t.geometry.TriangleMesh.from_legacy(mm))
    tray = _o3.camera.PinholeCameraTrajectory(); params = []
    rgbds = []
    DS2 = 3   # depth+color a 1/3 (480x640): suficiente y rapido
    for nom, cid, R, t in vistas:
        w, h, fx, fy, cx, cy = cams2[cid]
        w2, h2 = w//DS2, h//DS2
        fx2, fy2, cx2, cy2 = fx/DS2, fy/DS2, cx/DS2, cy/DS2
        C = -R.T @ t
        xs = (_np.arange(w2)+0.5-cx2)/fx2; ys = (_np.arange(h2)+0.5-cy2)/fy2
        gx, gy = _np.meshgrid(xs, ys)
        dirs_c = _np.stack([gx, gy, _np.ones_like(gx)], -1).reshape(-1,3)
        dirs_w = dirs_c @ R
        rays = _np.concatenate([_np.broadcast_to(C, dirs_w.shape), dirs_w], 1).astype(_np.float32)
        th = scn.cast_rays(_o3.core.Tensor(rays))["t_hit"].numpy().reshape(h2, w2)
        depth = _np.where(_np.isfinite(th), th, 0.0).astype(_np.float32)
        col = Image.open(os.path.join(IMGD, nom)).convert("RGB").resize((w2, h2), Image.BILINEAR)
        rgbds.append(_o3.geometry.RGBDImage.create_from_color_and_depth(
            _o3.geometry.Image(_np.asarray(col)),
            _o3.geometry.Image(depth), depth_scale=1.0,
            depth_trunc=50.0, convert_rgb_to_intensity=False))
        pc = _o3.camera.PinholeCameraParameters()
        pc.intrinsic = _o3.camera.PinholeCameraIntrinsic(w2, h2, fx2, fy2, cx2, cy2)
        E = _np.eye(4); E[:3,:3] = R; E[:3,3] = t
        pc.extrinsic = E
        params.append(pc)
    tray.parameters = params
    opt = _o3.pipelines.color_map.RigidOptimizerOption(
        maximum_iteration=POSE_ITERS,
        maximum_allowable_depth=50.0,
        depth_threshold_for_visibility_check=0.03)
    _, tray2 = _o3.pipelines.color_map.run_rigid_optimizer(mm, rgbds, tray, opt)
    # reescribir images.txt con las poses refinadas
    lineas = ["# Image list (poses refinadas Zhou-Koltun)\n"]
    for (nom, cid, _R, _t), pc in zip(vistas, tray2.parameters):
        E = _np.asarray(pc.extrinsic); R2 = E[:3,:3]; t2 = E[:3,3]
        tr = _np.trace(R2)
        if tr > 0:
            S = (tr+1.0)**0.5*2; qw=0.25*S; qx=(R2[2,1]-R2[1,2])/S; qy=(R2[0,2]-R2[2,0])/S; qz=(R2[1,0]-R2[0,1])/S
        else:
            i0 = int(_np.argmax([R2[0,0],R2[1,1],R2[2,2]]))
            if i0==0:
                S=(1+R2[0,0]-R2[1,1]-R2[2,2])**0.5*2; qw=(R2[2,1]-R2[1,2])/S; qx=0.25*S; qy=(R2[0,1]+R2[1,0])/S; qz=(R2[0,2]+R2[2,0])/S
            elif i0==1:
                S=(1-R2[0,0]+R2[1,1]-R2[2,2])**0.5*2; qw=(R2[0,2]-R2[2,0])/S; qx=(R2[0,1]+R2[1,0])/S; qy=0.25*S; qz=(R2[1,2]+R2[2,1])/S
            else:
                S=(1-R2[0,0]-R2[1,1]+R2[2,2])**0.5*2; qw=(R2[1,0]-R2[0,1])/S; qx=(R2[0,2]+R2[2,0])/S; qy=(R2[1,2]+R2[2,1])/S; qz=0.25*S
        lineas.append("%d %.9f %.9f %.9f %.9f %.7f %.7f %.7f %d %s\n\n"
                      % (cid, qw, qx, qy, qz, t2[0], t2[1], t2[2], cid, nom))
    open(os.path.join(SPD, "images.txt"), "w").writelines(lineas)
    log("POSE-OPT Zhou-Koltun: %d vistas refinadas en %.1f min (iters=%d)"
        % (len(vistas), (time.time()-t0)/60.0, POSE_ITERS))

if POSE_OPT:
    try:
        _pose_opt_zhou()
    except Exception as _po:
        log("POSE-OPT fallo (%s): sigo con las poses de MASt3R" % _po)
else:
    log("POSE-OPT apagado (OMVS_POSEOPT=0): poses de MASt3R tal cual")

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
    def _absxp(a, gpu):
        return a.abs() if gpu else _np.abs(a)
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

    # ── 1) PARCHES = caras unidas por (MISMO vertice 3D + MISMA coordenada UV) ──
    #   v8.9 fallo: uni por INDICE de UV, pero OpenMVS repite las UV por cara
    #   (cada cara trae su propio vt) -> cada cara salia como un parche y 0
    #   costuras. Aqui suelto las UV por VALOR (no por indice) y exijo ademas
    #   que compartan el vertice 3D: eso ES el parche. Verificado sobre el .glb
    #   real: da ~116k parches y ~62k pares de vecinos (antes: 1.09M y 0).
    uvq = _np.round(Tn, 6)
    _, uid = _np.unique(_np.c_[uvq, _np.zeros(len(uvq))], axis=0, return_inverse=True)
    NU = int(uid.max()) + 2
    cf = _np.repeat(_np.arange(NF), 3)
    cv = F.reshape(-1)                                   # vertice 3D (el OBJ SI los comparte)
    cu = uid[FT.reshape(-1)] + _np.repeat(FM, 3) * NU    # UV soldada, separada por atlas
    key = cv.astype(_np.int64) * (NU * (FM.max() + 2)) + cu
    o = _np.argsort(key, kind="stable"); key_s = key[o]; cf_s = cf[o]
    st = _np.r_[0, _np.flatnonzero(_np.diff(key_s)) + 1]
    msk = _np.ones(len(cf_s), bool); msk[st] = False
    pos = _np.flatnonzero(msk)
    NISL, isl = _cc(_sp.coo_matrix((_np.ones(len(pos), _np.int8), (cf_s[pos], cf_s[pos-1])),
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

    # ── 3) costuras: vertices 3D donde se tocan DOS parches (vectorizado) ──
    g = okf[cf]
    cvg = cv[g]; cpg = isl[cf[g]]; cfg = cf[g]
    o2 = _np.lexsort((cpg, cvg)); cvg = cvg[o2]; cpg = cpg[o2]; cfg = cfg[o2]
    lg = _np.log(_np.maximum(col, 1e-4))
    nuevo = _np.r_[True, (_np.diff(cvg) != 0) | (_np.diff(cpg) != 0)]
    gs = _np.flatnonzero(nuevo)
    gcnt = _np.diff(_np.r_[gs, len(cvg)])
    gmean = _np.add.reduceat(lg[cfg], gs) / gcnt[:, None]     # color medio por (vertice,parche)
    gv = cvg[gs]; gp = cpg[gs]
    vs = _np.flatnonzero(_np.r_[True, _np.diff(gv) != 0])
    ve = _np.r_[vs[1:], len(gv)]
    npat = ve - vs
    A_l = []; B_l = []; D_l = []
    two = _np.flatnonzero(npat == 2)                          # el caso comun: vectorizado
    if len(two):
        i0 = vs[two]; i1 = i0 + 1
        A_l.append(gp[i0]); B_l.append(gp[i1]); D_l.append(gmean[i1] - gmean[i0])
    for gi in _np.flatnonzero(npat > 2):                      # los pocos con 3+ parches
        a, b = vs[gi], ve[gi]
        for x in range(a, b):
            for y in range(x + 1, b):
                A_l.append(_np.array([gp[x]])); B_l.append(_np.array([gp[y]]))
                D_l.append((gmean[y] - gmean[x])[None, :])
    if not A_l:
        log("TONO: no encontre costuras entre parches; textura sin tocar"); return False
    A = _np.concatenate(A_l); B = _np.concatenate(B_l); D = _np.concatenate(D_l)
    sw = A > B
    A2 = _np.where(sw, B, A); B2 = _np.where(sw, A, B)
    D = _np.where(sw[:, None], -D, D)
    D = _np.clip(D, -0.7, 0.7)
    pk = A2 * NISL + B2
    uk, inv = _np.unique(pk, return_inverse=True)
    cnt = _np.bincount(inv).astype(_np.float64)
    sums = _np.stack([_np.bincount(inv, weights=D[:, c]) for c in range(3)], 1)
    keep = cnt >= TONE_MINF
    PA = (uk[keep] // NISL).astype(_np.int64); PB = (uk[keep] % NISL).astype(_np.int64)
    PD = sums[keep] / cnt[keep][:, None]; PN = cnt[keep]
    NP = len(PA)
    tocados = _np.zeros(NISL, bool); tocados[PA] = True; tocados[PB] = True
    cover = 100.0 * tocados[isl].mean()
    log("TONO: %d parches, %d costuras entre vecinos, cubren el %.0f%% de las caras"
        % (NISL, NP, cover))
    if NP < 500 or cover < 20.0:
        log("TONO: cobertura insuficiente -> NO nivelo (textura sin tocar)"); return False

    # ── 4) una ganancia por canal por parche ──
    ri = _np.repeat(_np.arange(NP), 2)
    ci = _np.empty(NP * 2, _np.int64); dv = _np.empty(NP * 2, _np.float64)
    w = _np.minimum(PN, 60.0) ** 0.5
    ci[0::2] = PA; ci[1::2] = PB; dv[0::2] = w; dv[1::2] = -w
    lam = 0.05
    M = _sp.vstack([_sp.coo_matrix((dv, (ri, ci)), shape=(NP, NISL)),
                    _sp.identity(NISL, format="coo") * lam]).tocsr()
    G = _np.ones((NISL, 3))
    for c in range(3):
        b = _np.r_[PD[:, c] * w, _np.zeros(NISL)]
        G[:, c] = _np.exp(_spl.lsqr(M, b, atol=1e-7, btol=1e-7, iter_lim=600)[0])
    lo, hi = 1.0 / TONE_CLAMP, TONE_CLAMP
    nclamp = int(((G < lo) | (G > hi)).sum()); G = _np.clip(G, lo, hi)
    lgG = _np.log(G).mean(1)
    d0 = _np.abs(PD.mean(1)); d1 = _np.abs(PD.mean(1) - (lgG[PA] - lgG[PB]))
    red = 100.0 * (1.0 - d1.mean() / max(d0.mean(), 1e-9))

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


def bake_multiview(objf, texfiles, mtl2tex):
    """HORNEADOR MULTI-VISTA v9.2 (P1 de la investigacion; el metodo Polycam).

    salida = BAJA (promedio ponderado de TODAS las fotos borrosas: tono parejo
             por construccion) + ALTA (mejor foto nitida - su borrosa: detalle
             sin fantasmas). Fotos 12MP, atlas x2 -> ~0.075 cm/texel.

    v9.1 MURIO EN EL POD sin decir nada tras rasterizar 120M texeles: el
    sistema mato el proceso (memoria). v9.2 = misma matematica (validada en
    sintetico: escalon de tono 1.8 niveles vs 20-36 del mejor-vista), pero:
      - pico de RAM recortado: numeros compactos (int32/f16/f32), trozos por
        PRESUPUESTO de muestras, SIN deduplicado global (duplicados raros e
        inofensivos: se escriben dos veces con el mismo valor);
      - LATIDOS en el log con la RAM usada en cada fase y cada 16 camaras:
        si vuelve a morir, el log dice exactamente donde y con cuanto;
      - si la GPU se queda corta, reintenta con trozos a la mitad;
      - triangulos grandes ya no quedan con huecos (tope de muestras alto).
    Si algo falla: False y los atlas de OpenMVS quedan intactos."""
    import numpy as _np, gc as _gc
    from scipy import ndimage as _ndi
    _t0 = time.time()

    def _rss():
        try:
            for _l in open("/proc/self/status"):
                if _l.startswith("VmRSS"): return int(_l.split()[1])/1048576.0
        except Exception: pass
        return -1.0
    def _s2l(c): return _np.where(c <= 0.04045, c/12.92, ((c+0.055)/1.055)**2.4)
    def _l2s(c):
        c = _np.clip(c, 0.0, 1.0)
        return _np.where(c <= 0.0031308, c*12.92, 1.055*(c**(1.0/2.4)) - 0.055)
    def _absxp(a, gpu):
        return a.abs() if gpu else _np.abs(a)

    # ── 1) leer el OBJ ──
    Vn=[]; Tn=[]; F=[]; FT=[]; FM=[]; cur=-1
    with open(objf) as fh:
        for ln in fh:
            if ln.startswith("v "):
                p=ln.split(); Vn.append((float(p[1]),float(p[2]),float(p[3])))
            elif ln.startswith("vt "):
                p=ln.split(); Tn.append((float(p[1]),float(p[2])))
            elif ln.startswith("usemtl"):
                cur=mtl2tex.get(ln.split(None,1)[1].strip(),-1)
            elif ln.startswith("f "):
                p=ln.split()
                if len(p)<4: continue
                a=[];b=[]
                for c in p[1:4]:
                    q=c.split("/"); a.append(int(q[0])-1)
                    b.append(int(q[1])-1 if len(q)>1 and q[1] else -1)
                F.append(a); FT.append(b); FM.append(cur)
    V=_np.asarray(Vn,_np.float64); F=_np.asarray(F,_np.int64)
    FT=_np.asarray(FT,_np.int64); FM=_np.asarray(FM,_np.int64); Tn=_np.asarray(Tn,_np.float64)
    del Vn; _gc.collect()
    NF=len(F)
    if NF<1000 or len(Tn)==0 or (FT<0).any() or (FM<0).any():
        log("BAKE: OBJ sin UVs utilizables o muy chico; no horneo"); return False
    V32=V.astype(_np.float32)
    e1=V32[F[:,1]]-V32[F[:,0]]; e2=V32[F[:,2]]-V32[F[:,0]]
    FN=_np.cross(e1,e2); FN/= (_np.linalg.norm(FN,axis=1,keepdims=True)+1e-12)
    FN16=FN.astype(_np.float16); del e1,e2,FN; _gc.collect()

    # ── 2) camaras del sparse reescalado ──
    cams2, vistas = _leer_sparse(SPD)
    if not vistas:
        log("BAKE: sin camaras en el sparse; no horneo"); return False

    # ── 3) voto de orientacion V contra el relleno gris ──
    votes=[0,0]; cuv=Tn[FT].mean(1)
    for ti in range(len(texfiles)):
        m=FM==ti
        if not m.any(): continue
        a=_np.asarray(Image.open(texfiles[ti]).convert("RGB"))
        H0,W0=a.shape[:2]
        fill=(_np.abs(a[:,:,0].astype(_np.int16)-128)<6)&(_np.abs(a[:,:,1].astype(_np.int16)-128)<6)&(_np.abs(a[:,:,2].astype(_np.int16)-128)<6)
        px=_np.clip((cuv[m,0]*(W0-1)).astype(int),0,W0-1)
        for fl in (0,1):
            vv=(1.0-cuv[m,1]) if fl==0 else cuv[m,1]
            py=_np.clip((vv*(H0-1)).astype(int),0,H0-1)
            votes[fl]+=int((~fill[py,px]).sum())
        del a,fill
    flip=0 if votes[0]>=votes[1] else 1
    del cuv; _gc.collect()
    log("BAKE: arranque (v9.4 loop unificado) | RAM %.1f GB" % _rss())

    # ── 4) tabla de texeles por splat (trozos por PRESUPUESTO de muestras) ──
    SC=max(1,BAKE_SCALE)
    if SC>1 and len(texfiles)>BAKE_MAXATL:
        log("BAKE: OpenMVS devolvio %d atlas (tope %d para el x%d) -> horneo a escala 1 "
            "para no producir un archivo gigante" % (len(texfiles),BAKE_MAXATL,SC))
        SC=1
    at_lin=[]; at_pos=[]; at_nrm=[]; at_id=[]; at_dims=[]
    _NMU=0; _NBIG=0; _MUBIG=0
    PRESU=25_000_000
    for ti in range(len(texfiles)):
        with Image.open(texfiles[ti]) as _im0: W0,H0=_im0.size
        W2,H2=W0*SC,H0*SC; at_dims.append((W2,H2))
        m=_np.flatnonzero(FM==ti)
        if len(m)==0: continue
        u=Tn[FT[m]].astype(_np.float32)
        pu=(u[:,:,0]*(W2-1)).astype(_np.float32)
        pv=(((1.0-u[:,:,1]) if flip==0 else u[:,:,1])*(H2-1)).astype(_np.float32)
        del u
        area2=_np.abs((pu[:,1]-pu[:,0])*(pv[:,2]-pv[:,0])-(pu[:,2]-pu[:,0])*(pv[:,1]-pv[:,0]))
        ns=_np.clip((area2*2.0).astype(_np.int64)+3,3,400000)
        _NMU+=int(ns.sum()); _bg=ns>10000
        _NBIG+=int(_bg.sum()); _MUBIG+=int(ns[_bg].sum())
        cs=_np.cumsum(ns)
        f0=0
        while f0 < len(m):
            f1=int(_np.searchsorted(cs, (cs[f0-1] if f0>0 else 0)+PRESU))+1
            f1=min(max(f1,f0+1),len(m))
            nrep=ns[f0:f1]
            fid=_np.repeat(_np.arange(f0,f1,dtype=_np.int64),nrep)
            S=len(fid)
            r1=_np.random.rand(S).astype(_np.float32); r2=_np.random.rand(S).astype(_np.float32)
            sq=_np.sqrt(r1); ba=1-sq; bb=sq*(1-r2); bc=sq*r2
            del r1,r2,sq
            ix=_np.clip((ba*pu[fid,0]+bb*pu[fid,1]+bc*pu[fid,2]+0.5).astype(_np.int32),0,W2-1)
            iy=_np.clip((ba*pv[fid,0]+bb*pv[fid,1]+bc*pv[fid,2]+0.5).astype(_np.int32),0,H2-1)
            lin=iy*_np.int32(W2)+ix
            del ix,iy
            uq,first=_np.unique(lin,return_index=True)
            del lin
            gf=m[fid[first]]
            baf=ba[first,None]; bbf=bb[first,None]; bcf=bc[first,None]
            P=(baf*V32[F[gf,0]]+bbf*V32[F[gf,1]]+bcf*V32[F[gf,2]])
            at_lin.append(uq.astype(_np.int32)); at_pos.append(P)
            at_nrm.append(FN16[gf]); at_id.append(_np.full(len(uq),ti,_np.uint8))
            del fid,ba,bb,bc,baf,bbf,bcf,uq,first,gf,P
            f0=f1
        del pu,pv,area2,ns,cs; _gc.collect()
    if not at_lin:
        log("BAKE: no pude rasterizar texeles; no horneo"); return False
    LIN=_np.concatenate(at_lin); POS=_np.concatenate(at_pos)
    NRM=_np.concatenate(at_nrm); AID=_np.concatenate(at_id)
    del at_lin,at_pos,at_nrm,at_id
    if not BAKE_VFILL:
        del Tn,FT          # el parche de vacios los necesita al final
    _gc.collect()
    NT=len(LIN)
    log("BAKE: %d texeles rasterizados (con duplicados raros de borde, inofensivos) "
        "| RAM %.1f GB | %.1f min" % (NT,_rss(),(time.time()-_t0)/60.0))
    # DIAGNOSTICO DE LENTITUD (medido en la (61): 198M muestras para 50M texeles;
    # 1.822 caras de un millon pedian el 47% del trabajo. Eran ASTILLAS, la forma
    # que deja el cosido de huecos. Si esta linea vuelve a dispararse, el culpable
    # es HOLE_MAX otra vez.)
    if _NMU:
        log("BAKE: %.0fM muestras pedidas | %d caras piden >10k cada una (%.0f%% del "
            "trabajo). Sano: <5%%; si sube mucho, baja HOLE_MAX"
            % (_NMU/1e6,_NBIG,100*_MUBIG/max(_NMU,1)))

    # ── 5) mapas de profundidad (media resolucion, f16) ──
    import open3d as _o3
    scn=_o3.t.geometry.RaycastingScene()
    _mm=_o3.geometry.TriangleMesh(_o3.utility.Vector3dVector(V),_o3.utility.Vector3iVector(F.astype(_np.int32)))
    scn.add_triangles(_o3.t.geometry.TriangleMesh.from_legacy(_mm))
    DSD=2
    deps={}
    for nom,cid,R,t in vistas:
        w14,h14,fx,fy,cx,cy=cams2[cid]
        wD,hD=max(2,w14//DSD),max(2,h14//DSD)
        C=(-R.T@t)
        xs=(_np.arange(wD)+0.5-cx/DSD)/(fx/DSD); ys=(_np.arange(hD)+0.5-cy/DSD)/(fy/DSD)
        gx,gy=_np.meshgrid(xs,ys)
        dc=_np.stack([gx,gy,_np.ones_like(gx)],-1).reshape(-1,3)
        dw=dc@R
        rays=_np.concatenate([_np.broadcast_to(C,dw.shape),dw],1).astype(_np.float32)
        th=scn.cast_rays(_o3.core.Tensor(rays))["t_hit"].numpy().reshape(hD,wD)
        deps[nom]=_np.where(_np.isfinite(th),th,0.0).astype(_np.float16)
        del gx,gy,dc,dw,rays,th
    _Vkeep=V.astype(_np.float32)   # copia para el nivelado de tono (V se borra aqui)
    del scn,_mm,V; _gc.collect()
    log("BAKE: profundidades de %d camaras listas | RAM %.1f GB | %.1f min"
        % (len(vistas),_rss(),(time.time()-_t0)/60.0))

    # ── 6) acumulacion UNIFICADA (un solo cuerpo; adaptador GPU/CPU) ──
    # Una sola implementacion corre en GPU (torch) o CPU (numpy) via un
    # adaptador minimo. El test local en CPU ejercita EXACTAMENTE estas
    # lineas -> se acaba la clase de bug "la rama GPU no estaba probada"
    # (que costo 2 renders: el OOM silencioso y el 'i12u before assignment').
    usa_gpu=False
    try:
        import torch as _th
        usa_gpu=_th.cuda.is_available() and os.environ.get("OMVS_BAKE_CPU","0")!="1"
    except Exception:
        _th=None

    class _XPnp:                      # adaptador NumPy (CPU)
        name="CPU"
        def zeros(self,shp,f16=False): return _np.zeros(shp,_np.float16 if f16 else _np.float32)
        def zc(self,n): return _np.zeros(n,_np.uint8)
        def put(self,a): return a     # ya es numpy
        def get(self,a): return a
        def idx(self,a): return a.astype(_np.int64)
        def clipi(self,a,hi): return _np.clip(a.astype(_np.int64),0,hi)
        def where(self,c,a,b): return _np.where(c,a,b)
        def zeros_like(self,a): return _np.zeros_like(a)
        def col(self,a): return a[:,None]
        def af(self,a): return a.astype(_np.float32)
        def norm1(self,v): return _np.linalg.norm(v,axis=1,keepdims=True)
        def u8add(self,c,m): return _np.minimum(c.astype(_np.int32)+m,250).astype(_np.uint8)
        def f16(self,a): return a.astype(_np.float16)

    class _XPth:                      # adaptador Torch (GPU)
        name="GPU"
        def __init__(self,dev): self.d=dev
        def zeros(self,shp,f16=False): return _th.zeros(shp,dtype=_th.float16 if f16 else _th.float32,device=self.d)
        def zc(self,n): return _th.zeros(n,dtype=_th.uint8,device=self.d)
        def put(self,a): return _th.from_numpy(_np.ascontiguousarray(a)).to(self.d)
        def get(self,a): return a.detach().cpu().numpy()
        def idx(self,a): return a.long()
        def clipi(self,a,hi): return _th.clamp(a.long(),0,hi)
        def where(self,c,a,b): return _th.where(c,a,b)
        def zeros_like(self,a): return _th.zeros_like(a)
        def col(self,a): return a.unsqueeze(1)
        def af(self,a): return a.float()
        def norm1(self,v): return _th.clamp(v.norm(dim=1,keepdim=True),min=1e-9)
        def u8add(self,c,m): return c+(m.to(_th.uint8)*(c<250).to(_th.uint8))
        def f16(self,a): return a.half()

    xp = _XPth("cuda") if usa_gpu else _XPnp()
    sumL=xp.zeros((NT,3)); sumW=xp.zeros(NT); bestW=xp.zeros(NT)
    bestS=xp.zeros((NT,3),f16=True); bestL=xp.zeros((NT,3),f16=True); cnt=xp.zc(NT)
    POSg=xp.put(POS); NRMg=xp.put(NRM)
    if usa_gpu: del POS,NRM; _gc.collect()
    CH=[20_000_000 if usa_gpu else 4_000_000]
    log("BAKE: mezclando %d camaras en %s | RAM %.1f GB" % (len(vistas),xp.name,_rss()))

    # ── NORMALIZACION DE EXPOSICION por foto (BAKE_EXPO): el Xiaomi cambia la
    #    auto-exposicion foto a foto (medido: hasta 4x en este set). Antes de
    #    mezclar, se lleva cada foto a la MEDIANA global de luminancia lineal.
    #    Ataca "no hay armonia / distintos tonos" en la RAIZ, no solo tapa. ──
    _gain={}; _wb={}; _VIGC={}
    _vig_on=BAKE_VIG        # local: NO reasignar el global (bug del V)
    if BAKE_EXPO or BAKE_WB or _vig_on:
        _ms=[]; _rgb=[]
        _NR=24                      # bins radiales del perfil de vineteado
        _vsum=_np.zeros(_NR); _vcnt=_np.zeros(_NR)
        for nom,cid,_Rc,_tc in vistas:
            cr=CROPS.get(nom)
            try:
                if cr:
                    _p,_l,_t,_cw,_ch,_rw,_rh=cr
                    _q=Image.open(_p).convert("RGB").crop((_l,_t,_l+_cw,_t+_ch)).resize((160,213),Image.BILINEAR)
                else:
                    _q=Image.open(os.path.join(IMGD,nom)).convert("RGB").resize((160,213),Image.BILINEAR)
            except Exception:
                _q=Image.open(os.path.join(IMGD,nom)).convert("RGB").resize((160,213),Image.BILINEAR)
            _a=_s2l(_np.asarray(_q,_np.float32)/255.0)      # luz LINEAL
            _ms.append((nom,float(_np.median(_a))))
            # (1) color medio por canal -> para igualar el BALANCE DE BLANCOS
            _rgb.append((nom,_a.reshape(-1,3).mean(0)))
            # (2) perfil radial: brillo normalizado por el de la propia foto.
            #     Promediado sobre 185 fotos, el contenido de la escena se
            #     cancela y queda la caida del LENTE (flat-field retrospectivo).
            if _vig_on:
                _hq,_wq=_a.shape[:2]
                _yy,_xx=_np.mgrid[0:_hq,0:_wq]
                _rr=_np.sqrt(((_xx-(_wq-1)/2.0)/((_wq-1)/2.0))**2
                             +((_yy-(_hq-1)/2.0)/((_hq-1)/2.0))**2)
                _rr=_np.clip(_rr/_np.sqrt(2.0),0,0.999999)
                _lum=_a.mean(2); _mn=float(_lum.mean())
                if _mn>1e-6:
                    _bi=(_rr*_NR).astype(_np.int64)
                    _np.add.at(_vsum,_bi.ravel(),(_lum/_mn).ravel())
                    _np.add.at(_vcnt,_bi.ravel(),1.0)
        _med=_np.median([x[1] for x in _ms])
        for nom,mi in _ms:
            _gain[nom]=float(_np.clip(_med/max(mi,1e-4),_EXPLO,_EXPHI))
        _sp=_np.array([_gain[n] for n,_ in _ms])
        _nsat=int(((_sp<=_EXPLO+1e-6)|(_sp>=_EXPHI-1e-6)).sum())
        log("BAKE: exposicion normalizada | ganancias %.2f-%.2f (rango %.2f-%.2f) | "
            "%d de %d fotos TOCAN el limite"
            % (_sp.min(),_sp.max(),_EXPLO,_EXPHI,_nsat,len(_sp)))
        # ── BALANCE DE BLANCOS: llevar TODAS las fotos al color de la MEDIANA ──
        # No se fuerza "gris" (gray-world puro), que le cambiaria el color al
        # cuarto entero: se igualan entre si. Es lo que hace falta, porque el
        # defecto es que las fotos NO COINCIDEN, no que esten mal en absoluto.
        if BAKE_WB and _rgb:
            _M=_np.array([v for _,v in _rgb],_np.float64)      # (N,3)
            _M=_np.maximum(_M,1e-6)
            _rg=_M[:,0]/_M[:,1]; _bg=_M[:,2]/_M[:,1]           # razones al verde
            _rgm=float(_np.median(_rg)); _bgm=float(_np.median(_bg))
            _lo,_hi=1.0/BAKE_WB_TOPE,BAKE_WB_TOPE
            _nw=0
            for _i,(nom,_) in enumerate(_rgb):
                _gr=float(_np.clip(_rgm/max(_rg[_i],1e-6),_lo,_hi))
                _gb=float(_np.clip(_bgm/max(_bg[_i],1e-6),_lo,_hi))
                _wb[nom]=_np.array([_gr,1.0,_gb],_np.float32)
                if abs(_gr-1)>0.02 or abs(_gb-1)>0.02: _nw+=1
            _dr=float(_np.percentile(_rg,90)-_np.percentile(_rg,10))
            _db=float(_np.percentile(_bg,90)-_np.percentile(_bg,10))
            log("BAKE WB: %d de %d fotos con tinte distinto; dispersion R/G %.3f, B/G %.3f "
                "-> igualadas a la mediana" % (_nw,len(_rgb),_dr,_db))
        # ── VINETEADO: invertir el perfil radial medido ──
        if _vig_on and _vcnt.sum()>0:
            _ok=_vcnt>0
            _prof=_np.ones(_NR)
            _prof[_ok]=_vsum[_ok]/_vcnt[_ok]
            # OJO (bug cazado en la prueba sintetica): hay que suavizar ANTES de
            # normalizar, y rellenando el borde con el propio valor. Al reves,
            # convolve rellena con CEROS y hunde el bin del centro: el perfil
            # salia 0.67 en el centro cuando debe ser 1.00, y minimum.accumulate
            # congelaba ese error -> la correccion habria quedado invertida.
            _prof=_np.convolve(_np.pad(_prof,1,mode="edge"),
                               _np.ones(3)/3.0,mode="same")[1:-1]
            _prof=_prof/max(_prof[0],1e-6)                     # normalizar al centro
            _prof=_np.minimum.accumulate(_np.clip(_prof,0.2,1.0))
            _caida=float(1.0-_prof[-1])
            if _caida<0.02:
                _vig_on=False
                log("BAKE VIG: la caida al borde es solo %.1f%% -> no hace falta corregir" % (100*_caida))
            else:
                _corr=_np.clip(1.0/_np.maximum(_prof,1e-6),1.0,BAKE_VIG_TOPE)
                _VIGC["prof"]=_corr
                log("BAKE VIG: el borde del cuadro llega %.0f%% mas oscuro que el centro "
                    "-> corregido (esto pintaba parches segun donde caia cada foto)"
                    % (100*_caida))
        else:
            _vig_on=False

    for ki,(nom,cid,Rc,tc) in enumerate(vistas):
        if ki % 16 == 0 and ki:
            log("BAKE: camara %d/%d | RAM %.1f GB | %.1f min"
                % (ki,len(vistas),_rss(),(time.time()-_t0)/60.0))
        w14,h14,fx,fy,cx,cy=cams2[cid]
        cr=CROPS.get(nom)
        try:
            if cr:
                pth,left,top,cw,ch_,rw,rh=cr
                im=Image.open(pth).convert("RGB").crop((left,top,left+cw,top+ch_))
            else:
                im=Image.open(os.path.join(IMGD,nom)).convert("RGB")
        except Exception:
            im=Image.open(os.path.join(IMGD,nom)).convert("RGB")
        W12,H12=im.size; s12=W12/float(w14)
        low_im=im.resize((max(1,W12//BAKE_DS),max(1,H12//BAKE_DS)),Image.LANCZOS).resize((W12,H12),Image.BILINEAR)
        g_i=_gain.get(nom,1.0)
        # exposicion (escalar) x balance de blancos (por canal)
        _gv=_np.float32([g_i,g_i,g_i])
        if BAKE_WB and nom in _wb: _gv=_gv*_wb[nom]
        sharp=_s2l(_np.asarray(im,_np.float32)/255.0)*_gv[None,None,:]; del im
        low=_s2l(_np.asarray(low_im,_np.float32)/255.0)*_gv[None,None,:]; del low_im
        # vineteado: mapa radial, calculado UNA vez por tamano de foto y cacheado
        if _vig_on and "prof" in _VIGC:
            _k=(sharp.shape[1],sharp.shape[0])
            if _k not in _VIGC:
                _wv,_hv=_k
                _yv,_xv=_np.mgrid[0:_hv,0:_wv]
                _rv=_np.sqrt(((_xv-(_wv-1)/2.0)/((_wv-1)/2.0))**2
                             +((_yv-(_hv-1)/2.0)/((_hv-1)/2.0))**2)
                _rv=_np.clip(_rv/_np.sqrt(2.0),0,0.999999)
                _pr=_VIGC["prof"]
                _fi=_rv*(len(_pr)-1)
                _i0=_np.floor(_fi).astype(_np.int32); _i1=_np.minimum(_i0+1,len(_pr)-1)
                _fr=(_fi-_i0).astype(_np.float32)
                _VIGC[_k]=(_pr[_i0]*(1-_fr)+_pr[_i1]*_fr).astype(_np.float32)
                del _yv,_xv,_rv,_fi,_i0,_i1,_fr
            _vm=_VIGC[_k][:,:,None]
            sharp=sharp*_vm; low=low*_vm
        dep=deps[nom].astype(_np.float32); hD,wD=dep.shape; sD=wD/float(w14)
        Cc=(-Rc.T@tc).astype(_np.float32)
        # subir esta camara al backend
        Rq=xp.put(Rc.astype(_np.float32)); tq=xp.put(tc.astype(_np.float32)); Cq=xp.put(Cc)
        shq=xp.f16(xp.put(sharp)); lwq=xp.f16(xp.put(low)); dpq=xp.put(dep)
        del sharp,low,dep
        c0=0
        while c0<NT:
            sl=slice(c0,min(c0+CH[0],NT))
            try:
                P=POSg[sl]; N=xp.af(NRMg[sl])
                Xc=P@Rq.T+tq; z=Xc[:,2]
                zc=z if usa_gpu else _np.maximum(z,1e-6)
                ok=z>0.05
                u=fx*Xc[:,0]/zc+cx; v=fy*Xc[:,1]/zc+cy
                ok=ok&(u>=0)&(u<w14-1)&(v>=0)&(v<h14-1)
                iu=xp.clipi(u*sD,wD-1); iv=xp.clipi(v*sD,hD-1)
                d=dpq[iv,iu]
                ok=ok&(d>0)&(xp.af(_absxp(z-d,usa_gpu))<BAKE_TOL*z)
                dv=P-Cq; dv=dv/xp.norm1(dv)
                cosv=_absxp((dv*N).sum(1),usa_gpu)
                w=xp.where(ok,(cosv**BAKE_COSK)/(z*z+1e-6),xp.zeros_like(z))
                # muestreo (bilineal por defecto) de las fotos 12MP
                fu=u*s12; fv=v*s12
                if usa_gpu:
                    fu=_th.clamp(fu,0.0,W12-1.001); fv=_th.clamp(fv,0.0,H12-1.001)
                else:
                    fu=_np.clip(fu,0.0,W12-1.001); fv=_np.clip(fv,0.0,H12-1.001)
                x0=xp.idx(fu); y0=xp.idx(fv)
                if BAKE_BILIN:
                    ax=xp.col(fu-xp.af(x0)); ay=xp.col(fv-xp.af(y0))
                    x1=xp.clipi(x0+1,W12-1); y1=xp.clipi(y0+1,H12-1)
                    cS=((xp.af(shq[y0,x0])*(1-ax)+xp.af(shq[y0,x1])*ax)*(1-ay)
                        +(xp.af(shq[y1,x0])*(1-ax)+xp.af(shq[y1,x1])*ax)*ay)
                    cL=((xp.af(lwq[y0,x0])*(1-ax)+xp.af(lwq[y0,x1])*ax)*(1-ay)
                        +(xp.af(lwq[y1,x0])*(1-ax)+xp.af(lwq[y1,x1])*ax)*ay)
                else:
                    cS=xp.af(shq[y0,x0]); cL=xp.af(lwq[y0,x0])
                sumL[sl]=sumL[sl]+xp.col(w)*cL; sumW[sl]=sumW[sl]+w
                cnt[sl]=xp.u8add(cnt[sl],(w>0))
                mej=w>bestW[sl]
                bestW[sl]=xp.where(mej,w,bestW[sl])
                bestS[sl]=xp.where(xp.col(mej),xp.f16(cS),bestS[sl])
                bestL[sl]=xp.where(xp.col(mej),xp.f16(cL),bestL[sl])
                c0=sl.stop
            except RuntimeError as _oe:
                if usa_gpu and "out of memory" in str(_oe).lower() and CH[0]>2_000_000:
                    _th.cuda.empty_cache(); CH[0]//=2
                    log("BAKE: GPU corta de memoria -> trozos a %.0fM" % (CH[0]/1e6))
                else:
                    raise
        del Rq,tq,Cq,shq,lwq,dpq
    del deps; _gc.collect()
    if usa_gpu:
        sumL=sumL.cpu().numpy(); sumW=sumW.cpu().numpy(); bestW=bestW.cpu().numpy()
        bestS=bestS.cpu().numpy(); bestL=bestL.cpu().numpy(); cnt=cnt.cpu().numpy()
        del POSg,NRMg
        _th.cuda.empty_cache()
    log("BAKE: mezcla terminada | RAM %.1f GB | %.1f min" % (_rss(),(time.time()-_t0)/60.0))

    seen=(cnt>=1)&(sumW>1e-9)
    cov1=100.0*seen.mean(); cov3=100.0*(cnt>=3).mean()
    if cov1<30.0:
        log("BAKE: cobertura muy baja (%.0f%% con >=1 foto); no toco los atlas" % cov1); return False
    # componer POR TROZOS: v9.2 armaba aqui ~8 temporales del tamano completo
    # (el pico de 10.4 GB visto en el pod). Ahora: un solo buffer de salida
    # uint8 y trozos de 4M texeles con temporales diminutos.
    outs=_np.empty((NT,3),_np.uint8)
    for c0 in range(0,NT,4_000_000):
        sl=slice(c0,min(c0+4_000_000,NT))
        lb=sumL[sl]/_np.maximum(sumW[sl],1e-9)[:,None]
        ol=_np.clip(lb+(bestS[sl].astype(_np.float32)-bestL[sl].astype(_np.float32)),0.0,1.0)
        outs[sl]=(_np.clip(_l2s(ol),0,1)*255.0+0.5).astype(_np.uint8)
    del sumL,sumW,bestW,bestS,bestL; _gc.collect()
    log("BAKE: composicion lista | RAM %.1f GB" % _rss())

    # ── 7) escribir atlas: horneado donde hay fotos + CORRECCION DE TONO donde no ──
    #
    # PROBLEMA QUE ARREGLA (visto en el render 45): el horneador solo repinta
    # los texeles que alguna foto VE. El 5-15% restante (rincones, zonas
    # ocluidas, detras de muebles) se quedaba con el pixel CRUDO de OpenMVS
    # -> islas poligonales con el tono de UNA sola foto, en medio de la
    # superficie ya armonizada. Eso es lo que se ve como "figuras
    # geometricas" y "cicatrices/mesetas" en las paredes.
    #
    # SOLUCION: en vez de dejar el crudo, calculamos el campo de CORRECCION
    #   C = horneado - crudo      (solo donde SI hay horneado)
    # lo extendemos suavemente a todo el atlas (relleno por vecino mas
    # cercano + desenfoque) y se lo sumamos al crudo. Asi:
    #   - donde hay fotos: queda el horneado exacto
    #   - donde no: queda el crudo con el MISMO tono de sus vecinos, y
    #     conserva su detalle fino. No sobrevive ninguna isla de tono ajeno.
    _nfix=0; _nvac=0; _nvpx=0; _ndeg=0
    # color por vertice de la malla original (viene de las fotos): se mapea a
    # los vertices del OBJ de OpenMVS por posicion (el OBJ los renumera).
    VCOL=None
    if BAKE_VFILL:
        try:
            import open3d as _o3b
            from scipy.spatial import cKDTree as _KD
            _mo=_o3b.io.read_triangle_mesh(MESH)
            _vc=_np.asarray(_mo.vertex_colors)
            if len(_vc)==len(_mo.vertices) and len(_vc)>0:
                _kd=_KD(_np.asarray(_mo.vertices))
                _,_ii=_kd.query(V32,k=1,workers=-1)
                VCOL=(_np.clip(_vc[_ii],0,1)*255.0).astype(_np.float32)
                log("BAKE: color de la malla listo para tapar vacios (%d vertices)" % len(V32))
            else:
                log("BAKE: la malla no trae color por vertice; no puedo tapar vacios")
            del _mo,_vc
        except Exception as _vc_e:
            log("BAKE: no pude leer el color de la malla (%s)" % _vc_e)
    for ti,tf in enumerate(texfiles):
        W2,H2=at_dims[ti]
        base=_np.asarray(Image.open(tf).convert("RGB").resize((W2,H2),Image.BILINEAR)).copy()
        m=(AID==ti)&seen
        if m.any():
            lin=LIN[m].astype(_np.int64); iy=lin//W2; ix=lin%W2
            crudo=base[iy,ix].astype(_np.float32)
            base[iy,ix]=outs[m]
            filled=_np.zeros((H2,W2),bool); filled[iy,ix]=True
            # ── campo de correccion en REJILLA GRUESA ──
            # la correccion de tono es de baja frecuencia, asi que se calcula
            # a 1/BAKE_FIXDS de resolucion: cuesta megas en vez de gigas
            # (a resolucion completa el mapa de distancias pedia >3 GB y el
            #  sistema mataba el proceso).
            # se guarda la SUMA del horneado y la SUMA del crudo por celda:
            # la correccion es una RAZON (ganancia), no una resta. Una
            # diferencia de exposicion entre fotos es multiplicativa: sumar
            # un numero fijo arregla los tonos medios y estropea los claros
            # y oscuros (medido: sumar quitaba 31% de las islas, multiplicar
            # las quita casi todas).
            DS=max(1,BAKE_FIXDS); GW=(W2+DS-1)//DS; GH=(H2+DS-1)//DS
            ghor=_np.zeros((GH,GW,3),_np.float32); gcru=_np.zeros((GH,GW,3),_np.float32)
            gcnt=_np.zeros((GH,GW),_np.float32)
            gy_=iy//DS; gx_=ix//DS
            _oh=outs[m].astype(_np.float32)
            for _c in range(3):
                _np.add.at(ghor[:,:,_c],(gy_,gx_),_oh[:,_c])
                _np.add.at(gcru[:,:,_c],(gy_,gx_),crudo[:,_c])
            _np.add.at(gcnt,(gy_,gx_),1.0)
            del lin,iy,ix,crudo,_oh,gy_,gx_
            # dilatacion normal (borde de parche, para mipmaps)
            for _ in range(BAKE_DILA):
                nb=filled[1:,:]&~filled[:-1,:]; base[:-1][nb]=base[1:][nb]; filled[:-1][nb]=True
                nb=filled[:-1,:]&~filled[1:,:]; base[1:][nb]=base[:-1][nb]; filled[1:][nb]=True
                nb=filled[:,1:]&~filled[:,:-1]; base[:,:-1][nb]=base[:,1:][nb]; filled[:,:-1][nb]=True
                nb=filled[:,:-1]&~filled[:,1:]; base[:,1:][nb]=base[:,:-1][nb]; filled[:,1:][nb]=True
            # ── que texeles quedaron con el color CRUDO de OpenMVS ──
            # OJO (bug de v9.5-a): antes se usaba la lista de texeles
            # rasterizados (LIN) para saber que es superficie. Pero LIN trae
            # repetidos y NO cubre todos los texeles de cada parche, asi que
            # la correccion solo llegaba a la MITAD de los texeles crudos
            # (medido: 21.7M crudos, solo 10.7M corregidos). Ahora se detecta
            # al reves y sin huecos: todo lo que NO es el relleno gris vacio
            # de OpenMVS y NO lo repinto el horneador, es superficie cruda.
            # (por FRANJAS: convertir el atlas entero a int16 pedia 0.4 GB de golpe)
            sinh=_np.zeros((H2,W2),bool)
            for _r0 in range(0,H2,512):
                _r1=min(_r0+512,H2); _b=base[_r0:_r1]
                _g=((_b[:,:,0]>=125)&(_b[:,:,0]<=131)&(_b[:,:,1]>=125)&(_b[:,:,1]<=131)
                    &(_b[:,:,2]>=125)&(_b[:,:,2]<=131))
                sinh[_r0:_r1]=(~filled[_r0:_r1])&(~_g)
                del _g,_b
            nsin=int(sinh.sum())
            if nsin>0 and BAKE_FIX:
                gok=gcnt>=4          # celdas con muestras suficientes
                if gok.any():
                    gan=_np.ones_like(ghor)
                    _num=ghor[gok]+8.0; _den=gcru[gok]+8.0
                    gan[gok]=_np.clip(_num/_den,0.45,2.2)
                    # extender la ganancia a las celdas sin muestras (vecino mas cercano)
                    gi=_ndi.distance_transform_edt(~gok,return_distances=False,return_indices=True)
                    gan=gan[gi[0],gi[1]]
                    # suavizar: un degradado de ganancia, no calcos con borde
                    gan=_ndi.uniform_filter(gan,size=(BAKE_FIXBLUR,BAKE_FIXBLUR,1))
                    # MEDIDO en la (61): hasta aqui todo bien, pero mas abajo la
                    # ganancia se ampliaba de la rejilla (1/16) al atlas por
                    # VECINO MAS CERCANO -> cada celda de 16x16 texeles quedaba
                    # con un tono plano distinto del vecino. En el atlas real eso
                    # deja un escalon EXTRA en los limites de 16, por encima del
                    # bloqueo de JPEG: 6.017 vs 5.709 (atlas 1) y 2.827 vs 2.534
                    # (atlas 2), igual en horizontal y vertical. Son LOS
                    # CUADRADITOS, sobre 41.9M de texeles (31% del atlas).
                    # Aqui se prepara la version SUAVE: coordenadas fraccionarias
                    # para interpolar bilinealmente al aplicar.
                    # aplicar SOLO donde ninguna foto vio (el horneado no se toca),
                    # por FRANJAS: sacar las coordenadas de 20M de texeles de
                    # golpe pedia 0.35 GB y mataba el proceso.
                    # BILINEAL: el centro de la celda g cae en el texel
                    # g*DS + DS/2, asi que la coordenada de rejilla de un texel x
                    # es (x + 0.5)/DS - 0.5. Se interpola entre las dos celdas
                    # vecinas en vez de copiar la mas cercana -> la ganancia queda
                    # como un degradado continuo y desaparece el escalon de 16.
                    _fx=_np.clip((_np.arange(W2)+0.5)/DS-0.5,0,GW-1)
                    _x0g=_np.floor(_fx).astype(_np.int64)
                    _x1g=_np.minimum(_x0g+1,GW-1)
                    _tx=(_fx-_x0g).astype(_np.float32)[None,:,None]
                    for _r0 in range(0,H2,512):
                        _r1=min(_r0+512,H2)
                        _m2=sinh[_r0:_r1]
                        if not _m2.any(): continue
                        _fy=_np.clip((_np.arange(_r0,_r1)+0.5)/DS-0.5,0,GH-1)
                        _y0g=_np.floor(_fy).astype(_np.int64)
                        _y1g=_np.minimum(_y0g+1,GH-1)
                        _ty=(_fy-_y0g).astype(_np.float32)[:,None,None]
                        _g00=gan[_y0g[:,None],_x0g[None,:]]
                        _g01=gan[_y0g[:,None],_x1g[None,:]]
                        _g10=gan[_y1g[:,None],_x0g[None,:]]
                        _g11=gan[_y1g[:,None],_x1g[None,:]]
                        _gg=((_g00*(1-_tx)+_g01*_tx)*(1-_ty)
                             +(_g10*(1-_tx)+_g11*_tx)*_ty)
                        _blk=_np.clip(base[_r0:_r1].astype(_np.float32)*_gg,0,255).astype(_np.uint8)
                        base[_r0:_r1]=_np.where(_m2[:,:,None],_blk,base[_r0:_r1])
                        del _gg,_blk,_m2,_fy,_y0g,_y1g,_ty,_g00,_g01,_g10,_g11
                    _nfix+=nsin
                    del gan,gi,_num,_den,_fx,_x0g,_x1g,_tx
            del ghor,gcru,gcnt,filled,sinh
        # ── (c) RELLENO DE GRIS ───────────────────────────────────────────
        # MEDIDO en la malla (53): el 19.3% de las caras (219.817) quedaban
        # apuntando al gris de relleno. El parche anterior las buscaba con un
        # pre-filtro de 4 puntos por cara y solo encontraba 11.403 (el 5%):
        # ese pre-filtro es el que fallaba. Aqui se elimina: se recorren TODAS
        # las caras y se pinta el color de la malla (que viene de las fotos)
        # en cada texel que siga gris. La prueba es POR TEXEL, no por cara, asi
        # que no puede saltarse ninguno ni pisar nada de lo ya horneado.
        #   - toda cara recibe un punto en su texel central (cubre las caras
        #     mas chicas que un texel: el 45% de las grises)
        #   - ademas se rellena su huella completa (cubre las medianas)
        if VCOL is not None:
            try:
                mf=_np.flatnonzero(FM==ti)
                NCH=150000
                for _c0 in range(0,len(mf),NCH):
                    idc=mf[_c0:_c0+NCH]
                    uvf=Tn[FT[idc]]
                    pu=(uvf[:,:,0]*(W2-1)).astype(_np.float32)
                    pv=(((1.0-uvf[:,:,1]) if flip==0 else uvf[:,:,1])*(H2-1)).astype(_np.float32)
                    colf=((VCOL[F[idc,0]]+VCOL[F[idc,1]]+VCOL[F[idc,2]])/3.0)
                    # (1) PUNTO en el texel central de cada cara
                    cxd=_np.clip(pu.mean(1),0,W2-1).astype(_np.int64)
                    cyd=_np.clip(pv.mean(1),0,H2-1).astype(_np.int64)
                    R=max(0,BAKE_VDOT)
                    _hit=_np.zeros(len(idc),bool)
                    for _oy in range(-R,R+1):
                        for _ox in range(-R,R+1):
                            yy=_np.clip(cyd+_oy,0,H2-1); xx=_np.clip(cxd+_ox,0,W2-1)
                            _cur=base[yy,xx].astype(_np.int16)
                            gz=(_np.abs(_cur-128).max(1)<=6)
                            if gz.any():
                                base[yy[gz],xx[gz]]=_np.clip(colf[gz],0,255).astype(_np.uint8)
                                _nvpx+=int(gz.sum()); _hit|=gz
                            del _cur,gz,yy,xx
                    # (2) HUELLA completa de la cara (por grupos de tamano)
                    x0=_np.clip(_np.floor(pu.min(1)),0,W2-1).astype(_np.int64)
                    x1=_np.clip(_np.ceil (pu.max(1)),0,W2-1).astype(_np.int64)
                    y0=_np.clip(_np.floor(pv.min(1)),0,H2-1).astype(_np.int64)
                    y1=_np.clip(_np.ceil (pv.max(1)),0,H2-1).astype(_np.int64)
                    lado=_np.maximum(x1-x0,y1-y0)+1
                    exp=_np.zeros(len(idc),_np.int32); _l=lado.copy()
                    while (_l>1).any():
                        exp+=(_l>1); _l=(_l+1)//2
                    exp=_np.minimum(exp,6)          # tope 64 px de lado
                    for e in range(int(exp.max())+1 if len(exp) else 0):
                        sel=_np.flatnonzero(exp==e)
                        if len(sel)==0: continue
                        K=1<<e; dxy=_np.arange(K,dtype=_np.int64)
                        paso=max(1,3_000_000//(K*K))
                        for f0 in range(0,len(sel),paso):
                            ii=sel[f0:f0+paso]; n=len(ii)
                            X=x0[ii][:,None,None]+dxy[None,None,:]
                            Y=y0[ii][:,None,None]+dxy[None,:,None]
                            ok=(X<=x1[ii][:,None,None])&(Y<=y1[ii][:,None,None])
                            ok=_np.broadcast_to(ok,(n,K,K)).copy()
                            Xf=_np.broadcast_to(X,(n,K,K)).astype(_np.float32)
                            Yf=_np.broadcast_to(Y,(n,K,K)).astype(_np.float32)
                            ax=pu[ii,0][:,None,None]; ay=pv[ii,0][:,None,None]
                            bx=pu[ii,1][:,None,None]; by=pv[ii,1][:,None,None]
                            cx2=pu[ii,2][:,None,None]; cy2=pv[ii,2][:,None,None]
                            den=(by-cy2)*(ax-cx2)+(cx2-bx)*(ay-cy2)
                            den=_np.where(_np.abs(den)<1e-9,1e-9,den)
                            l0=((by-cy2)*(Xf-cx2)+(cx2-bx)*(Yf-cy2))/den
                            l1=((cy2-ay)*(Xf-cx2)+(ax-cx2)*(Yf-cy2))/den
                            ok&=(l0>=-0.02)&(l1>=-0.02)&((1.0-l0-l1)>=-0.02)
                            del Xf,Yf,den,l0,l1
                            if ok.any():
                                ixp=_np.broadcast_to(X,(n,K,K))[ok]
                                iyp=_np.broadcast_to(Y,(n,K,K))[ok]
                                fid=_np.broadcast_to(_np.arange(n)[:,None,None],(n,K,K))[ok]
                                _cur=base[iyp,ixp].astype(_np.int16)
                                gz=(_np.abs(_cur-128).max(1)<=6)
                                if gz.any():
                                    base[iyp[gz],ixp[gz]]=_np.clip(colf[ii][fid[gz]],0,255).astype(_np.uint8)
                                    _nvpx+=int(gz.sum())
                                    _hit[ii[fid[gz]]]=True
                                del ixp,iyp,fid,_cur,gz
                            del X,Y,ok
                    _nvac+=int(_hit.sum())
                    del uvf,pu,pv,colf,cxd,cyd,x0,x1,y0,y1,lado,exp,_hit
                _gc.collect()
            except Exception as _ve:
                log("BAKE: relleno de gris fallo en atlas %d (%s)" % (ti+1,_ve))
        if tf.lower().endswith((".jpg",".jpeg")):
            Image.fromarray(base).save(tf,quality=BAKE_JQ,subsampling=2)
        else:
            Image.fromarray(base).save(tf)
        del base; _gc.collect()
        log("BAKE: atlas %d/%d escrito (%dx%d) | RAM %.1f GB"
            % (ti+1,len(texfiles),W2,H2,_rss()))
    if _nfix:
        log("BAKE: %.1fM texeles de relleno recibieron correccion de tono" % (_nfix/1e6))
    if _nvac:
        log("BAKE: %d caras apuntaban al gris de relleno -> pintadas con el color de la "
            "malla (%.1fM texeles). Esto es lo que se veia como parches y huecos en paredes y piso."
            % (_nvac,_nvpx/1e6))
    # ══ NIVELADO DE TONO DE PAREDES ═════════════════════════════════════════
    # MEDIDO en la malla (58): la variacion de brillo de las paredes NO es
    # excesiva en total (es menor que la de Polycam), pero esta CONCENTRADA en
    # la escala grande: banda de 1-3 m = 22.3 niveles de gris, mientras 5-30 cm
    # y 30 cm-1 m valen ~14. Un gradiente suave de metro y medio es lo que el
    # ojo lee como "esta pared no esta de un solo tono"; el grano fino lo lee
    # como textura. Asi que se quita SOLO la banda grande.
    # Probado sobre la (58) antes de tocar nada: banda 1-3 m 22.3 -> 7.6 (-66%),
    # 30cm-1m 14.8 -> 12.3, 5-30cm 13.4 -> 13.5 (igual), grano fino 9.2 -> 10.1.
    # COMO: por cada plano grande (pared/piso/techo) se arma una rejilla de 2 cm
    # EN COORDENADAS DEL PLANO (no del atlas: el atlas parte cada pared en miles
    # de parches y un filtro ahi mezclaria zonas que no se tocan), se saca su
    # gradiente de baja frecuencia y se corrige de forma MULTIPLICATIVA hacia el
    # tono medio de esa pared. La ganancia es por CARA porque varia a escala de
    # metros: no hace falta por texel.
    # COSTO ACEPTADO: los gradientes de luz REALES (ventana, lampara) tambien se
    # aplanan. Es el precio de "un solo tono"; se baja con OMVS_TONO_FZA<1.
    if BAKE_TONO and len(F) > 5000:
        try:
            _tt0=time.time()
            from scipy import ndimage as _ndi
            # COLOR POR CARA: se lee el atlas ya horneado y se muestrea el texel
            # del centro de cada cara. Con una muestra por cara sobra, porque lo
            # que se busca es el gradiente a escala de METROS, no el detalle.
            FCOL=_np.zeros((len(F),3),_np.float32)
            _uvc=(Tn[FT[:,0]]+Tn[FT[:,1]]+Tn[FT[:,2]])/3.0
            for _ti,_tf in enumerate(texfiles):
                _mt=_np.flatnonzero(FM==_ti)
                if not len(_mt): continue
                _im=Image.open(_tf).convert("RGB")
                _A=_np.asarray(_im,dtype=_np.uint8); del _im
                _Ha,_Wa=_A.shape[:2]
                _px=_np.clip((_uvc[_mt,0]*(_Wa-1)).astype(_np.int64),0,_Wa-1)
                _vv=(1.0-_uvc[_mt,1]) if flip==0 else _uvc[_mt,1]
                _py=_np.clip((_vv*(_Ha-1)).astype(_np.int64),0,_Ha-1)
                FCOL[_mt]=_A[_py,_px].astype(_np.float32)
                del _A; _gc.collect()
            # las caras que quedaron en el gris de relleno no aportan tono
            _gris=_np.abs(FCOL-128.0).max(1)<=6
            FCOL[_gris]=_np.nan
            _o3=None
            try:
                import open3d as _o3
            except Exception:
                _o3=None
            _cen=(_Vkeep[F[:,0]]+_Vkeep[F[:,1]]+_Vkeep[F[:,2]])/3.0
            _e0=_Vkeep[F[:,1]]-_Vkeep[F[:,0]]; _e1v=_Vkeep[F[:,2]]-_Vkeep[F[:,0]]
            _fn=_np.cross(_e0,_e1v)
            _ln=_np.linalg.norm(_fn,axis=1); _ok=_ln>1e-12
            _fn[_ok]/=_ln[_ok][:,None]
            _CG=_Vkeep.mean(0)
            # --- planos: PRIMERO los que ya encontro el script de malla ---
            # Esos ya pasaron los tres candados (tamano >=1.2 m, periferia por
            # percentil, normales alineadas). Recalcularlos aqui era duplicar el
            # trabajo con peor informacion: en el job e13efea4 la busqueda de aqui
            # devolvio CERO planos y el nivelado no corrio, mientras el script de
            # malla habia encontrado 6 sin problema.
            _planos=[]
            try:
                _pf=os.environ.get("PLANOS_NPY","/workspace/job/planos.npy")
                if os.path.exists(_pf):
                    _pa=_np.load(_pf)
                    for _row in _pa:
                        _nn=_np.asarray(_row[:3],_np.float64)
                        _L=_np.linalg.norm(_nn)
                        if _L>1e-9: _planos.append((_nn/_L,float(_row[3])/_L))
                    log("BAKE TONO: %d planos leidos del paso de malla (ya filtrados)"
                        % len(_planos))
                else:
                    log("BAKE TONO: no hay planos guardados; los busco yo (respaldo)")
            except Exception as _pe:
                log("BAKE TONO: no pude leer los planos guardados (%s); los busco yo" % _pe)
            if not _planos and _o3 is not None:
                _Vs=_Vkeep[::4].astype(_np.float64)
                _rest=_np.arange(len(_Vs))
                for _r in range(8):
                    if len(_rest)<4000: break
                    _pc=_o3.geometry.PointCloud()
                    _pc.points=_o3.utility.Vector3dVector(_Vs[_rest])
                    try:
                        _mod,_inl=_pc.segment_plane(distance_threshold=0.02,ransac_n=3,num_iterations=800)
                    except Exception:
                        break
                    if len(_inl)<4000: break
                    _a,_b,_c,_d=_mod; _nn=_np.array([_a,_b,_c],dtype=_np.float64)
                    _L=_np.linalg.norm(_nn)
                    if _L<1e-9: break
                    _planos.append((_nn/_L,_d/_L))
                    _rest=_np.setdiff1d(_rest,_rest[_np.asarray(_inl)])
            if not _planos:
                log("BAKE TONO: no encontre planos grandes; lo salto")
            else:
                _gan=_np.ones(len(F),_np.float32)
                _npar=0; _desc=[]
                for _nrm,_dd in _planos:
                    _lado=_np.sign(float(_np.dot(_CG,_nrm))+_dd) or 1.0
                    _dist=(_cen@_nrm+_dd)*_lado
                    _sel=(_np.abs(_fn@_nrm)>0.90)&(_np.abs(_dist)<0.06)
                    if int(_sel.sum())<8000:
                        _desc.append("plano con solo %d caras (<8000)" % int(_sel.sum()))
                        continue
                    _ix=_np.flatnonzero(_sel)
                    _u=_np.array([1.0,0,0]) if abs(_nrm[0])<0.9 else _np.array([0,1.0,0])
                    _a1=_np.cross(_nrm,_u); _a1/=_np.linalg.norm(_a1)
                    _a2=_np.cross(_nrm,_a1)
                    _x=_cen[_ix]@_a1; _y=_cen[_ix]@_a2
                    _P=0.02
                    _gx=((_x-_x.min())/_P).astype(_np.int64)
                    _gy=((_y-_y.min())/_P).astype(_np.int64)
                    _W=int(_gx.max())+1; _H=int(_gy.max())+1
                    if _W*_H>4_000_000 or _W<60 or _H<60:
                        _desc.append("rejilla %dx%d fuera de rango" % (_W,_H)); continue
                    # color medio por cara (basta: se busca la escala de METROS)
                    _bue=~_np.isnan(FCOL[_ix]).any(1)
                    if _bue.sum()<4000:
                        _desc.append("solo %d caras con color leible (<4000)" % int(_bue.sum()))
                        continue
                    _ix=_ix[_bue]
                    _gx=_gx[_bue]; _gy=_gy[_bue]
                    _cf=FCOL[_ix]
                    _S=_np.zeros((_H,_W,3)); _Cn=_np.zeros((_H,_W))
                    for _k in range(3): _np.add.at(_S[:,:,_k],(_gy,_gx),_cf[:,_k])
                    _np.add.at(_Cn,(_gy,_gx),1.0)
                    _mk=_Cn>0
                    if _mk.mean()<0.05:
                        _desc.append("rejilla casi vacia (%.1f%% llena)" % (100*_mk.mean()))
                        continue
                    _G=_np.where(_mk[:,:,None],_S/_np.maximum(_Cn,1.0)[:,:,None],0.0)
                    _ii=_ndi.distance_transform_edt(~_mk,return_distances=False,return_indices=True)
                    _G=_G[_ii[0],_ii[1]]
                    _kk=max(3,int(BAKE_TONO_ESC/_P))
                    _BJ=_np.stack([_ndi.uniform_filter(_G[:,:,_k],size=_kk) for _k in range(3)],-1)
                    _obj=_G[_mk].mean(0)
                    _g3=_np.clip(_obj[None,None,:]/_np.maximum(_BJ,1.0),
                                 1.0/BAKE_TONO_TOPE,BAKE_TONO_TOPE)
                    _gl=_g3.mean(2)                       # una ganancia por celda
                    _gf=_gl[_np.clip(_gy,0,_H-1),_np.clip(_gx,0,_W-1)]
                    _gf=1.0+BAKE_TONO_FZA*(_gf-1.0)
                    _gan[_ix]=_gf.astype(_np.float32)
                    _npar+=1
                if _npar==0:
                    log("BAKE TONO: ningun plano cumplio el minimo de %d. Motivos: %s"
                        % (len(_planos), " | ".join(_desc[:6]) if _desc else "sin detalle"))
                else:
                    _apl=_np.flatnonzero(_np.abs(_gan-1.0)>0.01)
                    log("BAKE TONO: %d superficies, %d caras a corregir (escala %.1f m, fuerza %.2f)"
                        % (_npar,len(_apl),BAKE_TONO_ESC,BAKE_TONO_FZA))
                    _ntx=0
                    for _ti,_tf in enumerate(texfiles):
                        _sf=_apl[FM[_apl]==_ti]
                        if not len(_sf): continue
                        _im=Image.open(_tf).convert("RGB")
                        _bs=_np.asarray(_im,dtype=_np.uint8).copy(); del _im
                        _H2,_W2=_bs.shape[:2]
                        _uv=Tn[FT[_sf]]
                        _pu=(_uv[:,:,0]*(_W2-1)).astype(_np.float32)
                        _pv=(((1.0-_uv[:,:,1]) if flip==0 else _uv[:,:,1])*(_H2-1)).astype(_np.float32)
                        _x0=_np.clip(_np.floor(_pu.min(1)),0,_W2-1).astype(_np.int64)
                        _x1=_np.clip(_np.ceil (_pu.max(1)),0,_W2-1).astype(_np.int64)
                        _y0=_np.clip(_np.floor(_pv.min(1)),0,_H2-1).astype(_np.int64)
                        _y1=_np.clip(_np.ceil (_pv.max(1)),0,_H2-1).astype(_np.int64)
                        _ld=_np.maximum(_x1-_x0,_y1-_y0)+1
                        _ex=_np.zeros(len(_sf),_np.int32); _l=_ld.copy()
                        while (_l>1).any(): _ex+=(_l>1); _l=(_l+1)//2
                        _ex=_np.minimum(_ex,6)
                        _gs=_gan[_sf]
                        for _e in range(int(_ex.max())+1 if len(_ex) else 0):
                            _sg=_np.flatnonzero(_ex==_e)
                            if not len(_sg): continue
                            _K=1<<_e; _dxy=_np.arange(_K,dtype=_np.int64)
                            _paso=max(1,3_000_000//(_K*_K))
                            for _c0 in range(0,len(_sg),_paso):
                                _s=_sg[_c0:_c0+_paso]; _n=len(_s)
                                _X=_x0[_s][:,None,None]+_dxy[None,None,:]
                                _Y=_y0[_s][:,None,None]+_dxy[None,:,None]
                                _okb=(_X<=_x1[_s][:,None,None])&(_Y<=_y1[_s][:,None,None])
                                _okb=_np.broadcast_to(_okb,(_n,_K,_K)).copy()
                                _Xf=_np.broadcast_to(_X,(_n,_K,_K)).astype(_np.float32)
                                _Yf=_np.broadcast_to(_Y,(_n,_K,_K)).astype(_np.float32)
                                _ax=_pu[_s,0][:,None,None]; _ay=_pv[_s,0][:,None,None]
                                _bx=_pu[_s,1][:,None,None]; _by=_pv[_s,1][:,None,None]
                                _cx=_pu[_s,2][:,None,None]; _cy=_pv[_s,2][:,None,None]
                                _den=(_by-_cy)*(_ax-_cx)+(_cx-_bx)*(_ay-_cy)
                                _den=_np.where(_np.abs(_den)<1e-9,1e-9,_den)
                                _l0=((_by-_cy)*(_Xf-_cx)+(_cx-_bx)*(_Yf-_cy))/_den
                                _l1=((_cy-_ay)*(_Xf-_cx)+(_ax-_cx)*(_Yf-_cy))/_den
                                _okb&=(_l0>=-0.02)&(_l1>=-0.02)&((1.0-_l0-_l1)>=-0.02)
                                del _Xf,_Yf,_den,_l0,_l1
                                if _okb.any():
                                    _ixp=_np.broadcast_to(_X,(_n,_K,_K))[_okb]
                                    _iyp=_np.broadcast_to(_Y,(_n,_K,_K))[_okb]
                                    _fid=_np.broadcast_to(_np.arange(_n)[:,None,None],(_n,_K,_K))[_okb]
                                    _cur=_bs[_iyp,_ixp].astype(_np.float32)
                                    _bs[_iyp,_ixp]=_np.clip(_cur*_gs[_s][_fid][:,None],0,255).astype(_np.uint8)
                                    _ntx+=int(_okb.sum())
                                    del _ixp,_iyp,_fid,_cur
                                del _X,_Y,_okb
                        if _tf.lower().endswith((".jpg",".jpeg")):
                            Image.fromarray(_bs).save(_tf,quality=max(BAKE_JQ,90),subsampling=2)
                        else:
                            Image.fromarray(_bs).save(_tf)
                        del _bs; _gc.collect()
                    log("BAKE TONO: %.1fM texeles nivelados en %.1f min "
                        "(esto es lo que se veia como paredes con tonos distintos)"
                        % (_ntx/1e6,(time.time()-_tt0)/60.0))
        except Exception as _te:
            log("BAKE TONO: fallo, sigo sin el (%s)" % _te)
    # ══ NIVELADO DE COSTURAS ENTRE PARCHES ══════════════════════════════════
    # MEDIDO en la malla (58): OpenMVS parte la malla en parches y a cada uno le
    # asigna UNA foto. Al cruzar de parche cambia la foto y salta el tono. En las
    # paredes el salto medio entre caras vecinas es 4.44 niveles DENTRO del mismo
    # parche y 9.15 CRUZANDO (2.1x), y el 13.7% de los bordes de parche salta mas
    # de 15 niveles. Como el borde sigue aristas de triangulos, se ve como
    # RECTANGULOS Y PARALELOGRAMOS de bordes rectos y diagonales con distinto
    # gris: es justo lo que se reportaba como "bloques" y "cicatrices".
    # OpenMVS trae dos correcciones para esto (--global/--local-seam-leveling)
    # pero estan en 0, y aunque se encendieran NO SERVIRIA: el horneador escribe
    # 48M texeles ENCIMA, borrandolas. Por eso el arreglo va aqui.
    # COMO: se resuelve un campo de compensacion (un offset por cara) que anula
    # el salto en los bordes de parche y se mantiene liso dentro de cada parche.
    # Es un sistema disperso (Laplaciano) resuelto por gradiente conjugado.
    # PROBADO sobre la (58) antes de escribir esto:
    #   cruzando parche  9.70 -> 1.34 niveles (-86%), p95 38.3 -> 3.4
    #   dentro del parche 4.35 -> 4.39 (+1%)  <- el detalle NO se toca
    #   bordes con salto >15 niveles: 14.9% -> 2.2%
    if BAKE_SEAM and len(F) > 5000:
        try:
            _st0=time.time()
            import scipy.sparse as _sps
            from scipy.sparse.linalg import cg as _cg
            # --- color por cara, leido de los atlas ya horneados ---
            _uvc=(Tn[FT[:,0]]+Tn[FT[:,1]]+Tn[FT[:,2]])/3.0
            _FC=_np.zeros((len(F),3),_np.float32)
            for _ti,_tf in enumerate(texfiles):
                _mt=_np.flatnonzero(FM==_ti)
                if not len(_mt): continue
                _im=Image.open(_tf).convert("RGB")
                _A=_np.asarray(_im,dtype=_np.uint8); del _im
                _Ha,_Wa=_A.shape[:2]
                _px=_np.clip((_uvc[_mt,0]*(_Wa-1)).astype(_np.int64),0,_Wa-1)
                _vv=(1.0-_uvc[_mt,1]) if flip==0 else _uvc[_mt,1]
                _py=_np.clip((_vv*(_Ha-1)).astype(_np.int64),0,_Ha-1)
                _FC[_mt]=_A[_py,_px].astype(_np.float32)
                del _A; _gc.collect()
            # --- caras vecinas (comparten arista) ---
            _E=_np.sort(_np.stack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]],1).reshape(-1,2),axis=1)
            _fi=_np.repeat(_np.arange(len(F)),3)
            _o=_np.lexsort((_E[:,1],_E[:,0])); _Es=_E[_o]; _fs=_fi[_o]
            _ig=_np.flatnonzero((_Es[:-1]==_Es[1:]).all(1))
            _f1=_fs[_ig]; _f2=_fs[_ig+1]
            _Ea=_Es[_ig,0].copy(); _Eb=_Es[_ig,1].copy()   # vertices de la arista
            del _E,_fi,_o,_Es,_fs,_ig
            if len(_f1)<1000:
                log("BAKE COSTURA: la malla no comparte aristas; lo salto")
            else:
                # DETECCION EXACTA DE COSTURA (v2)
                # FALLO del intento anterior (job d87eae06): marcaba costura con
                # un umbral RELATIVO (distancia UV entre centroides > 5x la
                # mediana). En mi banco daba 1.7% de las aristas; EN PRODUCCION
                # dio 25.2% (407.966 de 1.621.751) y compensacion media de 7.26
                # niveles en vez de 0.21. Marco como costura una de cada cuatro
                # aristas SANAS y pinto la pared de retazos: eso fabrico los
                # cuadraditos que aparecieron en ese render.
                # AHORA la prueba es exacta y no depende de ninguna estadistica:
                # dos caras vecinas comparten una arista de GEOMETRIA (2 vertices).
                # Si sus UV EN ESOS DOS VERTICES coinciden, la textura es continua
                # y NO hay costura; si diferen, es costura de verdad.
                _p1=_np.zeros((len(_f1),2),_np.int64)
                _p2=_np.zeros((len(_f1),2),_np.int64)
                for _k in range(3):
                    _m=F[_f1,_k]==_Ea; _p1[_m,0]=_k
                    _m=F[_f1,_k]==_Eb; _p1[_m,1]=_k
                    _m=F[_f2,_k]==_Ea; _p2[_m,0]=_k
                    _m=F[_f2,_k]==_Eb; _p2[_m,1]=_k
                _uA1=Tn[FT[_f1,_p1[:,0]]]; _uA2=Tn[FT[_f1,_p1[:,1]]]
                _uB1=Tn[FT[_f2,_p2[:,0]]]; _uB2=Tn[FT[_f2,_p2[:,1]]]
                _cont=((_np.abs(_uA1-_uB1).max(1)<1e-5)&(_np.abs(_uA2-_uB2).max(1)<1e-5))
                _sm=(FM[_f1]!=FM[_f2])|(~_cont)
                del _p1,_p2,_uA1,_uA2,_uB1,_uB2,_cont
                _frac=float(_sm.mean())
                log("BAKE COSTURA: %.2f%% de las aristas son costura real" % (100*_frac))
                # CANDADO: en una malla sana son ~2%. Si sale mucho mas, la
                # deteccion esta rota y se aborta ANTES de tocar un solo texel,
                # en vez de repetir el desastre de los cuadraditos.
                if _frac > float(os.environ.get("OMVS_SEAM_MAXFRAC","0.12")):
                    raise RuntimeError("deteccion sospechosa (%.1f%% de aristas); no toco nada" % (100*_frac))
                _N=len(F)
                def _lap(_a,_b,_w):
                    _n=len(_a)
                    _r=_np.concatenate([_a,_b,_a,_b]); _c=_np.concatenate([_a,_b,_b,_a])
                    _v=_np.concatenate([_np.full(_n,_w),_np.full(_n,_w),
                                        _np.full(_n,-_w),_np.full(_n,-_w)])
                    return _sps.csr_matrix((_v,(_r,_c)),shape=(_N,_N))
                _Aop=(_lap(_f1[_sm],_f2[_sm],1.0)
                      +_lap(_f1[~_sm],_f2[~_sm],BAKE_SEAM_MU)
                      +BAKE_SEAM_LAM*_sps.identity(_N,format='csr')).tocsr()
                _OFF=_np.zeros((_N,3),_np.float32)
                for _ch in range(3):
                    _d=(_FC[_f1[_sm],_ch]-_FC[_f2[_sm],_ch]).astype(_np.float64)
                    _b=_np.zeros(_N)
                    _np.add.at(_b,_f1[_sm],-_d); _np.add.at(_b,_f2[_sm],_d)
                    try:
                        _x,_inf=_cg(_Aop,_b,rtol=1e-6,maxiter=300)
                    except TypeError:
                        _x,_inf=_cg(_Aop,_b,tol=1e-6,maxiter=300)
                    _OFF[:,_ch]=_np.clip(_x,-BAKE_SEAM_TOPE,BAKE_SEAM_TOPE)
                if float(_np.abs(_OFF).max())>=BAKE_SEAM_TOPE-0.01:
                    raise RuntimeError("la compensacion se saturo en el tope (%.0f): "
                        "el sistema no convergio, no aplico nada" % BAKE_SEAM_TOPE)
                log("BAKE COSTURA: %d costuras de %d aristas; compensacion "
                    "media %.2f niveles, maxima %.1f"
                    % (int(_sm.sum()),len(_f1),float(_np.abs(_OFF).mean()),
                       float(_np.abs(_OFF).max())))
                # --- aplicar el offset a la huella de cada cara ---
                _apl=_np.flatnonzero(_np.abs(_OFF).max(1)>0.5)
                _ntx=0
                for _ti,_tf in enumerate(texfiles):
                    _sf=_apl[FM[_apl]==_ti]
                    if not len(_sf): continue
                    _im=Image.open(_tf).convert("RGB")
                    _bs=_np.asarray(_im,dtype=_np.uint8).copy(); del _im
                    _H2,_W2=_bs.shape[:2]
                    _uv=Tn[FT[_sf]]
                    _pu=(_uv[:,:,0]*(_W2-1)).astype(_np.float32)
                    _pv=(((1.0-_uv[:,:,1]) if flip==0 else _uv[:,:,1])*(_H2-1)).astype(_np.float32)
                    _x0=_np.clip(_np.floor(_pu.min(1)),0,_W2-1).astype(_np.int64)
                    _x1=_np.clip(_np.ceil (_pu.max(1)),0,_W2-1).astype(_np.int64)
                    _y0=_np.clip(_np.floor(_pv.min(1)),0,_H2-1).astype(_np.int64)
                    _y1=_np.clip(_np.ceil (_pv.max(1)),0,_H2-1).astype(_np.int64)
                    _ld=_np.maximum(_x1-_x0,_y1-_y0)+1
                    _ex=_np.zeros(len(_sf),_np.int32); _l=_ld.copy()
                    while (_l>1).any(): _ex+=(_l>1); _l=(_l+1)//2
                    _ex=_np.minimum(_ex,6)
                    _os=_OFF[_sf]
                    for _e in range(int(_ex.max())+1 if len(_ex) else 0):
                        _sg=_np.flatnonzero(_ex==_e)
                        if not len(_sg): continue
                        _K=1<<_e; _dxy=_np.arange(_K,dtype=_np.int64)
                        _paso=max(1,3_000_000//(_K*_K))
                        for _c0 in range(0,len(_sg),_paso):
                            _s=_sg[_c0:_c0+_paso]; _n=len(_s)
                            _X=_x0[_s][:,None,None]+_dxy[None,None,:]
                            _Y=_y0[_s][:,None,None]+_dxy[None,:,None]
                            _okb=(_X<=_x1[_s][:,None,None])&(_Y<=_y1[_s][:,None,None])
                            _okb=_np.broadcast_to(_okb,(_n,_K,_K)).copy()
                            _Xf=_np.broadcast_to(_X,(_n,_K,_K)).astype(_np.float32)
                            _Yf=_np.broadcast_to(_Y,(_n,_K,_K)).astype(_np.float32)
                            _ax=_pu[_s,0][:,None,None]; _ay=_pv[_s,0][:,None,None]
                            _bx=_pu[_s,1][:,None,None]; _by=_pv[_s,1][:,None,None]
                            _cx=_pu[_s,2][:,None,None]; _cy=_pv[_s,2][:,None,None]
                            _den=(_by-_cy)*(_ax-_cx)+(_cx-_bx)*(_ay-_cy)
                            _den=_np.where(_np.abs(_den)<1e-9,1e-9,_den)
                            _l0=((_by-_cy)*(_Xf-_cx)+(_cx-_bx)*(_Yf-_cy))/_den
                            _l1=((_cy-_ay)*(_Xf-_cx)+(_ax-_cx)*(_Yf-_cy))/_den
                            _okb&=(_l0>=-0.02)&(_l1>=-0.02)&((1.0-_l0-_l1)>=-0.02)
                            del _Xf,_Yf,_den,_l0,_l1
                            if _okb.any():
                                _ixp=_np.broadcast_to(_X,(_n,_K,_K))[_okb]
                                _iyp=_np.broadcast_to(_Y,(_n,_K,_K))[_okb]
                                _fid=_np.broadcast_to(_np.arange(_n)[:,None,None],(_n,_K,_K))[_okb]
                                _cur=_bs[_iyp,_ixp].astype(_np.float32)
                                _bs[_iyp,_ixp]=_np.clip(_cur+_os[_s][_fid],0,255).astype(_np.uint8)
                                _ntx+=int(_okb.sum())
                                del _ixp,_iyp,_fid,_cur
                            del _X,_Y,_okb
                    if _tf.lower().endswith((".jpg",".jpeg")):
                        Image.fromarray(_bs).save(_tf,quality=max(BAKE_JQ,90),subsampling=2)
                    else:
                        Image.fromarray(_bs).save(_tf)
                    del _bs; _gc.collect()
                log("BAKE COSTURA: %.1fM texeles compensados en %.1f min "
                    "(esto es lo que se veia como bloques y cicatrices de distinto tono)"
                    % (_ntx/1e6,(time.time()-_st0)/60.0))
        except Exception as _se:
            log("BAKE COSTURA: fallo, sigo sin ella (%s)" % _se)
    log("BAKE listo: %.1fM texeles | cobertura >=1 foto %.0f%%, >=3 fotos %.0f%% | %s | "
        "atlas x%d en %.1f min"
        % (NT/1e6,cov1,cov3,"GPU" if usa_gpu else "CPU",SC,(time.time()-_t0)/60.0))
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
        log("TONO: superado por el horneador (OMVS_TONE=0)")
    # (a3) HORNEADOR MULTI-VISTA v9.1 — repinta los atlas mezclando las fotos
    if BAKE and texfiles and mtl2tex:
        try:
            if not bake_multiview(objf, texfiles, mtl2tex):
                log("BAKE: no se aplico (los atlas de OpenMVS quedan como estaban)")
        except Exception as _bk:
            log("BAKE: horneador fallo (%s); los atlas de OpenMVS quedan" % _bk)
    elif not BAKE:
        log("BAKE apagado (OMVS_BAKE=0): atlas de OpenMVS tal cual")
    # (b) exportar SOLDANDO vertices: el OBJ repite cada esquina (v/vt) y el
    #     glb salia con el TRIPLE de vertices (3.29M para 1.1M caras; Polycam
    #     usa 0.57 por cara). Aqui cada par unico (vertice, uv) es UN vertice:
    #     ~25 MB menos. Si algo falla, cae al exportador simple de siempre.
    try:
        import numpy as _np
        _Vs=[]; _Ts=[]; _Fs={}; _cur=None
        with open(objf) as _fh:
            for _ln in _fh:
                if _ln.startswith("v "):
                    _p=_ln.split(); _Vs.append((float(_p[1]),float(_p[2]),float(_p[3])))
                elif _ln.startswith("vt "):
                    _p=_ln.split(); _Ts.append((float(_p[1]),float(_p[2])))
                elif _ln.startswith("usemtl"):
                    _cur=_ln.split(None,1)[1].strip(); _Fs.setdefault(_cur,[])
                elif _ln.startswith("f ") and _cur is not None:
                    _p=_ln.split()
                    if len(_p)<4: continue
                    _tri=[]
                    for _c in _p[1:4]:
                        _q=_c.split("/"); _tri.append((int(_q[0])-1, int(_q[1])-1))
                    _Fs[_cur].append(_tri)
        _V=_np.asarray(_Vs,_np.float64); _T=_np.asarray(_Ts,_np.float64)
        # soldar por (vertice, VALOR de UV redondeado) — NO por indice de vt:
        # OpenMVS le da un indice de vt DISTINTO a cada esquina aunque el uv
        # sea identico, asi que agrupar por indice no soldaba nada (leccion
        # v8.9). Redondeamos el uv a ~1/8000 (un texel de 8192) y agrupamos
        # por (indice de vertice, uv_x_red, uv_y_red): dos esquinas que
        # comparten posicion Y uv se funden en un vertice.
        _uvq=_np.round(_T*8000).astype(_np.int64)          # cuantizar uv
        _uvmap={}                                          # uv_red -> id compacto
        _esc=trimesh.Scene(); _tot_v=0; _tot_c=0
        for _mtl,_caras in _Fs.items():
            if not _caras or _mtl not in mtl2tex: continue
            _fa=_np.asarray(_caras,_np.int64)              # (n,3,2) = (vi,ti)
            _vig=_fa[:,:,0].reshape(-1)                    # indice de vertice
            _tig=_fa[:,:,1].reshape(-1)                    # indice de vt
            _uvk=_uvq[_tig]                                # (M,2) uv cuantizado
            # clave = vertice * BIG + hash(uv cuantizado)
            _BIGU=16000                                    # rango de uv cuantizado (0..8000)
            _clave=_vig.astype(_np.int64)*(_BIGU*_BIGU) + _uvk[:,0]*_BIGU + _uvk[:,1]
            _uq,_first,_inv=_np.unique(_clave,return_index=True,return_inverse=True)
            _vsel=_vig[_first]; _tsel=_tig[_first]   # vertice y vt representativo por clave
            _mesh=trimesh.Trimesh(vertices=_V[_vsel], faces=_inv.reshape(-1,3),
                                  process=False)
            _mesh.visual=trimesh.visual.TextureVisuals(
                uv=_T[_tsel], image=Image.open(texfiles[mtl2tex[_mtl]]))
            _esc.add_geometry(_mesh, geom_name=_mtl)
            _tot_v+=len(_vsel); _tot_c+=len(_fa)
        _esc.export(outglb)
        log("export SOLDADO: %d vertices para %d caras (antes: %d)"
            % (_tot_v,_tot_c,_tot_c*3))
    except Exception as _we:
        log("export soldado fallo (%s) -> exportador simple" % _we)
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
# NOTA CLAVE v9.5: el horneador repinta CADA texel mezclando todas las fotos,
# asi que la "mejor vista por cara" que elige OpenMVS ya no afecta el color
# final: lo unico que importa es el MAPA UV. Por eso subimos la compactacion
# de parches (cost-smoothness-ratio) y usamos el empaque de mejor ajuste:
# menos parches y mas grandes -> menos borde desperdiciado entre ellos.
# MEDIDO en el (45): nuestro atlas aprovecha 26% del area; Polycam 61%.
CFG1 = ["--virtual-face-images", str(VFACES),
        "--patch-packing-heuristic", PACKH,
        "--local-seam-leveling", str(LOCAL_SEAM),
        "--sharpness-weight", str(SHARP),
        "--global-seam-leveling", str(GLOBAL_SEAM)]
CFG2 = ["--virtual-face-images", str(VFACES),
        "--patch-packing-heuristic", PACKH,
        "--local-seam-leveling", "0",
        "--sharpness-weight", "0",
        "--global-seam-leveling", "0"]
CONFIGS = [(MAX_TEX, CFG1, OMP_HI), (4096, CFG2, OMP_HI)]   # cfg2 = respaldo con el 4096 probado en produccion
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
# FASE 0 (investigacion): el prior de PROFUNDIDAD queda en 0 a proposito.
# Fue el que causo las LAMINAS: la profundidad monocular se alinea con
# escala+desplazamiento FOTO POR FOTO, cada foto pide la pared a otra
# distancia y el entrenamiento apila una capa por version. Evidencia dura
# en los logs de este proyecto: 17.949 -> 116.896 pedazos sueltos, error de
# orientacion de surfels 12 -> 45 grados.
# El prior de NORMALES no tiene ese problema: una normal es una DIRECCION,
# no depende de escala ni de desplazamiento, asi que es consistente entre
# fotos. Es justo lo que aplana paredes lisas sin textura (DN-Splatter,
# GaussianRoom). Subir esto de 0 solo si se quiere reproducir el fallo.
_L_DEPTH = float(os.environ.get("MONO_LAMBDA_DEPTH", "0.0"))
_L_NORM  = float(os.environ.get("MONO_LAMBDA_NORMAL", "0.1"))
_ABS_COS = os.environ.get("MONO_ABS_COS", "1") == "1"   # |cos|: inmune al signo
_COS_DIAG = False
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
    # DIAGNOSTICO (una sola vez): si el coseno medio sale NEGATIVO, la normal
    # del prior viene invertida respecto a la de 2DGS y la perdida estaba
    # empujando los surfels justo al reves. Eso explicaria el desastre medido.
    global _COS_DIAG
    if not _COS_DIAG:
        try:
            _cm = float((cosine * m.float()).sum() / (m.float().sum() + 1e-6))
            print("[priors] DIAGNOSTICO coseno medio prior-vs-render = %+.3f  (%s)"
                  % (_cm, "SIGNO INVERTIDO" if _cm < -0.2 else
                          ("alineado" if _cm > 0.2 else "SIN CORRELACION")), flush=True)
            _COS_DIAG = True
        except Exception:
            _COS_DIAG = True
    # |coseno|: penaliza que el EJE no coincida, sin depender del signo.
    if _ABS_COS:
        L_n = ((1.0 - cosine.abs()) * m.float()).sum() / (m.float().sum() + 1e-6)
    else:
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
        _bn_pr = ("priorsOFF" if os.environ.get("MONO_PRIORS", "0") != "1"
                  else ("priorsNORM" if float(os.environ.get("MONO_LAMBDA_DEPTH","0.0"))==0.0
                        else "priorsON"))
        _bn_sm = os.environ.get("SMOOTH_MODE", "twostep")
        _bn_sn = "snapON" if os.environ.get("PLANE_SNAP", "1") == "1" else "snapOFF"
        _bn_tr = int(os.environ.get("MESH_TRIS", "2500000")) // 1000
        _bn_st = os.environ.get("PAINT_STORE", "linear")
        _bn_au = "audit" if os.environ.get("AUDIT","1")=="1" else "noaudit"
        _bn_uv = "uv" if os.environ.get("UV_TEXTURE","1")=="1" else "noUV"
        log(f"═══ render-gs-worker 2DGS · v9-{_bn_pr}-{_bn_sm}-{_bn_sn}-{_bn_tr}k-{_bn_st}-"
            f"{os.environ.get('MESH_ENGINE','2dgs').lower() + '-bake99-snap2b-bilin' if os.environ.get('UV_TEXTURE','1')=='1' else 'vertexB'}"
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
        # ── PASO 1b: PREFILTRO con las poses que manda el celular (Felipe) ──
        # La app escribe poses.json/.jsonl dentro del ZIP y hasta ahora se
        # IGNORABA. Usarlo aqui sale gratis: pasa ANTES de cargar el modelo, o
        # sea cero GPU, y cada foto que cae son ~25 pares de MASt3R menos.
        # SEGURIDAD: sin ARCore el celular solo sabe hacia DONDE apunta, no
        # donde esta. Por eso solo se comparan fotos CERCANAS EN EL TIEMPO: eso
        # ataca el caso real (quedarse quieto) sin borrar la cocina por
        # parecerse al baño.
        imgs = sorted(imgs)
        poses_app = {}
        if DEDUP and PREFILTER:
            try:
                for pf in list(raw.rglob("poses.json")) + list(raw.rglob("poses.jsonl")):
                    txt = pf.read_text(encoding="utf-8", errors="ignore")
                    registros = []
                    if pf.suffix == ".jsonl":
                        for ln in txt.splitlines():
                            ln = ln.strip()
                            if ln:
                                try: registros.append(json.loads(ln))
                                except Exception: pass
                    else:
                        try: registros = json.loads(txt).get("poses", [])
                        except Exception: registros = []
                    for r in registros:
                        nom = r.get("image", "")
                        if nom:
                            poses_app[nom] = r
                # VALIDACION DE TIMESTAMPS: toda la seguridad descansa en la
                # ventana de tiempo. Si vinieran vacios o iguales, la ventana no
                # expiraria nunca y compararíamos todas contra todas.
                _ts = [float(r.get("timestamp", 0) or 0) for r in poses_app.values()]
                _ts_ok = (len(set(_ts)) > max(2, len(_ts) // 10)
                          and (max(_ts) - min(_ts)) > 1000.0) if _ts else False
                if len(poses_app) < 2:
                    log("   PREFILTRO: el ZIP no trae poses.json usable, se salta")
                elif not _ts_ok:
                    log("   PREFILTRO: los timestamps del poses.json no son fiables "
                        "(vacios o todos iguales). NO filtro aqui: sin reloj no puedo "
                        "distinguir 'quieto en el baño' de 'otra habitacion'. "
                        "El filtro fino del PASO 2 sigue activo.")
                else:
                    def _quat_de(r):
                        """Cuaternion del registro. COMPATIBLE con la app vieja:
                        antes solo escribia 'rotation' = [azimut,pitch,roll] en
                        radianes; aqui se convierte. No es capricho: arregla el
                        gimbal lock (apuntando al piso el azimut se dispara sin
                        haber girado, y restar angulos de Euler daria diferencias
                        enormes falsas)."""
                        q = r.get("quaternion")
                        if q and len(q) >= 4:
                            return list(q[:4])                    # app nueva: [x,y,z,w]
                        e = r.get("rotation")
                        if not e or len(e) < 3:
                            return None
                        import math as _m
                        az, pi_, ro = float(e[0]), float(e[1]), float(e[2])
                        def _mul(a, b):                           # [x,y,z,w]
                            x1,y1,z1,w1 = a; x2,y2,z2,w2 = b
                            return [w1*x2 + x1*w2 + y1*z2 - z1*y2,
                                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                                    w1*w2 - x1*x2 - y1*y2 - z1*z2]
                        qz = [0.0, 0.0, _m.sin(az/2), _m.cos(az/2)]
                        qx = [_m.sin(pi_/2), 0.0, 0.0, _m.cos(pi_/2)]
                        qy = [0.0, _m.sin(ro/2), 0.0, _m.cos(ro/2)]
                        return _mul(_mul(qz, qx), qy)
                    def _ang_deg(q1, q2):
                        # angulo geodesico: 2*acos(|q1 . q2|). El valor absoluto
                        # hace que q y -q (misma rotacion) den lo mismo.
                        import math
                        d = abs(sum(a*b for a, b in zip(q1, q2)))
                        d = min(1.0, max(0.0, d))
                        return math.degrees(2.0 * math.acos(d))
                    conservadas, recientes = [], []
                    sin_datos = 0; _via_euler = 0
                    for img in imgs:
                        r = poses_app.get(img.name)
                        q = _quat_de(r) if r else None
                        if not r or not q:
                            conservadas.append(img); sin_datos += 1
                            continue
                        if not r.get("quaternion"):
                            _via_euler += 1
                        ts = float(r.get("timestamp", 0)) / 1000.0
                        pos = r.get("position") or [0.0, 0.0, 0.0]
                        hay_pos = bool(r.get("has_position", False))
                        recientes = [x for x in recientes if ts - x[0] <= PREFILTER_TIME_S]
                        repetida = False
                        for (ts2, q2, pos2, hay2) in recientes:
                            if _ang_deg(q, q2) >= DEDUP_ROT_DEG:
                                continue                     # otro angulo -> sirve
                            if hay_pos and hay2:
                                d = sum((a-b)**2 for a, b in zip(pos, pos2)) ** 0.5
                                if d >= DEDUP_DIST_M:
                                    continue                 # otro punto -> sirve
                            repetida = True
                            break
                        if not repetida:
                            conservadas.append(img)
                            recientes.append((ts, q, pos, hay_pos))
                    quitadas = len(imgs) - len(conservadas)
                    if len(conservadas) < DEDUP_MIN_KEEP:
                        log(f"   PREFILTRO: quedarian {len(conservadas)} fotos "
                            f"(menos de {DEDUP_MIN_KEEP}). NO filtro, es muy arriesgado.")
                    elif quitadas > 0:
                        log(f"   PREFILTRO: {len(imgs)} -> {len(conservadas)} fotos "
                            f"({quitadas} repetidas quitadas sin gastar GPU, "
                            f"~{quitadas*25} pares de MASt3R ahorrados)")
                        if sin_datos:
                            log(f"   PREFILTRO: {sin_datos} fotos sin pose en el JSON, conservadas")
                        if _via_euler:
                            log(f"   PREFILTRO: {_via_euler} fotos venian de una version vieja de "
                                f"la app (sin cuaternion); convertidas desde los angulos de Euler")
                        imgs = conservadas
                    else:
                        log("   PREFILTRO: no habia repetidas evidentes")
            except Exception as e:
                log(f"   PREFILTRO fallo ({e}); sigo con todas las fotos")
        images_dir = WORK / "images"; images_dir.mkdir(exist_ok=True)
        _stamps = {}
        for i, img in enumerate(imgs):
            nuevo = f"foto_{i:04d}{img.suffix.lower()}"
            shutil.copy(img, images_dir / nuevo)
            # timestamp con el NOMBRE NUEVO: el PASO 2 ya no conoce los
            # originales, y lo usa el seguro anti-baldosa del filtro fino.
            try:
                r = poses_app.get(img.name)
                if r and float(r.get("timestamp", 0) or 0) > 0:
                    _stamps[Path(nuevo).stem] = float(r["timestamp"]) / 1000.0
            except Exception:
                pass
        if _stamps:
            (WORK / "timestamps.json").write_text(json.dumps(_stamps))
            log(f"   {len(_stamps)} timestamps guardados (seguro anti-textura-repetida activo)")
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
        run(["python", str(mast3r_py), str(images_dir), str(dataset),
             str(WORK / "timestamps.json")],
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

        # ── PASO 2c: priors monoculares — APAGADOS (2 experimentos fallidos) ──
        # INTENTO 2 (fase 0, job 47349919): se probo el prior de NORMALES SOLO,
        # con la profundidad en 0, pensando que el culpable era la profundidad.
        # RESULTADO: MISMO desastre. 139.570 pedazos sueltos (peor que los
        # 116.896 del intento 1), gaussianas 1.1M -> 2.19M, PSNR 33.5 -> 29.1,
        # malla cruda 272 MB, y el aplanado NO ENCONTRO NI UN PLANO (la malla
        # quedo tan laminada que RANSAC no reconocia una pared).
        # Asi que la normal tampoco sirve TAL COMO ESTA APLICADA. Sospecha
        # concreta pendiente de verificar: el signo. DSINE entrega la normal en
        # SU convencion de camara y aqui se usa cruda, sin comprobar que
        # coincida con la de 2DGS. Una normal invertida empuja los surfels a la
        # orientacion contraria: eso produce exactamente lo que se midio.
        # Queda el arreglo |coseno| (insensible al signo) + un diagnostico que
        # imprime el coseno medio; se activa con MONO_PRIORS=1 para UNA prueba.
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
            log("   PASO 2c saltado (MONO_PRIORS=0)")   # apagable con MONO_PRIORS=0

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
                log("   FASE 0: prior de NORMALES activo (peso %s); prior de PROFUNDIDAD "
                    "en %s (0 = apagado, fue el que causaba las laminas)"
                    % (os.environ.get("MONO_LAMBDA_NORMAL","0.1"),
                       os.environ.get("MONO_LAMBDA_DEPTH","0.0")))
            else:
                log("   AVISO: no encontré las anclas en train.py — entreno SIN priors")
        except Exception as e:
            log(f"   (no se pudo parchear priors en train.py: {e})")

        # ── PASO 3: entrenar 2DGS ──
        fase(0.45, f"PASO 3/5 — Entrenando superficie ({ITERS} iter)")
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
        # ── FASE 1: motor de superficie seleccionable ────────────────────────
        # MESH_ENGINE=pgsr (por defecto) -> PGSR: impone planaridad DURANTE el
        # entrenamiento (profundidad insesgada + consistencia multi-vista
        # fotometrica y geometrica). Es lo que ataca la ondulacion de 20-50 cm
        # que medimos como el 52% del ruido de pared y que ningun ajuste del
        # TSDF ni del aplanado posterior podia tocar.
        # MESH_ENGINE=2dgs -> el motor de siempre, intacto, como respaldo.
        # Banderas de PGSR recomendadas por sus autores para escenas de POCA
        # TEXTURA (exactamente paredes lisas): --max_abs_split_points 0 evita
        # que se sobreajuste partiendo puntos, y --opacity_cull_threshold 0.05
        # baja el numero de gaussianas.
        # PGSR se probo (job eefc0681) y NO mejoro las paredes: rugosidad 18.1 vs
        # 19.0 mm de 2DGS (5%, ruido) y ondulacion 20-50cm 13.3 vs 14.1 mm, pero
        # el render paso de 58 a 109 minutos y el PSNR bajo de 33.6 a 31.3.
        # Se vuelve a 2DGS. PGSR sigue en la imagen: MESH_ENGINE=pgsr lo enciende.
        _ENGINE = os.environ.get("MESH_ENGINE", "2dgs").lower()
        _PGSR_DIR = os.environ.get("PGSR_DIR", "/opt/pgsr")
        if _ENGINE == "pgsr" and not os.path.exists(os.path.join(_PGSR_DIR, "train.py")):
            log(f"   ⚠ PGSR no esta en la imagen ({_PGSR_DIR}): uso 2DGS")
            _ENGINE = "2dgs"
        log(f"   MOTOR DE SUPERFICIE: {_ENGINE.upper()}")
        if _ENGINE == "pgsr":
            # FALLO REAL (job 4660e8f8): PGSR lee el sparse de <dataset>/sparse/
            # SIN el /0, mientras MASt3R lo escribe en <dataset>/sparse/0/.
            # Su lector intenta primero BINARIO en sparse/, y al fallar cae a
            # TEXTO en sparse/ (nunca mira dentro de 0/), asi que reventaba con
            #   FileNotFoundError: .../dataset/sparse/images.txt
            # Se copian los tres .txt un nivel arriba. No se mueven ni se borran:
            # 2DGS sigue leyendo su sparse/0 igual que siempre.
            try:
                _sp0 = dataset / "sparse" / "0"
                _sp1 = dataset / "sparse"
                _cop = []
                for _f in ("cameras.txt", "images.txt", "points3D.txt"):
                    if (_sp0 / _f).exists():
                        shutil.copy(str(_sp0 / _f), str(_sp1 / _f)); _cop.append(_f)
                log("   PGSR: sparse copiado a %s -> %s (PGSR lo lee SIN el /0)"
                    % (_sp1, ", ".join(_cop) if _cop else "NADA"))
                if len(_cop) < 2:
                    log("   ⚠ PGSR: faltan archivos del sparse; uso 2DGS")
                    _ENGINE = "2dgs"
            except Exception as _ce:
                log(f"   ⚠ PGSR: no pude preparar el sparse ({_ce}); uso 2DGS")
                _ENGINE = "2dgs"
        def _entrenar(_dd):
            if _ENGINE == "pgsr":
                return run(["python", os.path.join(_PGSR_DIR, "train.py"),
                     "-s", str(dataset), "-m", str(dgs_out),
                     "--iterations", str(ITERS),
                     "-r", "1",
                     "--max_abs_split_points",
                     os.environ.get("PGSR_SPLIT_PTS", "0"),
                     "--opacity_cull_threshold",
                     os.environ.get("PGSR_OPACITY_CULL", "0.05"),
                     "--data_device", _dd],
                    fase_label="PASO 3/5 — Entrenando PGSR", check=False)
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
            raise RuntimeError(f"El entrenamiento falló (código {_rc_tr}).")
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
        # FASE 0: ajustables sin tocar codigo. MEDIDO en la malla (55): del ruido
        # total de pared (17.8 mm RMS), solo el 29% esta por debajo de 5 cm; el
        # 52% es ondulacion de 20-50 cm que NINGUN ajuste de voxel arregla. O sea
        # que subir el voxel tiene un techo de ~10-20%. Se dejan los knobs para
        # poder barrer sin gastar codigo: TSDF_DIV mas chico = voxel mas grande.
        voxel = max(_maxext / float(os.environ.get("TSDF_DIV", "500")), 0.005)
        sdf_trunc = float(os.environ.get("TSDF_TRUNC_K", "5.0")) * voxel          # banda ~5 voxeles (antes 4): cierra mejor los HUECOS
        #                                  en zonas de poca observación, a cambio de redondear
        #                                  un poquito los detalles finos (compromiso aceptable).
        depth_trunc = _diag * float(os.environ.get("TSDF_DEPTH_K", "1.3"))        # cubre el cuarto + margen; corta agujas lejanas
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
        if _ENGINE == "pgsr":
            # PGSR fusiona su propia malla (TSDF) con profundidad insesgada.
            # --use_depth_filter lo recomiendan sus autores cuando hay puntos
            # flotantes o vistas insuficientes: es justo nuestro caso y ataca
            # las laminas despegadas ANTES de que existan.
            _pg_cmd = ["python", os.path.join(_PGSR_DIR, "render.py"),
                       "-s", str(dataset), "-m", str(dgs_out),
                       "--skip_test",
                       "--max_depth", f"{depth_trunc:.6f}",
                       "--voxel_size", f"{voxel:.6f}",
                       "--num_cluster", os.environ.get("PGSR_CLUSTER", "50")]
            if os.environ.get("PGSR_DEPTH_FILTER", "1") == "1":
                _pg_cmd.append("--use_depth_filter")
            log("$ " + " ".join(_pg_cmd))
            rc_mesh, _salida_mesh = run(_pg_cmd, env=env_mesh,
                fase_label="PASO 4/5 — Extrayendo malla (PGSR)", check=False)
        else:
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
        candidatos = [c for c in candidatos
                      if "point_cloud" not in str(c).lower()]   # esa es la NUBE, no la malla
        for nombre in ("tsdf_fusion_post.ply", "tsdf_fusion.ply",   # PGSR
                       "fuse_post.ply", "fuse.ply",                 # 2DGS
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
            "    _nuevo = _V.copy(); _nplanos = 0; _planos = []\n"
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
            "        _planos.append((_nrm.copy(), float(_d)))\n"
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
            "    # GUARDAR LOS PLANOS PARA EL HORNEADOR.\n"
            "    # FALLO REAL (job e13efea4): el horneador volvia a buscar los planos por\n"
            "    # su cuenta con RANSAC sobre la malla de OpenMVS, y fallo: escribio\n"
            "    # 'BAKE TONO: no encontre planos grandes' y el nivelado NO corrio, aunque\n"
            "    # AQUI se habian encontrado 6 planos sin problema. Era una busqueda\n"
            "    # duplicada, peor y sin diagnostico. Ahora se guardan los de aqui, que ya\n"
            "    # pasaron los tres candados (tamano, periferia, normales), y el horneador\n"
            "    # los lee. Si el archivo no existe, el horneador cae a su metodo de antes.\n"
            "    try:\n"
            "        if _planos:\n"
            "            _pa = np.array([[float(n[0]), float(n[1]), float(n[2]), float(d)]\n"
            "                            for n, d in _planos], dtype=np.float64)\n"
            "            np.save('/workspace/job/planos.npy', _pa)\n"
            "            print('PLANOS guardados para el horneador: %d' % len(_pa), flush=True)\n"
            "    except Exception as _pe:\n"
            "        print('PLANOS (no pude guardarlos, el horneador buscara solo):', _pe, flush=True)\n"
            "    # ---- TAPAR HUECOS DE LA MALLA ----\n"
            "    # MEDIDO en la malla (58), soldando por POSICION (no por UV, que es lo que\n"
            "    # hace el .glb y confunde toda costura con un borde): la malla tiene 64.718\n"
            "    # aristas de borde reales = 4.627 HUECOS, de los cuales 186 son grandes, con\n"
            "    # 310 m de perimetro cayendo dentro de las PAREDES. Eso es geometria que\n"
            "    # FALTA: son los huecos que se ven en la pared.\n"
            "    # Se tapan con pymeshlab, que ya esta en la imagen. El tope de tamano es\n"
            "    # deliberado: un hueco pequeno o mediano es un defecto y se cierra; un hueco\n"
            "    # enorme suele ser una PUERTA o una VENTANA de verdad y NO se debe tapar.\n"
            "    try:\n"
            "        # MEDIDO en la (61): con 300 el cosido metio 23.571 caras nuevas en\n"
            "        # forma de ASTILLA (la mas larga, 1.402 veces mas larga que ancha).\n"
            "        # Esas astillas piden una huella enorme en el atlas: 198M muestras\n"
            "        # para 50M texeles, y el horneado paso de 0.7 a 17.6 minutos.\n"
            "        # Con 60 se tapan igual los huecos pequenos (la gran mayoria de los\n"
            "        # 4.627 medidos, incluido el que se veia en la pared) SIN astillas.\n"
            "        # REVERTIDO A 300. Bajarlo a 60 fue un error mio: culpe al cosido\n"
            "        # de huecos de las caras gigantes, y el log lo desmintio (195M\n"
            "        # muestras con 60 vs 198M con 300 = igual). A cambio, con 60\n"
            "        # REAPARECIO el hueco de la pared que con 300 ya no salia.\n"
            "        HOLE_MAX = int(os.environ.get('HOLE_MAX', '300'))\n"
            "        if HOLE_MAX > 0:\n"
            "            import pymeshlab as _pml\n"
            "            _Vh = np.array(m.vertices, dtype=np.float64)\n"
            "            _Th = np.asarray(m.triangles)\n"
            "            _msh = _pml.MeshSet()\n"
            "            _msh.add_mesh(_pml.Mesh(vertex_matrix=_Vh, face_matrix=_Th))\n"
            "            _nf0 = _Th.shape[0]\n"
            "            try:\n"
            "                _msh.meshing_close_holes(maxholesize=HOLE_MAX,\n"
            "                                         newfaceselected=False,\n"
            "                                         selfintersection=True)\n"
            "            except Exception:\n"
            "                _msh.apply_filter('meshing_close_holes', maxholesize=HOLE_MAX)\n"
            "            _Vn2 = _msh.current_mesh().vertex_matrix()\n"
            "            _Tn2 = _msh.current_mesh().face_matrix()\n"
            "            _add = _Tn2.shape[0] - _nf0\n"
            "            # CANDADO: si el filtro anade una barbaridad, algo salio mal y no se aplica\n"
            "            if _add < 0 or _add > 0.10 * _nf0:\n"
            "                print('HUECOS: el cierre anadiria %d caras (%.1f%% de la malla): '\n"
            "                      'lo descarto por seguridad' % (_add, 100.0*_add/max(_nf0,1)), flush=True)\n"
            "            else:\n"
            "                m.vertices = o3d.utility.Vector3dVector(_Vn2)\n"
            "                m.triangles = o3d.utility.Vector3iVector(_Tn2.astype(np.int32))\n"
            "                m.compute_vertex_normals()\n"
            "                print('HUECOS: %d caras nuevas tapando huecos de hasta %d aristas '\n"
            "                      '(esto es lo que se veia como huecos en la pared)'\n"
            "                      % (_add, HOLE_MAX), flush=True)\n"
            "            del _msh, _Vh, _Th\n"
            "    except Exception as _he:\n"
            "        print('HUECOS (fallo, sigo sin tapar):', _he, flush=True)\n"
            "    # ---- SNAP2: LAMINAS DESPEGADAS ----\n"
            "    # MEDIDO en la malla (54): 46 m2 de 83 despegados de su plano (el doble\n"
            "    # que Polycam en proporcion). El SNAP de arriba solo agarra lo que ya\n"
            "    # esta a menos de 2.5 cm; los trozos LEVANTADOS viven entre 3 y 15 cm y\n"
            "    # quedaban fuera. Eso se ve como trozos alzados en paredes y piso, y\n"
            "    # tambien como manchas de tono (cada lamina capta la luz distinto).\n"
            "    # Candados medidos sobre la malla real: las laminas estan a 4-9 cm, son\n"
            "    # planas (grosor <6 cm) y grandes; lo que esta a ~26 cm son MUEBLES y no\n"
            "    # se toca. Probado en la (54): 127 trozos pegados, 17.5 m2, movimiento\n"
            "    # medio 74 mm y maximo 231 mm, 0.45% de triangulos estirados.\n"
            "    try:\n"
            "        import scipy.sparse.csgraph as _csg\n"
            "        _BAND2 = float(os.environ.get('SNAP2_BAND', '0.15'))\n"
            "        _ALIN2 = float(os.environ.get('SNAP2_ALIN', '0.94'))\n"
            "        _AR2 = float(os.environ.get('SNAP2_AREA', '0.005'))\n"
            "        _NV2 = int(os.environ.get('SNAP2_NV', '20'))\n"
            "        _FL2 = float(os.environ.get('SNAP2_FLAT', '0.02'))\n"
            "        _GR2 = float(os.environ.get('SNAP2_GROS', '0.06'))\n"
            "        if _planos and os.environ.get('SNAP2', '1') == '1':\n"
            "            _V2 = np.asarray(m.vertices)\n"
            "            m.compute_vertex_normals()\n"
            "            _N2 = np.asarray(m.vertex_normals); _T2 = np.asarray(m.triangles)\n"
            "            _q0 = _V2[_T2[:,0]]; _q1 = _V2[_T2[:,1]]; _q2 = _V2[_T2[:,2]]\n"
            "            _arf = 0.5*np.linalg.norm(np.cross(_q1-_q0, _q2-_q0), axis=1)\n"
            "            _Vn2 = _V2.copy(); _nt2 = 0; _at2 = 0.0\n"
            "            for _nrm2, _d2 in _planos:\n"
            "                _dv = _V2 @ _nrm2 + _d2\n"
            "                _cand = ((np.abs(_N2 @ _nrm2) > _ALIN2) & (np.abs(_dv) > 0.03) & (np.abs(_dv) < _BAND2))\n"
            "                if _cand.sum() < 200: continue\n"
            "                _ix2 = np.flatnonzero(_cand)\n"
            "                _nc, _lb = _csg.connected_components(_A[_ix2][:,_ix2], directed=False)\n"
            "                _mp = -np.ones(len(_V2), dtype=np.int64); _mp[_ix2] = _lb\n"
            "                _fl = _mp[_T2]\n"
            "                _fok = (_fl[:,0]>=0)&(_fl[:,0]==_fl[:,1])&(_fl[:,1]==_fl[:,2])\n"
            "                _am = np.bincount(_fl[_fok,0], weights=_arf[_fok], minlength=_nc)\n"
            "                _nv2 = np.bincount(_lb, minlength=_nc); _dd = _dv[_ix2]\n"
            "                _s1 = np.bincount(_lb, weights=_dd, minlength=_nc)\n"
            "                _s2 = np.bincount(_lb, weights=_dd*_dd, minlength=_nc)\n"
            "                _mu = _s1/np.maximum(_nv2,1)\n"
            "                _sd = np.sqrt(np.maximum(_s2/np.maximum(_nv2,1) - _mu*_mu, 0))\n"
            "                _mx = np.full(_nc,-1e9); _mn = np.full(_nc,1e9)\n"
            "                np.maximum.at(_mx,_lb,_dd); np.minimum.at(_mn,_lb,_dd)\n"
            "                _kp = (_am>_AR2)&(_nv2>=_NV2)&(_sd<_FL2)&((_mx-_mn)<_GR2)\n"
            "                if not _kp.any(): continue\n"
            "                _mv = _ix2[_kp[_lb]]\n"
            "                _w2 = np.zeros(len(_V2)); _w2[_mv] = 1.0\n"
            "                _fr2 = np.zeros(len(_V2), dtype=bool); _fr2[_mv] = True\n"
            "                for _pw2 in (0.75, 0.5, 0.25):\n"
            "                    _vc2 = np.zeros(len(_V2), dtype=np.int8); _vc2[_fr2] = 1\n"
            "                    _nb2 = (_A @ _vc2) > 0\n"
            "                    _nu2 = _nb2 & (_w2 == 0); _w2[_nu2] = _pw2; _fr2 = _nu2\n"
            "                _mk2 = (_w2 > 0) & (np.abs(_dv) < _BAND2*1.5)\n"
            "                _Vn2[_mk2] -= (_w2[_mk2][:,None]) * ((_V2[_mk2] @ _nrm2 + _d2)[:,None] * _nrm2[None,:])\n"
            "                _nt2 += int(_kp.sum()); _at2 += float(_am[_kp].sum())\n"
            "            if _nt2:\n"
            "                _ds2 = np.linalg.norm(_Vn2-_V2, axis=1)\n"
            "                m.vertices = o3d.utility.Vector3dVector(_Vn2)\n"
            "                m.compute_vertex_normals()\n"
            "                print('SNAP2 laminas despegadas: %d trozos pegados (%.1f m2); movimiento medio %.0f mm, maximo %.0f mm' % (_nt2, _at2, 1000*_ds2[_ds2>1e-9].mean(), 1000*_ds2.max()), flush=True)\n"
            "            else:\n"
            "                print('SNAP2: no habia laminas despegadas que pegar', flush=True)\n"
            "    except Exception as _e2:\n"
            "        print('SNAP2 (fallo, sigo sin el):', _e2, flush=True)\n"
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

        # ── PASO 4d: TEXTURA UV + HORNEADOR v9.1 (plan P1 de la investigacion) ──
        #   OpenMVS solo pone el MAPA UV; luego el horneador repinta cada texel
        #   MEZCLANDO todas las fotos 12MP que lo ven (oclusion + peso angular),
        #   en dos bandas: baja=promedio (tono parejo por construccion, adios
        #   escalones) + alta=mejor foto (nitidez sin fantasmas). Atlas x2 ->
        #   ~0.075 cm/texel (UV viejo 0.15, vertice 0.84; Polycam 0.05).
        #   Probado en sintetico con camaras a exposiciones 0.75/1.0/1.3:
        #   salto de tono en costura 1.8 niveles (mejor-vista daria 20-36).
        #   Si el horneado falla -> atlas OpenMVS quedan; si todo el paso UV
        #   falla -> se sube el color por vertice (vertexB) como siempre.
        #   OMVS_POSEOPT=1 enciende el refinador Zhou-Koltun (P2) sin tocar codigo.
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
            log("   PASO 4d saltado (OPCION B, UV_TEXTURE=0): se entrega el COLOR POR VERTICE (tono uniforme, sin costuras)")

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
