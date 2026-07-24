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
BAKE_COSK     = float(os.environ.get("OMVS_BAKE_COSK", "2"))     # peso angular cos^k (k=2 rec