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
# SCRIPT DE TEXTURIZADO UV (corre en el pod como subproceso, estilo Polycam)
# ────────────────────────────────────────────────────────────────────────────
# Separa el COLOR de la GEOMETRÍA: en vez de pintar cada vértice (que obliga a
# malla densa/rugosa = "braille"), hornea una TEXTURA UV (imagen) sobre la malla.
# ANTI-DESALINEACIÓN (lo que arruinó OpenMVS): en vez de pegar UNA foto por cara
# (que se tuerce con cualquier error de pose), PROMEDIA todas las fotos visibles
# por texel, usando las MISMAS imágenes y poses del entrenamiento 2DGS. La
# visibilidad/oclusión se resuelve con raycasting (cada foto solo aporta a las
# superficies que realmente ve). El error de pose se reparte en un leve desenfoque,
# no en cortes torcidos. Validado localmente con datos sintéticos.
TEXTURE_SCRIPT = r'''
import sys, os, gc, json, struct
import numpy as np
from PIL import Image
print("   [tex] iniciando horneado de textura UV", flush=True)
try:
    import xatlas
except Exception:
    print("   [tex] instalando xatlas...", flush=True)
    os.system(sys.executable + " -m pip install xatlas --quiet")
    import xatlas
import open3d as o3d
import trimesh

MESH_PLY   = sys.argv[1]
IMAGES_DIR = sys.argv[2]
SPARSE_DIR = sys.argv[3]
OUT_GLB    = sys.argv[4]
TEXSIZE    = int(sys.argv[5]) if len(sys.argv) > 5 else 2048

def log(s): print("   [tex] " + s, flush=True)

# Conversion sRGB <-> lineal. Las fotos vienen en sRGB (gamma). Promediar en sRGB
# OSCURECE (promedio de blanco+negro da 50% de valor pero solo ~22% de luz). Hay que
# convertir a LINEAL, promediar, y volver a sRGB. Arregla la oscuridad en bordes/sombras.
def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def linear_to_srgb(c):
    c = np.maximum(c, 0.0)
    return np.clip(np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055), 0, 1)

# 1) malla
m = o3d.io.read_triangle_mesh(MESH_PLY)
V = np.asarray(m.vertices); F = np.asarray(m.triangles)
if len(V) == 0 or len(F) == 0:
    log("malla vacia, abortando"); sys.exit(1)
log("malla %d vert %d caras" % (len(V), len(F)))

# 2) UV unwrap (xatlas duplica vertices en costuras → vmapping mapea al original)
vmapping, indices, uvs = xatlas.parametrize(V, F)
Vn = np.ascontiguousarray(V[vmapping].astype(np.float64))
Fn = np.ascontiguousarray(indices.astype(np.int32))
UV = np.ascontiguousarray(uvs.astype(np.float64))
log("UV unwrap: %d vert, uv[%.3f..%.3f]" % (len(Vn), UV.min(), UV.max()))

# 3) escena de raycasting (para visibilidad/oclusion)
mn = o3d.geometry.TriangleMesh()
mn.vertices = o3d.utility.Vector3dVector(Vn)
mn.triangles = o3d.utility.Vector3iVector(Fn)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mn))
INVALID = scene.INVALID_ID

# 4) intrinsecos (cameras.txt) + poses world->cam (images.txt)
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

# 5) HORNEAR: por cada foto, raycast (visibilidad) + splat al texel promediando
acc  = np.zeros((TEXSIZE, TEXSIZE, 3), np.float64)
wsum = np.zeros((TEXSIZE, TEXSIZE), np.float64)
Ktens = {}   # cache de tensores K (reutilizar EVITA un segfault de Open3D)
nbaked = 0
for cid, name, E, R, t in views:
    if cid not in cams: continue
    W, H, fx, fy, cx, cy = cams[cid]
    path = os.path.join(IMAGES_DIR, name)
    if not os.path.exists(path): continue
    photo = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    photo = srgb_to_linear(photo)   # a LINEAL para promediar bien (anti-oscuro)
    Hp, Wp = photo.shape[:2]
    sx = Wp / float(W); sy = Hp / float(H)   # por si la foto difiere de cameras.txt
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
    uvt = UV[Fn[ti]]
    uvp = b0[:, None]*uvt[:, 0] + b1[:, None]*uvt[:, 1] + b2[:, None]*uvt[:, 2]
    col = photo[yy, xx]
    # peso por angulo de vista: la camara que ve la superficie DE FRENTE pesa mas
    P3 = Vn[Fn[ti]]
    pos = b0[:, None]*P3[:, 0] + b1[:, None]*P3[:, 1] + b2[:, None]*P3[:, 2]
    Ccam = -R.T @ t
    vd = Ccam[None, :] - pos
    vd /= (np.linalg.norm(vd, axis=1, keepdims=True) + 1e-9)
    nh = nrm[hit]; nh /= (np.linalg.norm(nh, axis=1, keepdims=True) + 1e-9)
    w = np.clip(np.abs((nh*vd).sum(1)), 0.05, 1.0)
    # ARREGLO BORROSO: elevar el peso a la 4 → la vista que ve la superficie DE FRENTE
    # domina mucho más que las oblicuas. Como las poses tienen error (~0.1°), promediar
    # muchas vistas desalineadas desenfoca; al dar casi todo el peso a la mejor vista, la
    # textura sale MÁS NÍTIDA (menos "tela"/borroso). Sin costuras duras (sigue siendo
    # promedio, solo muy sesgado a la mejor vista).
    w = w ** 4
    tx_i = np.clip(uvp[:, 0]*(TEXSIZE-1), 0, TEXSIZE-1).astype(int)
    ty_i = np.clip((1-uvp[:, 1])*(TEXSIZE-1), 0, TEXSIZE-1).astype(int)
    np.add.at(acc,  (ty_i, tx_i), col*w[:, None])
    np.add.at(wsum, (ty_i, tx_i), w)
    nbaked += 1
    del photo, tri, bary, nrm, thit, hit, col; gc.collect()
log("horneadas %d/%d camaras" % (nbaked, len(views)))
if nbaked == 0:
    log("ninguna camara horneada, abortando"); sys.exit(1)

# 6) normalizar (en LINEAL) + volver a sRGB + realce de brillo
cov = wsum > 0
tex = np.zeros((TEXSIZE, TEXSIZE, 3), np.float32)
tex_lin = np.zeros((TEXSIZE, TEXSIZE, 3), np.float32)
tex_lin[cov] = (acc[cov] / wsum[cov, None]).astype(np.float32)   # promedio en LINEAL
tex[cov] = linear_to_srgb(tex_lin[cov])                          # de vuelta a sRGB
tex[cov] = np.clip(tex[cov] ** 0.8, 0, 1)                        # gamma 0.8 = realce de brillo (anti-oscuro)
log("cobertura textura %.1f%%" % (cov.mean()*100))
mask = cov.copy()
for _ in range(16):
    if mask.all(): break
    s = np.zeros_like(tex); c = np.zeros(mask.shape)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        sh = np.roll(np.roll(tex, dy, 0), dx, 1)
        mh = np.roll(np.roll(mask, dy, 0), dx, 1).astype(np.float64)
        s += sh * mh[..., None]; c += mh
    fill = (~mask) & (c > 0)
    tex[fill] = (s[fill] / c[fill, None]).astype(np.float32)
    mask = mask | fill

# 7) exportar .glb con la textura UV (color separado de la geometria)
tex_img = Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8))
mesh_out = trimesh.Trimesh(vertices=Vn, faces=Fn, process=False)
# ── ARREGLO 1 (anti-facetado): forzar NORMALES SUAVES. Sin esto el .glb sale SIN
#    normales y el visor calcula caras planas → se ven los triangulos. Al acceder a
#    vertex_normals, trimesh promedia las caras por vertice (suave) y las mete al .glb.
_ = mesh_out.vertex_normals
# ── ARREGLO 2 (anti-oscuro): material MATE NO-METALICO. Por defecto glTF asume
#    metallic=1.0 → la superficie sale OSCURA sin reflejos de entorno. metallic=0 +
#    roughness=1 = superficie mate que muestra bien la textura.
_mat = trimesh.visual.material.PBRMaterial(
    baseColorTexture=tex_img, metallicFactor=0.0, roughnessFactor=1.0)
mesh_out.visual = trimesh.visual.TextureVisuals(uv=UV, image=tex_img, material=_mat)
mesh_out.export(OUT_GLB)

# ── ARREGLO 3 (anti-oscuro, definitivo): marcar el material como KHR_materials_unlit.
#    Las fotos YA traen la iluminacion real horneada, asi que NO queremos que el visor
#    la re-ilumine. "unlit" = el visor muestra la textura tal cual (ni oscura ni quemada).
try:
    _d = bytearray(open(OUT_GLB, "rb").read())
    _jlen = struct.unpack("<I", _d[12:16])[0]
    _g = json.loads(_d[20:20+_jlen].decode("utf-8"))
    _g.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in _g["extensionsUsed"]:
        _g["extensionsUsed"].append("KHR_materials_unlit")
    for _m in _g.get("materials", []):
        _m.setdefault("extensions", {})["KHR_materials_unlit"] = {}
    _nj = json.dumps(_g, separators=(",", ":")).encode("utf-8")
    while len(_nj) % 4:
        _nj += b" "
    _bin = _d[20+_jlen:]
    _out = bytearray()
    _out += _d[:12]
    _out += struct.pack("<I", len(_nj)) + b"JSON" + _nj
    _out += _bin
    struct.pack_into("<I", _out, 8, len(_out))
    open(OUT_GLB, "wb").write(_out)
    log("material marcado unlit (textura sin re-iluminacion)")
except Exception as _e:
    log("no pude marcar unlit (%s); sigo con material mate" % _e)
log("textura UV %dx%d exportada a .glb" % (TEXSIZE, TEXSIZE))
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

        # ── PASO 3: entrenar 2DGS ──
        fase(0.45, f"PASO 3/5 — Entrenando 2DGS ({ITERS} iter)")
        dgs_out = WORK / "output"; dgs_out.mkdir(exist_ok=True)
        # --lambda_dist 25 : regularizador de DISTORSIÓN de profundidad. ANTES estaba
        # en 100 — DEMASIADO ALTO. La investigación encontró que ese valor era la
        # CAUSA RAÍZ de que el render saliera dañado: en paredes lisas sin textura el
        # término de distorsión se disparaba (picos de 0.17-0.49) y SACUDÍA las
        # gaussianas en vez de asentarlas → geometría deformada, huecos, cuarto
        # incompleto, y la calidad (PSNR) llegó a CAER de 27 a 25.8 con más iteraciones.
        # Además era INESTABLE: a veces salía bien (PSNR 32.5) y a veces fatal (25.8)
        # con el mismo input. Bajándolo a 25 el entrenamiento es estable y reproducible
        # (junto con la semilla fija). --lambda_normal 0.05 es el default oficial (antes
        # 0.1, el doble): mantiene las paredes planas sin el riesgo de la distorsión alta.
        _rc_tr, _out_tr = run(["python", "/opt/2dgs/train.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--iterations", str(ITERS),
             "--lambda_dist", "25",
             "--lambda_normal", "0.05",
             "-r", "1",                   # usar la resolución COMPLETA de las imágenes (1000px), no reducir
             "--data_device", "cpu"],     # imágenes en RAM (no en VRAM) → evita quedarse sin memoria a alta resolución
            fase_label="PASO 3/5 — Entrenando 2DGS")
        log("   2DGS entrenado")
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
                if psnr_final >= 30:
                    log(f"   ✓ CALIDAD OK: PSNR final {psnr_final:.1f} (buena base estable)")
                else:
                    log(f"   ⚠⚠⚠ CALIDAD BAJA: PSNR final {psnr_final:.1f} (< 30). La malla "
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
        log(f"$ python /opt/2dgs/render.py (BOUNDED) --depth_ratio 0 "
            f"--voxel_size {voxel:.4f} --sdf_trunc {sdf_trunc:.4f} "
            f"--depth_trunc {depth_trunc:.2f} --num_cluster 50  (OMP=8)")
        rc_mesh, _salida_mesh = run(
            ["python", "/opt/2dgs/render.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--skip_train", "--skip_test",
             "--depth_ratio", "0",
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
            "try:\n"
            "    m = m.filter_smooth_taubin(number_of_iterations=8)\n"
            "    print('SMOOTH Taubin 8 iter (alisa rugosidad/braille) OK', flush=True)\n"
            "except Exception as e:\n"
            "    print('SMOOTH (fallo, sigo):', e, flush=True)\n"
            # --- 3) DECIMAR a ~1.2M — la config que MEJOR se vio (Fase 0) ---
            #     Decimar agresivo (a 300k/600k) FRAGMENTÓ la malla en >1000 pedazos
            #     → eso causaba los HUECOS. 1.2M es una reducción suave (~2×) que NO
            #     rompe la malla. Con NORMALES SUAVES en el .glb no se ven triángulos
            #     aunque sea densa. FAIL-SAFE: si pesa mucho, probar 900k (sin bajar
            #     de ahí, porque por debajo empieza a fragmentar y salen huecos).
            "target = 1200000\n"
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
                if isinstance(sc, trimesh.Scene):
                    for _g in sc.geometry.values():
                        _ = _g.vertex_normals
                else:
                    _ = sc.vertex_normals
                log("   normales suaves forzadas en el .glb (anti-facetado)")
            except Exception as _ne:
                log(f"   ⚠ no pude forzar normales ({_ne}); el visor podría facetar")
            sc.export(str(glb_final))
            log(f"   .glb (color por vértice + AO): {glb_final.stat().st_size/1e6:.1f} MB")
        except Exception as e:
            log(f"   ⚠ no se pudo exportar .glb ({e}); subo el .ply")

        # ══════════════════════════════════════════════════════════════════════
        # PASO 4c: HORNEAR TEXTURA UV (estilo Polycam) — FASE 0 de validación
        # ──────────────────────────────────────────────────────────────────────
        # Separa el color de la geometría: hornea una imagen de alta resolución
        # sobre la malla (en vez de color por vértice). PROMEDIA todas las fotos
        # visibles por texel usando las MISMAS imágenes y poses del entrenamiento
        # 2DGS → no se tuerce como OpenMVS. Si falla, se sube el .glb de color por
        # vértice (no se pierde el render). Esta corrida valida si la textura ALINEA.
        glb_tex = WORK / "mesh_textured.glb"
        try:
            fase(0.94, "PASO 4c/5 — Horneando textura UV (estilo Polycam)")
            bake_py = WORK / "bake_texture.py"
            bake_py.write_text(TEXTURE_SCRIPT)
            run(["python", str(bake_py), str(malla), str(dataset / "images"),
                 str(dataset / "sparse" / "0"), str(glb_tex), "2048"],
                fase_label="PASO 4c/5 — Horneando textura UV", check=False)
            if glb_tex.exists() and glb_tex.stat().st_size > 1000:
                log(f"   ✓ textura UV horneada: {glb_tex.stat().st_size/1e6:.1f} MB")
            else:
                log("   ⚠ el texturizado no produjo archivo; uso color por vértice")
        except Exception as e:
            log(f"   ⚠ texturizado falló ({e}); uso color por vértice (.glb)")

        # Archivo a subir (orden de preferencia):
        #   1º textura UV (estilo Polycam)  2º color por vértice + AO  3º .ply crudo
        if glb_tex.exists() and glb_tex.stat().st_size > 1000:
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
        callback("completed", frames_used=n_fotos, ply_mb=round(ply_mb, 1),
                 seconds=round(seconds), log="\n".join(_LOG))

    except Exception as e:
        _estado["vivo"] = False
        log(f"✗ ERROR: {e}")
        callback("error", error_message=str(e), log="\n".join(_LOG))
        sys.exit(1)


if __name__ == "__main__":
    main()
