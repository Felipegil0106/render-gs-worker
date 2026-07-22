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
    """NIVELADO DE TONO POR PARCHE (Opcion C).

    El escalon de tono NO esta entre los 6 materiales (medido: ya balanceados),
    sino entre los miles de PARCHES (islas del atlas) dentro de cada material:
    cada parche viene de una foto distinta. Aqui, en el atlas, SI conozco las
    islas, asi que:
      1) cada isla = un componente conexo de pixeles NO-relleno del atlas;
      2) dos islas son VECINAS si comparten un vertice 3D de la malla;
      3) por cada costura mido la diferencia de tono entre los dos lados;
      4) resuelvo UNA ganancia por canal por isla (minimos cuadrados en log,
         luz LINEAL, con prior hacia 1) para que las vecinas queden iguales;
      5) aplico la ganancia a los pixeles de esa isla.
    Como es UNA ganancia por isla, solo mueve el TONO: el detalle fino de la
    textura queda intacto (verificado en el archivo real: detalle 13.2 -> 13.6).
    Si algo falla, devuelve False y la textura queda como estaba.
    """
    import numpy as _np
    from PIL import Image
    from scipy import ndimage as _ndi
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spl
    _t = time.time()

    def _s2l(c): return _np.where(c <= 0.04045, c/12.92, ((c+0.055)/1.055)**2.4)
    def _l2s(c):
        c = _np.clip(c, 0.0, 1.0)
        return _np.where(c <= 0.0031308, c*12.92, 1.055*(c**(1.0/2.4)) - 0.055)

    # ── leer el OBJ: vertices 3D, UVs, caras y a que textura va cada cara ──
    Vn = []; Tn = []; F = []; FT = []; FM = []
    cur = -1
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
    if NF < 1000 or len(Tn) == 0 or (FT < 0).any():
        log("TONO: el OBJ no trae UVs utilizables; dejo la textura como esta"); return False

    # ── islas del atlas: componentes conexos de pixeles NO-relleno ──
    labs = []; ncomp = []; imgs_lin = []
    for tf in texfiles:
        a = _np.asarray(Image.open(tf).convert("RGB"))
        fill = (_np.abs(a[:, :, 0].astype(_np.int16) - 128) < 6) & \
               (_np.abs(a[:, :, 1].astype(_np.int16) - 128) < 6) & \
               (_np.abs(a[:, :, 2].astype(_np.int16) - 128) < 6)
        lb, nc = _ndi.label(~fill)
        labs.append(lb.astype(_np.int32)); ncomp.append(int(nc))
        imgs_lin.append(_s2l(a.astype(_np.float32) / 255.0))
    off = _np.cumsum([0] + [n + 1 for n in ncomp])       # id global de isla
    NISL = int(off[-1])

    # ── a que isla y con que color cae cada cara (centro del triangulo) ──
    cuv = Tn[FT].mean(axis=1)
    isl = _np.full(NF, -1, _np.int64)
    col = _np.zeros((NF, 3), _np.float32)
    _flip_votes = [0, 0]
    for ti in range(len(texfiles)):
        m = FM == ti
        if not m.any(): continue
        H, W = labs[ti].shape
        px = _np.clip((cuv[m, 0] * (W - 1)).astype(_np.int64), 0, W - 1)
        for _fl in (0, 1):                                # auto-detectar el sentido de V
            vv = (1.0 - cuv[m, 1]) if _fl == 0 else cuv[m, 1]
            py = _np.clip((vv * (H - 1)).astype(_np.int64), 0, H - 1)
            _flip_votes[_fl] += int((labs[ti][py, px] > 0).sum())
    _flip = 0 if _flip_votes[0] >= _flip_votes[1] else 1
    for ti in range(len(texfiles)):
        m = FM == ti
        if not m.any(): continue
        H, W = labs[ti].shape
        px = _np.clip((cuv[m, 0] * (W - 1)).astype(_np.int64), 0, W - 1)
        vv = (1.0 - cuv[m, 1]) if _flip == 0 else cuv[m, 1]
        py = _np.clip((vv * (H - 1)).astype(_np.int64), 0, H - 1)
        lb = labs[ti][py, px]
        isl[m] = _np.where(lb > 0, off[ti] + lb, -1)
        col[m] = imgs_lin[ti][py, px]
    ok = (isl >= 0) & (col.min(1) > 0.002)
    if ok.sum() < 1000:
        log("TONO: no pude ubicar las caras en el atlas; dejo la textura como esta"); return False

    # ── costuras: vertices 3D donde se tocan DOS islas distintas ──
    vi = F.reshape(-1); fi = _np.repeat(_np.arange(NF), 3)
    good = ok[fi]
    vi = vi[good]; fi = fi[good]
    o = _np.argsort(vi, kind="stable"); vi = vi[o]; fi = fi[o]
    isf = isl[fi]
    starts = _np.r_[0, _np.flatnonzero(_np.diff(vi)) + 1]
    mn = _np.minimum.reduceat(isf, starts); mx = _np.maximum.reduceat(isf, starts)
    seamg = _np.flatnonzero(mn != mx)
    ends = _np.r_[starts[1:], len(vi)]
    lg = _np.log(_np.maximum(col, 1e-4))
    acc = {}
    for gi in seamg:
        a, b = starts[gi], ends[gi]
        ii = isf[a:b]; ff = fi[a:b]
        uq = _np.unique(ii)
        if len(uq) < 2: continue
        med = {int(u): lg[ff[ii == u]].mean(0) for u in uq}
        for x in range(len(uq)):
            for y in range(x + 1, len(uq)):
                A, B = int(uq[x]), int(uq[y])
                d = _np.clip(med[B] - med[A], -0.7, 0.7)
                k = (A, B)
                if k in acc: acc[k][0] += d; acc[k][1] += 1
                else: acc[k] = [d.copy(), 1]
    pairs = [(k, v) for k, v in acc.items() if v[1] >= TONE_MINF]
    if len(pairs) < 50:
        log("TONO: solo %d costuras utiles; dejo la textura como esta" % len(pairs)); return False

    # ── resolver UNA ganancia por canal por isla (minimos cuadrados) ──
    NP = len(pairs)
    ri = _np.repeat(_np.arange(NP), 2)
    ci = _np.empty(NP * 2, _np.int64); dv = _np.empty(NP * 2, _np.float64)
    w = _np.empty(NP); rhs = _np.empty((NP, 3))
    for i, ((A, B), (sd, n)) in enumerate(pairs):
        ww = min(n, 60) ** 0.5
        ci[2*i] = A; ci[2*i+1] = B
        dv[2*i] = ww; dv[2*i+1] = -ww
        w[i] = ww; rhs[i] = sd / n
    lam = 0.10                                            # prior: ganancia ~ 1
    M = _sp.vstack([
        _sp.coo_matrix((dv, (ri, ci)), shape=(NP, NISL)),
        _sp.identity(NISL, format="coo") * lam]).tocsr()
    G = _np.ones((NISL, 3))
    for c in range(3):
        b = _np.r_[rhs[:, c] * w, _np.zeros(NISL)]
        x = _spl.lsqr(M, b, atol=1e-6, btol=1e-6, iter_lim=400)[0]
        G[:, c] = _np.exp(x)
    lo, hi = 1.0 / TONE_CLAMP, TONE_CLAMP
    nclamp = int(((G < lo) | (G > hi)).sum()); G = _np.clip(G, lo, hi)

    # ── cuanto baja el escalon (medido sobre las mismas costuras) ──
    dif0 = _np.array([abs(float((sd / n).mean())) for (_k, (sd, n)) in pairs])
    lgG = _np.log(G).mean(1)
    dif1 = _np.array([abs(float((sd / n).mean()) - (lgG[A] - lgG[B]))
                      for ((A, B), (sd, n)) in pairs])
    red = 100.0 * (1.0 - (dif1.mean() / max(dif0.mean(), 1e-9)))

    # ── aplicar la ganancia a los pixeles de cada isla y guardar ──
    for ti, tf in enumerate(texfiles):
        g = _np.ones((ncomp[ti] + 1, 3))
        g[1:] = G[off[ti] + 1: off[ti] + ncomp[ti] + 1]
        out = _l2s(imgs_lin[ti] * g[labs[ti]])
        out = (_np.clip(out, 0, 1) * 255.0 + 0.5).astype(_np.uint8)
        im = Image.fromarray(out)
        if tf.lower().endswith((".jpg", ".jpeg")): im.save(tf, quality=95)
        else: im.save(tf)
    log("TONO: %d islas, %d costuras -> nivelado por parche en %.1f min"
        % (NISL, NP, (time.time() - _t) / 60.0))
    log("TONO: escalon medio en costuras BAJO %.0f%% | ganancias %.2f-%.2f "
        "(%d en tope; solo mueve el tono, el detalle queda intacto)"
        % (red, float(G.min()), float(G.max()), nclamp))
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
