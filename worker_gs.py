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
#  v8.2 (esta): pido OBJ (que OpenMVS escribe perfecto) y lo convierto a glb
#        limpio con trimesh (texturas bien incrustadas). Ademas: el "nivelado
#        GLOBAL de costuras" de OpenMVS tiene un bug (_Map_base::at) que
#        crashea con esta malla -> lo dejo APAGADO por defecto (queda el local).
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
SMOOTH_RATIO  = os.environ.get("OMVS_SMOOTH", "1.5")             # ALTO = islas GRANDES (menos fragmentacion; mata el vitral)
GLOBAL_SEAM   = os.environ.get("OMVS_GLOBAL_SEAM", "0")          # 0 = apagado (su version global tiene un bug)
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

# ── 3) decimar la malla al conteo del glb final ────────────────────────────
import open3d as o3d
m = o3d.io.read_triangle_mesh(MESH)
nt0 = len(m.triangles)
if nt0 > TEX_MESH_TRIS:
    m = m.simplify_quadric_decimation(target_number_of_triangles=TEX_MESH_TRIS)
    m.remove_unreferenced_vertices()
m.remove_degenerate_triangles(); m.remove_duplicated_vertices()
MFT = os.path.join(WORK, "mesh_for_tex.ply")
m2 = o3d.geometry.TriangleMesh(m.vertices, m.triangles)
o3d.io.write_triangle_mesh(MFT, m2)
log("malla para textura: %d -> %d caras" % (nt0, len(m2.triangles)))

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
def texcmd(max_tex, gseam):
    c = [TEXM, "-i", SCENE, "-m", MFT, "-o", BASE + ".obj",
         "--export-type", "obj",
         "--resolution-level", str(RES_LEVEL),
         "--max-texture-size", str(max_tex),
         "--outlier-threshold", str(OUTLIER),
         "--cost-smoothness-ratio", str(SMOOTH_RATIO),
         "--global-seam-leveling", str(gseam)]
    return c

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
    texfiles = []
    if mtlpath and os.path.exists(mtlpath):
        for l in open(mtlpath):
            p = l.split()
            if p and p[0] == "map_Kd":
                tf = os.path.join(objdir, l.split(None, 1)[1].strip())
                if os.path.exists(tf): texfiles.append(tf)
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
    # (b) cargar la Scene (texturas ya corregidas) y exportar SIN concatenar
    obj = trimesh.load(objf, process=False)
    obj.export(outglb)
    # (c) material MATE + UNLIT
    _patch_unlit_matte(outglb)
    return os.path.exists(outglb) and os.path.getsize(outglb) > 200000


# AUTO-SANADOR: intento bueno (4096); si se cae por RAM, uno mas liviano (2048).
CONFIGS = [(MAX_TEX, GLOBAL_SEAM, OMP_HI), (2048, GLOBAL_SEAM, "2")]
final = None
for ci, (mt, gs, omp) in enumerate(CONFIGS):
    for f in glob.glob(BASE + ".*"):
        try: os.remove(f)
        except Exception: pass
    envt = dict(os.environ); envt["OMP_NUM_THREADS"] = str(omp)
    tag = "TextureMesh cfg%d (tex=%d gseam=%s omp=%s)" % (ci+1, mt, gs, omp)
    rc = run(texcmd(mt, gs), tag, env=envt)
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
