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
             # DSP-SIFT: forma afín + domain-size pooling. Recomendación OFICIAL
             # de COLMAP para registrar MÁS fotos en zonas difíciles (paredes
             # lisas, poca textura). Features mucho más discriminativos → engancha
             # donde el SIFT normal fallaba (las 16 fotos que no se registraban).
             # Es más lento (CPU) pero es la causa #1 del cuarto incompleto.
             "--SiftExtraction.estimate_affine_shape", "1",
             "--SiftExtraction.domain_size_pooling", "1",
             # CLAVE del error -9 (OOM): COLMAP-CPU abre 1 hilo por núcleo y
             # cada uno carga una foto en RAM. En máquinas con muchos vCPU
             # (como la A6000) eso revienta la memoria. Con 4 hilos, máximo
             # 4 fotos en RAM a la vez → estable en cualquier GPU.
             "--SiftExtraction.num_threads", "4"])
        fase(0.25, "PASO 2/5 — COLMAP emparejando fotos")
        run(["colmap", "exhaustive_matcher",
             "--database_path", str(db), "--SiftMatching.use_gpu", "0",
             # Emparejamiento guiado: usa la geometría para encontrar MÁS
             # correspondencias (ayuda a registrar las fotos difíciles).
             "--SiftMatching.guided_matching", "1",
             # Mismo motivo que arriba: limitar hilos evita el OOM (-9).
             "--SiftMatching.num_threads", "4"])
        fase(0.35, "PASO 2/5 — COLMAP reconstruyendo (mapper)")
        # Mapper ESTRICTO (umbrales por defecto): la investigación demostró que
        # el mapper TOLERANTE de la ronda pasada (que subió el registro de 111 a
        # 118 fotos) metió poses de BAJA CALIDAD. Y las poses imprecisas ROMPEN la
        # fusión del TSDF → paredes onduladas y huecos. CALIDAD de poses > cantidad
        # de fotos. Volvemos a estricto: menos fotos, pero poses consistentes.
        # ba_refine_focal_length: refina la cámara durante el bundle adjustment
        # (mejor precisión de poses → TSDF fusiona mejor las superficies).
        run(["colmap", "mapper",
             "--database_path", str(db), "--image_path", str(images_dir),
             "--output_path", str(sparse),
             "--Mapper.ba_refine_focal_length", "1",
             "--Mapper.ba_refine_principal_point", "0"])
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
        # DIAGNÓSTICO: ¿CUÁLES fotos quedaron sin registrar? (las que cubren las
        # zonas faltantes del cuarto). Leemos los nombres del modelo y los
        # comparamos con las fotos en disco. No es crítico: si falla, seguimos.
        try:
            def _nombres_registrados(images_bin):
                nombres = set()
                with open(images_bin, "rb") as f:
                    num = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(num):
                        f.read(4)                              # image_id
                        f.read(32); f.read(24); f.read(4)      # qvec, tvec, camera_id
                        bs = bytearray()
                        while True:                            # nombre (termina en \0)
                            c = f.read(1)
                            if c in (b"\x00", b""):
                                break
                            bs += c
                        nombres.add(bs.decode("utf-8", "ignore"))
                        npts = struct.unpack("<Q", f.read(8))[0]
                        f.read(npts * 24)                      # saltar puntos 2D
                return nombres
            reg = _nombres_registrados(modelo / "images.bin")
            todas = {p.name for p in images_dir.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
            faltan = sorted(todas - reg)
            if faltan:
                muestra = ", ".join(faltan[:20]) + (" ..." if len(faltan) > 20 else "")
                log(f"   FOTOS SIN REGISTRAR ({len(faltan)}): {muestra}")
                log("   (esas fotos son las zonas que faltan; míralas para ver qué cubren)")
            else:
                log("   ✓ TODAS las fotos se registraron")
        except Exception as e:
            log(f"   (no se pudo listar fotos sin registrar: {e})")
        if registradas < n_fotos * 0.5:
            log(f"   ⚠ OJO: solo se registró {registradas}/{n_fotos} fotos. "
                f"La malla puede salir incompleta (poco solape entre fotos).")
        # DIAGNÓSTICO DE CALIDAD DE POSES: error de reproyección y longitud de
        # tracks. Poses sanas = error < ~1px y tracks largos. Si el error es alto,
        # las poses son imprecisas y el TSDF no fusionará bien (paredes onduladas).
        # No es crítico: si falla, seguimos.
        try:
            import subprocess as _sp
            r = _sp.run(["colmap", "model_analyzer", "--path", str(modelo)],
                        capture_output=True, text=True, timeout=120)
            for linea in (r.stdout + r.stderr).splitlines():
                l = linea.strip()
                if any(k in l for k in ("Cameras:", "Images:", "Points:",
                                        "reprojection error", "Mean track",
                                        "Mean observations")):
                    log(f"   [poses] {l}")
        except Exception as e:
            log(f"   (model_analyzer no disponible: {e})")
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
        def _centros_camara(images_bin):
            centros = []
            with open(images_bin, "rb") as f:
                num = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num):
                    f.read(4)                                  # image_id
                    qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                    tx, ty, tz = struct.unpack("<3d", f.read(24))
                    f.read(4)                                  # camera_id
                    while f.read(1) not in (b"\x00", b""):     # nombre (termina en \0)
                        pass
                    npts = struct.unpack("<Q", f.read(8))[0]
                    f.read(npts * 24)                          # saltar puntos 2D
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
            cps = _centros_camara(dataset / "sparse" / "0" / "images.bin")
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
        log(f"$ python /opt/2dgs/render.py --unbounded --mesh_res 1024 --num_cluster 100  (OMP=8)")
        rc_mesh, _salida_mesh = run(
            ["python", "/opt/2dgs/render.py",
             "-s", str(dataset), "-m", str(dgs_out),
             "--skip_train", "--skip_test",
             "--depth_ratio", "0",
             "--unbounded",
             "--mesh_res", "1024",
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
            "print('DIAG vertices', len(m.vertices), 'triangulos', n0)\n"
            # --- DIAGNÓSTICOS (envueltos: si fallan, NO rompen la malla) ---
            # Nos dicen: tamaño del cuarto (bbox), cuántos pedazos sueltos hay, y
            # de cada pedazo grande su tamaño y QUÉ TAN LEJOS está del centro.
            # La "ventana flotante" = un pedazo pequeño MUY lejos del centro.
            "try:\n"
            "    aabb = m.get_axis_aligned_bounding_box()\n"
            "    ext = aabb.get_extent(); cg = aabb.get_center()\n"
            "    print('DIAG bbox_global X=%.2f Y=%.2f Z=%.2f (unidades COLMAP)' % (ext[0], ext[1], ext[2]))\n"
            "    cl = m.cluster_connected_triangles()\n"
            "    lab = np.asarray(cl[0]); ntri = np.asarray(cl[1])\n"
            "    print('DIAG componentes_conexas', len(ntri))\n"
            "    V = np.asarray(m.vertices); T = np.asarray(m.triangles)\n"
            "    order = np.argsort(ntri)[::-1]\n"
            "    for k, i in enumerate(order[:8]):\n"
            "        mask = lab == i\n"
            "        vidx = np.unique(T[mask].reshape(-1)); vv = V[vidx]\n"
            "        cmin = vv.min(0); cmax = vv.max(0); c = (cmin+cmax)/2; sz = cmax-cmin\n"
            "        d = float(np.linalg.norm(c - cg)); pct = 100.0*ntri[i]/max(n0,1)\n"
            "        print('DIAG comp%d: %d tri (%.1f%%) tamano(%.2f,%.2f,%.2f) dist_al_centro=%.2f' % (k, int(ntri[i]), pct, sz[0], sz[1], sz[2], d))\n"
            "except Exception as e:\n"
            "    print('DIAG (fallo diagnostico, sigo):', e)\n"
            # --- DECIMACIÓN ---
            "target = 500000\n"
            "if n0 > target:\n"
            "    m = m.simplify_quadric_decimation(target_number_of_triangles=target)\n"
            "m.remove_unreferenced_vertices()\n"
            "m.remove_degenerate_triangles()\n"
            "m.remove_duplicated_vertices()\n"
            "m.compute_vertex_normals()\n"
            "m.compute_triangle_normals()\n"
            f"o3d.io.write_triangle_mesh(r'{decimada}', m)\n"
            "print('DECIMATE triangulos', n0, '->', len(m.triangles))\n"
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
