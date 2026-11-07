import argparse, os, subprocess, shutil, sys, time
from contextlib import contextmanager
import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

W, H = 1280, 720
PW, PH = 96, 64
FW, FH = 36, 24
EMA_ALPHA = 0.25
MOSAIC_OX = 1000
TOTAL_STAGES = 8
GREEN, RED, GRAY = (0, 220, 0), (0, 0, 230), (170, 170, 170)
MIN_VIS_TOTAL = 0.60     
MIN_VIS_CENTER = 0.97    
_ONES = np.ones((H, W), np.uint8)

def say(msg):
    print(msg, flush=True)


@contextmanager
def stage(i, title):
    say(f"\n[{i}/{TOTAL_STAGES}] {title}")
    t0 = time.time()
    yield
    say(f"     етап {i}: {time.time() - t0:.1f} с")

def load_frames(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames, fps

def estimate_homographies(frames, lags=(1, 2, 4, 8, 12, 16)):
    """H[t]: кадр t -> кадр 0. Щоб не накопичувати дрейф ланцюжком попарних гомографій,
    кадр t зіставляється одразу з кількома попередніми (t-1, t-2, t-4, ...), їхні збіги
    переводяться в систему кадру 0 і спільно (RANSAC) дають одну гомографію."""
    orb = cv2.ORB_create(3000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kd = [orb.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None) for f in frames]
    Hs, inl = [np.eye(3)], []
    for t in range(1, len(frames)):
        k1, d1 = kd[t]; P1, Q = [], []
        for j in lags:
            if t - j < 0:
                continue
            k0, d0 = kd[t - j]
            m = sorted(bf.match(d1, d0), key=lambda x: x.distance)[:500]
            if len(m) < 12:
                continue
            p = np.float32([k1[x.queryIdx].pt for x in m]); q = np.float32([k0[x.trainIdx].pt for x in m])
            P1.append(p)
            Q.append(cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hs[t - j]).reshape(-1, 2))
        h, mask = (None, None)
        if P1:
            h, mask = cv2.findHomography(np.vstack(P1), np.vstack(Q), cv2.RANSAC, 2.5)
        if h is None:                   
            h, mask = Hs[-1], np.zeros(1)
        inl.append(int(mask.sum())); Hs.append(h)
        if t % 50 == 0 or t == len(frames) - 1: print(f"    оброблено кадрів: {t}/{len(frames) - 1}")
    return Hs, inl

def build_mosaic(frames, Hs, ox=MOSAIC_OX, width=2300, height=780, step=10):
    T = np.array([[1, 0, ox], [0, 1, 0], [0, 0, 1]], float)
    canvas = np.zeros((height, width, 3), np.uint8)
    for t in range(len(frames) - 1, -1, -step):
        w = cv2.warpPerspective(frames[t], T @ Hs[t], (width, height)); m = w.sum(2) > 0; canvas[m] = w[m]
    return canvas

def project(poly_ref, Ht):
    pts = np.float32(poly_ref).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(Ht)).reshape(-1, 2)

def fully_visible(poly, margin=6):
    return bool(np.all((poly[:, 0] > margin) & (poly[:, 0] < W - margin) &
                       (poly[:, 1] > margin) & (poly[:, 1] < H - margin)))

def _rect_M(poly):
    dst = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])
    return cv2.getPerspectiveTransform(np.float32(poly), dst)

def rectify(frame, poly):
    """Патч місця. Частину, що вийшла за межі кадру, заповнюємо медіанним кольором видимої частини."""
    M = _rect_M(poly)
    p = cv2.warpPerspective(frame, M, (PW, PH), flags=cv2.INTER_AREA)
    ok = cv2.warpPerspective(_ONES, M, (PW, PH), flags=cv2.INTER_NEAREST)
    ok = cv2.erode(ok, np.ones((3, 3), np.uint8)) > 0
    if not ok.all():
        if ok.any():
            p[~ok] = np.median(p[ok], axis=0).astype(np.uint8)
        else:
            p[:] = 0
    return p

def slot_coverage(poly):
    """(частка площі місця в кадрі, частка центральної частини в кадрі)."""
    ok = cv2.warpPerspective(_ONES, _rect_M(poly), (PW, PH), flags=cv2.INTER_NEAREST)
    c = ok[int(.18 * PH):int(.82 * PH), int(.12 * PW):int(.88 * PW)]
    return float(ok.mean()), float(c.mean())

def usable_view(poly):
    try:
        tot, cen = slot_coverage(poly)
    except cv2.error:
        return False
    return tot >= MIN_VIS_TOTAL and cen >= MIN_VIS_CENTER

def features(patch):
    """Ознаки, стійкі до кольору авто: гістограми насиченості та яскравості центру місця +
    орієнтації градієнтів у сітці 4x8 комірок (форма авто: контури, вікна)."""
    c = patch[int(.18 * PH):int(.82 * PH), int(.12 * PW):int(.88 * PW)]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hs = np.histogram(hsv[..., 1], bins=8, range=(0, 256))[0]
    hv = np.histogram(hsv[..., 2], bins=8, range=(0, 256))[0]
    hist = np.concatenate([hs, hv]).astype(np.float32) / (c.shape[0] * c.shape[1])
    g = cv2.cvtColor(cv2.resize(patch, (64, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy); ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 8
    cells = []
    for i in range(0, 32, 8):
        for j in range(0, 64, 8):
            b = np.minimum(ang[i:i + 8, j:j + 8].astype(int), 7).ravel(); w = mag[i:i + 8, j:j + 8].ravel()
            h = np.bincount(b, weights=w, minlength=8)
            cells.append(h / (h.sum() + 1e-6) * np.log1p(w.sum()))
    dsc = patch_descriptor(patch) / np.array([1, 1, 100, 1, 100])    
    dsc[4] = abs(dsc[4] - 1.1)
    return np.concatenate([hist, 3 * np.concatenate(cells), 3 * dsc])

def collect(frames, Hs, slots, stride=1, stats=None):
    """Патчі місць, придатних для аналізу (в т.ч. частково видимих).
    stats[si] = [повністю видимих кадрів, частково видимих кадрів]."""
    recs = []
    for t in range(0, len(frames), stride):
        for si, s in enumerate(slots):
            poly = project(s["poly"], Hs[t])
            if usable_view(poly):
                recs.append((t, si, rectify(frames[t], poly), poly))
                if stats is not None:
                    st = stats.setdefault(si, [0, 0])
                    st[0 if fully_visible(poly) else 1] += 1
    return recs

def augment(frame, poly, rng, n=3):
    out = []
    for _ in range(n):
        q = poly + rng.normal(0, 4.0, poly.shape)
        p = rectify(frame, q).astype(np.float32)
        if rng.random() < .5: p = p[:, ::-1]
        if rng.random() < .5: p = p[::-1]
        p = np.clip(p * rng.uniform(.75, 1.25) + rng.normal(0, 4, p.shape), 0, 255).astype(np.uint8)
        out.append(p)
    return out

def make_mlp(seed=0):
    return make_pipeline(StandardScaler(), PCA(n_components=30, random_state=seed),
        MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=1e-1,
                      learning_rate_init=1e-3, max_iter=500, early_stopping=True,
                      validation_fraction=0.15, n_iter_no_change=15, random_state=seed))

def make_baseline(seed=0):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.05))

def train_set(frames, Hs, slots, recs, idx, rng, n_aug=3):
    X, y = [], []
    for k in idx:
        t, si, patch, poly = recs[k]; lab = slots[si]["label"]
        X.append(features(patch)); y.append(lab)
        for p in augment(frames[t], poly, rng, n_aug):
            X.append(features(p)); y.append(lab)
    return np.array(X), np.array(y)

def cross_validate(frames, Hs, slots, recs, n_splits, make_model, seed=0, name="модель"):
    rng = np.random.default_rng(seed)
    groups = np.array([r[1] for r in recs])
    Xe = np.array([features(r[2]) for r in recs])
    ye = np.array([slots[r[1]]["label"] for r in recs])
    prob = np.zeros(len(recs))
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(Xe, ye, groups), 1):
        Xtr, ytr = train_set(frames, Hs, slots, recs, tr, rng)
        if len(set(ytr.tolist())) < 2:         
            prob[te] = float(ytr[0]); print(f"    [{name}] фолд {k}: один клас у навчанні"); continue
        m = make_model(seed).fit(Xtr, ytr)
        prob[te] = m.predict_proba(Xe[te])[:, 1]
        fold_acc = ((prob[te] >= .5) == ye[te]).mean()
        print(f"    [{name}] фолд {k}/{n_splits}: train={len(Xtr)}, test={len(te)}, acc={fold_acc:.3f}")
    return prob, ye, groups

def visible_fraction(poly):
    """Частка площі чотирикутника, що потрапила в кадр."""
    total = abs(cv2.contourArea(np.float32(poly)))
    if total < 1:
        return 0.0
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.int32(poly)], 1)
    return float(m.sum()) / total


def render_video(frames, Hs, slots, recs, prob, fps, path):
    """EMA-прогноз оновлюється в кадрах, де місце придатне для аналізу (в т.ч. частково видиме).
    Рамка малюється, якщо видно >=40% місця; ще не класифіковані місця - сірим."""
    P = {(t, si): p for (t, si, _, _), p in zip(recs, prob)}
    ema = [None] * len(slots)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t, f in enumerate(frames):
        out = f.copy(); in_view = free_view = 0
        for si, s in enumerate(slots):
            if (t, si) in P:
                ema[si] = P[(t, si)] if ema[si] is None else (1 - EMA_ALPHA) * ema[si] + EMA_ALPHA * P[(t, si)]
            poly = project(s["poly"], Hs[t])
            if not fully_visible(poly) and visible_fraction(poly) < 0.4:
                continue
            if ema[si] is None:                  
                cv2.polylines(out, [np.int32(poly)], True, GRAY, 2, cv2.LINE_AA)
                continue
            occ = ema[si] >= .5; in_view += 1; free_view += (not occ)
            cv2.polylines(out, [np.int32(poly)], True, RED if occ else GREEN, 3, cv2.LINE_AA)
            c = poly.mean(0)
            cx, cy = int(np.clip(c[0], 30, W - 60)), int(np.clip(c[1], 20, H - 10))
            cv2.putText(out, f"{ema[si]:.2f}", (cx - 22, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
        known = [e for e in ema if e is not None]; free_known = sum(e < .5 for e in known)
        cv2.rectangle(out, (0, 0), (430, 92), (0, 0, 0), -1)
        cv2.putText(out, f"U kadri: vilnykh {free_view} z {in_view}", (10, 30), 0, .8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"Na karti: vilnykh {free_known} z {len(known)} (vsogo {len(slots)})", (10, 62), 0, .7, (200, 255, 200), 2, cv2.LINE_AA)
        vw.write(out)
        if t % 50 == 0 or t == len(frames) - 1:
            print(f"    записано кадрів: {t + 1}/{len(frames)}")
    vw.release()
    return ema


def reencode_h264(path):
    if shutil.which("ffmpeg") is None:
        return path
    tmp = path.replace(".mp4", "_h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", tmp])
    if r.returncode == 0: os.replace(tmp, path)
    return path

def find_default(exts, preferred=()):
    here = os.path.dirname(os.path.abspath(__file__))
    for n in preferred:
        p = os.path.join(here, n)
        if os.path.exists(p): return p
    for f in sorted(os.listdir(here)):
        if f.lower().endswith(exts): return os.path.join(here, f)
    return None

def _paint_mask(mosaic):
    """Тонкі світлі малонасичені структури (смуги розмітки), без білих кузовів авто."""
    valid = (mosaic.sum(2) > 0).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(mosaic, cv2.COLOR_BGR2HSV)[..., 1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)   
    m = ((tophat > 35) & (gray > 140) & (sat < 70) & (valid > 0)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m


def _segments(mask):
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=35, minLineLength=60, maxLineGap=25)
    if segs is None:
        return np.zeros((0, 4), np.float32)
    segs = np.asarray(segs, np.float32)
    return segs.reshape(-1, 4)       


def _dominant_angle(segs, tol_deg=8):
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    ang = (ang + 90) % 180 - 90                                 
    ln = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    hist = np.zeros(180)
    for a, l in zip(ang, ln):
        hist[int(round(a + 90)) % 180] += l
    hist = np.convolve(np.r_[hist[-4:], hist, hist[:4]], np.ones(9), "valid")
    a0 = np.argmax(hist) - 90.0
    d = (ang - a0 + 90) % 180 - 90
    ok = np.abs(d) < tol_deg
    a = a0 + np.average(d[ok], weights=ln[ok]) if ok.any() else a0
    return a, ok


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def _ridge_image(mosaic, A, size):
    """Відгук на світлі тонкі смуги (темні з обох боків) у повернутій системі, де лінії горизонтальні."""
    g = cv2.warpAffine(cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY), A, size).astype(np.float32)
    valid = cv2.warpAffine((mosaic.sum(2) > 0).astype(np.uint8) * 255, A, size)
    valid = cv2.erode(valid, np.ones((25, 25), np.uint8)) > 0
    g = cv2.GaussianBlur(g, (1, 5), 0)
    up, dn = np.roll(g, 14, 0), np.roll(g, -14, 0)
    ridge = np.minimum(g - 0.5 * (up + dn), g - np.maximum(up, dn) + 10)
    ridge[~valid] = 0
    ridge[g < 110] = 0
    return np.clip(ridge, 0, None), g


def detect_slots(mosaic, ox=0, min_len=70, verbose=True):
    """-> (slots, debug). Місця у координатах кадру 0. Алгоритм:
    кут розмітки (Hough) -> вирівнювання -> відгук на смуги -> довгі смуги групуються в колонки ->
    у колонці крок і зсув ґратки підбираються за профілем смуг (стійко до затулених ліній)."""
    mask = _paint_mask(mosaic)
    segs = _segments(mask)
    if len(segs) < 4:
        raise RuntimeError("Не знайдено ліній розмітки на мозаїці.")
    theta, ok = _dominant_angle(segs)
    th = np.radians(theta)
    d = np.array([np.cos(th), np.sin(th)])       
    n = np.array([-np.sin(th), np.cos(th)])       
    hh, ww = mask.shape
    cr = np.array([[0, 0], [ww, 0], [ww, hh], [0, hh]], np.float32)
    uc, vc = cr @ d, cr @ n
    u0, v0 = float(uc.min()), float(vc.min())
    A = np.float32([[d[0], d[1], -u0], [n[0], n[1], -v0]])
    size = (int(uc.max() - u0) + 1, int(vc.max() - v0) + 1)
    R, _ = _ridge_image(mosaic, A, size)

    sm = cv2.blur(R, (81, 1))
    mk = (sm > 28).astype(np.uint8) * 255
    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))
    nlab, _, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
    lines = []
    for i in range(1, nlab):
        x, y, w_, h_, _a = st[i]
        if w_ >= min_len and h_ <= 26:
            lines.append(dict(v=float(y + h_ / 2), a=float(x), b=float(x + w_)))
    if verbose:
        print(f"    довгих смуг розмітки: {len(lines)}, кут розмітки: {theta:.1f}°")

    dsu = _DSU(len(lines))
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            ov = min(lines[i]["b"], lines[j]["b"]) - max(lines[i]["a"], lines[j]["a"])
            sh = min(lines[i]["b"] - lines[i]["a"], lines[j]["b"] - lines[j]["a"])
            cdist = abs((lines[i]["a"] + lines[i]["b"]) - (lines[j]["a"] + lines[j]["b"])) / 2
            if ov > 0.5 * sh and cdist < 100:
                dsu.u(i, j)
    cols = {}
    for i in range(len(lines)):
        cols.setdefault(dsu.f(i), []).append(lines[i])
    cols = [sorted(c, key=lambda L: L["v"]) for c in cols.values() if len(c) >= 2]

    valid_m = (mosaic.sum(2) > 0).astype(np.uint8)
    polys, pitches = [], []
    for c in cols:
        ua_ = float(np.percentile([L["a"] for L in c], 25)); ub_ = float(np.percentile([L["b"] for L in c], 75))
        lo, hi = int(ua_), int(ub_)
        prof = R[:, lo:hi].mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 5), 0).ravel()
        pk = np.maximum.reduce([np.roll(prof, k) for k in range(-3, 4)])
        vmin, vmax = c[0]["v"], c[-1]["v"]
        best = (-1, None, None)
        for p in np.arange(105, 151, 1.0):
            for o in np.arange(vmin - p, vmin + 1, 1.0):
                ks = np.arange(0, int((vmax - o) // p) + 2)
                pos = o + ks * p
                pos = pos[(pos >= vmin - 6) & (pos <= vmax + 6)]
                if len(pos) < 2:
                    continue
                sc = float(np.mean(pk[pos.astype(int)])) * (1 + 0.05 * len(pos))
                if sc > best[0]:
                    best = (sc, p, o)
        _, p, o = best
        if p is None:
            continue
        ks = np.arange(0, int((vmax - o) // p) + 2)
        pos = o + ks * p
        pos = pos[(pos >= vmin - 6) & (pos <= vmax + 6)]
        pitches.append(p)
        thr = 0.5 * float(np.median(pk[pos.astype(int)]))
        ref = []
        for v_ in pos:
            lo_, hi_ = int(max(v_ - 14, 0)), int(min(v_ + 15, len(prof)))
            j = lo_ + int(np.argmax(prof[lo_:hi_]))
            ref.append(float(j) if prof[j] > thr and prof[j] > 1.5 * np.median(prof[lo_:hi_]) else float(v_))
        pos = np.array(ref)

        def poly_at(v_a, v_b):
            cs = [(ua_, v_a), (ub_, v_a), (ub_, v_b), (ua_, v_b)]
            return np.array([(uu + u0) * d + (vv + v0) * n for uu, vv in cs])

        def inside_frac(poly):
            m_ = np.zeros(valid_m.shape, np.uint8)
            cv2.fillPoly(m_, [np.int32(poly)], 1)
            return float((m_ & valid_m).sum()) / max(float(m_.sum()), 1.0)

        pos = list(pos)
        for _ in range(2):
            if inside_frac(poly_at(pos[-1], pos[-1] + p)) >= 0.5:
                pos.append(pos[-1] + p)
            else:
                break
        for _ in range(2):
            if inside_frac(poly_at(pos[0] - p, pos[0])) >= 0.5:
                pos.insert(0, pos[0] - p)
            else:
                break
        for v_a, v_b in zip(pos[:-1], pos[1:]):
            polys.append(poly_at(v_a, v_b))
    if verbose and pitches:
        print(f"    ширина місця по колонках: {[round(x) for x in pitches]}px, колонок: {len(pitches)}")
    slots = polys
    slots.sort(key=lambda p: (round(p[:, 0].mean() / 150), p[:, 1].mean()))
    out = [{"id": f"S{i + 1:02d}",
            "poly": [[round(float(x - ox), 1), round(float(y), 1)] for x, y in p],
            "label": None} for i, p in enumerate(slots)]
    if verbose:
        print(f"    автоматично виявлено місць: {len(out)}")
    return out, dict(mask=mask, theta=theta, lines=lines, R=R)

def patch_descriptor(patch):
    """Ознаки «відхилення від асфальту» для центральної частини місця:
    частка дуже світлих і дуже темних пікселів, насиченість, густина країв, середня яскравість."""
    h, w = patch.shape[:2]
    c = patch[int(.18 * h):int(.82 * h), int(.12 * w):int(.88 * w)]
    g0 = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g0, (3, 3), 0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    sat = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    return np.array([(g0 > 170).mean(), (g0 < 45).mean(), sat.mean(), (mag > 60).mean(), g0.mean()])


def auto_label(slots, recs, verbose=True):
    """Ставить slots[i]['label'] без учителя: медіанний дескриптор місця за всі кадри ->
    оцінка відхилення від типового асфальту (макс. z-оцінка за ознаками); поріг 0.7σ -> зайняте.
    Повертає (labels, дескриптори)."""
    per = {}
    for t, si, patch, poly in recs:
        per.setdefault(si, []).append(patch_descriptor(patch))
    ids = sorted(per)
    D = np.array([np.median(per[i], axis=0) for i in ids])
   
    D2 = D.copy()
    D2[:, 4] = np.abs(D[:, 4] - np.median(D[:, 4]))
    med = np.median(D2, axis=0)
    score = (np.maximum(D2 - med, 0) / (D2.std(0) + 1e-9)).max(1)
    lab = (score >= 0.7).astype(int)            
    for k, i in enumerate(ids):
        slots[i]["label"] = int(lab[k])
        slots[i]["score"] = float(score[k])
    for s in slots:                              
        if s["label"] is None:
            s["label"] = 0
    if verbose:
        print(f"    авторозмітка: зайнятих {int(lab.sum())}, вільних {int((1 - lab).sum())} "
              f"(із {len(ids)} видимих місць)")
    return lab, dict(zip(ids, D.tolist()))


def report_slots(slots, stats, usable):
    say("    місце | повних | часткових | мітка   | score | у навчанні")
    for i, s in enumerate(slots):
        f, p = stats.get(i, (0, 0))
        lab = "зайняте" if s["label"] else "вільне"
        say(f"    {s['id']:>5} | {f:6d} | {p:9d} | {lab:7} | {s.get('score', 0):5.2f} | {'так' if i in usable else 'ні'}")


def save_debug(mosaic, slots, usable, path):
    img = mosaic.copy()
    for i, s in enumerate(slots):
        p = np.int32(np.array(s["poly"]) + [MOSAIC_OX, 0])
        col = GRAY if i not in usable else (RED if s["label"] else GREEN)
        cv2.polylines(img, [p], True, col, 2, cv2.LINE_AA)
        c = p.mean(0).astype(int)
        cv2.putText(img, s["id"], (int(c[0]) - 18, int(c[1])), 0, .5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=find_default((".mp4", ".avi", ".mov", ".mkv"), ("video.mp4",)))
    ap.add_argument("--output", default=os.path.join(here, "result.mp4"), help="вихідне відео")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--min-line", type=int, default=80, help="мін. довжина лінії розмітки (px на мозаїці)")
    ap.add_argument("--debug", action="store_true", help="зберегти мозаїку з виявленими місцями (mosaic_debug.jpg)")
    a = ap.parse_args()
    if not a.video or not os.path.exists(a.video):
        raise SystemExit("Не знайдено відео. Покладіть його поруч зі скриптом або вкажіть --video шлях.")
    t_start = time.time()

    with stage(1, "Читання відео"):
        frames, fps = load_frames(a.video)
        if not frames: raise SystemExit("Не вдалося прочитати кадри з відео.")
        say(f"    файл: {os.path.basename(a.video)}; кадрів: {len(frames)}; fps: {fps:.1f}; "
            f"тривалість: {len(frames) / fps:.1f} с; робочий розмір: {W}x{H}")

    with stage(2, "Рух камери (ORB + RANSAC-гомографії)"):
        Hs, inl = estimate_homographies(frames)
        ia = np.array(inl) if inl else np.zeros(1, int)
        say(f"    інлайєрів на кадр: мін {ia.min()}, медіана {np.median(ia):.0f}, макс {ia.max()}; "
            f"слабких (<30): {int((ia < 30).sum())}")
        ctr = np.array([cv2.perspectiveTransform(np.float32([[[W / 2, H / 2]]]), h)[0, 0] for h in Hs])
        say(f"    зсув центру кадру відносно кадру 0: x {ctr[:, 0].min():.0f}..{ctr[:, 0].max():.0f}, "
            f"y {ctr[:, 1].min():.0f}..{ctr[:, 1].max():.0f} px")

    with stage(3, "Мозаїка"):
        mosaic = build_mosaic(frames, Hs)
        cov = (mosaic.sum(2) > 0).mean()
        say(f"    розмір: {mosaic.shape[1]}x{mosaic.shape[0]}, покрито: {cov * 100:.0f}%")

    with stage(4, "Пошук місць за розміткою"):
        slots, _ = detect_slots(mosaic, ox=MOSAIC_OX, min_len=a.min_line)
        if not slots: raise SystemExit("Місця не знайдено. Спробуйте зменшити --min-line.")

    with stage(5, "Видимість місць у кадрах"):
        stats = {}
        recs = collect(frames, Hs, slots, stats=stats)
        n_by_slot = np.bincount([r[1] for r in recs], minlength=len(slots))
        usable = {i for i, n in enumerate(n_by_slot) if n >= 6}
        partial_only = sum(1 for i in usable if stats[i][0] == 0)
        say(f"    патчів: {len(recs)} (повних {sum(v[0] for v in stats.values())}, "
            f"часткових {sum(v[1] for v in stats.values())})")
        say(f"    місць придатних для навчання (>=6 кадрів): {len(usable)} з {len(slots)}; "
            f"із них лише з часткових кадрів: {partial_only}")

    with stage(6, "Автоматична розмітка зайняте/вільне"):
        auto_label(slots, recs)
        recs = [r for r in recs if r[1] in usable]
        report_slots(slots, stats, usable)
        if a.debug:
            p = os.path.join(here, "mosaic_debug.jpg"); save_debug(mosaic, slots, usable, p)
            say(f"    налагоджувальна мозаїка: {p}")
        n_occ = sum(slots[i]["label"] for i in usable)
        say(f"    у навчанні: зайнятих {n_occ}, вільних {len(usable) - n_occ}")
    if len(usable) < 3 or n_occ in (0, len(usable)):
        raise SystemExit("Недостатньо місць обох класів для навчання.")
    folds = min(a.folds, len(usable))

    with stage(7, f"Навчання та крос-валідація по місцях ({folds} фолдів)"):
        prob, ye, groups = cross_validate(frames, Hs, slots, recs, folds, make_mlp, name="MLP")
        prob_b, _, _ = cross_validate(frames, Hs, slots, recs, folds, make_baseline, name="LogReg")
        acc = ((prob >= .5) == ye).mean(); acc_b = ((prob_b >= .5) == ye).mean()
        pred = prob >= .5
        tp = int((pred & (ye == 1)).sum()); tn = int((~pred & (ye == 0)).sum())
        fp = int((pred & (ye == 0)).sum()); fn = int((~pred & (ye == 1)).sum())
        say(f"    точність поза-навчальних місць: MLP {acc:.3f}, LogReg {acc_b:.3f}")
        say(f"    MLP матриця помилок: TP={tp} TN={tn} FP={fp} FN={fn}")
        bad = [slots[g]["id"] for g in sorted(set(groups.tolist()))
               if (pred[groups == g] != (ye[groups == g] == 1)).mean() > 0.5]
        say(f"    місця, де MLP переважно помиляється: {bad if bad else 'немає'}")

    with stage(8, "Рендер відео"):
        ema = render_video(frames, Hs, slots, recs, prob, fps, a.output)
        reencode_h264(a.output)
        known = [e for e in ema if e is not None]
        say(f"    підсумок по карті: вільних {sum(e < .5 for e in known)} з {len(known)} класифікованих "
            f"(усього місць {len(slots)})")
    say(f"\nГотово за {time.time() - t_start:.0f} с -> {os.path.abspath(a.output)}")


if __name__ == "__main__":
    main()