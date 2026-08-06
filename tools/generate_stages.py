"""Generate one PNG per stage of the world generation process (colour renders)."""

import os
import sys

import numpy

sys.path.insert(0, "/workspace")

from worldengine.biome import biome_name_to_index
from worldengine.draw import _biome_colors
from worldengine.generation import (
    Step,
    add_noise_to_elevation,
    center_land,
    initialize_ocean_and_thresholds,
    place_oceans_at_map_borders,
)
from worldengine.image_io import PNGWriter
from worldengine.plates import _plates_simulation
from worldengine.simulations.biome import BiomeSimulation
from worldengine.simulations.erosion import ErosionSimulation
from worldengine.simulations.humidity import HumiditySimulation
from worldengine.simulations.hydrology import WatermapSimulation
from worldengine.simulations.icecap import IcecapSimulation
from worldengine.simulations.irrigation import IrrigationSimulation
from worldengine.simulations.permeability import PermeabilitySimulation
from worldengine.simulations.precipitation import PrecipitationSimulation
from worldengine.simulations.temperature import TemperatureSimulation

OUT = "/workspace/stage_images"
W, H = 512, 512
SEED = 1
NUM_PLATES = 10

# ---------------- colour maps (normalised 0..1) ----------------

TERRAIN = [  # hypsometric-style: deep sea -> shallow sea -> lowland -> hills -> mountains -> snow
    (0.00, (20, 40, 110)),
    (0.18, (40, 90, 160)),
    (0.32, (80, 145, 200)),
    (0.42, (95, 175, 205)),
    (0.48, (70, 150, 85)),
    (0.60, (135, 185, 95)),
    (0.72, (200, 175, 115)),
    (0.85, (160, 135, 115)),
    (1.00, (250, 250, 250)),
]
TEMPERATURE = [
    (0.00, (0, 0, 150)),
    (0.25, (30, 110, 210)),
    (0.50, (120, 200, 130)),
    (0.75, (255, 200, 50)),
    (1.00, (220, 30, 20)),
]
PRECIPITATION = [  # dry -> wet
    (0.00, (255, 255, 210)),
    (0.33, (190, 230, 160)),
    (0.66, (70, 180, 210)),
    (1.00, (10, 70, 180)),
]
SEA_DEPTH = [  # shallow -> deep
    (0.00, (120, 190, 230)),
    (0.50, (40, 100, 190)),
    (1.00, (5, 20, 90)),
]
HUMIDITY = [  # arid -> humid
    (0.00, (230, 220, 150)),
    (0.50, (120, 190, 90)),
    (1.00, (20, 110, 40)),
]
PERMEABILITY = [  # low -> high
    (0.00, (150, 120, 80)),
    (0.50, (190, 170, 110)),
    (1.00, (120, 190, 110)),
]
WATERMAP = [  # none -> creek -> river -> main river
    (0.00, (245, 245, 235)),
    (0.60, (90, 180, 235)),
    (0.90, (20, 90, 220)),
    (1.00, (5, 30, 120)),
]
ICECAP = [
    (0.00, (150, 190, 225)),
    (1.00, (255, 255, 255)),
]

PLATE_PALETTE = numpy.array(
    [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
        (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
        (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    ],
    dtype=float,
)


def ensure_out():
    os.makedirs(OUT, exist_ok=True)


def save_color(array, name, stops):
    path = os.path.join(OUT, name)
    stops = sorted(stops, key=lambda s: s[0])
    xs = numpy.array([s[0] for s in stops], dtype=float)
    cs = numpy.array([s[1] for s in stops], dtype=float)
    a = numpy.asarray(array, dtype=float)
    amin, amax = a.min(), a.max()
    if amax == amin:
        amax = amin + 1.0
    n = (a - amin) / (amax - amin)
    rgb = numpy.stack(
        [numpy.interp(n, xs, cs[:, i]) for i in range(3)], axis=-1
    ).astype(numpy.uint8)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def save_plates(array, name):
    path = os.path.join(OUT, name)
    n_plates = int(array.max()) + 1
    palette = numpy.resize(PLATE_PALETTE, (n_plates, 3))
    rgb = palette[array.astype(int)].astype(numpy.uint8)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def _kmeans(cent, k, offset):
    """Cluster a small set of centroids into k groups. Returns {plate_index: group_id}."""
    n = len(cent)
    if n == 0:
        return {}
    if n <= k:
        return {i: offset + i for i in range(n)}
    centroids = numpy.array([cent[p] for p in cent], dtype=float)
    order = numpy.argsort(centroids[:, 0])
    init = sorted(order[numpy.linspace(0, n - 1, k).round().astype(int)])
    centers = centroids[init].copy()
    assign = numpy.zeros(n, dtype=int)
    for _ in range(100):
        dist = ((centroids[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        nassign = dist.argmin(1)
        if numpy.array_equal(nassign, assign):
            break
        assign = nassign
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = centroids[m].mean(0)
    return {i: offset + a for i, a in enumerate(assign)}


def _kmeans_centers(points, k, seed=0):
    """K-means over 2D point coordinates. Returns k centers."""
    rng = numpy.random.RandomState(seed)
    n = len(points)
    if n == 0:
        return numpy.zeros((0, 2))
    if n <= k:
        return points.astype(float).copy()
    init = rng.choice(n, k, replace=False)
    centers = points[init].astype(float)
    assign = numpy.zeros(n, dtype=int)
    for _ in range(60):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        nassign = dist.argmin(1)
        if numpy.array_equal(nassign, assign):
            break
        assign = nassign
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = points[m].mean(0)
    return centers


def _nearest_mask_pixel(mask, y, x, limit=160):
    """Closest pixel where mask is True to (y, x); falls back to (y, x)."""
    h, w = mask.shape
    y = int(round(y)); x = int(round(x))
    y = min(max(y, 0), h - 1); x = min(max(x, 0), w - 1)
    for r in range(limit + 1):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        pts = numpy.argwhere(mask[y0:y1, x0:x1])
        if len(pts):
            d2 = ((pts - numpy.array([y - y0, x - x0])) ** 2).sum(-1)
            best = pts[int(d2.argmin())]
            return (best[0] + y0, best[1] + x0)
    return (y, x)


def _soft_seeds(ocean, n_ocean, n_land):
    """Seed positions: oceanic seeds near the map centre (left/right), land
    seeds around the rim. Each is snapped onto the closest matching pixel."""
    h, w = ocean.shape
    cy, cx = h / 2.0, w / 2.0
    ideal_ocean = [(cy, cx * 0.58), (cy, cx * 1.42)]
    ideal_land = [(h * 0.13, cx), (h * 0.87, cx), (cy, w * 0.11), (cy, w * 0.89)]
    seeds = [_nearest_mask_pixel(ocean, y, x) for (y, x) in ideal_ocean]
    seeds += [_nearest_mask_pixel(numpy.logical_not(ocean), y, x) for (y, x) in ideal_land]
    return seeds


def _fractal_noise(shape, seed=7, octaves=4, base_low=60):
    """Multi-octave fractal noise: large waviness + mid ripples + fine jitter.

    Each octave halves the effective low-res scale and amplitude so borders
    look like natural coastlines instead of a single smooth bend."""
    from PIL import Image
    n = numpy.zeros(shape, dtype=float)
    for o in range(octaves):
        low = max(int(base_low / (2 ** o)), 4)
        amp = 0.5 ** o
        rng = numpy.random.RandomState(seed + o * 31)
        small = rng.rand(low, low)
        img = Image.fromarray((small * 255.0).astype(numpy.uint8))
        img = img.resize(shape, Image.BILINEAR)
        n += amp * (numpy.asarray(img) / 255.0)
    n -= n.min()
    n /= n.max() + 1e-12
    return n


def _downsample_labels(labels, factor=2):
    """Majority-vote downsampling: each factor x factor block -> mode label."""
    h, w = labels.shape
    nh, nw = h // factor, w // factor
    out = numpy.empty((nh, nw), dtype=labels.dtype)
    for y in range(nh):
        yy = y * factor
        for x in range(nw):
            xx = x * factor
            block = labels[yy : yy + factor, xx : xx + factor]
            vals, cnt = numpy.unique(block, return_counts=True)
            out[y, x] = vals[int(cnt.argmax())]
    return out


def _noise_perturbed_split(shape, seeds, w=130.0, weight=None):
    """Nearest-seed split with fractal noise perturbing each distance field.

    Multi-octave noise (large bends + mid ripples + fine jitter) gives the
    borders a natural, coast-like irregularity.  When a per-pixel weight map
    (e.g. from elevation) is passed, noise amplitude varies spatially so
    continental boundaries are wilder and deep-ocean boundaries smoother."""
    h_w, wdt = shape
    k = len(seeds)
    yy, xx = numpy.mgrid[0:h_w, 0:wdt].astype(float)
    dist = numpy.empty((h_w, wdt, k))
    for j, (sy, sx) in enumerate(seeds):
        dist[:, :, j] = numpy.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)
    for j in range(k):
        n = _fractal_noise(shape, seed=7 + j, octaves=5, base_low=50)
        if weight is not None:
            n = n * weight
        dist[:, :, j] += w * n
    return dist.argmin(-1)


def _soft_voronoi(shape, seeds, iterations=80, t_hi=45.0, t_lo=12.0):
    """Annealed softmax Voronoi with isotropic diffusion -> smooth round borders.

    Uses euclidean (not squared) distance so the exp() stays in range, and the
    temperature is scaled against the map size."""
    h, w = shape
    k = len(seeds)
    yy, xx = numpy.mgrid[0:h, 0:w].astype(float)
    dist = numpy.empty((h, w, k))
    for j, (sy, sx) in enumerate(seeds):
        dist[:, :, j] = numpy.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)
    rng = numpy.random.RandomState(0)
    P = rng.rand(h, w, k)
    P /= P.sum(-1, keepdims=True)
    for it in range(iterations):
        t = t_hi + (t_lo - t_hi) * it / max(iterations - 1, 1)
        target = numpy.exp(-dist / t)
        target /= target.sum(-1, keepdims=True)
        Psm = numpy.zeros_like(P)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                Psm += numpy.roll(numpy.roll(P, dy, 0), dx, 1)
        Psm /= 9.0
        P = numpy.sqrt(P * Psm)
        P *= target
        P /= P.sum(-1, keepdims=True)
    return P.argmax(-1)


def _label_components(mask):
    """8-connected component labelling via flood fill."""
    from collections import deque
    h, w = mask.shape
    lab = numpy.zeros((h, w), dtype=int)
    cur = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and lab[y, x] == 0:
                cur += 1
                q = deque([(y, x)])
                lab[y, x] = cur
                while q:
                    cy, cx = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                                lab[ny, nx] = cur
                                q.append((ny, nx))
    return lab, cur


def _smooth_labels(labels, iters=14):
    """Anti-alias a hard label map: one-hot, isotropic box blur, re-argmax.

    The blur radius (iterations) removes pixel-staircase jaggies while the
    low-frequency waviness of the borders is preserved."""
    k = int(labels.max()) + 1
    h, w = labels.shape
    P = numpy.zeros((h, w, k), dtype=float)
    for j in range(k):
        P[:, :, j] = labels == j
    for _ in range(iters):
        Psm = numpy.zeros_like(P)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                Psm += numpy.roll(numpy.roll(P, dy, 0), dx, 1)
        P = Psm / 9.0
    return P.argmax(-1)


def _majority_smooth(plates, iters=5):
    """Median-of-9 filter rounds off jagged pixel edges."""
    out = plates.astype(int).copy()
    for _ in range(iters):
        stacked = numpy.empty((9,) + out.shape, dtype=out.dtype)
        i = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                stacked[i] = numpy.roll(numpy.roll(out, dy, 0), dx, 1)
                i += 1
        out = numpy.median(stacked, axis=0).astype(out.dtype)
    return out


def _remove_enclaves(plates, min_frac=0.0005):
    """Merge tiny components (enclaves) into the surrounding plate."""
    from collections import Counter
    h, w = plates.shape
    min_area = int(min_frac * h * w)
    out = plates.astype(int).copy()
    for g in numpy.unique(plates):
        mask = out == g
        lab, n = _label_components(mask)
        for c in range(1, n + 1):
            comp = lab == c
            if int(comp.sum()) < min_area:
                cnt = Counter()
                ys, xs = numpy.where(comp)
                for y, x in zip(ys, xs):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and out[ny, nx] != g:
                                cnt[out[ny, nx]] += 1
                if cnt:
                    dominant = cnt.most_common(1)[0][0]
                    out[comp] = dominant
    return out


def merge_plates(world, n_ocean=2, n_land=4):
    """Merge the many small raw plates into n_ocean + n_land large plates.

    Uses plate-adjacency BFS from oceanic and continental seeds to group raw
    platec plates while preserving their natural tectonic boundaries.  The
    downstream renderer draws those raw boundaries as outlines."""
    from collections import deque

    plates = world.layers["plates"].data.astype(int)
    ocean = world.layers["ocean"].data
    n_raw = int(plates.max()) + 1
    k = n_ocean + n_land
    if n_raw <= k:
        return plates

    # Raw-plate adjacency
    adj = {p: set() for p in range(n_raw)}
    h, w = plates.shape
    for y in range(h):
        row = plates[y]
        for x in range(w):
            p = int(row[x])
            if x + 1 < w:
                q = int(plates[y, x + 1])
                if q != p:
                    adj[p].add(q)
                    adj[q].add(p)
            if y + 1 < h:
                q = int(plates[y + 1, x])
                if q != p:
                    adj[p].add(q)
                    adj[q].add(p)

    # Ocean ratio per raw plate
    sea_ratio = []
    for p in range(n_raw):
        mask = plates == p
        total = int(mask.sum())
        sea_ratio.append(float(ocean[mask].sum()) / total if total else 0.0)

    sea_order = sorted(range(n_raw), key=lambda p: sea_ratio[p], reverse=True)
    land_order = sorted(range(n_raw), key=lambda p: sea_ratio[p])

    ocean_seeds = sea_order[:n_ocean]
    land_seeds = []
    for p in land_order:
        if p not in ocean_seeds and len(land_seeds) < n_land:
            land_seeds.append(p)
    seeds = ocean_seeds + land_seeds
    group_of = {s: gi for gi, s in enumerate(seeds)}

    assigned = dict(group_of)
    q = deque(seeds)
    while q:
        p = q.popleft()
        g = assigned[p]
        for nxt in adj[p]:
            if nxt not in assigned:
                assigned[nxt] = g
                q.append(nxt)

    merged = numpy.empty_like(plates)
    for p, g in assigned.items():
        merged[plates == p] = g
    merged = _remove_enclaves(merged, min_frac=0.0005)
    return merged


# blue shades for oceanic plates, green/brown for continental ones
MERGED_PALETTE = numpy.array(
    [
        (20, 70, 190), (70, 150, 235),   # 2 oceanic plates
        (90, 170, 80), (200, 180, 90),   # 4 continental plates
        (180, 140, 80), (140, 190, 120),
    ],
    dtype=float,
)


def save_merged_plates(world, plates, name):
    from PIL import Image
    path = os.path.join(OUT, name)
    n_plates = int(plates.max()) + 1
    palette = numpy.resize(MERGED_PALETTE, (n_plates, 3))

    h, w = plates.shape
    big_h, big_w = h * 2, w * 2

    # Upscale labels with NEAREST (preserves label values)
    labels_img = Image.fromarray(plates.astype(numpy.uint8))
    labels_big = numpy.asarray(labels_img.resize((big_w, big_h), Image.NEAREST))
    rgb_big = palette[labels_big.astype(int)].astype(numpy.uint8).copy()

    # Draw group boundaries at 2x
    boundary = numpy.zeros((big_h, big_w), dtype=bool)
    boundary[:, 1:] |= labels_big[:, 1:] != labels_big[:, :-1]
    boundary[1:, :] |= labels_big[1:, :] != labels_big[:-1, :]
    rgb_big[boundary] = [0, 0, 0]

    # Downsample with BILINEAR → anti-aliases the stair-step boundaries
    img_big = Image.fromarray(rgb_big)
    img_small = img_big.resize((w, h), Image.BILINEAR)
    rgb = numpy.asarray(img_small)

    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)

    ocean = world.layers["ocean"].data
    for g in range(n_plates):
        mask = plates == g
        area = mask.sum()
        sea = ocean[mask].sum() / area if area else 0.0
        kind = "oceanic" if g < 2 else "continental"
        print(f"  plate {g} ({kind}): area {area} px, ocean ratio {sea:.2f}")


def save_ocean_mask(world, name):
    path = os.path.join(OUT, name)
    ocean = world.layers["ocean"].data
    rgb = numpy.zeros((H, W, 3), dtype=numpy.uint8)
    rgb[:] = (95, 160, 90)
    rgb[ocean] = (70, 130, 210)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def save_rivers(world, name):
    path = os.path.join(OUT, name)
    ocean = world.layers["ocean"].data
    river = world.layers["river_map"].data
    lake = world.layers["lake_map"].data
    rgb = numpy.zeros((H, W, 3), dtype=numpy.uint8)
    rgb[:] = (185, 205, 150)
    rgb[ocean] = (80, 135, 205)
    rgb[lake != 0] = (50, 130, 205)
    rgb[river > 0] = (10, 55, 180)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def save_biome(world, name):
    path = os.path.join(OUT, name)
    bm = world.layers["biome"].data
    h, w = bm.shape
    rgb = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    for name_, color in _biome_colors.items():
        rgb[bm == name_] = color[:3]
    rgb[bm == "bare rock"] = (128, 128, 128)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def main():
    ensure_out()

    step = Step.full()
    world = _plates_simulation("stages", W, H, SEED, num_plates=NUM_PLATES, step=step)

    # 1. raw plates simulation output
    save_color(world.layers["elevation"].data, "01_plates_raw_elevation.png", TERRAIN)
    save_plates(world.layers["plates"].data, "02_plates_map.png")

    # 2. land centered
    center_land(world)
    save_color(world.layers["elevation"].data, "03_center_land_elevation.png", TERRAIN)

    # 3. noise added
    add_noise_to_elevation(world, numpy.random.randint(0, 4096))
    save_color(world.layers["elevation"].data, "04_noise_elevation.png", TERRAIN)

    # 4. ocean borders + ocean/thresholds init
    place_oceans_at_map_borders(world)
    save_color(world.layers["elevation"].data, "05_ocean_borders_elevation.png", TERRAIN)
    initialize_ocean_and_thresholds(world)
    save_color(world.layers["elevation"].data, "06_ocean_initialized_elevation.png", TERRAIN)
    save_ocean_mask(world, "07_ocean_mask.png")
    save_color(world.layers["sea_depth"].data, "08_sea_depth.png", SEA_DEPTH)

    # 4b. merge into 2 oceanic + 4 continental plates
    merged = merge_plates(world)
    save_merged_plates(world, merged, "02b_plates_merged.png")

    rng = numpy.random.RandomState(SEED)
    sub_seeds = rng.randint(0, numpy.iinfo(numpy.int32).max, size=100)

    # 5. temperature
    TemperatureSimulation().execute(world, sub_seeds[4])
    save_color(world.layers["temperature"].data, "09_temperature.png", TEMPERATURE)

    # 6. precipitation
    PrecipitationSimulation().execute(world, sub_seeds[0])
    save_color(world.layers["precipitation"].data, "10_precipitation.png", PRECIPITATION)

    # 7. erosion -> rivers + lakes
    ErosionSimulation().execute(world, sub_seeds[1])
    save_rivers(world, "11_rivermap.png")
    save_rivers(world, "12_lakemap.png")
    save_color(world.layers["elevation"].data, "13_eroded_elevation.png", TERRAIN)

    # 8. watermap
    WatermapSimulation().execute(world, sub_seeds[2])
    save_color(world.layers["watermap"].data, "14_watermap.png", WATERMAP)

    # 9. irrigation
    IrrigationSimulation().execute(world, sub_seeds[3])
    save_color(world.layers["irrigation"].data, "15_irrigation.png", PRECIPITATION)

    # 10. humidity
    HumiditySimulation().execute(world, sub_seeds[5])
    save_color(world.layers["humidity"].data, "16_humidity.png", HUMIDITY)

    # 11. permeability
    PermeabilitySimulation().execute(world, sub_seeds[6])
    save_color(world.layers["permeability"].data, "17_permeability.png", PERMEABILITY)

    # 12. biome
    BiomeSimulation().execute(world, sub_seeds[7])
    save_biome(world, "18_biome.png")

    # 13. icecap
    IcecapSimulation().execute(world, sub_seeds[8])
    save_color(world.layers["icecap"].data, "19_icecap.png", ICECAP)

    print("all stages done")


if __name__ == "__main__":
    main()
