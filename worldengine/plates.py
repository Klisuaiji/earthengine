"""Tectonic plate generation.

Historically this module wrapped ``platec``, a C extension running a physical
plate-tectonics simulation.  That dependency has been removed: plates are now
produced analytically by :mod:`worldengine.spherical_voronoi` (domain-warped
spherical Voronoi + area-balanced merging), which is pure Python/NumPy/SciPy.

The public API is unchanged.  ``generate_plates_simulation`` still returns
``(heightmap, platesmap)`` on the historic platec value scale - elevations are
positive floats where ``1.0`` is sea level, ``~0.07`` the abyssal floor and
``~9.0`` the highest peaks - so every downstream consumer (the CLI, the
simulations, saved ``.world`` files) keeps working untouched.
"""

import time

import numpy

from worldengine.generation import (
    Step,
    add_noise_to_elevation,
    center_land,
    generate_world,
    get_verbose,
    initialize_ocean_and_thresholds,
    place_oceans_at_map_borders,
)
from worldengine.model.world import GenerationParameters, Size, World
from worldengine.spherical_voronoi import generate_spherical_plates, synthesize_elevation

# Historic platec heightmap conventions, preserved for backwards compatibility.
_HM_ABYSS = 0.07
_HM_SEA = 1.0
_HM_PEAK = 9.0

# Every raw micro-plate needs a reasonable pixel budget, otherwise the Voronoi
# partition degenerates on tiny maps (the unit tests use 32x16).
_MIN_PIXELS_PER_PLATE = 64
# spherical_voronoi._place_seeds reserves indices 4 and 5 for the oceanic seeds.
_MIN_RAW_PLATES = 6


def _plate_counts(width, height, num_plates):
    """Derive (n_raw, n_big) micro/major plate counts that fit the map size."""
    budget = max(_MIN_RAW_PLATES, (width * height) // _MIN_PIXELS_PER_PLATE)
    n_raw = min(max(30, 3 * int(num_plates)), budget)
    n_raw = max(_MIN_RAW_PLATES, n_raw)
    n_big = max(3, min(int(num_plates), n_raw - 1))
    return n_raw, n_big


def _to_platec_scale(elevation, sea_level):
    """Piecewise-linear remap of raw metres onto the platec heightmap scale.

    ``sea_level`` is the fraction of the map that should end up at or below
    sea level, matching the meaning of platec's own ``sea_level`` argument.
    The mapping pins three anchors - min -> 0.07, the sea-level quantile ->
    1.0, max -> 9.0 - so it is monotonic and never produces negative values
    (``place_oceans_at_map_borders`` multiplies elevations towards zero and
    relies on that).
    """
    e = numpy.asarray(elevation, dtype=numpy.float64)
    lo, hi = float(e.min()), float(e.max())
    if hi - lo < 1e-9:
        return numpy.full(e.shape, _HM_SEA, dtype=numpy.float64)

    sea_level = min(max(float(sea_level), 0.01), 0.99)
    sea = float(numpy.quantile(e, sea_level))
    eps = (hi - lo) * 1e-6
    sea = min(max(sea, lo + eps), hi - eps)

    out = numpy.empty_like(e)
    below = e <= sea
    out[below] = _HM_ABYSS + (e[below] - lo) * (_HM_SEA - _HM_ABYSS) / (sea - lo)
    out[~below] = _HM_SEA + (e[~below] - sea) * (_HM_PEAK - _HM_SEA) / (hi - sea)
    return out


def generate_plates_simulation(
    seed,
    width,
    height,
    sea_level=0.65,
    erosion_period=60,
    folding_ratio=0.02,
    aggr_overlap_abs=1000000,
    aggr_overlap_rel=0.33,
    cycle_count=2,
    num_plates=10,
    verbose=get_verbose(),
):
    """Generate a heightmap and a plate-id map.

    ``erosion_period``, ``folding_ratio``, ``aggr_overlap_abs``,
    ``aggr_overlap_rel`` and ``cycle_count`` were parameters of the physical
    platec simulation.  They are accepted (so existing callers and saved
    command lines keep working) but ignored by the Voronoi generator.

    :return: ``(heightmap, platesmap)``, both ``(height, width)`` arrays.
    """
    if verbose:
        start_time = time.time()

    n_raw, n_big = _plate_counts(width, height, num_plates)
    raw, merged = generate_spherical_plates(seed, w=width, h=height, n_raw=n_raw, n_big=n_big)
    elevation = synthesize_elevation(merged, height, width, seed, n_big=n_big)
    hm = _to_platec_scale(elevation, sea_level)
    pm = raw.astype(numpy.uint16)

    if verbose:
        elapsed_time = time.time() - start_time
        print(
            "...plates.generate_plates_simulation() complete. "
            f"{n_raw} micro-plates -> {n_big} major plates. "
            f"Elapsed time {elapsed_time} seconds."
        )
    return hm, pm


def _plates_simulation(
    name,
    width,
    height,
    seed,
    temps=[0.874, 0.765, 0.594, 0.439, 0.366, 0.124],
    humids=[0.941, 0.778, 0.507, 0.236, 0.073, 0.014, 0.002],
    gamma_curve=1.25,
    curve_offset=0.2,
    num_plates=10,
    ocean_level=1.0,
    step=Step.full(),
    verbose=get_verbose(),
):
    e_as_array, p_as_array = generate_plates_simulation(seed, width, height, num_plates=num_plates, verbose=verbose)

    world = World(
        name,
        Size(width, height),
        seed,
        GenerationParameters(num_plates, ocean_level, step),
        temps,
        humids,
        gamma_curve,
        curve_offset,
    )
    world.elevation = (numpy.array(e_as_array).reshape(height, width), None)
    world.plates = numpy.array(p_as_array, dtype=numpy.uint16).reshape(height, width)
    return world


def world_gen(
    name,
    width,
    height,
    seed,
    temps=[0.874, 0.765, 0.594, 0.439, 0.366, 0.124],
    humids=[0.941, 0.778, 0.507, 0.236, 0.073, 0.014, 0.002],
    num_plates=10,
    ocean_level=1.0,
    step=Step.full(),
    gamma_curve=1.25,
    curve_offset=0.2,
    fade_borders=True,
    verbose=get_verbose(),
):
    if verbose:
        start_time = time.time()
    world = _plates_simulation(
        name, width, height, seed, temps, humids, gamma_curve, curve_offset, num_plates, ocean_level, step, verbose
    )

    center_land(world)
    if verbose:
        elapsed_time = time.time() - start_time
        print(
            "...plates.world_gen: set_elevation, set_plates, center_land "
            + "complete. Elapsed time "
            + str(elapsed_time)
            + " seconds."
        )

    if verbose:
        start_time = time.time()
    add_noise_to_elevation(
        world, numpy.random.randint(0, 4096)
    )  # uses the global RNG; this is the very first call to said RNG - should that change, this needs to be taken care of
    if verbose:
        elapsed_time = time.time() - start_time
        print("...plates.world_gen: elevation noise added. Elapsed time " + str(elapsed_time) + " seconds.")

    if verbose:
        start_time = time.time()
    if fade_borders:
        place_oceans_at_map_borders(world)
    initialize_ocean_and_thresholds(world)
    if verbose:
        elapsed_time = time.time() - start_time
        print("...plates.world_gen: oceans initialized. Elapsed time " + str(elapsed_time) + " seconds.")

    return generate_world(world, step)
