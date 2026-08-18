"""
Two-radar sensor setup — Phase 1 of the blueprint (section 2.4):

    "Sensors: two radar sensors (state dimension 6; measures elevation,
    bearing, range), one airborne/moving, one ground-based; assumed mutually
    blind (no shared field of view); max range 3000; clutter modeled with a
    specified clutter rate and uniform spatial distribution."

Both sensors use Stone Soup's RadarElevationBearingRange, which measures
(elevation, bearing, range) relative to the sensor and is exactly the model
the paper's radars use. "Mutually blind" just means we never fuse raw
detections across sensors — only their independent tracks (via Covariance
Intersection downstream), so it needs no special code here: each sensor only
ever sees the shared ground-truth targets, never the other sensor.
"""
import numpy as np
from datetime import datetime

from stonesoup.types.state import State
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import TrueDetection
from stonesoup.movable.movable import MovingMovable
from stonesoup.sensor.radar.radar import RadarElevationBearingRange
from stonesoup.models.clutter.clutter import ClutterModel

from sim.scenario import build_transition_model

MAX_RANGE = 3000.0
POSITION_MAPPING = (0, 2, 4)  # x, y, z within the 6D [x,vx,y,vy,z,vz] state

# Elevation (rad), bearing (rad), range (m) noise std devs -> covariance.
# Not specified numerically in the paper; these are a reasonable radar-like
# default (~0.3 deg angular noise, 15 m range noise) — revisit per scenario.
DEFAULT_NOISE_COVAR = CovarianceMatrix(np.diag([
    np.deg2rad(0.3) ** 2,
    np.deg2rad(0.3) ** 2,
    15.0 ** 2,
]))


def default_clutter_model(clutter_rate, world_extent):
    """Uniform clutter over a Cartesian box, per blueprint's "uniform spatial
    distribution" requirement.

    Parameters
    ----------
    clutter_rate : float
        Mean number of clutter detections per time step (Poisson).
    world_extent : tuple of (min, max) pairs, length 3
        Cartesian (x, y, z) box that clutter is generated within, e.g.
        ((-3000, 3000), (-3000, 3000), (0, 2000)).
    """
    return ClutterModel(clutter_rate=clutter_rate, dist_params=world_extent)


def build_ground_radar(
    position,
    noise_covar=DEFAULT_NOISE_COVAR,
    clutter_rate=1.0,
    world_extent=((-3000, 3000), (-3000, 3000), (0, 2000)),
    max_range=MAX_RANGE,
):
    """Fixed, ground-based radar at a stationary Cartesian ``position`` (x, y, z)."""
    return RadarElevationBearingRange(
        position=StateVector(position),
        ndim_state=6,
        position_mapping=POSITION_MAPPING,
        noise_covar=noise_covar,
        max_range=max_range,
        clutter_model=default_clutter_model(clutter_rate, world_extent),
    )


def build_airborne_radar(
    initial_state,
    noise_covar=DEFAULT_NOISE_COVAR,
    clutter_rate=1.0,
    world_extent=((-3000, 3000), (-3000, 3000), (0, 2000)),
    max_range=MAX_RANGE,
    start_time=None,
):
    """Moving, airborne radar. ``initial_state`` is [x, vx, y, vy, z, vz];
    the platform follows its own NCV motion (independent of any target),
    advanced by calling :func:`advance_moving_sensors` once per time step.
    """
    start_time = start_time or datetime.now()
    movement_controller = MovingMovable(
        states=[State(StateVector(initial_state), timestamp=start_time)],
        transition_model=build_transition_model(),
        position_mapping=POSITION_MAPPING,
    )
    return RadarElevationBearingRange(
        movement_controller=movement_controller,
        ndim_state=6,
        position_mapping=POSITION_MAPPING,
        noise_covar=noise_covar,
        max_range=max_range,
        clutter_model=default_clutter_model(clutter_rate, world_extent),
    )


def advance_moving_sensors(sensors, timestamp):
    """Propagate every sensor with its own ``movement_controller`` (i.e. the
    airborne radar) to ``timestamp``. Call once per simulation time step,
    before ``sensor.measure(...)``. Fixed (ground) sensors are unaffected.
    """
    for sensor in sensors:
        controller = sensor.movement_controller
        if isinstance(controller, MovingMovable):
            controller.move(timestamp)


def apply_measurement_bias(detections, bias):
    """Add a constant offset to every *real* detection's measurement vector
    (elevation, bearing, range), leaving clutter untouched — a persistent
    sensor miscalibration, as opposed to clutter/geometry, which only ever
    affect a track's *availability*, not its *accuracy* (see
    fusion/kl_objective.py's Phase 6 update for why that distinction turned
    out to matter for Approach 1's KL-omega labels).

    Parameters
    ----------
    detections : set of Detection
        As returned by ``sensor.measure(...)``.
    bias : array-like, shape (3,), or None
        [elevation, bearing, range] offset, in the sensor's own measurement
        units (radians, radians, metres). None or all-zero is a no-op.

    Returns
    -------
    set of Detection
    """
    if bias is None or not np.any(np.asarray(bias, dtype=float)):
        return detections

    bias_vector = np.asarray(bias, dtype=float).reshape(-1, 1)
    biased = set()
    for detection in detections:
        if isinstance(detection, TrueDetection):
            biased.add(TrueDetection(
                StateVector(detection.state_vector + bias_vector),
                measurement_model=detection.measurement_model,
                timestamp=detection.timestamp,
                groundtruth_path=detection.groundtruth_path,
            ))
        else:
            biased.add(detection)
    return biased
