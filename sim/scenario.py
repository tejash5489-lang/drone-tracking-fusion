"""
Ground-truth scenario generator — Phase 1 of the blueprint.

Produces one or more 3D, linear-Gaussian, nearly-constant-velocity (NCV)
target trajectories, matching blueprint section 2.4 ("Ground truth: linear
Gaussian nearly-constant-velocity (NCV) motion model, 3-D (x, y, z)").

State vector convention used throughout this project: [x, vx, y, vy, z, vz]
(position_mapping = (0, 2, 4)), so it lines up directly with the radar
sensors' position_mapping in sim/sensors.py.
"""
from datetime import datetime, timedelta

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity,
)

# Process noise (velocity diffusion) per axis — modest values so tracks stay
# close to straight lines, consistent with a "nearly" constant velocity model.
DEFAULT_PROCESS_NOISE = (0.05, 0.05, 0.05)


def build_transition_model(process_noise=DEFAULT_PROCESS_NOISE):
    """3D NCV transition model: independent constant-velocity models on x, y, z."""
    qx, qy, qz = process_noise
    return CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(qx), ConstantVelocity(qy), ConstantVelocity(qz)]
    )


def generate_ground_truth(
    initial_state,
    num_steps,
    transition_model=None,
    start_time=None,
    time_step=timedelta(seconds=1),
    seed=None,
):
    """Simulate one target's 3D NCV ground-truth path.

    Parameters
    ----------
    initial_state : array-like, length 6
        [x, vx, y, vy, z, vz] at t=0.
    num_steps : int
        Number of time steps to simulate after t=0.
    transition_model : TransitionModel, optional
        Defaults to :func:`build_transition_model`.
    start_time : datetime, optional
        Defaults to ``datetime.now()``.
    seed : int, optional
        Seeds numpy's global RNG for reproducibility (Stone Soup's Gaussian
        noise sampling draws from ``numpy.random``).

    Returns
    -------
    GroundTruthPath
    """
    import numpy as np
    if seed is not None:
        np.random.seed(seed)

    transition_model = transition_model or build_transition_model()
    start_time = start_time or datetime.now()

    truth = GroundTruthPath([GroundTruthState(initial_state, timestamp=start_time)])
    for k in range(1, num_steps + 1):
        truth.append(GroundTruthState(
            transition_model.function(truth[-1], noise=True, time_interval=time_step),
            timestamp=start_time + k * time_step,
        ))
    return truth


def generate_scenario(target_initial_states, num_steps, **kwargs):
    """Simulate several targets at once (multi-target scenario).

    Parameters
    ----------
    target_initial_states : list of array-like
        One [x, vx, y, vy, z, vz] per target.
    num_steps : int
    kwargs : forwarded to :func:`generate_ground_truth` (shared across targets).

    Returns
    -------
    set of GroundTruthPath
    """
    return {
        generate_ground_truth(state, num_steps, **kwargs)
        for state in target_initial_states
    }
