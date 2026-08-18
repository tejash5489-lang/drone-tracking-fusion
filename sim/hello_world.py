"""
Phase 0 smoke test: one target, one radar sensor, one Kalman-filter tracker,
plotted against ground truth. Confirms the Stone Soup + SciPy + PyTorch stack
installed in venv/ actually works end to end before any of the fusion
approaches (Phases 1-5) are built on top of it.

Run: venv\\Scripts\\python.exe sim\\hello_world.py
"""
import numpy as np
from datetime import datetime, timedelta

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.detection import Detection
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity,
)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.plotter import Plotter

np.random.seed(1991)
start_time = datetime.now()

# --- Ground truth: constant-velocity motion in 2D (x, vx, y, vy) ---
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)]
)

truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
num_steps = 20
for k in range(1, num_steps + 1):
    truth.append(GroundTruthState(
        transition_model.function(truth[-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k),
    ))

# --- Sensor: linear-Gaussian measurement of (x, y) only ---
measurement_model = LinearGaussian(
    ndim_state=4, mapping=(0, 2), noise_covar=np.diag([1.0, 1.0])
)
detections = [
    Detection(
        measurement_model.function(state, noise=True),
        timestamp=state.timestamp,
        measurement_model=measurement_model,
    )
    for state in truth
]

# --- Tracker: Kalman predictor/updater (no data association needed, 1 target) ---
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
track = Track([prior])
for detection in detections:
    prediction = predictor.predict(track[-1], timestamp=detection.timestamp)
    hypothesis_state = updater.predict_measurement(prediction)
    from stonesoup.types.hypothesis import SingleHypothesis
    hypothesis = SingleHypothesis(prediction, detection)
    post = updater.update(hypothesis)
    track.append(post)

# --- Plot ground truth vs. tracked estimate ---
plotter = Plotter()
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(detections, [0, 2])
plotter.plot_tracks(track, [0, 2])
plotter.fig.savefig("data/hello_world_smoke_test.png")
print(f"OK — {len(track)} track states produced from {len(detections)} detections.")
print("Plot saved to data/hello_world_smoke_test.png")
