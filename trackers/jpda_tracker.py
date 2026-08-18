"""
Per-sensor JPDA tracker — Phase 1 of the blueprint (sensor 1 / airborne radar):

    "Trackers compared per sensor: JPDA (airborne radar) and GM-LCC
    (ground radar)"

Stone Soup ships JPDA directly (dataassociator.probability.JPDA +
hypothesiser.probability.PDAHypothesiser), so this is largely wiring rather
than new algorithm code. Because the radar measurement (elevation, bearing,
range) is nonlinear in Cartesian state, prediction/update use the Extended
Kalman Filter, matching blueprint 2.2's "EKF ... used to predict and update
each individual sensor's track".

Driven step-by-step (rather than via Stone Soup's generator-based
MultiTargetTracker) so the caller stays in control of the simulation loop —
same loop that advances the moving sensor and generates ground truth.
"""
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.probability import PDAUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.types.state import GaussianState


class JPDATracker:
    """Wraps Stone Soup's JPDA components behind a simple ``step`` API.

    Parameters
    ----------
    transition_model : TransitionModel
        Same NCV model used to generate ground truth (sim.scenario).
    prior_state : GaussianState
        Vague prior used to initiate new tracks from unassociated detections.
    prob_detect : float
        Target detection probability (blueprint doesn't specify a value;
        0.9 is a standard JPDA-tutorial default).
    covar_trace_thresh : float
        Track deleted once its covariance trace exceeds this (diverged /
        no longer supported by detections).
    """

    def __init__(
        self,
        transition_model,
        prior_state=None,
        prob_detect=0.9,
        covar_trace_thresh=5000.0,
    ):
        self.predictor = ExtendedKalmanPredictor(transition_model)
        # PDAUpdater (not plain ExtendedKalmanUpdater) because JPDA returns a
        # MultipleHypothesis (weighted mixture over detections) per track, not
        # a single hypothesis — PDAUpdater knows how to combine that mixture.
        # measurement_model is left unset here (not fixed at construction):
        # for a moving (airborne) sensor its measurement_model's position
        # offset changes every step, so it must be supplied fresh to update()
        # each time rather than captured once — see step() below.
        self.updater = PDAUpdater(measurement_model=None)
        self.hypothesiser = PDAHypothesiser(
            predictor=self.predictor,
            updater=self.updater,
            clutter_spatial_density=1e-8,
            prob_detect=prob_detect,
        )
        self.associator = JPDA(hypothesiser=self.hypothesiser)

        prior_state = prior_state or GaussianState(
            [[0], [0], [0], [0], [0], [0]],
            [[1e6, 0, 0, 0, 0, 0],
             [0, 1e4, 0, 0, 0, 0],
             [0, 0, 1e6, 0, 0, 0],
             [0, 0, 0, 1e4, 0, 0],
             [0, 0, 0, 0, 1e6, 0],
             [0, 0, 0, 0, 0, 1e4]],
        )
        self.initiator = SimpleMeasurementInitiator(prior_state=prior_state)
        # mapping=(0,2,4) restricts the trace check to position variance only:
        # the prior's velocity variance (1e4) alone would exceed most sensible
        # thresholds and delete every brand-new track before it gets a second
        # update, since velocity is unobservable from a single detection.
        self.deleter = CovarianceBasedDeleter(
            covar_trace_thresh=covar_trace_thresh, mapping=[0, 2, 4])

        self.tracks = set()

    def step(self, detections, timestamp, measurement_model):
        """Advance all tracks by one time step given this step's detections.

        Parameters
        ----------
        detections : set of Detection
        timestamp : datetime.datetime
        measurement_model : MeasurementModel
            The sensor's *current* measurement model, i.e. ``sensor.measurement_model``
            evaluated fresh this step (for a moving sensor this carries this
            step's position, not a stale one) — needed for the "missed
            detection" hypothesis, which has no real detection to draw a
            model from.

        Returns the current ``self.tracks`` (also updated in place).
        """
        if self.tracks:
            hypotheses = self.associator.associate(self.tracks, detections, timestamp)
            associated_detections = set()
            for track, multi_hypothesis in hypotheses.items():
                track.append(self.updater.update(
                    multi_hypothesis, measurement_model=measurement_model))
                associated_detections |= {
                    hyp.measurement for hyp in multi_hypothesis if hyp.measurement
                }
        else:
            associated_detections = set()

        self.tracks -= self.deleter.delete_tracks(self.tracks)
        self.tracks |= self.initiator.initiate(detections - associated_detections, timestamp)
        return self.tracks
