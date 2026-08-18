"""
OSPA + SIAP evaluation — Phase 1 of the blueprint (section 2.4):

    "evaluated with SIAP ambiguity, completeness, positional accuracy,
    spuriousness, and OSPA distance (average +/- std. dev., plus
    Friedman-test p-values for significance)."

Stone Soup ships both metric generators directly
(metricgenerator.ospametric.OSPAMetric, metricgenerator.tracktotruthmetrics.SIAPMetrics)
so, like the JPDA tracker, this is wiring rather than new algorithm code.
The Friedman test across methods/scenarios belongs in Phase 6 (needs many
runs to compare) — this module only computes the per-run metrics that
Phase 6 will later aggregate.
"""
import numpy as np

from stonesoup.dataassociator.tracktotrack import TrackToTruth
from stonesoup.measures import Euclidean
from stonesoup.metricgenerator.manager import SimpleManager
from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics

POSITION_MAPPING = (0, 2, 4)
VELOCITY_MAPPING = (1, 3, 5)


def build_metric_manager(
    ospa_cutoff=500.0,
    ospa_p=2,
    association_threshold=100.0,
):
    """Wire up OSPA + SIAP generators behind a SimpleManager.

    Parameters
    ----------
    ospa_cutoff : float
        OSPA's "c" — max per-point distance penalty (caps the cost of a
        badly-missed target so one outlier doesn't dominate the metric).
    ospa_p : float
        OSPA's order "p" (2 = Euclidean-style penalty, standard choice).
    association_threshold : float
        Max distance (m) for TrackToTruth to consider a track associated
        with a ground truth, used by the SIAP metrics.
    """
    position_measure = Euclidean(mapping=POSITION_MAPPING)
    velocity_measure = Euclidean(mapping=VELOCITY_MAPPING)

    ospa_generator = OSPAMetric(c=ospa_cutoff, p=ospa_p)
    siap_generator = SIAPMetrics(
        position_measure=position_measure,
        velocity_measure=velocity_measure,
    )
    associator = TrackToTruth(association_threshold=association_threshold)

    return SimpleManager(
        generators=[ospa_generator, siap_generator],
        associator=associator,
    )


def compute_metrics(manager, tracks, ground_truth_paths):
    """Run OSPA + SIAP for one set of (tracks, ground truth) and return a
    flat dict of scalar summary values (mean over the run's time steps,
    where the metric is a time series).
    """
    manager.add_data(groundtruth_paths=ground_truth_paths, tracks=tracks, overwrite=True)
    raw_metrics = manager.generate_metrics()
    return {name: _summarise(metric) for name, metric in raw_metrics.items()}


def _summarise(metric):
    """Reduce a Stone Soup Metric (SingleTimeMetric or TimeRangeMetric-of-
    SingleTimeMetric) down to a scalar mean, for easy reporting/aggregation.
    """
    value = metric.value
    if isinstance(value, list):
        scalars = [_summarise(v) if hasattr(v, "value") else v for v in value]
        scalars = [float(s) for s in scalars]
        return float(np.mean(scalars)) if scalars else float("nan")
    return float(value)
