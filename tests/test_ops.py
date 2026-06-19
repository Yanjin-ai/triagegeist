"""P6 guards — bucket validity + priority monotonicity + senior-review logic."""
import numpy as np
import pandas as pd

from src.ops.prioritize import needs_senior_review, priority_score, rank_queue
from src.ops.resource_map import BUCKETS, assign_bucket


def test_buckets_are_valid_and_acuity_ordered():
    acuity = np.array([1, 2, 3, 4, 5])
    b = assign_bucket(acuity, p_admit=np.zeros(5), pred_los=np.full(5, 3.0))
    assert set(b).issubset(set(BUCKETS))
    assert b.iloc[0] == BUCKETS[0] and b.iloc[4] == BUCKETS[4]


def test_priority_decreases_with_acuity():
    # holding everything else fixed, a more urgent (lower) acuity must score higher
    base = dict(p_critical=0.2, entropy=0.5, wait_hours=1.0)
    scores = [priority_score([a], [base["p_critical"]], [base["entropy"]],
                             [base["wait_hours"]])[0] for a in (1, 3, 5)]
    assert scores[0] > scores[1] > scores[2]


def test_senior_review_flags_uncertain_critical():
    flag = needs_senior_review(acuity=[2, 5], p_critical=[0.6, 0.0],
                               entropy=[1.5, 0.01], ent_hi=1.0)
    assert flag[0] and not flag[1]


def test_rank_queue_orders_by_priority():
    df = pd.DataFrame({"acuity": [5, 1, 3], "p_critical": [0.0, 0.9, 0.3],
                       "entropy": [0.1, 0.2, 0.5], "wait_hours": [0.0, 0.0, 0.0]})
    rq = rank_queue(df)
    assert rq["priority"].is_monotonic_decreasing
    assert rq.iloc[0]["acuity"] == 1
