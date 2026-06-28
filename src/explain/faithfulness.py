"""Remove-a-reason perturbation check.

Faithfulness test: remove a reason (an aspect/edge the explanation cites) and
check whether the recommendation / explanation changes as it should. The core
honest-vs-fluent comparison leans on this.
"""


def check(*args, **kwargs):
    raise NotImplementedError
