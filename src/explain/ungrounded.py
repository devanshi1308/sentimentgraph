"""Fluent-but-ungrounded (study only).

The ungrounded condition is a study variable, not a product feature. It is never
wired into the live recommender. Kept in a separate file from honest.py so the
conditions never bleed together.
"""


def explain(*args, **kwargs):
    raise NotImplementedError
