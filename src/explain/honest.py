"""Graph-grounded explanation.

The honest condition: only rephrases facts already extracted from the graph
(aspect→sentiment, trust). No new content is invented. Kept apart from
ungrounded.py so study conditions never bleed together.
"""


def explain(*args, **kwargs):
    raise NotImplementedError
