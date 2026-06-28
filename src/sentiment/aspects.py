"""The fixed aspect list.

Runs in the ABSA env only. The frozen set of aspects for Video Games
(e.g. gameplay, graphics, price, difficulty), fixed after eyeballing PyABSA
output on ~300 reviews so the sentiment table stays comparable across runs.
"""

# Fixed aspect list (locked after step 5 sanity check).
ASPECTS: list[str] = [
    "gameplay",
    "graphics",
    "price",
    "difficulty",
]
