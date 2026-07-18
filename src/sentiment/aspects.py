"""Canonical aspect vocabulary + synonym map for the extract-all-normalize-after policy.

EXTENDS the PRD step-5 fixed list (gameplay, graphics, price, difficulty) under the
"extract all, normalize after" policy decided by the project lead (see fixes.md #6).
The four PRD-locked aspects are preserved verbatim; up to six additional video-game-domain
canonicals have been added based on aspects that raw PyABSA output plausibly produces.

Matching is case-insensitive substring: for each raw aspect (lowercased and stripped),
we check whether any variant string appears as a substring of that raw term. The FIRST
canonical whose any-variant matches wins (dictionary insertion order = tie-break). The
canonical vocabulary should be reviewed by the project lead after the first full ABSA
run; unmatched terms are surfaced by normalize_aspects.py's coverage report to inform
future curation.

This file runs in the ABSA env but is also safe to import from the rec env —
no pyabsa or recbole imports here.
"""

# CANONICAL_ASPECTS maps canonical aspect name -> list of lowercase raw-variant substrings
# that identify it. Variants are checked as substrings of the lowercased raw aspect term
# emitted by PyABSA. Order within each list does not affect matching (any match wins).
# Order of canonicals in this dict IS significant: the first canonical matched is used
# when a raw term contains variants from multiple canonicals (tie-break by insertion order).
#
# PRD-locked four (gameplay, graphics, price, difficulty) come first to preserve priority.
# Additional canonicals follow in rough frequency-of-occurrence order in game reviews.

CANONICAL_ASPECTS: dict[str, list[str]] = {
    # ---- PRD-locked four ----
    "gameplay": [
        "gameplay",
        "game mechanic",
        "mechanic",
        "game play",
        "playability",
        "game loop",
    ],
    "graphics": [
        "graphic",
        "visual",
        "art style",
        "art direction",
        "texture",
        "resolution",
        "render",
        "animation",
        "look",        # "look and feel", "looks great" — short; safe as last resort
    ],
    "price": [
        "price",
        "cost",
        "value for",   # "value for money" — intentionally more specific than "value"
        "worth the",   # "worth the money/price"
        "overpriced",
        "expensive",
        "cheap",
        "affordable",
        "sale",
        "discount",
        "$",
    ],
    "difficulty": [
        "difficul",    # difficulty, difficult
        "challeng",    # challenge, challenging
        "hard",
        "easy",
        "punishing",
        "brutal",
        "forgiving",
        "learning curve",
        "steep curve",
    ],
    # ---- Additional video-game-domain canonicals ----
    "story": [
        "story",
        "narrative",
        "plot",
        "lore",
        "writing",
        "dialogue",
        "character",
        "ending",
        "cutscene",
        "quest",
    ],
    "controls": [
        "control",
        "input",
        "button",
        "keybind",
        "key binding",
        "controller",
        "mouse",
        "keyboard",
        "responsiv",   # responsive, responsiveness
        "handling",
    ],
    "sound": [
        "sound",
        "music",
        "audio",
        "soundtrack",
        "voice acting",
        "voice act",
        "sfx",
        "ambient",
        "score",
    ],
    "multiplayer": [
        "multiplayer",
        "online",
        "coop",
        "co-op",
        "pvp",
        "matchmaking",
        "server",
        "lobby",
        "netcode",
        "lag",         # lag can appear under performance too; multiplayer wins by order
    ],
    "performance": [
        "performance",
        "fps",
        "frame rate",
        "framerate",
        "stutter",
        "crash",
        "bug",
        "glitch",
        "optimization",
        "optimisation",
        "loading",
        "load time",
    ],
    "value": [
        "value",       # standalone "value" (not "value for" — that hits price first)
        "worth it",
        "worth buying",
        "bang for",
        "deal",
        "content",     # "amount of content", "content for the price"
    ],
}
