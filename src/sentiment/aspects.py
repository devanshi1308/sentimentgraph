"""Canonical aspect vocabulary + synonym map for the extract-all-normalize-after policy.

EXTENDS the PRD step-5 fixed list (gameplay, graphics, price, difficulty) under the
"extract all, normalize after" policy decided by the project lead (see fixes.md #6).
The four PRD-locked aspects are preserved verbatim; up to six additional video-game-domain
canonicals have been added based on aspects that raw PyABSA output plausibly produces.

Matching semantics (two-mode)
------------------------------
Each variant string is interpreted as one of two modes:

  * PREFIX variant (ends with '*'):
    The '*' is stripped and the stem is matched as a WORD-START-ANCHORED PREFIX.
    Regex: ``(?<!\\w)<stem>`` — the stem must start at a word boundary (not preceded
    by a word character) but may continue freely. For example, ``"graphic*"`` matches
    "graphics", "graphical", "graphically" but NOT "autographic".

  * WHOLE-WORD / WHOLE-PHRASE variant (no '*'):
    The variant is matched with word boundaries on BOTH sides:
    ``(?<!\\w)<variant>(?!\\w)``. For example, ``"hard"`` as a variant would only match
    the standalone word "hard", not "hardware" or "hard drive". Since 'hard' and 'easy'
    have been removed from difficulty, those substring false positives are eliminated.

Special case — '$':
    The dollar sign is a non-word character so ``(?<!\\w)\\$(?!\\w)`` would require
    it to be flanked by non-word chars. In practice raw aspects like "$60" or "60$"
    have a digit on one side, so boundary anchors would make '$' unmatchable as a
    standalone price indicator. '$' is therefore matched as a plain substring (``"$" in
    raw_lower``) by special-casing it in the normalizer. This behaviour is documented
    and tested in normalize_aspects.py.

Matching is always case-insensitive (normalize to lowercase before matching).
The FIRST canonical (in insertion order) whose any variant matches wins. This is the
tie-break rule; the four PRD-locked aspects always beat the additional ones.

This file runs in the ABSA env but is also safe to import from the rec env —
no pyabsa or recbole imports here.
"""

# CANONICAL_ASPECTS maps canonical aspect name -> list of variant strings.
# Each variant is either a PREFIX (ends with '*') or a WHOLE-WORD/PHRASE (no '*').
# See the module docstring for the exact regex semantics of each mode.
#
# PRD-locked four (gameplay, graphics, price, difficulty) come first to preserve priority.
# Additional canonicals follow in rough frequency-of-occurrence order in game reviews.

CANONICAL_ASPECTS: dict[str, list[str]] = {
    # ---- PRD-locked four ----
    "gameplay": [
        "gameplay*",       # gameplay, gameplays
        "game play",       # whole phrase
        "game mechanic*",  # game mechanic, game mechanics — phrase-anchored; safe for 'mechanical keyboard'
        "mechanics",       # whole word — "the mechanics are great"; NOT prefix so 'mechanical' is safe
        "playability",     # whole word
        "game loop",       # whole phrase
    ],
    "graphics": [
        "graphic*",        # graphic, graphics, graphical
        "visual*",         # visual, visuals, visually
        "art style",       # whole phrase
        "art direction",   # whole phrase
        "texture*",        # texture, textures
        "resolution",      # whole word
        "render*",         # render, rendering, renderer
        "animation*",      # animation, animations
        # REMOVED: "look" — caused 'outlook' -> graphics false positive
    ],
    "price": [
        "price*",          # price, prices, pricey, pricing
        "cost",            # whole word
        "costs",           # whole word (plural)
        "value for",       # whole phrase — "value for money"
        "worth the",       # whole phrase — "worth the price/money"
        "overpriced",      # whole word
        "expensive",       # whole word
        "cheap",           # whole word
        "affordable",      # whole word
        "discount*",       # discount, discounted, discounts
        "$",               # special-case plain substring (see docstring)
        # REMOVED: "sale" — caused 'salesman'/'wholesale' false positives
    ],
    "difficulty": [
        "difficul*",       # difficulty, difficult
        "challeng*",       # challenge, challenging, challenged
        "punishing",       # whole word
        "brutal",          # whole word
        "forgiving",       # whole word
        "learning curve",  # whole phrase
        "steep curve",     # whole phrase
        # REMOVED: "hard" — caused 'hardware'/'hard drive' false positives
        # REMOVED: "easy" — caused 'easy setup' false positive
    ],
    # ---- Additional video-game-domain canonicals ----
    "story": [
        "story",           # whole word
        "storyline",       # whole word
        "narrative",       # whole word
        "plot",            # whole word
        "lore",            # whole word
        "writing",         # whole word
        "dialogue",        # whole word
        "character*",      # character, characters, characterization
        "ending",          # whole word
        "cutscene*",       # cutscene, cutscenes
        "quest*",          # quest, quests
    ],
    "controls": [
        "control*",        # control, controls, controller, controllable
        "input*",          # input, inputs
        "button*",         # button, buttons
        "keybind*",        # keybind, keybinds, keybinding, keybindings
        "key binding",     # whole phrase
        "mouse",           # whole word
        # NOTE: bare 'keyboard' removed — 'mechanical keyboard' (a product) must not match controls.
        # 'keybind*' and 'key binding' cover in-game key remapping contexts adequately.
        "responsiv*",      # responsive, responsiveness
        "handling",        # whole word
    ],
    "sound": [
        "sound*",          # sound, sounds, soundtrack (also matched by own entry)
        "music",           # whole word
        "audio",           # whole word
        "soundtrack",      # whole word
        "voice act*",      # voice acting, voice actor, voice actors
        "sfx",             # whole word
        "ambient",         # whole word
        # REMOVED: "score" — caused 'scoreboard' false positive
    ],
    "multiplayer": [
        "multiplayer*",    # multiplayer, multiplayers
        "online play",     # whole phrase — avoids bare 'online' hitting 'online manual'
        "online mode",     # whole phrase
        "online multiplayer",  # whole phrase
        "coop",            # whole word
        "co-op",           # whole phrase (hyphen treated as non-word char — boundaries work)
        "pvp",             # whole word
        "matchmaking",     # whole word
        "server*",         # server, servers
        "lobby",           # whole word
        "netcode",         # whole word
        # REMOVED: bare "online" — caused 'online manual' false positive
        # REMOVED: bare "lag" — moved to performance below to fix 'flagship' false positive
    ],
    "performance": [
        "performance",     # whole word
        "fps",             # whole word
        "frame rate",      # whole phrase
        "framerate",       # whole word
        "stutter*",        # stutter, stuttering, stutters
        "crash*",          # crash, crashes, crashing
        "bug*",            # bug, bugs, buggy
        "glitch*",         # glitch, glitches, glitchy
        "optimization",    # whole word
        "optimisation",    # whole word (British)
        "loading",         # whole word
        "load time",       # whole phrase
        "lag*",            # lag, lagging, laggy — word-start prefix; 'flagship' is safe because
                           # 'lag' is NOT at a word start in 'flagship' (preceded by 's')
    ],
    "value": [
        "value",           # whole word — standalone "value" (not "value for" — that hits price first)
        "worth it",        # whole phrase
        "worth buying",    # whole phrase
        "bang for",        # whole phrase
        # REMOVED: "deal" — caused 'dealer' false positive
        # REMOVED: "content" — caused 'content creator' false positive
    ],
}
