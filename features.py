"""Shared feature definitions and fallbacks.

Imported by `feature_engineering.ipynb` (to build the training data), by
`model.ipynb` (to train), and by `ranking_system.py` (to serve). Anything that
has to behave identically at training time and at inference time lives here so
there is exactly one definition of it.

Label note: in the SLAM corpus the label is 1 when the learner got the token
*wrong*. The `p_recall` column therefore holds an error indicator, not a recall
indicator -- see ERROR_LABEL below.
"""

import re

# The label is 1 = learner made a mistake on this token, 0 = answered correctly.
# Confirmed empirically: the most common tokens ("Je" 0.04, "suis" 0.06) carry the
# lowest rate, the rate falls with each repeat (0.22 -> 0.11 by the 10th), and
# label=1 attempts take longer (17.2s vs 13.1s). All three only hold for an error
# label. Models trained on it predict P(error), so a HIGH score = a HARD card.
ERROR_LABEL = "p_recall"

# The 11 columns the OrdinalEncoder / OneHotEncoder are fitted on, in the order
# the ColumnTransformer receives them. The transformed frame puts these first
# (indices 0..10) and passes the numerics through after them.
CATEGORICAL_FEATURES = [
    "POS",
    "Dependency-Relation",
    "Definite",
    "Gender",
    "Number",
    "PronType",
    "Mood",
    "Tense",
    "VerbForm",
    "Reflex",
    "Person",
]

# Keys unpacked out of the raw Morpho-Syntactic_Features column.
MORPHO_COLS = [
    "Definite",
    "Gender",
    "Number",
    "Person",
    "PronType",
    "Mood",
    "Tense",
    "VerbForm",
    "Reflex",
]

# Static per-token columns: everything a flashcard knows about a word before the
# learner has seen it. These get precomputed into models/token_features.parquet.
TOKEN_FEATURE_COLS = [
    "Dependency-Relation",
    "Dependancy-Head",
    "syllable_count",
    "ortho_freq",
    "lev_ratio",
] + MORPHO_COLS

# Per-user history columns, rebuilt at runtime from UserState.
HISTORY_FEATURE_COLS = [
    "nth_occurrence",
    "overall_recall_rate",
    "token_recall_rate",
    "token_streak",
]

# Fill value for an absent categorical. Training filled nulls with this literal
# string, so inference must produce the same string and not a real NaN.
NA_TOKEN = "N/A"

_VOWELS = re.compile(r"[aeiouyàâäéèêëïîôöùûü]+")


def estimate_syllables_fr(token):
    """Heuristic French syllable count for a token missing from Lexique.

    Collapses vowel clusters into single nuclei and discards a silent final
    'e'. Returns 0 for pure punctuation, otherwise at least 1.
    """
    word = str(token).lower()

    # Pure punctuation / non-alphabetic has no syllables.
    if not any(char.isalpha() for char in word):
        return 0

    count = len(_VOWELS.findall(word))
    # A silent final 'e' doesn't usually carry its own syllable.
    if word.endswith("e") and count > 1:
        count -= 1

    return max(count, 1)


def parse_morpho(morpho_features):
    """Split "Key=Value|Key=Value|..." into a dict, dropping the redundant fPOS.

    Accepts NaN/None for tokens that carry no morphological features.
    """
    parsed = {}
    if morpho_features is None or morpho_features != morpho_features:  # NaN check
        return parsed

    for pair in str(morpho_features).split("|"):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        if key == "fPOS":  # restates POS, already its own column
            continue
        parsed[key] = value

    return parsed
