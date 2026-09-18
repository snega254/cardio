"""
Known clinical ECG layout templates.

Each template defines:
    - name
    - rows: number of trace rows in the image
    - cols: number of trace columns in the image
    - leads: 2D list of lead names, indexed [row][col]
    - has_rhythm_strip: bool
    - notes

Order follows clinical convention. Layout is detected by matching
the number of ink bands to `rows` and ink columns to `cols`.
"""

LAYOUT_TEMPLATES = {
    "6x2_standard": {
        "name": "6x2 standard (limb + chest split)",
        "rows": 6,
        "cols": 2,
        "leads": [
            ["I",   "V1"],
            ["II",  "V2"],
            ["III", "V3"],
            ["aVR", "V4"],
            ["aVL", "V5"],
            ["aVF", "V6"],
        ],
        "has_rhythm_strip": True,
        "notes": "2 columns: left=limb leads, right=chest leads. "
                 "Very common in India, GE/Philips printouts.",
    },

    "4x3_standard": {
        "name": "4x3 standard (3 columns x 4 rows)",
        "rows": 4,
        "cols": 3,
        "leads": [
            ["I",   "aVR", "V1"],
            ["II",  "aVL", "V2"],
            ["III", "aVF", "V3"],
            ["V4",  "V5",  "V6"],
        ],
        "has_rhythm_strip": True,
        "notes": "Most common US/European 12-lead format. "
                 "3 leads per row, 4 rows + rhythm strip.",
    },

    "3x4_standard": {
        "name": "3x4 standard (4 columns x 3 rows)",
        "rows": 3,
        "cols": 4,
        "leads": [
            ["I",   "aVR", "V1", "V4"],
            ["II",  "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ],
        "has_rhythm_strip": True,
        "notes": "Classic 4-column × 3-row format.",
    },

    "12x1_column": {
        "name": "12x1 single column",
        "rows": 12,
        "cols": 1,
        "leads": [
            ["I"], ["II"], ["III"], ["aVR"], ["aVL"], ["aVF"],
            ["V1"], ["V2"], ["V3"], ["V4"], ["V5"], ["V6"],
        ],
        "has_rhythm_strip": False,
        "notes": "Older format, one lead per row.",
    },

    "cabrera_6x2": {
        "name": "Cabrera 6x2 (ordered: aVR, aVL, aVF)",
        "rows": 6,
        "cols": 2,
        "leads": [
            ["I",   "V1"],
            ["II",  "V2"],
            ["III", "V3"],
            ["aVR", "V4"],
            ["aVL", "V5"],
            ["aVF", "V6"],
        ],
        "has_rhythm_strip": True,
        "notes": "Same as 6x2 but with Cabrera lead ordering convention.",
    },
}


# Leads required for the CNN model (only independent ones)
REQUIRED_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# All 12 leads (in case you need them for QA)
ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

# Derivation formulas for the redundant leads
# III = II - I
# aVR = -(I + II) / 2
# aVL = I - II / 2
# aVF = II - I / 2
DERIVED_LEADS = {"III", "aVR", "aVL", "aVF"}