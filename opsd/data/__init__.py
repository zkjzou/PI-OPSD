"""Data helpers for the OPSD experiments.

``python -m opsd.data.fetch`` downloads (or verifies) the published data bundle -- splits, hints,
reference trajectories, metric inputs -- into ``$OPSD_DATA_ROOT`` and checks every file against the
bundle's ``manifest.json``. See ``fetch.py``.
"""
