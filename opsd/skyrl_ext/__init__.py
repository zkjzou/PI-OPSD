"""Every OPSD touchpoint with SkyRL internals.

SkyRL (an external checkout at the pinned commit, ``../SkyRL-ecdaa946`` by default) is used unmodified. Where
OPSD needs behaviour SkyRL does not provide, this package supplies it through SkyRL's own extension points
(subclasses of ``RayPPOTrainer`` / the FSDP policy worker / ``SkyRLTrainConfig``, ``BasePPOExp`` hooks, the
skyrl_gym registry). Importing this package has no side effects.
"""
