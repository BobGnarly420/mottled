"""Model backends and family adapters.

Every backend produces a StateTrajectory; transformers are just one backend.
"""

from models.families import FamilyAdapter, resolve_family  # noqa: F401
