from .base import BaseModel, MODEL_REGISTRY, register_model

# Import all models so they auto-register
from . import trivial  # noqa: F401
from . import unet3d  # noqa: F401
from . import fno3d  # noqa: F401
from . import deeponet  # noqa: F401
from . import vit3d  # noqa: F401
from . import meshgraphnet  # noqa: F401
