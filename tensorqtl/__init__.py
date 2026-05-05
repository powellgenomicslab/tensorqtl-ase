import importlib.metadata
from .tensorqtl import *
from .mixqtl import mixqtl, trc, asc, meta_analyze
from .mixqtl_cis import cis_nominal
from .mixqtl_perm import mixqtl_perm

__version__ = importlib.metadata.version(__name__)
