from .about import __version__
from .manager import Manager
from .ontology import Ontology
import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
