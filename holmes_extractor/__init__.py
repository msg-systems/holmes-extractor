import logging
logging.getLogger("rdflib").setLevel(logging.WARNING) # avoid INFO console message on startup
from holmes_extractor.manager import Manager
from holmes_extractor.manager import MultiprocessingManager
from holmes_extractor.ontology import Ontology
