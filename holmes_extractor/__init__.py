import logging
logging.getLogger("rdflib").setLevel(logging.WARNING) # avoid INFO console message on startup
from holmes_extractor.manager import Manager as Manager
from holmes_extractor.manager import MultiprocessingManager as MultiprocessingManager
from holmes_extractor.ontology import Ontology as Ontology
