from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    # versions of spaCy > 2.0.12 do not currently work with neuralcoref
    install_requires=['spacy==2.0.12','neuralcoref==3.1','numpy','scipy','sklearn','bs4',
        'rdflib','jsonpickle','msgpack-numpy<0.4.4.0']
)
