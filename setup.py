from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    # versions of spaCy > 2.1.0 do not currently work with neuralcoref
    install_requires=['spacy==2.1.0','neuralcoref==4.0.0','numpy','scipy','sklearn','bs4',
        'rdflib','jsonpickle','msgpack-numpy', 'falcon']
)
