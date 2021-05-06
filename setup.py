from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    package_data={
        "holmes_extractor": ["data/*"]
    },
    install_requires=[
        'spacy>=3.0.6', 'coreferee~=1.0.2', 'numpy', 'scipy', 'sklearn', 'bs4',
        'rdflib', 'jsonpickle', 'msgpack-numpy', 'falcon', 'torch>=1.8.1']
)
