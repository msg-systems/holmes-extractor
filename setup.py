from setuptools import setup, find_packages

long_description = """
**Holmes** is a Python 3 library (tested with version 3.7.2) that supports a number of
use cases involving information extraction from English and German texts. See
[Github](https://github.com/msg-systems/holmes-extractor) for more details.
"""

setup(
    name='holmes-extractor',
    version='2.0',
    description='Information extraction from English and German texts based on predicate logic',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/msg-systems/holmes-extractor',
    author='Richard Paul Hudson, msg systems ag',
    author_email='richard.hudson@msg.group',
    license='gpl-3.0',
    keywords=['nlp', 'information-extraction', 'spacy', 'spacy-extension', 'python',
        'machine-learning', 'ontology', 'semantics'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Legal Industry',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Natural Language :: German',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic'
    ],
    packages=find_packages(),
    # versions of spaCy > 2.0.12 do not currently work with neuralcoref
    install_requires=['spacy==2.0.12','neuralcoref==3.1','numpy','scipy','sklearn','bs4',
        'rdflib','jsonpickle']
)
