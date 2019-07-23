Holmes
======
Author: <a href="mailto:richard.hudson@msg.group">Richard Paul Hudson, msg systems ag</a>

-   [1. Introduction](#introduction)
    -   [1.1 The basic idea](#the-basic-idea)
    -   [1.2 Installation](#installation)
        -   [1.2.1 Prerequisites](#prerequisites)
        -   [1.2.2 Library installation](#library-installation)
        -   [1.2.3 Installing the spaCy models](#installing-the-spacy-models)
        -   [1.2.4 Comments about deploying Holmes in an
            enterprise
            environment](#comments-about-deploying-holmes-in-an-enterprise-environment)
    -   [1.3 Getting started](#getting-started)
-   [2. Word-level matching strategies](#word-level-matching-strategies)
    -   [2.1 Direct matching](#direct-matching)
    -   [2.2 Named entity matching](#named-entity-matching)
    -   [2.3 Ontology-based matching](#ontology-based-matching)
    -   [2.4 Embedding-based matching](#embedding-based-matching)
-   [3. Coreference resolution](#coreference-resolution)
-   [4. Writing effective search
    phrases](#writing-effective-search-phrases)
    -   4.1 [General comments](#general-comments)
        -   [4.1.1 Lexical versus grammatical words](#lexical-versus-grammatical-words)
        -   [4.1.2 Use of the present active](#use-of-the-present-active)
        -   [4.1.3 Generic pronouns](#generic-pronouns)
        -   [4.1.4 Prepositions](#prepositions)
    -   [4.2 Structures not permitted in search
        phrases](#structures-not-permitted-in-search-phrases)
        -   [4.2.1 Multiple clauses](#multiple-clauses)
        -   [4.2.2 Negation](#negation)
        -   [4.2.3 Conjunction](#conjunction)
        -   [4.2.4 Lack of lexical words](#lack-of-lexical-words)
        -   [4.2.5 Coreferring pronouns](#coreferring-pronouns)
    -   [4.3 Structures strongly discouraged in search
        phrases](#structures-strongly-discouraged-in-search-phrases)
        -   [4.3.1 Ungrammatical
            expressions](#ungrammatical-expressions)
        -   [4.3.2 Complex verb tenses](#complex-verb-tenses)
        -   [4.3.3 Questions](#questions)
    -   [4.4 Structures to be used with caution in search
        phrases](#structures-to-be-used-with-caution-in-search-phrases)
        -   [4.4.1 Very complex
            structures](#very-complex-structures)
        -   [4.4.2 Deverbal noun phrases](#deverbal-noun-phrases)
-   [5. Use cases and examples](#use-cases-and-examples)
    -   [5.1 Chatbot](#chatbot)
    -   [5.2 Structural matching](#structural-matching)
    -   [5.3 Topic matching](#topic-matching)
    -   [5.4 Supervised document classification](#supervised-document-classification)
-   [6 Interfaces intended for public
    use](#interfaces-intended-for-public-use)
    -   [6.1 `Manager`](#manager)
    -   [6.2 `Ontology`](#ontology)
    -   [6.3 `SupervisedTopicTrainingBasis`](#supervised-topic-training-basis)
    (returned from `Manager.get_supervised_topic_training_basis()`)
    -   [6.4 `SupervisedTopicModelTrainer`](#supervised-topic-model-trainer)
    (returned from `SupervisedTopicTrainingBasis.train()`)
    -   [6.5 `SupervisedTopicClassifier`](#supervised-topic-classifier)
    (returned from `SupervisedTopicModelTrainer.classifier()` and
    `Manager.deserialize_supervised_topic_classifier()`)
    -   [6.6 `Match` (returned from
          `Manager.match()`)](#match)
    -   [6.7 `WordMatch` (returned from
        `Manager.match().word_matches`)](#wordmatch)
    -   [6.8 Dictionary returned from
        `Manager.match_returning_dictionaries()`)](#dictionary)
    -   [6.9 `TopicMatch`](#topic-match)
    (returned from `Manager.topic_match_documents_against()`)
-   [7 A note on the license](#a-note-on-the-license)
-   [8 Information for developers](#information-for-developers)
    -   [8.1 How it works](#how-it-works)
        - [8.1.1 Structural matching](#how-it-works-structural-matching)
        - [8.1.2 Topic matching](#how-it-works-topic-matching)
        - [8.1.3 Supervised document classification](#how-it-works-supervised-document-classification)
    -   [8.2 Development and testing
        guidelines](#development-and-testing-guidelines)
    -   [8.3 Areas for further
        development](#areas-for-further-development)
        -   [8.3.1 Incorporation into the spaCy
        multithreading architecture](#incorporation-into-the-spacy-multithreading-architecture)
        -   [8.3.2 Additional languages](#additional-languages)
        -   [8.3.3 Use of machine learning to improve
            matching](#use-of-machine-learning-to-improve-matching)
        -   [8.3.4 Upgrade to latest library versions](#upgrade-to-latest-library-versions)
        -   [8.3.5 Remove names from supervised document classification models](#remove-names-from-supervised-document-classification-models)
        -   [8.3.6 Improve the performance of supervised document classification training](#improve-performance-of-supervised-document-classification-training)
        -   [8.3.7 Explore the optimal hyperparameters for topic matching and supervised document classification](#explore-hyperparameters)


<a id="introduction"></a>
### 1. Introduction

<a id="the-basic-idea"></a>
#### 1.1 The basic idea

**Holmes** is a Python 3 library (tested with version 3.7.2) that supports a number of
use cases involving information extraction from English and German texts. In all use cases, the information extraction
is based on analysing the semantic relationships expressed by the component parts of each sentence:

- In the [chatbot](#getting-started) use case, the system is configured using one or more **search phrases**.
Holmes then looks for structures whose meanings correspond to those of these search phrases within
a searched **document**, which in this case corresponds to an individual snippet of text or speech
entered or uttered by the end user. Within a match, each non-grammatical word in the search phrase
corresponds to one or more non-grammatical words in the document, which can then be extracted as structured information.

- The [structural matching](#structural-matching) use case uses exactly the same technological basis as the chatbot use
case, but searching takes place with respect to a pre-existing document or documents that are typically much
longer than the snippets analysed in the chatbot use case.

- The [topic matching](#topic-matching) use case aims to find passages in a document or documents whose meaning
is close to that of another document, which takes on the role of the **query document**, or to that of a
**query phrase** entered ad-hoc by the user. Holmes extracts a number of small **phraselets** from the query phrase or
query document, matches the documents being searched against each phraselet, and conflates the results to find
the most relevant passages within the documents. Because there is no strict requirement that every non-grammatical
word in the query document match a specific word or words in the searched documents, more matches are found
than in the structural matching use case, but the matches do not contain structured information that can be
used in subsequent processing.

- The [supervised document classification](#supervised-document-classification) use case uses training data to
learn a classifier that assigns one or more **classification labels** to new documents based on what they are about.
It classifies a new document by matching it against phraselets that were extracted from the training documents in the
same way that phraselets are extracted from the query document in the topic matching use case. The technique is
inspired by bag-of-words-based classification algorithms that use n-grams, but aims to derive n-grams whose component
words are related semantically rather than that just happen to be neighbours in the surface representation of a language.

In all four use cases, the **individual words** are matched using a [number of strategies](#word-level-matching-strategies).
To work out whether two grammatical structures that contain individually matching words correspond logically and
constitute a match, Holmes transforms the syntactic parse information provided by the [spaCy](https://spacy.io/) library
into semantic structures that allow texts to be compared using predicate logic. As a user of Holmes, you do not need to
understand the intricacies of how this works, although there are some
[important tips](#writing-effective-search-phrases) around writing effective search phrases for the chatbot and
structured matching use cases that you should try and take on board.

Holmes aims to offer generalist solutions that can be used more or less out of the box with
relatively little tuning, tweaking or training and that are rapidly applicable to a wide range of use cases.
At its core lies a logical, programmed, rule-based system that describes how syntactic representations in each
language express semantic relationships. Although the supervised document classification use case does incorporate a
neural network and although the spaCy library upon which Holmes builds has itself been pre-trained using machine
learning, the essentially rule-based nature of Holmes means that the chatbot, structural matching and topic matching use
cases can be put to use out of the box without any training and that the supervised document classification use case
typically requires relatively little training data, which is a great advantage because pre-labelled training data is
not available for many real-world problems.

<a id="installation"></a>
#### 1.2 Installation

<a id="prerequisites"></a>
##### 1.2.1 Prerequisites

If you do not already have [Python 3](https://realpython.com/installing-python/) and
[pip](https://pypi.org/project/pip/) on your machine, you will need to install them
before installing Holmes.

<a id="library-installation"></a>
##### 1.2.2 Library installation

Because of a conflict between the install scripts of two of Holmes' dependencies
(`neuralcoref` and `numpy`), `numpy` has to be installed before the Holmes installation
script runs. Install Holmes using the following commands:

*Linux:*
```
pip3 install numpy
pip3 install holmes-extractor
```

*Windows:*
```
pip install numpy
pip install holmes-extractor
```

If you are working on Windows and have not used Python before,
several of Holmes' dependencies require you to download Visual Studio and then
rerun the installation. During the Visual Studio install, it is imperative to select
the **Desktop Development with C++** option, which is not checked by default.

If you wish to use the examples and tests, clone the source code using

```
git clone https://github.com/msg-systems/holmes-extractor
```

Note that at present spaCy version 2.0.12 is installed rather than the current version
because of a conflict between later versions of spaCy and the version of `neuralcoref` that
was available when Holmes 2.0 was developed. This problem has been resolved in the latest version
of `neuralcoref`, and updating Holmes to work with the latest versions of both libraries is
[on the to-do list](#upgrade-to-latest-library-versions).

If you wish to experiment with changing the source code, you can
override the installed code by starting Python (type `python3` (Linux) or `python`
(Windows)) in the parent directory of the directory where your altered `holmes_extractor`
module code is. If you have checked Holmes out of Git, this will be the `holmes-extractor` directory.

If you wish to uninstall Holmes again, this is achieved by deleting the installed
file(s) directly from the file system. These can be found by issuing the
following from the Python command prompt started from any directory **other**
than the parent directory of `holmes_extractor`:

```
import holmes_extractor
print(holmes_extractor.__file__)
```

<a id="installing-the-spacy-models"></a>
##### 1.2.3 Installing the spaCy models

The spaCy library that Holmes builds upon requires
[language-specific models](https://spacy.io/usage/models) that have to be downloaded
separately before Holmes can be used. The following models are for English without
coreference resolution, English with coreference resolution, and German respectively:

*Linux:*
```
python3 -m spacy download en_core_web_lg
pip3 install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_lg-3.0.0/en_coref_lg-3.0.0.tar.gz
python3 -m spacy download de_core_news_sm
```

*Windows:*
```
python -m spacy download en_core_web_lg
pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_lg-3.0.0/en_coref_lg-3.0.0.tar.gz
python -m spacy download de_core_news_sm
```

Note that, for English, other, smaller models are also available. Users of Holmes are nonetheless urged to stick to the `en_core_web_lg` and `en_coref_lg` models as they have consistently been found to yield the best results.

<a id="comments-about-deploying-holmes-in-an-enterprise-environment"></a>
##### 1.2.4 Comments about deploying Holmes in an enterprise environment

Python 3 is a language that is absent from the architecture standards of
many large enterprises. For a number of reasons, however, it was the
only serious contender with which to develop Holmes.

The best way of integrating Holmes into a non-Python environment is to
wrap it as a RESTful HTTP service and to deploy it as a
microservice.

<a id="getting-started"></a>
#### 1.3 Getting started

The easiest use case with which to get a quick basic idea of how Holmes works is the **chatbot** use case.

Here one or more search phrases are defined to Holmes in advance, and the
searched 'documents' are short sentences or paragraphs typed in
interactively by an end user. In a real-life setting, the extracted
information would probably be stored in a database and/or used to
determine the flow of interaction with the end user. For testing and
demonstration purposes, there is a console that displays
its matched findings interactively. It can be easily and
quickly started from the Python command line (which is itself started from the
operating system prompt by typing `python3` (Linux) or `python` (Windows))
or from within a [Jupyter notebook](https://jupyter.org/).

The following code snippet can be entered line for line into the Python command
line, into a Jupyter notebook or into an IDE. It registers the fact that you are
interested in sentences about big dogs chasing cats and starts a
demonstration chatbot console:

*English:*

```
import holmes_extractor as holmes
holmes_manager = holmes.Manager(model='en_coref_lg')
holmes_manager.register_search_phrase('A big dog chases a cat')
holmes_manager.start_chatbot_mode_console()
```

*German:*

```
import holmes_extractor as holmes
holmes_manager = holmes.Manager(model='de_core_news_sm')
holmes_manager.register_search_phrase('Ein großer Hund jagt eine Katze')
holmes_manager.start_chatbot_mode_console()
```

If you now enter a sentence that corresponds to the search phrase, the
console will display a match:

*English:*

```
Ready for input

A big dog chased a cat


Matched search phrase 'A big dog chases a cat':
'big'->'big' (direct); 'A big dog'->'dog' (direct); 'chased'->'chase' (direct); 'a cat'->'cat' (direct)
```

*German:*

```
Ready for input

Ein großer Hund jagte eine Katze


Matched search phrase 'Ein großer Hund jagt eine Katze':
'großer'->'groß' (direct); 'Ein großer Hund'->'hund' (direct); 'jagte'->'jagen' (direct); 'eine Katze'->'katze' (direct)
```

This could easily have been achieved with a simple matching algorithm, so type
in a few more complex sentences to convince yourself that Holmes is
really grasping them and that matches are still returned:

*English:*

```
The big dog would not stop chasing the cat
The big dog who was tired chased the cat
The cat was chased by the big dog
The cat always used to be chased by the big dog
The big dog was going to chase the cat
The big dog decided to chase the cat
The cat was afraid of being chased by the big dog
I saw a cat-chasing big dog
You saw a big-dog-chased cat
The cat the big dog chased was scared
The big dog chasing the cat was a problem
There was a big dog that was chasing a cat
```

*German:*

```
Der große Hund hat die Katze ständig gejagt
Der große Hund, der müde war, jagte die Katze
Die Katze wurde vom großen Hund gejagt
Die Katze wurde immer wieder durch den großen Hund gejagt
Der große Hund wollte die Katze jagen
Der große Hund entschied sich, die Katze zu jagen
Die Katze hatte die Nase voll, vom großen Hund gejagt zu werden
Die Katze, die der große Hund gejagt hatte, hatte Angst
Dass der große Hund die Katze jagte, war ein Problem
Es gab einen großen Hund, der eine Katze jagte
```

In English but not presently in German, [coreference resolution](#coreference-resolution)
is active. This means that the system can link pronouns and nouns to other pronouns and nouns
nearby in the same text that refer to the same entities. It increases the variety of
structures that Holmes can recognise:

*English:*

```
There was a big dog and it was chasing a cat.
I saw a big dog. My cat was afraid of being chased by the dog.
The big dog was called Fido. He was chasing my cat.
A dog appeared. It was chasing a cat. It was very big.
The cat sneaked back into our lounge because a big dog had been chasing her outside.
Our big dog was excited because he had been chasing a cat.
```

The demonstration is not complete without trying other sentences that
contain the same words but do not express the same idea and observing that they
are **not** matched:

*English:*

```
The dog chased a big cat
The big dog and the cat chased about
The big dog chased a mouse but the cat was tired
The big dog always used to be chased by the cat
I saw a big-dog-chasing cat
The big dog the cat chased was scared
Our big dog was upset because he had been chased by a cat.
```

*German:*

```
Der Hund jagte eine große Katze
Den großen Hund jagt die Katze
Der große Hund und die Katze jagten
Der große Hund jagte eine Maus aber die Katze war müde
Der große Hund wurde ständig von der Katze gejagt
Der große Hund entschloss sich, von der Katze gejagt zu werden
```

In the above examples, Holmes has matched a variety of different
sentence-level structures that share the same meaning, but the base
forms of the three words in the matched documents have always been the
same as the three words in the search phrase. Holmes provides
several further strategies for matching at the individual word level. In
combination with Holmes's ability to match different sentence
structures, these can enable a search phrase to be matched to a document
sentence that shares its meaning even where the two share no words and
are grammatically completely different.

One of these additional word-matching strategies is [named-entity
matching](#named-entity-matching): special words can be included in search phrases
that match whole classes of names like people or places. Exit the
console by typing `exit`, then register a second search phrase and
restart the console:

*English:*

```
holmes_manager.register_search_phrase('An ENTITYPERSON goes into town')
holmes_manager.start_chatbot_mode_console()
```

*German:*

```
holmes_manager.register_search_phrase('Ein ENTITYPER geht in die Stadt')
holmes_manager.start_chatbot_mode_console()
```

You have now registered your interest in people going into town and can
enter appropriate sentences into the console:

*English:*

```
Ready for input

I met Richard Hudson and John Doe last week. They didn't want to go into town.


Matched search phrase 'An ENTITYPERSON goes into town'; negated; uncertain; involves coreference:
'Richard Hudson'->'ENTITYPERSON' (entity); 'go'->'go' (direct); 'into'->'into' (direct); 'town'->'town' (direct)

Matched search phrase 'An ENTITYPERSON goes into town'; negated; uncertain; involves coreference:
'John Doe'->'ENTITYPERSON' (entity); 'go'->'go' (direct); 'into'->'into' (direct); 'town'->'town' (direct)
```

*German:*

```
Ready for input

Richard Hudson und Max Mustermann wollten nicht mehr in die Stadt gehen


Matched search phrase 'Ein ENTITYPER geht in die Stadt'; negated; uncertain:
'Richard Hudson'->'ENTITYPER' (entity); 'gehen'->'gehen' (direct); 'in'->'in' (direct); 'die Stadt'->'stadt' (direct)

Matched search phrase 'Ein ENTITYPER geht in die Stadt'; negated; uncertain:
'Max Mustermann'->'ENTITYPER' (entity); 'gehen'->'gehen' (direct); 'in'->'in' (direct); 'die Stadt'->'stadt' (direct)
```

In each of the two languages, this last example demonstrates several
further features of Holmes:

-   It can match not only individual words, but also **multiword**
    phrases like *Richard Hudson*.
-   When two or more words or phrases are linked by **conjunction**
    (*and* or *or*), Holmes extracts a separate match for each.
-   When a sentence is **negated** (*not*), Holmes marks the match
    accordingly.
-   Like several of the matches yielded by the more complex entry
    sentences in the above example about big dogs and cats, Holmes marks the
    two matches as **uncertain**. This means that the search phrase was
    not matched exactly, but rather in the context of some other, more
    complex relationship ('wanting to go into town' is not the same
    thing as 'going into town').

For more examples, please see [section 5](#use-cases-and-examples).

<a id="word-level-matching-strategies"></a>
### 2. Word-level matching strategies

The same word-level matching strategies are employed with [all use cases](#use-cases-and-examples) and most
of the comments that follow apply equally to all use cases. The exception to this principle
is that [ontology-based matching](#ontology-based-matching) works differently depending on
the use case.

<a id="direct-matching"></a>
#### 2.1 Direct matching (`word_match.type=='direct'`)

Direct matching between search phrase words and document words is always
active. The strategy relies mainly on matching stem forms of words,
e.g. matching English *buy* and *child* for *bought* and *children*,
German *steigen* and *Kind* for *stieg* and *Kinder*. However, in order to
increase the chance of direct matching working when the parser delivers an
incorrect stem form for a word, the raw text forms of both search-phrase and
document words are also taken into consideration during direct matching.

<a id="named-entity-matching"></a>
#### 2.2 Named entity matching (`word_match.type=='entity'`)

Named entity matching is activated by inserting a special named-entity
identifier at the desired point in a search phrase in place of a noun,
e.g.

***An ENTITYPERSON goes into town*** (English)  
***Ein ENTITYPER geht in die Stadt*** (German).

The supported named-entity identifiers depend directly on the named
entity information supplied by the spaCy models for each language
(descriptions copied from the [spaCy
documentation](https://spacy.io/usage/linguistic-features#section-named-entities)):

*English:*

|Identifier           | Meaning|
|---------------------| ------------------------------------------------------|
|ENTITYNOUN           | Any noun phrase.|
|ENTITYPERSON         | People, including fictional.|
|ENTITYNORP           | Nationalities or religious or political groups.|
|ENTITYFAC            | Buildings, airports, highways, bridges, etc.|
|ENTITYORG            | Companies, agencies, institutions, etc.|
|ENTITYGPE            | Countries, cities, states.|
|ENTITYLOC            | Non-GPE locations, mountain ranges, bodies of water.|
|ENTITYPRODUCT        | Objects, vehicles, foods, etc. (Not services.)|
|ENTITYEVENT          | Named hurricanes, battles, wars, sports events, etc.|
|ENTITYWORK_OF_ART    | Titles of books, songs, etc.|
|ENTITYLAW            | Named documents made into laws.|
|ENTITYLANGUAGE       | Any named language.|
|ENTITYDATE           | Absolute or relative dates or periods.|
|ENTITYTIME           | Times smaller than a day.|
|ENTITYPERCENT        | Percentage, including "%".|
|ENTITYMONEY          | Monetary values, including unit.|
|ENTITYQUANTITY       | Measurements, as of weight or distance.|
|ENTITYORDINAL        | "first", "second", etc.|
|ENTITYCARDINAL       | Numerals that do not fall under another type.|


*German:*

|Identifier|                                Meaning|
|----|----|
|ENTITYNOUN |                               Any noun phrase.|
|ENTITYPER   |                              Named person or family.|
|ENTITYLOC    |                             Name of politically or  geographically defined                                        location (cities, provinces, countries, international regions, bodies of water,                                          mountains).|
|ENTITYORG     |                            Named corporate, governmental, or other                                          organizational entity.|
|ENTITYMISC     |                           Miscellaneous entities, e.g. events, nationalities, products or works of art.|

We have added `ENTITYNOUN` to the genuine named-entity identifiers. As
it matches any noun phrase, it behaves in a similar fashion to [generic pronouns](#generic-pronouns).
The differences are that `ENTITYNOUN` has to match a specific noun phrase within a document
and that this specific noun phrase is extracted and available for further processing.

<a id="ontology-based-matching"></a>
#### 2.3 Ontology-based matching (`word_match.type=='ontology'`)

An ontology enables the user to define relationships between words that
are then taken into account when matching documents to search phrases.
The three relevant relationship types are *hyponyms* (something is a
subtype of something), *synonyms* (something means the same as
something) and *named individuals* (something is a specific instance of
something). The three relationship types are exemplified in Figure 1:

![Figure 1](https://github.com/msg-systems/holmes-extractor/blob/master/docs/ontology_example.png)

Ontologies are defined to Holmes using the [OWL ontology
standard](https://www.w3.org/OWL/) serialized using
[RDF/XML](https://www.w3.org/2001/sw/wiki/RDF). Such ontologies
can be generated with a variety of tools. For the Holmes [examples](#use-cases-and-examples) and
[tests](#development-and-testing-guidelines), the free tool
[Protege](https://protege.stanford.edu/) was used. It is recommended
that you use Protege both to define your own ontologies and to browse
the ontologies that ship with the examples and tests. When saving an
ontology under Protege, please select *RDF/XML* as the format. Protege
assigns standard labels for the hyponym, synonym and named-individual relationships
that Holmes [understands as defaults](#ontology) but that can also be
overridden.

Ontology entries are defined using an Internationalized Resource
Identifier (IRI),
e.g. `http://www.semanticweb.org/hudsonr/ontologies/2019/0/animals#dog`.
Holmes only uses the final fragment for matching, which allows homonyms
(words with the same form but multiple meanings) to be defined at
multiple points in the ontology tree.

Ontology-based matching gives the best results with Holmes when small
ontologies are used that have been built for specific subject domains
and use cases. For example, if you are implementing a chatbot for a
building insurance use case, you should create a small ontology capturing the
terms and relationships within that specific domain. On the other hand,
it is not recommended to use large ontologies built
for all domains within an entire language such as
[WordNet](https://wordnet.princeton.edu/). This is because the many
homonyms and relationships that only apply in narrow subject
domains will tend to lead to a large number of incorrect matches. For
general use cases, [embedding-based matching](#embedding-based-matching) will tend to yield better results.

Each word in an ontology can be regarded as heading a subtree consisting
of its hyponyms, synonyms and named individuals, those words' hyponyms,
synonyms and named individuals, and so on. With an ontology set up in the standard fashion that
is appropriate for the [chatbot](#chatbot) and [structural matching](#structural-matching) use cases,
a word in a Holmes search phrase matches a word in a document if the document word is within the
subtree of the search phrase word. Were the ontology in Figure 1 defined to Holmes, in addition to the
[direct matching strategy](#direct-matching), which would match each word to itself, the
following combinations would match:

-   *animal* in a search phrase would match *hound*, *dog*, *cat*,
    *pussy*, *puppy*, *Fido*, *kitten* and *Mimi Momo* in documents;
-   *hound* in a search phrase would match *dog*, *puppy* and *Fido* in
    documents;
-   *dog* in a search phrase would match *hound*, *puppy* and *Fido* in
    documents;
-   *cat* in a search phrase would match *pussy*, *kitten* and *Mimi
    Momo* in documents;
-   *pussy* in a search phrase would match *cat*, *kitten* and *Mimi
    Momo* in documents.

English phrasal verbs like *eat up* and German separable verbs like *aufessen*  
must be defined as single items within ontologies. When Holmes is analysing a text and
comes across such a verb, the main verb and the particle are conflated into a single
logical word that can then be matched via an ontology. This means that *eat up* within
a text would match the subtree of *eat up* within the ontology but not the subtree of
*eat* within the ontology.

In situations where finding relevant sentences is more important than
ensuring the logical correspondence of document matches to search phrases,
it may make sense to specify **symmetric matching** when defining the ontology.
Symmetric matching is recommended for the [topic matching](#topic-matching) use case, but
is unlikely to be appropriate for the [chatbot](#chatbot) or [structural matching](#structural-matching) use cases.
It means that the hypernym (reverse hyponym) relationship is taken into account as well as the
hyponym and synonym relationships when matching, thus leading to a more symmetric relationship
between documents and search phrases. An important rule applied when matching via a symmetric ontology is that a match path may not contain both hypernym and hyponym relationships, i.e. you cannot go back on yourself. Were the
ontology above defined as symmetric, the following combinations would match:

-   *animal* in a search phrase would match *hound*, *dog*, *cat*,
    *pussy*, *puppy*, *Fido*, *kitten* and *Mimi Momo* in documents;
-   *hound* in a search phrase would match *animal*, *dog*, *puppy* and *Fido* in
    documents;
-   *dog* in a search phrase would match *animal*, *hound*, *puppy* and *Fido* in
    documents;
-   *puppy* in a search phrase would match *animal*, *dog* and *hound* in documents;
-   *Fido* in a search phrase would match *animal*, *dog* and *hound* in documents;    
-   *cat* in a search phrase would match *animal*, *pussy*, *kitten* and *Mimi
    Momo* in documents;
-   *pussy* in a search phrase would match *animal*, *cat*, *kitten* and *Mimi
    Momo* in documents.
-   *kitten* in a search phrase would match *animal*, *cat* and *pussy* in documents;
-   *Mimi Momo* in a search phrase would match *animal*, *cat* and *pussy* in documents.

In the [supervised document classification](#supervised-document-classification) use case,
two separate ontologies can be used:

- The **structural matching** ontology is used to analyse the content of both training
and test documents. Each word from a document that is found in the ontology is replaced by its most general hypernym
ancestor. It is important to realise that an ontology is only likely to work with structural matching for
supervised document classification if it was built specifically for the purpose: such an ontology
should consist of a number of separate trees representing the main classes of object in the documents
to be classified. In the example ontology shown above, all words in the ontology would be replaced with
*animal*; in an extreme case with a WordNet-style ontology, all nouns would end up being replaced with
*thing*, which is clearly not a desirable outcome!

- The **classification** ontology is used to capture relationships between classification labels: that a document
has a certain classification implies it also has any classifications to whose subtree that classification belongs.
Synonyms should be used sparingly if at all in classification ontologies because they add to the complexity of the
neural network without adding any tangible value; and although it is technically possible to set up a classification
ontology to use symmetric matching, there is no sensible reason for doing so. Note that a label within the
classification ontology that is not directly defined as the label of any training document
[has to be registered specifically](#supervised-topic-training-basis) using the
`SupervisedTopicTrainingBasis.register_additional_classification_label()` method if it is to be taken into
account when training the classifier.

<a id="embedding-based-matching"></a>
#### 2.4 Embedding-based matching (`word_match.type=='embedding'`)

For English but not presently for German, spaCy offers **word
embeddings**: machine-learning-generated numerical vector
representations of words that capture the contexts in which each word
tends to occur. Two words with similar meaning tend to emerge with word
embeddings that are close to each other, and spaCy can measure the
**similarity** between any two words' embeddings expressed as a decimal
between 0.0 (no similarity) and 1.0 (the same word). Because *dog* and
*cat* tend to appear in similar contexts, they have a similarity of
0.80; *dog* and *horse* have less in common and have a similarity of
0.62; and *dog* and *iron* have a similarity of only 0.25.

Holmes makes use of word-embedding-based similarities using a globally
defined **overall similarity threshold**. A match is detected between a
search phrase and a structure within a document whenever the geometric
mean of the similarities between the individual corresponding word pairs
is greater than the threshold. The intuition behind this technique is
that where a search phrase with e.g. six lexical words has matched a
document structure where five of these words match exactly and only one
corresponds via an embedding, the similarity that should be required to match this sixth word is less than
when only three of the words matched exactly and all of the other three only correspond via embeddings.

It is important to understand that the fact that two words have similar
embeddings does not imply the same sort of logical relationship between
the two as when [ontology-based matching](#ontology-based-matching) is used: for example, the
fact that *dog* and *cat* have similar embeddings means neither that a
dog is a type of cat nor that a cat is a type of dog. Whether or not
embedding-based matching is nonetheless an appropriate choice depends on
the use case. It is more likely to be appropriate for the [topic matching](#topic-matching) and
[supervised document classification](#supervised-document-classification) use cases than for the
[chatbot](#chatbot) and [structural matching](#structural-matching) use cases.

Matching a search phrase to a document begins by finding words
in the document that match the word at the root (syntactic head) of the
search phrase. Holmes then investigates the structure around each of
these matched document words to check whether the document structure matches
the search phrase structure in its entirity.
The document words that match the search phrase root word are normally found
using an index. However, if embeddings have to be taken into account when
finding document words that match a search phrase root word, **every** word in
**every** document has to be compared for similarity to that search phrase root word.
This has a very noticeable performance hit that renders all use cases except the
[chatbot](#chatbot) use case unusable if large numbers of documents
are being analysed.

At the same time, the root words of typical Holmes search phrases
and phraselets are verbs, and embedding-based matching often yields few results for verbs
in any case. To avoid the typically unnecessary performance hit that results from embedding-based matching
of search phrase root words, it is [controlled separately](#manager) from embedding-based matching in general
using the `embedding_based_matching_on_root_words` parameter, and the default and advised setting is
that it should remain switched off (value `False`).

<a id="coreference-resolution"></a>
### 3. Coreference resolution

As explained in the [initial examples](#getting-started), Holmes can be configured to use
**coreference resolution** when analysing English (but not yet German). This
means that situations are recognised where pronouns and nouns that are located near one another
within a text refer to the same entities. The information from one mention can then
be applied to the analysis of further mentions:

I saw a *big dog*. *It* was chasing a cat.   
I saw a *big dog*. *The dog* was chasing a cat.

Coreference resolution is performed using the [neuralcoref](https://github.com/huggingface/neuralcoref)
library running on top of spaCy. The neuralcoref version used to build Holmes published specific spaCy models that
are additionally trained for coreference resolution and uses them when it runs; Holmes makes use of
coreference resolution information by importing the relevant neuralcoref model instead of the vanilla spaCy model.
For example, Holmes can use the `en_coref_lg` neuralcoref model in place of the `en_core_web_lg` standard spaCy model.
The reason why Holmes cannot currently consider coreference resolution for German is that, at the time of writing, a
German neuralcoref model has not yet been published. As and when such a model becomes available,
Holmes should theoretically be able to use it immediately without additional development work, although this
would obviously need to be tested.

The `neuralcoref` library detects chains of coreferring nouns and pronouns that can
grow to considerable lengths when longer texts are analysed. For Holmes, it has been found
to be appropriate to limit the consideration of coreference resolution information to a small
number of mentions either side of a noun or pronoun within a chain: the threshold is currently set to 3.

Alongside the main use of coreference resolution information to increase the scope of
structural matching between search phrases and documents, Holmes also looks for situations
where a matched word is in a coreference chain with another word that is linked to the
matched word in an [ontology](#ontology-based-matching) and that is more specific than the
matched word:

We discussed *msg systems*. *The company* had made a profit.

If this example were to match the search phrase ***A company makes a profit*** and if
*msg systems* were defined as a named-individual instance of *company* in the ontology, the
coreference information that the company under discussion is msg systems is clearly
relevant and worth extracting in addition to the word(s) directly matched to the search
phrase. Such information is captured in the [word_match.extracted_word](#wordmatch) field.

A caveat applies when using coreference resolution in the context of the
[structural matching](#structural-matching) use case. The `neuralcoref` library yields excellent results with
grammatical structures of low or average complexity. However, with very complex texts, the proportion of errors in
the detected coreference chains seems to increase significantly to an extent that is not observed either for the
underlying spaCy syntactic parses or for the Holmes semantic interpretations of them. This is presumably because humans
performing coreference resolution rely partially on information about the world to which the library does
not have access. This should be borne in mind when extracting structured information from very complex documents:
there is a danger that using coreference resolution will lead to an unacceptable proportion of the
extracted information being incorrect.

The `neuralcoref` library does not [currently](#upgrade-to-latest-library-versions) support
[serialization](#manager-serialize-function): an
attempt to serialize a document parsed using a model that supports coreference resolution will result in
an error being raised. Note that this is the case irrespective of whether coreference resolution
is switched on in the [Manager](#manager) class. This may be a further consideration when deciding whether
to use a `neuralcoref` model or an original spaCy model.

<a id="writing-effective-search-phrases"></a>
### 4. Writing effective search phrases

<a id="general-comments"></a>
#### 4.1 General comments

The concept of search phrases has [already been introduced](#getting-started) and is relevant to the
chatbot use case, the structured matching use case and to [preselection](#preselection) within the supervised
document classification use case.

Structural matching between search phrases and documents is not symmetric: there
are many situations in which sentence X as a search phrase would match
sentence Y within a document but where the converse would not be true.
Although Holmes does its best to understand any search phrases, the
results are better when the user writing them follows certain patterns
and tendencies, and getting to grips with these patterns and tendencies is
the key to using the relevant features of Holmes successfully.

<a id="lexical-versus-grammatical-words"></a>
##### 4.1.1 Lexical versus grammatical words

Holmes distinguishes between: **lexical words** like *dog*, *chase* and
*cat* (English) or *Hund*, *jagen* and *Katze* (German) in the [initial
example above](#getting-started); and **grammatical words** like *a* (English)
or *ein* and *eine* (German) in the initial example above. Only lexical words match
words in documents, but grammatical words still play a crucial role within a
search phrase: they enable Holmes to understand it.

***Dog chase cat*** (English)  
***Hund jagen Katze*** (German)

contain the same lexical words as the search phrases in the [initial
example above](#getting-started), but as they are not grammatical sentences Holmes is
liable to misunderstand them if they are used as search phrases. This is a major difference
between Holmes search phrases and the search phrases you use instinctively with
standard search engines like Google, and it can take some getting used to.

<a id="use-of-the-present-active"></a>
##### 4.1.2 Use of the present active

A search phrase need not contain a verb:

***ENTITYPERSON*** (English)  
***A big dog*** (English)  
***Interest in fishing*** (English)  
***ENTITYPER*** (German)  
***Ein großer Hund*** (German)  
***Interesse am Angeln*** (German)

are all perfectly valid and potentially useful search phrases.

Where a verb is present, however, Holmes delivers the best results when the verb
is in the **present active**, as *chases* and *jagt* are in the [initial
example above](#getting-started). This gives Holmes the best chance of understanding
the relationship correctly and of matching the
widest range of document structures that share the target meaning.

<a id="generic-pronouns"></a>
##### 4.1.3 Generic pronouns

Sometimes you may only wish to extract the object of a verb. For
example, you might want to find sentences that are discussing a cat
being chased regardless of who is doing the chasing. In order to avoid a
search phrase containing a passive expression like

***A cat is chased*** (English)  
***Eine Katze wird gejagt*** (German)

you can use a **generic pronoun**. This is a word that Holmes treats
like a grammatical word in that it is not matched to documents; its sole
purpose is to help the user form a grammatically optimal search phrase
in the present active. Recognised generic pronouns are English
*something*, *somebody* and *someone* and German *jemand* (and inflected forms of *jemand*) and *etwas*:
Holmes treats them all as equivalent. Using generic pronouns,
the passive search phrases above could be re-expressed as

***Somebody chases a cat*** (English)  
***Jemand jagt eine Katze*** (German).

<a id="prepositions"></a>
##### 4.1.4 Prepositions

Experience shows that different **prepositions** are often used with the
same meaning in equivalent phrases and that this can prevent search
phrases from matching where one would intuitively expect it. For
example, the search phrases

***Somebody is at the market*** (English)  
***Jemand ist auf dem Marktplatz*** (German)

would fail to match the document phrases

*Richard was in the market* (English)  
*Richard war am Marktplatz* (German)

The best way of solving this problem is to define the prepositions in
question as synonyms in an [ontology](#ontology-based-matching).

<a id="structures-not-permitted-in-search-phrases"></a>
#### 4.2 Structures not permitted in search phrases

The following types of structures are prohibited in search phrases and
result in Python user-defined errors:

<a id="multiple-clauses"></a>
##### 4.2.1 Multiple clauses

***A dog chases a cat. A cat chases a dog*** (English)  
***Ein Hund jagt eine Katze. Eine Katze jagt einen Hund*** (German)

Each clause must be separated out into its own search phrase and
registered individually.

<a id="negation"></a>
##### 4.2.2 Negation

***A dog does not chase a cat.*** (English)  
***Ein Hund jagt keine Katze.*** (German)

Negative expressions are recognised as such in documents and the generated
matches marked as negative; allowing search phrases themselves to be
negative would overcomplicate the library without offering any benefits.

<a id="conjunction"></a>
##### 4.2.3 Conjunction

***A dog and a lion chase a cat.*** (English)  
***Ein Hund und ein Löwe jagen eine Katze.*** (German)

Wherever conjunction occurs in documents, Holmes distributes the
information among multiple matches as explained [above](#getting-started). In the
unlikely event that there should be a requirement to capture conjunction explicitly
when matching, this could be achieved by using the
[`Manager.match()` function](#manager-match-function) and looking for situations
where the document token objects are shared by multiple match objects.

<a id="lack-of-lexical-words"></a>
##### 4.2.4 Lack of lexical words

***The*** (English)  
***Der*** (German)

A search phrase cannot be processed if it does not contain any words
that can be matched to documents.

<a id="coreferring-pronouns"></a>
##### 4.2.5 Coreferring pronouns

***A dog chases a cat and he chases a mouse*** (English)  

Pronouns that corefer with nouns elsewhere in the search phrase are not permitted as this
would overcomplicate the library without offering any benefits.
Whether or not this applies to a specific pronoun depends not only on the search phrase
content, but also on whether or not [coreference resolution](#coreference-resolution)
is available for the model being used and is [switched on](#manager). Because coreference
resolution is not currently available for German, only an English example is given.

<a id="structures-strongly-discouraged-in-search-phrases"></a>
#### 4.3 Structures strongly discouraged in search phrases

The following types of structures are strongly discouraged in search
phrases:

<a id="ungrammatical-expressions"></a>
##### 4.3.1 Ungrammatical expressions

***Dog chase cat*** (English)  
***Hund jagen Katze*** (German)

Although these will sometimes work, the results will be better if search
phrases are expressed grammatically.

<a id="complex-verb-tenses"></a>
##### 4.3.2 Complex verb tenses

***A cat is chased by a dog*** (English)  
***A dog will have chased a cat*** (English)  
***Eine Katze wird durch einen Hund gejagt*** (German)  
***Ein Hund wird eine Katze gejagt haben*** (German)

Although these will sometimes work, the results will be better if verbs in
search phrases are expressed in the present active.

<a id="questions"></a>
##### 4.3.3 Questions

***Who chases the cat?*** (English)  
***Wer jagt die Katze?*** (German)

Although questions are supported in a limited sense as query phrases in the
[topic matching](#topic-matching) use case, they are not appropriate as search phrases.
Questions should be re-phrased as statements, in this case

***Something chases the cat*** (English)  
***Etwas jagt die Katze*** (German).

<a id="structures-to-be-used-with-caution-in-search-phrases"></a>
#### 4.4 Structures to be used with caution in search phrases

The following types of structures should be used with caution in search
phrases:

<a id="very-complex-structures"></a>
##### 4.4.1 Very complex structures

***A fierce dog chases a scared cat on the way to the theatre***
(English)  
***Ein kämpferischer Hund jagt eine verängstigte Katze auf dem
Weg ins Theater*** (German)

Holmes can handle any level of complexity within search phrases, but the
more complex a structure, the less likely it becomes that a document
sentence will match it. If it is really necessary to match complex relationships
with search phrases rather than with [topic matching](#topic-matching), such complex relationships
are typically better extracted by splitting the search phrase up, e.g.

***A fierce dog*** (English)  
***A scared cat*** (English)  
***A dog chases a cat*** (English)  
***Something chases something on the way to the theatre*** (English)  

***Ein kämpferischer Hund*** (German)  
***Eine verängstigte Katze*** (German)   
***Ein Hund jagt eine Katze*** (German)  
***Etwas jagt etwas auf dem Weg ins Theater*** (German)

Correlations between the resulting matches can then be established by
matching via the [`Manager.match()` function](#manager-match-function) and looking for
situations where the document token objects are shared across multiple match objects.

One important exception to this piece of advice is when
[embedding-based matching](#embedding-based-matching) is active. Because
whether or not each word in a search phrase matches then depends on whether
or not other words in the same search phrase have been matched, large, complex
search phrases can sometimes yield results that a combination of smaller,
simpler search phrases would not.

<a id="deverbal-noun-phrases"></a>
##### 4.4.2 Deverbal noun phrases

***The chasing of a cat*** (English)  
***Die Jagd einer Katze*** (German)

If an [ontology](#ontology-based-matching) is being used, it is generally better practice
to use verbal search phrases like

***Something chases a cat*** (English)  
***Etwas jagt eine Katze*** (German)

and to define the verbs and their correponding nouns as synonyms in the
ontology (*chasing* as a synonym of *chase*, *Jagd* as a synonym of
*jagen*) as this yields the same results with a smaller number of search
phrases. Holmes can match the dependency relationships within deverbal
noun phrases to the corresponding relationships within matching verb
phrases, but will not match the deverbal nouns themselves to the
corresponding verbs at word level unless they are defined as synonyms in the ontology
or happen to be identical words (the document expressions *The chase of a cat*
(English) and *Das Jagen einer Katze* (German) would match the verbal search
phrases even in the absence of an ontology).

<a id="use-cases-and-examples"></a>
### 5. Use cases and examples

<a id="chatbot"></a>
#### 5.1 Chatbot

The chatbot use case has already been introduced in [section 1.3](#getting-started):
a predefined set of search phrases is used to extract
information from phrases entered or spoken interactively by an end user, which in
this use case act as the 'documents'.

The Holmes source code ships with two examples demonstrating the chatbot
use case, one for each language, with predefined ontologies. Having
[cloned the source code and installed the Holmes library](#installation),
navigate to the `/examples` directory and type the following (Linux):

*English:*

    python3 example_chatbot_EN_insurance.py

*German:*

    python3 example_chatbot_DE_insurance.py

or click on the files in Windows Explorer (Windows).

Holmes matches syntactically distinct structures that are semantically
equivalent, i.e. that share the same meaning. In a real chatbot use
case, users will typically enter equivalent information with phrases that
are semantically distinct as well, i.e. that have different meanings.
Because the effort involved in registering a search phrase is barely
greater than the time it takes to type it in, it makes sense to register
a large number of search phrases for each relationship you are trying to
extract: essentially *all ways people have been observed to express the
information you are interested in* or *all ways you can imagine somebody
might express the information you are interested in*. To assist this,
search phrases can be registered with labels that do not need
to be unique: a label can then be used to express the relationship
an entire group of search phrases is designed to extract. Note that when many search
phrases have been defined to extract the same relationship, a single user entry
is likely to be sometimes matched by multiple search phrases. This must be handled
appropriately by the calling application.

One obvious weakness of Holmes in the chatbot setting is its sensitivity
to correct spelling and, to a lesser extent, to correct grammar.
Strategies for mitigating this weakness include:

-   Defining common misspellings as synonyms in the ontology
-   Defining specific search phrases including common misspellings
-   Putting user entry through a spellchecker before submitting it to
    Holmes
-   Explaining the importance of correct spelling and grammar to users

<a id="structural-matching"></a>
#### 5.2 Structural matching

The structural matching use case performs the same procedure as the [chatbot](#chatbot) use case,
and many of the same comments and tips apply to it. The principal difference is that pre-existing and
often lengthy documents are scanned rather than text snippets entered ad-hoc by the user.

The most useful setting for the structural matching use case is probably when a set of predefined search phrases
are matched against a stream of documents. However, the use case is most easily *demonstrated* with a
console where the user enters single search phrases that are then matched against a set of
documents that have been pre-loaded into memory. The [example scripts for topic matching](#examples-topic-matching)
perform structural matching as well as topic matching whenever the entered text forms a
[valid search phrase](#structures-not-permitted-in-search-phrases).

Some search phrases you might want to try are:

*English:*

```
A girl sings
An ENTITYPERSON goes to ENTITYGPE
A huckster sells butter
```

*German:*

```
Ein ENTITYNOUN kündigt etwas mit einer Frist von einem ENTITYNOUN
Eine Richtlinie einer ENTITYORG
```

<a id="topic-matching"></a>
#### 5.3 Topic matching

The topic matching use case matches a **query document**, or alternatively a **query phrase**
entered ad-hoc by the user, against a set of documents pre-loaded into memory. The aim is to find the passages
in the documents whose topic most closely corresponds to the topic of the query document; the output is
a ordered list of passages scored according to topic similarity.

Unlike the [structural matching](#structural-matching) use case, the topic matching use case places no
restrictions on the grammatical structures permissible within the query document. This means that query phrases
can be expressed as questions, and indeed questions may well be the most natural way for many users to formulate query
phrases. However, it is important to understand that Holmes is not a dedicated question answering system in that it
makes no attempt to retrieve content based on the meanings of question words. Instead, question words are
ignored as grammatical words; the lexical words within the question are analysed and used as a basis for
matching in the same way as if they had been contained within a statement.

<a id="examples-topic-matching"></a>
The Holmes source code ships with two examples demonstrating both the
topic matching and the structural matching use cases, one for each language. The English example
downloads and registers the collected works of Hans Christian Andersen,
while the German example downloads and registers the
Versicherungsvertragsgesetz and Versicherungsaufsichtsgesetz,
the German federal laws relating respectively to insurance contracts and
to the statutory supervision of insurance companies. You will need to be online to run both examples.
Unfortunately, the extraction of the raw website text using
[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) does not
work quite as well for the German example as for the English example,
although this does not seem to have that great an impact on the
subsequent Holmes matching. Having [cloned the source code and installed the Holmes library](#installation),
navigate to the `/examples` directory and type the following (Linux):

*English:*

```
python3 example_search_EN_literature.py
```

*German:*

```
python3 example_search_DE_law.py
```

or click on the files in Windows Explorer (Windows).

Some query phrases you might want to try are:

*English:*

```
Some people checked to see whether a traveller was a princess based on whether or not she needed a comfortable bed
A prince dreams about marrying his princess
A mermaid is sad because she is in love
```

*German:*

```
Der Versicherer darf den Vertrag fristlos kündigen, wenn der Versicherungsnehmer beim Abschluss des Vertrags die vorvertragliche Anzeigepflicht verletzt hat.
Wann darf der Versicherer Leistungen verweigern?
Wann darf der Versicherer die Prämie anpassen?
```

[Embedding-based matching](#embedding-based-matching) is switched off
as standard for the English search example (the feature is not
available for German in the first place) as the results it yielded did not seem
to be very convincing, but you may wish to change this setting at the top of the
script and observe the changes in the returned results.

The interior workings of topic matching are explained [here](#how-it-works-topic-matching).

<a id="supervised-document-classification"></a>
#### 5.4 Supervised document classification

In the supervised document classification use case, a classifier is trained with a number of documents that
are each pre-labelled with a classification. The trained classifier then assigns one or more labels to new documents
according to what each new document is about. As explained [here](#ontology-based-matching), ontologies can be
used both to enrichen the comparison of the content of the various documents and to capture implication
relationships between classification labels.

A classifier makes use of a neural network (a [multilayer perceptron](https://machinelearningcatalogue.com/algorithm/alg_perceptron.html)) whose topology can either
be determined automatically by Holmes or [specified explicitly by the user](#supervised-topic-training-basis).
With a large number of training documents, the automatically determined topology can easily exhaust the memory
available on a typical machine; if there is no opportunity to scale up the memory, this problem can be
remedied by specifying a smaller number of hidden layers or a smaller number of nodes in one or more of the layers.

A trained document classification model retains no references to its training data. This is an advantage
from a data protection viewpoint, although it
[cannot presently be guaranteed](#remove-names-from-supervised-document-classification-models) that models will
not contain individual personal or company names. It also means that models can be serialized even when
[the training documents were not serializable](#coreference-resolution).

<a id="preselection"></a>
A typical problem with the execution of many document classification use cases is that a new classification label
is added when the system is already live but that there are initially no examples of this new classification with
which to train a new model. The best course of action in such a situation is to define search phrases which
**preselect** the more obvious documents with the new classification using structural matching. Those documents that
are not preselected as having the new classification label are then passed to the existing, previously trained
classifier in the normal way. When enough documents exemplifying the new classification have accumulated in the system,
the model can be retrained and the preselection search phrases removed.

Holmes ships with an example script demonstrating supervised document classification for English with the
[BBC Documents dataset](http://mlg.ucd.ie/datasets/bbc.html). The script downloads the documents (for
this operation and for this operation alone, you will need to be online) and places them in a working directory.
When training is complete, the script saves the model to the working directory. If the model file is found
in the working directory on subsequent invocations of the script, the training phase is skipped and the script
goes straight to the testing phase. This means that if it is wished to repeat the training phase, either the model
has to be deleted from the working directory or a new working directory has to be specified to the script.

Having [cloned the source code and installed the Holmes library](#installation),
navigate to the `/examples` directory. Specify a working directory at the top of the
`example_supervised_topic_model_EN.py` file, then type `python3 example_supervised_topic_model_EN` (Linux)
or click on the script in Windows Explorer (Windows).

It is important to realise that Holmes learns to classify documents according to the words or semantic
relationships they contain, taking any structural matching ontology into account in the process. For many
classification tasks, this is exactly what is required; but there are tasks (e.g. author attribution according
to the frequency of grammatical constructions typical for each author) where it is not. For the right task,
Holmes achieves impressive results. For the BBC Documents benchmark
processed by the example script, Holmes predicts the correct classification 97.9% of the time; in 0.6% of the
remaining cases it predicts two labels and the less probable of the two is the correct one. This is
slightly better than benchmarks available online (see [here](https://github.com/suraj-deshmukh/BBC-Dataset-News-Classification)
and [here](https://cloud.google.com/blog/products/gcp/problem-solving-with-ml-automatic-document-classification))
although the difference is probably too slight to be significant, especially given that the different
training/test splits were used in each case. At the same time, however, the fact that zero and multiple
classifications are permitted outcomes make the results more usable than in the online benchmarks because it allows
the model to communicate uncertainty to the user explicitly.

The interior workings of supervised document classification are explained [here](#how-it-works-supervised-document-classification).

<a id="interfaces-intended-for-public-use"></a>
### 6 Interfaces intended for public use

<a id="manager"></a>
#### 6.1 `Manager`

``` {.python}
holmes_extractor.Manager(self, model, *, overall_similarity_threshold=1.0,
  embedding_based_matching_on_root_words=False, ontology=None,
  perform_coreference_resolution=None, debug=False)

The facade class for the Holmes library.

Args:

model -- the name of the spaCy model, e.g. 'en_core_web_lg'  
overall_similarity_threshold -- the overall similarity threshold for
  embedding-based matching. Defaults to '1.0', which deactivates
  embedding-based matching.  
embedding_based_matching_on_root_words -- determines whether or not embedding-based
  matching should be attempted on search-phrase root tokens, which has a considerable
  performance hit. Defaults to 'False'.
ontology -- an 'Ontology' object. Defaults to 'None' (no ontology).  
perform_coreference_resolution -- 'True', 'False', or 'None' if coreference resolution
  should be performed depending on whether the model supports it. Defaults to 'None'.
debug -- a boolean value specifying whether debug representations should
be outputted for parsed sentences. Defaults to 'False'.
```

``` {.python}
Manager.parse_and_register_document(self, document_text, label='')

Args:

document_text -- the raw document text.  
label -- a label for the document which must be unique. Defaults to the
  empty string, which is intended for use cases where single documents
  (user entries) are matched to predefined search phrases.
```

``` {.python}
Manager.register_parsed_document(self, document, label='')

Args:

document -- a preparsed Holmes document.  
label -- a label for the document which must be unique. Defaults to the
  empty string, which is intended for the chatbot use case where single documents
  (user entries) are matched to predefined search phrases.
```

``` {.python}
Manager.deserialize_and_register_document(self, document, label='')

Raises a 'WrongModelDeserializationError' if the model used to parse the serialized
  document does not correspond to the model with which this Manager object was created.

Args:

document -- a Holmes document serialized using the
  'serialize_document()' function.  
label -- a label for the document which must be unique. Defaults to the
  empty string, which is intended for the chatbot use case where single documents
  (user entries) are matched to predefined search phrases.
```

``` {.python}
Manager.remove_document(self, label)

Args:

label -- the label of the document to be removed.
```

``` {.python}
Manager.remove_all_documents(self)
```

``` {.python}
Manager.remove_all_search_phrases(self)
```

``` {.python}
Manager.remove_all_search_phrases_with_label(self, label)
```

``` {.python}
Manager.document_labels(self)

Returns a list of the labels of the currently registered documents.
```

<a id="manager-serialize-function"></a>
``` {.python}
Manager.serialize_document(self, label)

Returns a serialized representation of a Holmes document that can be
  persisted to a file. If 'label' is not the label of a registered document,
  'None' is returned instead. Serialization is not supported for documents
  created with neuralcoref models.

Args:

label -- the label of the document to be serialized.
```

``` {.python}
Manager.register_search_phrase(self, search_phrase_text, label=None)

Args:

search_phrase_text -- the raw search phrase text.  
label -- a label for the search phrase which need not be unique.
  If label==None, the assigned label defaults to the raw search phrase text.
```
<a id="manager-match-function"></a>
``` {.python}
Manager.match(self)

Matches the registered search phrases to the registered documents.
  Returns a list of Match objects sorted by their overall similarity
  measures in descending order. Should be called by applications wishing
  to retain references to the spaCy and Holmes information that was used
  to derive the matches.
```

``` {.python}
Manager.match_returning_dictionaries(self)

Matches the registered search phrases to the registered documents.
  Returns a list of dictionaries describing any matches, sorted by their
  overall similarity measures in descending order. Callers of this method
  do not have to manage any further dependencies on spaCy or Holmes.
```


``` {.python}
Manager.match_search_phrases_against(self, entry)

Convenience method matching the registered search phrases against a
  single document supplied to the method and returning dictionaries
  describing any matches. Any pre-existing registered documents are
  removed.
```


``` {.python}
Manager.match_documents_against(self, search_phrase)

Convenience method matching the registered documents against a single
  search phrase supplied to the method and returning dictionaries
  describing any matches. Any pre-existing registered searched phrases are
  removed.
```

``` {.python}
Manager.topic_match_documents_against(self, text_to_match, *,
  maximum_activation_distance=75, relation_score=30, single_word_score=5,
  overlapping_relation_multiplier=1.5, overlap_memory_size=10,
  maximum_activation_value=1000, sideways_match_extent=100, number_of_results=10)

Returns the results of a topic match between an entered text and the loaded documents.

Args:

text_to_match -- the text to match against the loaded documents.
maximum_activation_distance -- the number of words it takes for a pre-existing
activation to reduce to zero when the library is reading through a document.
relation_score -- the activation score added when a two-word relation is matched.
single_word_score -- the activation score added when a single word is matched.
overlapping_relation_multiplier -- the value by which the activation score is multiplied
   when two relations were matched and the matches involved a common document word.
overlap_memory_size -- the size of the memory for previous matches that is taken into
   consideration when searching for overlaps (matches are sorted according to the head
   word, and the dependent word that overlaps may be removed from the head word by
   some distance within the document text).
maximum_activation_value -- the maximum permissible activation value.
sideways_match_extent -- the maximum number of words that may be incorporated into a
   topic match either side of the word where the activation peaked.
number_of_results -- the number of topic match objects to return.
```

``` {.python}
Manager.get_supervised_topic_training_basis(self, *, classification_ontology=None,
  overlap_memory_size=10, oneshot=True, match_all_words=False, verbose=True)

Returns an object that is used to train and generate a model for the
supervised document classification use case.

Args:

classification_ontology -- an Ontology object incorporating relationships between
    classification labels, or 'None' if no such ontology is to be used.
overlap_memory_size -- how many non-word phraselet matches to the left should be
    checked for words in common with a current match.
oneshot -- whether the same word or relationship matched multiple times within a
    single document should be counted once only (value 'True') or multiple times
    (value 'False')
match_all_words -- whether all single words should be taken into account
          (value 'True') or only single words with noun tags (value 'False')          
verbose -- if 'True', information about training progress is outputted to the console.
```

``` {.python}
Manager.deserialize_supervised_topic_classifier(self, serialized_model)

Returns a classifier for the supervised document classification use case
that will use a supplied pre-trained model.

Args:

serialized_model -- the pre-trained model.
```

``` {.python}
Manager.start_chatbot_mode_console(self)

Starts a chatbot mode console enabling the matching of pre-registered
  search phrases to documents (chatbot entries) entered ad-hoc by the
  user.
```


``` {.python}
Manager.start_search_mode_console(self)

Starts a search mode console enabling the matching of pre-registered
  documents to phrases entered ad-hoc by the user. Topic matching is
  always carried out; structural matching is carried out as well whenever
  the entered phrase is a valid search phrase.
```

<a id="ontology"></a>
#### 6.2 `Ontology`

``` {.python}
holmes_extractor.Ontology(self, ontology_path,
  owl_class_type='http://www.w3.org/2002/07/owl#Class',
  owl_individual_type='http://www.w3.org/2002/07/owl#NamedIndividual',
  owl_type_link='http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
  owl_synonym_type='http://www.w3.org/2002/07/owl#equivalentClass',
  owl_hyponym_type='http://www.w3.org/2000/01/rdf-schema#subClassOf',
  symmetric_matching=False)

Loads information from an existing ontology and manages ontology
matching.

The ontology must follow the W3C OWL 2 standard. Search phrase words are
matched to hyponyms, synonyms and instances from within documents being
searched.

This class is designed for small ontologies that have been constructed
by hand for specific use cases. Where the aim is to model a large number
of semantic relationships, word embeddings are likely to offer
better results.

Matching is case-insensitive.

Args:

ontology_path -- the path from where the ontology is to be loaded. See https://github.com/RDFLib/rdflib/.  
owl_class_type -- optionally overrides the OWL 2 URL for types.  
owl_individual_type -- optionally overrides the OWL 2 URL for individuals.  
owl_type_link -- optionally overrides the RDF URL for types.  
owl_synonym_type -- optionally overrides the OWL 2 URL for synonyms.  
owl_hyponym_type -- optionally overrides the RDF URL for hyponyms.
symmetric_matching -- if 'True', means hypernym relationships are also taken into account.
```

<a id="supervised-topic-training-basis"></a>
#### 6.3 `SupervisedTopicTrainingBasis` (returned from `Manager.get_supervised_topic_training_basis`)

Holder object for training documents and their classifications from which one or more
[SupervisedTopicModelTrainer](#supervised-topic-model-trainer) objects can be derived.

``` {.python}
SupervisedTopicTrainingBasis.parse_and_register_training_document(self, text, classification, label=None)

Parses and registers a document to use for training.

Args:

text -- the document text
classification -- the classification label
label -- a label with which to identify the document in verbose training output,
  or 'None' if a random label should be assigned.
```

``` {.python}
SupervisedTopicTrainingBasis.register_training_document(self, text, classification, label=None)

Registers a pre-parsed document to use for training.

Args:

doc -- the document
classification -- the classification label
label -- a label with which to identify the document in verbose training output,
  or 'None' if a random label should be assigned.
```

``` {.python}
SupervisedTopicTrainingBasis.register_additional_classification_label(self, classification)

Register an additional classification label which no training document poessesses explicitly
  but that should be assigned to documents whose explicit labels are related to the
  additional classification label via the classification ontology.
```

``` {.python}
SupervisedTopicTrainingBasis.prepare()

Matches the phraselets derived from the training documents against the training
  documents to generate frequencies that also include combined labels, and examines the
  explicit classification labels, the additional classification labels and the
  classification ontology to derive classification implications.

  Once this method has been called, the instance no longer accepts new training documents
  or additional classification labels.
```

``` {.python}
SupervisedTopicTrainingBasis.train(self, *, minimum_occurrences=4, cv_threshold=1.0, mlp_activation='relu',
  mlp_solver='adam', mlp_learning_rate='constant', mlp_learning_rate_init=0.001,
  mlp_max_iter=200, mlp_shuffle=True, mlp_random_state=42, oneshot=True,
  overlap_memory_size=10, hidden_layer_sizes=None):

Trains a model based on the prepared state.

Args:

minimum_occurrences -- the minimum number of times a word or relationship has to
  occur in the context of the same classification for the phraselet
  to be accepted into the final model.
cv_threshold -- the minimum coefficient of variation with which a word or relationship has
  to occur across the explicit classification labels for the phraselet to be
  accepted into the final model.
mlp_* -- see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html.
oneshot -- whether the same word or relationship matched multiple times within a single
  document should be counted once only (value 'True') or multiple times (value 'False')
overlap_memory_size -- how many non-word phraselet matches to the left should be
  checked for words in common with a current match.
hidden_layer_sizes -- a list where each entry is the size of a hidden layer, or 'None'
  if the topology should be determined automatically.
```

<a id="supervised-topic-model-trainer"></a>
#### 6.4 `SupervisedTopicModelTrainer` (returned from `SupervisedTopicTrainingBasis.train()`)

Worker object used to train and generate models. This object could be removed from the public interface
(`SupervisedTopicTrainingBasis.train()` could return a `SupervisedTopicClassifier` directly) but has
been retained to facilitate testability.

``` {.python}
SupervisedTopicModelTrainer.classifier()

Returns a supervised topic classifier which contains no explicit references to the training data and that
can be serialized.
```

<a id="supervised-topic-classifier"></a>
#### 6.5 `SupervisedTopicClassifier` (returned from
`SupervisedTopicModelTrainer.classifier()` and
`Manager.deserialize_supervised_topic_classifier()`))

``` {.python}
SupervisedTopicModelTrainer.parse_and_classify(self, text)

Returns a list containing zero, one or many document classifications. Where more
than one classifications are returned, the labels are ordered by decreasing
probability.

Args:

text -- the text to parse and classify.
```

``` {.python}
SupervisedTopicModelTrainer.classify(self, doc)

Returns a list containing zero, one or many document classifications. Where more
than one classifications are returned, the labels are ordered by decreasing
probability.

Args:

doc -- the pre-parsed document to classify.
```

``` {.python}
SupervisedTopicModelTrainer.serialize_model(self)
```

``` {.python}
SupervisedTopicModelTrainer.deserialize_model(self, serialized_model)
```

<a id="match"></a>
#### 6.6 `Match` (returned from `Manager.match()`)

``` {.python}
A match between a search phrase and a document.

Properties:

search_phrase_label -- the label of the search phrase that matched.
document_label -- the label of the document that matched.
is_negated -- 'True' if this match is negated.
is_uncertain -- 'True' if this match is uncertain.
involves_coreference -- 'True' if this match was found using
  coreference resolution.
overall_similarity_measure -- the overall similarity of the match, or
  '1.0' if embedding-based matching was not involved in the match.  
word_matches -- a list of WordMatch objects.
index_within_document -- the index of the document token that matched
  the search phrase root token.
```

<a id="wordmatch"></a>
#### 6.7 `WordMatch` (returned from `Manager.match().word_matches`)

``` {.python}
A match between a searched phrase word and a document word.

Properties:

search_phrase_token -- the spaCy token from the search phrase.
search_phrase_word -- the string that matched from the search phrase.
document_token -- the spaCy token from the document.
document_word -- the string that matched from the document.
type -- 'direct', 'entity', 'embedding' or 'ontology'.
similarity_measure -- for type 'embedding', the similarity between the
  two tokens, otherwise '1.0'.
is_negated -- 'True' if this word match leads to a match of which it
  is a part being negated.
is_uncertain -- 'True' if this word match leads to a match of which it
  is a part being uncertain.
structurally_matched_document_token -- the spaCy token from the document that matched
  the parent dependencies, which may be different from *document_token* if coreference
  resolution is active.
involves_coreference -- 'True' if document_token and
  structurally_matched_document_token are different.
extracted_word -- within the coreference chain, the most specific term that corresponded to
  document_word in the ontology.
depth -- the number of hyponym relationships linking search_phrase_word and
  extracted_word, or '0' if ontology-based matching is not active. Can be negative
  if symmetric matching is active.
```

<a id="dictionary"></a>
#### 6.8 Dictionary returned from `Manager.match_returning_dictionaries()`)

``` {.python}
A text-only representation of a match between a search phrase and a
document.

Properties:

search_phrase -- the label of the search phrase.
document -- the label of the document.
index_within_document -- the character index of the match within the document.
sentences_within_document -- the raw text of the sentences within the document that matched.
negated -- 'True' if this match is negated.
uncertain -- 'True' if this match is uncertain.
involves_coreference -- 'True' if this match was found using coreference resolution.
overall_similarity_measure -- the overall similarity of the match, or
  '1.0' if embedding-based matching was not involved in the match.  
word_matches -- an array of dictionaries with the properties:

  search_phrase_word -- the string that matched from the search phrase.
  document_word -- the string that matched from the document.
  document_phrase -- the phrase headed by the word that matched from the
  document.
  match_type -- 'direct', 'entity', 'embedding' or 'ontology'.
  similarity_measure -- for type 'embedding', the similarity between the
    two tokens, otherwise '1.0'.
  involves_coreference -- 'True' if the word was matched using coreference resolution.
  extracted_word -- within the coreference chain, the most specific term that corresponded to
    document_word in the ontology.
```

<a id="topic-match"></a>
#### 6.9 `TopicMatch` (returned from `Manager.topic_match_documents_against()`))

``` {.python}
A topic match between some text and part of a document.

Properties:

document_label -- the document label.
start_index -- the start index of the topic match within the document.
end_index -- the end index of the topic match within the document.
sentences_start_index -- the start index within the document of the sentence that contains
    'start_index'.
sentences_end_index -- the end index within the document of the sentence that contains
    'end_index'.
relative_start_index -- the start index of the topic match relative to 'sentences_start_index'.
relative_end_index -- the end index of the topic match relative to 'sentences_start_index'.
score -- the similarity score of the topic match.
text -- the text between 'sentences_start_index' and 'sentences_end_index'.
```

<a id="a-note-on-the-license"></a>
### 7 A note on the license

Holmes encompasses several concepts that build on work that the author, Richard
Paul Hudson, carried out as a young graduate and for which his former
employer, [Definiens](https://www.definiens.com), has since been granted a
[U.S. patent](https://patents.google.com/patent/US8155946B2/en).
Definiens has kindly permitted the author to publish Holmes under the GNU General Public
License ("GPL"). As long as you abide by the terms of the GPL, this means you can
use the library without worrying about the patent, even if your activities take place
in the United States of America.

The GPL is often misunderstood to be a license for non-commercial use. In reality, it
certainly does permit commercial use as well in various scenarios, especially if you
are building bespoke software in an enterprise context: consult the very
comprehensive [GPL FAQ](https://www.gnu.org/licenses/gpl-faq.html) to determine whether
it is suitable for your needs.

If you wish to use Holmes in a way that is not permitted by
the GPL, please <a href="mailto:richard.hudson@msg.group">get in touch with the author</a> and
we can try and find a solution which will obviously need to involve Definiens as well if whatever
you are proposing involves the USA in any way.

<a id="information-for-developers"></a>
### 8 Information for developers

<a id="how-it-works"></a>
#### 8.1 How it works

<a id="how-it-works-structural-matching"></a>
##### 8.1.1 Structural matching

The word-level matching and the high-level operation of structural
matching between search-phrase and document subgraphs both work more or
less as one would expect. What is perhaps more in need of further
comment is the semantic analysis code subsumed in the `semantics.py`
script.

`SemanticAnalyzer` is an abstract class that is subclassed for each new
language: at present by `EnglishSemanticAnalyzer` and
`GermanSemanticAnalyzer`. At present, all functionality that is common
to the two languages is realised in the abstract parent class.
Especially because English and German are closely related languages, it
is probable that functionality will need to be moved from the abstract
parent class to specific implementing children classes when new semantic
analyzers are added for new languages.

The `HolmesDictionary` class is defined as a [spaCy extension
attribute](https://spacy.io/usage/processing-pipelines#section-custom-components-attributes)
that is accessed using the syntax `token._.holmes`. The most important
information in the dictionary is a list of `SemanticDependency` objects.
These are derived from the dependency relationships in the spaCy output
(`token.dep_`) but go through a considerable amount of processing to
make them 'less syntactic' and 'more semantic'. To give but a few
examples:

-   Where coordination occurs, dependencies are added to and from all
    siblings.
-   In passive structures, the dependencies are swapped around to capture
    the fact that the syntactic subject is the semantic object and
    vice versa.
-   Relationships are added spanning main and subordinate clauses to
    capture the fact that the syntactic subject of a main clause also
    plays a semantic role in the subordinate clause.

Some new semantic dependency labels that do not occur in spaCy outputs
as values of `token.dep_` are added for Holmes semantic dependencies.
It is important to understand that Holmes semantic dependencies are used
exclusively for matching and are therefore neither intended nor required
to form a coherent set of linguistic theoretical entities or relationships;
whatever works best for matching is assigned on an ad-hoc basis.

For each language, the `_matching_dep_dict` dictionary maps search-phrase semantic dependencies to matching
document semantic dependencies and is responsible for the [asymmetry of matching between search phrases
and documents](#general-comments).

<a id="how-it-works-topic-matching"></a>
##### 8.1.2 Topic matching

Topic matching involves the following steps:

1. The query document or query phrase is parsed and a number of **phraselets**
are extracted from it. Wherever a noun is found, a single-word phraselet is
extracted. Two-word phraselets are extracted wherever certain grammatical structures
are found. The structures that trigger two-word phraselets differ from language to language
but typically include verb-subject, verb-object and noun-adjective pairs. The relevant
phraselet structures for a given language are defined in `SemanticAnalyzer.phraselet_templates`.
Care should be taken to avoid defining phraselet templates whose head token belongs to a closed
word class e.g. prepositions. This is because such head tokens would match a large number of document
tokens, so that the resulting phraselets would give rise to a large number of potential matches:
the effort required to investigate a potential two-word phraselet match is much higher than the
effort required to match single-word phraselets.
2. The phraselets are matched against the documents to be searched and the matches held in memory. If no
matches are found, step (1) is repeated, but this time extracting single-word phraselets from
all non-grammatical words rather than just from nouns.
3. Each document is scanned from beginning to end and a psychologically inspired **activation score**
is determined for each word in each document.

  - The activation score begins at zero.
  - For as long as the activation score has a value above zero, it is reduced by 1 divided by a
  configurable number ('maximum_activation_distance'; default: 75) as each new word is read and before
  any other scores are added to it.
  - The activation score is increased by a configurable number of points ('single_word_score'; default: 5)
  at each word where a single-word phraselet was matched. However, if the previous match was also against the same
  single-word phraselet, this match is ignored unless ignoring it would cause the activation score to
  fall below the number of points awarded for a single-word phraselet match ('single_word_score'; default: 5),
  in which case the score is set to this number of points.
  - The activation score is increased by a configurable number of points ('relation_score': default: 30)
  at each word that is at the head of a match against a two-word phraselet.
  - When the same word was involved in matches against more than one two-word phraselets, this
  implies that a structure involving three or more words has been matched. For each such overlap, the activation score
  as it stands after the increases for the individual matches is multiplied by a configurable factor
  ('overlapping_relation_multiplier'; default: 1.5). Overlaps are determined by going back through the preceding
  matches within the document to check whether any of them have words in common with the current match. For
  performance reasons, it is important to place a sensible upper bound on how many preceding matches into the past to
  check ('overlap_memory_size'; default: 10).
  - An upper bound is placed on the activation score ('maximum_activation_value': default: 1000), both because
  this corresponds to psychological/neurological reality and in order to prevent one strong match from giving rise to
  erroneous weak matches in the passages following it within a document. However, in order to enable the correct
  ordering of results in situations where a number of matches have reached the maximum score, the unbounded activation
  score is maintained in parallel to the bounded activation score.
4. The most relevant passages are then determined by the highest activation score peaks within the documents, where
necessary additionally ordered by their unconstrained scores. Areas to either side of each peak up to an upper bound
('sideways_match_extent'; default: 100 words) in which the activation score is higher than the number of points
awarded for a single-word phraselet match (default: 5) are regarded as belonging to a contiguous passage around the peak.

<a id="how-it-works-supervised-document-classification"></a>
##### 8.1.3 Supervised document classification

The supervised document classification use case relies on the same phraselets as the
[topic matching use case](#how-it-works-topic-matching). Classifiers are built and trained as follows:

1. All phraselets are extracted from all training documents and registered with a structural matcher.
2. Each training document is then matched against the totality of extracted phraselets and the number of times
each phraselet is matched within training documents with each classification label is recorded. Whether multiple
occurrences within a single document are taken into account depends on the value of `oneshot`; whether
single-word phraselets are generated for all matchable words or only for those matchable words whose
part-of-speech tags match the single-word phraselet template specification depends on the value
of `match_all_words`. Wherever two phraselet matches overlap, a combined match is recorded. Combined matches are
treated in the same way as other phraselet matches in further processing. This means that effectively the
algorithm picks up one-word, two-word and three-word semantic combinations.
See [here](#improve-performance-of-supervised-document-classification-training) for a discussion of the
performance of this step.
3. The results for each phraselet are examined and phraselets are removed from the model that do not play a
statistically significant role in predicting classifications. Phraselets are removed that did not match within
the documents of any classification a minimum number of times ('minimum_occurrences'; default: 4) or where the
coefficient of variation (the standard deviation divided by the arithmetic mean) of the occurrences across the
categories is below a [threshold](#supervised-topic-training-basis) ('cv_threshold'; default: 1.0).
4. The phraselets that made it into the model are once again matched against each document. Matches against each
phraselet are used to determine the input values to a multilayer perceptron: the input nodes can either record
occurrence (binary) or match frequency (scalar) (`oneshot==True` vs. `oneshot==False` respectively). The outputs are the
category labels, including any additional labels determined via a classification ontology.  By default, the multilayer
perceptron has three hidden layers where the first hidden layer has the same number of neurons as the input layer and
the second and third layers have sizes in between the input and the output layer with an equally sized step between
each size; the user is however [free to specify any other topology](#supervised-topic-training-basis).
5. The resulting model is serializable, i.e. can be saved and reloaded. When a new document is classified, the output
is zero, one or many suggested classifications; when more than one classification is suggested, the classifications
are ordered by decreasing probabilility.

<a id="development-and-testing-guidelines"></a>
#### 8.2 Development and testing guidelines

Holmes code adheres broadly to the
[PEP-8](https://www.python.org/dev/peps/pep-0008/) standard. Because of
the complexity of some of the code, Holmes adheres to a 100-character
rather than an 80-character line width as permitted as an option there.

The complexity of what Holmes does makes development impossible without
a robust set of regression tests. These can be executed individually
with `unittest` or all at once by running the
[pytest](https://docs.pytest.org/en/latest/) utility from the Holmes
source code root directory. (Note that the Python 3 command on Linux
is `pytest-3`.) The pytest variant will only work on machines
with sufficient memory resources.

<a id="areas-for-further-development"></a>
#### 8.3 Areas for further development

<a id="incorporation-into-the-spacy-multithreading-architecture"></a>
##### 8.3.1 Incorporation into the spaCy multithreading architecture

SpaCy defines an [architecture for multithreading](https://spacy.io/usage/processing-pipelines#section-multithreading) for situations in which
large numbers of documents are to be parsed at once. At present, Holmes does not support
this architecture. For the time being, a workaround is to perform the spaCy parsing separately
and submit the spaCy documents to the `SemanticAnalyzer` class directly via the `holmes_parse()` function
so that only the Holmes parsing has to take place in a single-threaded context.

<a id="additional-languages"></a>
##### 8.3.2 Additional languages

New languages can be added to Holmes by subclassing the
`SemanticAnalyzer` class as explained in [8.1.1](#how-it-works-structural-matching). Because [some of
the linguistic features](https://spacy.io/api/annotation) returned by
spaCy are the same for all languages except English and German, the
additional effort required to add a *fourth* language may well be less
than the additional effort required to add a third language.

<a id="use-of-machine-learning-to-improve-matching"></a>
##### 8.3.3 Use of machine learning to improve matching

The sets of matching semantic dependencies captured in the
`_matching_dep_dict` dictionary for each language have been obtained on
the basis of a mixture of linguistic theoretical expectations and trial
and error. The results would probably be improved if the `_matching_dep_dict` dictionaries
could be derived using machine learning instead; as yet this has not been
attempted because of the lack of appropriate training data.

<a id="upgrade-to-latest-library-versions"></a>
##### 8.3.4 Upgrade to latest library versions

Holmes should be upgraded to use the latest versions of the spaCy and neuralcoref libraries.
<a id="remove-names-from-supervised-document-classification-models"></a>
##### 8.3.5 Remove names from supervised document classification models

An attempt should be made to remove personal data from supervised document classification models to
make them more compliant with data protection laws.

<a id="improve-performance-of-supervised-document-classification-training"></a>
##### 8.3.6 Improve the performance of supervised document classification training

As long as [embedding-based matching](#embedding-based-matching) is not active, the second step of the
[supervised document classification](#how-it-works-supervised-document-classification) procedure repeats
a considerable amount of processing from the first step. Retaining the relevant information from the first
step of the procedure would greatly improve training performance. This has not been attempted up to now
because a large number of tests would be required to prove that such performance improvements did not
have any inadvertent impacts on functionality.

<a id="explore-hyperparameters"></a>
##### 8.3.7 Explore the optimal hyperparameters for topic matching and supervised document classification

The [topic matching](#topic-matching) and [supervised document classification](#supervised-document-classification)
use cases are both configured with a number of hyperparameters that are presently set to best-guess values
derived on a purely theoretical basis. Results could be further improved by testing the use cases with a variety
of hyperparameters to learn the optimal values.
