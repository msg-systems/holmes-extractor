Holmes
======
Author: <a href="mailto:richard@explosion.ai">Richard Paul Hudson, Explosion AI</a>

-   [1. Introduction](#introduction)
    -   [1.1 The basic idea](#the-basic-idea)
    -   [1.2 Installation](#installation)
        -   [1.2.1 Prerequisites](#prerequisites)
        -   [1.2.2 Library installation](#library-installation)
        -   [1.2.3 Installing the spaCy and coreferee models](#installing-the-spacy-and-coreferee-models)
        -   [1.2.4 Comments about deploying Holmes in an
            enterprise
            environment](#comments-about-deploying-holmes-in-an-enterprise-environment)
        -   [1.2.5 Resource requirements](#resource-requirements)
    -   [1.3 Getting started](#getting-started)
-   [2. Word-level matching strategies](#word-level-matching-strategies)
    -   [2.1 Direct matching](#direct-matching)
    -   [2.2 Derivation-based matching](#derivation-based-matching)
    -   [2.3 Named-entity matching](#named-entity-matching)
    -   [2.4 Ontology-based matching](#ontology-based-matching)
    -   [2.5 Embedding-based matching](#embedding-based-matching)
    -   [2.6 Named-entity-embedding-based matching](#named-entity-embedding-based-matching)
    -   [2.7 Initial-question-word matching](#initial-question-word-matching)
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
        -   [4.3.4 Compound words](#compound-words)
    -   [4.4 Structures to be used with caution in search
        phrases](#structures-to-be-used-with-caution-in-search-phrases)
        -   [4.4.1 Very complex
            structures](#very-complex-structures)
        -   [4.4.2 Deverbal noun phrases](#deverbal-noun-phrases)
-   [5. Use cases and examples](#use-cases-and-examples)
    -   [5.1 Chatbot](#chatbot)
    -   [5.2 Structural extraction](#structural-extraction)
    -   [5.3 Topic matching](#topic-matching)
    -   [5.4 Supervised document classification](#supervised-document-classification)
-   [6 Interfaces intended for public
    use](#interfaces-intended-for-public-use)
    -   [6.1 `Manager`](#manager)
    -   [6.2 `manager.nlp`](#manager.nlp)
    -   [6.3 `Ontology`](#ontology)
    -   [6.4 `SupervisedTopicTrainingBasis`](#supervised-topic-training-basis)
    (returned from `Manager.get_supervised_topic_training_basis()`)
    -   [6.5 `SupervisedTopicModelTrainer`](#supervised-topic-model-trainer)
    (returned from `SupervisedTopicTrainingBasis.train()`)
    -   [6.6 `SupervisedTopicClassifier`](#supervised-topic-classifier)
    (returned from `SupervisedTopicModelTrainer.classifier()` and
    `Manager.deserialize_supervised_topic_classifier()`)
    -   [6.7 Dictionary returned from
        `Manager.match()`)](#dictionary)
    -   [6.8 Dictionary returned from
        `Manager.topic_match_documents_against()`](#topic-match-dictionary)
-   [7 A note on the license](#a-note-on-the-license)
-   [8 Information for developers](#information-for-developers)
    -   [8.1 How it works](#how-it-works)
        - [8.1.1 Structural matching (chatbot and structural extraction)](#how-it-works-structural-matching)
        - [8.1.2 Topic matching](#how-it-works-topic-matching)
        - [8.1.3 Supervised document classification](#how-it-works-supervised-document-classification)
    -   [8.2 Development and testing
        guidelines](#development-and-testing-guidelines)
    -   [8.3 Areas for further
        development](#areas-for-further-development)
        -   [8.3.1 Additional languages](#additional-languages)
        -   [8.3.2 Use of machine learning to improve
            matching](#use-of-machine-learning-to-improve-matching)
        -   [8.3.3 Remove names from supervised document classification models](#remove-names-from-supervised-document-classification-models)
        -   [8.3.4 Improve the performance of supervised document classification training](#improve-performance-of-supervised-document-classification-training)
        -   [8.3.5 Explore the optimal hyperparameters for topic matching and supervised document classification](#explore-hyperparameters)
    -   [8.4 Version history](#version-history)
        -   [8.4.1 Version 2.0.x](#version-20x)
        -   [8.4.2 Version 2.1.0](#version-210)
        -   [8.4.3 Version 2.2.0](#version-220)
        -   [8.4.4 Version 2.2.1](#version-221)
        -   [8.4.5 Version 3.0.0](#version-300)
        -   [8.4.6 Version 4.0.0](#version-400)

<a id="introduction"></a>
### 1. Introduction

<a id="the-basic-idea"></a>
#### 1.1 The basic idea

**Holmes** is a Python 3 library (v3.6—v3.10) running on top of
[spaCy](https://spacy.io/) (v3.1—v3.3) that supports a number of use cases
involving information extraction from English and German texts. In all use cases, the information
extraction is based on analysing the semantic relationships expressed by the component parts of
each sentence:

- In the [chatbot](#getting-started) use case, the system is configured using one or more **search phrases**.
Holmes then looks for structures whose meanings correspond to those of these search phrases within
a searched **document**, which in this case corresponds to an individual snippet of text or speech
entered by the end user. Within a match, each word with its own meaning (i.e. that does not merely fulfil a grammatical function) in the search phrase
corresponds to one or more such words in the document. Both the fact that a search phrase was matched and any structured information the search phrase extracts can be used to drive the chatbot.

- The [structural extraction](#structural-extraction) use case uses exactly the same
[structural matching](#how-it-works-structural-matching) technology as the chatbot use
case, but searching takes place with respect to a pre-existing document or documents that are typically much
longer than the snippets analysed in the chatbot use case, and the aim is to extract and store structured information. For example, a set of business articles could be searched to find all the places where one company is said to be planning to
take over a second company. The identities of the companies concerned could then be stored in a database.

- The [topic matching](#topic-matching) use case aims to find passages in a document or documents whose meaning
is close to that of another document, which takes on the role of the **query document**, or to that of a **query phrase** entered ad-hoc by the user. Holmes extracts a number of small **phraselets** from the query phrase or
query document, matches the documents being searched against each phraselet, and conflates the results to find
the most relevant passages within the documents. Because there is no strict requirement that every
word with its own meaning in the query document match a specific word or words in the searched documents, more matches are found
than in the structural extraction use case, but the matches do not contain structured information that can be
used in subsequent processing. The topic matching use case is demonstrated by [a website allowing searches within
the Harry Potter corpus (for English) and around 350 traditional stories (for German)](https://demo.holmes.prod.demos.explosion.services/).

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
structural extraction use cases that you should try and take on board.

Holmes aims to offer generalist solutions that can be used more or less out of the box with
relatively little tuning, tweaking or training and that are rapidly applicable to a wide range of use cases.
At its core lies a logical, programmed, rule-based system that describes how syntactic representations in each
language express semantic relationships. Although the supervised document classification use case does incorporate a
neural network and although the spaCy library upon which Holmes builds has itself been pre-trained using machine
learning, the essentially rule-based nature of Holmes means that the chatbot, structural extraction and topic matching use
cases can be put to use out of the box without any training and that the supervised document classification use case
typically requires relatively little training data, which is a great advantage because pre-labelled training data is
not available for many real-world problems.

Holmes has a long and complex history and we are now able to publish it under the MIT license thanks to the goodwill and openness of several companies. I, Richard Hudson, wrote the versions up to 3.0.0 while working at [msg systems](https://www.msg.group/en), a large international software consultancy based near Munich. In late 2021, I changed employers and now work for [Explosion](https://explosion.ai/), the creators of [spaCy](https://spacy.io/) and [Prodigy](https://prodi.gy/). Elements of the Holmes library are covered by a [US patent](https://patents.google.com/patent/US8155946B2/en) that I myself wrote in the early 2000s while working at a startup called Definiens that has since been acquired by [AstraZeneca](https://www.astrazeneca.com/). With the kind permission of both AstraZeneca and msg systems, I am now maintaining Holmes at Explosion and can offer it for the first time under a permissive license: anyone can now use Holmes under the terms of the MIT
license without having to worry about the patent.

<a id="installation"></a>
#### 1.2 Installation

<a id="prerequisites"></a>
##### 1.2.1 Prerequisites

If you do not already have [Python 3](https://realpython.com/installing-python/) and
[pip](https://pypi.org/project/pip/) on your machine, you will need to install them
before installing Holmes.

<a id="library-installation"></a>
##### 1.2.2 Library installation

Install Holmes using the following commands:

*Linux:*
```
pip3 install holmes-extractor
```

*Windows:*
```
pip install holmes-extractor
```

To upgrade from a previous Holmes version, issue the following commands and then
[reissue the commands to download the spaCy and coreferee models](#installing-the-spacy-and-coreferee-models) to ensure
you have the correct versions of them:

*Linux:*
```
pip3 install --upgrade holmes-extractor
```

*Windows:*
```
pip install --upgrade holmes-extractor
```

If you wish to use the examples and tests, clone the source code using

```
git clone https://github.com/explosion/holmes-extractor
```

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

<a id="installing-the-spacy-and-coreferee-models"></a>
##### 1.2.3 Installing the spaCy and coreferee models

The spaCy and coreferee libraries that Holmes builds upon require
language-specific models that have to be downloaded separately before Holmes can be used:

*Linux/English:*
```
python3 -m spacy download en_core_web_trf
python3 -m spacy download en_core_web_lg
python3 -m coreferee install en
```

*Linux/German:*
```
pip3 install spacy-lookups-data # (from spaCy 3.3 onwards)
python3 -m spacy download de_core_news_lg
python3 -m coreferee install de
```

*Windows/English:*
```
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
python -m coreferee install en
```

*Windows/German:*
```
pip install spacy-lookups-data # (from spaCy 3.3 onwards)
python -m spacy download de_core_news_lg
python -m coreferee install de
```

and if you plan to run the [regression tests](#development-and-testing-guidelines):

*Linux:*
```
python3 -m spacy download en_core_web_sm
```

*Windows:*
```
python -m spacy download en_core_web_sm
```

You specify a spaCy model for Holmes to use [when you instantiate the Manager facade class](#getting-started). `en_core_web_trf` and `de_core_web_lg` are the models that have been found to yield the best results for English and German respectively. Because `en_core_web_trf` does not have its own word vectors, but Holmes requires word vectors for [embedding-based-matching](#embedding-based-matching), the `en_core_web_lg` model is loaded as a vector source whenever `en_core_web_trf` is specified to the Manager class as the main model.

The `en_core_web_trf` model requires sufficiently more resources than the other models; in a siutation where resources are scarce, it may be a sensible compromise to use `en_core_web_lg` as the main model instead.

<a id="comments-about-deploying-holmes-in-an-enterprise-environment"></a>
##### 1.2.4 Comments about deploying Holmes in an enterprise environment

The best way of integrating Holmes into a non-Python environment is to
wrap it as a RESTful HTTP service and to deploy it as a
microservice. See [here](https://github.com/explosion/holmes-extractor/blob/master/examples/example_search_EN_literature.py) for an example.

<a id="resource-requirements"></a>
##### 1.2.5 Resource requirements

Because Holmes performs complex, intelligent analysis, it is inevitable that it requires more hardware resources than more traditional search frameworks. The use cases that involve loading documents — [structural extraction](#structural-extraction) and [topic matching](#topic-matching) — are most immediately applicable to large but not massive corpora (e.g. all the documents belonging to a certain organisation, all the patents on a certain topic, all the books by a certain author). For cost reasons, Holmes would not be an appropriate tool with which to analyse the content of the entire internet!

That said, Holmes is both vertically and horizontally scalable. With sufficient hardware, both these use cases can be applied to an essentially unlimited number of documents by running Holmes on multiple machines, processing a different set of documents on each one and conflating the results. Note that this strategy is already employed to distribute matching amongst multiple cores on a single machine: the [Manager](#manager) class starts a number of worker processes and distributes registered documents between them.

Holmes holds loaded documents in memory, which ties in with its intended use with large but not massive corpora. The performance of document loading, [structural extraction](#structural-extraction) and [topic matching](#topic-matching) all degrade heavily if the operating system has to swap memory pages to secondary storage, because Holmes can require memory from a variety of pages to be addressed when processing a single sentence. This means it is important to supply enough RAM on each machine to hold all loaded documents.

Please note the [above comments](#installing-the-spacy-and-coreferee-models) about the relative resource requirements of the different models.

<a id="getting-started"></a>
#### 1.3 Getting started

The easiest use case with which to get a quick basic idea of how Holmes works is the **chatbot** use case.

Here one or more search phrases are defined to Holmes in advance, and the
searched documents are short sentences or paragraphs typed in
interactively by an end user. In a real-life setting, the extracted
information would be used to
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
holmes_manager = holmes.Manager(model='en_core_web_lg', number_of_workers=1)
holmes_manager.register_search_phrase('A big dog chases a cat')
holmes_manager.start_chatbot_mode_console()
```

*German:*

```
import holmes_extractor as holmes
holmes_manager = holmes.Manager(model='de_core_news_lg', number_of_workers=1)
holmes_manager.register_search_phrase('Ein großer Hund jagt eine Katze')
holmes_manager.start_chatbot_mode_console()
```

If you now enter a sentence that corresponds to the search phrase, the
console will display a match:

*English:*

```
Ready for input

A big dog chased a cat


Matched search phrase with text 'A big dog chases a cat':
'big'->'big' (Matches BIG directly); 'A big dog'->'dog' (Matches DOG directly); 'chased'->'chase' (Matches CHASE directly); 'a cat'->'cat' (Matches CAT directly)
```

*German:*

```
Ready for input

Ein großer Hund jagte eine Katze


Matched search phrase 'Ein großer Hund jagt eine Katze':
'großer'->'groß' (Matches GROSS directly); 'Ein großer Hund'->'hund' (Matches HUND directly); 'jagte'->'jagen' (Matches JAGEN directly); 'eine Katze'->'katze' (Matches KATZE directly)
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
The cat the big dog chased was scared
The big dog chasing the cat was a problem
There was a big dog that was chasing a cat
The cat chase by the big dog
There was a big dog and it was chasing a cat.
I saw a big dog. My cat was afraid of being chased by the dog.
There was a big dog. His name was Fido. He was chasing my cat.
A dog appeared. It was chasing a cat. It was very big.
The cat sneaked back into our lounge because a big dog had been chasing her.
Our big dog was excited because he had been chasing a cat.
```

*German:*

```
Der große Hund hat die Katze ständig gejagt
Der große Hund, der müde war, jagte die Katze
Die Katze wurde vom großen Hund gejagt
Die Katze wurde immer wieder durch den großen Hund gejagt
Der große Hund wollte die Katze jagen
Der große Hund entschied sich, die Katze zu jagen
Die Katze, die der große Hund gejagt hatte, hatte Angst
Dass der große Hund die Katze jagte, war ein Problem
Es gab einen großen Hund, der eine Katze jagte
Die Katzenjagd durch den großen Hund
Es gab einmal einen großen Hund, und er jagte eine Katze
Es gab einen großen Hund. Er hieß Fido. Er jagte meine Katze
Es erschien ein Hund. Er jagte eine Katze. Er war sehr groß.
Die Katze schlich sich in unser Wohnzimmer zurück, weil ein großer Hund sie draußen gejagt hatte
Unser großer Hund war aufgeregt, weil er eine Katze gejagt hatte
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
The big dog the cat chased was scared
Our big dog was upset because he had been chased by a cat.
The dog chase of the big cat
```

*German:*

```
Der Hund jagte eine große Katze
Die Katze jagte den großen Hund
Der große Hund und die Katze jagten
Der große Hund jagte eine Maus aber die Katze war müde
Der große Hund wurde ständig von der Katze gejagt
Der große Hund entschloss sich, von der Katze gejagt zu werden
Die Hundejagd durch den große Katze
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


Matched search phrase with text 'An ENTITYPERSON goes into town'; negated; uncertain; involves coreference:
'Richard Hudson'->'ENTITYPERSON' (Has an entity label matching ENTITYPERSON); 'go'->'go' (Matches GO directly); 'into'->'into' (Matches INTO directly); 'town'->'town' (Matches TOWN directly)

Matched search phrase with text 'An ENTITYPERSON goes into town'; negated; uncertain; involves coreference:
'John Doe'->'ENTITYPERSON' (Has an entity label matching ENTITYPERSON); 'go'->'go' (Matches GO directly); 'into'->'into' (Matches INTO directly); 'town'->'town' (Matches TOWN directly)
```

*German:*

```
Ready for input

Letzte Woche sah ich Richard Hudson und Max Mustermann. Sie wollten nicht mehr in die Stadt gehen.


Matched search phrase with text 'Ein ENTITYPER geht in die Stadt'; negated; uncertain; involves coreference:
'Richard Hudson'->'ENTITYPER' (Has an entity label matching ENTITYPER); 'gehen'->'gehen' (Matches GEHEN directly); 'in'->'in' (Matches IN directly); 'die Stadt'->'stadt' (Matches STADT directly)

Matched search phrase with text 'Ein ENTITYPER geht in die Stadt'; negated; uncertain; involves coreference:
'Max Mustermann'->'ENTITYPER' (Has an entity label matching ENTITYPER); 'gehen'->'gehen' (Matches GEHEN directly); 'in'->'in' (Matches IN directly); 'die Stadt'->'stadt' (Matches STADT directly)
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

The following strategies are implemented with 
[one Python module per strategy](https://github.com/explosion/holmes-extractor/tree/master/holmes_extractor/word_matching). 
Although the standard library does not support adding bespoke strategies via the [Manager](#manager)
class, it would be relatively easy for anyone with Python programming skills to
change the code to enable this.

<a id="direct-matching"></a>
#### 2.1 Direct matching (`word_match.type=='direct'`)

Direct matching between search phrase words and document words is always
active. The strategy relies mainly on matching stem forms of words,
e.g. matching English *buy* and *child* to *bought* and *children*,
German *steigen* and *Kind* to *stieg* and *Kinder*. However, in order to
increase the chance of direct matching working when the parser delivers an
incorrect stem form for a word, the raw-text forms of both search-phrase and
document words are also taken into consideration during direct matching.

<a id="derivation-based-matching"></a>
#### 2.2 Derivation-based matching (`word_match.type=='derivation'`)

Derivation-based matching involves distinct but related words that typically
belong to different word classes, e.g. English *assess* and *assessment*,
German *jagen* and *Jagd*. It is active by default but can be switched off using
the `analyze_derivational_morphology` parameter, which is set when instantiating the [Manager](#manager) class.

<a id="named-entity-matching"></a>
#### 2.3 Named-entity matching (`word_match.type=='entity'`)

Named-entity matching is activated by inserting a special named-entity
identifier at the desired point in a search phrase in place of a noun,
e.g.

***An ENTITYPERSON goes into town*** (English)  
***Ein ENTITYPER geht in die Stadt*** (German).

The supported named-entity identifiers depend directly on the named-entity information supplied
by the spaCy models for each language (descriptions copied from an earlier version of the spaCy
documentation):

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
and that this specific noun phrase is extracted and available for further processing. `ENTITYNOUN` is not supported within the topic matching use case.

<a id="ontology-based-matching"></a>
#### 2.4 Ontology-based matching (`word_match.type=='ontology'`)

An ontology enables the user to define relationships between words that
are then taken into account when matching documents to search phrases.
The three relevant relationship types are *hyponyms* (something is a
subtype of something), *synonyms* (something means the same as
something) and *named individuals* (something is a specific instance of
something). The three relationship types are exemplified in Figure 1:

![Figure 1](https://github.com/explosion/holmes-extractor/blob/master/docs/ontology_example.png)

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
is appropriate for the [chatbot](#chatbot) and [structural extraction](#structural-extraction) use cases,
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

English phrasal verbs like *eat up* and German separable verbs like *aufessen*  must be defined as single items within ontologies. When Holmes is analysing a text and
comes across such a verb, the main verb and the particle are conflated into a single
logical word that can then be matched via an ontology. This means that *eat up* within
a text would match the subtree of *eat up* within the ontology but not the subtree of
*eat* within the ontology.

If [derivation-based matching](#derivation-based-matching) is active, it is taken into account
on both sides of a potential ontology-based match. For example, if *alter* and *amend* are
defined as synonyms in an ontology, *alteration* and *amendment* would also match each other.

In situations where finding relevant sentences is more important than
ensuring the logical correspondence of document matches to search phrases,
it may make sense to specify **symmetric matching** when defining the ontology.
Symmetric matching is recommended for the [topic matching](#topic-matching) use case, but
is unlikely to be appropriate for the [chatbot](#chatbot) or [structural extraction](#structural-extraction) use cases.
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
to be classified. In the example ontology shown above, all words in the ontology would be replaced with *animal*; in an extreme case with a WordNet-style ontology, all nouns would end up being replaced with
 *thing*, which is clearly not a desirable outcome!

- The **classification** ontology is used to capture relationships between classification labels: that a document
has a certain classification implies it also has any classifications to whose subtree that classification belongs.
Synonyms should be used sparingly if at all in classification ontologies because they add to the complexity of the
neural network without adding any value; and although it is technically possible to set up a classification
ontology to use symmetric matching, there is no sensible reason for doing so. Note that a label within the
classification ontology that is not directly defined as the label of any training document
[has to be registered specifically](#supervised-topic-training-basis) using the
`SupervisedTopicTrainingBasis.register_additional_classification_label()` method if it is to be taken into
account when training the classifier.

<a id="embedding-based-matching"></a>
#### 2.5 Embedding-based matching (`word_match.type=='embedding'`)

spaCy offers **word embeddings**:
machine-learning-generated numerical vector representations of words
that capture the contexts in which each word
tends to occur. Two words with similar meaning tend to emerge with word
embeddings that are close to each other, and spaCy can measure the
**cosine similarity** between any two words' embeddings expressed as a decimal
between 0.0 (no similarity) and 1.0 (the same word). Because *dog* and
*cat* tend to appear in similar contexts, they have a similarity of
0.80; *dog* and *horse* have less in common and have a similarity of
0.62; and *dog* and *iron* have a similarity of only 0.25. Embedding-based matching
is only activated for nouns, adjectives and adverbs because the results have been found to be
unsatisfactory with other word classes.

It is important to understand that the fact that two words have similar
embeddings does not imply the same sort of logical relationship between
the two as when [ontology-based matching](#ontology-based-matching) is used: for example, the
fact that *dog* and *cat* have similar embeddings means neither that a
dog is a type of cat nor that a cat is a type of dog. Whether or not
embedding-based matching is nonetheless an appropriate choice depends on
the functional use case.

For the [chatbot](#chatbot), [structural extraction](#structural-extraction) and [supervised document classification](#supervised-document-classification) use cases, Holmes makes use of word-
embedding-based similarities using a `overall_similarity_threshold` parameter defined globally on
the [Manager](#manager) class. A match is detected between a
search phrase and a structure within a document whenever the geometric
mean of the similarities between the individual corresponding word pairs
is greater than this threshold. The intuition behind this technique is
that where a search phrase with e.g. six lexical words has matched a
document structure where five of these words match exactly and only one
corresponds via an embedding, the similarity that should be required to match this sixth word is
less than when only three of the words matched exactly and two of the other words also correspond
via embeddings.

Matching a search phrase to a document begins by finding words
in the document that match the word at the root (syntactic head) of the
search phrase. Holmes then investigates the structure around each of
these matched document words to check whether the document structure matches
the search phrase structure in its entirity.
The document words that match the search phrase root word are normally found
using an index. However, if embeddings have to be taken into account when
finding document words that match a search phrase root word, **every** word in
**every** document with a valid word class has to be compared for similarity to that
search phrase root word. This has a very noticeable performance hit that renders all use cases
except the [chatbot](#chatbot) use case essentially unusable.

To avoid the typically unnecessary performance hit that results from embedding-based matching
of search phrase root words, it is controlled separately from embedding-based matching in general
using the `embedding_based_matching_on_root_words` parameter, which is set when instantiating the
[Manager](#manager) class. You are advised to keep this setting switched off (value `False`) for most use cases.

Neither the `overall_similarity_threshold` nor the `embedding_based_matching_on_root_words` parameter has any effect on the [topic matching](#topic-matching) use case. Here word-level embedding similarity thresholds are set using the `word_embedding_match_threshold` and  `initial_question_word_embedding_match_threshold` parameters when calling the [`topic_match_documents_against` function on the Manager class](#manager-topic-match-function).

<a id="named-entity-embedding-based-matching"></a>
#### 2.6 Named-entity-embedding-based matching (`word_match.type=='entity_embedding'`)

A named-entity-embedding based match obtains between a searched-document word that has a certain entity label and a search phrase or query document word whose embedding is sufficiently similar to the underlying meaning of that entity label, e.g. the word *individual* in a search phrase has a similar word embedding to the underlying meaning of the *PERSON* entity label. Note that named-entity-embedding-based matching is never active on root words regardless of the `embedding_based_matching_on_root_words` setting.

<a id="initial-question-word-matching"></a>
#### 2.7 Initial-question-word matching (`word_match.type=='question'`)

Initial-question-word matching is only active during [topic matching](#topic-matching). Initial question words in query phrases match entities in the searched documents that represent potential answers to the question, e.g. when comparing the query phrase *When did Peter have breakfast* to the searched-document phrase *Peter had breakfast at 8 a.m.*, the question word *When* would match the temporal adverbial phrase *at 8 a.m.*.

Initial-question-word matching is switched on and off using the `initial_question_word_behaviour` parameter when calling the [`topic_match_documents_against` function on the Manager class](#manager-topic-match-function). It is only likely to be useful when topic matching is being performed in an interactive setting where the user enters short query phrases, as opposed to when it is being used to find documents on a similar topic to an pre-existing query document: initial question words are only processed at the beginning of the first sentence of the query phrase or query document.

Linguistically speaking, if a query phrase consists of a complex question with several elements dependent on the main verb, a finding in a searched document is only an 'answer' if contains matches to all these elements. Because recall is typically more important than precision when performing topic matching with interactive query phrases, however, Holmes will match an initial question word to a searched-document phrase wherever they correspond semantically (e.g. wherever *when* corresponds to a temporal adverbial phrase) and each depend on verbs that themselves match at the word level. One possible strategy to filter out 'incomplete answers' would be to calculate the maximum possible score for a query phrase and reject topic matches that score below a threshold scaled to this maximum.

<a id="coreference-resolution"></a>
### 3. Coreference resolution

Before Holmes analyses a searched document or query document, coreference resolution is performed using the [Coreferee](https://github.com/explosion/coreferee)
library running on top of spaCy.  This means that situations are recognised where pronouns and nouns that are located near one another within a text refer to the same entities. The information from one mention can then be applied to the analysis of further mentions:

I saw a *big dog*. *It* was chasing a cat.   
I saw a *big dog*. *The dog* was chasing a cat.

Coreferee also detects situations where a noun refers back to a named entity:

We discussed *AstraZeneca*. *The company* had given us permission to publish this library under the MIT license.

If this example were to match the search phrase ***A company gives permission to publish something***, the
coreference information that the company under discussion is AstraZeneca is clearly
relevant and worth extracting in addition to the word(s) directly matched to the search
phrase. Such information is captured in the [word_match.extracted_word](#dictionary) field.

<a id="writing-effective-search-phrases"></a>
### 4. Writing effective search phrases

<a id="general-comments"></a>
#### 4.1 General comments

The concept of search phrases has [already been introduced](#getting-started) and is relevant to the
chatbot use case, the structural extraction use case and to [preselection](#preselection) within the supervised
document classification use case.

**It is crucial to understand that the tips and limitations set out in Section 4 do not apply in any way to query phrases in topic matching. If you are using Holmes for topic matching only, you can completely ignore this section!**

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
***Ein Hund jagt eine Katze und er jagt eine Maus*** (German)

Pronouns that corefer with nouns elsewhere in the search phrase are not permitted as this
would overcomplicate the library without offering any benefits.

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

Although questions are supported as query phrases in the
[topic matching](#topic-matching) use case, they are not appropriate as search phrases.
Questions should be re-phrased as statements, in this case

***Something chases the cat*** (English)  
***Etwas jagt die Katze*** (German).

<a id="compound-words"></a>
##### 4.3.4 Compound words (relates to German only)

***Informationsextraktion*** (German)  
***Ein Stadtmittetreffen*** (German)

The internal structure of German compound words is analysed within searched documents as well as
within query phrases in the [topic matching](#topic-matching) use case, but not within search
phrases. In search phrases, compound words should be reexpressed as genitive constructions even in cases
where this does not strictly capture their meaning:

***Extraktion der Information*** (German)  
***Ein Treffen der Stadtmitte*** (German)

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
sentence will match it. If it is really necessary to match such complex relationships
with search phrases rather than with [topic matching](#topic-matching), they are typically better extracted by splitting the search phrase up, e.g.

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

One possible exception to this piece of advice is when
[embedding-based matching](#embedding-based-matching) is active. Because
whether or not each word in a search phrase matches then depends on whether
or not other words in the same search phrase have been matched, large, complex
search phrases can sometimes yield results that a combination of smaller,
simpler search phrases would not.

<a id="deverbal-noun-phrases"></a>
##### 4.4.2 Deverbal noun phrases

***The chasing of a cat*** (English)  
***Die Jagd einer Katze*** (German)

These will often work, but it is generally better practice
to use verbal search phrases like

***Something chases a cat*** (English)  
***Etwas jagt eine Katze*** (German)

and to allow the corresponding nominal phrases to be matched via [derivation-based matching](#derivation-based-matching).

<a id="use-cases-and-examples"></a>
### 5. Use cases and examples

<a id="chatbot"></a>
#### 5.1 Chatbot

The chatbot use case has [already been introduced](#getting-started):
a predefined set of search phrases is used to extract
information from phrases entered interactively by an end user, which in
this use case act as the documents.

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

<a id="structural-extraction"></a>
#### 5.2 Structural extraction

The structural extraction use case uses [structural matching](#how-it-works-structural-matching) in the same way as the [chatbot](#chatbot) use case,
and many of the same comments and tips apply to it. The principal differences are that pre-existing and
often lengthy documents are scanned rather than text snippets entered ad-hoc by the user, and that the
returned match objects are not used to
drive a dialog flow; they are examined solely to extract and store structured information.

Code for performing structural extraction would typically perform the following tasks:

-   Initialize the Holmes manager object.
-   Call `Manager.register_search_phrase()` several times to define a number of search phrases specifying the information to be extracted.
-   Call `Manager.parse_and_register_document()` several times to load a number of documents within which to search.
-   Call `Manager.match()` to perform the matching.
-   Query the returned match objects to obtain the extracted information and store it in a database.

<a id="topic-matching"></a>
#### 5.3 Topic matching

The topic matching use case matches a **query document**, or alternatively a **query phrase**
entered ad-hoc by the user, against a set of documents pre-loaded into memory. The aim is to find the passages
in the documents whose topic most closely corresponds to the topic of the query document; the output is
a ordered list of passages scored according to topic similarity. Additionally, if a query phrase contains an [initial question word](#initial-question-word-matching), the output will contain potential answers to the question.

Topic matching queries may contain [generic pronouns](#generic-pronouns) and
[named-entity identifiers](#named-entity-matching) just like search phrases, although the `ENTITYNOUN`
token is not supported. However, an important difference from
search phrases is that the topic matching use case places no
restrictions on the grammatical structures permissible within the query document.

The Holmes source code ships with three examples demonstrating the topic matching use case with an English literature
corpus, a German literature corpus and a German legal corpus respectively. The two literature examples are hosted at
the [Holmes demonstration website](https://demo.holmes.prod.demos.explosion.services/), although users are encouraged to run [the scripts](https://github.com/explosion/holmes-extractor/blob/master/examples/)
locally as well to get a feel for how they work. The German law example starts a simple interactive console and its [script](https://github.com/explosion/holmes-extractor/blob/master/examples/example_search_DE_law.py) contains some example queries as comments.

Topic matching uses a variety of strategies to find text passages that are relevant to the query. These include
resource-hungry procedures like investigating semantic relationships and comparing embeddings. Because applying these
across the board would prevent topic matching from scaling, Holmes only attempts them for specific areas of the text
that less resource-intensive strategies have already marked as looking promising. This and the other interior workings
of topic matching are explained [here](#how-it-works-topic-matching).

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
not contain individual personal or company names.

<a id="preselection"></a>
A typical problem with the execution of many document classification use cases is that a new classification label
is added when the system is already live but that there are initially no examples of this new classification with
which to train a new model. The best course of action in such a situation is to define search phrases which
**preselect** the more obvious documents with the new classification using structural matching. Those documents that
are not preselected as having the new classification label are then passed to the existing, previously trained
classifier in the normal way. When enough documents exemplifying the new classification have accumulated in the system,
the model can be retrained and the preselection search phrases removed.

Holmes ships with an example [script](https://github.com/explosion/holmes-extractor/blob/master/examples/example_supervised_topic_model_EN.py) demonstrating supervised document classification for English with the
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
processed by the example script, Holmes performs slightly better than benchmarks available online
(see e.g. [here](https://github.com/suraj-deshmukh/BBC-Dataset-News-Classification))
although the difference is probably too slight to be significant, especially given that the different
training/test splits were used in each case: Holmes has been observed to learn models that predict the
correct result between 96.9% and 98.7% of the time. The range is explained by the fact that the behaviour
of the neural network is not fully deterministic.

The interior workings of supervised document classification are explained [here](#how-it-works-supervised-document-classification).

<a id="interfaces-intended-for-public-use"></a>
### 6 Interfaces intended for public use

<a id="manager"></a>
#### 6.1 `Manager`

``` {.python}
holmes_extractor.Manager(self, model, *, overall_similarity_threshold=1.0,
  embedding_based_matching_on_root_words=False, ontology=None,
  analyze_derivational_morphology=True, perform_coreference_resolution=None,
  number_of_workers=None, verbose=False)

The facade class for the Holmes library.

Parameters:

model -- the name of the spaCy model, e.g. *en_core_web_trf*
overall_similarity_threshold -- the overall similarity threshold for embedding-based
  matching. Defaults to *1.0*, which deactivates embedding-based matching. Note that this
  parameter is not relevant for topic matching, where the thresholds for embedding-based
  matching are set on the call to *topic_match_documents_against*.
embedding_based_matching_on_root_words -- determines whether or not embedding-based
  matching should be attempted on search-phrase root tokens, which has a considerable
  performance hit. Defaults to *False*. Note that this parameter is not relevant for topic
  matching.
ontology -- an *Ontology* object. Defaults to *None* (no ontology).
analyze_derivational_morphology -- *True* if matching should be attempted between different
  words from the same word family. Defaults to *True*.
perform_coreference_resolution -- *True* if coreference resolution should be taken into account
  when matching. Defaults to *True*.
use_reverse_dependency_matching -- *True* if appropriate dependencies in documents can be
  matched to dependencies in search phrases where the two dependencies point in opposite
  directions. Defaults to *True*.
number_of_workers -- the number of worker processes to use, or *None* if the number of worker
  processes should depend on the number of available cores. Defaults to *None*
verbose -- a boolean value specifying whether multiprocessing messages should be outputted to
  the console. Defaults to *False*
```

``` {.python}
Manager.register_serialized_document(self, serialized_document:bytes, label:str="") -> None

Parameters:

document -- a preparsed Holmes document.
label -- a label for the document which must be unique. Defaults to the empty string,
    which is intended for use cases involving single documents (typically user entries).
```

<a id="manager-register-serialized-documents-function"></a>
``` {.python}
Manager.register_serialized_documents(self, document_dictionary:dict[str, bytes]) -> None

Note that this function is the most efficient way of loading documents.

Parameters:

document_dictionary -- a dictionary from labels to serialized documents.
```

``` {.python}
Manager.parse_and_register_document(self, document_text:str, label:str='') -> None

Parameters:

document_text -- the raw document text.
label -- a label for the document which must be unique. Defaults to the empty string,
    which is intended for use cases involving single documents (typically user entries).
```

``` {.python}
Manager.remove_document(self, label:str) -> None
```

``` {.python}
Manager.remove_all_documents(self, labels_starting:str=None) -> None

Parameters:

labels_starting -- a string starting the labels of documents to be removed,
    or 'None' if all documents are to be removed.
```

``` {.python}
Manager.list_document_labels(self) -> List[str]

Returns a list of the labels of the currently registered documents.
```

``` {.python}
Manager.serialize_document(self, label:str) -> Optional[bytes]

Returns a serialized representation of a Holmes document that can be
  persisted to a file. If 'label' is not the label of a registered document,
  'None' is returned instead.

Parameters:

label -- the label of the document to be serialized.
```

``` {.python}
Manager.get_document(self, label:str='') -> Optional[Doc]

Returns a Holmes document. If *label* is not the label of a registered document, *None*
  is returned instead.

Parameters:

label -- the label of the document to be serialized.
```

``` {.python}
Manager.debug_document(self, label:str='') -> None

Outputs a debug representation for a loaded document.

Parameters:

label -- the label of the document to be serialized.
```

``` {.python}
Manager.register_search_phrase(self, search_phrase_text:str, label:str=None) -> SearchPhrase

Registers and returns a new search phrase.

Parameters:

search_phrase_text -- the raw search phrase text.  
label -- a label for the search phrase, which need not be unique.
  If label==None, the assigned label defaults to the raw search phrase text.
```

``` {.python}
Manager.remove_all_search_phrases_with_label(self, label:str) -> None
```

```
Manager.remove_all_search_phrases(self) -> None
```

```
Manager.list_search_phrase_labels(self) -> List[str]
```

<a id="manager-match-function"></a>
``` {.python}
Manager.match(self, search_phrase_text:str=None, document_text:str=None) -> List[Dict]

Matches search phrases to documents and returns the result as match dictionaries.

Parameters:

search_phrase_text -- a text from which to generate a search phrase, or 'None' if the
    preloaded search phrases should be used for matching.
document_text -- a text from which to generate a document, or 'None' if the preloaded
    documents should be used for matching.
```

<a id="manager-topic-match-function"></a>
``` {.python}
topic_match_documents_against(self, text_to_match:str, *,
    use_frequency_factor:bool=True,
    maximum_activation_distance:int=75,
    word_embedding_match_threshold:float=0.8,
    initial_question_word_embedding_match_threshold:float=0.7,
    relation_score:int=300,
    reverse_only_relation_score:int=200,
    single_word_score:int=50,
    single_word_any_tag_score:int=20,
    initial_question_word_answer_score:int=600,
    initial_question_word_behaviour:str='process',
    different_match_cutoff_score:int=15,
    overlapping_relation_multiplier:float=1.5,
    embedding_penalty:float=0.6,
    ontology_penalty:float=0.9,
    relation_matching_frequency_threshold:float=0.25,
    embedding_matching_frequency_threshold:float=0.5,
    sideways_match_extent:int=100,
    only_one_result_per_document:bool=False,
    number_of_results:int=10,
    document_label_filter:str=None,
    tied_result_quotient:float=0.9) -> List[Dict]:

Returns a list of dictionaries representing the results of a topic match between an entered text
and the loaded documents.

Properties:

text_to_match -- the text to match against the loaded documents.
use_frequency_factor -- *True* if scores should be multiplied by a factor between 0 and 1
  expressing how rare the words matching each phraselet are in the corpus. Note that,
  even if this parameter is set to *False*, the factors are still calculated as they are 
  required for determining which relation and embedding matches should be attempted.
maximum_activation_distance -- the number of words it takes for a previous phraselet
  activation to reduce to zero when the library is reading through a document.
word_embedding_match_threshold -- the cosine similarity above which two words match where
  the search phrase word does not govern an interrogative pronoun.
initial_question_word_embedding_match_threshold -- the cosine similarity above which two
  words match where the search phrase word governs an interrogative pronoun.
relation_score -- the activation score added when a normal two-word relation is matched.
reverse_only_relation_score -- the activation score added when a two-word relation
  is matched using a search phrase that can only be reverse-matched.
single_word_score -- the activation score added when a single noun is matched.
single_word_any_tag_score -- the activation score added when a single word is matched
  that is not a noun.
initial_question_word_answer_score -- the activation score added when a question word is
  matched to an potential answer phrase.
initial_question_word_behaviour -- 'process' if a question word in the sentence
  constituent at the beginning of *text_to_match* is to be matched to document phrases
  that answer it and to matching question words; 'exclusive' if only topic matches that 
  answer questions are to be permitted; 'ignore' if question words are to be ignored.
different_match_cutoff_score -- the activation threshold under which topic matches are
  separated from one another. Note that the default value will probably be too low if
  *use_frequency_factor* is set to *False*.
overlapping_relation_multiplier -- the value by which the activation score is multiplied
  when two relations were matched and the matches involved a common document word.
embedding_penalty -- a value between 0 and 1 with which scores are multiplied when the
  match involved an embedding. The result is additionally multiplied by the overall
  similarity measure of the match.
ontology_penalty -- a value between 0 and 1 with which scores are multiplied for each
  word match within a match that involved the ontology. For each such word match,
  the score is multiplied by the value (abs(depth) + 1) times, so that the penalty is
  higher for hyponyms and hypernyms than for synonyms and increases with the
  depth distance.
relation_matching_frequency_threshold -- the frequency threshold above which single
  word matches are used as the basis for attempting relation matches.
embedding_matching_frequency_threshold -- the frequency threshold above which single
  word matches are used as the basis for attempting relation matches with
  embedding-based matching on the second word.
sideways_match_extent -- the maximum number of words that may be incorporated into a
  topic match either side of the word where the activation peaked.
only_one_result_per_document -- if 'True', prevents multiple results from being returned
  for the same document.
number_of_results -- the number of topic match objects to return.
document_label_filter -- optionally, a string with which document labels must start to
  be considered for inclusion in the results.
tied_result_quotient -- the quotient between a result and following results above which
  the results are interpreted as tied.
```

``` {.python}
Manager.get_supervised_topic_training_basis(self, *, classification_ontology:Ontology=None,
  overlap_memory_size:int=10, oneshot:bool=True, match_all_words:bool=False,
  verbose:bool=True) -> SupervisedTopicTrainingBasis:

Returns an object that is used to train and generate a model for the
supervised document classification use case.

Parameters:

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
Manager.deserialize_supervised_topic_classifier(self,
  serialized_model:bytes, verbose:bool=False) -> SupervisedTopicClassifier:

Returns a classifier for the supervised document classification use case
that will use a supplied pre-trained model.

Parameters:

serialized_model -- the pre-trained model as returned from `SupervisedTopicClassifier.serialize_model()`.
verbose -- if 'True', information about matching is outputted to the console.
```

``` {.python}
Manager.start_chatbot_mode_console(self)

Starts a chatbot mode console enabling the matching of pre-registered
  search phrases to documents (chatbot entries) entered ad-hoc by the
  user.
```

``` {.python}
Manager.start_structural_search_mode_console(self)

Starts a structural extraction mode console enabling the matching of pre-registered
  documents to search phrases entered ad-hoc by the user.
```

``` {.python}
Manager.start_topic_matching_search_mode_console(self,    
  only_one_result_per_document:bool=False, word_embedding_match_threshold:float=0.8,
  initial_question_word_embedding_match_threshold:float=0.7):

Starts a topic matching search mode console enabling the matching of pre-registered
  documents to query phrases entered ad-hoc by the user.

Parameters:

only_one_result_per_document -- if 'True', prevents multiple topic match
  results from being returned for the same document.
word_embedding_match_threshold -- the cosine similarity above which two words match where the  
  search phrase word does not govern an interrogative pronoun.
initial_question_word_embedding_match_threshold -- the cosine similarity above which two
  words match where the search phrase word governs an interrogative pronoun.
```

``` {.python}
Manager.close(self) -> None

Terminates the worker processes.
```

<a id="manager.nlp"></a>
#### 6.2 `manager.nlp`

`manager.nlp` is the underlying spaCy [Language](https://spacy.io/api/language/) object on which both Coreferee and Holmes have been registered as custom pipeline components. The most efficient way of parsing documents for use with Holmes is to call [`manager.nlp.pipe()`](https://spacy.io/api/language/#pipe). This yields an iterable of documents that can then be loaded into Holmes via [`manager.register_serialized_documents()`](#manager-register-serialized-documents-function).

The [`pipe()` method](https://spacy.io/api/language#pipe) has an argument `n_process` that specifies the number of processors to use. With `_lg`, `_md` and `_sm` spaCy models, there are [some situations](https://github.com/explosion/spaCy/discussions/8402#multiprocessing) where it can make sense to specify a value other than 1 (the default). Note however that with transformer spaCy models (`_trf`) values other than 1 are not supported.

<a id="ontology"></a>
#### 6.3 `Ontology`

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

Holmes is not designed to support changes to a loaded ontology via direct
calls to the methods of this class. It is also not permitted to share a single instance
of this class between multiple Manager instances: instead, a separate Ontology instance
pointing to the same path should be created for each Manager.

Matching is case-insensitive.

Parameters:

ontology_path -- the path from where the ontology is to be loaded,
or a list of several such paths. See https://github.com/RDFLib/rdflib/.  
owl_class_type -- optionally overrides the OWL 2 URL for types.  
owl_individual_type -- optionally overrides the OWL 2 URL for individuals.  
owl_type_link -- optionally overrides the RDF URL for types.  
owl_synonym_type -- optionally overrides the OWL 2 URL for synonyms.  
owl_hyponym_type -- optionally overrides the RDF URL for hyponyms.
symmetric_matching -- if 'True', means hypernym relationships are also taken into account.
```

<a id="supervised-topic-training-basis"></a>
#### 6.4 `SupervisedTopicTrainingBasis` (returned from `Manager.get_supervised_topic_training_basis()`)

Holder object for training documents and their classifications from which one or more
[SupervisedTopicModelTrainer](#supervised-topic-model-trainer) objects can be derived. This class is NOT threadsafe.

``` {.python}
SupervisedTopicTrainingBasis.parse_and_register_training_document(self, text:str, classification:str,
  label:Optional[str]=None) -> None

Parses and registers a document to use for training.

Parameters:

text -- the document text
classification -- the classification label
label -- a label with which to identify the document in verbose training output,
  or 'None' if a random label should be assigned.
```

``` {.python}
SupervisedTopicTrainingBasis.register_training_document(self, doc:Doc, classification:str, 
  label:Optional[str]=None) -> None

Registers a pre-parsed document to use for training.

Parameters:

doc -- the document
classification -- the classification label
label -- a label with which to identify the document in verbose training output,
  or 'None' if a random label should be assigned.
```

``` {.python}
SupervisedTopicTrainingBasis.register_additional_classification_label(self, label:str) -> None

Register an additional classification label which no training document possesses explicitly
  but that should be assigned to documents whose explicit labels are related to the
  additional classification label via the classification ontology.
```

``` {.python}
SupervisedTopicTrainingBasis.prepare(self) -> None

Matches the phraselets derived from the training documents against the training
  documents to generate frequencies that also include combined labels, and examines the
  explicit classification labels, the additional classification labels and the
  classification ontology to derive classification implications.

  Once this method has been called, the instance no longer accepts new training documents
  or additional classification labels.
```

<a id="supervised-topic-training-basis-train"></a>
``` {.python}
SupervisedTopicTrainingBasis.train(
        self,
        *,
        minimum_occurrences: int = 4,
        cv_threshold: float = 1.0,
        learning_rate: float = 0.001,
        batch_size: int = 5,
        max_epochs: int = 200,
        convergence_threshold: float = 0.0001,
        hidden_layer_sizes: Optional[List[int]] = None,
        shuffle: bool = True,
        normalize: bool = True
    ) -> SupervisedTopicModelTrainer:

Trains a model based on the prepared state.

Parameters:

minimum_occurrences -- the minimum number of times a word or relationship has to
  occur in the context of the same classification for the phraselet
  to be accepted into the final model.
cv_threshold -- the minimum coefficient of variation with which a word or relationship has
  to occur across the explicit classification labels for the phraselet to be
  accepted into the final model.
learning_rate -- the learning rate for the Adam optimizer.
batch_size -- the number of documents in each training batch.
max_epochs -- the maximum number of training epochs.
convergence_threshold -- the threshold below which loss measurements after consecutive
  epochs are regarded as equivalent. Training stops before 'max_epochs' is reached
  if equivalent results are achieved after four consecutive epochs.
hidden_layer_sizes -- a list containing the number of neurons in each hidden layer, or
  'None' if the topology should be determined automatically.
shuffle -- 'True' if documents should be shuffled during batching.
normalize -- 'True' if normalization should be applied to the loss function.
```

<a id="supervised-topic-model-trainer"></a>
#### 6.5 `SupervisedTopicModelTrainer` (returned from `SupervisedTopicTrainingBasis.train()`)

Worker object used to train and generate models. This object could be removed from the public interface
(`SupervisedTopicTrainingBasis.train()` could return a `SupervisedTopicClassifier` directly) but has
been retained to facilitate testability.

This class is NOT threadsafe.

``` {.python}
SupervisedTopicModelTrainer.classifier(self)

Returns a supervised topic classifier which contains no explicit references to the training data and that
can be serialized.
```

<a id="supervised-topic-classifier"></a>
#### 6.6 `SupervisedTopicClassifier` (returned from
`SupervisedTopicModelTrainer.classifier()` and
`Manager.deserialize_supervised_topic_classifier()`))

``` {.python}
SupervisedTopicClassifier.def parse_and_classify(self, text: str) -> Optional[OrderedDict]:

Returns a dictionary from classification labels to probabilities
  ordered starting with the most probable, or *None* if the text did
  not contain any words recognised by the model.

Parameters:

text -- the text to parse and classify.
```

``` {.python}
SupervisedTopicClassifier.classify(self, doc: Doc) -> Optional[OrderedDict]:

Returns a dictionary from classification labels to probabilities
  ordered starting with the most probable, or *None* if the text did
  not contain any words recognised by the model.


Parameters:

doc -- the pre-parsed document to classify.
```

``` {.python}
SupervisedTopicClassifier.serialize_model(self) -> str

Returns a serialized model that can be reloaded using
  *Manager.deserialize_supervised_topic_classifier()*
```

<a id="dictionary"></a>
#### 6.7 Dictionary returned from `Manager.match_returning_dictionaries()`)

``` {.python}
A text-only representation of a match between a search phrase and a
document. The indexes refer to tokens.

Properties:

search_phrase_label -- the label of the search phrase.
search_phrase_text -- the text of the search phrase.
document -- the label of the document.
index_within_document -- the index of the match within the document.
sentences_within_document -- the raw text of the sentences within the document that matched.
negated -- 'True' if this match is negated.
uncertain -- 'True' if this match is uncertain.
involves_coreference -- 'True' if this match was found using coreference resolution.
overall_similarity_measure -- the overall similarity of the match, or
  '1.0' if embedding-based matching was not involved in the match.  
word_matches -- an array of dictionaries with the properties:

  search_phrase_token_index -- the index of the token that matched from the search phrase.
  search_phrase_word -- the string that matched from the search phrase.
  document_token_index -- the index of the token that matched within the document.
  first_document_token_index -- the index of the first token that matched within the document.
    Identical to 'document_token_index' except where the match involves a multiword phrase.
  last_document_token_index -- the index of the last token that matched within the document
    (NOT one more than that index). Identical to 'document_token_index' except where the match
    involves a multiword phrase.
  structurally_matched_document_token_index -- the index of the token within the document that
    structurally matched the search phrase token. Is either the same as 'document_token_index' or
    is linked to 'document_token_index' within a coreference chain.
  document_subword_index -- the index of the token subword that matched within the document, or
    'None' if matching was not with a subword but with an entire token.
  document_subword_containing_token_index -- the index of the document token that contained the
    subword that matched, which may be different from 'document_token_index' in situations where a
    word containing multiple subwords is split by hyphenation and a subword whose sense
    contributes to a word is not overtly realised within that word.
  document_word -- the string that matched from the document.
  document_phrase -- the phrase headed by the word that matched from the document.
  match_type -- 'direct', 'derivation', 'entity', 'embedding', 'ontology', 'entity_embedding'
    or 'question'.
  negated -- 'True' if this word match is negated.
  uncertain -- 'True' if this word match is uncertain.
  similarity_measure -- for types 'embedding' and 'entity_embedding', the similarity between the
    two tokens, otherwise '1.0'.
  involves_coreference -- 'True' if the word was matched using coreference resolution.
  extracted_word -- within the coreference chain, the most specific term that corresponded to
    the document_word.
  depth -- the number of hyponym relationships linking 'search_phrase_word' and
    'extracted_word', or '0' if ontology-based matching is not active. Can be negative
    if symmetric matching is active.
  explanation -- creates a human-readable explanation of the word match from the perspective of the
    document word (e.g. to be used as a tooltip over it).
```

<a id="topic-match-dictionary"></a>
#### 6.8 Dictionary returned from `Manager.topic_match_documents_returning_dictionaries_against()`

``` {.python}
A text-only representation of a topic match between a search text and a document.

Properties:

document_label -- the label of the document.
text -- the document text that was matched.
text_to_match -- the search text.
rank -- a string representation of the scoring rank which can have the form e.g. '2=' in case of a tie.
index_within_document -- the index of the document token where the activation peaked.
subword_index -- the index of the subword within the document token where the activation peaked, or
  'None' if the activation did not peak at a specific subword.
start_index -- the index of the first document token in the topic match.
end_index -- the index of the last document token in the topic match (NOT one more than that index).
sentences_start_index -- the token start index within the document of the sentence that contains
  'start_index'
sentences_end_index -- the token end index within the document of the sentence that contains
  'end_index' (NOT one more than that index).
sentences_character_start_index_in_document -- the character index of the first character of 'text'
  within the document.
sentences_character_end_index_in_document -- one more than the character index of the last
  character of 'text' within the document.
score -- the score
word_infos -- an array of arrays with the semantics:

  [0] -- 'relative_start_index' -- the index of the first character in the word relative to
    'sentences_character_start_index_in_document'.
  [1] -- 'relative_end_index' -- one more than the index of the last character in the word
    relative to 'sentences_character_start_index_in_document'.  
  [2] -- 'type' -- 'single' for a single-word match, 'relation' if within a relation match
    involving two words, 'overlapping_relation' if within a relation match involving three
    or more words.
  [3] -- 'is_highest_activation' -- 'True' if this was the word at which the highest activation
    score reported in 'score' was achieved, otherwise 'False'.
  [4] -- 'explanation' -- a human-readable explanation of the word match from the perspective of
    the document word (e.g. to be used as a tooltip over it).

answers -- an array of arrays with the semantics:

  [0] -- the index of the first character of a potential answer to an initial question word.
  [1] -- one more than the index of the last character of a potential answer to an initial question
    word.
```

<a id="a-note-on-the-license"></a>
### 7 A note on the license

Earlier versions of Holmes could only be published under a restrictive license because of patent issues. As explained in the
[introduction](#introduction), this is no longer the case thanks to the generosity of [AstraZeneca](https://www.astrazeneca.com/):
versions from 4.0.0 onwards are licensed under the MIT license.

<a id="information-for-developers"></a>
### 8 Information for developers

<a id="how-it-works"></a>
#### 8.1 How it works

<a id="how-it-works-structural-matching"></a>
##### 8.1.1 Structural matching (chatbot and structural extraction)

The word-level matching and the high-level operation of structural
matching between search-phrase and document subgraphs both work more or
less as one would expect. What is perhaps more in need of further
comment is the semantic analysis code subsumed in the [parsing.py](https://github.com/explosion/holmes-extractor/blob/master/holmes_extractor/parsing.py)
script as well as in the `language_specific_rules.py` script for each
language.

`SemanticAnalyzer` is an abstract class that is subclassed for each
language: at present by `EnglishSemanticAnalyzer` and
`GermanSemanticAnalyzer`. These classes contain most of the semantic analysis code.
`SemanticMatchingHelper` is a second abstract class, again with an concrete
implementation for each language, that contains semantic analysis code
that is required at matching time. Moving this out to a separate class family
was necessary because, on operating systems that spawn processes rather
than forking processes (e.g. Windows), `SemanticMatchingHelper` instances
have to be serialized when the worker processes are created: this would
not be possible for `SemanticAnalyzer` instances because not all
spaCy models are serializable, and would also unnecessarily consume
large amounts of memory.

At present, all functionality that is common
to the two languages is realised in the two abstract parent classes.
Especially because English and German are closely related languages, it
is probable that functionality will need to be moved from the abstract
parent classes to specific implementing children classes if and when new
semantic analyzers are added for new languages.

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

For each language, the `match_implication_dict` dictionary maps search-phrase semantic dependencies
to matching document semantic dependencies and is responsible for the [asymmetry of matching
between search phrases and documents](#general-comments).

<a id="how-it-works-topic-matching"></a>
##### 8.1.2 Topic matching

Topic matching involves the following steps:

1. The query document or query phrase is parsed and a number of **phraselets**
are derived from it. Single-word phraselets are extracted for every word (or subword in German) with its own meaning within the query phrase apart from a handful of stop words defined within the semantic matching helper (`SemanticMatchingHelper.topic_matching_phraselet_stop_lemmas`), which are
consistently ignored throughout the whole process.
2. Two-word or **relation** phraselets are extracted from the query document or query phrase wherever certain grammatical structures
are found. The structures that trigger two-word phraselets differ from language to language
but typically include verb-subject, verb-object and noun-adjective pairs as well as verb-noun and noun-noun relations spanning prepositions. Each relation phraselet
has a parent (governor) word or subword and a child (governed) word or subword. The relevant
phraselet structures for a given language are defined in `SemanticMatchingHelper.phraselet_templates`.
3. Both types of phraselet are assigned a **frequency factor** expressing how common or rare its word or words are in the corpus. Frequency factors are determined using a logarithmic calculation and range from 0.0 (very common) to 1.0 (very rare). Each word within a relation phraselet is also assigned its own frequency factor.
4. Phraselet templates where the parent word belongs to a closed word class, e.g. prepositions, can be defined as 'reverse_only'. This signals that matching with derived phraselets should only be attempted starting from the child word rather than from the parent word as normal. Phraselets are also defined as reverse-only when the parent word is one of a handful of words defined within the semantic matching helper (`SemanticMatchingHelper.topic_matching_reverse_only_parent_lemmas`) or when the frequency factor for the parent word is below the threshold for relation matching ( `relation_matching_frequency_threshold`, default: 0.25).  These measures are necessary because matching on e.g. a parent preposition would lead to a large number of
potential matches that would take a lot of resources to investigate: it is better to start
investigation from the less frequent word within a given relation.
5. All single-word phraselets are matched against the document corpus.
6. Normal [structural matching](#how-it-works-structural-matching) is used to match against the document corpus all relation phraselets
that are not set to reverse-matching.
7. Reverse matching starts at all words in the corpus that match a relation phraselet child word. Every word governing one of these words is a potential match for the corresponding relation phraselet parent word, so structural matching is attempted starting at all these parent words. Reverse matching is only attempted for relation phraselets where the child word's frequency factor is above the threshold for relation matching ( `relation_matching_frequency_threshold`, default: 0.25).
8. If either the parent or the child word of a relation template has a frequency factor above a configurable threshold (`embedding_matching_frequency_threshold`, default: 0.5), matching at all of those words where the relation template has not already been
matched is retried using embeddings at the other word within the relation. A pair of words is then regarded as matching when their mutual cosine similarity is above `initial_question_word_embedding_match_threshold` (default: 0.7) in situations where the document word has an initial question word in its phrase or `word_embedding_match_threshold` (default: 0.8) in all other situations.
9. The set of structural matches collected up to this point is filtered to cover cases where the same
document words were matched by multiple phraselets, where multiple sibling words have been matched by the same
phraselet where one sibling has a higher [embedding-based similarity](#embedding-based-matching) than the
other, and where a phraselet has matched multiple words that [corefer](#coreference-resolution) with one another.
10. Each document is scanned from beginning to end and a psychologically inspired **activation score**
is determined for each word in each document.

  - Activation is tracked separately for each phraselet. Each time
  a match for a phraselet is encountered, the activation for that phraselet is set to the score returned by
  the match, unless the existing activation is already greater than that score. If the parameter `use_frequency_factor` is set to `True` (the default), each score is scaled by the frequency factor of its phraselet, meaning that words that occur less frequently in the corpus give rise to higher scores.
  - For as long as the activation score for a phraselet has a value above zero, 1 divided by a
  configurable number (`maximum_activation_distance`; default: 75) of its value is subtracted from it as each new word is read.
  - The score returned by a match depends on whether the match was produced by a single-word noun phraselet that matched an entire word (`single_word_score`; default: 50), a non-noun single-word phraselet or a noun phraselet that matched a subword (`single_word_any_tag_score`; default: 20),
  a relation phraselet produced by a reverse-only template (`reverse_only_relation_score`; default: 200),
  any other (normally matched) relation phraselet (`relation_score`; default: 300), or a relation
  phraselet involving an initial question word (`initial_question_word_answer_score`; default: 600).
  - Where a match involves embedding-based matching, the resulting inexactitude is
  captured by multiplying the potential new activation score with the value of the
  similarity measure that was returned for the match multiplied by a penalty value (`embedding_penalty`; default: 0.6).
  - Where a match involves ontology-based matching, the resulting inexactitude is captured
  by multiplying the potential new activation score by a penalty value (`ontology_penalty`;
  default: 0.9) once more often than the difference in depth between the two ontology entries,
  i.e. once for a synonym, twice for a child, three times for a grandchild and so on.
  - When the same word was involved in matches against more than one two-word phraselets, this
  implies that a structure involving three or more words has been matched. The activation score returned by
  each match within such a structure is multiplied by a configurable factor
  (`overlapping_relation_multiplier`; default: 1.5).

11. The most relevant passages are then determined by the highest activation score peaks within the documents. Areas to either side of each peak up to a certain distance
(`sideways_match_extent`; default: 100 words) within which the activation score is higher than the `different_match_cutoff_score` (default: 15) are regarded as belonging to a contiguous passage around the peak that is then returned as a `TopicMatch` object. (Note that this default will almost certainly turn out to be too low if `use_frequency_factor`is set to `False`.) A word whose activation equals the threshold exactly is included at the beginning of the area as long as the next word where
activation increases has a score above the threshold. If the topic match peak is below the
threshold, the topic match will only consist of the peak word.
12. If `initial_question_word_behaviour` is set to `process` (the default) or to `exclusive`, where a document word has [matched an initial question word](#initial-question-word-matching) from the query phrase, the subtree of the matched document word is identified as a potential answer to the question and added to the dictionary to be returned. If `initial_question_word_behaviour` is set to `exclusive`, any topic matches that do not contain answers to initial question words are discarded.
13. Setting `only_one_result_per_document = True` prevents more than one result from being returned from the same
document; only the result from each document with the highest score will then be returned.
14. Adjacent topic matches whose scores differ by less than `tied_result_quotient` (default: 0.9) are labelled as tied.

<a id="how-it-works-supervised-document-classification"></a>
##### 8.1.3 Supervised document classification

The supervised document classification use case relies on the same phraselets as the
[topic matching use case](#how-it-works-topic-matching), although reverse-only templates are ignored and
a different set of stop words is used (`SemanticMatchingHelper.supervised_document_classification_phraselet_stop_lemmas`).
Classifiers are built and trained as follows:

1. All phraselets are extracted from all training documents and registered with a structural matcher.
2. Each training document is then matched against the totality of extracted phraselets and the number of times
each phraselet is matched within training documents with each classification label is recorded. Whether multiple
occurrences within a single document are taken into account depends on the value of `oneshot`; whether
single-word phraselets are generated for all words with their own meaning or only for those such words whose
part-of-speech tags match the single-word phraselet template specification (essentially: noun phraselets) depends on the value
of `match_all_words`. Wherever two phraselet matches overlap, a combined match is recorded. Combined matches are
treated in the same way as other phraselet matches in further processing. This means that effectively the
algorithm picks up one-word, two-word and three-word semantic combinations.
See [here](#improve-performance-of-supervised-document-classification-training) for a discussion of the
performance of this step.
3. The results for each phraselet are examined and phraselets are removed from the model that do not play a
statistically significant role in predicting classifications. Phraselets are removed that did not match within
the documents of any classification a minimum number of times (`minimum_occurrences`; default: 4) or where the
coefficient of variation (the standard deviation divided by the arithmetic mean) of the occurrences across the
categories is below a threshold (`cv_threshold`; default: 1.0).
4. The phraselets that made it into the model are once again matched against each document. Matches against each
phraselet are used to determine the input values to a multilayer perceptron: the input nodes can either record
occurrence (binary) or match frequency (scalar) (`oneshot==True` vs. `oneshot==False` respectively). The outputs are the
category labels, including any additional labels determined via a classification ontology.  By default, the multilayer
perceptron has three hidden layers where the first hidden layer has the same number of neurons as the input layer and
the second and third layers have sizes in between the input and the output layer with an equally sized step between
each size; the user is however [free to specify any other topology](#supervised-topic-training-basis-train).
5. The resulting model is serializable, i.e. can be saved and reloaded.
6. When a new document is classified, the output
is zero, one or many suggested classifications; when more than one classification is suggested, the classifications
are ordered by decreasing probabilility.

<a id="development-and-testing-guidelines"></a>
#### 8.2 Development and testing guidelines

Holmes code is formatted with [black](https://black.readthedocs.io/en/stable/).

The complexity of what Holmes does makes development impossible without
a robust set of over 1400 regression tests. These can be executed individually
with `unittest` or all at once by running the
[pytest](https://docs.pytest.org/en/latest/) utility from the Holmes
source code root directory. (Note that the Python 3 command on Linux
is `pytest-3`.)

The `pytest` variant will only work on machines with sufficient memory resources. To
reduce this problem, the tests are distributed across three subdirectories, so that
`pytest` can be run three times, once from each subdirectory:

-   [en](https://github.com/explosion/holmes-extractor/blob/master/tests/en): tests relating to English
-   [de](https://github.com/explosion/holmes-extractor/blob/master/tests/de): tests relating to German
-   [common](https://github.com/explosion/holmes-extractor/blob/master/tests/common): language-independent tests

<a id="areas-for-further-development"></a>
#### 8.3 Areas for further development

<a id="additional-languages"></a>
##### 8.3.1 Additional languages

New languages can be added to Holmes by subclassing the
`SemanticAnalyzer` and `SemanticMatchingHelper` classes as explained
[here](#how-it-works-structural-matching).

<a id="use-of-machine-learning-to-improve-matching"></a>
##### 8.3.2 Use of machine learning to improve matching

The sets of matching semantic dependencies captured in the
`_matching_dep_dict` dictionary for each language have been obtained on
the basis of a mixture of linguistic-theoretical expectations and trial
and error. The results would probably be improved if the `_matching_dep_dict` dictionaries
could be derived using machine learning instead; as yet this has not been
attempted because of the lack of appropriate training data.

<a id="remove-names-from-supervised-document-classification-models"></a>
##### 8.3.3 Remove names from supervised document classification models

An attempt should be made to remove personal data from supervised document classification models to
make them more compliant with data protection laws.

<a id="improve-performance-of-supervised-document-classification-training"></a>
##### 8.3.4 Improve the performance of supervised document classification training

In cases where [embedding-based matching](#embedding-based-matching) is not active, the second step of the
[supervised document classification](#how-it-works-supervised-document-classification) procedure repeats
a considerable amount of processing from the first step. Retaining the relevant information from the first
step of the procedure would greatly improve training performance. This has not been attempted up to now
because a large number of tests would be required to prove that such performance improvements did not
have any inadvertent impacts on functionality.

<a id="explore-hyperparameters"></a>
##### 8.3.5 Explore the optimal hyperparameters for topic matching and supervised document classification

The [topic matching](#topic-matching) and [supervised document classification](#supervised-document-classification)
use cases are both configured with a number of hyperparameters that are presently set to best-guess values
derived on a purely theoretical basis. Results could be further improved by testing the use cases with a variety
of hyperparameters to learn the optimal values.

<a id="version-history"></a>
#### 8.4 Version history

<a id="version-20x"></a>
##### 8.4.1 Version 2.0.x

The initial open-source version.

<a id="version-210"></a>
##### 8.4.2 Version 2.1.0

-  Upgrade to spaCy 2.1.0 and neuralcoref 4.0.0.
-  Addition of new dependency `pobjp` linking parents of prepositions directly with their children.
-  Development of the multiprocessing architecture, which has the `MultiprocessingManager` object
as its facade.
-  Complete overhaul of [topic matching](#how-it-works-topic-matching).
-  Incorporation of coreference information into Holmes document structures so it no longer needs to be calculated on the fly.
-  New literature examples for both languages and the facility to serve them over RESTful HTTP.
-  Numerous minor improvements and bugfixes.

<a id="version-220"></a>
##### 8.4.3 Version 2.2.0

-  Addition of derivational morphology analysis allowing the matching of related words with the
same stem.
-  Addition of new dependency types and dependency matching rules to make full use of the new derivational morphology information.
-  For German, analysis of and matching with subwords (constituent parts of compound words), e.g. *Information* and *Extraktion* are the subwords within *Informationsextraktion*.
-  It is now possible to supply multiple ontology files to the [Ontology](#ontology) constructor.
-  Ontology implication rules are now calculated eagerly to improve runtime performance.
-  [Ontology-based matching](#ontology-based-matching) now includes special, language-specific rules to handle hyphens within ontology entries.
-  Word-match information is now included in all matches including single-word matches.
-  Word matches and dictionaries derived from them now include human-readable explanations designed to be used as tooltips.
-  In [topic matching](#manager-topic-match-function), a penalty is now applied to ontology-based matches as well as to embedding-based matches.
-  [Topic matching](#manager-topic-match-function) now includes a filter facility to specify
that only documents whose labels begin with a certain string should be searched.
-  Error handling and reporting have been improved for the MultiprocessingManager.
-  Numerous minor improvements and bugfixes.
-  The [demo website](https://demo.holmes.prod.demos.explosion.services/) has been updated to reflect the changes.

<a id="version-221"></a>
##### 8.4.4 Version 2.2.1

-  Fixed bug with reverse derived lemmas and subwords (only affects German).
-  Removed dead code.

<a id="version-300"></a>
##### 8.4.5 Version 3.0.0

-  Moved to [coreferee](https://github.com/explosion/coreferee) as the source of coreference information, meaning that coreference resolution is now active for German as well as English; all documents can be serialized; and the latest spaCy version can be supported.
-  The corpus frequencies of words are now taken into account when scoring topic matches.
-  Reverse dependencies are now taken into account, so that e.g. *a man dies* can match *the dead man* although the dependencies in the two phrases point in opposite directions.
-  Merged the pre-existing `Manager` and `MultiprocessingManager` classes into a single `Manager` class, with a redesigned public interface, that uses worker threads for everything except supervised document classification.
-  Added support for [initial question words](#initial-question-word-matching).
-  The [demo website](https://demo.holmes.prod.demos.explosion.services/) has been updated to reflect the changes.

<a id="version-400"></a>
##### 8.4.6 Version 4.0.0

- The license has been changed from GPL3 to MIT.
- The word matching code has been refactored and now uses the Strategy pattern, making it easy to add additional word-matching strategies.
- With the exception of [rdflib](https://github.com/RDFLib/rdflib), all direct dependencies are now from within the Explosion stack, making
installation much faster and more trouble-free.
- Holmes now supports a wide range of Python (3.6—3.10) and spaCy (3.1—3.3) versions.
- A new [demo website](https://demo.holmes.prod.demos.explosion.services/) has been developed by <a href="mailto:edward@explosion.ai">Edward Schmuhl</a> based on Streamlit.