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

For more information, please see the [main documentation on Github](https://github.com/explosion/holmes-extractor).
