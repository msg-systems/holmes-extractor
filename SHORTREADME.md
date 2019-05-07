**Holmes** is a Python 3 library (tested with version 3.7.2) that supports a number of
use cases involving information extraction from English and German texts. In all use cases, the information extraction
is based on analysing the semantic relationships expressed by the component parts of each sentence:

- In the [chatbot](https://github.com/msg-systems/holmes-extractor/#getting-started) use case, the system is configured using one or more **search phrases**.
Holmes then looks for structures whose meanings correspond to those of these search phrases within
a searched **document**, which in this case corresponds to an individual snippet of text or speech
entered or uttered by the end user. Within a match, each non-grammatical word in the search phrase
corresponds to one or more non-grammatical words in the document, which can then be extracted as structured information.

- The [structural matching](https://github.com/msg-systems/holmes-extractor/#structural-matching) use case uses exactly the same technological basis as the chatbot use
case, but searching takes place with respect to a pre-existing document or documents that are typically much
longer than the snippets analysed in the chatbot use case.

- The [topic matching](https://github.com/msg-systems/holmes-extractor/#topic-matching) use case aims to find passages in a document or documents whose meaning
is close to that of another document, which takes on the role of the **query document**, or to that of a
**query phrase** entered ad-hoc by the user. Holmes extracts a number of small **phraselets** from the query phrase or
query document, matches the documents being searched against each phraselet, and conflates the results to find
the most relevant passages within the documents. Because there is no strict requirement that every non-grammatical
word in the query document match a specific word or words in the searched documents, more matches are found
than in the structural matching use case, but the matches do not contain structured information that can be
used in subsequent processing.

- The [supervised document classification](https://github.com/msg-systems/holmes-extractor/#supervised-document-classification) use case uses training data to
learn a classifier that assigns one or more **classification labels** to new documents based on what they are about.
It classifies a new document by matching it against phraselets that were extracted from the training documents in the
same way that phraselets are extracted from the query document in the topic matching use case. The technique is
inspired by bag-of-words-based classification algorithms that use n-grams, but aims to derive n-grams whose component
words are related semantically rather than that just happen to be neighbours in the surface representation of a language.

In all four use cases, the **individual words** are matched using a [number of strategies](https://github.com/msg-systems/holmes-extractor/#word-level-matching-strategies).
To work out whether two grammatical structures that contain individually matching words correspond logically and
constitute a match, Holmes transforms the syntactic parse information provided by the [spaCy](https://spacy.io/) library
into semantic structures that allow texts to be compared using predicate logic. As a user of Holmes, you do not need to
understand the intricacies of how this works, although there are some
[important tips](https://github.com/msg-systems/holmes-extractor/#writing-effective-search-phrases) around writing effective search phrases for the chatbot and
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

For more information, please see the [main documentation on Github](https://github.com/msg-systems/holmes-extractor).
