import math
import pickle
import importlib
from abc import ABC, abstractmethod
from functools import total_ordering
from typing import Callable
import jsonpickle
import srsly
import pkg_resources
from spacy.tokens import Token, Doc
from .errors import WrongModelDeserializationError, WrongVersionDeserializationError,\
        DocumentTooBigError, SearchPhraseContainsNegationError,\
        SearchPhraseContainsConjunctionError, SearchPhraseWithoutMatchableWordsError,\
        SearchPhraseContainsMultipleClausesError, SearchPhraseContainsCoreferringPronounError

SERIALIZED_DOCUMENT_VERSION = '3.1'

class SemanticDependency:
    """A labelled semantic dependency between two tokens."""

    def __init__(self, parent_index, child_index, label=None, is_uncertain=False):
        """Args:

        parent_index -- the index of the parent token within the document. The dependency will
            always be managed by the parent token, but the index is maintained within the
            object for convenience.
        child_index -- the index of the child token within the document, or one less than the zero
            minus the index of the child token within the document to indicate a grammatical
            dependency. A grammatical dependency means that the parent should be replaced by the
            child during matching.
        label -- the label of the semantic dependency, which must be *None* for grammatical
            dependencies.
        is_uncertain -- if *True*, any match involving this dependency will itself be uncertain.
        """
        if child_index < 0 and label is not None:
            raise RuntimeError(
                'Semantic dependency with negative child index may not have a label.')
        if parent_index == child_index:
            raise RuntimeError(' '.join((
                'Attempt to create self-referring semantic dependency with index',
                str(parent_index))))
        self.parent_index = parent_index
        self.child_index = child_index
        self.label = label
        self.is_uncertain = is_uncertain

    def parent_token(self, doc):
        """Convenience method to return the parent token of this dependency.

        doc -- the document containing the token.
        """
        index = self.parent_index
        if index < 0:
            index = -1 - index
        return doc[index]

    def child_token(self, doc):
        """Convenience method to return the child token of this dependency.

        doc -- the document containing the token.
        """
        index = self.child_index
        if index < 0:
            index = -1 - index
        return doc[index]

    def __str__(self):
        """e.g. *2:nsubj* or *2:nsubj(U)* to represent uncertainty."""
        working_label = str(self.label)
        if self.is_uncertain:
            working_label = ''.join((working_label, '(U)'))
        return ':'.join((str(self.child_index), working_label))

    def __eq__(self, other):
        return isinstance(other, SemanticDependency) and \
            self.parent_index == other.parent_index and self.child_index == other.child_index \
            and self.label == other.label and self.is_uncertain == other.is_uncertain

    def __hash__(self):
        return hash((self.parent_index, self.child_index, self.label, self.is_uncertain))

class Mention:
    """ Simplified information about a coreference mention with respect to a specific token. """

    def __init__(self, root_index, indexes):
        self.root_index = root_index
        self.indexes = indexes

    def __str__(self):
        return ''.join(('[', str(self.root_index), '; ', str(self.indexes), ']'))

class Subword:
    """A semantically atomic part of a word. Currently only used for German.

        containing_token_index -- the index of the containing token within the document.
        index -- the index of the subword within the word.
        text -- the original subword string.
        lemma -- the model-normalized representation of the subword string.
        derived_lemma -- where relevant, another lemma with which *lemma* is derivationally related
        and which can also be useful for matching in some usecases; otherwise *None*
        vector -- the vector representation of *lemma*, or *None* if there is none available.
        char_start_index -- the character index of the subword within the containing word.
        dependent_index -- the index of a subword that is dependent on this subword, or *None*
            if there is no such subword.
        dependency_label -- the label of the dependency between this subword and its dependent,
            or *None* if it has no dependent.
        governor_index -- the index of a subword on which this subword is dependent, or *None*
            if there is no such subword.
        governing_dependency_label -- the label of the dependency between this subword and its
            governor, or *None* if it has no governor.
    """
    def __init__(
            self, containing_token_index, index, text, lemma, derived_lemma, vector,
            char_start_index, dependent_index, dependency_label, governor_index,
            governing_dependency_label):
        self.containing_token_index = containing_token_index
        self.index = index
        self.text = text
        self.lemma = lemma
        self.derived_lemma = derived_lemma
        self.vector = vector
        self.char_start_index = char_start_index
        self.dependent_index = dependent_index
        self.dependency_label = dependency_label
        self.governor_index = governor_index
        self.governing_dependency_label = governing_dependency_label

    def lemma_or_derived_lemma(self):
        if self.derived_lemma is not None:
            return self.derived_lemma
        else:
            return self.lemma

    @property
    def is_head(self):
        return self.governor_index is None

    def __str__(self):
        if self.derived_lemma is not None:
            lemma_string = ''.join((self.lemma, '(', self.derived_lemma, ')'))
        else:
            lemma_string = self.lemma
        return '/'.join((self.text, lemma_string))


@total_ordering
class Index:
    """ The position of a word or subword within a document. """

    def __init__(self, token_index, subword_index):
        self.token_index = token_index
        self.subword_index = subword_index

    def is_subword(self):
        return self.subword_index is not None

    def __eq__(self, other):
        return isinstance(other, Index) and \
            self.token_index == other.token_index and self.subword_index == other.subword_index

    def __lt__(self, other):
        if not isinstance(other, Index):
            raise RuntimeError('Comparison between Index and another type.')
        if self.token_index < other.token_index:
            return True
        if not self.is_subword() and other.is_subword():
            return True
        if self.is_subword() and other.is_subword() and self.subword_index < other.subword_index:
            return True
        return False

    def __hash__(self):
        return hash((self.token_index, self.subword_index))

class CorpusWordPosition:
    def __init__(self, document_label, index):
        if document_label is None:
            raise RuntimeError('CorpusWordPosition.document_label must have a value.')
        self.document_label = document_label
        self.index = index

    def __eq__(self, other):
        return isinstance(other, CorpusWordPosition) and self.index == other.index and \
                self.document_label == other.document_label

    def __hash__(self):
        return hash((self.document_label, self.index))

    def __str__(self):
        return ':'.join((self.document_label, str(self.index)))

class MultiwordSpan:

    def __init__(self, text, lemma, derived_lemma, tokens):
        """Args:

        text -- the raw text representation of the multiword span
        lemma - the lemma representation of the multiword span
        derived_lemma - the lemma representation with individual words that have derived
            lemmas replaced by those derived lemmas
        tokens -- a list of tokens that make up the multiword span
        """
        self.text = text
        self.lemma = lemma
        self.derived_lemma = derived_lemma
        self.tokens = tokens

class MatchImplication:
    """Entry describing which document dependencies match a given search phrase dependency.

        Parameters:

        search_phrase_dependency -- the search phrase dependency.
        document_dependencies -- the matching document dependencies.
        reverse_document_dependencies -- document dependencies that match when the polarity is
            opposite to the polarity of *search_phrase_dependency*.
    """

    def __init__(
            self, *, search_phrase_dependency, document_dependencies,
                reverse_document_dependencies=[]):
        self.search_phrase_dependency = search_phrase_dependency
        self.document_dependencies = document_dependencies
        self.reverse_document_dependencies = reverse_document_dependencies

class QuestionWordStrategy:
    """Instructions for recognising a specific question word and for matching it within documents.

        Parameters:

        name -- a human-readable name designed for output within a user interface.
        lemma -- the Holmes lemma matching question words must have.
        tag -- the tag matching question words must have.
        deps -- a list of dependency labels matching question words may have.
        matcher - a function to determine whether a document token heads a phrase that
            matches the question word.
    """

    def __init__(
        self, *, name:str, lemma:str, tag:str, deps:list[str],
        matcher:Callable[[Token], bool]):
        self.name = name
        self.lemma = lemma
        self.tag = tag
        self.deps = deps
        self.matcher = matcher

class HolmesDocumentInfo:
    def __init__(self, semantic_analyzer):
        self.model = semantic_analyzer.model
        self.serialized_document_version = SERIALIZED_DOCUMENT_VERSION

    @srsly.msgpack_encoders("holmes_document_info_holder")
    def serialize_obj(obj, chain=None):
        if isinstance(obj, HolmesDocumentInfo):
            return {'__holmes_document_info_holder__': pickle.dumps(obj)}
        return obj if chain is None else chain(obj)

    @srsly.msgpack_decoders("holmes_document_info_holder")
    def deserialize_obj(obj, chain=None):
        if '__holmes_document_info_holder__' in obj:
            return pickle.loads(obj['__holmes_document_info_holder__'])
        return obj if chain is None else chain(obj)

class HolmesDictionary:
    """The holder object for token-level semantic information managed by Holmes

    Holmes dictionaries are accessed using the syntax *token._.holmes*.

    index -- the index of the token
    lemma -- the value returned from *._.holmes.lemma* for the token.
    derived_lemma -- the value returned from *._.holmes.derived_lemma for the token; where relevant,
        another lemma with which *lemma* is derivationally related and which can also be useful for
        matching in some usecases; otherwise *None*.
    vector -- the vector representation of *lemma*, unless *lemma* is a multiword, in which case
        the vector representation of *token.lemma_* is used instead. *None* where there is no
        vector for the lexeme.
    """

    def __init__(self, index, lemma, derived_lemma, vector):
        self.index = index
        self.lemma = lemma
        self._derived_lemma = derived_lemma
        self.vector = vector
        self.children = [] # list of *SemanticDependency* objects where this token is the parent.
        self.parents = [] # list of *SemanticDependency* objects where this token is the child.
        self.righthand_siblings = [] # list of tokens to the right of this token that stand in a
        # conjunction relationship to this token and that share its semantic parents.
        self.token_or_lefthand_sibling_index = None # the index of this token's lefthand sibling,
        # or this token's own index if this token has no lefthand sibling.
        self.is_involved_in_or_conjunction = False
        self.is_negated = None
        self.is_matchable = None
        self.is_initial_question_word = False
        self.coreference_linked_child_dependencies = [] # list of [index, label] specifications of
        # dependencies where this token is the parent, taking any coreference resolution into
        # account. Used in topic matching.
        self.coreference_linked_parent_dependencies = [] # list of [index, label] specifications of
        # dependencies where this token is the child, taking any coreference resolution into
        # account. Used in topic matching.
        self.token_and_coreference_chain_indexes = None # where no coreference, only the token
        # index; where coreference, the token index followed by the indexes of coreferring tokens
        self.mentions = []
        self.subwords = []

    @property
    def derived_lemma(self):
        if self.lemma == self._derived_lemma: # can occur with phraselets
            return None
        else:
            return self._derived_lemma

    @derived_lemma.setter
    def derived_lemma(self, derived_lemma):
        self._derived_lemma = derived_lemma

    def lemma_or_derived_lemma(self):
        if self._derived_lemma is not None:
            return self._derived_lemma
        else:
            return self.lemma

    @property
    def is_uncertain(self):
        """if *True*, a match involving this token will itself be uncertain."""
        return self.is_involved_in_or_conjunction

    def loop_token_and_righthand_siblings(self, doc):
        """Convenience generator to loop through this token and any righthand siblings."""
        indexes = [self.index]
        indexes.extend(self.righthand_siblings)
        indexes = sorted(indexes) # in rare cases involving truncated nouns in German, righthand
                                  #siblings can actually end up to the left of the head word.
        for index in indexes:
            yield doc[index]

    def get_sibling_indexes(self, doc):
        """ Returns the indexes of this token and any siblings, ordered from left to right. """
        # with truncated nouns in German, the righthand siblings may occasionally occur to the left
        # of the head noun
        head_sibling = doc[self.token_or_lefthand_sibling_index]
        indexes = [self.token_or_lefthand_sibling_index]
        indexes.extend(head_sibling._.holmes.righthand_siblings)
        return sorted(indexes)

    def has_dependency_with_child_index(self, index):
        for dependency in self.children:
            if dependency.child_index == index:
                return True
        return False

    def get_label_of_dependency_with_child_index(self, index):
        for dependency in self.children:
            if dependency.child_index == index:
                return dependency.label
        return None

    def has_dependency_with_label(self, label):
        for dependency in self.children:
            if dependency.label == label:
                return True
        return False

    def has_dependency_with_child_index_and_label(self, index, label):
        for dependency in self.children:
            if dependency.child_index == index and dependency.label == label:
                return True
        return False

    def remove_dependency_with_child_index(self, index):
        self.children = [dep for dep in self.children if dep.child_index != index]

    def string_representation_of_children(self):
        children = sorted(
            self.children, key=lambda dependency: dependency.child_index)
        return '; '.join(str(child) for child in children)

    def string_representation_of_parents(self):
        parents = sorted(
            self.parents, key=lambda dependency: dependency.parent_index)
        return '; '.join(':'.join((str(parent.parent_index), parent.label)) for parent in parents)

    def is_involved_in_coreference(self):
        return len(self.mentions) > 0

    @srsly.msgpack_encoders("holmes_dictionary_holder")
    def serialize_obj(obj, chain=None):
        if isinstance(obj, HolmesDictionary):
            return {'__holmes_dictionary_holder__': pickle.dumps(obj)}
        return obj if chain is None else chain(obj)

    @srsly.msgpack_decoders("holmes_dictionary_holder")
    def deserialize_obj(obj, chain=None):
        if '__holmes_dictionary_holder__' in obj:
            return pickle.loads(obj['__holmes_dictionary_holder__'])
        return obj if chain is None else chain(obj)

class PhraseletTemplate:
    """A template for a phraselet used in topic matching.

    Properties:

    label -- a label for the relation which will be used to form part of the labels of phraselets
        derived from this template.
    template_sentence -- a sentence with the target grammatical structure for phraselets derived
        from this template.
    template_doc -- a spacy Doc representing *template_sentence* (set by the *Manager* object)
    parent_index -- the index within *template_sentence* of the parent participant in the dependency
        (for relation phraselets) or of the word (for single-word phraselets).
    child_index -- the index within *template_sentence* of the child participant in the dependency
        (for relation phraselets) or 'None' for single-word phraselets.
    dependency_labels -- the labels of dependencies that match the template
        (for relation phraselets) or 'None' for single-word phraselets.
    parent_tags -- the tag_ values of parent participants in the dependency (for parent phraselets)
        of of the word (for single-word phraselets) that match the template.
    child_tags -- the tag_ values of child participants in the dependency (for parent phraselets)
        that match the template, or 'None' for single-word phraselets.
    reverse_only -- specifies that relation phraselets derived from this template should only be
        reverse-matched, e.g. matching should only be attempted during topic matching when the
        possible child token has already been matched to a single-word phraselet. This
        is used for performance reasons when the parent tag belongs to a closed word class like
        prepositions. Reverse-only phraselets are ignored in supervised document classification.
    assigned_dependency_label -- if a value other than 'None', specifies a dependency label that
        should be used to relabel the relationship between the parent and child participants.
        Has no effect if child_index is None.
    """

    def __init__(
            self, label, template_sentence, parent_index, child_index,
            dependency_labels, parent_tags, child_tags, *, reverse_only,
            assigned_dependency_label=None):
        self.label = label
        self.template_sentence = template_sentence
        self.parent_index = parent_index
        self.child_index = child_index
        self.dependency_labels = dependency_labels
        self.parent_tags = parent_tags
        self.child_tags = child_tags
        self.reverse_only = reverse_only
        self.assigned_dependency_label = assigned_dependency_label

    def single_word(self):
        """ 'True' if this is a template for single-word phraselets, otherwise 'False'. """
        return self.child_index is None

class PhraseletInfo:
    """Information describing a topic matching phraselet.

        Parameters:

        label -- the phraselet label.
        template_label -- the value of 'PhraseletTemplate.label'.
        parent_lemma -- the parent lemma, or the lemma for single-word phraselets.
        parent_derived_lemma -- the parent derived lemma, or the derived lemma for single-word
            phraselets.
        parent_pos -- the part of speech tag of the token that supplied the parent word.
        parent_ent_type -- the parent entity label, or the entity label for single-word
            phraselets. '' if there is none.
        child_lemma -- the child lemma, or 'None' for single-word phraselets.
        child_derived_lemma -- the child derived lemma, or 'None' for single-word phraselets.
        child_pos -- the part of speech tag of the token that supplied the child word, or 'None'
            for single-word phraselets.
        child_ent_type -- the child entity label. '' if there is none; 'None' for single-word
            phraselets.
        created_without_matching_tags -- 'True' if created without matching tags.
        reverse_only_parent_lemma -- 'True' if the parent lemma is in the reverse matching list.
        frequency_factor -- a multiplication factor between 0.0 and 1.0 which is lower the more
            frequently words occur in the corpus, relating to the whole phraselet.
        parent_frequency_factor -- a multiplication factor between 0.0 and 1.0 which is lower the
            more frequently words occur in the corpus, relating to the parent token.
        child_frequency_factor -- a multiplication factor between 0.0 and 1.0 which is lower the
            more frequently words occur in the corpus, relating to the child token.
    """

    def __init__(
            self, label, template_label, parent_lemma, parent_derived_lemma,
            parent_pos, parent_ent_type, child_lemma, child_derived_lemma,
            child_pos, child_ent_type, created_without_matching_tags,
            reverse_only_parent_lemma, frequency_factor, parent_frequency_factor,
            child_frequency_factor):
        self.label = label
        self.template_label = template_label

        self.parent_lemma = parent_lemma
        self.parent_derived_lemma = parent_derived_lemma
        self.parent_pos = parent_pos
        self.parent_ent_type = parent_ent_type
        self.child_lemma = child_lemma
        self.child_derived_lemma = child_derived_lemma
        self.child_pos = child_pos
        self.child_ent_type = child_ent_type
        self.created_without_matching_tags = created_without_matching_tags
        self.reverse_only_parent_lemma = reverse_only_parent_lemma
        self.frequency_factor = frequency_factor
        self.parent_frequency_factor = parent_frequency_factor
        self.child_frequency_factor = child_frequency_factor

    def __eq__(self, other):
        return isinstance(other, PhraseletInfo) and \
            self.label == other.label and \
            self.template_label == other.template_label and \
            self.parent_lemma == other.parent_lemma and \
            self.parent_derived_lemma == other.parent_derived_lemma and \
            self.parent_pos == other.parent_pos and \
            self.child_lemma == other.child_lemma and \
            self.child_derived_lemma == other.child_derived_lemma and \
            self.child_pos == other.child_pos and \
            self.created_without_matching_tags == other.created_without_matching_tags and \
            self.reverse_only_parent_lemma == other.reverse_only_parent_lemma and \
            str(self.frequency_factor) == str(other.frequency_factor) and \
            str(self.parent_frequency_factor) == str(other.parent_frequency_factor) and \
            str(self.child_frequency_factor) == str(other.child_frequency_factor)

    def __hash__(self):
        return hash((
            self.label, self.template_label, self.parent_lemma, self.parent_derived_lemma,
            self.parent_pos, self.child_lemma, self.child_derived_lemma,
            self.child_pos, self.created_without_matching_tags, self.reverse_only_parent_lemma,
            str(self.frequency_factor), str(self.parent_frequency_factor),
            str(self.child_frequency_factor)))

class SearchPhrase:

    def __init__(self, doc, matchable_token_indexes, root_token_index,
            matchable_non_entity_tokens_to_vectors, label, topic_match_phraselet,
            topic_match_phraselet_created_without_matching_tags, reverse_only,
            treat_as_reverse_only_during_initial_relation_matching, words_matching_root_token,
            root_word_to_match_info_dict):
        """Args:

        doc -- the Holmes document created for the search phrase
        matchable_token_indexes -- a list of indexes of tokens all of which must have counterparts
            in the document to produce a match
        root_token_index -- the index of the token at which recursive matching starts
        matchable_non_entity_tokens_to_vectors -- dictionary from token indexes to vectors.
            Only used when embedding matching is active.
        label -- a label for the search phrase.
        ontology -- a reference to the ontology held by the outer *StructuralMatcher* object.
        topic_match_phraselet -- 'True' if a topic match phraselet, otherwise 'False'.
        topic_match_phraselet_created_without_matching_tags -- 'True' if a topic match
        phraselet created without matching tags (match_all_words), otherwise 'False'.
        reverse_only -- 'True' if a phraselet that should only be reverse-matched.
        treat_as_reverse_only_during_initial_relation_matching -- phraselets are
            set to *True* in the context of topic matching to prevent them from being taken into
            account during initial relation matching because the parent relation occurs too
            frequently during the corpus. *reverse_only* cannot be used instead because it
            has an effect on scoring.
        words_matching_root_token -- a list of words that match the root token.
        root_word_to_match_info_dict -- a dictionary from words in *words_matching_root_token*
            to match information tuples.
        """
        self.doc = doc
        self.doc_text = doc.text
        self.matchable_token_indexes = matchable_token_indexes
        self.root_token_index = root_token_index
        self.matchable_non_entity_tokens_to_vectors = matchable_non_entity_tokens_to_vectors
        self.label = label
        self.topic_match_phraselet = topic_match_phraselet
        self.topic_match_phraselet_created_without_matching_tags = \
                topic_match_phraselet_created_without_matching_tags
        self.reverse_only = reverse_only
        self.treat_as_reverse_only_during_initial_relation_matching = \
            treat_as_reverse_only_during_initial_relation_matching
        self.words_matching_root_token = words_matching_root_token
        self.root_word_to_match_info_dict = root_word_to_match_info_dict
        self.has_single_matchable_word = len(matchable_token_indexes) == 1

    @property
    def matchable_tokens(self):
        return [self.doc[index] for index in self.matchable_token_indexes]

    @property
    def root_token(self):
        return self.doc[self.root_token_index]

    def pack(self):
        self.serialized_doc = self.doc.to_bytes()
        self.doc = None

    def unpack(self, vocab):
        self.doc = Doc(vocab).from_bytes(self.serialized_doc)
        self.serialized_doc = None

class SemanticAnalyzerFactory():
    """Returns the correct *SemanticAnalyzer* for the model language.
        This class must be added to if additional implementations are added for new languages.
    """

    def semantic_analyzer(self, *, nlp, vectors_nlp):
        language = nlp.meta['lang']
        try:
            language_specific_rules_module = importlib.import_module(
                '.'.join(('.lang', language, 'language_specific_rules')),
                'holmes_extractor')
        except ModuleNotFoundError:
            raise ValueError(' '.join(('Language', language, 'not supported')))
        return language_specific_rules_module.\
            LanguageSpecificSemanticAnalyzer(nlp=nlp, vectors_nlp=vectors_nlp)

class SemanticAnalyzer(ABC):
    """Abstract *SemanticAnalyzer* parent class. Functionality is placed here that is common to all
        current implementations. It follows that some functionality will probably have to be moved
        out to specific implementations whenever an implementation for a new language is added.

    For explanations of the abstract variables and methods, see the *EnglishSemanticAnalyzer*
        implementation where they can be illustrated with direct examples.
    """

    def __init__(self, *, nlp, vectors_nlp):
        """Args:

        nlp -- the spaCy model
        vectors_nlp -- the spaCy model to use for vocabularies and vectors
        """
        self.nlp = nlp
        self.vectors_nlp = vectors_nlp
        self.model = '_'.join((self.nlp.meta['lang'], self.nlp.meta['name']))
        self.derivational_dictionary = self.load_derivational_dictionary()
        self.serialized_document_version = SERIALIZED_DOCUMENT_VERSION

    def load_derivational_dictionary(self):
        in_package_filename = ''.join(('lang/', self.nlp.meta['lang'], '/data/derivation.csv'))
        absolute_filename = pkg_resources.resource_filename(__name__, in_package_filename)
        dictionary = {}
        with open(absolute_filename, "r", encoding="utf-8") as file:
            for line in file.readlines():
                words = [word.strip() for word in line.split(',')]
                for index, _ in enumerate(words):
                    dictionary[words[index]] = words[0]
        return dictionary

    _maximum_document_size = 1000000

    def spacy_parse(self, text):
        """Performs a standard spaCy parse on a string.
        """
        if len(text) > self._maximum_document_size:
            raise DocumentTooBigError(' '.join((
                'size:', str(len(text)), 'max:', str(self._maximum_document_size))))
        return self.nlp(text, disable=['coreferee', 'holmes'])

    def parse(self, text):
        return self.nlp(text)

    def get_vector(self, lemma):
        """ Returns a vector representation of *lemma*, or *None* if none is available.
        """
        lexeme = self.vectors_nlp.vocab[lemma]
        return lexeme.vector if lexeme.has_vector and lexeme.vector_norm > 0 else None

    def holmes_parse(self, spacy_doc):
        """Adds the Holmes-specific information to each token within a spaCy document.
        """
        spacy_doc._.set('holmes_document_info', HolmesDocumentInfo(self))
        for token in spacy_doc:
            lemma = self.holmes_lemma(token)
            derived_lemma = self.derived_holmes_lemma(token, lemma)
            lexeme = self.vectors_nlp.vocab[token.lemma_ if len(lemma.split()) > 1 else lemma]
            vector = lexeme.vector if lexeme.has_vector and lexeme.vector_norm > 0 else None
            token._.set('holmes', HolmesDictionary(token.i, lemma, derived_lemma, vector))
        for token in spacy_doc:
            self.set_negation(token)
        for token in spacy_doc:
            self.initialize_semantic_dependencies(token)
        self.set_initial_question_words(spacy_doc)
        for token in spacy_doc:
            self.mark_if_righthand_sibling(token)
            token._.holmes.token_or_lefthand_sibling_index = self._lefthand_sibling_recursively(
                token)
        for token in spacy_doc:
            self.copy_any_sibling_info(token)
        subword_cache = {}
        for token in spacy_doc:
            self.add_subwords(token, subword_cache)
        for token in spacy_doc:
            self.set_coreference_information(token)
        for token in spacy_doc:
            self.set_matchability(token)
        for token in spacy_doc:
            self.correct_auxiliaries_and_passives(token)
        for token in spacy_doc:
            self.copy_any_sibling_info(token)
        for token in spacy_doc:
            self.handle_relative_constructions(token)
        for token in spacy_doc:
            self.normalize_predicative_adjectives(token)
        for token in spacy_doc:
            self.create_additional_preposition_phrase_semantic_dependencies(token)
        for token in spacy_doc:
            self.perform_language_specific_tasks(token)
        for token in spacy_doc:
            self.create_convenience_dependencies(token)
        return spacy_doc

    def _lefthand_sibling_recursively(self, token):
        """If *token* is a righthand sibling, return the index of the token that has a sibling
            reference to it, otherwise return the index of *token* itself.
        """
        if token.dep_ not in self.conjunction_deps:
            return token.i
        else:
            return self._lefthand_sibling_recursively(token.head)

    def debug_structures(self, doc):
        for token in doc:
            if token._.holmes.derived_lemma is not None:
                lemma_string = ''.join((
                    token._.holmes.lemma, '(', token._.holmes.derived_lemma, ')'))
            else:
                lemma_string = token._.holmes.lemma
            subwords_strings = ';'.join(str(subword) for subword in token._.holmes.subwords)
            subwords_strings = ''.join(('[', subwords_strings, ']'))
            negation_string = 'negative' if token._.holmes.is_negated else 'positive'
            uncertainty_string = 'uncertain' if token._.holmes.is_uncertain else 'certain'
            matchability_string = 'matchable' if token._.holmes.is_matchable else 'unmatchable'
            if token._.holmes.is_involved_in_coreference():
                coreference_string = '; '.join(
                    str(mention) for mention in token._.holmes.mentions)
            else:
                coreference_string = ''
            print(
                token.i, token.text, lemma_string, subwords_strings, token.pos_, token.tag_,
                token.dep_, token.ent_type_, token.head.i,
                token._.holmes.string_representation_of_children(),
                token._.holmes.righthand_siblings, negation_string,
                uncertainty_string, matchability_string, coreference_string)

    def set_coreference_information(self, token):
        token._.holmes.token_and_coreference_chain_indexes = [token.i]
        token._.holmes.most_specific_coreferring_term_index = None
        for chain in token._.coref_chains:
            this_token_mention_index = -1
            for mention_index, mention in enumerate(chain):
                if token.i in mention.token_indexes:
                    this_token_mention_index = mention_index
                    break
            if this_token_mention_index > -1:
                for mention_index, mention in enumerate(chain):
                    if this_token_mention_index - mention_index > \
                            self.maximum_mentions_in_coreference_chain or \
                            abs (mention.root_index - token.i) > \
                            self.maximum_word_distance_in_coreference_chain:
                        continue
                    if mention_index - this_token_mention_index > \
                            self.maximum_mentions_in_coreference_chain:
                        break
                    token._.holmes.mentions.append(Mention(mention.root_index,
                            [token.i] if token.i in mention.token_indexes
                            else mention.token_indexes))
            if len(chain[0]) == 1: # chains with coordinated mentions are not relevant to
                                   # most specific mentions
                token._.holmes.most_specific_coreferring_term_index = \
                    chain[chain.most_specific_mention_index][0]
        working_set = set()
        for mention in (m for m in token._.holmes.mentions if token.i not in m.indexes):
            working_set.update(mention.indexes)
        token._.holmes.token_and_coreference_chain_indexes.extend(sorted(working_set))

    language_name = NotImplemented

    noun_pos = NotImplemented

    predicate_head_pos = NotImplemented

    matchable_pos = NotImplemented

    adjectival_predicate_head_pos = NotImplemented

    adjectival_predicate_subject_pos = NotImplemented

    adjectival_predicate_subject_dep = NotImplemented

    adjectival_predicate_predicate_dep = NotImplemented

    adjectival_predicate_predicate_pos = NotImplemented

    modifier_dep = NotImplemented

    spacy_noun_to_preposition_dep = NotImplemented

    spacy_verb_to_preposition_dep = NotImplemented

    holmes_noun_to_preposition_dep = NotImplemented

    holmes_verb_to_preposition_dep = NotImplemented

    conjunction_deps = NotImplemented

    interrogative_pronoun_tags = NotImplemented

    semantic_dependency_excluded_tags = NotImplemented

    generic_pronoun_lemmas = NotImplemented

    or_lemma = NotImplemented

    mark_child_dependencies_copied_to_siblings_as_uncertain = NotImplemented

    maximum_mentions_in_coreference_chain = NotImplemented

    maximum_word_distance_in_coreference_chain = NotImplemented

    sibling_marker_deps = NotImplemented

    @abstractmethod
    def add_subwords(self, token, subword_cache):
        pass

    @abstractmethod
    def set_negation(self, token):
        pass

    @abstractmethod
    def correct_auxiliaries_and_passives(self, token):
        pass

    @abstractmethod
    def perform_language_specific_tasks(self, token):
        pass

    @abstractmethod
    def handle_relative_constructions(self, token):
        pass

    @abstractmethod
    def holmes_lemma(self, token):
        pass

    @abstractmethod
    def is_interrogative_pronoun(self, token):
        pass

    def derived_holmes_lemma(self, token, lemma):
        if lemma in self.derivational_dictionary:
            derived_lemma = self.derivational_dictionary[lemma]
            if lemma == derived_lemma: # basis entry, so do not call language specific method
                return None
            else:
                return derived_lemma
        else:
            return self.language_specific_derived_holmes_lemma(token, lemma)

    @abstractmethod
    def language_specific_derived_holmes_lemma(self, token, lemma):
        pass

    def initialize_semantic_dependencies(self, token):
        for child in (
                child for child in token.children if child.dep_ != 'punct' and
                child.tag_ not in self.semantic_dependency_excluded_tags):
            token._.holmes.children.append(SemanticDependency(token.i, child.i, child.dep_))

    def set_initial_question_words(self, doc:Doc):
        """ True on a token that represents an interrogative pronoun within an initial phrase. """
        initial_sentence = next(doc.sents, None)
        if initial_sentence is not None:
            visited = set()
            working_first_phrase_head = doc[0]
            while working_first_phrase_head.head is not None and not \
                    working_first_phrase_head.head.pos_ in ('VERB', 'AUX') \
                    and not working_first_phrase_head.head in visited:
                visited.add(working_first_phrase_head)
                working_first_phrase_head = working_first_phrase_head.head
            for token in initial_sentence:
                if self.is_interrogative_pronoun(token) and \
                        token in working_first_phrase_head.subtree:
                    token._.holmes.is_initial_question_word = True

    def mark_if_righthand_sibling(self, token):
        if token.dep_ in self.sibling_marker_deps:  # i.e. is righthand sibling
            working_token = token
            working_or_conjunction_flag = False
            # work up through the tree until the lefthandmost sibling element with the
            # semantic relationships to the rest of the sentence is reached
            while working_token.dep_ in self.conjunction_deps:
                working_token = working_token.head
                for working_child in working_token.children:
                    if working_child.lemma_ == self.or_lemma:
                        working_or_conjunction_flag = True
            # add this element to the lefthandmost sibling as a righthand sibling
            working_token._.holmes.righthand_siblings.append(token.i)
            if working_or_conjunction_flag:
                working_token._.holmes.is_involved_in_or_conjunction = True

    def copy_any_sibling_info(self, token):
        # Copy the or conjunction flag to righthand siblings
        if token._.holmes.is_involved_in_or_conjunction:
            for righthand_sibling in token._.holmes.righthand_siblings:
                token.doc[righthand_sibling]._.holmes.is_involved_in_or_conjunction = True
        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.child_index >= 0):
            # where a token has a dependent token and the dependent token has righthand siblings,
            # add dependencies from the parent token to the siblings
            for child_righthand_sibling in \
                    token.doc[dependency.child_index]._.holmes.righthand_siblings:
                # Check this token does not already have the dependency
                if len([dependency for dependency in token._.holmes.children if
                        dependency.child_index == child_righthand_sibling]) == 0:
                    child_index_to_add = child_righthand_sibling
                    # If this token is a grammatical element, it needs to point to new
                    # child dependencies as a grammatical element as well
                    if dependency.child_index < 0:
                        child_index_to_add = 0 - (child_index_to_add + 1)
                    # Check adding the new dependency will not result in a loop and that
                    # this token still does not have the dependency now its index has
                    # possibly been changed
                    if token.i != child_index_to_add and not \
                            token._.holmes.has_dependency_with_child_index(child_index_to_add):
                        token._.holmes.children.append(SemanticDependency(
                            token.i, child_index_to_add, dependency.label, dependency.is_uncertain))
            # where a token has a dependent token and the parent token has righthand siblings,
            # add dependencies from the siblings to the dependent token, unless the dependent
            # token is to the right of the parent token but to the left of the sibling.
            for righthand_sibling in (
                    righthand_sibling for righthand_sibling in
                    token._.holmes.righthand_siblings if righthand_sibling !=
                    dependency.child_index and (
                        righthand_sibling < dependency.child_index or
                        dependency.child_index < token.i)):
                # unless the sibling already contains a dependency with the same label
                # or the sibling has this token as a dependent child
                righthand_sibling_token = token.doc[righthand_sibling]
                if len([sibling_dependency for sibling_dependency in
                        righthand_sibling_token._.holmes.children if
                        sibling_dependency.label == dependency.label and not
                        token._.holmes.has_dependency_with_child_index(
                        sibling_dependency.child_index)]) == 0 and \
                        dependency.label not in self.conjunction_deps and not \
                        righthand_sibling_token._.holmes.has_dependency_with_child_index(
                            dependency.child_index) \
                        and righthand_sibling != dependency.child_index:
                    righthand_sibling_token._.holmes.children.append(SemanticDependency(
                        righthand_sibling, dependency.child_index, dependency.label,
                        self.mark_child_dependencies_copied_to_siblings_as_uncertain
                        or dependency.is_uncertain))

    def normalize_predicative_adjectives(self, token):
        """Change phrases like *the town is old* and *the man is poor* so their
            semantic structure is equivalent to *the old town* and *the poor man*.
        """
        if token.pos_ in self.adjectival_predicate_head_pos:
            altered = False
            for predicative_adjective_index in (
                    dependency.child_index for dependency in \
                    token._.holmes.children if dependency.label ==
                    self.adjectival_predicate_predicate_dep and
                    token.doc[dependency.child_index].pos_ ==
                    self.adjectival_predicate_predicate_pos and
                    dependency.child_index >= 0):
                for subject_index in (
                        dependency.child_index for dependency in
                        token._.holmes.children if dependency.label ==
                        self.adjectival_predicate_subject_dep and (
                            dependency.child_token(token.doc).pos_ in
                            self.adjectival_predicate_subject_pos or
                            dependency.child_token(token.doc)._.holmes.is_involved_in_coreference()
                        and dependency.child_index >= 0
                        and dependency.child_index != predicative_adjective_index)):
                    token.doc[subject_index]._.holmes.children.append(
                        SemanticDependency(
                            subject_index, predicative_adjective_index, self.modifier_dep))
                    altered = True
            if altered:
                token._.holmes.children = [SemanticDependency(
                    token.i, 0 - (subject_index + 1), None)]

    def create_additional_preposition_phrase_semantic_dependencies(self, token):
        """In structures like 'Somebody needs insurance for a period' it seems to be
            mainly language-dependent whether the preposition phrase is analysed as being
            dependent on the preceding noun or the preceding verb. We add an additional, new
            dependency to whichever of the noun or the verb does not already have one. In English,
            the new label is defined in *match_implication_dict* in such a way that original
            dependencies in search phrases match new dependencies in documents but not vice versa.
            This restriction is not applied in German because the fact the verb can be placed in
            different positions within the sentence means there is considerable variation around
            how prepositional phrases are analyzed by spaCy.
        """

        def add_dependencies_pointing_to_preposition_and_siblings(parent, label):
            for working_preposition in token._.holmes.loop_token_and_righthand_siblings(token.doc):
                if parent.i != working_preposition.i:
                    parent._.holmes.children.append(SemanticDependency(
                        parent.i, working_preposition.i, label, True))

        # token is a preposition ...
        if token.pos_ == 'ADP':
            # directly preceded by a noun
            if token.i > 0 and token.doc[token.i-1].sent == token.sent and \
                    token.doc[token.i-1].pos_ in ('NOUN', 'PROPN', 'PRON'):
                preceding_noun = token.doc[token.i-1]
                # and the noun is governed by at least one verb
                governing_verbs = [
                    working_token for working_token in token.sent
                    if working_token.pos_ == 'VERB' and
                    working_token._.holmes.has_dependency_with_child_index(
                        preceding_noun.i)]
                if len(governing_verbs) == 0:
                    return
                # if the noun governs the preposition, add new possible dependencies
                # from the verb(s)
                for governing_verb in governing_verbs:
                    if preceding_noun._.holmes.has_dependency_with_child_index_and_label(
                            token.i, self.spacy_noun_to_preposition_dep) and not \
                            governing_verb._.holmes.has_dependency_with_child_index_and_label(
                                token.i, self.spacy_verb_to_preposition_dep):
                        add_dependencies_pointing_to_preposition_and_siblings(
                            governing_verb, self.holmes_verb_to_preposition_dep)
                # if the verb(s) governs the preposition, add new possible dependencies
                # from the noun
                if governing_verbs[0]._.holmes.has_dependency_with_child_index_and_label(
                        token.i, self.spacy_verb_to_preposition_dep) and not \
                        preceding_noun._.holmes.has_dependency_with_child_index_and_label(
                            token.i, self.spacy_noun_to_preposition_dep):
                    # check the preposition is not pointing back to a relative clause
                    for preposition_dep_index in (
                            dep.child_index for dep in token._.holmes.children
                            if dep.child_index >= 0):
                        if token.doc[preposition_dep_index]._.holmes.\
                                has_dependency_with_label('relcl'):
                            return
                    add_dependencies_pointing_to_preposition_and_siblings(
                        preceding_noun, self.holmes_noun_to_preposition_dep)

    def set_matchability(self, token):
        """Marks whether this token, if it appears in a search phrase, should require a counterpart
        in a document being matched.
        """
        token._.holmes.is_matchable = (
            token.pos_ in self.matchable_pos or token._.holmes.is_involved_in_coreference()
            or len(token._.holmes.subwords) > 0) \
            and token.tag_ not in self.interrogative_pronoun_tags and \
            token._.holmes.lemma not in self.generic_pronoun_lemmas

    def move_information_between_tokens(self, from_token, to_token):
        """Moves semantic child and sibling information from one token to another.

        Args:

        from_token -- the source token, which will be marked as a grammatical token
        pointing to *to_token*.
        to_token -- the destination token.
        """
        linking_dependencies = [
            dependency for dependency in from_token._.holmes.children
            if dependency.child_index == to_token.i]
        if len(linking_dependencies) == 0:
            return  # should only happen if there is a problem with the spaCy structure
        # only loop dependencies whose label or index are not already present at the destination
        for dependency in (
                dependency for dependency in from_token._.holmes.children
                if not to_token._.holmes.has_dependency_with_child_index(dependency.child_index)
                and to_token.i != dependency.child_index and
                to_token.i not in to_token.doc[dependency.child_index]._.holmes.righthand_siblings
                and dependency.child_index not in to_token._.holmes.righthand_siblings):
            to_token._.holmes.children.append(SemanticDependency(
                to_token.i, dependency.child_index, dependency.label, dependency.is_uncertain))
        from_token._.holmes.children = [SemanticDependency(from_token.i, 0 - (to_token.i + 1))]
        to_token._.holmes.righthand_siblings.extend(
            from_token._.holmes.righthand_siblings)
        from_token._.holmes.righthand_siblings = []
        if from_token._.holmes.is_involved_in_or_conjunction:
            to_token._.holmes.is_involved_in_or_conjunction = True
        if from_token._.holmes.is_negated:
            to_token._.holmes.is_negated = True
        # If from_token is the righthand sibling of some other token within the same sentence,
        # replace that token's reference with a reference to to_token
        for token in from_token.sent:
            if from_token.i in token._.holmes.righthand_siblings:
                token._.holmes.righthand_siblings.remove(from_token.i)
                if token.i != to_token.i:
                    token._.holmes.righthand_siblings.append(to_token.i)

    def create_convenience_dependencies(self, token):
        for child_dependency in (
                child_dependency for child_dependency in token._.holmes.children
                if child_dependency.child_index >= 0):
            child_token = child_dependency.child_token(token.doc)
            child_token._.holmes.parents.append(child_dependency)
        for linked_parent_index in token._.holmes.token_and_coreference_chain_indexes:
            linked_parent = token.doc[linked_parent_index]
            for child_dependency in (
                    child_dependency for child_dependency in linked_parent._.holmes.children
                    if child_dependency.child_index >= 0):
                child_token = child_dependency.child_token(token.doc)
                for linked_child_index in \
                        child_token._.holmes.token_and_coreference_chain_indexes:
                    linked_child = token.doc[linked_child_index]
                    token._.holmes.coreference_linked_child_dependencies.append([
                        linked_child.i, child_dependency.label])
                    linked_child._.holmes.coreference_linked_parent_dependencies.append([
                        token.i, child_dependency.label])

class LinguisticObjectFactory:

    def __init__(
            self, semantic_analyzer, semantic_matching_helper, ontology,
            overall_similarity_threshold, embedding_based_matching_on_root_words,
            analyze_derivational_morphology, perform_coreference_resolution):
        """Args:

        semantic_analyzer -- the *SemanticAnalyzer* object to use
        semantic_matching_helper -- the *SemanticMatchingHelper* object to use
        ontology -- optionally, an *Ontology* object to use in matching
        overall_similarity_threshold -- if embedding-based matching is to be activated, a float
            value between 0 and 1. A match between a search phrase and a document is then valid
            if the geometric mean of all the similarities between search phrase tokens and
            document tokens is this value or greater. If this value is set to 1.0,
            embedding-based matching is deactivated.
        embedding_based_matching_on_root_words -- determines whether or not embedding-based
            matching should be attempted on search-phrase root tokens, which has a considerable
            performance hit. Defaults to *False*.
        analyze_derivational_morphology -- *True* if matching should be attempted between different
            words from the same word family. Defaults to *True*.
        perform_coreference_resolution -- *True* if coreference resolution should be performed.
        """
        self.semantic_analyzer = semantic_analyzer
        self.semantic_matching_helper = semantic_matching_helper
        self.ontology = ontology
        self.overall_similarity_threshold = overall_similarity_threshold
        self.embedding_based_matching_on_root_words = embedding_based_matching_on_root_words
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.perform_coreference_resolution = perform_coreference_resolution

    def add_phraselets_to_dict(
            self, doc, *, phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors, match_all_words,
            ignore_relation_phraselets, include_reverse_only, stop_lemmas, stop_tags,
            reverse_only_parent_lemmas, words_to_corpus_frequencies, maximum_corpus_frequency,
            process_initial_question_words):
        """ Creates topic matching phraselets extracted from a matching text.

        Properties:

        doc -- the Holmes-parsed document
        phraselet_labels_to_phraselet_infos -- a dictionary from labels to phraselet info objects
            that are used to generate phraselet search phrases.
        replace_with_hypernym_ancestors -- if 'True', all words present in the ontology
            are replaced with their most general (highest) ancestors.
        match_all_words -- if 'True', word phraselets are generated for all matchable words
            rather than just for words whose tags match the phraselet template; multiwords
            are not taken into account when processing single-word phraselets; and single-word
            phraselets are generated for subwords.
        ignore_relation_phraselets -- if 'True', only single-word phraselets are processed.
        include_reverse_only -- whether to generate phraselets that are only reverse-matched.
            Reverse matching is used in topic matching but not in supervised document
            classification.
        stop_lemmas -- lemmas that should prevent all types of phraselet production.
        stop_tags -- tags that should prevent all types of phraselet production.
        reverse_only_parent_lemmas -- lemma / part-of-speech combinations that, when present at
            the parent pole of a relation phraselet, should cause that phraselet to be
            reverse-matched.
        words_to_corpus_frequencies -- a dictionary from words to the number of times each
            word occurs in the indexed documents.
        maximum_corpus_frequency -- the maximum value within *words_to_corpus_frequencies*.
        process_initial_question_words -- *True* if interrogative pronouns are permitted within phraselets.
        """

        index_to_lemmas_cache = {}
        def get_lemmas_from_index(index):
            """ Returns the lemma and the derived lemma. Phraselets form a special case where
                the derived lemma is set even if it is identical to the lemma. This is necessary
                because the lemma may be set to a different value during the lifecycle of the
                object. The property getter in the SemanticDictionary class ensures that
                derived_lemma is None is always returned where the two strings are identical.
            """
            if index in index_to_lemmas_cache:
                return index_to_lemmas_cache[index]
            token = doc[index.token_index]
            if self.semantic_matching_helper.is_entity_search_phrase_token(token, False):
                # False in order to get text rather than lemma
                index_to_lemmas_cache[index] = token.text, token.text
                return token.text, token.text
                # keep the text, because the lemma will be lowercase
            if index.is_subword():
                lemma = token._.holmes.subwords[index.subword_index].lemma
                if self.analyze_derivational_morphology:
                    derived_lemma = token._.holmes.subwords[index.subword_index].\
                        lemma_or_derived_lemma()
                else:
                    derived_lemma = lemma
                if self.ontology is not None and self.analyze_derivational_morphology:
                    for reverse_derived_word in self.semantic_matching_helper.\
                            reverse_derived_lemmas_in_ontology(
                            token._.holmes.subwords[index.subword_index]):
                        derived_lemma = reverse_derived_word.lower()
                        break
            else:
                lemma = token._.holmes.lemma
                if self.analyze_derivational_morphology:
                    derived_lemma = token._.holmes.lemma_or_derived_lemma()
                else:
                    derived_lemma = lemma
                if self.ontology is not None and not self.ontology.contains(lemma):
                    if self.ontology.contains(token.text.lower()):
                        lemma = derived_lemma = token.text.lower()
                    # ontology contains text but not lemma, so return text
                if self.ontology is not None and self.analyze_derivational_morphology:
                    for reverse_derived_word in self.semantic_matching_helper.\
                            reverse_derived_lemmas_in_ontology(token):
                        derived_lemma = reverse_derived_word.lower()
                        break
                        # ontology contains a word pointing to the same derived lemma,
                        # so return that. Note that if there are several such words the same
                        # one will always be returned.
            index_to_lemmas_cache[index] = lemma, derived_lemma
            return lemma, derived_lemma

        def replace_lemmas_with_most_general_ancestor(lemma, derived_lemma):
            new_derived_lemma = self.ontology.get_most_general_hypernym_ancestor(
                derived_lemma).lower()
            if derived_lemma != new_derived_lemma:
                lemma = derived_lemma = new_derived_lemma
            return lemma, derived_lemma

        def lemma_replacement_indicated(existing_lemma, existing_pos, new_lemma, new_pos):
            if existing_lemma is None:
                return False
            if not existing_pos in self.semantic_matching_helper.preferred_phraselet_pos and \
                    new_pos in self.semantic_matching_helper.preferred_phraselet_pos:
                return True
            if existing_pos in self.semantic_matching_helper.preferred_phraselet_pos and \
                     new_pos not in self.semantic_matching_helper.preferred_phraselet_pos:
                return False
            return len(new_lemma) < len(existing_lemma)

        def add_new_phraselet_info(
                phraselet_label, phraselet_template, created_without_matching_tags,
                is_reverse_only_parent_lemma, parent_lemma, parent_derived_lemma,
                parent_pos, parent_ent_type, child_lemma, child_derived_lemma, child_pos,
                child_ent_type):

            def get_frequency_factor_for_pole(parent): # pole is 'True' -> parent, 'False' -> child
                original_word_set = {parent_lemma, parent_derived_lemma} if parent else \
                    {child_lemma, child_derived_lemma}
                word_set_including_any_ontology = original_word_set.copy()
                if self.ontology is not None:
                    for word in original_word_set:
                        for word_matching, _ in \
                                self.ontology.get_words_matching_and_depths(word):
                            word_set_including_any_ontology.add(word_matching)
                frequencies = []
                for word in word_set_including_any_ontology:
                    if word in words_to_corpus_frequencies:
                        frequencies.append(float(words_to_corpus_frequencies[word]))
                if len(frequencies) == 0:
                    return 1.0
                adjusted_max_frequency = max(frequencies) - 1.0
                if adjusted_max_frequency <= 0.0:
                    return 1.0
                return 1 - (math.log(adjusted_max_frequency) / math.log(maximum_corpus_frequency))

            frequency_factor = parent_frequency_factor = child_frequency_factor = None
            if words_to_corpus_frequencies is not None:
                parent_frequency_factor = get_frequency_factor_for_pole(True)
                frequency_factor = parent_frequency_factor
                if child_lemma is not None:
                    child_frequency_factor = get_frequency_factor_for_pole(False)
                    frequency_factor *= child_frequency_factor
            if phraselet_label not in phraselet_labels_to_phraselet_infos:
                phraselet_labels_to_phraselet_infos[phraselet_label] = PhraseletInfo(
                    phraselet_label, phraselet_template.label, parent_lemma,
                    parent_derived_lemma, parent_pos, parent_ent_type, child_lemma,
                    child_derived_lemma, child_pos, child_ent_type, created_without_matching_tags,
                    is_reverse_only_parent_lemma, frequency_factor, parent_frequency_factor,
                    child_frequency_factor)
            else:
                existing_phraselet = phraselet_labels_to_phraselet_infos[phraselet_label]
                if lemma_replacement_indicated(
                        existing_phraselet.parent_lemma, existing_phraselet.parent_pos,
                        parent_lemma, parent_pos):
                    existing_phraselet.parent_lemma = parent_lemma
                    existing_phraselet.parent_pos = parent_pos
                if lemma_replacement_indicated(
                        existing_phraselet.child_lemma, existing_phraselet.child_pos, child_lemma,
                        child_pos):
                    existing_phraselet.child_lemma = child_lemma
                    existing_phraselet.child_pos = child_pos

        def process_single_word_phraselet_templates(
                token, subword_index, checking_tags, token_indexes_to_multiword_lemmas):
            for phraselet_template in (
                    phraselet_template for phraselet_template in
                    self.semantic_matching_helper.phraselet_templates if
                    phraselet_template.single_word() and (
                        token._.holmes.is_matchable or subword_index is not None)):
                        # see note below for explanation
                if (not checking_tags or token.tag_ in phraselet_template.parent_tags) and \
                        token.tag_ not in stop_tags:
                    if token.i in token_indexes_to_multiword_lemmas and not match_all_words:
                        lemma = derived_lemma = token_indexes_to_multiword_lemmas[token.i]
                    else:
                        lemma, derived_lemma = get_lemmas_from_index(Index(token.i, subword_index))
                    if self.ontology is not None and replace_with_hypernym_ancestors:
                        lemma, derived_lemma = replace_lemmas_with_most_general_ancestor(
                            lemma, derived_lemma)
                    phraselet_label = ''.join((phraselet_template.label, ': ', derived_lemma))
                    if derived_lemma not in stop_lemmas and derived_lemma != 'ENTITYNOUN':
                        # ENTITYNOUN has to be excluded as single word although it is still
                        # permitted as the child of a relation phraselet template
                        add_new_phraselet_info(
                            phraselet_label, phraselet_template, not checking_tags,
                            None, lemma, derived_lemma, token.pos_, token.ent_type_,
                            None, None, None, None)

        def add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(index_list):
            # for each token in the list, find out whether it has subwords and if so add the
            # head subword to the list
            for index in index_list.copy():
                token = doc[index.token_index]
                for subword in (
                        subword for subword in token._.holmes.subwords if
                        subword.is_head and subword.containing_token_index == token.i):
                    index_list.append(Index(token.i, subword.index))
            # if one or more subwords do not belong to this token, it is a hyphenated word
            # within conjunction and the whole word should not be used to build relation phraselets.
                if len([
                        subword for subword in token._.holmes.subwords if
                        subword.containing_token_index != token.i]) > 0:
                    index_list.remove(index)

        self.redefine_multiwords_on_head_tokens(doc)
        token_indexes_to_multiword_lemmas = {}
        token_indexes_within_multiwords_to_ignore = []
        for token in (token for token in doc if len(token._.holmes.lemma.split()) == 1):
            entity_defined_multiword, indexes = \
                self.semantic_matching_helper.get_entity_defined_multiword(token)
            if entity_defined_multiword is not None:
                for index in indexes:
                    if index == token.i:
                        token_indexes_to_multiword_lemmas[token.i] = entity_defined_multiword
                    else:
                        token_indexes_within_multiwords_to_ignore.append(index)
        for token in doc:
            if token.i in token_indexes_within_multiwords_to_ignore:
                if match_all_words:
                    process_single_word_phraselet_templates(
                        token, None, False, token_indexes_to_multiword_lemmas)
                continue
            if len([
                    subword for subword in token._.holmes.subwords if
                    subword.containing_token_index != token.i]) == 0:
                # whole single words involved in subword conjunction should not be included as
                # these are partial words including hyphens.
                process_single_word_phraselet_templates(
                    token, None, not match_all_words, token_indexes_to_multiword_lemmas)
            if match_all_words:
                for subword in (
                        subword for subword in token._.holmes.subwords if
                        token.i == subword.containing_token_index):
                    process_single_word_phraselet_templates(
                        token, subword.index, False, token_indexes_to_multiword_lemmas)
            if ignore_relation_phraselets:
                continue
            if self.perform_coreference_resolution:
                parents = [
                    Index(token_index, None) for token_index in
                    token._.holmes.token_and_coreference_chain_indexes]
            else:
                parents = [Index(token.i, None)]
            add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(parents)
            for parent in parents:
                for dependency in (
                        dependency for dependency in doc[parent.token_index]._.holmes.children
                        if dependency.child_index not in token_indexes_within_multiwords_to_ignore):
                    if self.perform_coreference_resolution:
                        children = [
                            Index(token_index, None) for token_index in
                            dependency.child_token(doc)._.holmes.
                            token_and_coreference_chain_indexes]
                    else:
                        children = [Index(dependency.child_token(doc).i, None)]
                    add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(
                        children)
                    for child in children:
                        for phraselet_template in (
                                phraselet_template for phraselet_template in
                                self.semantic_matching_helper.phraselet_templates if not
                                phraselet_template.single_word() and (
                                    not phraselet_template.reverse_only or include_reverse_only)):
                            if dependency.label in \
                                    phraselet_template.dependency_labels and \
                                    doc[parent.token_index].tag_ in phraselet_template.parent_tags\
                                    and doc[child.token_index].tag_ in \
                                    phraselet_template.child_tags and \
                                    doc[parent.token_index]._.holmes.is_matchable and \
                                    (doc[child.token_index]._.holmes.is_matchable or
                                    (process_initial_question_words and
                                    doc[child.token_index]._.holmes.is_initial_question_word)):
                                if parent.token_index in token_indexes_to_multiword_lemmas:
                                    parent_lemma = parent_derived_lemma = \
                                        token_indexes_to_multiword_lemmas[parent.token_index]
                                else:
                                    parent_lemma, parent_derived_lemma = \
                                        get_lemmas_from_index(parent)
                                if self.ontology is not None and replace_with_hypernym_ancestors:
                                    parent_lemma, parent_derived_lemma = \
                                        replace_lemmas_with_most_general_ancestor(
                                            parent_lemma, parent_derived_lemma)
                                if child.token_index in token_indexes_to_multiword_lemmas:
                                    child_lemma = child_derived_lemma = \
                                        token_indexes_to_multiword_lemmas[child.token_index]
                                else:
                                    child_lemma, child_derived_lemma = get_lemmas_from_index(child)
                                if self.ontology is not None and replace_with_hypernym_ancestors:
                                    child_lemma, child_derived_lemma = \
                                        replace_lemmas_with_most_general_ancestor(
                                            child_lemma, child_derived_lemma)
                                phraselet_label = ''.join((
                                    phraselet_template.label, ': ', parent_derived_lemma,
                                    '-', child_derived_lemma))
                                is_reverse_only_parent_lemma = False
                                if reverse_only_parent_lemmas is not None:
                                    for entry in reverse_only_parent_lemmas:
                                        if entry[0] == doc[parent.token_index]._.holmes.lemma \
                                                and entry[1] == doc[parent.token_index].pos_:
                                            is_reverse_only_parent_lemma = True
                                if parent_lemma not in stop_lemmas and child_lemma not in \
                                        stop_lemmas and not (
                                            is_reverse_only_parent_lemma
                                            and not include_reverse_only):
                                    add_new_phraselet_info(
                                        phraselet_label, phraselet_template, match_all_words,
                                        is_reverse_only_parent_lemma,
                                        parent_lemma, parent_derived_lemma,
                                        doc[parent.token_index].pos_,
                                        doc[parent.token_index].ent_type_,
                                        child_lemma, child_derived_lemma,
                                        doc[child.token_index].pos_,
                                        doc[child.token_index].ent_type_)

            # We do not check for matchability in order to catch pos_='X', tag_='TRUNC'. This
            # is not a problem as only a limited range of parts of speech receive subwords in
            # the first place.
            for subword in (
                    subword for subword in token._.holmes.subwords if
                    subword.dependent_index is not None):
                parent_subword_index = subword.index
                child_subword_index = subword.dependent_index
                if token._.holmes.subwords[parent_subword_index].containing_token_index != \
                        token.i and \
                        token._.holmes.subwords[child_subword_index].containing_token_index != \
                        token.i:
                    continue
                for phraselet_template in (
                        phraselet_template for phraselet_template in
                        self.semantic_matching_helper.phraselet_templates if not
                        phraselet_template.single_word() and (
                            not phraselet_template.reverse_only or include_reverse_only)
                        and subword.dependency_label in phraselet_template.dependency_labels and
                        token.tag_ in phraselet_template.parent_tags):
                    parent_lemma, parent_derived_lemma = get_lemmas_from_index(Index(
                        token.i, parent_subword_index))
                    if self.ontology is not None and replace_with_hypernym_ancestors:
                        parent_lemma, parent_derived_lemma = \
                            replace_lemmas_with_most_general_ancestor(
                                parent_lemma, parent_derived_lemma)
                    child_lemma, child_derived_lemma = get_lemmas_from_index(Index(
                        token.i, child_subword_index))
                    if self.ontology is not None and replace_with_hypernym_ancestors:
                        child_lemma, child_derived_lemma = \
                                replace_lemmas_with_most_general_ancestor(
                                    child_lemma, child_derived_lemma)
                    phraselet_label = ''.join((
                        phraselet_template.label, ': ', parent_derived_lemma, '-',
                        child_derived_lemma))
                    add_new_phraselet_info(
                        phraselet_label, phraselet_template, match_all_words,
                        False, parent_lemma, parent_derived_lemma, token.pos_, token.ent_type_,
                        child_lemma, child_derived_lemma, token.pos_, token.ent_type_)
        if len(phraselet_labels_to_phraselet_infos) == 0 and not match_all_words:
            for token in doc:
                process_single_word_phraselet_templates(
                    token, None, False, token_indexes_to_multiword_lemmas)

    def create_search_phrases_from_phraselet_infos(self, phraselet_infos,
            reverse_matching_frequency_threshold=None):
        """ Creates search phrases from phraselet info objects, returning a dictionary from
            phraselet labels to the created search phrases.

            reverse_matching_frequency_threshold: an optional threshold between 0.0 and 1.0.
                Where the parent word in a phraselet has a frequency factor below the threshold,
                the search phrase will be set to
                *treat_as_reverse_only_during_initial_relation_matching=True*.
        """

        def create_phraselet_label(phraselet_info):
            if phraselet_info.child_lemma is not None:
                return ''.join((
                    phraselet_info.template_label, ': ', phraselet_info.parent_derived_lemma, '-',
                    phraselet_info.child_derived_lemma))
            else:
                return ''.join((
                    phraselet_info.template_label, ': ', phraselet_info.parent_derived_lemma))

        def create_search_phrase_from_phraselet(phraselet_info):
            for phraselet_template in self.semantic_matching_helper.phraselet_templates:
                if phraselet_info.template_label == phraselet_template.label:
                    phraselet_doc = phraselet_template.template_doc.copy()
                    phraselet_doc[phraselet_template.parent_index]._.holmes.lemma = \
                        phraselet_info.parent_lemma
                    phraselet_doc[phraselet_template.parent_index]._.holmes.derived_lemma = \
                        phraselet_info.parent_derived_lemma
                    phraselet_doc[phraselet_template.parent_index]._.holmes.ent_type = \
                        phraselet_info.parent_ent_type
                    if phraselet_info.child_lemma is not None:
                        phraselet_doc[phraselet_template.child_index]._.holmes.lemma = \
                            phraselet_info.child_lemma
                        phraselet_doc[phraselet_template.child_index]._.holmes.derived_lemma = \
                            phraselet_info.child_derived_lemma
                        phraselet_doc[phraselet_template.child_index]._.holmes.ent_type = \
                            phraselet_info.child_ent_type
                    return self.create_search_phrase(
                        'topic match phraselet', phraselet_doc,
                        create_phraselet_label(phraselet_info), phraselet_template,
                        phraselet_info.created_without_matching_tags,
                        reverse_matching_frequency_threshold is not None and
                        phraselet_info.parent_frequency_factor <
                        reverse_matching_frequency_threshold and
                        phraselet_info.child_lemma is not None,
                        phraselet_info.reverse_only_parent_lemma,
                        True)
            raise RuntimeError(''.join((
                'Phraselet template', phraselet_info.template_label, 'not found.')))

        return {
            create_phraselet_label(phraselet_info) :
            create_search_phrase_from_phraselet(phraselet_info) for phraselet_info in
            phraselet_infos}

    def redefine_multiwords_on_head_tokens(self, doc):

        def loop_textual_representations(multiword_span):
            for representation, _ in self.semantic_matching_helper.loop_textual_representations(
                    multiword_span):
                yield representation, multiword_span.derived_lemma
            if self.analyze_derivational_morphology:
                for reverse_derived_lemma in \
                        self.semantic_matching_helper.reverse_derived_lemmas_in_ontology(
                        multiword_span):
                    yield reverse_derived_lemma, multiword_span.derived_lemma

        if self.ontology is not None:
            for token in (token for token in doc if len(token._.holmes.lemma.split()) == 1):
                matched = False
                for multiword_span in \
                        self.semantic_matching_helper.multiword_spans_with_head_token(token):
                    for representation, derived_lemma in \
                            loop_textual_representations(multiword_span):
                        if self.ontology.contains_multiword(representation):
                            matched = True
                            token._.holmes.lemma = representation.lower()
                            token._.holmes.derived_lemma = derived_lemma
                            # mark the dependent tokens as grammatical and non-matchable
                            for multiword_token in (
                                    multiword_token for multiword_token in multiword_span.tokens
                                    if multiword_token.i != token.i):
                                multiword_token._.holmes.children = [SemanticDependency(
                                    multiword_token.i, 0 - (token.i + 1), None)]
                                multiword_token._.holmes.is_matchable = False
                            break
                    if matched:
                        break

    def get_phraselet_labels_to_phraselet_infos(self, *, text_to_match_doc,
            words_to_corpus_frequencies, maximum_corpus_frequency, process_initial_question_words):
        phraselet_labels_to_phraselet_infos = {}
        self.add_phraselets_to_dict(
            text_to_match_doc,
            phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors=False,
            match_all_words=False,
            ignore_relation_phraselets=False,
            include_reverse_only=True,
            stop_tags=self.semantic_matching_helper.topic_matching_phraselet_stop_tags,
            stop_lemmas=self.semantic_matching_helper.topic_matching_phraselet_stop_lemmas,
            reverse_only_parent_lemmas=
            self.semantic_matching_helper.topic_matching_reverse_only_parent_lemmas,
            words_to_corpus_frequencies=words_to_corpus_frequencies,
            maximum_corpus_frequency=maximum_corpus_frequency,
            process_initial_question_words=process_initial_question_words)

        # now add the single word phraselets whose tags did not match.
        self.add_phraselets_to_dict(
            text_to_match_doc,
            phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors=False,
            match_all_words=True,
            ignore_relation_phraselets=True,
            include_reverse_only=False, # value is irrelevant with
                                        # ignore_relation_phraselets == True
            stop_lemmas=self.semantic_matching_helper.topic_matching_phraselet_stop_lemmas,
            stop_tags=self.semantic_matching_helper.topic_matching_phraselet_stop_tags,
            reverse_only_parent_lemmas=
            self.semantic_matching_helper.topic_matching_reverse_only_parent_lemmas,
            words_to_corpus_frequencies=words_to_corpus_frequencies,
            maximum_corpus_frequency=maximum_corpus_frequency,
            process_initial_question_words=False)
        return phraselet_labels_to_phraselet_infos

    def create_search_phrase(
            self, search_phrase_text, search_phrase_doc,
            label, phraselet_template, topic_match_phraselet_created_without_matching_tags,
            treat_as_reverse_only_during_initial_relation_matching,
            is_reverse_only_parent_lemma, process_initial_question_words):
        """phraselet_template -- 'None' if this search phrase is not a topic match phraselet"""

        def replace_grammatical_root_token_recursively(token):
            """Where the syntactic root of a search phrase document is a grammatical token or is
                marked as non-matchable, loop through the semantic dependencies to find the
                semantic root.
            """
            for dependency in token._.holmes.children:
                if dependency.child_index < 0:
                    return replace_grammatical_root_token_recursively(
                        token.doc[(0 - dependency.child_index) - 1])
            if not token._.holmes.is_matchable:
                for dependency in token._.holmes.children:
                    if dependency.child_index >= 0 and \
                            dependency.child_token(token.doc)._.holmes.is_matchable:
                        return replace_grammatical_root_token_recursively(
                            token.doc[dependency.child_index])
            return token

        def add_word_information(word, match_type, depth):
            if word not in words_matching_root_token:
                words_matching_root_token.append(word)
            if not word in root_word_to_match_info_dict:
                root_word_to_match_info_dict[word] = (match_type, depth)

        def add_word_information_from_ontology(word):
            for entry_word, entry_depth in \
                    self.semantic_matching_helper.ontology.get_words_matching_and_depths(word):
                add_word_information(entry_word, 'ontology', entry_depth)
                if self.semantic_matching_helper.analyze_derivational_morphology:
                    working_derived_lemma = self.semantic_analyzer.derived_holmes_lemma(
                            None, entry_word.lower())
                    if working_derived_lemma is not None:
                        add_word_information(working_derived_lemma, 'ontology', entry_depth)

        if phraselet_template is None:
            self.redefine_multiwords_on_head_tokens(search_phrase_doc)
            # where a multiword exists as an ontology entry, the multiword should be used for
            # matc0hing rather than the individual words. Not relevant for topic matching
            # phraselets because the multiword will already have been set as the Holmes
            # lemma of the word.

        for token in search_phrase_doc:
            if len(token._.holmes.righthand_siblings) > 0:
                # SearchPhrases may not themselves contain conjunctions like 'and'
                # because then the matching becomes too complicated
                raise SearchPhraseContainsConjunctionError(search_phrase_text)
            if token._.holmes.is_negated:
                # SearchPhrases may not themselves contain negation
                # because then the matching becomes too complicated
                raise SearchPhraseContainsNegationError(search_phrase_text)
            if self.perform_coreference_resolution and token.pos_ == 'PRON' and \
                    token._.holmes.is_involved_in_coreference():
                # SearchPhrases may not themselves contain coreferring pronouns
                # because then the matching becomes too complicated
                raise SearchPhraseContainsCoreferringPronounError(search_phrase_text)

        root_tokens = []
        tokens_to_match = []
        matchable_non_entity_tokens_to_vectors = {}
        for token in search_phrase_doc:
            # check whether grammatical token
            if phraselet_template is not None and phraselet_template.parent_index != token.i and \
                    phraselet_template.child_index != token.i:
                token._.holmes.is_matchable = False
            if phraselet_template is not None and phraselet_template.parent_index == token.i and \
                    not phraselet_template.single_word() and \
                    phraselet_template.assigned_dependency_label is not None:
                for dependency in (
                        dependency for dependency in token._.holmes.children if \
                        dependency.child_index == phraselet_template.child_index):
                    dependency.label = phraselet_template.assigned_dependency_label
            if token._.holmes.is_matchable and not (
                    len(token._.holmes.children) > 0 and
                    token._.holmes.children[0].child_index < 0):
                tokens_to_match.append(token)
                if not self.semantic_matching_helper.is_entity_search_phrase_token(
                        token, phraselet_template is not None):
                    if phraselet_template is None and len(token._.holmes.lemma.split()) > 1:
                        working_lexeme = self.semantic_analyzer.vectors_nlp.vocab[token.lemma_]
                    else:
                        working_lexeme = \
                            self.semantic_analyzer.vectors_nlp.vocab[token._.holmes.lemma]
                    if working_lexeme.has_vector and working_lexeme.vector_norm > 0:
                        matchable_non_entity_tokens_to_vectors[token.i] = \
                            working_lexeme.vector
                    else:
                        matchable_non_entity_tokens_to_vectors[token.i] = None
            if process_initial_question_words and token.tag_ in \
                    self.semantic_analyzer.interrogative_pronoun_tags:
                tokens_to_match.append(token)
                matchable_non_entity_tokens_to_vectors[token.i] = None
            if token.dep_ == 'ROOT': # syntactic root
                root_tokens.append(replace_grammatical_root_token_recursively(token))
        if len(tokens_to_match) == 0:
            raise SearchPhraseWithoutMatchableWordsError(search_phrase_text)
        if len(root_tokens) > 1:
            raise SearchPhraseContainsMultipleClausesError(search_phrase_text)
        root_token = root_tokens[0]
        if phraselet_template is None:
            reverse_only = False
        else:
            reverse_only = is_reverse_only_parent_lemma or phraselet_template.reverse_only

        words_matching_root_token = []
        root_word_to_match_info_dict = {}

        add_word_information(root_token._.holmes.lemma, 'direct', 0)
        if phraselet_template is None and root_token.lemma_.lower() == \
                root_token._.holmes.lemma.lower():
            add_word_information(root_token.text.lower(), 'direct', 0)
            hyphen_normalized_text = \
                self.semantic_matching_helper.normalize_hyphens(root_token.text)
            if root_token.text != hyphen_normalized_text:
                add_word_information(hyphen_normalized_text.lower(), 'direct', 0)
        if self.semantic_matching_helper.analyze_derivational_morphology and \
                root_token._.holmes.derived_lemma is not None:
            add_word_information(root_token._.holmes.derived_lemma, 'derivation', 0)
        if self.semantic_matching_helper.ontology is not None and not \
                self.semantic_matching_helper.is_entity_search_phrase_token(
                    root_token, phraselet_template is not None):
            add_word_information_from_ontology(root_token._.holmes.lemma)
            if self.semantic_matching_helper.analyze_derivational_morphology and \
                    root_token._.holmes.derived_lemma is not None:
                add_word_information_from_ontology(root_token._.holmes.derived_lemma)
            if phraselet_template is None and root_token.lemma == \
                    root_token._.holmes.lemma:
                add_word_information_from_ontology(root_token.text.lower())
                if root_token.text != hyphen_normalized_text:
                    add_word_information_from_ontology(hyphen_normalized_text.lower())
            if self.semantic_matching_helper.analyze_derivational_morphology:
                for reverse_derived_lemma in \
                        self.semantic_matching_helper.reverse_derived_lemmas_in_ontology(
                        root_token):
                    add_word_information_from_ontology(reverse_derived_lemma)

        return SearchPhrase(
            search_phrase_doc, [token.i for token in tokens_to_match], root_token.i,
            matchable_non_entity_tokens_to_vectors, label, phraselet_template is not None,
            topic_match_phraselet_created_without_matching_tags,
            reverse_only, treat_as_reverse_only_during_initial_relation_matching,
            words_matching_root_token, root_word_to_match_info_dict)

    def get_ontology_reverse_derivational_dict(self):
        """During structural matching, a lemma or derived lemma matches any words in the ontology
            that yield the same word as their derived lemmas. This method generates a dictionary
            from derived lemmas to ontology words that yield them to facilitate such matching.
        """
        if self.analyze_derivational_morphology and self.ontology is not None:
            ontology_reverse_derivational_dict = {}
            for ontology_word in self.ontology.words:
                derived_lemmas = []
                normalized_ontology_word = \
                    self.semantic_matching_helper.normalize_hyphens(ontology_word)
                for textual_word in normalized_ontology_word.split():
                    derived_lemma = self.semantic_analyzer.derived_holmes_lemma(
                        None, textual_word.lower())
                    if derived_lemma is None:
                        derived_lemma = textual_word
                    derived_lemmas.append(derived_lemma)
                derived_ontology_word = ' '.join(derived_lemmas)
                if derived_ontology_word != ontology_word:
                    if derived_ontology_word in ontology_reverse_derivational_dict:
                        ontology_reverse_derivational_dict[derived_ontology_word].append(
                            ontology_word)
                    else:
                        ontology_reverse_derivational_dict[derived_ontology_word] = [ontology_word]
            # sort entry lists to ensure deterministic behaviour
            for derived_ontology_word in ontology_reverse_derivational_dict:
                ontology_reverse_derivational_dict[derived_ontology_word] = \
                    sorted(ontology_reverse_derivational_dict[derived_ontology_word])
            return ontology_reverse_derivational_dict
        else:
            return None

class SemanticMatchingHelperFactory():
    """Returns the correct *SemanticMatchingHelperFactory* for the language in use.
        This class must be added to if additional implementations are added for new languages.
    """

    def semantic_matching_helper(self, *, language, ontology, analyze_derivational_morphology):
        language_specific_rules_module = importlib.import_module(
            '.'.join(('.lang', language, 'language_specific_rules')),
            'holmes_extractor')
        return language_specific_rules_module.\
            LanguageSpecificSemanticMatchingHelper(ontology, analyze_derivational_morphology)

class SemanticMatchingHelper(ABC):
    """Abstract *SemanticMatchingHelper* parent class containing language-specific properties and
        methods that are required for matching and can be successfully and efficiently serialized.
        Functionality is placed here that is common to all current implementations. It follows that
        some functionality will probably have to be moved out to specific implementations whenever
        an implementation for a new language is added.

        For explanations of the abstract variables and methods, see the
        *EnglishSemanticMatchingHelper* implementation where they can be illustrated with direct
        examples.
    """

    noun_pos = NotImplemented

    permissible_embedding_pos = NotImplemented

    noun_kernel_dep = NotImplemented

    minimum_embedding_match_word_length = NotImplemented

    topic_matching_phraselet_stop_lemmas = NotImplemented

    topic_matching_reverse_only_parent_lemmas = NotImplemented

    topic_matching_phraselet_stop_tags = NotImplemented

    supervised_document_classification_phraselet_stop_lemmas = NotImplemented

    match_implication_dict = NotImplemented

    phraselet_templates = NotImplemented

    preferred_phraselet_pos = NotImplemented

    entity_defined_multiword_pos = NotImplemented

    entity_defined_multiword_entity_types = NotImplemented

    sibling_marker_deps = NotImplemented

    def __init__(self, ontology, analyze_derivational_morphology):
        self.ontology = ontology
        self.analyze_derivational_morphology = analyze_derivational_morphology
        for key, match_implication in self.match_implication_dict.items():
            assert key == match_implication.search_phrase_dependency
            assert key not in match_implication.document_dependencies
            assert len([dep for dep in match_implication.document_dependencies
                if match_implication.document_dependencies.count(dep) > 1]) == 0
            assert key not in match_implication.reverse_document_dependencies
            assert len([dep for dep in match_implication.reverse_document_dependencies
                if match_implication.reverse_document_dependencies.count(dep) > 1]) == 0

    @abstractmethod
    def normalize_hyphens(self, word):
        pass

    @abstractmethod
    def question_word_matches(self, search_phrase_token:Token, document_token:Token):
        pass

    @abstractmethod
    def get_subtree_list_without_conjunction(self, token:Token):
        pass

    def add_to_corpus_index(self, corpus_index_dict, parsed_document, document_label):

        def add_dict_entry(dictionary, word, token_index, subword_index, match_type):
            index = Index(token_index, subword_index)
            corpus_word_position = CorpusWordPosition(document_label, index)
            if match_type == 'entity':
                key_word = word
            else:
                key_word = word.lower()
            if key_word in dictionary.keys():
                if index not in dictionary[key_word]:
                    dictionary[key_word].append((
                        corpus_word_position, word, match_type == 'derivation'))
            else:
                dictionary[key_word] = [(corpus_word_position, word, match_type == 'derivation')]

        def get_ontology_defined_multiword(token):
            for multiword_span in \
                    self.multiword_spans_with_head_token(token):
                if self.ontology.contains_multiword(multiword_span.text):
                    return multiword_span.text, 'direct'
                hyphen_normalized_text = self.normalize_hyphens(multiword_span.text)
                if self.ontology.contains_multiword(hyphen_normalized_text):
                    return hyphen_normalized_text, 'direct'
                elif self.ontology.contains_multiword(multiword_span.lemma):
                    return multiword_span.lemma, 'direct'
                elif self.ontology.contains_multiword(multiword_span.derived_lemma):
                    return multiword_span.derived_lemma, 'derivation'
                if self.analyze_derivational_morphology and self.ontology is not None:
                    for reverse_lemma in \
                            self.reverse_derived_lemmas_in_ontology(multiword_span):
                        return reverse_lemma, 'derivation'
            return None, None

        for token in parsed_document:

            # parent check is necessary so we only find multiword entities once per
            # search phrase. sibling_marker_deps applies to siblings which would
            # otherwise be excluded because the main sibling would normally also match the
            # entity root word.
            if len(token.ent_type_) > 0 and (
                    token.dep_ == 'ROOT' or token.dep_ in self.sibling_marker_deps
                    or token.ent_type_ != token.head.ent_type_):
                entity_label = ''.join(('ENTITY', token.ent_type_))
                add_dict_entry(corpus_index_dict, entity_label, token.i, None, 'entity')
            if self.ontology is not None:
                ontology_defined_multiword, match_type = get_ontology_defined_multiword(token)
                if ontology_defined_multiword is not None:
                    add_dict_entry(
                        corpus_index_dict, ontology_defined_multiword, token.i, None,
                        match_type)
                    continue
            entity_defined_multiword, _ = \
                self.get_entity_defined_multiword(token)
            if entity_defined_multiword is not None:
                add_dict_entry(
                    corpus_index_dict, entity_defined_multiword, token.i, None, 'direct')
            for representation, match_type in self.loop_textual_representations(token):
                add_dict_entry(
                    corpus_index_dict, representation, token.i, None, match_type)
            for subword in token._.holmes.subwords:
                for representation, match_type in self.loop_textual_representations(subword):
                    add_dict_entry(
                        corpus_index_dict, representation, token.i, subword.index,
                        match_type)

    def get_corpus_index_removing_document(self, corpus_index_dict, document_label):
        new_corpus_index_dict = {}
        for entry in corpus_index_dict:
            new_value = ([(c, w, m) for (c, w, m) in corpus_index_dict[entry] if c.document_label !=
                document_label])
            if len(new_value) > 0:
                new_corpus_index_dict[entry] = new_value
        return new_corpus_index_dict

    def dependency_labels_match(self, *, search_phrase_dependency_label, document_dependency_label,
            inverse_polarity:bool):
        """Determines whether a dependency label in a search phrase matches a dependency label in
            a document being searched.
            inverse_polarity: *True* if the matching dependencies have to point in opposite
            directions.
        """
        if not inverse_polarity:
            if search_phrase_dependency_label == document_dependency_label:
                return True
            if search_phrase_dependency_label not in self.match_implication_dict.keys():
                return False
            return document_dependency_label in \
                self.match_implication_dict[search_phrase_dependency_label].document_dependencies
        else:
            return search_phrase_dependency_label in self.match_implication_dict.keys() and \
                document_dependency_label in self.match_implication_dict[
                search_phrase_dependency_label].reverse_document_dependencies

    def multiword_spans_with_head_token(self, token):
        """Generator over *MultiwordSpan* objects with *token* at their head. Dependent phrases
            are only returned for nouns because e.g. for verbs the whole sentence would be returned.
        """

        if not token.pos_ in self.noun_pos:
            return
        pointer = token.left_edge.i
        while pointer <= token.right_edge.i:
            working_text = ''
            working_lemma = ''
            working_derived_lemma = ''
            working_tokens = []
            inner_pointer = pointer
            while inner_pointer <= token.right_edge.i and \
                    (token.doc[inner_pointer]._.holmes.is_matchable or
                    token.doc[inner_pointer].text == '-'):
                if token.doc[inner_pointer].text != '-':
                    working_text = ' '.join((working_text, token.doc[inner_pointer].text))
                    working_lemma = ' '.join((
                        working_lemma, token.doc[inner_pointer]._.holmes.lemma))
                    if self.analyze_derivational_morphology and \
                            token.doc[inner_pointer]._.holmes.derived_lemma is not None:
                        this_token_derived_lemma = token.doc[inner_pointer]._.holmes.derived_lemma
                    else:
                        # if derivational morphology analysis is switched off, the derived lemma
                        # will be identical to the lemma and will not be yielded by
                        # _loop_textual_representations().
                        this_token_derived_lemma = token.doc[inner_pointer]._.holmes.lemma
                    working_derived_lemma = ' '.join((
                        working_derived_lemma, this_token_derived_lemma))
                    working_tokens.append(token.doc[inner_pointer])
                inner_pointer += 1
            if pointer + 1 < inner_pointer and token in working_tokens:
                yield MultiwordSpan(
                    working_text.strip(), working_lemma.strip(), working_derived_lemma.strip(),
                    working_tokens)
            pointer += 1

    def reverse_derived_lemmas_in_ontology(self, obj):
        """ Returns all ontology entries that point to the derived lemma of a token or token-like
            object.
        """
        if isinstance(obj, Token):
            derived_lemma = obj._.holmes.lemma_or_derived_lemma()
        elif isinstance(obj, Subword):
            derived_lemma = obj.lemma_or_derived_lemma()
        elif isinstance(obj, MultiwordSpan):
            derived_lemma = obj.derived_lemma
        else:
            raise RuntimeError(': '.join(('Unsupported type', str(type(obj)))))
        derived_lemma = self.normalize_hyphens(derived_lemma)
        if derived_lemma in self.ontology_reverse_derivational_dict:
            return self.ontology_reverse_derivational_dict[derived_lemma]
        else:
            return []

    def is_entity_search_phrase_token(
            self, search_phrase_token, examine_lemma_rather_than_text):
        if examine_lemma_rather_than_text:
            word_to_check = search_phrase_token._.holmes.lemma
        else:
            word_to_check = search_phrase_token.text
        return word_to_check[:6] == 'ENTITY' and len(word_to_check) > 6

    def is_entitynoun_search_phrase_token(
            self, search_phrase_token):
        return search_phrase_token.text == 'ENTITYNOUN'

    def entity_search_phrase_token_matches(
            self, search_phrase_token, topic_match_phraselet, document_token):
        if topic_match_phraselet:
            word_to_check = search_phrase_token._.holmes.lemma
        else:
            word_to_check = search_phrase_token.text
        return (
            document_token.ent_type_ == word_to_check[6:] and
            len(document_token._.holmes.lemma.strip()) > 0) or (
                word_to_check == 'ENTITYNOUN' and
                document_token.pos_ in self.noun_pos)
                # len(document_token._.holmes.lemma.strip()) > 0: in German spaCy sometimes
                # classifies whitespace as entities.

    def loop_textual_representations(self, object):
        if isinstance(object, Token):
            yield object.text, 'direct'
            hyphen_normalized_text = self.normalize_hyphens(object.text)
            if hyphen_normalized_text != object.text:
                yield hyphen_normalized_text, 'direct'
            if object._.holmes.lemma != object.text:
                yield object._.holmes.lemma, 'direct'
            if self.analyze_derivational_morphology and object._.holmes.derived_lemma is not None:
                yield object._.holmes.derived_lemma, 'derivation'
        elif isinstance(object, Subword):
            yield object.text, 'direct'
            hyphen_normalized_text = self.normalize_hyphens(object.text)
            if hyphen_normalized_text != object.text:
                yield hyphen_normalized_text, 'direct'
            if object.text != object.lemma:
                yield object.lemma, 'direct'
            if self.analyze_derivational_morphology and object.derived_lemma is not None:
                yield object.derived_lemma, 'derivation'
        elif isinstance(object, MultiwordSpan):
            yield object.text, 'direct'
            hyphen_normalized_text = self.normalize_hyphens(object.text)
            if hyphen_normalized_text != object.text:
                yield hyphen_normalized_text, 'direct'
            if object.text != object.lemma:
                yield object.lemma, 'direct'
            if object.lemma != object.derived_lemma:
                yield object.derived_lemma, 'derivation'
        else:
            raise RuntimeError(': '.join(('Unsupported type', str(type(object)))))

    def belongs_to_entity_defined_multiword(self, token):
        return token.pos_ in self.entity_defined_multiword_pos and token.ent_type_ in \
                self.entity_defined_multiword_entity_types

    def get_entity_defined_multiword(self, token):
        """ If this token is at the head of a multiword recognized by spaCy named entity processing,
            returns the multiword string in lower case and the indexes of the tokens that make up
            the multiword, otherwise *None, None*.
        """
        if not self.belongs_to_entity_defined_multiword(token) or (
                token.dep_ != 'ROOT' and self.belongs_to_entity_defined_multiword(token.head)) or \
                token.ent_type_ == '' or token.left_edge.i == token.right_edge.i:
            return None, None
        working_ent = token.ent_type_
        working_text = ''
        working_indexes = []
        for counter in range(token.left_edge.i, token.right_edge.i +1):
            multiword_token = token.doc[counter]
            if not self.belongs_to_entity_defined_multiword(multiword_token) or \
                    multiword_token.ent_type_ != working_ent:
                if working_text != '':
                    return None, None
                else:
                    continue
            working_text = ' '.join((working_text, multiword_token.text))
            working_indexes.append(multiword_token.i)
        if len(working_text.split()) > 1:
            return working_text.strip().lower(), working_indexes
        else:
            return None, None

    def get_dependent_phrase(self, token, subword):
        """Return the dependent phrase of a token, with an optional subword reference. Used in
            building match dictionaries"""
        if subword is not None:
            return subword.text
        if not token.pos_ in self.noun_pos:
            return token.text
        return_string = ''
        pointer = token.left_edge.i - 1
        while True:
            pointer += 1
            if token.doc[pointer].pos_ not in self.noun_pos and \
                    token.doc[pointer].dep_ not in self.noun_kernel_dep and pointer > token.i:
                return return_string.strip()
            if return_string == '':
                return_string = token.doc[pointer].text
            else:
                return_string = ' '.join((return_string, token.doc[pointer].text))
            if token.right_edge.i <= pointer:
                return return_string

    def has_question_word_in_phrase(self, token:Token):
        if token.i > 20: # performance quick win
            return False
        return token.pos_ in self.noun_pos and len([1 for c in token._.holmes.children
            if c.child_token(token.doc)._.holmes.is_initial_question_word]) > 0
