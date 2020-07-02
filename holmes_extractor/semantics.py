from abc import ABC, abstractmethod
import spacy
import neuralcoref
import jsonpickle
import pkg_resources
from spacy.tokens import Token, Doc
from .errors import WrongModelDeserializationError, WrongVersionDeserializationError, \
        DocumentTooBigError

SERIALIZED_DOCUMENT_VERSION = 3

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

    def child_token(self, doc):
        """Convenience method to return the child token of this dependency.

        doc -- the document containing the token.
        """
        return doc[self.child_index]

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
            self, containing_token_index, index, text, lemma, derived_lemma, char_start_index,
            dependent_index, dependency_label, governor_index, governing_dependency_label):
        self.containing_token_index = containing_token_index
        self.index = index
        self.text = text
        self.lemma = lemma
        self.derived_lemma = derived_lemma
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

class HolmesDictionary:
    """The holder object for token-level semantic information managed by Holmes

    Holmes dictionaries are accessed using the syntax *token._.holmes*.

    index -- the index of the token
    lemma -- the value returned from *._.holmes.lemma* for the token.
    derived_lemma -- the value returned from *._.holmes.derived_lemma for the token; where relevant,
        another lemma with which *lemma* is derivationally related and which can also be useful for
        matching in some usecases; otherwise *None*.
    """

    def __init__(self, index, lemma, derived_lemma):
        self.index = index
        self.lemma = lemma
        self._derived_lemma = derived_lemma
        self.children = [] # list of *SemanticDependency* objects where this token is the parent.
        self.righthand_siblings = [] # list of tokens to the right of this token that stand in a
        # conjunction relationship to this token and that share its semantic parents.
        self.token_or_lefthand_sibling_index = None # the index of this token's lefthand sibling,
        # or this token's own index if this token has no lefthand sibling.
        self.is_involved_in_or_conjunction = False
        self.is_negated = None
        self.is_matchable = None
        self.parent_dependencies = [] # list of [index, label] specifications of dependencies
        # where this token is the child. Takes any coreference resolution into account. Used in
        # topic matching.
        self.token_and_coreference_chain_indexes = None # where no coreference, only the token
        # index; where coreference, the token index followed by the indexes of coreferring tokens
        self.mentions = []
        self.mention_root_index = None  # the lefthandmost token within of the mention that contains
                                        # this token within the first cluster to which this token
                                        # belongs, which will most often be this token itself
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
        if self.derived_lemma is not None:
            return self.derived_lemma
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


class SerializedHolmesDocument:
    """Consists of the spaCy represention returned by *get_bytes()* plus a jsonpickle representation
        of each token's *SemanticDictionary*.
    """

    def __init__(self, serialized_spacy_document, dictionaries, model):
        self._serialized_spacy_document = serialized_spacy_document
        self._dictionaries = dictionaries
        self._model = model
        self._version = SERIALIZED_DOCUMENT_VERSION

    def holmes_document(self, semantic_analyzer):
        doc = Doc(semantic_analyzer.nlp.vocab).from_bytes(
            self._serialized_spacy_document)
        for token in doc:
            token._.holmes = self._dictionaries[token.i]
        return doc

class PhraseletTemplate:
    """A template for a phraselet used in topic matching.

    Properties:

    label -- a label for the relation which will be used to form part of the labels of phraselets
        derived from this template.
    template_sentence -- a sentence with the target grammatical structure for phraselets derived
        from this template.
    parent_index -- the index within 'template_sentence' of the parent participant in the dependency
        (for relation phraselets) or of the word (for single-word phraselets).
    child_index -- the index within 'template_sentence' of the child participant in the dependency
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

class SemanticAnalyzerFactory():
    """Returns the correct *SemanticAnalyzer* for the model language. This class must be added to
        if additional *SemanticAnalyzer* implementations are added for new languages.
    """

    def semantic_analyzer(self, *, model, perform_coreference_resolution, debug=False):
        language = model[0:2]
        if language == 'en':
            return EnglishSemanticAnalyzer(
                model=model, perform_coreference_resolution=perform_coreference_resolution,
                debug=debug)
        elif language == 'de':
            return GermanSemanticAnalyzer(
                model=model, perform_coreference_resolution=perform_coreference_resolution,
                debug=debug)
        else:
            raise ValueError(
                ' '.join(['No semantic analyzer for model', language]))

class SemanticAnalyzer(ABC):
    """Abstract *SemanticAnalyzer* parent class. Functionality is placed here that is common to all
        current implementations. It follows that some functionality will probably have to be moved
        out to specific implementations whenever an implementation for a new language is added.

    For explanations of the abstract variables and methods, see the *EnglishSemanticAnalyzer*
        implementation where they can be illustrated with direct examples.
    """

    def __init__(self, *, model, perform_coreference_resolution, debug):
        """Args:

        model -- the name of the spaCy model
        perform_coreference_resolution -- *True* if neuralcoref should be added to the pipe,
                *None* if neuralcoref should be added to the pipe if coreference resolution is
                available for the model
        debug -- *True* if the object should print a representation of each parsed document
        """
        self.nlp = spacy.load(model)
        if perform_coreference_resolution is None and self.model_supports_coreference_resolution():
            perform_coreference_resolution = True
        if perform_coreference_resolution:
            neuralcoref.add_to_pipe(self.nlp)
        self.model = model
        self.perform_coreference_resolution = perform_coreference_resolution
        self.debug = debug
        self._derivational_dictionary = self._load_derivational_dictionary()

    Token.set_extension('holmes', default='')

    def _load_derivational_dictionary(self):
        in_package_filename = ''.join(('data/derivation_', self.model[0:2], '.csv'))
        absolute_filename = pkg_resources.resource_filename(__name__, in_package_filename)
        dictionary = {}
        with open(absolute_filename, "r", encoding="utf-8") as file:
            for line in file.readlines():
                words = [word.strip() for word in line.split(',')]
                for index in range(len(words)):
                    dictionary[words[index]] = words[0]
        return dictionary

    def reload_model(self):
        spacy.load(self.model)

    def parse(self, text):
        """Performs a full spaCy and Holmes parse on a string.
        """
        spacy_doc = self.spacy_parse(text)
        holmes_doc = self.holmes_parse(spacy_doc)
        return holmes_doc

    _maximum_document_size = 1000000

    def spacy_parse(self, text):
        """Performs a standard spaCy parse on a string.
        """
        if len(text) > self._maximum_document_size:
            raise DocumentTooBigError(' '.join((
                'size:', str(len(text)), 'max:', str(self._maximum_document_size))))
        return self.nlp(text)

    def holmes_parse(self, spacy_doc):
        """Adds the Holmes-specific information to each token within a spaCy document.
        """
        for token in spacy_doc:
            lemma = self._holmes_lemma(token)
            derived_lemma = self.derived_holmes_lemma(token, lemma)
            token._.set('holmes', HolmesDictionary(token.i, lemma, derived_lemma))
        for token in spacy_doc:
            self._set_negation(token)
        for token in spacy_doc:
            self._initialize_semantic_dependencies(token)
        for token in spacy_doc:
            self._mark_if_righthand_sibling(token)
            token._.holmes.token_or_lefthand_sibling_index = self._lefthand_sibling_recursively(
                token)
        for token in spacy_doc:
            self._copy_any_sibling_info(token)
        subword_cache = {}
        for token in spacy_doc:
            self._add_subwords(token, subword_cache)
        for token in spacy_doc:
            self._set_coreference_information(token)
        for token in spacy_doc:
            self._set_matchability(token)
        for token in spacy_doc:
            self._correct_auxiliaries_and_passives(token)
        for token in spacy_doc:
            self._copy_any_sibling_info(token)
        for token in spacy_doc:
            self._normalize_predicative_adjectives(token)
        for token in spacy_doc:
            self._handle_relative_constructions(token)
        for token in spacy_doc:
            self._create_additional_preposition_phrase_semantic_dependencies(token)
        for token in spacy_doc:
            self._perform_language_specific_tasks(token)
        for token in spacy_doc:
            self._create_parent_dependencies(token)
        self.debug_structures(spacy_doc)
        return spacy_doc

    def model_supports_embeddings(self):
        return self.nlp.meta['vectors']['vectors'] > 0

    def model_supports_coreference_resolution(self):
        return self._model_supports_coreference_resolution

    def dependency_labels_match(self, *, search_phrase_dependency_label, document_dependency_label):
        """Determines whether a dependency label in a search phrase matches a dependency label in
            a document being searched.
        """
        if search_phrase_dependency_label == document_dependency_label:
            return True
        if search_phrase_dependency_label not in self._matching_dep_dict.keys():
            return False
        return document_dependency_label in self._matching_dep_dict[search_phrase_dependency_label]

    def _lefthand_sibling_recursively(self, token):
        """If *token* is a righthand sibling, return the index of the token that has a sibling
            reference to it, otherwise return the index of *token* itself.
        """
        if token.dep_ not in self._conjunction_deps:
            return token.i
        else:
            return self._lefthand_sibling_recursively(token.head)

    def debug_structures(self, doc):
        if self.debug:
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
                if self.is_involved_in_coreference(token):
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

    def to_serialized_string(self, spacy_doc):
        dictionaries = []
        for token in spacy_doc:
            dictionaries.append(token._.holmes)
            token._.holmes = None
        serialized_document = SerializedHolmesDocument(
            spacy_doc.to_bytes(), dictionaries, self.model)
        for token in spacy_doc:
            token._.holmes = dictionaries[token.i]
        return jsonpickle.encode(serialized_document)

    def from_serialized_string(self, serialized_spacy_doc):
        serialized_document = jsonpickle.decode(serialized_spacy_doc)
        if serialized_document._model != self.model:
            raise WrongModelDeserializationError(serialized_document._model)
        if serialized_document._version != SERIALIZED_DOCUMENT_VERSION:
            raise WrongVersionDeserializationError(serialized_document._version)
        return serialized_document.holmes_document(self)

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
            if token.doc[pointer].pos_ not in self.noun_pos and token.doc[pointer].dep_ not in \
                    self.noun_kernel_dep and pointer > token.i:
                return return_string.strip()
            if return_string == '':
                return_string = token.doc[pointer].text
            else:
                return_string = ' '.join((return_string, token.doc[pointer].text))
            if token.right_edge.i <= pointer:
                return return_string

    def is_involved_in_coreference(self, token):
        return len(token._.holmes.mentions) > 0

    def _set_coreference_information(self, token):
        token._.holmes.token_and_coreference_chain_indexes = [token.i]
        if not self.perform_coreference_resolution or not token.doc._.has_coref or not \
                token._.in_coref:
            return
        for cluster in token._.coref_clusters:
            counter = 0
            this_token_mention_index = -1
            for span in cluster:
                for candidate in span.root._.holmes.loop_token_and_righthand_siblings(
                        token.doc):
                    if candidate.i == token.i and candidate.i >= span.start and candidate.i < \
                            span.end:
                        this_token_mention_index = counter
                        if token._.holmes.mention_root_index is None:
                            token._.holmes.mention_root_index = span.root.i
                        break
                if this_token_mention_index > -1:
                    break
                counter += 1
            counter = 0
            if this_token_mention_index > -1:
                for span in cluster:
                    if abs(counter - this_token_mention_index) <= \
                            self._maximum_mentions_in_coreference_chain and \
                            abs(span.root.i - token.i) < \
                            self._maximum_word_distance_in_coreference_chain:
                        siblings_of_span_root = [span.root.i]
                        siblings_of_span_root.extend(span.root._.holmes.righthand_siblings)
                        indexes_within_mention = []
                        for candidate in siblings_of_span_root:
                            if span.start <= candidate < span.end and not \
                                    (candidate != token.i and token.i in siblings_of_span_root):
                                indexes_within_mention.append(candidate)
                        token._.holmes.mentions.append(Mention(span.root.i, indexes_within_mention))
                    counter += 1
        working_set = set()
        for mention in token._.holmes.mentions:
            working_set.update(mention.indexes)
        if len(working_set) > 1:
            working_set.remove(token.i)
            token._.holmes.token_and_coreference_chain_indexes.extend(sorted(working_set))
            # this token must always be the first in the list to ensure it is recorded as the
            # structurally matched token during structural matching

    def belongs_to_entity_defined_multiword(self, token):
        return token.pos_ in self._entity_defined_multiword_pos and token.ent_type_ in \
                self._entity_defined_multiword_entity_types

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

    def embedding_matching_permitted(self, obj):
        if isinstance(obj, Token):
            if len(obj._.holmes.lemma.split()) > 1:
                working_lemma = obj.lemma_
            else:
                working_lemma = obj._.holmes.lemma
            return obj.pos_ in self._permissible_embedding_pos and \
                len(working_lemma) >= self._minimum_embedding_match_word_length
        elif isinstance(obj, Subword):
            return len(obj.lemma) >= self._minimum_embedding_match_word_length

    language_name = NotImplemented

    noun_pos = NotImplemented

    _matchable_pos = NotImplemented

    _adjectival_predicate_head_pos = NotImplemented

    _adjectival_predicate_subject_pos = NotImplemented

    noun_kernel_dep = NotImplemented

    sibling_marker_deps = NotImplemented

    _adjectival_predicate_subject_dep = NotImplemented

    _adjectival_predicate_predicate_dep = NotImplemented

    _modifier_dep = NotImplemented

    _spacy_noun_to_preposition_dep = NotImplemented

    _spacy_verb_to_preposition_dep = NotImplemented

    _holmes_noun_to_preposition_dep = NotImplemented

    _holmes_verb_to_preposition_dep = NotImplemented

    _conjunction_deps = NotImplemented

    _interrogative_pronoun_tags = NotImplemented

    _semantic_dependency_excluded_tags = NotImplemented

    _generic_pronoun_lemmas = NotImplemented

    _or_lemma = NotImplemented

    _matching_dep_dict = NotImplemented

    _mark_child_dependencies_copied_to_siblings_as_uncertain = NotImplemented

    _maximum_mentions_in_coreference_chain = NotImplemented

    _maximum_word_distance_in_coreference_chain = NotImplemented

    _model_supports_coreference_resolution = NotImplemented

    _entity_defined_multiword_pos = NotImplemented

    _entity_defined_multiword_entity_types = NotImplemented

    phraselet_templates = NotImplemented

    topic_matching_phraselet_stop_lemmas = NotImplemented

    supervised_document_classification_phraselet_stop_lemmas = NotImplemented

    topic_matching_reverse_only_parent_lemmas = NotImplemented

    preferred_phraselet_pos = NotImplemented

    _permissible_embedding_pos = NotImplemented

    _minimum_embedding_match_word_length = NotImplemented

    @abstractmethod
    def _add_subwords(self, token, subword_cache):
        pass

    @abstractmethod
    def _set_negation(self, token):
        pass

    @abstractmethod
    def _correct_auxiliaries_and_passives(self, token):
        pass

    @abstractmethod
    def _perform_language_specific_tasks(self, token):
        pass

    @abstractmethod
    def _handle_relative_constructions(self, token):
        pass

    @abstractmethod
    def _holmes_lemma(self, token):
        pass

    def derived_holmes_lemma(self, token, lemma):
        if lemma in self._derivational_dictionary:
            derived_lemma = self._derivational_dictionary[lemma]
            if lemma == derived_lemma: # basis entry, so do not call language specific method
                return None
            else:
                return derived_lemma
        else:
            return self._language_specific_derived_holmes_lemma(token, lemma)

    @abstractmethod
    def normalize_hyphens(self, word):
        pass

    @abstractmethod
    def _language_specific_derived_holmes_lemma(self, token, lemma):
        pass

    def _initialize_semantic_dependencies(self, token):
        for child in (
                child for child in token.children if child.dep_ != 'punct' and
                child.tag_ not in self._semantic_dependency_excluded_tags):
            token._.holmes.children.append(SemanticDependency(token.i, child.i, child.dep_))

    def _mark_if_righthand_sibling(self, token):
        if token.dep_ in self.sibling_marker_deps:  # i.e. is righthand sibling
            working_token = token
            working_or_conjunction_flag = False
            # work up through the tree until the lefthandmost sibling element with the
            # semantic relationships to the rest of the sentence is reached
            while working_token.dep_ in self._conjunction_deps:
                working_token = working_token.head
                for working_child in working_token.children:
                    if working_child.lemma_ == self._or_lemma:
                        working_or_conjunction_flag = True
            # add this element to the lefthandmost sibling as a righthand sibling
            working_token._.holmes.righthand_siblings.append(token.i)
            if working_or_conjunction_flag:
                working_token._.holmes.is_involved_in_or_conjunction = True

    def _copy_any_sibling_info(self, token):
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
                        sibling_dependency.label == dependency.label]) == 0 and \
                        dependency.label not in self._conjunction_deps and not \
                        righthand_sibling_token._.holmes.has_dependency_with_child_index(
                            dependency.child_index) \
                        and righthand_sibling != dependency.child_index:
                    righthand_sibling_token._.holmes.children.append(SemanticDependency(
                        righthand_sibling, dependency.child_index, dependency.label,
                        self._mark_child_dependencies_copied_to_siblings_as_uncertain
                        or dependency.is_uncertain))

    def _normalize_predicative_adjectives(self, token):
        """Change phrases like *the town is old* and *the man is poor* so their
            semantic structure is equivalent to *the old town* and *the poor man*.
        """
        if token.pos_ == self._adjectival_predicate_head_pos:
            altered = False
            for predicative_adjective_index in (
                    dependency.child_index for dependency in \
                    token._.holmes.children if dependency.label ==
                    self._adjectival_predicate_predicate_dep and
                    token.doc[dependency.child_index].pos_ == 'ADJ' and
                    dependency.child_index >= 0):
                for subject_index in (
                        dependency.child_index for dependency in
                        token._.holmes.children if dependency.label ==
                        self._adjectival_predicate_subject_dep and (
                            dependency.child_token(token.doc).pos_ in
                            self._adjectival_predicate_subject_pos or
                            self.is_involved_in_coreference(dependency.child_token(token.doc))) and
                        dependency.child_index >= 0 and \
                        dependency.child_index != predicative_adjective_index):
                    token.doc[subject_index]._.holmes.children.append(
                        SemanticDependency(
                            subject_index, predicative_adjective_index, self._modifier_dep))
                    altered = True
            if altered:
                token._.holmes.children = [SemanticDependency(
                    token.i, 0 - (subject_index + 1), None)]

    def _create_additional_preposition_phrase_semantic_dependencies(self, token):
        """In structures like 'Somebody needs insurance for a period' it seems to be
            mainly language-dependent whether the preposition phrase is analysed as being
            dependent on the preceding noun or the preceding verb. We add an additional, new
            dependency to whichever of the noun or the verb does not already have one. In English,
            the new label is defined in *_matching_dep_dict* in such a way that original
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
            if token.i > 0 and token.doc[token.i-1].sent == token.sent and (
                    token.doc[token.i-1].pos_ in ('NOUN', 'PROPN') or
                    self.is_involved_in_coreference(token.doc[token.i-1])):
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
                            token.i, self._spacy_noun_to_preposition_dep) and not \
                            governing_verb._.holmes.has_dependency_with_child_index_and_label(
                                token.i, self._spacy_verb_to_preposition_dep):
                        add_dependencies_pointing_to_preposition_and_siblings(
                            governing_verb, self._holmes_verb_to_preposition_dep)
                # if the verb(s) governs the preposition, add new possible dependencies
                # from the noun
                if governing_verbs[0]._.holmes.has_dependency_with_child_index_and_label(
                        token.i, self._spacy_verb_to_preposition_dep) and not \
                        preceding_noun._.holmes.has_dependency_with_child_index_and_label(
                            token.i, self._spacy_noun_to_preposition_dep):
                    # check the preposition is not pointing back to a relative clause
                    for preposition_dep_index in (
                            dep.child_index for dep in token._.holmes.children
                            if dep.child_index >= 0):
                        if token.doc[preposition_dep_index]._.holmes.\
                                has_dependency_with_label('relcl'):
                            return
                    add_dependencies_pointing_to_preposition_and_siblings(
                        preceding_noun, self._holmes_noun_to_preposition_dep)

    def _set_matchability(self, token):
        """Marks whether this token, if it appears in a search phrase, should require a counterpart
        in a document being matched.
        """
        token._.holmes.is_matchable = (
            token.pos_ in self._matchable_pos or self.is_involved_in_coreference(token)) \
            and token.tag_ not in self._interrogative_pronoun_tags and \
            token._.holmes.lemma not in self._generic_pronoun_lemmas

    def _move_information_between_tokens(self, from_token, to_token):
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
        linking_dependency_label = linking_dependencies[0].label
        # only loop dependencies whose label or index are not already present at the destination
        for dependency in (
                dependency for dependency in from_token._.holmes.children
                if dependency.label != linking_dependency_label and not
                to_token._.holmes.has_dependency_with_child_index(dependency.child_index) and
                to_token.i != dependency.child_index):
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

    def _create_parent_dependencies(self, token):
        if self.perform_coreference_resolution:
            for linked_parent_index in token._.holmes.token_and_coreference_chain_indexes:
                linked_parent = token.doc[linked_parent_index]
                for child_dependency in (
                        child_dependency for child_dependency in linked_parent._.holmes.children
                        if child_dependency.child_index >= 0):
                    child_token = child_dependency.child_token(token.doc)
                    for linked_child_index in \
                            child_token._.holmes.token_and_coreference_chain_indexes:
                        linked_child = token.doc[linked_child_index]
                        linked_child._.holmes.parent_dependencies.append([
                            token.i, child_dependency.label])
        else:
            for child_dependency in (
                    child_dependency for child_dependency in token._.holmes.children
                    if child_dependency.child_index >= 0):
                child_token = child_dependency.child_token(token.doc)
                child_token._.holmes.parent_dependencies.append([
                    token.i, child_dependency.label])

class EnglishSemanticAnalyzer(SemanticAnalyzer):

    language_name = 'English'

    # The part of speech tags that require a match in the search sentence when they occur within a
    # search_phrase
    _matchable_pos = ('ADJ', 'ADP', 'ADV', 'NOUN', 'NUM', 'PROPN', 'VERB')

    # The part of speech tags that can refer to nouns
    noun_pos = ('NOUN', 'PROPN')

    # The part of speech tags that can refer to the head of an adjectival predicate phrase
    # ("is" in "The dog is tired")
    _adjectival_predicate_head_pos = 'VERB'

    # The part of speech tags that can refer to the subject of a adjectival predicate
    # ("dog" in "The dog is tired")
    _adjectival_predicate_subject_pos = ('NOUN', 'PROPN', 'PRON')

    # Dependency labels that mark noun kernel elements that are not the head noun
    noun_kernel_dep = ('nmod', 'compound', 'appos', 'nummod')

    # Dependency labels that can mark righthand siblings
    sibling_marker_deps = ('conj', 'appos')

    # Dependency label that marks the subject of an adjectival predicate
    _adjectival_predicate_subject_dep = 'nsubj'

    # Dependency label that marks the predicate of an adjectival predicate
    _adjectival_predicate_predicate_dep = 'acomp'

    # Dependency label that marks a modifying adjective
    _modifier_dep = 'amod'

    # Original dependency label from nouns to prepositions
    _spacy_noun_to_preposition_dep = 'prep'

    # Original dependency label from verbs to prepositions
    _spacy_verb_to_preposition_dep = 'prep'

    # Added possible dependency label from nouns to prepositions
    _holmes_noun_to_preposition_dep = 'prepposs'

    # Added possible dependency label from verbs to prepositions
    _holmes_verb_to_preposition_dep = 'prepposs'

    # Dependency labels that occur in a conjunction phrase (righthand siblings and conjunctions)
    _conjunction_deps = ('conj', 'appos', 'cc')

    # Syntactic tags that can mark interrogative pronouns
    _interrogative_pronoun_tags = ('WDT', 'WP', 'WRB')

    # Syntactic tags that exclude a token from being the child token within a semantic dependency
    _semantic_dependency_excluded_tags = ('DT')

    # Generic pronouns
    _generic_pronoun_lemmas = ('something', 'somebody', 'someone')

    # The word for 'or' in this language
    _or_lemma = 'or'

    # Map from dependency tags as occurring within search phrases to corresponding dependency tags
    # as occurring within documents being searched. This is the main source of the asymmetry
    # in matching from search phrases to documents versus from documents to search phrases.
    _matching_dep_dict = {
        'nsubj': ['csubj', 'poss', 'pobjb', 'pobjo', 'advmodsubj', 'arg'],
        'acomp': ['amod', 'advmod', 'npmod', 'advcl'],
        'amod': ['acomp', 'advmod', 'npmod', 'advcl'],
        'advmod': ['acomp', 'amod', 'npmod', 'advcl'],
        'arg': [
            'nsubj', 'csubj', 'poss', 'pobjb', 'advmodsubj', 'dobj', 'pobjo', 'relant',
            'nsubjpass', 'csubjpass', 'compound', 'advmodobj', 'dative', 'pobjp'],
        'compound': [
            'nmod', 'appos', 'nounmod', 'nsubj', 'csubj', 'poss', 'pobjb',
            'advmodsubj', 'dobj', 'pobjo', 'relant', 'pobjp',
            'nsubjpass', 'csubjpass', 'arg', 'advmodobj', 'dative'],
        'dative': ['pobjt', 'relant', 'nsubjpass'],
        'pobjt': ['dative', 'relant'],
        'nsubjpass': [
            'dobj', 'pobjo', 'poss', 'relant', 'csubjpass',
            'compound', 'advmodobj', 'arg', 'dative'],
        'dobj': [
            'pobjo', 'poss', 'relant', 'nsubjpass', 'csubjpass',
            'compound', 'advmodobj', 'arg', 'xcomp'],
        'nmod': ['appos', 'compound', 'nummod'],
        'poss': [
            'pobjo', 'nsubj', 'csubj', 'pobjb', 'advmodsubj', 'arg', 'relant',
            'nsubjpass', 'csubjpass', 'compound', 'advmodobj'],
        'pobjo': [
            'poss', 'dobj', 'relant', 'nsubjpass', 'csubjpass',
            'compound', 'advmodobj', 'arg', 'xcomp', 'nsubj', 'csubj', 'advmodsubj'],
        'pobjb': ['nsubj', 'csubj', 'poss', 'advmodsubj', 'arg'],
        'pobjp': ['compound'],
        'prep': ['prepposs'],
        'xcomp': [
            'pobjo', 'poss', 'relant', 'nsubjpass', 'csubjpass',
            'compound', 'advmodobj', 'arg', 'dobj']}

    # Where dependencies from a parent to a child are copied to the parent's righthand siblings,
    # it can make sense to mark the dependency as uncertain depending on the underlying spaCy
    # representations for the individual language
    _mark_child_dependencies_copied_to_siblings_as_uncertain = True

    # Coreference chains are only processed up to this number of mentions away from the currently
    # matched document location
    _maximum_mentions_in_coreference_chain = 3

    # Coreference chains are only processed up to this number of words away from the currently
    # matched document location
    _maximum_word_distance_in_coreference_chain = 300

    # Presently depends purely on the language
    _model_supports_coreference_resolution = True

    # The part-of-speech labels permitted for elements of an entity-defined multiword.
    _entity_defined_multiword_pos = ('NOUN', 'PROPN')

    # The entity labels permitted for elements of an entity-defined multiword.
    _entity_defined_multiword_entity_types = ('PERSON', 'ORG', 'GPE', 'WORK_OF_ART')

    # The templates used to generate topic matching phraselets.
    phraselet_templates = [
        PhraseletTemplate(
            "predicate-actor", "A thing does", 2, 1,
            ['nsubj', 'csubj', 'pobjb', 'advmodsubj'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'], reverse_only=False),
        PhraseletTemplate(
            "predicate-patient", "Somebody does a thing", 1, 3,
            ['dobj', 'relant', 'advmodobj', 'xcomp'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            reverse_only=False),
        PhraseletTemplate(
            "word-ofword", "A thing of a thing", 1, 4,
            ['pobjo', 'poss'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            reverse_only=False),
        PhraseletTemplate(
            "predicate-toughmovedargument", "A thing is easy to do", 5, 1,
            ['arg'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'], reverse_only=False),
        PhraseletTemplate(
            "predicate-passivesubject", "A thing is done", 3, 1,
            ['nsubjpass', 'csubjpass'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'], reverse_only=False),
        PhraseletTemplate(
            "be-attribute", "Something is a thing", 1, 3,
            ['attr'],
            ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'], reverse_only=True),
        PhraseletTemplate(
            "predicate-recipient", "Somebody gives a thing something", 1, 3,
            ['dative', 'pobjt'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'], reverse_only=False),
        PhraseletTemplate(
            "governor-adjective", "A described thing", 2, 1,
            ['acomp', 'amod', 'advmod', 'npmod', 'advcl'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ['JJ', 'JJR', 'JJS', 'VBN', 'RB', 'RBR', 'RBS'], reverse_only=False),
        PhraseletTemplate(
            "noun-noun", "A thing thing", 2, 1,
            ['nmod', 'appos', 'compound', 'nounmod'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'], reverse_only=False),
        PhraseletTemplate(
            "number-noun", "Seven things", 1, 0,
            ['nummod'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'],
            ['CD'], reverse_only=False),
        PhraseletTemplate(
            "prepgovernor-noun", "A thing in a thing", 1, 4,
            ['pobjp'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'], reverse_only=False),
        PhraseletTemplate(
            "prep-noun", "in a thing", 0, 2,
            ['pobj'],
            ['IN'], ['FW', 'NN', 'NNP', 'NNPS', 'NNS'], reverse_only=True),
        PhraseletTemplate(
            "word", "thing", 0, None,
            None,
            ['FW', 'NN', 'NNP', 'NNPS', 'NNS'],
            None, reverse_only=False)
        ]

    # Lemmas that should be suppressed within relation phraselets or as words of
    # single-word phraselets during topic matching.
    topic_matching_phraselet_stop_lemmas = ('then', 'therefore', 'so', '-pron-')

    # Lemmas that should be suppressed within relation phraselets or as words of
    # single-word phraselets during supervised document classification.
    supervised_document_classification_phraselet_stop_lemmas = ('be', 'have')

    # Parent lemma / part-of-speech combinations that should lead to phraselets being
    # reverse-matched only during topic matching.
    topic_matching_reverse_only_parent_lemmas = (
        ('be', 'VERB'), ('have', 'VERB'), ('do', 'VERB'),
        ('say', 'VERB'), ('go', 'VERB'), ('get', 'VERB'), ('make', 'VERB'))

    # Parts of speech that are preferred as lemmas within phraselets
    preferred_phraselet_pos = ('NOUN', 'PROPN')

    # Parts of speech for which embedding matching is attempted
    _permissible_embedding_pos = ('NOUN', 'PROPN', 'ADJ', 'ADV')

    # Minimum length of a word taking part in an embedding-based match.
    # Necessary because of the proliferation of short nonsense strings in the vocabularies.
    _minimum_embedding_match_word_length = 3

    def _add_subwords(self, token, subword_cache):
        """ Analyses the internal structure of the word to find atomic semantic elements. Is
            relevant for German and not currently implemented for English.
        """
        pass

    def _set_negation(self, token):
        """Marks the negation on the token. A token is negative if it or one of its ancestors
            has a negation word as a syntactic (not semantic!) child.
        """
        if token._.holmes.is_negated is not None:
            return
        for child in token.children:
            if child._.holmes.lemma in (
                    'nobody', 'nothing', 'nowhere', 'noone', 'neither', 'nor', 'no') \
                    or child.dep_ == 'neg':
                token._.holmes.is_negated = True
                return
            if child._.holmes.lemma in ('more', 'longer'):
                for grandchild in child.children:
                    if grandchild._.holmes.lemma == 'no':
                        token._.holmes.is_negated = True
                        return
        if token.dep_ == 'ROOT':
            token._.holmes.is_negated = False
            return
        self._set_negation(token.head)
        token._.holmes.is_negated = token.head._.holmes.is_negated

    def _correct_auxiliaries_and_passives(self, token):
        """Wherever auxiliaries and passives are found, derive the semantic information
            from the syntactic information supplied by spaCy.
        """
        # 'auxpass' means an auxiliary used in a passive context. We mark its subject with
        # a new dependency label 'nsubjpass'.
        if len([
                dependency for dependency in token._.holmes.children
                if dependency.label == 'auxpass']) > 0:
            for dependency in token._.holmes.children:
                if dependency.label == 'nsubj':
                    dependency.label = 'nsubjpass'

        # Structures like 'he used to' and 'he is going to'
        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.label == 'xcomp'):
            child = dependency.child_token(token.doc)
            # distinguish 'he used to ...' from 'he used it to ...'
            if token._.holmes.lemma == 'use' and token.tag_ == 'VBD' and len([
                    element for element in token._.holmes.children
                    if element.label == 'dobj']) == 0:
                self._move_information_between_tokens(token, child)
            elif token._.holmes.lemma == 'go':
                # 'was going to' is marked as uncertain, 'is going to' is not marked as uncertain
                uncertainty_flag = False
                for other_dependency in (
                        other_dependency for other_dependency in
                        token._.holmes.children if other_dependency.label == 'aux'):
                    other_dependency_token = other_dependency.child_token(token.doc)
                    if other_dependency_token._.holmes.lemma == 'be' and \
                            other_dependency_token.tag_ == 'VBD':  # 'was going to'
                        uncertainty_flag = True
                self._move_information_between_tokens(token, child)
                if uncertainty_flag:
                    for child_dependency in child._.holmes.children:
                        child_dependency.is_uncertain = True
            else:
                # constructions like:
                #
                #'she told him to close the contract'
                #'he decided to close the contract'
                for other_dependency in token._.holmes.children:
                    if other_dependency.label in ('dobj', 'nsubjpass') or (
                            other_dependency.label == 'nsubj' and \
                            len([
                                element for element in token._.holmes.children
                                if element.label == 'dobj'])
                            == 0):
                        if len([
                                element for element in child._.holmes.children
                                if element.label == 'auxpass']) > 0:
                            if not child._.holmes.has_dependency_with_child_index(
                                    other_dependency.child_index) and \
                                    dependency.child_index > other_dependency.child_index:
                                child._.holmes.children.append(SemanticDependency(
                                    dependency.child_index, other_dependency.child_index,
                                    'nsubjpass', True))
                        else:
                            if not child._.holmes.has_dependency_with_child_index(
                                    other_dependency.child_index) and \
                                    dependency.child_index > other_dependency.child_index:
                                child._.holmes.children.append(SemanticDependency(
                                    dependency.child_index, other_dependency.child_index,
                                    'nsubj', True))

    def _handle_relative_constructions(self, token):
        if token.dep_ == 'relcl':
            for dependency in token._.holmes.children:
                child = dependency.child_token(token.doc)
                # handle 'whose' clauses
                for child_dependency in (
                        child_dependency for child_dependency in
                        child._.holmes.children if child_dependency.child_index >= 0
                        and child_dependency.label == 'poss' and
                        child_dependency.child_token(token.doc).tag_ == 'WP$'):
                    whose_pronoun_token = child_dependency.child_token(
                        token.doc)
                    working_index = whose_pronoun_token.i
                    while working_index >= token.sent.start:
                        # find the antecedent (possessed entity)
                        for dependency in (
                                dependency for dependency in
                                whose_pronoun_token.doc[working_index]._.holmes.children
                                if dependency.label == 'relcl'):
                            working_token = child.doc[working_index]
                            working_token = working_token.doc[
                                working_token._.holmes.token_or_lefthand_sibling_index]
                            for lefthand_sibling_of_antecedent in \
                                    working_token._.holmes.loop_token_and_righthand_siblings(
                                        token.doc):
                                # find the possessing noun
                                for possessing_noun in (
                                        possessing_noun for possessing_noun in
                                        child._.holmes.loop_token_and_righthand_siblings(token.doc)
                                        if possessing_noun.i != lefthand_sibling_of_antecedent.i):
                                    # add the semantic dependency
                                    possessing_noun._.holmes.children.append(
                                        SemanticDependency(
                                            possessing_noun.i,
                                            lefthand_sibling_of_antecedent.i, 'poss',
                                            lefthand_sibling_of_antecedent.i != working_index))
                                    # remove the syntactic dependency
                                    possessing_noun._.holmes.remove_dependency_with_child_index(
                                        whose_pronoun_token.i)
                                whose_pronoun_token._.holmes.children = [SemanticDependency(
                                    whose_pronoun_token.i, 0 - (working_index + 1), None)]
                            return
                        working_index -= 1
                    return
                if child.tag_ in ('WP', 'WRB', 'WDT'):  # 'that' or 'which'
                    working_dependency_label = dependency.label
                    child._.holmes.children = [SemanticDependency(
                        child.i, 0 - (token.head.i + 1), None)]
                else:
                    # relative antecedent, new dependency tag, 'the man I saw yesterday'
                    working_dependency_label = 'relant'
                last_righthand_sibling_of_predicate = list(
                    token._.holmes.loop_token_and_righthand_siblings(token.doc))[-1]
                for preposition_dependency in (
                        dep for dep in last_righthand_sibling_of_predicate._.holmes.children
                        if dep.label == 'prep' and
                        dep.child_token(token.doc)._.holmes.is_matchable):
                    preposition = preposition_dependency.child_token(token.doc)
                    for grandchild_dependency in (
                            dep for dep in preposition._.holmes.children if
                            dep.child_token(token.doc).tag_ in ('WP', 'WRB', 'WDT')
                            and dep.child_token(token.doc).i >= 0):
                            # 'that' or 'which'
                        complementizer = grandchild_dependency.child_token(token.doc)
                        preposition._.holmes.remove_dependency_with_child_index(
                            grandchild_dependency.child_index)
                        # a new relation pointing directly to the antecedent noun
                        # will be added in the section below
                        complementizer._.holmes.children = [SemanticDependency(
                            grandchild_dependency.child_index, 0 - (token.head.i + 1), None)]
                displaced_preposition_dependencies = [
                    dep for dep in
                    last_righthand_sibling_of_predicate._.holmes.children
                    if dep.label == 'prep'
                    and len(dep.child_token(token.doc)._.holmes.children) == 0
                    and dep.child_token(token.doc)._.holmes.is_matchable]
                antecedent = token.doc[token.head._.holmes.token_or_lefthand_sibling_index]
                if len(displaced_preposition_dependencies) > 0:
                    displaced_preposition = \
                        displaced_preposition_dependencies[0].child_token(token.doc)
                    for lefthand_sibling_of_antecedent in (
                            lefthand_sibling_of_antecedent for lefthand_sibling_of_antecedent in
                            antecedent._.holmes.loop_token_and_righthand_siblings(token.doc)
                            if displaced_preposition.i != lefthand_sibling_of_antecedent.i):
                        displaced_preposition._.holmes.children.append(SemanticDependency(
                            displaced_preposition.i, lefthand_sibling_of_antecedent.i,
                            'pobj', lefthand_sibling_of_antecedent.i != token.head.i))
                        #Where the antecedent is not the final one before the relative
                        #clause, mark the dependency as uncertain
                    for sibling_of_pred in \
                            token._.holmes.loop_token_and_righthand_siblings(token.doc):
                        if not sibling_of_pred._.holmes.has_dependency_with_child_index(
                                displaced_preposition.i) and \
                                sibling_of_pred.i != displaced_preposition.i:
                            sibling_of_pred._.holmes.children.append(SemanticDependency(
                                sibling_of_pred.i, displaced_preposition.i, 'prep', True))
                        if working_dependency_label != 'relant':
                        # if 'that' or 'which', remove it
                            sibling_of_pred._.holmes.remove_dependency_with_child_index(
                                child.i)
                else:
                    for lefthand_sibling_of_antecedent in \
                            antecedent._.holmes.loop_token_and_righthand_siblings(token.doc):
                        for sibling_of_predicate in (
                                sibling_of_predicate for sibling_of_predicate
                                in token._.holmes.loop_token_and_righthand_siblings(token.doc)
                                if sibling_of_predicate.i != lefthand_sibling_of_antecedent.i):
                            sibling_of_predicate._.holmes.children.append(SemanticDependency(
                                sibling_of_predicate.i, lefthand_sibling_of_antecedent.i,
                                working_dependency_label,
                                lefthand_sibling_of_antecedent.i != token.head.i))
                            #Where the antecedent is not the final one before the relative
                            #clause, mark the dependency as uncertain
                            if working_dependency_label != 'relant':
                                sibling_of_predicate._.holmes.remove_dependency_with_child_index(
                                    child.i)
                break

    def _holmes_lemma(self, token):
        """Relabel the lemmas of phrasal verbs in sentences like 'he gets up' to incorporate
            the entire phrasal verb to facilitate matching.
        """
        if token.pos_ == 'VERB':
            for child in token.children:
                if child.tag_ == 'RP':
                    return ' '.join([token.lemma_.lower(), child.lemma_.lower()])
        return token.lemma_.lower()

    def normalize_hyphens(self, word):
        """ Normalizes hyphens for ontology matching. Depending on the language,
            this may involve replacing them with spaces (English) or deleting them entirely
            (German).
        """
        if word.strip().startswith('-') or word.endswith('-'):
            return word
        else:
            return word.replace('-', ' ')

    def _language_specific_derived_holmes_lemma(self, token, lemma):
        """Generates and returns a derived lemma where appropriate, otherwise returns *None*."""
        if (token is None or token.pos_ == 'NOUN') and len(lemma) >= 10:
            possible_lemma = None
            if lemma.endswith('isation') or lemma.endswith('ization'):
                possible_lemma = ''.join((lemma[:-5], 'e')) # 'isation', 'ization' -> 'ise', 'ize'
                if possible_lemma.endswith('ise'):
                    lemma_to_test_in_vocab = ''.join((possible_lemma[:-3], 'ize'))
                    # only American spellings in vocab
                else:
                    lemma_to_test_in_vocab = possible_lemma
            elif lemma.endswith('ication'):
                possible_lemma = ''.join((lemma[:-7], 'y')) # implication -> imply
                lemma_to_test_in_vocab = possible_lemma
            if (possible_lemma is None or self.nlp.vocab[lemma_to_test_in_vocab].is_oov) and \
                    lemma.endswith('ation'):
                possible_lemma = ''.join((lemma[:-3], 'e')) # manipulation -> manipulate
                lemma_to_test_in_vocab = possible_lemma
            if possible_lemma is not None and not self.nlp.vocab[lemma_to_test_in_vocab].is_oov:
                return possible_lemma
        # deadjectival nouns in -ness
        if (token is None or token.pos_ == 'NOUN') and len(lemma) >= 7 and lemma.endswith('ness'):
            working_possible_lemma = lemma[:-4]
            # 'bawdiness'
            if working_possible_lemma[-1] == 'i':
                working_possible_lemma = ''.join((working_possible_lemma[:-1], 'y'))
            if not self.nlp.vocab[working_possible_lemma].is_oov:
                return working_possible_lemma
            else:
                return None
        # adverb with 'ly' -> adjective without 'ly'
        if token is None or token.tag_ == 'RB':
            # domestically -> domestic
            if lemma.endswith('ically'):
                return lemma[:-4]
            # 'regrettably', 'horribly' -> 'regrettable', 'horrible'
            if lemma.endswith('ably') or lemma.endswith('ibly'):
                return ''.join((lemma[:-1], 'e'))
            if lemma.endswith('ly'):
                derived_lemma = lemma[:-2]
                # 'happily' -> 'happy'
                if derived_lemma[-1] == 'i':
                    derived_lemma = ''.join((derived_lemma[:-1], 'y'))
                return derived_lemma
        # singing -> sing
        if (token is None or token.tag_ == 'NN') and lemma.endswith('ing'):
            lemmatization_sentence = ' '.join(('it is', lemma))
            lemmatization_doc = self.spacy_parse(lemmatization_sentence)
            return lemmatization_doc[2].lemma_.lower()
        return None

    def _perform_language_specific_tasks(self, token):

        # Because phrasal verbs are conflated into a single lemma, remove the dependency
        # from the verb to the preposition
        if token.tag_ == 'RP':
            token.head._.holmes.remove_dependency_with_child_index(token.i)

        # mark modal verb dependencies as uncertain
        if token.pos_ == 'VERB':
            for dependency in (
                    dependency for dependency in token._.holmes.children
                    if dependency.label == 'aux'):
                child = dependency.child_token(token.doc)
                if child.pos_ == 'VERB' and child._.holmes.lemma not in \
                        ('be', 'have', 'do', 'go', 'use', 'will', 'shall'):
                    for other_dependency in (
                            other_dependency for other_dependency in
                            token._.holmes.children if other_dependency.label != 'aux'):
                        other_dependency.is_uncertain = True

        # set auxiliaries as not matchable
        if token.dep_ in ('aux', 'auxpass'):
            token._.holmes.is_matchable = False

        # Add new dependencies to phrases with 'by', 'of' and 'to' to enable the matching
        # of deverbal nominal phrases with verb phrases; add 'dative' dependency to
        # nouns within dative 'to' phrases; add new dependency spanning other prepositions
        # to facilitate topic matching and supervised document classification
        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.label in ('prep', 'agent', 'dative')):
            child = dependency.child_token(token.doc)
            if child._.holmes.lemma == 'by':
                working_dependency_label = 'pobjb'
            elif child._.holmes.lemma == 'of':
                working_dependency_label = 'pobjo'
            elif child._.holmes.lemma == 'to':
                if dependency.label == 'dative':
                    working_dependency_label = 'dative'
                else:
                    working_dependency_label = 'pobjt'
            else:
                working_dependency_label = 'pobjp'
            # for 'by', 'of' and 'to' the preposition is marked as not matchable
            if working_dependency_label != 'pobjp':
                child._.holmes.is_matchable = False
            for child_dependency in (
                    child_dependency for child_dependency in child._.holmes.children
                    if child_dependency.label == 'pobj' and token.i !=
                    child_dependency.child_index):
                token._.holmes.children.append(SemanticDependency(
                    token.i, child_dependency.child_index, working_dependency_label,
                    dependency.is_uncertain or child_dependency.is_uncertain))

        # where a 'prepposs' dependency has been added and the preposition is not 'by', 'of' or
        #'to', add a corresponding uncertain 'pobjp'
        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.label == 'prepposs'):
            child = dependency.child_token(token.doc)
            for child_dependency in (
                    child_dependency for child_dependency in
                    child._.holmes.children if child_dependency.label == 'pobj' and token.i !=
                    child_dependency.child_index and child._.holmes.is_matchable):
                token._.holmes.children.append(
                    SemanticDependency(token.i, child_dependency.child_index, 'pobjp', True))

        # handle present active participles
        if token.dep_ == 'acl' and token.tag_ == 'VBG':
            lefthand_sibling = token.doc[token.head._.holmes.token_or_lefthand_sibling_index]
            for antecedent in \
                    lefthand_sibling._.holmes.loop_token_and_righthand_siblings(token.doc):
                if token.i != antecedent.i:
                    token._.holmes.children.append(
                        SemanticDependency(token.i, antecedent.i, 'nsubj'))

        # handle past passive participles
        if token.dep_ == 'acl' and token.tag_ == 'VBN':
            lefthand_sibling = token.doc[token.head._.holmes.token_or_lefthand_sibling_index]
            for antecedent in \
                    lefthand_sibling._.holmes.loop_token_and_righthand_siblings(token.doc):
                if token.i != antecedent.i:
                    token._.holmes.children.append(
                        SemanticDependency(token.i, antecedent.i, 'dobj'))

        # handle phrases like 'cat-eating dog' and 'dog-eaten cat', adding new dependencies
        if token.dep_ == 'amod' and token.pos_ == 'VERB':
            for dependency in (
                    dependency for dependency in token._.holmes.children
                    if dependency.label == 'npadvmod'):
                if token.tag_ == 'VBG':
                    dependency.label = 'advmodobj'
                    noun_dependency = 'advmodsubj'
                elif token.tag_ == 'VBN':
                    dependency.label = 'advmodsubj'
                    noun_dependency = 'advmodobj'
                else:
                    break
                for noun in token.head._.holmes.loop_token_and_righthand_siblings(token.doc):
                    if token.i != noun.i:
                        token._.holmes.children.append(SemanticDependency(
                            token.i, noun.i, noun_dependency, noun.i != token.head.i))
                break  # we only handle one antecedent, spaCy never seems to produce more anyway

        # handle phrases like 'he is thinking about singing', 'he keeps on singing'
        # find governed verb
        if token.pos_ == 'VERB' and token.dep_ == 'pcomp':
            # choose correct noun dependency for passive or active structure
            if len([
                    dependency for dependency in token._.holmes.children
                    if dependency.label == 'auxpass']) > 0:
                new_dependency_label = 'nsubjpass'
            else:
                new_dependency_label = 'nsubj'
            # check that governed verb does not already have a dependency with the same label
            if len([
                    target_token_dependency for target_token_dependency in token._.holmes.children
                    if target_token_dependency.label == new_dependency_label]) == 0:
                # Go back in the sentence to find the first subject phrase
                counter = token.i
                while True:
                    counter -= 1
                    if counter < token.sent.start:
                        return
                    if token.doc[counter].dep_ in ('nsubj', 'nsubjpass'):
                        break
                # From the subject phrase loop up through the syntactic parents
                # to handle relative constructions
                working_token = token.doc[counter]
                while True:
                    if working_token.tag_.startswith('NN') or \
                            self.is_involved_in_coreference(working_token):
                        for source_token in \
                                working_token._.holmes.loop_token_and_righthand_siblings(token.doc):
                            for target_token in \
                                    token._.holmes.loop_token_and_righthand_siblings(token.doc):
                                if target_token.i != source_token.i:
                                    # such dependencies are always uncertain
                                    target_token._.holmes.children.append(SemanticDependency(
                                        target_token.i, source_token.i, new_dependency_label, True))
                        return
                    if working_token.dep_ != 'ROOT':
                        working_token = working_token.head
                    else:
                        return

        # handle phrases like 'he is easy to find', 'he is ready to go'
        # There is no way of knowing from the syntax whether the noun is a semantic
        # subject or object of the verb, so the new dependency label 'arg' is added.
        if token.tag_.startswith('NN') or self.is_involved_in_coreference(token):
            for adjective_dep in (
                    dep for dep in token._.holmes.children if
                    dep.label == self._modifier_dep and dep.child_token(token.doc).pos_ == 'ADJ'):
                adj_token = adjective_dep.child_token(token.doc)
                for verb_dep in (
                        dep for dep in adj_token._.holmes.children if
                        dep.label == 'xcomp' and dep.child_token(token.doc).pos_ == 'VERB'):
                    verb_token = verb_dep.child_token(token.doc)
                    verb_token._.holmes.children.append(SemanticDependency(
                        verb_token.i, token.i, 'arg', True))

class GermanSemanticAnalyzer(SemanticAnalyzer):

    language_name = 'German'

    noun_pos = ('NOUN', 'PROPN', 'ADJ')

    _matchable_pos = ('ADJ', 'ADP', 'ADV', 'NOUN', 'NUM', 'PROPN', 'VERB', 'AUX')

    _adjectival_predicate_head_pos = 'AUX'

    _adjectival_predicate_subject_pos = ('NOUN', 'PROPN', 'PRON')

    noun_kernel_dep = ('nk', 'pnc')

    sibling_marker_deps = ('cj', 'app')

    _adjectival_predicate_subject_dep = 'sb'

    _adjectival_predicate_predicate_dep = 'pd'

    _modifier_dep = 'nk'

    _spacy_noun_to_preposition_dep = 'mnr'

    _spacy_verb_to_preposition_dep = 'mo'

    _holmes_noun_to_preposition_dep = 'mnrposs'

    _holmes_verb_to_preposition_dep = 'moposs'

    _conjunction_deps = ('cj', 'cd', 'punct', 'app')

    _interrogative_pronoun_tags = ('PWAT', 'PWAV', 'PWS')

    _semantic_dependency_excluded_tags = ('ART')

    _generic_pronoun_lemmas = ('jemand', 'etwas')

    _or_lemma = 'oder'

    _matching_dep_dict = {
        'sb': ['pobjb', 'ag', 'arg', 'intcompound'],
        'ag': ['nk', 'pobjo', 'intcompound'],
        'oa': ['pobjo', 'ag', 'arg', 'intcompound', 'og', 'oc'],
        'arg': ['sb', 'oa', 'ag', 'intcompound', 'pobjb', 'pobjo'],
        'mo': ['moposs', 'mnr', 'mnrposs', 'nk', 'oc'],
        'mnr': ['mnrposs', 'mo', 'moposs', 'nk', 'oc'],
        'nk': ['ag', 'pobjo', 'intcompound', 'oc', 'mo'],
        'pobjo': ['ag', 'intcompound'],
        'pobjp': ['intcompound'],
        # intcompound is only used within extensive matching because it is not assigned
        # in the context of registering search phrases.
        'intcompound': ['sb', 'oa', 'ag', 'og', 'nk', 'mo', 'pobjo', 'pobjp']
    }

    _mark_child_dependencies_copied_to_siblings_as_uncertain = False

    # Never used at the time of writing
    _maximum_mentions_in_coreference_chain = 3

    # Never used at the time of writing
    _maximum_word_distance_in_coreference_chain = 300

    _model_supports_coreference_resolution = False

    _entity_defined_multiword_pos = ('NOUN', 'PROPN')

    _entity_defined_multiword_entity_types = ('PER', 'LOC')

    phraselet_templates = [
        PhraseletTemplate(
            "verb-nom", "Eine Sache tut", 2, 1,
            ['sb', 'pobjb'],
            [
                'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP',
                'VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'FM', 'NE', 'NNE', 'NN'],
            ['FM', 'NE', 'NNE', 'NN'], reverse_only=False),
        PhraseletTemplate(
            "verb-acc", "Jemand tut eine Sache", 1, 3,
            ['oa', 'pobjo', 'ag', 'og', 'oc'],
            [
                'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP',
                'VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'FM', 'NE', 'NNE', 'NN'],
            ['FM', 'NE', 'NNE', 'NN'], reverse_only=False),
        PhraseletTemplate(
            "verb-dat", "Jemand gibt einer Sache etwas", 1, 3,
            ['da'],
            [
                'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP',
                'VAFIN', 'VAIMP', 'VAINF', 'VAPP'],
            ['FM', 'NE', 'NNE', 'NN'], reverse_only=False),
        PhraseletTemplate(
            "verb-pd", "Jemand ist eine Sache", 1, 3,
            ['pd'],
            [
                'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP',
                'VAFIN', 'VAIMP', 'VAINF', 'VAPP'],
            ['FM', 'NE', 'NNE', 'NN'], reverse_only=True),
        PhraseletTemplate(
            "noun-dependent", "Eine beschriebene Sache", 2, 1,
            ['nk'],
            ['FM', 'NE', 'NNE', 'NN'],
            ['FM', 'NE', 'NNE', 'NN', 'ADJA', 'ADJD', 'ADV', 'CARD'], reverse_only=False),
        PhraseletTemplate(
            "verb-adverb", "schnell machen", 1, 0,
            ['mo', 'moposs', 'oc'],
            [
                'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP',
                'VAFIN', 'VAIMP', 'VAINF', 'VAPP'],
            ['ADJA', 'ADJD', 'ADV'], reverse_only=False),
        PhraseletTemplate(
            "prepgovernor-noun", "Eine Sache in einer Sache", 1, 4,
            ['pobjp'],
            [
                'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP',
                'VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'FM', 'NE', 'NNE', 'NN'],
            ['FM', 'NE', 'NNE', 'NN'], reverse_only=False),
        PhraseletTemplate(
            "prep-noun", "in einer Sache", 0, 2,
            ['nk'],
            ['APPO', 'APPR', 'APPRART', 'APZR'],
            ['FM', 'NE', 'NNE', 'NN'], reverse_only=True),
        PhraseletTemplate(
            "verb-toughmovedargument", "Eine Sache ist schwer zu tun", 5, 1,
            ['arg'],
            [
                'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP',
                'VAFIN', 'VAIMP', 'VAINF', 'VAPP'],
            ['FM', 'NE', 'NNE', 'NN'], reverse_only=False),
        PhraseletTemplate(
            "intcompound", "Eine Sache in einer Sache", 1, 4,
            ['intcompound'],
            ['NE', 'NNE', 'NN', 'TRUNC', 'ADJA', 'ADJD', 'TRUNC'],
            ['NE', 'NNE', 'NN', 'TRUNC', 'ADJA', 'ADJD', 'TRUNC'], reverse_only=False,
            assigned_dependency_label='intcompound'),
        PhraseletTemplate(
            "word", "Sache", 0, None,
            None,
            ['FM', 'NE', 'NNE', 'NN'],
            None, reverse_only=False)]

    topic_matching_phraselet_stop_lemmas = ('dann', 'danach', 'so', 'ich')

    supervised_document_classification_phraselet_stop_lemmas = ('sein', 'haben')

    topic_matching_reverse_only_parent_lemmas = (
        ('sein', 'AUX'), ('werden', 'AUX'), ('haben', 'AUX'), ('sagen', 'VERB'),
        ('machen', 'VERB'), ('tun', 'VERB'))

    preferred_phraselet_pos = ('NOUN', 'PROPN')

    _permissible_embedding_pos = ('NOUN', 'PROPN', 'ADJ', 'ADV')

    _minimum_embedding_match_word_length = 4

    # Only words at least this long are examined for possible subwords
    _minimum_length_for_subword_search = 10

    # Part-of-speech tags examined for subwords
    # Verbs are not examined because the separable parts that would typically be found as
    # subwords are too short to be found.
    _tag_for_subword_search = ('NE', 'NNE', 'NN', 'TRUNC', 'ADJA', 'ADJD')

    # Absolute minimum length of a subword.
    _minimum_subword_length = 3

    # Subwords at least this long are more likely to be genuine (not nonsensical) vocab entries.
    _minimum_long_subword_length = 6

    # Subwords longer than this are likely not be atomic and solutions that split them up are
    # preferred
    _maximum_realistic_subword_length = 12

    # Scoring bonus where a Fugen-S follows a whitelisted ending
    # (one where a Fugen-S is normally expected)
    _fugen_s_after_whitelisted_ending_bonus = 5

    # Scoring bonus where a Fugen-S follows an ending where it is neither expected nor disallowed
    _fugen_s_after_non_whitelisted_non_blacklisted_ending_bonus = 3

    # Both words around a Fugen-S have to be at least this long for the scoring bonus to be applied
    _fugen_s_whitelist_bonus_surrounding_word_minimum_length = 5

    # Endings after which a Fugen-S is normally expected
    _fugen_s_ending_whitelist = (
        'tum', 'ling', 'ion', 'tät', 'heit', 'keit', 'schaft', 'sicht', 'ung')

    # Endings after which a Fugen_S is normally disallowed
    _fugen_s_ending_blacklist = (
        'a', 'ä', 'e', 'i', 'o', 'ö', 'u', 'ü', 'nt', 'sch', 's', 'ß', 'st', 'tz', 'z')

    # Blacklisted subwords
    _subword_blacklist = (
        'igkeit', 'igkeiten', 'digkeit', 'digkeiten', 'schaft', 'schaften',
        'keit', 'keiten', 'lichkeit', 'lichkeiten', 'tigten', 'tigung', 'tigungen', 'barkeit',
        'barkeiten', 'heit', 'heiten', 'ung', 'ungen', 'aften', 'erung', 'erungen', 'mungen')

    # Bigraphs of two consonants that can occur at the start of a subword.
    _subword_start_consonant_bigraph_whitelist = (
        'bl', 'br', 'ch', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gm', 'gn', 'gr', 'kl', 'kn', 'kr',
        'kw', 'pf', 'ph', 'pl', 'pn', 'pr', 'ps', 'rh', 'sc', 'sh', 'sk', 'sl', 'sm', 'sp', 'st',
        'sw', 'sz', 'th', 'tr', 'vl', 'vr', 'wr', 'zw')

    # Bigraphs of two consonants that can occur at the end of a subword.
    # Bigraphs where the second consonant is 's' are always allowed.
    _subword_end_consonant_bigraph_whitelist = (
        'bb', 'bs', 'bt', 'ch', 'ck', 'ct', 'dd', 'ds', 'dt', 'ff', 'fs', 'ft', 'gd', 'gg', 'gn',
        'gs', 'gt', 'hb', 'hd', 'hf', 'hg', 'hk', 'hl', 'hm', 'hn', 'hp', 'hr', 'hs', 'ht', 'ks',
        'kt', 'lb', 'lc', 'ld', 'lf', 'lg', 'lk', 'll', 'lm', 'ln', 'lp', 'ls', 'lt', 'lx', 'lz',
        'mb', 'md', 'mk', 'mm', 'mp', 'ms', 'mt', 'mx', 'nb', 'nd', 'nf', 'ng', 'nk', 'nn', 'np',
        'ns', 'nt', 'nx', 'nz', 'pf', 'ph', 'pp', 'ps', 'pt', 'rb', 'rc', 'rd', 'rf', 'rg', 'rk',
        'rl', 'rm', 'rn', 'rp', 'rr', 'rs', 'rt', 'rx', 'rz', 'sk', 'sl', 'sp', 'ss', 'st', 'th',
        'ts', 'tt', 'tz', 'xt', 'zt', 'ßt')

    # Letters that can represent vowel sounds
    _vowels = ('a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü', 'y')

    # Subwords used in analysis but not recorded on the Holmes dictionary instances. At present
    # the code only supports these in word-final position; word-initial position would require
    # a code change.
    _non_recorded_subword_list = ('lein', 'chen')

    # Subword solutions that scored higher than this are regarded as probably wrong and so are
    # not recorded.
    _maximum_acceptable_subword_score = 8

    def _add_subwords(self, token, subword_cache):

        class PossibleSubword:
            """ A subword within a possible solution.

                text -- the text
                char_start_index -- the character start index of the subword within the word.
                fugen_s_status --
                '1' if the preceding word has an ending that normally has a Fugen-s,
                '2' if the preceding word has an ending that precludes using a Fugen-s,
                '0' otherwise.
            """

            def __init__(self, text, char_start_index, fugen_s_status):
                self.text = text
                self.char_start_index = char_start_index
                self.fugen_s_status = fugen_s_status

        def get_subword(lemma, initial_index, length):
            # find the shortest subword longer than length, unless length is less than
            # _minimum_long_subword_length, in which case only that length is tried. This strategy
            # is necessary because of the large number of nonsensical short vocabulary entries.
            for end_index in range(initial_index + length, len(lemma) + 1):
                possible_word = lemma[initial_index: end_index]
                if not self.nlp.vocab[possible_word].is_oov and len(possible_word) >= 2 and \
                        (
                            possible_word[0] in self._vowels or possible_word[1] in self._vowels
                            or
                            possible_word[:2] in self._subword_start_consonant_bigraph_whitelist) \
                        and (
                            possible_word[-1] in self._vowels or possible_word[-2] in self._vowels
                            or
                            possible_word[-2:] in self._subword_end_consonant_bigraph_whitelist):
                    return possible_word
                elif length < self._minimum_long_subword_length:
                    break
            return None

        def score(possible_solution):
            # Lower scores are better.
            number = 0
            for subword in possible_solution:
                # subwords shorter than _minimum_long_subword_length: penalty of 2
                if len(subword.text) < self._minimum_long_subword_length:
                    number += 2 * (self._minimum_long_subword_length - len(subword.text))
                # subwords longer than 12: penalty of 1
                elif len(subword.text) > self._maximum_realistic_subword_length:
                    number += len(subword.text) - self._maximum_realistic_subword_length
                # fugen-s after a whitelist ending
                if subword.fugen_s_status == 2:
                    number -= self._fugen_s_after_whitelisted_ending_bonus
                # fugen-s after an ending that is neither whitelist nor blacklist
                elif subword.fugen_s_status == 1:
                    number -= self._fugen_s_after_non_whitelisted_non_blacklisted_ending_bonus
            return number

        def scan_recursively_for_subwords(lemma, initial_index=0):

            if initial_index == 0: # only need to check on the initial (outermost) call
                for char in lemma:
                    if not char.isalpha() and char != '-':
                        return
            if initial_index + 1 < len(lemma) and lemma[initial_index] == '-':
                return scan_recursively_for_subwords(lemma, initial_index + 1)
            lengths = list(range(self._minimum_subword_length, 1 + len(lemma) - initial_index))
            possible_solutions = []
            working_subword = None
            for length in lengths:
                if working_subword is not None and len(working_subword) >= length:
                    # we are catching up with the length already returned by get_subword
                    continue
                working_subword = get_subword(lemma, initial_index, length)
                if working_subword is None or working_subword in self._subword_blacklist or \
                        '-' in working_subword:
                    continue
                possible_solution = [PossibleSubword(working_subword, initial_index, 0)]
                if \
                        (
                            initial_index + len(working_subword) == len(lemma)) or (
                                initial_index + len(working_subword)
                            + 1 == len(lemma) and lemma[-1] == '-') \
                        or (
                            initial_index + len(working_subword) + 2 == len(lemma) and lemma[-2:] ==
                            's-'):
                    # we have reached the end of the word
                    possible_solutions.append(possible_solution)
                    break
                following_subwords = scan_recursively_for_subwords(
                    lemma, initial_index + len(working_subword))
                if following_subwords is not None:
                    possible_solution.extend(following_subwords)
                    possible_solutions.append(possible_solution)
                if initial_index + len(working_subword) + 2 < len(lemma) and lemma[
                        initial_index + len(working_subword): initial_index +
                        len(working_subword) + 2] == 's-':
                    following_initial_index = initial_index + len(working_subword) + 2
                elif initial_index + len(working_subword) + 1 < len(lemma) and \
                        lemma[initial_index + len(working_subword)] == 's':
                    following_initial_index = initial_index + len(working_subword) + 1
                else:
                    continue
                possible_solution = [PossibleSubword(working_subword, initial_index, 0)]
                following_subwords = scan_recursively_for_subwords(lemma, following_initial_index)
                if following_subwords is not None:
                    for ending in self._fugen_s_ending_whitelist:
                        if working_subword.endswith(ending):
                            following_subwords[0].fugen_s_status = 2
                    if following_subwords[0].fugen_s_status == 0 and len(working_subword) >= \
                            self._fugen_s_whitelist_bonus_surrounding_word_minimum_length and \
                            len(following_subwords[0].text) >= \
                            self._fugen_s_whitelist_bonus_surrounding_word_minimum_length:
                        # if the first does not have a whitelist ending and one of the words is
                        # short, do not give the score bonus
                        following_subwords[0].fugen_s_status = 1
                        for ending in self._fugen_s_ending_blacklist:
                            # blacklist ending: take the bonus away again
                            if working_subword.endswith(ending):
                                following_subwords[0].fugen_s_status = 0
                    possible_solution.extend(following_subwords)
                    possible_solutions.append(possible_solution)
            if len(possible_solutions) > 0:
                possible_solutions = sorted(
                    possible_solutions, key=lambda possible_solution: score(possible_solution))
                return possible_solutions[0]

        def get_lemmatization_doc(possible_subwords, pos):
            # We retrieve the lemma for each subword by calling Spacy. To reduce the
            # overhead, we concatenate the subwords in the form:
            # Subword1. Subword2. Subword3
            entry_words = []
            for counter in range(len(possible_subwords)):
                if counter + 1 == len(possible_subwords) and pos == 'ADJ':
                    entry_words.append(possible_subwords[counter].text)
                else:
                    entry_words.append(possible_subwords[counter].text.capitalize())
            subword_lemmatization_string = ' . '.join(entry_words)
            return self.spacy_parse(subword_lemmatization_string)

        if not token.tag_ in self._tag_for_subword_search or (
                len(token._.holmes.lemma) < self._minimum_length_for_subword_search and
                '-' not in token._.holmes.lemma):
            return
        if token.text in subword_cache:
            cached_subwords = subword_cache[token.text]
            for cached_subword in cached_subwords:
                token._.holmes.subwords.append(Subword(
                    token.i, cached_subword.index, cached_subword.text, cached_subword.lemma,
                    cached_subword.derived_lemma, cached_subword.char_start_index,
                    cached_subword.dependent_index, cached_subword.dependency_label,
                    cached_subword.governor_index, cached_subword.governing_dependency_label))
        else:
            working_subwords = []
            possible_subwords = scan_recursively_for_subwords(token._.holmes.lemma)
            if possible_subwords is None or score(possible_subwords) > \
                    self._maximum_acceptable_subword_score:
                return
            if len(possible_subwords) == 1 and token._.holmes.lemma.isalpha():
                # not ... isalpha(): hyphenation
                subword_cache[token.text] = []
            else:
                index = 0
                if token._.holmes.lemma[0] == '-':
                    # with truncated nouns, the righthand siblings may actually occur to the left
                    # of the head noun
                    head_sibling = token.doc[token._.holmes.token_or_lefthand_sibling_index]
                    if len(head_sibling._.holmes.righthand_siblings) > 0:
                        indexes = token._.holmes.get_sibling_indexes(token.doc)
                        first_sibling = token.doc[indexes[0]]
                        first_sibling_possible_subwords = \
                                scan_recursively_for_subwords(first_sibling._.holmes.lemma)
                        if first_sibling_possible_subwords is not None:
                            first_sibling_lemmatization_doc = get_lemmatization_doc(
                                first_sibling_possible_subwords, token.pos_)
                            final_subword_counter = len(first_sibling_possible_subwords) - 1
                            if final_subword_counter > 0 and \
                                    first_sibling_possible_subwords[
                                        final_subword_counter].text \
                                    in self._non_recorded_subword_list:
                                final_subword_counter -= 1
                            for counter in range(final_subword_counter):
                                first_sibling_possible_subword = \
                                    first_sibling_possible_subwords[counter]
                                if first_sibling_possible_subword.text in \
                                        self._non_recorded_subword_list:
                                    continue
                                text = first_sibling.text[
                                    first_sibling_possible_subword.char_start_index:
                                    first_sibling_possible_subword.char_start_index +
                                    len(first_sibling_possible_subword.text)]
                                lemma = first_sibling_lemmatization_doc[counter*2].lemma_.lower()
                                derived_lemma = self.derived_holmes_lemma(None, lemma)
                                working_subwords.append(Subword(
                                    first_sibling.i, index, text, lemma, derived_lemma,
                                    first_sibling_possible_subword.char_start_index,
                                    None, None, None, None))
                                index += 1
                lemmatization_doc = get_lemmatization_doc(possible_subwords, token.pos_)
                for counter, possible_subword in enumerate(possible_subwords):
                    possible_subword = possible_subwords[counter]
                    if possible_subword.text in self._non_recorded_subword_list:
                        continue
                    text = token.text[
                        possible_subword.char_start_index:
                        possible_subword.char_start_index + len(possible_subword.text)]
                    lemma = lemmatization_doc[counter*2].lemma_.lower()
                    derived_lemma = self.derived_holmes_lemma(None, lemma)
                    working_subwords.append(Subword(
                        token.i, index, text, lemma, derived_lemma,
                        possible_subword.char_start_index, None, None, None, None))
                    index += 1
                if token._.holmes.lemma[-1] == '-':
                    # with truncated nouns, the righthand siblings may actually occur to the left
                    # of the head noun
                    head_sibling = token.doc[token._.holmes.token_or_lefthand_sibling_index]
                    if len(head_sibling._.holmes.righthand_siblings) > 0:
                        indexes = token._.holmes.get_sibling_indexes(token.doc)
                        last_sibling_index = indexes[-1]
                        if token.i != last_sibling_index:
                            last_sibling = token.doc[last_sibling_index]
                            last_sibling_possible_subwords = \
                                scan_recursively_for_subwords(last_sibling._.holmes.lemma)
                            if last_sibling_possible_subwords is not None:
                                last_sibling_lemmatization_doc = get_lemmatization_doc(
                                    last_sibling_possible_subwords, token.pos_)
                                for counter in range(1, len(last_sibling_possible_subwords)):
                                    last_sibling_possible_subword = \
                                        last_sibling_possible_subwords[counter]
                                    if last_sibling_possible_subword.text in \
                                            self._non_recorded_subword_list:
                                        continue
                                    text = last_sibling.text[
                                        last_sibling_possible_subword.char_start_index:
                                        last_sibling_possible_subword.char_start_index +
                                        len(last_sibling_possible_subword.text)]
                                    lemma = last_sibling_lemmatization_doc[counter*2].lemma_.lower()
                                    derived_lemma = self.derived_holmes_lemma(None, lemma)
                                    working_subwords.append(Subword(
                                        last_sibling.i, index, text, lemma, derived_lemma,
                                        last_sibling_possible_subword.char_start_index,
                                        None, None, None, None))
                                    index += 1

                if index > 1: # if only one subword was found, no need to record it on ._.holmes
                    for counter, working_subword in enumerate(working_subwords):
                        if counter > 0:
                            dependency_label = 'intcompound'
                            dependent_index = counter - 1
                        else:
                            dependency_label = None
                            dependent_index = None
                        if counter + 1 < len(working_subwords):
                            governing_dependency_label = 'intcompound'
                            governor_index = counter + 1
                        else:
                            governing_dependency_label = None
                            governor_index = None
                        working_subword = working_subwords[counter]
                        token._.holmes.subwords.append(Subword(
                            working_subword.containing_token_index,
                            working_subword.index, working_subword.text, working_subword.lemma,
                            working_subword.derived_lemma, working_subword.char_start_index,
                            dependent_index, dependency_label, governor_index,
                            governing_dependency_label))
                if token._.holmes.lemma.isalpha(): # caching only where no hyphenation
                    subword_cache[token.text] = token._.holmes.subwords
        if len(token._.holmes.subwords) > 1 and 'nicht' in (
                subword.lemma for subword in token._.holmes.subwords):
            token._.holmes.is_negated = True

    def _set_negation(self, token):
        """Marks the negation on the token. A token is negative if it or one of its ancestors
            has a negation word as a syntactic (not semantic!) child.
        """
        if token._.holmes.is_negated is not None:
            return
        for child in token.children:
            if child._.holmes.lemma in ('nicht', 'kein', 'keine', 'nie') or \
                    child._.holmes.lemma.startswith('nirgend'):
                token._.holmes.is_negated = True
                return
        if token.dep_ == 'ROOT':
            token._.holmes.is_negated = False
            return
        self._set_negation(token.head)
        token._.holmes.is_negated = token.head._.holmes.is_negated

    def _correct_auxiliaries_and_passives(self, token):
        """Wherever auxiliaries and passives are found, derive the semantic information
            from the syntactic information supplied by spaCy.
        """

        def correct_auxiliaries_and_passives_recursively(token, processed_auxiliary_indexes):
            if token.i not in processed_auxiliary_indexes:
                processed_auxiliary_indexes.append(token.i)
                if (token.pos_ == 'AUX' or token.tag_.startswith('VM')) and len([
                        dependency for dependency in token._.holmes.children if
                        dependency.child_index >= 0 and
                        token.doc[dependency.child_index].tag_ == 'PTKVZ']) == 0: # 'vorhaben'
                    for dependency in (
                            dependency for dependency in token._.holmes.children
                            if token.doc[dependency.child_index].pos_ in ('VERB', 'AUX') and
                            token.doc[dependency.child_index].dep_ in ('oc', 'pd')):
                        token._.holmes.is_matchable = False
                        child = token.doc[dependency.child_index]
                        self._move_information_between_tokens(token, child)
                        # VM indicates a modal verb, which has to be marked as uncertain
                        if token.tag_.startswith('VM') or dependency.is_uncertain:
                            for child_dependency in child._.holmes.children:
                                child_dependency.is_uncertain = True
                        # 'er ist froh zu kommen' / 'er ist schwer zu erreichen'
                        # set dependency label to 'arg' because semantic role could be either
                        # subject or object
                        if token._.holmes.lemma == 'sein' and (
                                len([
                                    child_dependency for child_dependency in
                                    child._.holmes.children if child_dependency.label == 'pm' and
                                    child_dependency.child_token(token.doc).tag_ == 'PTKZU']) > 0
                                or child.tag_ == 'VVIZU'):
                            for new_dependency in (
                                    new_dependency for new_dependency in
                                    child._.holmes.children if new_dependency.label == 'sb'):
                                new_dependency.label = 'arg'
                                new_dependency.is_uncertain = True
                        # passive construction
                        if (token._.holmes.lemma == 'werden' and child.tag_ not in (
                                'VVINF', 'VAINF', 'VAFIN', 'VAINF')):
                            for child_or_sib in \
                                    child._.holmes.loop_token_and_righthand_siblings(token.doc):
                                #mark syntactic subject as semantic object
                                for grandchild_dependency in [
                                        grandchild_dependency for
                                        grandchild_dependency in child_or_sib._.holmes.children
                                        if grandchild_dependency.label == 'sb']:
                                    grandchild_dependency.label = 'oa'
                                #mark syntactic object as synctactic subject, removing the
                                #preposition 'von' or 'durch' from the construction and marking
                                #it as non-matchable
                                for grandchild_dependency in (
                                        gd for gd in
                                        child_or_sib._.holmes.children if gd.child_index >= 0):
                                    grandchild = token.doc[grandchild_dependency.child_index]
                                    if (
                                            grandchild_dependency.label == 'sbp' and
                                            grandchild._.holmes.lemma in ('von', 'vom')) or (
                                                grandchild_dependency.label == 'mo' and
                                                grandchild._.holmes.lemma in (
                                                    'von', 'vom', 'durch')):
                                        grandchild._.holmes.is_matchable = False
                                        for great_grandchild_dependency in \
                                                grandchild._.holmes.children:
                                            if child_or_sib.i != \
                                                    great_grandchild_dependency.child_index:
                                                child_or_sib._.holmes.children.append(
                                                    SemanticDependency(
                                                        child_or_sib.i,
                                                        great_grandchild_dependency.child_index,
                                                        'sb', dependency.is_uncertain))
                                        child_or_sib._.holmes.remove_dependency_with_child_index(
                                            grandchild_dependency.child_index)
            for syntactic_child in token.children:
                correct_auxiliaries_and_passives_recursively(
                    syntactic_child, processed_auxiliary_indexes)

        if token.dep_ == 'ROOT':
            correct_auxiliaries_and_passives_recursively(token, [])

    def _handle_relative_constructions(self, token):
        for dependency in (
                dependency for dependency in token._.holmes.children if
                dependency.child_index >= 0 and
                dependency.child_token(token.doc).tag_ in ('PRELS', 'PRELAT') and
                dependency.child_token(token.doc).dep_ != 'par'):
            counter = dependency.child_index
            while counter > token.sent.start:
                # find the antecedent
                counter -= 1
                working_token = token.doc[counter]
                if working_token.pos_ in ('NOUN', 'PROPN') and working_token.dep_ not in \
                        self.sibling_marker_deps:
                    working_dependency = None
                    for antecedent in (
                            antecedent for antecedent in
                            working_token._.holmes.loop_token_and_righthand_siblings(token.doc)
                            if antecedent.i != token.i):
                        # add new dependency from the verb to the antecedent
                        working_dependency = SemanticDependency(
                            token.i, antecedent.i, dependency.label, True)
                        token._.holmes.children.append(working_dependency)
                    # the last antecedent before the pronoun is not uncertain, so reclassify it
                    if working_dependency is not None:
                        working_dependency.is_uncertain = False
                        # remove the dependency from the verb to the relative pronoun
                        token._.holmes.remove_dependency_with_child_index(
                            dependency.child_index)
                        # label the relative pronoun as a grammatical token pointing to its
                        # direct antecedent
                        dependency.child_token(token.doc)._.holmes.children = [SemanticDependency(
                            dependency.child_index, 0 - (working_dependency.child_index + 1),
                            None)]

    def _holmes_lemma(self, token):
        """Relabel the lemmas of separable verbs in sentences like 'er steht auf' to incorporate
            the entire separable verb to facilitate matching.
        """
        if token.pos_ in ('VERB', 'AUX') and token.tag_ not in ('VAINF', 'VMINF', 'VVINF', 'VVIZU'):
            for child in token.children:
                if child.tag_ == 'PTKVZ':
                    child_lemma = child.lemma_.lower()
                    if child_lemma == 'einen':
                        child_lemma = 'ein'
                    return ''.join([child_lemma, token.lemma_.lower()])
        if token.tag_ == 'APPRART':
            if token.lemma_.lower() == 'im':
                return 'in'
            if token.lemma_.lower() == 'am':
                return 'an'
            if token.lemma_.lower() == 'beim':
                return 'bei'
            if token.lemma_.lower() == 'zum':
                return 'zu'
            if token.lemma_.lower() == 'zur':
                return 'zu'
        # sometimes adjectives retain their inflectional endings
        if token.tag_ == 'ADJA' and len(token.lemma_.lower()) > 5 and \
                token.lemma_.lower().endswith('en'):
            return token.lemma_.lower().rstrip('en')
        if token.tag_ == 'ADJA' and len(token.lemma_.lower()) > 5 and \
                token.lemma_.lower().endswith('e'):
            return token.lemma_.lower().rstrip('e')
        return token.lemma_.lower()

    _ung_ending_blacklist = ('sprung', 'schwung', 'nibelung')

    def normalize_hyphens(self, word):
        """ Normalizes hyphens in a multiword for ontology matching. Depending on the language,
            this may involve replacing them with spaces (English) or deleting them entirely
            (German).
        """
        if word.strip().startswith('-') or word.endswith('-'):
            return word
        else:
            return word.replace('-', '')

    def _language_specific_derived_holmes_lemma(self, token, lemma):
        """ token is None where *lemma* belongs to a subword """

        # verbs with 'ieren' -> 'ation'
        if (token is None or token.pos_ == 'VERB') and len(lemma) > 9 and \
                lemma.endswith('ieren'):
            working_lemma = ''.join((lemma[:-5], 'ation'))
            if not self.nlp.vocab[working_lemma].is_oov:
                return working_lemma
        # nouns with 'ierung' -> 'ation'
        if (token is None or token.pos_ == 'NOUN') and len(lemma) > 10 and \
                lemma.endswith('ierung'):
            working_lemma = ''.join((lemma[:-6], 'ation'))
            if not self.nlp.vocab[working_lemma].is_oov:
                return working_lemma
        # nominalization with 'ung'
        if (token is None or token.tag_ == 'NN') and lemma.endswith('ung'):
            for word in self._ung_ending_blacklist:
                if lemma.endswith(word):
                    return None
            if (lemma.endswith('erung') and not lemma.endswith('ierung')) or \
                    lemma.endswith('elung'):
                return ''.join((lemma[:-3], 'n'))
            elif lemma.endswith('lung') and len(lemma) >= 5 and \
                    lemma[-5] not in ('a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü', 'h'):
                return ''.join((lemma[:-4], 'eln'))
            return ''.join((lemma[:-3], 'en'))
        # nominalization with 'heit', 'keit'
        if (token is None or token.tag_ == 'NN') and (
                lemma.endswith('keit') or lemma.endswith('heit')):
            return lemma[:-4]
        if (token is None or token.pos_ in ('NOUN', 'PROPN')) and len(lemma) > 6 and \
                (lemma.endswith('chen') or lemma.endswith('lein')):
            # len > 6: because e.g. Dach and Loch have lemmas 'dachen' and 'lochen'
            working_lemma = lemma[-12:-4]
            # replace umlauts in the last 8 characters of the derived lemma
            working_lemma = working_lemma.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')
            working_lemma = ''.join((lemma[:-12], working_lemma))
            if not self.nlp.vocab[working_lemma].is_oov:
                return working_lemma
            if lemma[-4] == 'l': # 'lein' where original word ends in 'l'
                second_working_lemma = ''.join((working_lemma, 'l'))
                if not self.nlp.vocab[working_lemma].is_oov:
                    return second_working_lemma
            second_working_lemma = lemma[:-4] # 'Löffelchen'
            if not self.nlp.vocab[second_working_lemma].is_oov:
                return second_working_lemma
            if lemma[-4] == 'l': # 'Schlüsselein'
                second_working_lemma = ''.join((second_working_lemma, 'l'))
                if not self.nlp.vocab[second_working_lemma].is_oov:
                    return second_working_lemma
            return working_lemma
        if (token is None or token.tag_ == 'NN') and lemma.endswith('e') and len(lemma) > 1 and \
                not lemma[-2] in self._vowels:
            # for comparability with diminutive forms, e.g. äuglein <-> auge
            return lemma[:-1]
        return None

    def _perform_language_specific_tasks(self, token):

        # Because separable verbs are conflated into a single lemma, remove the dependency
        # from the verb to the preposition
        if token.tag_ == 'PTKVZ' and token.head.pos_ in ('VERB', 'AUX') and \
                token.head.tag_ not in ('VAINF', 'VMINF', 'VVINF', 'VVIZU'):
            token.head._.holmes.remove_dependency_with_child_index(token.i)

        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.label in ('mo', 'mnr', 'pg', 'op')):
            child = dependency.child_token(token.doc)
            for child_dependency in (
                    child_dependency for child_dependency in
                    child._.holmes.children if child_dependency.label == 'nk' and
                    token.i != child_dependency.child_index and child.pos_ == 'ADP'):
                if dependency.label in ('mnr', 'pg', 'op') and \
                        dependency.child_token(token.doc)._.holmes.lemma in ('von', 'vom'):
                    token._.holmes.children.append(SemanticDependency(
                        token.i, child_dependency.child_index, 'pobjo'))
                        # pobjO from English 'of'
                    child._.holmes.is_matchable = False
                elif dependency.label in ('mnr') and \
                        dependency.child_token(token.doc)._.holmes.lemma in ('durch'):
                    token._.holmes.children.append(SemanticDependency(
                        token.i, child_dependency.child_index, 'pobjb'))
                        # pobjB from English 'by'
                else:
                    token._.holmes.children.append(SemanticDependency(
                        token.i, child_dependency.child_index, 'pobjp',
                        dependency.is_uncertain or child_dependency.is_uncertain))

        # # where a 'moposs' or 'mnrposs' dependency has been added and the preposition is not
        # 'von' or 'vom' add a corresponding uncertain 'pobjp'
        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.label in ['moposs', 'mnrposs']):
            child = dependency.child_token(token.doc)
            for child_dependency in (
                    child_dependency for child_dependency in
                    child._.holmes.children if child_dependency.label == 'nk' and
                    token.i != child_dependency.child_index and child._.holmes.is_matchable):
                token._.holmes.children.append(SemanticDependency(
                    token.i, child_dependency.child_index, 'pobjp', True))

        # Loop through the structure around a dependent verb to find the lexical token at which
        # to add new dependencies, and find out whether it is active or passive so we know
        # whether to add an 'sb' or an 'oa'.
        def find_target_tokens_and_dependency_recursively(token, visited=[]):
            visited.append(token.i)
            tokens_to_return = []
            target_dependency = 'sb'
            # Loop through grammatical tokens. 'dependency.child_index + token.i != -1' would mean
            # a grammatical token were pointing to itself (should never happen!)
            if len([
                    dependency for dependency in token._.holmes.children
                    if dependency.child_index < 0 and dependency.child_index + token.i != -1]) > 0:
                for dependency in (
                        dependency for dependency in token._.holmes.children
                        if dependency.child_index < 0 and dependency.child_index + token.i != -1):
                    # resolve the grammatical token pointer
                    child_token = token.doc[0 - (dependency.child_index + 1)]
                    # passive construction
                    if (token._.holmes.lemma == 'werden' and child_token.tag_ not in
                            ('VVINF', 'VAINF', 'VAFIN', 'VAINF')):
                        target_dependency = 'oa'
                    if child_token.i not in visited:
                        new_tokens, new_target_dependency = \
                            find_target_tokens_and_dependency_recursively(child_token, visited)
                        tokens_to_return.extend(new_tokens)
                        if new_target_dependency == 'oa':
                            target_dependency = 'oa'
                    else:
                        tokens_to_return.append(token)
            else:
                # we have reached the target token
                tokens_to_return.append(token)
            return tokens_to_return, target_dependency

        # 'Der Mann hat xxx, es zu yyy' and similar structures
        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.label in ('oc', 'oa', 'mo', 're') and
                token.pos_ in ('VERB', 'AUX') and dependency.child_token(token.doc).pos_ in \
                ('VERB', 'AUX')):
            dependencies_to_add = []
            target_tokens, target_dependency = find_target_tokens_and_dependency_recursively(
                dependency.child_token(token.doc))
            # with um ... zu structures the antecedent subject is always the subject of the
            # dependent clause, unlike with 'zu' structures without the 'um'
            if len([other_dependency for other_dependency in target_tokens[0]._.holmes.children
                    if other_dependency.child_token(token.doc)._.holmes.lemma == 'um' and
                    other_dependency.child_token(token.doc).tag_ == 'KOUI']) == 0:
                # er hat ihm vorgeschlagen, etwas zu tun
                for other_dependency in (
                        other_dependency for other_dependency
                        in token._.holmes.children if other_dependency.label == 'da'):
                    dependencies_to_add.append(other_dependency)
                if len(dependencies_to_add) == 0:
                    # er hat ihn gezwungen, etwas zu tun
                    # We have to distinguish this type of 'oa' relationship from dependent
                    # clauses and reflexive pronouns ('er entschied sich, ...')
                    for other_dependency in (
                            other_dependency for other_dependency
                            in token._.holmes.children if other_dependency.label == 'oa' and
                            other_dependency.child_token(token.doc).pos_ not in ('VERB', 'AUX') and
                            other_dependency.child_token(token.doc).tag_ != 'PRF'):
                        dependencies_to_add.append(other_dependency)
            if len(dependencies_to_add) == 0:
                # We haven't found any object dependencies, so take the subject dependency
                for other_dependency in (
                        other_dependency for other_dependency
                        in token._.holmes.children if other_dependency.label == 'sb'):
                    dependencies_to_add.append(other_dependency)
            for target_token in target_tokens:
                for other_dependency in (
                        other_dependency for other_dependency in
                        dependencies_to_add if target_token.i != other_dependency.child_index):
                    # these dependencies are always uncertain
                    target_token._.holmes.children.append(SemanticDependency(
                        target_token.i, other_dependency.child_index, target_dependency, True))

        # 'Der Löwe bat den Hund, die Katze zu jagen' and similar structures
        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.label == 'oc' and token.pos_ == 'NOUN' and
                dependency.child_token(token.doc).pos_ in ('VERB', 'AUX')):
            target_tokens, target_dependency = find_target_tokens_and_dependency_recursively(
                dependency.child_token(token.doc))
            for target_token in target_tokens:
                target_token._.holmes.children.append(SemanticDependency(
                    target_token.i, token.i, target_dependency, True))

        # 'er dachte darüber nach, es zu tun' and similar structures
        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.label == 'op' and dependency.child_token(token.doc).tag_ == 'PROAV'):
            child_token = dependency.child_token(token.doc)
            for child_dependency in (
                    child_dependency for child_dependency in
                    child_token._.holmes.children if child_dependency.label == 're' and
                    child_dependency.child_token(token.doc).pos_ in ('VERB', 'AUX')):
                target_tokens, target_dependency = find_target_tokens_and_dependency_recursively(
                    child_dependency.child_token(token.doc))
                for other_dependency in (
                        other_dependency for other_dependency
                        in token._.holmes.children if other_dependency.label == 'sb'):
                    for target_token in target_tokens:
                        target_token._.holmes.children.append(SemanticDependency(
                            target_token.i, other_dependency.child_index, target_dependency, True))

        # 'er war froh, etwas zu tun'
        for dependency in (
                dependency for dependency in token._.holmes.children
                if dependency.label == 'nk' and token.pos_ in ('NOUN', 'PROPN')
                and token.dep_ == 'sb' and dependency.child_token(token.doc).pos_ == 'ADJ'):
            child_token = dependency.child_token(token.doc)
            for child_dependency in (
                    child_dependency for child_dependency in
                    child_token._.holmes.children if child_dependency.label in ('oc', 're') and
                    child_dependency.child_token(token.doc).pos_ in ('VERB', 'AUX')):
                target_tokens, target_dependency = find_target_tokens_and_dependency_recursively(
                    child_dependency.child_token(token.doc))
                for target_token in (
                        target_token for target_token in target_tokens
                        if target_token.i != dependency.parent_index):
                    # these dependencies are always uncertain
                    target_token._.holmes.children.append(SemanticDependency(
                        target_token.i, dependency.parent_index, target_dependency, True))

        # sometimes two verb arguments are interpreted as both subjects or both objects,
        # if this occurs reinterpret them

        # find first 'sb' dependency for verb
        dependencies = [
            dependency for dependency in token._.holmes.children
            if token.pos_ == 'VERB' and dependency.label == 'sb' and not
            dependency.is_uncertain]
        if len(dependencies) > 0 and len([
                object_dependency for object_dependency
                in dependencies if object_dependency.label == 'oa' and not
                dependency.is_uncertain]) == 0:
            dependencies.sort(key=lambda dependency: dependency.child_index)
            first_real_subject = dependencies[0].child_token(token.doc)
            for real_subject_index in \
                    first_real_subject._.holmes.get_sibling_indexes(token.doc):
                for dependency in dependencies:
                    if dependency.child_index == real_subject_index:
                        dependencies.remove(dependency)
            for dependency in (other_dependency for other_dependency in dependencies):
                dependency.label = 'oa'

        dependencies = [
            dependency for dependency in token._.holmes.children
            if token.pos_ == 'VERB' and dependency.label == 'oa' and not
            dependency.is_uncertain]
        if len(dependencies) > 0 and len([
                object_dependency for object_dependency
                in dependencies if object_dependency.label == 'sb' and not
                dependency.is_uncertain]) == 0:
            dependencies.sort(key=lambda dependency: dependency.child_index)
            first_real_subject = dependencies[0].child_token(token.doc)
            real_subject_indexes = first_real_subject._.holmes.get_sibling_indexes(token.doc)
            if len(dependencies) > len(real_subject_indexes):
                for dependency in (
                        dependency for dependency in dependencies if
                        dependency.child_index in real_subject_indexes):
                    dependency.label = 'sb'
