import copy
from .errors import *
from .semantics import SemanticDependency, Subword
from threading import Lock
from functools import total_ordering
from spacy.tokens.token import Token

ONTOLOGY_DEPTHS_TO_NAMES = {-4: 'an ancestor', -3: 'a great-grandparent', -2: 'a grandparent',
        -1: 'a parent', 0: 'a synonym', 1: 'a child', 2: 'a grandchild', 3: 'a great-grandchild',
        4: 'a descendant'}

class WordMatch:
    """A match between a searched phrase word and a document word.

    Properties:

    search_phrase_token -- the spaCy token from the search phrase.
    search_phrase_word -- the word that matched from the search phrase.
    document_token -- the spaCy token from the document.
    first_document_token -- the first token that matched from the document, which will equal
        *document_token* except with multiword matches.
    last_document_token -- the lst token that matched from the document, which will equal
        *document_token* except with multiword matches.
    document_subword -- the subword from the token that matched, or *None* if the match was
        with the whole token.
    document_word -- the word or subword that matched structurally from the document.
    type -- *direct*, *entity*, *embedding*, *ontology* or *derivation*.
    similarity_measure -- for type *embedding*, the similarity between the two tokens,
        otherwise 1.0.
    is_negated -- *True* if this word match leads to a match of which it
      is a part being negated.
    is_uncertain -- *True* if this word match leads to a match of which it
      is a part being uncertain.
    structurally_matched_document_token -- the spaCy token from the document that matched
        the dependency structure, which may be different from *document_token* if coreference
        resolution is active.
    involves_coreference -- *True* if *document_token* and *structurally_matched_document_token*
        are different.
    extracted_word -- within the coreference chain, the most specific term that corresponded to
      document_word in the ontology.
    depth -- the number of hyponym relationships linking *search_phrase_word* and
        *extracted_word*, or *0* if ontology-based matching is not active.
    """

    def __init__(self, search_phrase_token, search_phrase_word, document_token,
            first_document_token, last_document_token, document_subword, document_word,
            type, similarity_measure, is_negated, is_uncertain,
            structurally_matched_document_token, extracted_word, depth):

        self.search_phrase_token = search_phrase_token
        self.search_phrase_word = search_phrase_word
        self.document_token = document_token
        self.first_document_token = first_document_token
        self.last_document_token = last_document_token
        self.document_subword = document_subword
        self.document_word = document_word
        self.type = type
        self.similarity_measure = similarity_measure
        self.is_negated = is_negated
        self.is_uncertain = is_uncertain
        self.structurally_matched_document_token = structurally_matched_document_token
        self.extracted_word = extracted_word
        self.depth = depth

    @property
    def involves_coreference(self):
        return self.document_token != self.structurally_matched_document_token

    def get_document_index(self):
        if self.document_subword != None:
            subword_index = self.document_subword.index
        else:
            subword_index = None
        return Index(self.document_token.i, subword_index)

    def explain(self):
        """ Creates a human-readable explanation of the word match from the perspective of the
            document word (e.g. to be used as a tooltip over it)."""
        search_phrase_display_word = self.search_phrase_token._.holmes.lemma.upper()
        if self.type == 'direct':
            return ''.join(("Matches ", search_phrase_display_word, " directly."))
        elif self.type == 'derivation':
            return ''.join(("Has a common stem with ", search_phrase_display_word, "."))
        elif self.type == 'entity':
            return ''.join(("Matches the ", search_phrase_display_word, " placeholder."))
        elif self.type == 'embedding':
            printable_similarity = str(int(self.similarity_measure * 100))
            return ''.join(("Has a word embedding that is ", printable_similarity,
                    "% similar to ", search_phrase_display_word, "."))
        elif self.type == 'ontology':
            working_depth = self.depth
            if working_depth > 4:
                working_depth = 4
            elif working_depth < -4:
                working_depth = -4
            return ''.join(("Is ", ONTOLOGY_DEPTHS_TO_NAMES[working_depth], " of ",
                    search_phrase_display_word, " in the ontology."))
        else:
            raise RuntimeError(' '.join(('Unrecognized type', self.type)))

class Match:
    """A match between a search phrase and a document.

    Properties:

    word_matches -- a list of *WordMatch* objects.
    is_negated -- *True* if this match is negated.
    is_uncertain -- *True* if this match is uncertain.
    involves_coreference -- *True* if this match was found using coreference resolution.
    search_phrase_label -- the label of the search phrase that matched.
    document_label -- the label of the document that matched.
    from_single_word_phraselet -- *True* if this is a match against a single-word
        phraselet.
    from_topic_match_phraselet_created_without_matching_tags -- **True** or **False**
    from_reverse_only_topic_match_phraselet -- **True** or **False**
    overall_similarity_measure -- the overall similarity of the match, or *1.0* if the embedding
        strategy was not involved in the match.
    index_within_document -- the index of the document token that matched the search phrase
        root token.
    """

    def __init__(self, search_phrase_label, document_label, from_single_word_phraselet,
            from_topic_match_phraselet_created_without_matching_tags,
            from_reverse_only_topic_match_phraselet):
        self.word_matches = []
        self.is_negated = False
        self.is_uncertain = False
        self.search_phrase_label = search_phrase_label
        self.document_label = document_label
        self.from_single_word_phraselet = from_single_word_phraselet
        self.from_topic_match_phraselet_created_without_matching_tags = \
                from_topic_match_phraselet_created_without_matching_tags
        self.from_reverse_only_topic_match_phraselet = from_reverse_only_topic_match_phraselet
        self.index_within_document = None
        self.overall_similarity_measure = '1.0'

    @property
    def involves_coreference(self):
        for word_match in self.word_matches:
            if word_match.involves_coreference:
                return True
        return False

    def __copy__(self):
        match_to_return = Match(self.search_phrase_label, self.document_label,
                self.from_single_word_phraselet,
                self.from_topic_match_phraselet_created_without_matching_tags,
                self.from_reverse_only_topic_match_phraselet)
        match_to_return.word_matches = self.word_matches.copy()
        match_to_return.is_negated = self.is_negated
        match_to_return.is_uncertain = self.is_uncertain
        match_to_return.index_within_document = self.index_within_document
        return match_to_return

    def get_subword_index(self):
        if self.word_matches[0].document_subword == None:
            return None
        return self.word_matches[0].document_subword.index

    def get_subword_index_for_sorting(self):
        # returns *-1* rather than *None* in the absence of a subword
        if self.word_matches[0].document_subword == None:
            return -1
        return self.word_matches[0].document_subword.index

@total_ordering
class Index:
    """ The position of a word or subword within a document. """

    def __init__(self, token_index, subword_index):
        self.token_index = token_index
        self.subword_index = subword_index

    def is_subword(self):
        return self.subword_index != None

    def __eq__(self, other):
        return type(other) == Index and \
                self.token_index == other.token_index and self.subword_index == other.subword_index

    def __lt__(self, other):
        if type(other) != Index:
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

class PhraseletInfo:
    """Information describing a topic matching phraselet.

        Parameters:

        label -- the phraselet label.
        template_label -- the value of 'PhraseletTemplate.label'.
        parent_lemma -- the parent lemma, or the lemma for single-word phraselets.
        parent_derived_lemma -- the parent derived lemma, or the derived lemma for single-word
            phraselets.
        parent_pos -- the part of speech tag of the token that supplied the parent word.
        child_lemma -- the child lemma, or 'None' for single-word phraselets.
        child_derived_lemma -- the child derived lemma, or 'None' for single-word phraselets.
        child_pos -- the part of speech tag of the token that supplied the child word, or 'None'
            for single-word phraselets.
        created_without_matching_tags -- 'True' if created without matching tags.
        reverse_only_parent_lemma -- 'True' if the parent lemma is in the reverse matching list.
    """

    def __init__(self, label, template_label, parent_lemma, parent_derived_lemma, parent_pos,
            child_lemma, child_derived_lemma, child_pos, created_without_matching_tags,
            reverse_only_parent_lemma):
        self.label = label
        self.template_label = template_label

        self.parent_lemma = parent_lemma
        self.parent_derived_lemma = parent_derived_lemma
        self.parent_pos = parent_pos
        self.child_lemma = child_lemma
        self.child_derived_lemma = child_derived_lemma
        self.child_pos = child_pos
        self.created_without_matching_tags = created_without_matching_tags
        self.reverse_only_parent_lemma = reverse_only_parent_lemma

    def __eq__(self, other):
        return type(other) == PhraseletInfo and \
                self.label == other.label and \
                self.template_label == other.template_label and \
                self.parent_lemma == other.parent_lemma and \
                self.parent_derived_lemma == other.parent_derived_lemma and \
                self.parent_pos == other.parent_pos and \
                self.child_lemma == other.child_lemma and \
                self.child_derived_lemma == other.child_derived_lemma and \
                self.child_pos == other.child_pos and \
                self.created_without_matching_tags == other.created_without_matching_tags and \
                self.reverse_only_parent_lemma == other.reverse_only_parent_lemma

    def __hash__(self):
        return hash((self.label, self.template_label, self.parent_lemma, self.parent_derived_lemma,
                self.parent_pos, self.child_lemma, self.child_derived_lemma,
                self.child_pos, self.created_without_matching_tags, self.reverse_only_parent_lemma))

class ThreadsafeContainer:
    """Container for search phrases and documents that are registered and maintained on the
        manager object as opposed to being supplied with an individual query.
    """

    def __init__(self):
        self._search_phrases = []
        # Dict from document labels to IndexedDocument objects
        self._indexed_documents = {}
        self._lock = Lock()

    def remove_all_search_phrases(self):
        with self._lock:
            self._search_phrases = []

    def remove_all_search_phrases_with_label(self, label):
        with self._lock:
            self._search_phrases = [search_phrase for search_phrase in self._search_phrases if
                    search_phrase.label != label]

    def register_search_phrase(self, search_phrase):
        with self._lock:
            self._search_phrases.append(search_phrase)

    def list_search_phrase_labels(self):
        with self._lock:
            search_phrase_labels = sorted(set([search_phrase.label for search_phrase in
                    self._search_phrases]))
        return search_phrase_labels

    def register_document(self, indexed_document, label):
        with self._lock:
            if label in self._indexed_documents.keys():
                raise DuplicateDocumentError(label)
            self._indexed_documents[label] = indexed_document

    def remove_document(self, label):
        with self._lock:
            self._indexed_documents.pop(label)

    def remove_all_documents(self):
        with self._lock:
            self._indexed_documents = {}

    def document_labels(self):
        """Returns a list of the labels of the currently registered documents."""

        with self._lock:
            document_labels = self._indexed_documents.keys()
        return document_labels

    def get_document(self, label):
        with self._lock:
            if label in self._indexed_documents.keys():
                document = self._indexed_documents[label].doc
            else:
                document = None
        return document

    def get_indexed_documents(self):
        with self._lock:
            return self._indexed_documents.copy()

    def get_search_phrases(self):
        with self._lock:
            return self._search_phrases.copy()

class StructuralMatcher:
    """The class responsible for matching search phrases with documents."""

    def __init__(self, semantic_analyzer, ontology, overall_similarity_threshold,
             embedding_based_matching_on_root_words, analyze_derivational_morphology,
             perform_coreference_resolution):
        """Args:

        semantic_analyzer -- the *SemanticAnalyzer* object to use
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
        self.ontology = ontology
        self.overall_similarity_threshold = overall_similarity_threshold
        self.embedding_based_matching_on_root_words = embedding_based_matching_on_root_words
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.perform_coreference_resolution = perform_coreference_resolution
        self.populate_ontology_reverse_derivational_dict()

    def populate_ontology_reverse_derivational_dict(self):
        """During structural matching, a lemma or derived lemma matches any words in the ontology
            that yield the same word as their derived lemmas. This method generates a dictionary
            from derived lemmas to ontology words that yield them to facilitate such matching.
        """
        if self.analyze_derivational_morphology and self.ontology != None:
            ontology_reverse_derivational_dict = {}
            for ontology_word in self.ontology.words:
                derived_lemmas = []
                normalized_ontology_word = \
                        self.semantic_analyzer.normalize_hyphens(ontology_word)
                for textual_word in normalized_ontology_word.split():
                    derived_lemma = self.semantic_analyzer.derived_holmes_lemma(None,
                            textual_word.lower())
                    if derived_lemma == None:
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
            self.ontology_reverse_derivational_dict = ontology_reverse_derivational_dict
        else:
            self.ontology_reverse_derivational_dict = None

    def reverse_derived_lemmas_in_ontology(self, object):
        """ Returns all ontology entries that point to the derived lemma of a token.
        """
        if isinstance(object, Token):
            derived_lemma = object._.holmes.lemma_or_derived_lemma()
        elif isinstance(object, Subword):
            derived_lemma = object.lemma_or_derived_lemma()
        elif isinstance(object, self._MultiwordSpan):
            derived_lemma = object.derived_lemma
        else:
            raise RuntimeError(': '.join(('Unsupported type', str(type(object)))))
        derived_lemma = self.semantic_analyzer.normalize_hyphens(derived_lemma)
        if derived_lemma in self.ontology_reverse_derivational_dict:
            return self.ontology_reverse_derivational_dict[derived_lemma]
        else:
            return []

    class _SearchPhrase:

        def __init__(self, doc, matchable_tokens, root_token,
                matchable_non_entity_tokens_to_lexemes, single_token_similarity_threshold, label,
                ontology, topic_match_phraselet,
                topic_match_phraselet_created_without_matching_tags, reverse_only,
                structural_matcher):
            """Args:

            doc -- the Holmes document created for the search phrase
            matchable_tokens -- a list of tokens all of which must have counterparts in the
                document to produce a match
            root_token -- the token at which recursive matching starts
            matchable_non_entity_tokens_to_lexemes -- dictionary from token indexes to *Lexeme*
                objects. Only used when embedding matching is active.
            single_token_similarity_threshold -- the lowest similarity value that a single token
                within this search phrase could have with a matching document token to achieve
                the overall matching threshold for a match.
            label -- a label for the search phrase.
            ontology -- a reference to the ontology held by the outer *StructuralMatcher* object.
            topic_match_phraselet -- 'True' if a topic match phraselet, otherwise 'False'.
            topic_match_phraselet_created_without_matching_tags -- 'True' if a topic match
            phraselet created without matching tags (match_all_words), otherwise 'False'.
            reverse_only -- 'True' if a phraselet that should only be reverse-matched.
            structural_matcher -- the enclosing instance.
            """
            self.doc = doc
            self._matchable_token_indexes = [token.i for token in matchable_tokens]
            self._root_token_index = root_token.i
            self.matchable_non_entity_tokens_to_lexemes = matchable_non_entity_tokens_to_lexemes
            self.single_token_similarity_threshold = single_token_similarity_threshold
            self.label = label
            self.ontology = ontology
            self.topic_match_phraselet = topic_match_phraselet
            self.topic_match_phraselet_created_without_matching_tags = \
                    topic_match_phraselet_created_without_matching_tags
            self.reverse_only = reverse_only
            self.treat_as_reverse_only_during_initial_relation_matching = False # phraselets are
                # set to this value during topic matching to prevent them from being taken into
                # account during initial relation matching because the parent relation occurs too
                # frequently during the corpus. 'reverse_only' cannot be used instead because it
                # has an effect on scoring.
            self.words_matching_root_token, self.root_word_to_match_info_dict = \
                    self.get_words_matching_root_token_and_match_type_dict(structural_matcher)
            self.has_single_matchable_word = len(matchable_tokens) == 1

        @property
        def matchable_tokens(self):
            return [self.doc[index] for index in self._matchable_token_indexes]

        @property
        def root_token(self):
            return self.doc[self._root_token_index]

        def get_words_matching_root_token_and_match_type_dict(self, structural_matcher):
            """ Create list of all words that match the root token of the search phrase,
                taking any ontology into account; create a dictionary from these words
                to match types and depths.
            """

            def add_word_information(word, match_type, depth):
                if word not in list_to_return:
                    list_to_return.append(word)
                if not word in root_word_to_match_info_dict:
                    root_word_to_match_info_dict[word] = (match_type, depth)

            def add_word_information_from_ontology(word):
                for entry_word, entry_depth in \
                        structural_matcher.ontology.get_words_matching_and_depths(word):
                    add_word_information(entry_word, 'ontology', entry_depth)
                    if structural_matcher.analyze_derivational_morphology:
                        working_derived_lemma = \
                                structural_matcher.semantic_analyzer.derived_holmes_lemma(
                                None, entry_word.lower())
                        if working_derived_lemma != None:
                            add_word_information(working_derived_lemma, 'ontology', entry_depth)

            list_to_return = []
            root_word_to_match_info_dict = {}

            add_word_information(self.root_token._.holmes.lemma, 'direct', 0)
            if not self.topic_match_phraselet:
                add_word_information(self.root_token.text.lower(), 'direct', 0)
                hyphen_normalized_text = \
                        structural_matcher.semantic_analyzer.normalize_hyphens(self.root_token.text)
                if self.root_token.text != hyphen_normalized_text:
                    add_word_information(hyphen_normalized_text.lower(), 'direct', 0)
            if structural_matcher.analyze_derivational_morphology and \
                    self.root_token._.holmes.derived_lemma != None:
                add_word_information(self.root_token._.holmes.derived_lemma, 'derivation', 0)
            if structural_matcher.ontology != None and not \
                    structural_matcher._is_entity_search_phrase_token(self.root_token,
                    self.topic_match_phraselet):
                add_word_information_from_ontology(self.root_token._.holmes.lemma)
                if structural_matcher.analyze_derivational_morphology and \
                        self.root_token._.holmes.derived_lemma != None:
                    add_word_information_from_ontology(self.root_token._.holmes.derived_lemma)
                if not self.topic_match_phraselet:
                    add_word_information_from_ontology(self.root_token.text.lower())
                    if self.root_token.text != hyphen_normalized_text:
                        add_word_information_from_ontology(hyphen_normalized_text.lower())
                if structural_matcher.analyze_derivational_morphology:
                    for reverse_derived_lemma in \
                            structural_matcher.reverse_derived_lemmas_in_ontology(self.root_token):
                        add_word_information_from_ontology(reverse_derived_lemma)
            return list_to_return, root_word_to_match_info_dict

    class _IndexedDocument:
        """Args:

        doc -- the Holmes document
        words_to_token_info_dict -- a dictionary from words to tuples containing:
            - the token indexes where each word occurs in the document
            - the word representation
            - a boolean value specifying whether the index is based on derivation
        """

        def __init__(self, doc, words_to_token_info_dict):
            self.doc = doc
            self.words_to_token_info_dict = words_to_token_info_dict

    class _MultiwordSpan:

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

    def _multiword_spans_with_head_token(self, token):
        """Generator over *_MultiwordSpan* objects with *token* at their head. Dependent phrases
            are only returned for nouns because e.g. for verbs the whole sentence would be returned.
        """

        if not token.pos_ in self.semantic_analyzer.noun_pos:
            return
        pointer = token.left_edge.i
        while pointer <= token.right_edge.i:
            if token.doc[pointer].pos_ in self.semantic_analyzer.noun_pos \
                    and token.doc[pointer].dep_ in self.semantic_analyzer.noun_kernel_dep:
                working_text = ''
                working_lemma = ''
                working_derived_lemma = ''
                working_tokens = []
                inner_pointer = pointer
                while inner_pointer <= token.right_edge.i and \
                        token.doc[inner_pointer].pos_ in self.semantic_analyzer.noun_pos:
                    working_text = ' '.join((working_text, token.doc[inner_pointer].text))
                    working_lemma = ' '.join((working_lemma,
                            token.doc[inner_pointer]._.holmes.lemma))
                    if self.analyze_derivational_morphology and \
                            token.doc[inner_pointer]._.holmes.derived_lemma != None:
                        this_token_derived_lemma = token.doc[inner_pointer]._.holmes.derived_lemma
                    else:
                        # if derivational morphology analysis is switched off, the derived lemma
                        # will be identical to the lemma and will not be yielded by
                        # _loop_textual_representations().
                        this_token_derived_lemma = token.doc[inner_pointer]._.holmes.lemma
                    working_derived_lemma = ' '.join((working_derived_lemma,
                            this_token_derived_lemma))
                    working_tokens.append(token.doc[inner_pointer])
                    inner_pointer += 1
                if pointer + 1 < inner_pointer and token in working_tokens:
                    yield self._MultiwordSpan(working_text.strip(), working_lemma.strip(),
                            working_derived_lemma.strip(), working_tokens)
            pointer += 1

    def add_phraselets_to_dict(self, doc, *, phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors, match_all_words, returning_serialized_phraselets,
            ignore_relation_phraselets, include_reverse_only, stop_lemmas,
            reverse_only_parent_lemmas):
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
        returning_serialized_phraselets -- if 'True', serialized phraselets are returned.
        ignore_relation_phraselets -- if 'True', only single-word phraselets are processed.
        include_reverse_only -- whether to generate phraselets that are only reverse-matched.
            Reverse matching is used in topic matching but not in supervised document
            classification.
        stop_lemmas -- lemmas that should prevent all types of phraselet production.
        reverse_only_parent_lemmas -- lemma / part-of-speech combinations that, when present at
            the parent pole of a relation phraselet, should cause that phraselet to be
            reverse-matched.
        """

        index_to_lemmas_cache = {}
        def get_lemmas_from_index(index):
            """ Returns the lemma and the derived lemma. Phraselets form a special case where
                the derived lemma is set even if it is identical to the lemma. This is necessary
                because the lemma may be set to a different value during the lifecycle of the
                object. The property getter in the SemanticDictionary class ensures that
                derived_lemma == None is always returned where the two strings are identical.
            """
            if index in index_to_lemmas_cache:
                return index_to_lemmas_cache[index]
            token = doc[index.token_index]
            if self._is_entity_search_phrase_token(token, False):
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
            else:
                lemma = token._.holmes.lemma
                if self.analyze_derivational_morphology:
                    derived_lemma = token._.holmes.lemma_or_derived_lemma()
                else:
                    derived_lemma = lemma
            # the normal situation
            if self.ontology != None and not self.ontology.contains(lemma):
                if self.ontology.contains(token.text.lower()):
                    lemma = derived_lemma = token.text.lower()
                # ontology contains text but not lemma, so return text
            if self.ontology != None and self.analyze_derivational_morphology:
                for reverse_derived_word in self.reverse_derived_lemmas_in_ontology(token):
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
            if existing_lemma == None:
                return False
            if not existing_pos in self.semantic_analyzer.preferred_phraselet_pos and \
                    new_pos in self.semantic_analyzer.preferred_phraselet_pos:
                return True
            if existing_pos in self.semantic_analyzer.preferred_phraselet_pos and \
                    not new_pos in self.semantic_analyzer.preferred_phraselet_pos:
                return False
            return len(new_lemma) < len(existing_lemma)

        def add_new_phraselet_info(phraselet_label, phraselet_doc, phraselet_template,
                created_without_matching_tags, is_reverse_only_parent_lemma, parent_lemma,
                parent_derived_lemma, parent_pos, child_lemma, child_derived_lemma, child_pos):
            if phraselet_label not in phraselet_labels_to_phraselet_infos:
                phraselet_labels_to_phraselet_infos[phraselet_label] = \
                        PhraseletInfo(phraselet_label, phraselet_template.label, parent_lemma,
                        parent_derived_lemma, parent_pos, child_lemma, child_derived_lemma,
                        child_pos, created_without_matching_tags,
                        is_reverse_only_parent_lemma)
            else:
                existing_phraselet = phraselet_labels_to_phraselet_infos[phraselet_label]
                if lemma_replacement_indicated(existing_phraselet.parent_lemma,
                        existing_phraselet.parent_pos, parent_lemma, parent_pos):
                    existing_phraselet.parent_lemma = parent_lemma
                    existing_phraselet.parent_pos = parent_pos
                if lemma_replacement_indicated(existing_phraselet.child_lemma,
                        existing_phraselet.child_pos, child_lemma, child_pos):
                    existing_phraselet.child_lemma = child_lemma
                    existing_phraselet.child_pos = child_pos

        def process_single_word_phraselet_templates(token, subword_index, checking_tags,
                token_indexes_to_multiword_lemmas):
            for phraselet_template in (phraselet_template for phraselet_template in
                    self.semantic_analyzer.phraselet_templates if
                    phraselet_template.single_word() and (token._.holmes.is_matchable or
                    subword_index != None)): # see note below for explanation
                if not checking_tags or token.tag_ in phraselet_template.parent_tags:
                    phraselet_doc = self.semantic_analyzer.parse(
                        phraselet_template.template_sentence)
                    if token.i in token_indexes_to_multiword_lemmas and not match_all_words:
                        lemma = derived_lemma = token_indexes_to_multiword_lemmas[token.i]
                    else:
                        lemma, derived_lemma = get_lemmas_from_index(Index(token.i, subword_index))
                    if self.ontology != None and replace_with_hypernym_ancestors:
                        lemma, derived_lemma = replace_lemmas_with_most_general_ancestor(lemma,
                                derived_lemma)
                    phraselet_doc[phraselet_template.parent_index]._.holmes.lemma = lemma
                    phraselet_doc[phraselet_template.parent_index]._.holmes.derived_lemma = \
                            derived_lemma
                    phraselet_label = ''.join((phraselet_template.label, ': ', derived_lemma))
                    if derived_lemma not in stop_lemmas and derived_lemma != 'ENTITYNOUN':
                        # ENTITYNOUN has to be excluded as single word although it is still
                        # permitted as the child of a relation phraselet template
                        add_new_phraselet_info(phraselet_label, phraselet_doc, phraselet_template,
                                not checking_tags, None, lemma, derived_lemma, token.pos_, None,
                                None, None)

        def add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(index_list):
            # for each token in the list, find out whether it has subwords and if so add the
            # head subword to the list
            for index in index_list.copy():
                token = doc[index.token_index]
                for subword in (subword for subword in token._.holmes.subwords if
                        subword.is_head and subword.containing_token_index == token.i):
                    index_list.append(Index(token.i, subword.index))
            # if one or more subwords do not belong to this token, it is a hyphenated word
            # within conjunction and the whole word should not be used to build relation phraselets.
                if len ([subword for subword in token._.holmes.subwords if
                        subword.containing_token_index != token.i]) > 0:
                    index_list.remove(index)

        self._redefine_multiwords_on_head_tokens(doc)
        token_indexes_to_multiword_lemmas = {}
        token_indexes_within_multiwords_to_ignore = []
        for token in (token for token in doc if len(token._.holmes.lemma.split()) == 1):
            entity_defined_multiword, indexes = \
                    self.semantic_analyzer.get_entity_defined_multiword(token)
            if entity_defined_multiword != None:
                for index in indexes:
                    if index == token.i:
                        token_indexes_to_multiword_lemmas[token.i] = entity_defined_multiword
                    else:
                        token_indexes_within_multiwords_to_ignore.append(index)
        for token in doc:
            if token.i in token_indexes_within_multiwords_to_ignore:
                if match_all_words:
                    process_single_word_phraselet_templates(token, None, False,
                            token_indexes_to_multiword_lemmas)
                continue
            if len([subword for subword in token._.holmes.subwords if
                    subword.containing_token_index != token.i]) == 0:
                # whole single words involved in subword conjunction should not be included as
                # these are partial words including hyphens.
                process_single_word_phraselet_templates(token, None, not match_all_words,
                        token_indexes_to_multiword_lemmas)
            if match_all_words:
                for subword in (subword for subword in token._.holmes.subwords if
                        token.i == subword.containing_token_index):
                    process_single_word_phraselet_templates(token, subword.index,
                            False, token_indexes_to_multiword_lemmas)
            if ignore_relation_phraselets:
                continue
            if self.perform_coreference_resolution:
                parents = [Index(token_index, None) for token_index in
                        token._.holmes.token_and_coreference_chain_indexes]
            else:
                parents = [Index(token.i, None)]
            add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(parents)
            for parent in parents:
                for dependency in (dependency for dependency in
                        doc[parent.token_index]._.holmes.children
                        if dependency.child_index not in token_indexes_within_multiwords_to_ignore):
                    if self.perform_coreference_resolution:
                        children = [Index(token_index, None) for token_index in
                                dependency.child_token(doc)._.holmes.
                                token_and_coreference_chain_indexes]
                    else:
                        children = [Index(dependency.child_token(doc).i, None)]
                    add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(
                            children)
                    for child in children:
                        for phraselet_template in (phraselet_template for phraselet_template in
                                self.semantic_analyzer.phraselet_templates if not
                                phraselet_template.single_word() and (not
                                phraselet_template.reverse_only or include_reverse_only)):
                            if dependency.label in \
                                    phraselet_template.dependency_labels and \
                                    doc[parent.token_index].tag_ in phraselet_template.parent_tags\
                                    and doc[child.token_index].tag_ in \
                                    phraselet_template.child_tags and \
                                    doc[parent.token_index]._.holmes.is_matchable and \
                                    doc[child.token_index]._.holmes.is_matchable:
                                phraselet_doc = self.semantic_analyzer.parse(
                                        phraselet_template.template_sentence)
                                if parent.token_index in token_indexes_to_multiword_lemmas:
                                    parent_lemma = parent_derived_lemma = \
                                            token_indexes_to_multiword_lemmas[parent.token_index]
                                else:
                                    parent_lemma, parent_derived_lemma = \
                                            get_lemmas_from_index(parent)
                                if self.ontology != None and replace_with_hypernym_ancestors:
                                    parent_lemma, parent_derived_lemma = \
                                            replace_lemmas_with_most_general_ancestor(parent_lemma,
                                            parent_derived_lemma)
                                if child.token_index in token_indexes_to_multiword_lemmas:
                                    child_lemma = child_derived_lemma = \
                                            token_indexes_to_multiword_lemmas[child.token_index]
                                else:
                                    child_lemma, child_derived_lemma = get_lemmas_from_index(child)
                                if self.ontology != None and replace_with_hypernym_ancestors:
                                    child_lemma, child_derived_lemma = \
                                            replace_lemmas_with_most_general_ancestor(child_lemma,
                                            child_derived_lemma)
                                phraselet_doc[phraselet_template.parent_index]._.holmes.lemma = \
                                        parent_lemma
                                phraselet_doc[phraselet_template.parent_index]._.holmes.\
                                        derived_lemma = parent_derived_lemma
                                phraselet_doc[phraselet_template.child_index]._.holmes.lemma = \
                                        child_lemma
                                phraselet_doc[phraselet_template.child_index]._.holmes.\
                                        derived_lemma = child_derived_lemma
                                phraselet_label = ''.join((phraselet_template.label, ': ',
                                        parent_derived_lemma, '-', child_derived_lemma))
                                is_reverse_only_parent_lemma = False
                                if reverse_only_parent_lemmas != None:
                                    for entry in reverse_only_parent_lemmas:
                                        if entry[0] == doc[parent.token_index]._.holmes.lemma \
                                                and entry[1] == doc[parent.token_index].pos_:
                                            is_reverse_only_parent_lemma = True
                                if parent_lemma not in stop_lemmas and child_lemma not in \
                                        stop_lemmas and not (is_reverse_only_parent_lemma
                                        and not include_reverse_only):
                                    add_new_phraselet_info(phraselet_label, phraselet_doc,
                                            phraselet_template, match_all_words,
                                            is_reverse_only_parent_lemma,
                                            parent_lemma, parent_derived_lemma,
                                            doc[parent.token_index].pos_,
                                            child_lemma, child_derived_lemma,
                                            doc[child.token_index].pos_)

            # We do not check for matchability in order to catch pos_='X', tag_='TRUNC'. This
            # is not a problem as only a limited range of parts of speech receive subwords in
            # the first place.
            for subword in (subword for subword in token._.holmes.subwords if
                    subword.dependent_index != None):
                parent_subword_index = subword.index
                child_subword_index = subword.dependent_index
                if token._.holmes.subwords[parent_subword_index].containing_token_index != \
                        token.i and \
                        token._.holmes.subwords[child_subword_index].containing_token_index \
                        != token.i:
                    continue
                for phraselet_template in (phraselet_template for phraselet_template in
                        self.semantic_analyzer.phraselet_templates if not
                        phraselet_template.single_word() and (not
                        phraselet_template.reverse_only or include_reverse_only) and
                        subword.dependency_label in phraselet_template.dependency_labels and
                        token.tag_ in phraselet_template.parent_tags):
                    phraselet_doc = self.semantic_analyzer.parse(
                            phraselet_template.template_sentence)
                    parent_lemma, parent_derived_lemma = get_lemmas_from_index(Index(token.i,
                            parent_subword_index))
                    if self.ontology != None and replace_with_hypernym_ancestors:
                        parent_lemma, parent_derived_lemma = \
                                replace_lemmas_with_most_general_ancestor(parent_lemma,
                                parent_derived_lemma)
                    child_lemma, child_derived_lemma = get_lemmas_from_index(Index(token.i,
                            child_subword_index))
                    if self.ontology != None and replace_with_hypernym_ancestors:
                        child_lemma, child_derived_lemma = \
                                replace_lemmas_with_most_general_ancestor(child_lemma,
                                child_derived_lemma)
                    phraselet_doc[phraselet_template.parent_index]._.holmes.lemma = \
                            parent_lemma
                    phraselet_doc[phraselet_template.parent_index]._.holmes.derived_lemma = \
                            parent_derived_lemma
                    phraselet_doc[phraselet_template.child_index]._.holmes.lemma = \
                            child_lemma
                    phraselet_doc[phraselet_template.child_index]._.holmes.derived_lemma = \
                            child_derived_lemma
                    phraselet_label = ''.join((phraselet_template.label, ': ',
                            parent_derived_lemma, '-', child_derived_lemma))
                    add_new_phraselet_info(phraselet_label, phraselet_doc,
                            phraselet_template, match_all_words, False, parent_lemma,
                            parent_derived_lemma, token.pos_, child_lemma, child_derived_lemma,
                            token.pos_)
        if len(phraselet_labels_to_phraselet_infos) == 0 and not match_all_words:
            for token in doc:
                process_single_word_phraselet_templates(token, None, False,
                        token_indexes_to_multiword_lemmas)

    def create_search_phrases_from_phraselet_infos(self, phraselet_infos):
        """ Creates search phrases from phraselet info objects, returning a dictionary from
            phraselet labels to the created search phrases.
        """

        def create_phraselet_label(phraselet_info):
            if phraselet_info.child_lemma != None:
                return ''.join((phraselet_info.template_label, ': ',
                        phraselet_info.parent_derived_lemma, '-',
                        phraselet_info.child_derived_lemma))
            else:
                return ''.join((phraselet_info.template_label, ': ',
                        phraselet_info.parent_derived_lemma))

        def create_search_phrase_from_phraselet(phraselet_info):
            for phraselet_template in self.semantic_analyzer.phraselet_templates:
                if phraselet_info.template_label == phraselet_template.label:
                    phraselet_doc = self.semantic_analyzer.parse(
                            phraselet_template.template_sentence)
                    phraselet_doc[phraselet_template.parent_index]._.holmes.lemma = \
                            phraselet_info.parent_lemma
                    phraselet_doc[phraselet_template.parent_index]._.holmes.derived_lemma = \
                            phraselet_info.parent_derived_lemma
                    if phraselet_info.child_lemma != None:
                        phraselet_doc[phraselet_template.child_index]._.holmes.lemma = \
                                phraselet_info.child_lemma
                        phraselet_doc[phraselet_template.child_index]._.holmes.derived_lemma = \
                                phraselet_info.child_derived_lemma
                    return self.create_search_phrase('topic match phraselet',
                                phraselet_doc, create_phraselet_label(phraselet_info),
                                phraselet_template, phraselet_info.created_without_matching_tags,
                                phraselet_info.reverse_only_parent_lemma)
                    return
            raise RuntimeError(' '.join(('Phraselet template', phraselet_info.template_label,
                    'not found.')))

        return {create_phraselet_label(phraselet_info) :
                create_search_phrase_from_phraselet(phraselet_info) for phraselet_info in
                phraselet_infos}

    def _redefine_multiwords_on_head_tokens(self, doc):

        def loop_textual_representations(multiword_span):
            for representation, _ in self._loop_textual_representations(multiword_span):
                yield representation, multiword_span.derived_lemma
            if self.analyze_derivational_morphology:
                for reverse_derived_lemma in \
                        self.reverse_derived_lemmas_in_ontology(multiword_span):
                    yield reverse_derived_lemma, multiword_span.derived_lemma

        if self.ontology != None:
            for token in (token for token in doc if len(token._.holmes.lemma.split()) == 1):
                matched = False
                for multiword_span in self._multiword_spans_with_head_token(token):
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

    def create_search_phrase(self, search_phrase_text, search_phrase_doc,
            label, phraselet_template, topic_match_phraselet_created_without_matching_tags,
            is_reverse_only_parent_lemma = False):
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

        if phraselet_template == None:
            self._redefine_multiwords_on_head_tokens(search_phrase_doc)
            # where a multiword exists as an ontology entry, the multiword should be used for
            # matching rather than the individual words. Not relevant for topic matching
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
                    self.semantic_analyzer.is_involved_in_coreference(token):
                # SearchPhrases may not themselves contain coreferring pronouns
                # because then the matching becomes too complicated
                raise SearchPhraseContainsCoreferringPronounError(search_phrase_text)

        root_tokens = []
        tokens_to_match = []
        matchable_non_entity_tokens_to_lexemes = {}
        for token in search_phrase_doc:
            # check whether grammatical token
            if phraselet_template != None and phraselet_template.parent_index != token.i and \
                    phraselet_template.child_index != token.i:
                token._.holmes.is_matchable = False
            if phraselet_template != None and phraselet_template.parent_index == token.i and not \
                    phraselet_template.single_word() and \
                    phraselet_template.assigned_dependency_label != None:
                for dependency in (dependency for dependency in token._.holmes.children if \
                        dependency.child_index == phraselet_template.child_index):
                    dependency.label = phraselet_template.assigned_dependency_label
            if token._.holmes.is_matchable and not (len(token._.holmes.children) > 0 and
                    token._.holmes.children[0].child_index < 0):
                tokens_to_match.append(token)
                if self.overall_similarity_threshold < 1.0 and not \
                        self._is_entity_search_phrase_token(token, phraselet_template != None):
                    if phraselet_template == None and len(token._.holmes.lemma.split()) > 1:
                        matchable_non_entity_tokens_to_lexemes[token.i] = \
                                self.semantic_analyzer.nlp.vocab[token.lemma_]
                    else:
                        matchable_non_entity_tokens_to_lexemes[token.i] = \
                                self.semantic_analyzer.nlp.vocab[token._.holmes.lemma]
            if token.dep_ == 'ROOT': # syntactic root
                root_tokens.append(replace_grammatical_root_token_recursively(token))
        if len(tokens_to_match) == 0:
            raise SearchPhraseWithoutMatchableWordsError(search_phrase_text)
        if len(root_tokens) > 1:
            raise SearchPhraseContainsMultipleClausesError(search_phrase_text)
        single_token_similarity_threshold = 1.0
        if self.overall_similarity_threshold < 1.0 and \
                len(matchable_non_entity_tokens_to_lexemes) > 0:
            single_token_similarity_threshold = \
                    self.overall_similarity_threshold ** len(matchable_non_entity_tokens_to_lexemes)
        if phraselet_template == None:
            reverse_only = False
        else:
            reverse_only = is_reverse_only_parent_lemma or phraselet_template.reverse_only
        return self._SearchPhrase(search_phrase_doc, tokens_to_match,
                root_tokens[0], matchable_non_entity_tokens_to_lexemes,
                single_token_similarity_threshold, label, self.ontology,
                phraselet_template != None, topic_match_phraselet_created_without_matching_tags,
                reverse_only, self)

    def index_document(self, parsed_document):

        def add_dict_entry(dict, word, token_index, subword_index, match_type):
            index = Index(token_index, subword_index)
            if match_type == 'entity':
                key_word = word
            else:
                key_word = word.lower()
            if key_word in dict.keys():
                if index not in dict[key_word]:
                    dict[key_word].append((index, word, match_type == 'derivation'))
            else:
                dict[key_word] = [(index, word, match_type == 'derivation')]

        def get_ontology_defined_multiword(token):
            for multiword_span in self._multiword_spans_with_head_token(token):
                if self.ontology.contains_multiword(multiword_span.text):
                    return multiword_span.text, 'direct'
                hyphen_normalized_text = self.semantic_analyzer.normalize_hyphens(
                        multiword_span.text)
                if self.ontology.contains_multiword(hyphen_normalized_text):
                    return hyphen_normalized_text, 'direct'
                elif self.ontology.contains_multiword(multiword_span.lemma):
                    return multiword_span.lemma, 'direct'
                elif self.ontology.contains_multiword(multiword_span.derived_lemma):
                    return multiword_span.derived_lemma, 'derivation'
                if self.analyze_derivational_morphology and self.ontology != None:
                    for reverse_lemma in self.reverse_derived_lemmas_in_ontology(
                            multiword_span):
                        return reverse_lemma, 'derivation'
            return None, None

        words_to_token_info_dict = {}
        for token in parsed_document:

            # parent check is necessary so we only find multiword entities once per
            # search phrase. sibling_marker_deps applies to siblings which would
            # otherwise be excluded because the main sibling would normally also match the
            # entity root word.
            if len(token.ent_type_) > 0 and (token.dep_ == 'ROOT' or
                    token.dep_ in self.semantic_analyzer.sibling_marker_deps
                    or token.ent_type_ != token.head.ent_type_):
                entity_label = ''.join(('ENTITY', token.ent_type_))
                add_dict_entry(words_to_token_info_dict, entity_label, token.i, None, 'entity')
            if self.ontology != None:
                ontology_defined_multiword, match_type = get_ontology_defined_multiword(token)
                if ontology_defined_multiword != None:
                    add_dict_entry(words_to_token_info_dict, ontology_defined_multiword,
                            token.i, None, match_type)
                    continue
            entity_defined_multiword, _ = self.semantic_analyzer.get_entity_defined_multiword(token)
            if entity_defined_multiword != None:
                add_dict_entry(words_to_token_info_dict, entity_defined_multiword,
                        token.i, None, 'direct')
            for representation, match_type in self._loop_textual_representations(token):
                add_dict_entry(words_to_token_info_dict, representation, token.i, None,
                        match_type)
            for subword in token._.holmes.subwords:
                for representation, match_type in self._loop_textual_representations(subword):
                    add_dict_entry(words_to_token_info_dict, representation, token.i,
                            subword.index, match_type)
        return self._IndexedDocument(parsed_document, words_to_token_info_dict)

    def _match_type(self, search_phrase_and_document_derived_lemmas_identical, *match_types):
        if 'ontology' in match_types and search_phrase_and_document_derived_lemmas_identical:
            # an ontology entry happens to have created a derivation word match before the
            # derivation match itself was processed, so mark the type as 'derivation'.
            return 'derivation'
        elif 'ontology' in match_types:
            return 'ontology'
        elif 'derivation' in match_types:
            return 'derivation'
        else:
            return 'direct'

    def _match_recursively(self, *, search_phrase, search_phrase_token, document, document_token,
        document_subword_index, search_phrase_tokens_to_word_matches,
        search_phrase_and_document_visited_table, is_uncertain,
        structurally_matched_document_token, compare_embeddings_on_non_root_words):
        """Called whenever matching is attempted between a search phrase token and a document
            token."""

        def handle_match(search_phrase_word, document_word, match_type, depth,
                *, similarity_measure=1.0, first_document_token=document_token,
                last_document_token=document_token):
            """Most of the variables are set from the outer call.

            Args:

            search_phrase_word -- the textual representation of the search phrase word that matched.
            document_word -- the textual representation of the document word that matched.
            match_type -- *direct*, *entity*, *embedding*, *ontology* or *derivation*
            similarity_measure -- the similarity between the two tokens. Defaults to 1.0 if the
                match did not involve embeddings.
            """
            for dependency in (dependency for
                    dependency in search_phrase_token._.holmes.children
                    if dependency.child_token(search_phrase_token.doc)._.holmes.is_matchable):
                at_least_one_document_dependency_tried = False
                at_least_one_document_dependency_matched = False
                # Loop through this token and any tokens linked to it by coreference
                if self.perform_coreference_resolution and document_subword_index == None:
                    parents = [Index(token_index, None) for token_index in
                            document_token._.holmes.token_and_coreference_chain_indexes]
                else:
                    parents = [Index(document_token.i, document_subword_index)]
                for working_document_parent_index in parents:
                    working_document_child_indexes = []
                    document_parent_token = document_token.doc[
                            working_document_parent_index.token_index]
                    if not working_document_parent_index.is_subword() or \
                            document_parent_token._.holmes.subwords[
                            working_document_parent_index.subword_index].is_head:
                            # is_head: e.g. 'Polizeiinformation über Kriminelle' should match
                            # 'Information über Kriminelle'
                        for document_dependency in (document_dependency for document_dependency in
                                document_parent_token._.holmes.children if
                                self.semantic_analyzer.dependency_labels_match(
                                search_phrase_dependency_label= dependency.label,
                                document_dependency_label = document_dependency.label)):
                            document_child = document_dependency.child_token(document_token.doc)
                            if self.perform_coreference_resolution:
                                # wherever a dependency is found, loop through any tokens linked
                                # to the child by coreference
                                working_document_child_indexes = [Index(token_index, None) for
                                        token_index in
                                        document_child._.holmes.token_and_coreference_chain_indexes
                                        if document_token.doc[token_index].pos_ != 'PRON' or not
                                        self.semantic_analyzer.is_involved_in_coreference(
                                        document_token.doc[token_index])]
                                        # otherwise where matching starts with a noun and there is
                                        # a dependency pointing back to the noun, matching will be
                                        # attempted against the pronoun only and will then fail.
                            else:
                                working_document_child_indexes = \
                                        [Index(document_dependency.child_index, None)]
                            # Where a dependency points to an entire word that has subwords, check
                            # the head subword as well as the entire word
                            for working_document_child_index in \
                                    working_document_child_indexes.copy():
                                working_document_child = \
                                        document_token.doc[working_document_child_index.token_index]
                                for subword in (subword for subword in
                                        working_document_child._.holmes.subwords if
                                        subword.is_head):
                                    working_document_child_indexes.append(Index(
                                            working_document_child.i,
                                            subword.index))
                            # Loop through the dependencies from each token
                            for working_document_child_index in (working_index for working_index
                                    in working_document_child_indexes if working_index not in
                                    search_phrase_and_document_visited_table[dependency.child_index]
                                    ):
                                at_least_one_document_dependency_tried = True
                                if self._match_recursively(
                                        search_phrase=search_phrase,
                                        search_phrase_token=dependency.child_token(
                                                search_phrase_token.doc),
                                        document=document,
                                        document_token= document[
                                        working_document_child_index.token_index],
                                        document_subword_index =
                                        working_document_child_index.subword_index,
                                        search_phrase_tokens_to_word_matches=
                                                search_phrase_tokens_to_word_matches,
                                        search_phrase_and_document_visited_table=
                                                search_phrase_and_document_visited_table,
                                        is_uncertain=(document_dependency.is_uncertain and not
                                                dependency.is_uncertain),
                                        structurally_matched_document_token=document_child,
                                        compare_embeddings_on_non_root_words=
                                        compare_embeddings_on_non_root_words):
                                    at_least_one_document_dependency_matched = True
                    if working_document_parent_index.is_subword():
                        # examine relationship to dependent subword in the same word
                        document_parent_subword = document_token.doc[
                                working_document_parent_index.token_index]._.holmes.\
                                subwords[working_document_parent_index.subword_index]
                        if document_parent_subword.dependent_index != None and \
                                self.semantic_analyzer.dependency_labels_match(
                                search_phrase_dependency_label= dependency.label,
                                document_dependency_label =
                                document_parent_subword.dependency_label):
                            at_least_one_document_dependency_tried = True
                            if self._match_recursively(
                                    search_phrase=search_phrase,
                                    search_phrase_token=dependency.child_token(
                                            search_phrase_token.doc),
                                    document=document,
                                    document_token= document_token,
                                    document_subword_index =
                                            document_parent_subword.dependent_index,
                                    search_phrase_tokens_to_word_matches=
                                            search_phrase_tokens_to_word_matches,
                                    search_phrase_and_document_visited_table=
                                            search_phrase_and_document_visited_table,
                                    is_uncertain=False,
                                    structurally_matched_document_token=document_token,
                                    compare_embeddings_on_non_root_words=
                                    compare_embeddings_on_non_root_words):
                                at_least_one_document_dependency_matched = True
                if at_least_one_document_dependency_tried and not \
                        at_least_one_document_dependency_matched:
                        # it is already clear that the search phrase has not matched, so
                        # there is no point in pursuing things any further
                    return
            # store the word match
            if document_subword_index == None:
                document_subword = None
            else:
                document_subword = document_token._.holmes.subwords[document_subword_index]
            search_phrase_tokens_to_word_matches[search_phrase_token.i].append(WordMatch(
                    search_phrase_token, search_phrase_word, document_token,
                    first_document_token, last_document_token, document_subword,
                    document_word, match_type, similarity_measure, is_negated, is_uncertain,
                    structurally_matched_document_token, document_word, depth))

        def loop_search_phrase_word_representations():
            yield search_phrase_token._.holmes.lemma, 'direct', \
                    search_phrase_token._.holmes.lemma_or_derived_lemma()
            hyphen_normalized_word = self.semantic_analyzer.normalize_hyphens(
                    search_phrase_token._.holmes.lemma)
            if hyphen_normalized_word != search_phrase_token._.holmes.lemma:
                yield hyphen_normalized_word, 'direct', \
                        search_phrase_token._.holmes.lemma_or_derived_lemma()
            if self.analyze_derivational_morphology and \
                    search_phrase_token._.holmes.derived_lemma != None:
                yield search_phrase_token._.holmes.derived_lemma, 'derivation', \
                        search_phrase_token._.holmes.lemma_or_derived_lemma()
            if not search_phrase.topic_match_phraselet and \
                    search_phrase_token._.holmes.lemma == search_phrase_token.lemma_ and \
                    search_phrase_token._.holmes.lemma != search_phrase_token.text:
                # search phrase word is not multiword, phrasal or separable verb, so we can match
                # against its text as well as its lemma
                yield search_phrase_token.text, 'direct', \
                        search_phrase_token._.holmes.lemma_or_derived_lemma()
            if self.analyze_derivational_morphology and self.ontology != None:
                for reverse_lemma in self.reverse_derived_lemmas_in_ontology(
                        search_phrase_token):
                    yield reverse_lemma, 'ontology', \
                            search_phrase_token._.holmes.lemma_or_derived_lemma()

        def document_word_representations():
            list_to_return = []
            if document_subword_index != None:
                working_document_subword = document_token._.holmes.subwords[document_subword_index]
                list_to_return.append((working_document_subword.text, 'direct',
                        working_document_subword.lemma_or_derived_lemma()))
                hyphen_normalized_word = self.semantic_analyzer.normalize_hyphens(
                        working_document_subword.text)
                if hyphen_normalized_word != working_document_subword.text:
                    list_to_return.append((hyphen_normalized_word, 'direct',
                            working_document_subword.lemma_or_derived_lemma()))
                if working_document_subword.lemma != working_document_subword.text:
                    list_to_return.append((working_document_subword.lemma, 'direct',
                            working_document_subword.lemma_or_derived_lemma()))
                if self.analyze_derivational_morphology and \
                        working_document_subword.derived_lemma != None:
                    list_to_return.append((working_document_subword.derived_lemma,
                            'derivation', working_document_subword.lemma_or_derived_lemma()))
                if self.analyze_derivational_morphology and self.ontology != None:
                    for reverse_lemma in self.reverse_derived_lemmas_in_ontology(
                            working_document_subword):
                        list_to_return.append((reverse_lemma, 'ontology',
                                working_document_subword.lemma_or_derived_lemma()))
            else:
                list_to_return.append((document_token.text, 'direct',
                        document_token._.holmes.lemma_or_derived_lemma()))
                hyphen_normalized_word = self.semantic_analyzer.normalize_hyphens(
                        document_token.text)
                if hyphen_normalized_word != document_token.text:
                    list_to_return.append((hyphen_normalized_word, 'direct',
                            document_token._.holmes.lemma_or_derived_lemma()))
                if document_token._.holmes.lemma != document_token.text:
                        list_to_return.append((document_token._.holmes.lemma, 'direct',
                                document_token._.holmes.lemma_or_derived_lemma()))
                if self.analyze_derivational_morphology:
                    if document_token._.holmes.derived_lemma != None:
                        list_to_return.append((document_token._.holmes.derived_lemma,
                                'derivation', document_token._.holmes.lemma_or_derived_lemma()))
                if self.analyze_derivational_morphology and self.ontology != None:
                    for reverse_lemma in self.reverse_derived_lemmas_in_ontology(document_token):
                        list_to_return.append((reverse_lemma, 'ontology',
                                document_token._.holmes.lemma_or_derived_lemma()))
            return list_to_return

        def loop_document_multiword_representations(multiword_span):
            yield multiword_span.text, 'direct', multiword_span.derived_lemma
            hyphen_normalized_word = self.semantic_analyzer.normalize_hyphens(multiword_span.text)
            if hyphen_normalized_word != multiword_span.text:
                yield hyphen_normalized_word, 'direct', multiword_span.derived_lemma
            if multiword_span.text != multiword_span.lemma:
                yield multiword_span.lemma, 'direct', multiword_span.derived_lemma
            if multiword_span.derived_lemma != multiword_span.lemma:
                yield multiword_span.derived_lemma, 'derivation', multiword_span.derived_lemma
            if self.analyze_derivational_morphology and self.ontology != None:
                for reverse_lemma in self.reverse_derived_lemmas_in_ontology(multiword_span):
                    yield reverse_lemma, 'ontology', multiword_span.derived_lemma

        index = Index(document_token.i, document_subword_index)
        search_phrase_and_document_visited_table[search_phrase_token.i].add(index)
        is_negated = document_token._.holmes.is_negated
        if document_token._.holmes.is_uncertain:
            is_uncertain = True

        if self._is_entity_search_phrase_token(search_phrase_token,
                search_phrase.topic_match_phraselet) and document_subword_index == None:
            if self._entity_search_phrase_token_matches(search_phrase_token,
                    search_phrase.topic_match_phraselet, document_token):
                for multiword_span in self._multiword_spans_with_head_token(document_token):
                    for working_token in multiword_span.tokens:
                        if not self._entity_search_phrase_token_matches(
                                search_phrase_token, search_phrase.topic_match_phraselet,
                                document_token):
                            continue
                    for working_token in multiword_span.tokens:
                        search_phrase_and_document_visited_table[search_phrase_token.i].add(
                                working_token.i)
                    handle_match(search_phrase_token.text, multiword_span.text, 'entity', 0,
                            first_document_token=multiword_span.tokens[0],
                            last_document_token=multiword_span.tokens[-1])
                    return True
                search_phrase_and_document_visited_table[search_phrase_token.i].add(
                        document_token.i)
                handle_match(search_phrase_token.text, document_token.text, 'entity', 0)
                return True
            return False

        document_word_representations = document_word_representations()
        for search_phrase_word_representation, search_phrase_match_type, \
                search_phrase_derived_lemma in loop_search_phrase_word_representations():
            # multiword matches
            if document_subword_index == None:
                for multiword_span in self._multiword_spans_with_head_token(document_token):
                    for multiword_span_representation, document_match_type, \
                            multispan_derived_lemma in \
                            loop_document_multiword_representations(multiword_span):
                        if search_phrase_word_representation.lower() == \
                                multiword_span_representation.lower():
                            for working_token in multiword_span.tokens:
                                search_phrase_and_document_visited_table[search_phrase_token.i].add(
                                        working_token.i)
                            handle_match(search_phrase_token._.holmes.lemma,
                                    multiword_span_representation,
                                    self._match_type(search_phrase_derived_lemma ==
                                    multispan_derived_lemma, search_phrase_match_type,
                                    document_match_type),
                                    0, first_document_token=multiword_span.tokens[0],
                                    last_document_token=multiword_span.tokens[-1])
                            return True
                        if self.ontology != None:
                            entry = self.ontology.matches(
                                    search_phrase_word_representation.lower(),
                                    multiword_span_representation.lower())
                            if entry != None:
                                for working_token in multiword_span.tokens:
                                    search_phrase_and_document_visited_table[
                                            search_phrase_token.i].add(working_token.i)
                                handle_match(search_phrase_word_representation, entry.word,
                                        'ontology', entry.depth,
                                        first_document_token=multiword_span.tokens[0],
                                        last_document_token=multiword_span.tokens[-1])
                                return True
            for document_word_representation, document_match_type, document_derived_lemma in \
                    document_word_representations:
                if search_phrase_word_representation.lower() == \
                        document_word_representation.lower():
                    handle_match(search_phrase_word_representation, document_word_representation,
                            self._match_type(search_phrase_derived_lemma == document_derived_lemma,
                            search_phrase_match_type, document_match_type), 0)
                    return True
                if self.ontology != None:
                    entry = self.ontology.matches(search_phrase_word_representation.lower(),
                            document_word_representation.lower())
                    if entry != None:
                        handle_match(search_phrase_word_representation, entry.word, 'ontology',
                                entry.depth)
                        return True

        if self.overall_similarity_threshold < 1.0 and (compare_embeddings_on_non_root_words or
                search_phrase.root_token.i == search_phrase_token.i) and search_phrase_token.i in \
                search_phrase.matchable_non_entity_tokens_to_lexemes.keys() and \
                self.semantic_analyzer.embedding_matching_permitted(search_phrase_token):
            search_phrase_lexeme = \
                    search_phrase.matchable_non_entity_tokens_to_lexemes[search_phrase_token.i]
            if document_subword_index != None:
                if not self.semantic_analyzer.embedding_matching_permitted(
                        document_token._.holmes.subwords[document_subword_index]):
                    return False
                document_lemma = document_token._.holmes.subwords[document_subword_index].lemma
            else:
                if not self.semantic_analyzer.embedding_matching_permitted(document_token):
                    return False
                if len(document_token._.holmes.lemma.split()) > 1:
                    document_lemma = document_token.lemma_
                else:
                    document_lemma = document_token._.holmes.lemma

            document_lexeme = self.semantic_analyzer.nlp.vocab[document_lemma]
            if search_phrase_lexeme.vector_norm > 0 and document_lexeme.vector_norm > 0:
                similarity_measure = search_phrase_lexeme.similarity(document_lexeme)
                if similarity_measure > search_phrase.single_token_similarity_threshold:
                    if not search_phrase.topic_match_phraselet and \
                            len(search_phrase_token._.holmes.lemma.split()) > 1:
                        search_phrase_word_to_use = search_phrase_token.lemma_
                    else:
                        search_phrase_word_to_use = search_phrase_token._.holmes.lemma
                    handle_match(search_phrase_word_to_use, document_token.lemma_, 'embedding', 0,
                            similarity_measure = similarity_measure)
                    return True
        return False

    def _is_entity_search_phrase_token(self, search_phrase_token,
            examine_lemma_rather_than_text):
        if examine_lemma_rather_than_text:
            word_to_check = search_phrase_token._.holmes.lemma
        else:
            word_to_check = search_phrase_token.text
        return word_to_check[:6] == 'ENTITY' and len(word_to_check) > 6

    def _is_entitynoun_search_phrase_token(self, search_phrase_token,
            examine_lemma_rather_than_text):
        if examine_lemma_rather_than_text:
            word_to_check = search_phrase_token._.holmes.lemma
        else:
            word_to_check = search_phrase_token.text
        return word_to_check == 'ENTITYNOUN'

    def _entity_search_phrase_token_matches(self, search_phrase_token, topic_match_phraselet,
            document_token):
        if topic_match_phraselet:
            word_to_check = search_phrase_token._.holmes.lemma
        else:
            word_to_check = search_phrase_token.text
        return (document_token.ent_type_ == word_to_check[6:] and
                len(document_token._.holmes.lemma.strip()) > 0) or \
                (word_to_check == 'ENTITYNOUN' and
                document_token.pos_ in self.semantic_analyzer.noun_pos)
                # len(document_token._.holmes.lemma.strip()) > 0: in German spaCy sometimes
                # classifies whitespace as entities.

    def _loop_textual_representations(self, object):
        if isinstance(object, Token):
            yield object.text, 'direct'
            hyphen_normalized_text = self.semantic_analyzer.normalize_hyphens(object.text)
            if hyphen_normalized_text != object.text:
                yield hyphen_normalized_text, 'direct'
            if object._.holmes.lemma != object.text:
                yield object._.holmes.lemma, 'direct'
            if self.analyze_derivational_morphology and object._.holmes.derived_lemma != None:
                yield object._.holmes.derived_lemma, 'derivation'
        elif isinstance(object, Subword):
            yield object.text, 'direct'
            hyphen_normalized_text = self.semantic_analyzer.normalize_hyphens(object.text)
            if hyphen_normalized_text != object.text:
                yield hyphen_normalized_text, 'direct'
            if object.text != object.lemma:
                yield object.lemma, 'direct'
            if self.analyze_derivational_morphology and object.derived_lemma != None:
                yield object.derived_lemma, 'derivation'
        elif isinstance(object, self._MultiwordSpan):
            yield object.text, 'direct'
            hyphen_normalized_text = self.semantic_analyzer.normalize_hyphens(object.text)
            if hyphen_normalized_text != object.text:
                yield hyphen_normalized_text, 'direct'
            if object.text != object.lemma:
                yield object.lemma, 'direct'
            if object.lemma != object.derived_lemma:
                yield object.derived_lemma, 'derivation'
        else:
            raise RuntimeError(': '.join(('Unsupported type', str(type(object)))))

    def _build_matches(self, *, search_phrase, document, search_phrase_tokens_to_word_matches,
            document_label):
        """Investigate possible matches when recursion is complete."""

        def mention_root_or_token_index(token):
            mri = token._.holmes.mention_root_index
            if mri != None:
                return mri
            else:
                return token.i

        def filter_word_matches_based_on_coreference_resolution(word_matches):
            """ When coreference resolution is active, additional matches are sometimes
                returned that are filtered out again using this method. The use of
                mention_root_index means that only the first cluster is taken into account.
            """
            dict = {}
            # Find the structurally matching document tokens for this list of word matches
            for word_match in word_matches:
                structural_index = \
                        mention_root_or_token_index(word_match.structurally_matched_document_token)
                if structural_index in dict.keys():
                    dict[structural_index].append(word_match)
                else:
                    dict[structural_index] = [word_match]
            new_word_matches = []
            for structural_index in dict.keys():
                # For each structural token, find the best matching coreference mention
                relevant_word_matches = dict[structural_index]
                structurally_matched_document_token = \
                        relevant_word_matches[0].document_token.doc[structural_index]
                already_added_document_token_indexes = set()
                if self.semantic_analyzer.is_involved_in_coreference(
                        structurally_matched_document_token):
                    working_index = -1
                    for relevant_word_match in relevant_word_matches:
                        this_index = mention_root_or_token_index(relevant_word_match.document_token)
                        # The best mention should be as close to the structural
                        # index as possible; if they are the same distance, the preceding mention
                        # wins.
                        if working_index == -1 or \
                                (abs(structural_index - this_index) <
                                abs(structural_index - working_index)) or \
                                ((abs(structural_index - this_index) ==
                                abs(structural_index - working_index)) and
                                this_index < working_index):
                            working_index = this_index
                    # Filter out any matches from mentions other than the best mention
                    for relevant_word_match in relevant_word_matches:
                        if working_index == \
                                mention_root_or_token_index(relevant_word_match.document_token) \
                                and relevant_word_match.document_token.i not in \
                                already_added_document_token_indexes:
                                    already_added_document_token_indexes.add(
                                            relevant_word_match.document_token.i)
                                    new_word_matches.append(relevant_word_match)
                else:
                    new_word_matches.extend(relevant_word_matches)
            return new_word_matches

        def revise_extracted_words_based_on_coreference_resolution(word_matches):
            """ When coreference resolution and ontology-based matching are both active,
                there may be a more specific piece of information elsewhere in the coreference
                chain of a token that has been matched, in which case this piece of information
                should be recorded in *word_match.extracted_word*.

                If and when subwords and coreference resolution are analyzed together (at present
                they subwords are available only for German, coreference resolution only for
                English), this method will need to be updated to handle this.
            """


            for word_match in (word_match for word_match in word_matches if word_match.type
                    in ('direct', 'derivation', 'ontology')):
                working_entries = []
                # First loop through getting ontology entries for all mentions in the cluster
                for search_phrase_representation, _ in \
                        self._loop_textual_representations(word_match.search_phrase_token):
                    for mention in word_match.document_token._.holmes.mentions:
                        mention_root_token = word_match.document_token.doc[mention.root_index]
                        for mention_representation, _ in \
                                self._loop_textual_representations(mention_root_token):
                            working_entries.append(
                                    self.ontology.matches(
                                    search_phrase_representation, mention_representation))
                        for multiword_span in \
                                self._multiword_spans_with_head_token(mention_root_token):
                            for multiword_representation, _ in \
                                    self._loop_textual_representations(multiword_span):
                                working_entries.append(
                                        self.ontology.matches(
                                        search_phrase_representation, multiword_representation))

                # Now loop through the ontology entries to see if any are more specific than
                # the current value of *extracted_word*.
                for working_entry in working_entries:
                    if working_entry == None:
                        continue
                    if working_entry.is_individual:
                        word_match.extracted_word = working_entry.word
                        break
                    elif working_entry.depth > word_match.depth:
                        word_match.extracted_word = working_entry.word
            return word_matches

        def match_already_contains_structurally_matched_document_token(match, document_token,
                document_subword_index):
            """Ensure that the same document token or subword does not match multiple search phrase
                tokens.
            """
            for word_match in match.word_matches:
                if document_token.i == word_match.structurally_matched_document_token.i:
                    if word_match.document_subword != None and document_subword_index == \
                            word_match.document_subword.index:
                        return True
                    if word_match.document_subword == None and document_subword_index == None:
                        return True
            return False

        def check_document_tokens_are_linked_by_dependency(parent_token, parent_subword,
                child_token, child_subword):
            """ The recursive nature of the main matching algorithm can mean that all the tokens
                in the search phrase have matched but that two of them are linked by a dependency
                that is absent from the document, which invalidates the match.
            """
            if parent_subword != None:
                if child_subword != None and parent_subword.dependent_index == \
                        child_subword.index and parent_token.i == child_token.i:
                    return True
                elif parent_subword.is_head and (child_subword == None or (
                        child_subword.is_head and parent_subword.containing_token_index !=
                        child_subword.containing_token_index)):
                    return True
                else:
                    return False
            if child_subword != None and not child_subword.is_head:
                return False
            if self.perform_coreference_resolution and parent_subword == None:
                parents = parent_token._.holmes.token_and_coreference_chain_indexes
                children = child_token._.holmes.token_and_coreference_chain_indexes
            else:
                parents = [parent_token.i]
                children = [child_token.i]
            for parent in parents:
                for child in children:
                    if parent_token.doc[parent]._.holmes.\
                            has_dependency_with_child_index(child):
                        return True
            return False

        def match_with_subwords_involves_all_containing_document_tokens(word_matches):
            """ Where a match involves subwords and the subwords are involved in conjunction,
                we need to make sure there are no tokens involved in the match merely because they
                supply subwords to another token, as this would lead to double matching. An example
                is search phrase 'Extraktion der Information' and document
                'Informationsextraktionsüberlegungen und -probleme'.
            """
            token_indexes = []
            containing_subword_token_indexes = []
            for word_match in word_matches:
                if word_match.document_subword != None:
                    token_indexes.append(word_match.document_token.i)
                    containing_subword_token_indexes.append(
                            word_match.document_subword.containing_token_index)
            return len([token_index for token_index in token_indexes if not token_index in
                    containing_subword_token_indexes]) == 0

        matches = [Match(search_phrase.label, document_label,
                search_phrase.topic_match_phraselet and search_phrase.has_single_matchable_word,
                search_phrase.topic_match_phraselet_created_without_matching_tags,
                search_phrase.reverse_only)]
        for search_phrase_token in search_phrase.matchable_tokens:
            word_matches = search_phrase_tokens_to_word_matches[search_phrase_token.i]
            if len(word_matches) == 0:
                # if there is any search phrase token without a matching document token,
                # we have no match and can return
                return []
            if self.perform_coreference_resolution:
                word_matches = filter_word_matches_based_on_coreference_resolution(word_matches)
                if self.ontology != None:
                    word_matches = revise_extracted_words_based_on_coreference_resolution(
                            word_matches)
            # handle any conjunction by distributing the matches amongst separate match objects
            working_matches = []
            for word_match in word_matches:
                for match in matches:
                    working_match = copy.copy(match)
                    if word_match.document_subword == None:
                        subword_index = None
                    else:
                        subword_index = word_match.document_subword.index
                    if not match_already_contains_structurally_matched_document_token(working_match,
                            word_match.structurally_matched_document_token, subword_index):
                        working_match.word_matches.append(word_match)
                        if word_match.is_negated:
                            working_match.is_negated = True
                        if word_match.is_uncertain:
                            working_match.is_uncertain = True
                        if search_phrase_token.i == search_phrase.root_token.i:
                            working_match.index_within_document = word_match.document_token.i
                        working_matches.append(working_match)
            matches = working_matches

        matches_to_return = []
        for match in matches:
            failed = False
            not_normalized_overall_similarity_measure = 1.0
            # now carry out the coherence check, if there are two or fewer word matches (which
            # is the case during topic matching) no check is necessary
            if len(match.word_matches) > 2:
                for parent_word_match in match.word_matches:
                    for search_phrase_dependency in \
                            parent_word_match.search_phrase_token._.holmes.children:
                        for child_word_match in (cwm for cwm in match.word_matches if cwm.
                                search_phrase_token.i == search_phrase_dependency.child_index):
                            if not check_document_tokens_are_linked_by_dependency(
                                    parent_word_match.document_token,
                                    parent_word_match.document_subword,
                                    child_word_match.document_token,
                                    child_word_match.document_subword):
                                failed=True
                        if failed:
                            break
                    if failed:
                        break
                if failed:
                    continue

            if not match_with_subwords_involves_all_containing_document_tokens(match.word_matches):
                continue

            for word_match in match.word_matches:
                not_normalized_overall_similarity_measure *= word_match.similarity_measure
            if not_normalized_overall_similarity_measure < 1.0:
                overall_similarity_measure = \
                        round(not_normalized_overall_similarity_measure ** \
                        (1 / len(search_phrase.matchable_non_entity_tokens_to_lexemes)), 8)
            else:
                overall_similarity_measure = 1.0
            if overall_similarity_measure == 1.0 or \
                    overall_similarity_measure >= self.overall_similarity_threshold:
                match.overall_similarity_measure = str(
                    overall_similarity_measure)
                matches_to_return.append(match)
        return matches_to_return

    def _get_matches_starting_at_root_word_match(self, search_phrase, document,
            document_token, document_subword_index, document_label,
            compare_embeddings_on_non_root_words):
        """Begin recursive matching where a search phrase root token has matched a document
            token.
        """

        matches_to_return = []
        # array of arrays where each entry corresponds to a search_phrase token and is itself an
        # array of WordMatch instances
        search_phrase_tokens_to_word_matches = [[] for token in search_phrase.doc]
        # array of sets to guard against endless looping during recursion. Each set
        # corresponds to the search phrase token with its index and contains the Index objects
        # for the document words for which a match to that search phrase token has been attempted.
        search_phrase_and_document_visited_table = [set() for token in search_phrase.doc]
        self._match_recursively(
                search_phrase=search_phrase,
                search_phrase_token=search_phrase.root_token,
                document=document,
                document_token=document_token,
                document_subword_index = document_subword_index,
                search_phrase_tokens_to_word_matches=search_phrase_tokens_to_word_matches,
                search_phrase_and_document_visited_table=search_phrase_and_document_visited_table,
                is_uncertain=document_token._.holmes.is_uncertain,
                structurally_matched_document_token=document_token,
                compare_embeddings_on_non_root_words=compare_embeddings_on_non_root_words)
        working_matches = self._build_matches(
                search_phrase=search_phrase,
                document=document,
                search_phrase_tokens_to_word_matches=search_phrase_tokens_to_word_matches,
                document_label=document_label)
        matches_to_return.extend(working_matches)
        return matches_to_return

    def match(self, *, indexed_documents, search_phrases,
            output_document_matching_message_to_console,
            match_depending_on_single_words,
            compare_embeddings_on_root_words,
            compare_embeddings_on_non_root_words,
            document_labels_to_indexes_for_reverse_matching_sets,
            document_labels_to_indexes_for_embedding_reverse_matching_sets,
            document_label_filter = None):
        """Finds and returns matches between search phrases and documents.
        match_depending_on_single_words -- 'True' to match only single word search phrases,
            'False' to match only non-single-word search phrases and 'None' to match both.
        compare_embeddings_on_root_words -- if 'True', embeddings on root words are compared
            even if embedding_based_matching_on_root_words==False as long as
            overall_similarity_threshold < 1.0.
        compare_embeddings_on_non_root_words -- if 'False', embeddings on non-root words are not
            compared even if overall_similarity_threshold < 1.0.
        document_labels_to_indexes_for_reverse_matching_sets -- indexes for non-embedding
            reverse matching only.
        document_labels_to_indexes_for_embedding_reverse_matching_sets -- indexes for embedding
            and non-embedding reverse matching.
        document_label_filter -- a string with which the label of a document must begin for that
            document to be considered for matching, or 'None' if no filter is in use.
        """

        def get_indexes_to_consider(dictionary, document_label):
            if dictionary == None or document_label not in dictionary:
                return set()
            else:
                return dictionary[document_label]

        if self.embedding_based_matching_on_root_words:
            compare_embeddings_on_root_words=True
        if self.overall_similarity_threshold==1.0:
            compare_embeddings_on_root_words=False
            compare_embeddings_on_non_root_words=False
        match_specific_indexes = document_labels_to_indexes_for_reverse_matching_sets != None or \
                document_labels_to_indexes_for_embedding_reverse_matching_sets != None

        if len(indexed_documents) == 0:
            raise NoSearchedDocumentError(
                'At least one searched document is required to match.')
        if len(search_phrases) == 0:
            raise NoSearchPhraseError('At least one search_phrase is required to match.')
        matches = []
        for document_label, registered_document in indexed_documents.items():
            if document_label_filter != None and document_label != None and not \
                    document_label.startswith(str(document_label_filter)):
                continue
            if output_document_matching_message_to_console:
                print('Processing document', document_label)
            doc = registered_document.doc
            # Dictionary used to improve performance when embedding-based matching for root tokens
            # is active and there are multiple search phrases with the same root token word: the
            # same indexes in the document will then match all the search phrase root tokens.
            root_lexeme_to_indexes_to_match_dict = {}
            if match_specific_indexes:
                reverse_matching_indexes = get_indexes_to_consider(
                        document_labels_to_indexes_for_reverse_matching_sets, document_label)
                embedding_reverse_matching_indexes = get_indexes_to_consider(
                        document_labels_to_indexes_for_embedding_reverse_matching_sets,
                        document_label)

            for search_phrase in search_phrases:
                if not search_phrase.has_single_matchable_word and match_depending_on_single_words:
                    continue
                if search_phrase.has_single_matchable_word and \
                        match_depending_on_single_words == False:
                    continue
                if not match_specific_indexes and (search_phrase.reverse_only or \
                        search_phrase.treat_as_reverse_only_during_initial_relation_matching):
                    continue
                if search_phrase.has_single_matchable_word and \
                        not compare_embeddings_on_root_words and \
                        not self._is_entity_search_phrase_token(search_phrase.root_token,
                        search_phrase.topic_match_phraselet):
                    # We are only matching a single word without embedding, so to improve
                    # performance we avoid entering the subgraph matching code.
                    search_phrase_token = [token for token in search_phrase.doc if
                            token._.holmes.is_matchable][0]
                    existing_minimal_match_indexes = []
                    for word_matching_root_token in search_phrase.words_matching_root_token:
                        if word_matching_root_token in \
                                registered_document.words_to_token_info_dict.keys():
                            search_phrase_match_type, depth = \
                                    search_phrase.root_word_to_match_info_dict[
                                    word_matching_root_token]
                            for index, document_word_representation, \
                                    document_match_type_is_derivation in \
                                    registered_document.words_to_token_info_dict[
                                    word_matching_root_token]:
                                if index in existing_minimal_match_indexes:
                                    continue
                                if document_match_type_is_derivation:
                                    document_match_type = 'derivation'
                                else:
                                    document_match_type = 'direct'
                                match_type = self._match_type(False, search_phrase_match_type,
                                        document_match_type)
                                minimal_match = Match(search_phrase.label, document_label, True,
                                search_phrase.topic_match_phraselet_created_without_matching_tags,
                                search_phrase.reverse_only)
                                minimal_match.index_within_document = index.token_index
                                matched = False
                                if len(word_matching_root_token.split()) > 1:
                                    for multiword_span in self._multiword_spans_with_head_token(
                                            doc[index.token_index]):
                                        for textual_representation, _ in \
                                                self._loop_textual_representations(multiword_span):
                                            if textual_representation == \
                                                    word_matching_root_token:
                                                matched = True
                                                minimal_match.word_matches.append(WordMatch(
                                                    search_phrase_token,
                                                    search_phrase_token._.holmes.lemma,
                                                    doc[index.token_index],
                                                    multiword_span.tokens[0],
                                                    multiword_span.tokens[-1],
                                                    None,
                                                    document_word_representation,
                                                    match_type,
                                                    1.0, False, False, doc[index.token_index],
                                                    document_word_representation, depth))
                                                break
                                        if matched:
                                            break
                                if not matched:
                                    token = doc[index.token_index]
                                    if index.is_subword():
                                        subword = token._.holmes.subwords[index.subword_index]
                                    else:
                                        subword = None
                                    minimal_match.word_matches.append(WordMatch(
                                            search_phrase_token,
                                            search_phrase_token._.holmes.lemma,
                                            token,
                                            token,
                                            token,
                                            subword,
                                            document_word_representation,
                                            match_type,
                                            1.0, token._.holmes.is_negated, False, token,
                                            document_word_representation, depth))
                                    if token._.holmes.is_negated:
                                        minimal_match.is_negated = True
                                existing_minimal_match_indexes.append(index)
                                matches.append(minimal_match)
                    continue
                direct_matching_indexes = []
                if self._is_entitynoun_search_phrase_token(search_phrase.root_token,
                        search_phrase.topic_match_phraselet): # phraselets are not generated for
                                                              # ENTITYNOUN roots
                    for token in doc:
                        if token.pos_ in self.semantic_analyzer.noun_pos:
                            matches.extend(self._get_matches_starting_at_root_word_match(
                                    search_phrase, doc, token, None, document_label,
                                    compare_embeddings_on_non_root_words))
                    continue
                else:
                    matched_indexes_set = set()
                    if self._is_entity_search_phrase_token(search_phrase.root_token,
                        search_phrase.topic_match_phraselet):
                        if search_phrase.topic_match_phraselet:
                            entity_label = search_phrase.root_token._.holmes.lemma
                        else:
                            entity_label = search_phrase.root_token.text
                        if entity_label in registered_document.words_to_token_info_dict.keys():
                            entity_matching_indexes = [index for index, _, _ in
                                    registered_document.words_to_token_info_dict[
                                    entity_label]]
                            if match_specific_indexes:
                                entity_matching_indexes = [index for index in
                                        entity_matching_indexes if index in
                                        reverse_matching_indexes or index in
                                        embedding_reverse_matching_indexes and
                                        not index.is_subword()]
                            matched_indexes_set.update(entity_matching_indexes)
                    else:
                        for word_matching_root_token in search_phrase.words_matching_root_token:
                            if word_matching_root_token in \
                                    registered_document.words_to_token_info_dict.keys():
                                direct_matching_indexes = [index for index, _, _ in
                                        registered_document.words_to_token_info_dict[
                                        word_matching_root_token]]
                                if match_specific_indexes:
                                    direct_matching_indexes = [index for index in
                                            direct_matching_indexes if index in
                                            reverse_matching_indexes or index in
                                            embedding_reverse_matching_indexes]
                                matched_indexes_set.update(direct_matching_indexes)
                if compare_embeddings_on_root_words and not \
                        self._is_entity_search_phrase_token(search_phrase.root_token,
                        search_phrase.topic_match_phraselet) and not search_phrase.reverse_only and\
                        self.semantic_analyzer.embedding_matching_permitted(
                        search_phrase.root_token):
                    if not search_phrase.topic_match_phraselet and \
                            len(search_phrase.root_token._.holmes.lemma.split()) > 1:
                        root_token_lemma_to_use = search_phrase.root_token.lemma_
                    else:
                        root_token_lemma_to_use = search_phrase.root_token._.holmes.lemma
                    if root_token_lemma_to_use in root_lexeme_to_indexes_to_match_dict:
                        matched_indexes_set.update(root_lexeme_to_indexes_to_match_dict[
                                root_token_lemma_to_use])
                    else:
                        working_indexes_to_match_for_cache_set = set()
                        for document_word in registered_document.words_to_token_info_dict.keys():
                            indexes_to_match = [index for index, _, _ in
                                    registered_document.words_to_token_info_dict[document_word]]
                            if match_specific_indexes:
                                indexes_to_match = [index for index in indexes_to_match if
                                        index in embedding_reverse_matching_indexes and index
                                        not in direct_matching_indexes]
                                if len(indexes_to_match) == 0:
                                    continue
                            search_phrase_lexeme = \
                                    search_phrase.matchable_non_entity_tokens_to_lexemes[
                                    search_phrase.root_token.i]
                            example_index = indexes_to_match[0]
                            example_document_token = doc[example_index.token_index]
                            if example_index.is_subword():
                                if not self.semantic_analyzer.embedding_matching_permitted(
                                        example_document_token._.holmes.subwords[
                                        example_index.subword_index]):
                                    continue
                                document_lemma = example_document_token._.holmes.subwords[
                                        example_index.subword_index].lemma
                            else:
                                if not self.semantic_analyzer.embedding_matching_permitted(
                                        example_document_token):
                                    continue
                                if len(example_document_token._.holmes.lemma.split()) > 1:
                                    document_lemma = example_document_token.lemma_
                                else:
                                    document_lemma = example_document_token._.holmes.lemma
                            document_lexeme = self.semantic_analyzer.nlp.vocab[document_lemma]
                            if search_phrase_lexeme.vector_norm > 0 and \
                                    document_lexeme.vector_norm > 0:
                                similarity_measure = search_phrase_lexeme.similarity(
                                        document_lexeme)
                                if similarity_measure >= \
                                        search_phrase.single_token_similarity_threshold:
                                    matched_indexes_set.update(indexes_to_match)
                                    working_indexes_to_match_for_cache_set.update(indexes_to_match)
                        root_lexeme_to_indexes_to_match_dict[root_token_lemma_to_use] = \
                                working_indexes_to_match_for_cache_set
                for index_to_match in sorted(matched_indexes_set):
                    matches.extend(self._get_matches_starting_at_root_word_match(
                            search_phrase, doc, doc[index_to_match.token_index],
                            index_to_match.subword_index, document_label,
                            compare_embeddings_on_non_root_words))
        return sorted(matches, key=lambda match: 1 - float(match.overall_similarity_measure))
