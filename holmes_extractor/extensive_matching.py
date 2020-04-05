import collections
import jsonpickle
import uuid
import statistics
from scipy.sparse import dok_matrix
from sklearn.neural_network import MLPClassifier
from .structural_matching import Index
from .errors import WrongModelDeserializationError, FewerThanTwoClassificationsError, \
        DuplicateDocumentError, NoPhraseletsAfterFilteringError, \
        EmbeddingThresholdGreaterThanRelationThresholdError, \
        IncompatibleAnalyzeDerivationalMorphologyDeserializationError

class TopicMatch:
    """A topic match between some text and part of a document. Note that the end indexes refer
        to the token in question rather than to the following token.

    Properties:

    document_label -- the document label.
    index_within_document -- the index of the token within the document where 'score' was achieved.
    subword_index -- the index of the subword within the token within the document where 'score'
            was achieved, or *None* if the match involved the whole word.
    start_index -- the start index of the topic match within the document.
    end_index -- the end index of the topic match within the document.
    sentences_start_index -- the start index within the document of the sentence that contains
        'start_index'
    sentences_end_index -- the end index within the document of the sentence that contains
        'end_index'
    relative_start_index -- the start index of the topic match relative to 'sentences_start_index'
    relative_end_index -- the end index of the topic match relative to 'sentences_start_index'
    score -- the similarity score of the topic match
    text -- the text between 'sentences_start_index' and 'sentences_end_index'
    structural_matches -- a list of `Match` objects that were used to derive this object.
    """

    def __init__(self, document_label, index_within_document, subword_index, start_index, end_index,
            sentences_start_index, sentences_end_index, score, text, structural_matches):
        self.document_label = document_label
        self.index_within_document = index_within_document
        self.subword_index = subword_index
        self.start_index = start_index
        self.end_index = end_index
        self.sentences_start_index = sentences_start_index
        self.sentences_end_index = sentences_end_index
        self.score = score
        self.text = text
        self.structural_matches = structural_matches

    @property
    def relative_start_index(self):
        return self.start_index - self.sentences_start_index

    @property
    def relative_end_index(self):
        return self.end_index - self.sentences_start_index

class PhraseletActivationTracker:
    """ Tracks the activation for a specific phraselet - the most recent score
        and the position within the document at which that score was calculated.
    """
    def __init__(self, position, score):
        self.position = position
        self.score = score

class TopicMatcher:
    """A topic matcher object. See manager.py for details of the properties."""

    def __init__(self, *, semantic_analyzer, structural_matcher, indexed_documents,
            maximum_activation_distance, relation_score, reverse_only_relation_score,
            single_word_score, single_word_any_tag_score, overlapping_relation_multiplier,
            embedding_penalty, ontology_penalty,
            maximum_number_of_single_word_matches_for_relation_matching,
            maximum_number_of_single_word_matches_for_embedding_matching,
            sideways_match_extent, only_one_result_per_document, number_of_results,
            document_label_filter):
        if maximum_number_of_single_word_matches_for_embedding_matching > \
                maximum_number_of_single_word_matches_for_relation_matching:
            raise EmbeddingThresholdGreaterThanRelationThresholdError(' '.join((
                    'embedding',
                    str(maximum_number_of_single_word_matches_for_embedding_matching),
                    'relation',
                    str(maximum_number_of_single_word_matches_for_relation_matching))))
        self._semantic_analyzer = semantic_analyzer
        self.structural_matcher = structural_matcher
        self.indexed_documents = indexed_documents
        self._ontology = structural_matcher.ontology
        self.maximum_activation_distance = maximum_activation_distance
        self.relation_score = relation_score
        self.reverse_only_relation_score = reverse_only_relation_score
        self.single_word_score = single_word_score
        self.single_word_any_tag_score = single_word_any_tag_score
        self.overlapping_relation_multiplier = overlapping_relation_multiplier
        self.embedding_penalty = embedding_penalty
        self.ontology_penalty = ontology_penalty
        self.maximum_number_of_single_word_matches_for_relation_matching = \
                maximum_number_of_single_word_matches_for_relation_matching
        self.maximum_number_of_single_word_matches_for_embedding_matching = \
                maximum_number_of_single_word_matches_for_embedding_matching
        self.sideways_match_extent = sideways_match_extent
        self.only_one_result_per_document = only_one_result_per_document
        self.number_of_results = number_of_results
        self.document_label_filter = document_label_filter

    def _get_word_match_from_match(self, match, parent):
        ## child if parent==False
        for word_match in match.word_matches:
            if parent and word_match.search_phrase_token.dep_ == 'ROOT':
                return word_match
            if not parent and word_match.search_phrase_token.dep_ != 'ROOT':
                return word_match
        raise RuntimeError(''.join(('Word match not found with parent==', str(parent))))

    def _add_to_dict_list(self, dict, key, value):
        if key in dict:
            dict[key].append(value)
        else:
            dict[key] = [value]

    def _add_to_dict_set(self, dict, key, value):
        if not key in dict:
            dict[key] = set()
        dict[key].add(value)

    def topic_match_documents_against(self, text_to_match):
        """ Performs a topic match against the loaded documents.

        Property:

        text_to_match -- the text to match against the documents.
        """

        class CorpusWordPosition:
            def __init__(self, document_label, index):
                self.document_label = document_label
                self.index = index

            def __eq__(self, other):
                return type(other) == CorpusWordPosition and self.index == other.index and \
                        self.document_label == other.document_label

            def __hash__(self):
                return hash((self.document_label, self.index))

            def __str__(self):
                return ':'.join((self.document_label, str(self.index)))

        class PhraseletWordMatchInfo:
            def __init__(self):
                self.single_word_match_corpus_words = set()
                # The indexes at which the single word phraselet for this word was matched.

                self.phraselet_labels_to_parent_match_corpus_words = {}
                # Dictionary from phraselets with this word as the parent to indexes where the
                # phraselet was matched.

                self.phraselet_labels_to_child_match_corpus_words = {}
                # Dictionary from phraselets with this word as the child to indexes where the
                # phraselet was matched.

                self.parent_match_corpus_words_to_matches = {}
                # Dictionary from indexes where phraselets with this word as the parent were matched
                # to the match objects.

                self.child_match_corpus_words_to_matches = {}
                # Dictionary from indexes where phraselets with this word as the child were matched
                # to the match objects.

        def get_phraselet_word_match_info(word):
            if word in self._words_to_phraselet_word_match_infos:
                return self._words_to_phraselet_word_match_infos[word]
            else:
                phraselet_word_match_info = PhraseletWordMatchInfo()
                self._words_to_phraselet_word_match_infos[word] = phraselet_word_match_info
                return phraselet_word_match_info

        def set_phraselet_to_reverse_only_where_too_many_single_word_matches(phraselet):
            """ Where the parent word of a phraselet matched too often in the corpus, the phraselet
                is set to reverse matching only to improve performance.
            """
            parent_token = phraselet.root_token
            parent_word = parent_token._.holmes.lemma_or_derived_lemma()
            if parent_word in self._words_to_phraselet_word_match_infos:
                parent_phraselet_word_match_info = self._words_to_phraselet_word_match_infos[
                        parent_word]
                parent_single_word_match_corpus_words = \
                        parent_phraselet_word_match_info.single_word_match_corpus_words
                if len(parent_single_word_match_corpus_words) > \
                        self.maximum_number_of_single_word_matches_for_relation_matching:
                    phraselet.treat_as_reverse_only_during_initial_relation_matching = True

        def get_indexes_for_reverse_matching(*, phraselet,
                parent_document_labels_to_indexes_for_direct_retry_sets,
                parent_document_labels_to_indexes_for_embedding_retry_sets,
                child_document_labels_to_indexes_for_embedding_retry_sets):
            """
            parent_document_labels_to_indexes_for_direct_retry_sets -- indexes where matching
                against a reverse matching phraselet should be attempted. These are ascertained
                by examining the child words.
            parent_document_labels_to_indexes_for_embedding_retry_sets -- indexes where matching
                against a phraselet should be attempted with embedding-based matching on the
                parent (root) word. These are ascertained by examining the child words.
            child_document_labels_to_indexes_for_embedding_retry_sets -- indexes where matching
                against a phraselet should be attempted with embedding-based matching on the
                child (non-root) word. These are ascertained by examining the parent words.
            """

            parent_token = phraselet.root_token
            parent_word = parent_token._.holmes.lemma_or_derived_lemma()
            if parent_word in self._words_to_phraselet_word_match_infos and not \
                    phraselet.reverse_only and not \
                    phraselet.treat_as_reverse_only_during_initial_relation_matching:
                parent_phraselet_word_match_info = self._words_to_phraselet_word_match_infos[
                        parent_word]
                parent_single_word_match_corpus_words = \
                        parent_phraselet_word_match_info.single_word_match_corpus_words
                if phraselet.label in parent_phraselet_word_match_info.\
                        phraselet_labels_to_parent_match_corpus_words:
                    parent_relation_match_corpus_words = \
                            parent_phraselet_word_match_info.\
                            phraselet_labels_to_parent_match_corpus_words[phraselet.label]
                else:
                    parent_relation_match_corpus_words = []
                if len(parent_single_word_match_corpus_words) <= \
                        self.maximum_number_of_single_word_matches_for_embedding_matching:
                    # we deliberately use the number of single matches rather than the difference
                    # because the deciding factor should be whether or not enough match information
                    # has been returned without checking the embeddings
                    for corpus_word_position in \
                            parent_single_word_match_corpus_words.difference(
                            parent_relation_match_corpus_words):
                        self._add_to_dict_set(
                                child_document_labels_to_indexes_for_embedding_retry_sets,
                                corpus_word_position.document_label, Index(
                                corpus_word_position.index.token_index,
                                corpus_word_position.index.subword_index))
            child_token = [token for token in phraselet.matchable_tokens if token.i !=
                    parent_token.i][0]
            child_word = child_token._.holmes.lemma_or_derived_lemma()
            if child_word in self._words_to_phraselet_word_match_infos:
                child_phraselet_word_match_info = \
                        self._words_to_phraselet_word_match_infos[child_word]
                child_single_word_match_corpus_words = \
                        child_phraselet_word_match_info.single_word_match_corpus_words
                if phraselet.label in child_phraselet_word_match_info.\
                        phraselet_labels_to_child_match_corpus_words:
                    child_relation_match_corpus_words =  child_phraselet_word_match_info.\
                            phraselet_labels_to_child_match_corpus_words[phraselet.label]
                else:
                    child_relation_match_corpus_words = []
                if len(child_single_word_match_corpus_words) <= \
                        self.maximum_number_of_single_word_matches_for_embedding_matching:
                    set_to_add_to = parent_document_labels_to_indexes_for_embedding_retry_sets
                elif len(child_single_word_match_corpus_words) <= \
                        self.maximum_number_of_single_word_matches_for_relation_matching and (
                        phraselet.reverse_only or
                        phraselet.treat_as_reverse_only_during_initial_relation_matching):
                    set_to_add_to = parent_document_labels_to_indexes_for_direct_retry_sets
                else:
                    return
                linking_dependency = parent_token._.holmes.get_label_of_dependency_with_child_index(
                        child_token.i)
                for corpus_word_position in child_single_word_match_corpus_words.difference(
                        child_relation_match_corpus_words):
                    doc = self.indexed_documents[corpus_word_position.document_label].doc
                    working_index = corpus_word_position.index
                    working_token = doc[working_index.token_index]
                    if not working_index.is_subword() or \
                            working_token._.holmes.subwords[working_index.subword_index].is_head:
                        for parent_dependency in working_token._.holmes.parent_dependencies:
                            if self._semantic_analyzer.dependency_labels_match(
                                    search_phrase_dependency_label=linking_dependency,
                                    document_dependency_label=parent_dependency[1]):
                                self._add_to_dict_set(
                                        set_to_add_to,
                                        corpus_word_position.document_label,
                                        Index(parent_dependency[0], None))
                    else:
                        working_subword = \
                                working_token._.holmes.subwords[working_index.subword_index]
                        if working_subword.governor_index != None and \
                                self._semantic_analyzer.dependency_labels_match(
                                search_phrase_dependency_label=linking_dependency,
                                document_dependency_label=
                                working_subword.governing_dependency_label):
                            self._add_to_dict_set(
                                    set_to_add_to,
                                    corpus_word_position.document_label,
                                    Index(working_index.token_index,
                                    working_subword.governor_index))

        def rebuild_document_info_dict(matches, phraselet_labels_to_phraselet_infos):

            def process_word_match(match, parent): # 'True' -> parent, 'False' -> child
                word_match = self._get_word_match_from_match(match, parent)
                word = word_match.search_phrase_token._.holmes.lemma_or_derived_lemma()
                phraselet_word_match_info = get_phraselet_word_match_info(word)
                corpus_word_position = CorpusWordPosition(match.document_label,
                        word_match.get_document_index())
                if parent:
                    self._add_to_dict_list(
                            phraselet_word_match_info.parent_match_corpus_words_to_matches,
                            corpus_word_position, match)
                    self._add_to_dict_list(
                            phraselet_word_match_info.phraselet_labels_to_parent_match_corpus_words,
                            match.search_phrase_label, corpus_word_position)
                else:
                    self._add_to_dict_list(
                            phraselet_word_match_info.child_match_corpus_words_to_matches,
                            corpus_word_position, match)
                    self._add_to_dict_list(
                            phraselet_word_match_info.phraselet_labels_to_child_match_corpus_words,
                            match.search_phrase_label, corpus_word_position)

            self._words_to_phraselet_word_match_infos = {}
            for match in matches:
                if match.from_single_word_phraselet:
                    phraselet_info = phraselet_labels_to_phraselet_infos[match.search_phrase_label]
                    word = phraselet_info.parent_derived_lemma
                    phraselet_word_match_info = get_phraselet_word_match_info(word)
                    word_match = match.word_matches[0]
                    phraselet_word_match_info.single_word_match_corpus_words.add(
                            CorpusWordPosition(match.document_label,
                            word_match.get_document_index()))
                else:
                    process_word_match(match, True)
                    process_word_match(match, False)

        def filter_superfluous_matches(match):

            def get_other_matches_at_same_word(match, parent):  # 'True' -> parent, 'False' -> child
                word_match = self._get_word_match_from_match(match, parent)
                word = word_match.search_phrase_token._.holmes.lemma_or_derived_lemma()
                phraselet_word_match_info = get_phraselet_word_match_info(word)
                corpus_word_position = CorpusWordPosition(match.document_label,
                        word_match.get_document_index())
                if parent:
                    match_dict = phraselet_word_match_info.parent_match_corpus_words_to_matches
                else:
                    match_dict = phraselet_word_match_info.child_match_corpus_words_to_matches
                return match_dict[corpus_word_position]

            def check_for_sibling_match_with_higher_similarity(match, other_match,
                    word_match, other_word_match):
                    # We do not want the same phraselet to match multiple siblings, so choose
                    # the sibling that is most similar to the search phrase token.
                if self.structural_matcher.overall_similarity_threshold == 1.0:
                    return True
                if word_match.document_token.i == other_word_match.document_token.i:
                    return True
                working_sibling = word_match.document_token.doc[
                        word_match.document_token._.holmes.token_or_lefthand_sibling_index]
                for sibling in \
                        working_sibling._.holmes.loop_token_and_righthand_siblings(
                                word_match.document_token.doc):
                    if match.search_phrase_label == other_match.search_phrase_label and \
                            other_word_match.document_token.i == sibling.i and \
                            other_word_match.similarity_measure > word_match.similarity_measure:
                        return False
                return True

            def perform_checks_at_pole(match, parent): # pole is 'True' -> parent, 'False' -> child
                this_this_pole_word_match = self._get_word_match_from_match(match, parent)
                this_pole_index = this_this_pole_word_match.document_token.i
                this_other_pole_word_match = self._get_word_match_from_match(match, not parent)
                for other_this_pole_match in get_other_matches_at_same_word(match, parent):
                    other_other_pole_word_match = \
                            self._get_word_match_from_match(other_this_pole_match, not parent)
                    if this_other_pole_word_match.document_subword != None:
                        this_other_pole_subword_index = this_other_pole_word_match.\
                                document_subword.index
                    else:
                        this_other_pole_subword_index = None
                    if other_other_pole_word_match.document_subword != None:
                        other_other_pole_subword_index = other_other_pole_word_match.\
                                document_subword.index
                    else:
                        other_other_pole_subword_index = None
                    if this_other_pole_word_match.document_token.i == other_other_pole_word_match.\
                            document_token.i and this_other_pole_subword_index == \
                            other_other_pole_subword_index and \
                            other_other_pole_word_match.similarity_measure > \
                            this_other_pole_word_match.similarity_measure:
                        # The other match has a higher similarity measure at the other pole than
                        # this match. The matched tokens are the same. The matching phraselets
                        # must be different.
                        return False
                    if this_other_pole_word_match.document_token.i == other_other_pole_word_match.\
                            document_token.i and this_other_pole_subword_index != None \
                            and other_other_pole_subword_index == None:
                        # This match is with a subword where the other match has matched the entire
                        # word, so this match should be removed.
                        return False
                        # Check unnecessary if parent==True as it has then already
                        # been carried out during structural matching.
                    if not parent and this_other_pole_word_match.document_token.i != \
                            other_other_pole_word_match.document_token.i and \
                            other_other_pole_word_match.document_token.i in \
                            this_other_pole_word_match.document_token._.\
                            holmes.token_and_coreference_chain_indexes and \
                            match.search_phrase_label == other_this_pole_match.search_phrase_label \
                            and ((abs(this_pole_index -
                                    this_other_pole_word_match.document_token.i) \
                             > abs(this_pole_index - other_other_pole_word_match.document_token.i))\
                             or (abs(this_pole_index - this_other_pole_word_match.document_token.i)
                             == abs(this_pole_index - other_other_pole_word_match.document_token.i)
                             and this_other_pole_word_match.document_token.i >
                             other_other_pole_word_match.document_token.i)):
                        # The document tokens at the other poles corefer with each other and
                        # the other match's token is closer to the second document token (the
                        # one at this pole). Both matches are from the same phraselet.
                        # If the tokens from the two matches are the same distance from the document
                        # token at this pole but on opposite sides of it, the preceding one beats
                        # the succeeding one simply because we have to choose one or the other.
                        return False

                    if not check_for_sibling_match_with_higher_similarity(match,
                            other_this_pole_match, this_other_pole_word_match,
                            other_other_pole_word_match):
                        return False
                return True

            if match.from_single_word_phraselet:
                return True
            if not perform_checks_at_pole(match, True):
                return False
            if not perform_checks_at_pole(match, False):
                return False
            return True

        def remove_duplicates(matches):
            # Situations where the same document tokens have been matched by multiple phraselets
            matches_to_return = []
            if len(matches) == 0:
                return matches_to_return
            else:
                matches_to_return.append(matches[0])
            if len(matches) > 1:
                previous_whole_word_single_word_match = None
                for counter in range(1, len(matches)):
                    this_match = matches[counter]
                    previous_match = matches[counter-1]
                    if this_match.index_within_document == previous_match.index_within_document:
                        if previous_match.from_single_word_phraselet and \
                                previous_match.get_subword_index() == None:
                            previous_whole_word_single_word_match = previous_match
                        if this_match.get_subword_index() != None and \
                                previous_whole_word_single_word_match != None and \
                                this_match.index_within_document == \
                                previous_whole_word_single_word_match.index_within_document:
                            # This match is against a subword where the whole word has also been
                            # matched, so reject it
                            continue
                    if this_match.document_label != previous_match.document_label:
                        matches_to_return.append(this_match)
                    elif len(this_match.word_matches) != len(previous_match.word_matches):
                        matches_to_return.append(this_match)
                    else:
                        this_word_matches_indexes = [word_match.get_document_index() for
                                word_match in this_match.word_matches]
                        previous_word_matches_indexes = [word_match.get_document_index() for
                                word_match in previous_match.word_matches]
                        # In some circumstances the two phraselets may have matched the same
                        # tokens the opposite way round
                        if sorted(this_word_matches_indexes) != \
                                sorted(previous_word_matches_indexes):
                            matches_to_return.append(this_match)
            return matches_to_return

        doc = self._semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_phraselet_infos = {}
        self.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                replace_with_hypernym_ancestors=False,
                match_all_words = False,
                returning_serialized_phraselets = False,
                ignore_relation_phraselets = False,
                include_reverse_only = True,
                stop_lemmas = self._semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                reverse_only_parent_lemmas =
                        self._semantic_analyzer.topic_matching_reverse_only_parent_lemmas)

        # now add the single word phraselets whose tags did not match.
        self.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                replace_with_hypernym_ancestors=False,
                match_all_words = True,
                returning_serialized_phraselets = False,
                ignore_relation_phraselets = True,
                include_reverse_only = False, # value is irrelevant with
                                              # ignore_relation_phraselets == True
                stop_lemmas = self._semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                reverse_only_parent_lemmas =
                        self._semantic_analyzer.topic_matching_reverse_only_parent_lemmas)
        if len(phraselet_labels_to_phraselet_infos) == 0:
            return []
        phraselet_labels_to_search_phrases = \
                self.structural_matcher.create_search_phrases_from_phraselet_infos(
                phraselet_labels_to_phraselet_infos.values())
        # First get single-word matches
        structural_matches = self.structural_matcher.match(
                indexed_documents = self.indexed_documents,
                search_phrases = phraselet_labels_to_search_phrases.values(),
                output_document_matching_message_to_console = False,
                match_depending_on_single_words = True,
                compare_embeddings_on_root_words = False,
                compare_embeddings_on_non_root_words = False,
                document_labels_to_indexes_for_reverse_matching_sets = None,
                document_labels_to_indexes_for_embedding_reverse_matching_sets = None,
                document_label_filter = self.document_label_filter)
        if not self.structural_matcher.embedding_based_matching_on_root_words:
            rebuild_document_info_dict(structural_matches, phraselet_labels_to_phraselet_infos)
            for phraselet in (phraselet_labels_to_search_phrases[phraselet_info.label] for
                    phraselet_info in phraselet_labels_to_phraselet_infos.values() if
                    phraselet_info.child_lemma != None):
                set_phraselet_to_reverse_only_where_too_many_single_word_matches(phraselet)

        # Now get normally matched relations
        structural_matches.extend(self.structural_matcher.match(
                indexed_documents = self.indexed_documents,
                search_phrases = phraselet_labels_to_search_phrases.values(),
                output_document_matching_message_to_console = False,
                match_depending_on_single_words = False,
                compare_embeddings_on_root_words = False,
                compare_embeddings_on_non_root_words = False,
                document_labels_to_indexes_for_reverse_matching_sets = None,
                document_labels_to_indexes_for_embedding_reverse_matching_sets = None,
                document_label_filter = self.document_label_filter))

        rebuild_document_info_dict(structural_matches, phraselet_labels_to_phraselet_infos)
        parent_document_labels_to_indexes_for_direct_retry_sets = {}
        parent_document_labels_to_indexes_for_embedding_retry_sets = {}
        child_document_labels_to_indexes_for_embedding_retry_sets = {}
        for phraselet in (phraselet_labels_to_search_phrases[phraselet_info.label] for
                phraselet_info in phraselet_labels_to_phraselet_infos.values() if
                phraselet_info.child_lemma != None):
            get_indexes_for_reverse_matching(phraselet=phraselet,
                    parent_document_labels_to_indexes_for_direct_retry_sets =
                    parent_document_labels_to_indexes_for_direct_retry_sets,
                    parent_document_labels_to_indexes_for_embedding_retry_sets =
                    parent_document_labels_to_indexes_for_embedding_retry_sets,
                    child_document_labels_to_indexes_for_embedding_retry_sets =
                    child_document_labels_to_indexes_for_embedding_retry_sets)
        if len(parent_document_labels_to_indexes_for_embedding_retry_sets) > 0 or \
                len(parent_document_labels_to_indexes_for_direct_retry_sets) > 0:

            # Perform reverse matching at selected indexes
            structural_matches.extend(self.structural_matcher.match(
                    indexed_documents = self.indexed_documents,
                    search_phrases = phraselet_labels_to_search_phrases.values(),
                    output_document_matching_message_to_console = False,
                    match_depending_on_single_words = False,
                    compare_embeddings_on_root_words = True,
                    compare_embeddings_on_non_root_words = False,
                    document_labels_to_indexes_for_reverse_matching_sets =
                    parent_document_labels_to_indexes_for_direct_retry_sets,
                    document_labels_to_indexes_for_embedding_reverse_matching_sets =
                    parent_document_labels_to_indexes_for_embedding_retry_sets,
                    document_label_filter = self.document_label_filter))

        if len(child_document_labels_to_indexes_for_embedding_retry_sets) > 0:

            # Retry normal matching at selected indexes with embedding-based matching on children
            structural_matches.extend(self.structural_matcher.match(
                    indexed_documents = self.indexed_documents,
                    search_phrases = phraselet_labels_to_search_phrases.values(),
                    output_document_matching_message_to_console = False,
                    match_depending_on_single_words = False,
                    compare_embeddings_on_root_words = False,
                    compare_embeddings_on_non_root_words = True,
                    document_labels_to_indexes_for_reverse_matching_sets = None,
                    document_labels_to_indexes_for_embedding_reverse_matching_sets =
                    child_document_labels_to_indexes_for_embedding_retry_sets,
                    document_label_filter = self.document_label_filter))
        if len(parent_document_labels_to_indexes_for_direct_retry_sets) > 0 or \
                len(parent_document_labels_to_indexes_for_embedding_retry_sets) > 0 or \
                len(child_document_labels_to_indexes_for_embedding_retry_sets) > 0:
            rebuild_document_info_dict(structural_matches, phraselet_labels_to_phraselet_infos)
        structural_matches = list(filter(filter_superfluous_matches, structural_matches))
        position_sorted_structural_matches = \
                sorted(structural_matches, key=lambda match:
                (match.document_label, match.index_within_document,
                match.get_subword_index_for_sorting(), match.from_single_word_phraselet))
        position_sorted_structural_matches = remove_duplicates(position_sorted_structural_matches)
        # Read through the documents measuring the activation based on where
        # in the document structural matches were found
        score_sorted_structural_matches = self.perform_activation_scoring(
                position_sorted_structural_matches)
        return self.get_topic_matches(score_sorted_structural_matches,
                position_sorted_structural_matches)

    def perform_activation_scoring(self, position_sorted_structural_matches):
        """
        Read through the documents measuring the activation based on where
        in the document structural matches were found.
        """
        def get_set_from_dict(dict, key):
            if key in dict:
                return dict[key]
            else:
                return set()

        def get_current_activation_for_phraselet(phraselet_activation_tracker, current_index):
            distance_to_last_match = current_index - phraselet_activation_tracker.position
            tailoff_quotient = distance_to_last_match / self.maximum_activation_distance
            if tailoff_quotient > 1.0:
                tailoff_quotient = 1.0
            return (1-tailoff_quotient) * phraselet_activation_tracker.score

        document_labels_to_indexes_to_phraselet_labels = {}
        for match in (match for match in position_sorted_structural_matches if not
                match.from_single_word_phraselet):
            if match.document_label in document_labels_to_indexes_to_phraselet_labels:
                inner_dict = document_labels_to_indexes_to_phraselet_labels[match.document_label]
            else:
                inner_dict = {}
                document_labels_to_indexes_to_phraselet_labels[match.document_label] = inner_dict
            parent_word_match = self._get_word_match_from_match(match, True)
            self._add_to_dict_set(inner_dict, parent_word_match.get_document_index(),
                    match.search_phrase_label)
            child_word_match = self._get_word_match_from_match(match, False)
            self._add_to_dict_set(inner_dict, child_word_match.get_document_index(),
                    match.search_phrase_label)
        current_document_label = None
        for pssm_index, match in enumerate(position_sorted_structural_matches):
            match.original_index_within_list = pssm_index # store for later use after resorting
            if match.document_label != current_document_label or pssm_index == 0:
                current_document_label = match.document_label
                current_activation_score = 0
                phraselet_labels_to_phraselet_activation_trackers = {}
                if current_document_label in document_labels_to_indexes_to_phraselet_labels:
                    indexes_to_phraselet_labels = \
                            document_labels_to_indexes_to_phraselet_labels[current_document_label]
                else:
                    indexes_to_phraselet_labels = {}
            match.is_overlapping_relation = False
            if match.from_single_word_phraselet:
                if match.from_topic_match_phraselet_created_without_matching_tags:
                    this_match_score = self.single_word_any_tag_score
                else:
                    this_match_score = self.single_word_score
            else:
                if match.from_reverse_only_topic_match_phraselet:
                    this_match_score = self.reverse_only_relation_score
                else:
                    this_match_score = self.relation_score
                this_match_parent_word_match = self._get_word_match_from_match(match, True)
                this_match_parent_index = this_match_parent_word_match.get_document_index()
                this_match_child_word_match = self._get_word_match_from_match(match, False)
                this_match_child_index = this_match_child_word_match.get_document_index()
                other_relevant_phraselet_labels = get_set_from_dict(indexes_to_phraselet_labels,
                        this_match_parent_index) | get_set_from_dict(indexes_to_phraselet_labels,
                        this_match_child_index)
                other_relevant_phraselet_labels.remove(match.search_phrase_label)
                if len(other_relevant_phraselet_labels) > 0:
                    match.is_overlapping_relation=True
                    this_match_score *= self.overlapping_relation_multiplier
            overall_similarity_measure = float(match.overall_similarity_measure)
            if overall_similarity_measure < 1.0:
                this_match_score *= self.embedding_penalty * overall_similarity_measure
            for word_match in (word_match for word_match in match.word_matches \
                    if word_match.type == 'ontology'):
                this_match_score *= (self.ontology_penalty ** (abs(word_match.depth) + 1))
            if match.search_phrase_label in phraselet_labels_to_phraselet_activation_trackers:
                phraselet_activation_tracker = phraselet_labels_to_phraselet_activation_trackers[
                        match.search_phrase_label]
                current_score = get_current_activation_for_phraselet(phraselet_activation_tracker,
                        match.index_within_document)
                if this_match_score > current_score:
                    phraselet_activation_tracker.score = this_match_score
                else:
                    phraselet_activation_tracker.score = current_score
                phraselet_activation_tracker.position = match.index_within_document
            else:
                phraselet_labels_to_phraselet_activation_trackers[match.search_phrase_label] =\
                        PhraseletActivationTracker(match.index_within_document, this_match_score)
            match.topic_score = 0
            for phraselet_label in list(phraselet_labels_to_phraselet_activation_trackers):
                phraselet_activation_tracker = phraselet_labels_to_phraselet_activation_trackers[
                        phraselet_label]
                current_activation = \
                        get_current_activation_for_phraselet(phraselet_activation_tracker,
                        match.index_within_document)
                if current_activation <= 0:
                    del phraselet_labels_to_phraselet_activation_trackers[phraselet_label]
                else:
                    match.topic_score += current_activation
        return sorted(position_sorted_structural_matches, key=lambda match: 0-match.topic_score)

    def get_topic_matches(self, score_sorted_structural_matches,
            position_sorted_structural_matches):
        """Resort the matches starting with the highest (most active) and
            create topic match objects with information about the surrounding sentences.
        """

        def match_contained_within_existing_topic_match(topic_matches, match):
            for topic_match in topic_matches:
                if match.document_label == topic_match.document_label and \
                        match.index_within_document >= topic_match.start_index and \
                        match.index_within_document <= topic_match.end_index:
                    return True
            return False

        def alter_start_and_end_indexes_for_match(start_index, end_index, match,
                reference_match_index_within_document):
            if match.index_within_document < start_index:
                start_index = match.index_within_document
            for word_match in match.word_matches:
                if word_match.first_document_token.i < start_index:
                    start_index = word_match.first_document_token.i
                if word_match.document_subword != None and \
                        word_match.document_subword.containing_token_index < start_index:
                    start_index = word_match.document_subword.containing_token_index
            if match.index_within_document > end_index:
                end_index = match.index_within_document
            for word_match in match.word_matches:
                if word_match.last_document_token.i > end_index:
                    end_index = word_match.last_document_token.i
                if word_match.document_subword != None and \
                        word_match.document_subword.containing_token_index > end_index:
                    end_index = word_match.document_subword.containing_token_index
            return start_index, end_index

        if self.only_one_result_per_document:
            existing_document_labels = []
        topic_matches = []
        counter = 0
        for score_sorted_match in score_sorted_structural_matches:
            if counter >= self.number_of_results:
                break
            if match_contained_within_existing_topic_match(topic_matches,
                    score_sorted_match):
                continue
            if self.only_one_result_per_document and score_sorted_match.document_label \
                    in existing_document_labels:
                continue
            start_index, end_index = alter_start_and_end_indexes_for_match(
                    score_sorted_match.index_within_document,
                    score_sorted_match.index_within_document,
                    score_sorted_match, score_sorted_match.index_within_document)
            previous_index_within_list = score_sorted_match.original_index_within_list
            while previous_index_within_list > 0 and position_sorted_structural_matches[
                    previous_index_within_list-1].document_label == \
                    score_sorted_match.document_label and position_sorted_structural_matches[
                    previous_index_within_list].topic_score > self.single_word_score:
                    # previous_index_within_list rather than previous_index_within_list -1 :
                    # when a complex structure is matched, it will often begin with a single noun
                    # that should be included within the topic match indexes
                if match_contained_within_existing_topic_match(topic_matches,
                        position_sorted_structural_matches[
                        previous_index_within_list-1]):
                    break
                if score_sorted_match.index_within_document - position_sorted_structural_matches[
                        previous_index_within_list-1].index_within_document > \
                        self.sideways_match_extent:
                    break
                previous_index_within_list -= 1
                start_index, end_index = alter_start_and_end_indexes_for_match(
                        start_index, end_index,
                        position_sorted_structural_matches[previous_index_within_list],
                        score_sorted_match.index_within_document)
            next_index_within_list = score_sorted_match.original_index_within_list
            while next_index_within_list + 1 < len(score_sorted_structural_matches) and \
                    position_sorted_structural_matches[next_index_within_list+1].document_label == \
                    score_sorted_match.document_label and \
                    position_sorted_structural_matches[next_index_within_list+1].topic_score >= \
                    self.single_word_score:
                if match_contained_within_existing_topic_match(topic_matches,
                        position_sorted_structural_matches[
                        next_index_within_list+1]):
                    break
                if position_sorted_structural_matches[
                        next_index_within_list+1].index_within_document - \
                        score_sorted_match.index_within_document > self.sideways_match_extent:
                    break
                next_index_within_list += 1
                start_index, end_index = alter_start_and_end_indexes_for_match(
                        start_index, end_index,
                        position_sorted_structural_matches[next_index_within_list],
                        score_sorted_match.index_within_document)
            working_document = \
                    self.indexed_documents[score_sorted_match.document_label].doc
            relevant_sentences = [sentence for sentence in working_document.sents
                    if sentence.end > start_index and sentence.start <= end_index]
            sentences_start_index = relevant_sentences[0].start
            sentences_end_index = relevant_sentences[-1].end
            text = working_document[sentences_start_index: sentences_end_index].text
            topic_matches.append(TopicMatch(score_sorted_match.document_label,
                    score_sorted_match.index_within_document,
                    score_sorted_match.get_subword_index(),
                    start_index, end_index, sentences_start_index, sentences_end_index - 1,
                    score_sorted_match.topic_score, text, position_sorted_structural_matches[
                    previous_index_within_list:next_index_within_list+1]))
            if self.only_one_result_per_document:
                existing_document_labels.append(score_sorted_match.document_label)
            counter += 1
        # If two matches have the same score, order them by length
        return sorted(topic_matches, key=lambda topic_match: (0-topic_match.score,
                topic_match.start_index - topic_match.end_index))

    def topic_match_documents_returning_dictionaries_against(self, text_to_match,
            tied_result_quotient):
        """Returns a list of dictionaries representing the results of a topic match between an
            entered text and the loaded documents. Callers of this method do not have to manage any
            further dependencies on spaCy or Holmes.

        Properties:

        text_to_match -- the text to match against the loaded documents.
        tied_result_quotient -- the quotient between a result and following results above which
            the results are interpreted as tied
        """

        class WordInfo:

            def __init__(self, relative_start_index, relative_end_index, type, explanation):
                self.relative_start_index = relative_start_index
                self.relative_end_index = relative_end_index
                self.type = type
                self.explanation = explanation
                self.is_highest_activation = False

            def __eq__(self, other):
                return type(other) == WordInfo and self.relative_start_index == \
                        other.relative_start_index and self.relative_end_index == \
                        other.relative_end_index

            def __hash__(self):
                return hash((self.relative_start_index, self.relative_end_index))

        def get_containing_word_info_key(word_infos_to_word_infos, this_word_info):
            for other_word_info in word_infos_to_word_infos:
                if this_word_info.relative_start_index > other_word_info.relative_start_index and \
                        this_word_info.relative_end_index <= other_word_info.relative_end_index:
                    return other_word_info
                if this_word_info.relative_start_index >= other_word_info.relative_start_index and\
                        this_word_info.relative_end_index < other_word_info.relative_end_index:
                    return other_word_info
            return None

        topic_matches = self.topic_match_documents_against(text_to_match)
        topic_match_dicts = []
        for topic_match_counter in range(0, len(topic_matches)):
            topic_match = topic_matches[topic_match_counter]
            doc = self.indexed_documents[topic_match.document_label].doc
            sentences_character_start_index_in_document = doc[topic_match.sentences_start_index].idx
            sentences_character_end_index_in_document = doc[topic_match.sentences_end_index].idx + \
                    len(doc[topic_match.sentences_end_index].text)
            word_infos_to_word_infos = {}
            for match in topic_match.structural_matches:
                for word_match in match.word_matches:
                    if word_match.document_subword != None:
                        subword = word_match.document_subword
                        relative_start_index = doc[subword.containing_token_index].idx + \
                                subword.char_start_index - \
                                sentences_character_start_index_in_document
                        relative_end_index = relative_start_index + len(subword.text)
                    else:
                        relative_start_index = word_match.first_document_token.idx - \
                                sentences_character_start_index_in_document
                        relative_end_index = word_match.last_document_token.idx + \
                                len(word_match.last_document_token.text) - \
                                sentences_character_start_index_in_document
                    if match.is_overlapping_relation:
                        word_info = WordInfo(relative_start_index, relative_end_index,
                                'overlapping_relation', word_match.explain())
                    elif match.from_single_word_phraselet:
                        word_info = WordInfo(relative_start_index, relative_end_index,
                                'single', word_match.explain())
                    else:
                        word_info = WordInfo(relative_start_index, relative_end_index,
                                'relation', word_match.explain())
                    if word_info in word_infos_to_word_infos:
                        existing_word_info = word_infos_to_word_infos[word_info]
                        if not existing_word_info.type == 'overlapping_relation':
                            if match.is_overlapping_relation:
                                existing_word_info.type = 'overlapping_relation'
                            elif not match.from_single_word_phraselet:
                                existing_word_info.type = 'relation'
                    else:
                        word_infos_to_word_infos[word_info] = word_info
            for word_info in list(word_infos_to_word_infos.keys()):
                if get_containing_word_info_key(word_infos_to_word_infos, word_info) != None:
                    del word_infos_to_word_infos[word_info]
            if topic_match.subword_index != None:
                subword = doc[topic_match.index_within_document]._.holmes.subwords\
                        [topic_match.subword_index]
                highest_activation_relative_start_index = \
                        doc[subword.containing_token_index].idx + \
                        subword.char_start_index - \
                        sentences_character_start_index_in_document
                highest_activation_relative_end_index = \
                        highest_activation_relative_start_index + len(subword.text)
            else:
                highest_activation_relative_start_index = \
                        doc[topic_match.index_within_document].idx - \
                        sentences_character_start_index_in_document
                highest_activation_relative_end_index = doc[topic_match.index_within_document].idx \
                        + len(doc[topic_match.index_within_document].text) - \
                        sentences_character_start_index_in_document
            highest_activation_word_info = WordInfo(highest_activation_relative_start_index,
                    highest_activation_relative_end_index, 'temp', 'temp')
            containing_word_info = get_containing_word_info_key(word_infos_to_word_infos,
                    highest_activation_word_info)
            if containing_word_info != None:
                highest_activation_word_info = containing_word_info
            word_infos_to_word_infos[highest_activation_word_info].is_highest_activation=True
            word_infos = sorted(word_infos_to_word_infos.values(), key=lambda
                    word_info:(word_info.relative_start_index, word_info.relative_end_index))
            topic_match_dict = {
                'document_label': topic_match.document_label,
                'text': topic_match.text,
                'text_to_match': text_to_match,
                'rank': str(topic_match_counter + 1),   # ties are corrected by
                                                        # TopicMatchDictionaryOrderer
                'sentences_character_start_index_in_document':
                        sentences_character_start_index_in_document,
                'sentences_character_end_index_in_document':
                        sentences_character_end_index_in_document,
                'score': topic_match.score,
                'word_infos': [[word_info.relative_start_index, word_info.relative_end_index,
                        word_info.type, word_info.is_highest_activation, word_info.explanation]
                        for word_info in word_infos]
                # The word infos are labelled by array index alone to prevent the JSON from
                # becoming too bloated
            }
            topic_match_dicts.append(topic_match_dict)
        return TopicMatchDictionaryOrderer().order(topic_match_dicts, self.number_of_results,
                tied_result_quotient)

class TopicMatchDictionaryOrderer:
    # extracted into its own class to facilite use by MultiprocessingManager

    def order(self, topic_match_dicts, number_of_results, tied_result_quotient):

        topic_match_dicts = sorted(topic_match_dicts, key=lambda dict: (0-dict['score'],
        0-len(dict['text'].split()), dict['document_label'], dict['word_infos'][0][0]))
        topic_match_dicts = topic_match_dicts[0:number_of_results]
        topic_match_counter = 0
        while topic_match_counter < len(topic_match_dicts):
            topic_match_dicts[topic_match_counter]['rank'] = str(topic_match_counter + 1)
            following_topic_match_counter = topic_match_counter + 1
            while following_topic_match_counter < len(topic_match_dicts) and \
                    topic_match_dicts[following_topic_match_counter]['score'] / topic_match_dicts[
                    topic_match_counter]['score'] > tied_result_quotient:
                working_rank = ''.join((str(topic_match_counter + 1), '='))
                topic_match_dicts[topic_match_counter]['rank'] = working_rank
                topic_match_dicts[following_topic_match_counter]['rank'] = working_rank
                following_topic_match_counter += 1
            topic_match_counter = following_topic_match_counter
        return topic_match_dicts


class SupervisedTopicTrainingUtils:

    def __init__(self, overlap_memory_size, oneshot):
        self.overlap_memory_size = overlap_memory_size
        self.oneshot = oneshot

    def get_labels_to_classification_frequencies_dict(self, *, matches,
            labels_to_classifications_dict):
        """ Builds a dictionary from search phrase (phraselet) labels to classification
            frequencies. Depending on the training phase, which is signalled by the parameters, the
            dictionary tracks either raw frequencies for each search phrase label or points to a
            second dictionary from classification labels to frequencies.

            Parameters:

            matches -- the structural matches from which to build the dictionary
            labels_to_classifications_dict -- a dictionary from document labels to document
                classifications, or 'None' if the target dictionary should contain raw frequencies.
        """
        def increment(search_phrase_label, document_label):
            if labels_to_classifications_dict != None:
                if search_phrase_label not in labels_to_frequencies_dict:
                    classification_frequency_dict = {}
                    labels_to_frequencies_dict[search_phrase_label] = classification_frequency_dict
                else:
                    classification_frequency_dict = labels_to_frequencies_dict[search_phrase_label]
                classification = labels_to_classifications_dict[document_label]
                if classification in classification_frequency_dict:
                    classification_frequency_dict[classification] += 1
                else:
                    classification_frequency_dict[classification] = 1
            else:
                if search_phrase_label not in labels_to_frequencies_dict:
                    labels_to_frequencies_dict[search_phrase_label] = 1
                else:
                    labels_to_frequencies_dict[search_phrase_label] += 1

        def relation_match_involves_whole_word_containing_subwords(match):
            # Where there are subwords, we suppress relation matches with the
            # entire word. The same rule is not applied to single-word matches because
            # it still makes sense to track words with more than three subwords.
            return len(match.word_matches) > 1 and \
                    len([word_match for word_match in match.word_matches if
                    len(word_match.document_token._.holmes.subwords) > 0 and
                    word_match.document_subword == None]) > 0

        labels_to_frequencies_dict = {}
        matches = [match for match in matches if not
                relation_match_involves_whole_word_containing_subwords(match)]
        matches = sorted(matches,
                key=lambda match:(match.document_label, match.index_within_document,
                        match.get_subword_index_for_sorting()))
        for index, match in enumerate(matches):
            if self.oneshot:
                if ('this_document_label' not in locals()) or \
                        this_document_label != match.document_label:
                    this_document_label = match.document_label
                    search_phrases_added_for_this_document = set()
                if match.search_phrase_label not in search_phrases_added_for_this_document:
                    increment(match.search_phrase_label, match.document_label)
                    search_phrases_added_for_this_document.add(match.search_phrase_label)
            else:
                increment(match.search_phrase_label, match.document_label)
            if not match.from_single_word_phraselet:
                previous_match_index = index
                number_of_analyzed_matches_counter = 0
                while previous_match_index > 0 and number_of_analyzed_matches_counter \
                        <= self.overlap_memory_size:
                    previous_match_index -= 1
                    previous_match = matches[previous_match_index]
                    if previous_match.document_label != match.document_label:
                        break
                    if previous_match.from_single_word_phraselet:
                        continue
                    if previous_match.search_phrase_label == match.search_phrase_label:
                        continue # otherwise coreference resolution leads to phrases being
                                 # combined with themselves
                    number_of_analyzed_matches_counter += 1
                    previous_word_match_doc_indexes = [word_match.get_document_index() for
                            word_match in previous_match.word_matches]
                    for word_match in match.word_matches:
                        if word_match.get_document_index() in previous_word_match_doc_indexes:
                            # the same word is involved in both matches, so combine them
                            # into a new label
                            label_parts = sorted((previous_match.search_phrase_label,
                                    match.search_phrase_label))
                            combined_label = '/'.join((label_parts[0],
                                    label_parts[1]))
                            if self.oneshot:
                                if combined_label not in search_phrases_added_for_this_document:
                                    increment(combined_label, match.document_label)
                                    search_phrases_added_for_this_document.add(combined_label)
                            else:
                                increment(combined_label, match.document_label)
        return labels_to_frequencies_dict

    def record_matches(self, *, phraselet_labels_to_search_phrases, structural_matcher,
            sorted_label_dict, doc_label, doc, matrix, row_index, verbose):
        """ Matches a document against the currently stored phraselets and records the matches
            in a matrix.

            Parameters:

            phraselet_labels_to_search_phrases -- a dictionary from search phrase (phraselet)
                labels to search phrase objects.
            structural_matcher -- the structural matcher to use for comparisons.
            sorted_label_dict -- a dictionary from search phrase (phraselet) labels to their own
                alphabetic sorting indexes.
            doc_label -- the document label, or 'None' if there is none.
            doc -- the document to be matched.
            matrix -- the matrix within which to record the matches.
            row_index -- the row number within the matrix corresponding to the document.
            verbose -- if 'True', matching information is outputted to the console.
        """
        indexed_document = structural_matcher.index_document(doc)
        indexed_documents = {doc_label:indexed_document}
        found = False
        for label, occurrences in \
                self.get_labels_to_classification_frequencies_dict(
                matches = structural_matcher.match(
                indexed_documents = indexed_documents,
                search_phrases = phraselet_labels_to_search_phrases.values(),
                output_document_matching_message_to_console = verbose,
                match_depending_on_single_words = None,
                compare_embeddings_on_root_words = False,
                compare_embeddings_on_non_root_words = True,
                document_labels_to_indexes_for_reverse_matching_sets = None,
                document_labels_to_indexes_for_embedding_reverse_matching_sets = None),
                labels_to_classifications_dict=None).items():
            if self.oneshot:
                occurrences = 1
            if label in sorted_label_dict: # may not be the case for compound labels
                label_index = sorted_label_dict[label]
                matrix[row_index, label_index] = occurrences
                found = True
        return found

class SupervisedTopicTrainingBasis:
    """ Holder object for training documents and their classifications from which one or more
        'SupervisedTopicModelTrainer' objects can be derived. This class is *NOT* threadsafe.
    """
    def __init__(self, *, structural_matcher, classification_ontology, overlap_memory_size,
            oneshot, match_all_words, verbose):
        """ Parameters:

            structural_matcher -- the structural matcher to use.
            classification_ontology -- an Ontology object incorporating relationships between
                classification labels.
            overlap_memory_size -- how many non-word phraselet matches to the left should be
                checked for words in common with a current match.
            oneshot -- whether the same word or relationship matched multiple times should be
                counted once only (value 'True') or multiple times (value 'False')
            match_all_words -- whether all single words should be taken into account
                (value 'True') or only single words with noun tags (value 'False')
            verbose -- if 'True', information about training progress is outputted to the console.
        """
        self.semantic_analyzer = structural_matcher.semantic_analyzer
        self.structural_matcher = structural_matcher
        self.classification_ontology = classification_ontology
        self._utils = SupervisedTopicTrainingUtils(overlap_memory_size, oneshot)
        self._match_all_words = match_all_words
        self.verbose = verbose

        self.training_documents = {}
        self.training_documents_labels_to_classifications_dict = {}
        self.additional_classification_labels = set()
        self.classification_implication_dict = {}
        self.labels_to_classification_frequencies = None
        self.phraselet_labels_to_phraselet_infos = {}

    def parse_and_register_training_document(self, text, classification, label=None):
        """ Parses and registers a document to use for training.

            Parameters:

            text -- the document text
            classification -- the classification label
            label -- a label with which to identify the document in verbose training output,
                or 'None' if a random label should be assigned.
        """
        self.register_training_document(self.semantic_analyzer.parse(text), classification, label)

    def register_training_document(self, doc, classification, label):
        """ Registers a pre-parsed document to use for training.

            Parameters:

            doc -- the document
            classification -- the classification label
            label -- a label with which to identify the document in verbose training output,
                or 'None' if a random label should be assigned.
        """
        if self.labels_to_classification_frequencies != None:
            raise RuntimeError(
                    "register_training_document() may not be called once prepare() has been called")
        if label == None:
            label = str(uuid.uuid4())
        if label in self.training_documents:
            raise DuplicateDocumentError(label)
        if self.verbose:
            print('Registering document', label)
        indexed_document = self.structural_matcher.index_document(doc)
        self.training_documents[label] = indexed_document
        self.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_phraselet_infos=
                self.phraselet_labels_to_phraselet_infos,
                replace_with_hypernym_ancestors=True,
                match_all_words=self._match_all_words,
                returning_serialized_phraselets = True,
                ignore_relation_phraselets=False,
                include_reverse_only=False,
                stop_lemmas = self.semantic_analyzer.\
                supervised_document_classification_phraselet_stop_lemmas,
                reverse_only_parent_lemmas = None)
        self.training_documents_labels_to_classifications_dict[label] = classification

    def register_additional_classification_label(self, label):
        """ Register an additional classification label which no training document has explicitly
            but that should be assigned to documents whose explicit labels are related to the
            additional classification label via the classification ontology.
        """
        if self.labels_to_classification_frequencies != None:
            raise RuntimeError(
                    "register_additional_classification_label() may not be called once prepare() has been called")
        if self.classification_ontology != None and self.classification_ontology.contains(label):
            self.additional_classification_labels.add(label)

    def prepare(self):
        """ Matches the phraselets derived from the training documents against the training
            documents to generate frequencies that also include combined labels, and examines the
            explicit classification labels, the additional classification labels and the
            classification ontology to derive classification implications.

            Once this method has been called, the instance no longer accepts new training documents
            or additional classification labels.
        """
        if self.labels_to_classification_frequencies != None:
            raise RuntimeError(
                    "prepare() may only be called once")
        if self.verbose:
            print('Matching documents against all phraselets')
        search_phrases = self.structural_matcher.create_search_phrases_from_phraselet_infos(self.
                phraselet_labels_to_phraselet_infos.values()).values()
        self.labels_to_classification_frequencies = self._utils.\
                get_labels_to_classification_frequencies_dict(
                matches = self.structural_matcher.match(
                indexed_documents = self.training_documents,
                search_phrases = search_phrases,
                output_document_matching_message_to_console = self.verbose,
                match_depending_on_single_words = None,
                compare_embeddings_on_root_words = False,
                compare_embeddings_on_non_root_words = True,
                document_labels_to_indexes_for_reverse_matching_sets = None,
                document_labels_to_indexes_for_embedding_reverse_matching_sets = None),
                labels_to_classifications_dict=
                self.training_documents_labels_to_classifications_dict)
        self.classifications = \
                sorted(set(self.training_documents_labels_to_classifications_dict.values()
                        ).union(self.additional_classification_labels))
        if len(self.classifications) < 2:
            raise FewerThanTwoClassificationsError(len(self.classifications))
        if self.classification_ontology != None:
            for parent in self.classifications:
                for child in self.classifications:
                    if self.classification_ontology.matches(parent, child):
                        if child in self.classification_implication_dict.keys():
                            self.classification_implication_dict[child].append(parent)
                        else:
                            self.classification_implication_dict[child] = [parent]

    def train(self, *, minimum_occurrences=4, cv_threshold=1.0, mlp_activation='relu',
            mlp_solver='adam', mlp_learning_rate='constant', mlp_learning_rate_init=0.001,
            mlp_max_iter=200, mlp_shuffle=True, mlp_random_state=42, overlap_memory_size=10,
            hidden_layer_sizes=None):
        """ Trains a model based on the prepared state.

            Parameters:

            minimum_occurrences -- the minimum number of times a word or relationship has to
                occur in the context of at least one single classification for the phraselet
                to be accepted into the final model.
            cv_threshold -- the minimum coefficient of variation a word or relationship has
                to occur with respect to explicit classification labels for the phraselet to be
                accepted into the final model.
            mlp_* -- see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html.
            overlap_memory_size -- how many non-word phraselet matches to the left should be
                checked for words in common with a current match.
            hidden_layer_sizes -- a tuple containing the number of neurons in each hidden layer, or
                'None' if the topology should be determined automatically.
        """

        if self.labels_to_classification_frequencies == None:
            raise RuntimeError(
                    "train() may only be called after prepare() has been called")
        return SupervisedTopicModelTrainer(
                training_basis = self,
                semantic_analyzer = self.semantic_analyzer,
                structural_matcher = self.structural_matcher,
                labels_to_classification_frequencies = self.labels_to_classification_frequencies,
                phraselet_infos = self.phraselet_labels_to_phraselet_infos.values(),
                minimum_occurrences = minimum_occurrences,
                cv_threshold = cv_threshold,
                mlp_activation = mlp_activation,
                mlp_solver = mlp_solver,
                mlp_learning_rate = mlp_learning_rate,
                mlp_learning_rate_init = mlp_learning_rate_init,
                mlp_max_iter = mlp_max_iter,
                mlp_shuffle = mlp_shuffle,
                mlp_random_state = mlp_random_state,
                hidden_layer_sizes = hidden_layer_sizes,
                utils = self._utils
        )

class SupervisedTopicModelTrainer:
    """ Worker object used to train and generate models. This class is *NOT* threadsafe."""

    def __init__(self, *, training_basis, semantic_analyzer, structural_matcher,
            labels_to_classification_frequencies, phraselet_infos, minimum_occurrences,
            cv_threshold, mlp_activation, mlp_solver, mlp_learning_rate, mlp_learning_rate_init,
            mlp_max_iter, mlp_shuffle, mlp_random_state, hidden_layer_sizes, utils):

        self._utils = utils
        self._semantic_analyzer = semantic_analyzer
        self._structural_matcher = structural_matcher
        self._training_basis = training_basis
        self._minimum_occurrences = minimum_occurrences
        self._cv_threshold = cv_threshold
        self._labels_to_classification_frequencies, self._phraselet_infos = self._filter(
                labels_to_classification_frequencies, phraselet_infos)

        if len(self._phraselet_infos) == 0:
            raise NoPhraseletsAfterFilteringError(''.join(('minimum_occurrences: ',
                    str(minimum_occurrences), '; cv_threshold: ', str(cv_threshold))))

        phraselet_labels_to_search_phrases = \
                self._structural_matcher.create_search_phrases_from_phraselet_infos(
                self._phraselet_infos)
        self._sorted_label_dict = {}
        for index, label in enumerate(sorted(self._labels_to_classification_frequencies.keys())):
            self._sorted_label_dict[label] = index
        self._input_matrix = dok_matrix((len(self._training_basis.training_documents),
                len(self._sorted_label_dict)))
        self._output_matrix = dok_matrix((len(self._training_basis.training_documents),
                len(self._training_basis.classifications)))

        if self._training_basis.verbose:
            print('Matching documents against filtered phraselets')
        for index, document_label in enumerate(sorted(self._training_basis.
                training_documents.keys())):
            self._utils.record_matches(
                    structural_matcher = self._structural_matcher,
                    phraselet_labels_to_search_phrases =
                            phraselet_labels_to_search_phrases,
                    sorted_label_dict = self._sorted_label_dict,
                    doc_label = document_label,
                    doc=self._training_basis.training_documents[document_label].doc,
                    matrix=self._input_matrix,
                    row_index=index,
                    verbose=self._training_basis.verbose)
            self._record_classifications_for_training(document_label, index)
        self._hidden_layer_sizes = hidden_layer_sizes
        if self._hidden_layer_sizes == None:
            start = len(self._sorted_label_dict)
            step = (len(self._training_basis.classifications) - len(self._sorted_label_dict)) / 3
            self._hidden_layer_sizes = (start, int(start+step), int(start+(2*step)))
        if self._training_basis.verbose:
            print('Hidden layer sizes:', self._hidden_layer_sizes)
        self._mlp = MLPClassifier(
                activation=mlp_activation,
                solver=mlp_solver,
                hidden_layer_sizes= self._hidden_layer_sizes,
                learning_rate = mlp_learning_rate,
                learning_rate_init = mlp_learning_rate_init,
                max_iter = mlp_max_iter,
                shuffle = mlp_shuffle,
                verbose = self._training_basis.verbose,
                random_state = mlp_random_state)
        self._mlp.fit(self._input_matrix, self._output_matrix)
        if self._training_basis.verbose and self._mlp.n_iter_ < mlp_max_iter:
            print('MLP neural network converged after', self._mlp.n_iter_, 'iterations.')

    def _filter(self, labels_to_classification_frequencies, phraselet_infos):
        """ Filters the phraselets in memory based on minimum_occurrences and cv_threshold. """

        accepted = 0
        under_minimum_occurrences = 0
        under_minimum_cv = 0
        new_labels_to_classification_frequencies = {}
        for label, classification_frequencies in labels_to_classification_frequencies.items():
            at_least_minimum = False
            working_classification_frequencies = classification_frequencies.copy()
            for classification in working_classification_frequencies:
                if working_classification_frequencies[classification] >= self._minimum_occurrences:
                    at_least_minimum = True
            if not at_least_minimum:
                under_minimum_occurrences += 1
                continue
            frequency_list = list(working_classification_frequencies.values())
            # We only want to take explicit classification labels into account, i.e. ignore the
            # classification ontology.
            number_of_classification_labels = \
                len(set(
                self._training_basis.training_documents_labels_to_classifications_dict.values()))
            frequency_list.extend([0] * number_of_classification_labels)
            frequency_list = frequency_list[:number_of_classification_labels]
            if statistics.pstdev(frequency_list) / statistics.mean(frequency_list) >= \
                    self._cv_threshold:
                accepted += 1
                new_labels_to_classification_frequencies[label] = classification_frequencies
            else:
                under_minimum_cv += 1
        if self._training_basis.verbose:
            print('Filtered: accepted', accepted, '; removed minimum occurrences',
                under_minimum_occurrences, '; removed cv threshold',
                under_minimum_cv)
        new_phraselet_infos = [phraselet_info for phraselet_info in phraselet_infos if
                phraselet_info.label in new_labels_to_classification_frequencies.keys()]
        return new_labels_to_classification_frequencies, new_phraselet_infos

    def _record_classifications_for_training(self, document_label, index):
        classification = \
                self._training_basis.training_documents_labels_to_classifications_dict[
                document_label]
        classification_index = self._training_basis.classifications.index(classification)
        self._output_matrix[index, classification_index] = 1
        if classification in self._training_basis.classification_implication_dict:
            for implied_classification in \
                    self._training_basis.classification_implication_dict[classification]:
                implied_classification_index = self._training_basis.classifications.index(
                        implied_classification)
                self._output_matrix[index, implied_classification_index] = 1

    def classifier(self):
        """ Returns a supervised topic classifier which contains no explicit references to the
            training data and that can be serialized.
        """
        self._mlp.verbose=False # we no longer require output once we are using the model
                                # to classify new documents
        model = SupervisedTopicClassifierModel(
                semantic_analyzer_model = self._semantic_analyzer.model,
                structural_matcher_ontology = self._structural_matcher.ontology,
                phraselet_infos = self._phraselet_infos,
                mlp = self._mlp,
                sorted_label_dict = self._sorted_label_dict,
                classifications = self._training_basis.classifications,
                overlap_memory_size = self._utils.overlap_memory_size,
                oneshot = self._utils.oneshot,
                analyze_derivational_morphology=
                self._structural_matcher.analyze_derivational_morphology)
        return SupervisedTopicClassifier(self._semantic_analyzer, self._structural_matcher,
                model, self._training_basis.verbose)

class SupervisedTopicClassifierModel:
    """ A serializable classifier model.

        Parameters:

        semantic_analyzer_model -- a string specifying the spaCy model with which this instance
            was generated and with which it must be used.
        structural_matcher_ontology -- the ontology used for matching documents against this model
            (not the classification ontology!)
        phraselet_infos -- the phraselets used for structural matching
        mlp -- the neural network
        sorted_label_dict -- a dictionary from search phrase (phraselet) labels to their own
            alphabetic sorting indexes.
        classifications -- an ordered list of classification labels corresponding to the
            neural network outputs
        overlap_memory_size -- how many non-word phraselet matches to the left should be
            checked for words in common with a current match.
        oneshot -- whether the same word or relationship matched multiple times should be
            counted once only (value 'True') or multiple times (value 'False')
        analyze_derivational_morphology -- the value of this manager parameter that was in force
            when the model was built. The same value has to be in force when the model is
            deserialized and reused.
    """

    def __init__(self, semantic_analyzer_model, structural_matcher_ontology,
            phraselet_infos, mlp, sorted_label_dict, classifications, overlap_memory_size,
            oneshot, analyze_derivational_morphology):
        self.semantic_analyzer_model = semantic_analyzer_model
        self.structural_matcher_ontology = structural_matcher_ontology
        self.phraselet_infos = phraselet_infos
        self.mlp = mlp
        self.sorted_label_dict = sorted_label_dict
        self.classifications = classifications
        self.overlap_memory_size = overlap_memory_size
        self.oneshot = oneshot
        self.analyze_derivational_morphology = analyze_derivational_morphology

class SupervisedTopicClassifier:
    """ Classifies new documents based on a pre-trained model."""

    def __init__(self, semantic_analyzer, structural_matcher, model, verbose):
        self._semantic_analyzer = semantic_analyzer
        self._structural_matcher = structural_matcher
        self._model = model
        self._verbose = verbose
        self._utils = SupervisedTopicTrainingUtils(model.overlap_memory_size, model.oneshot)
        if self._semantic_analyzer.model != model.semantic_analyzer_model:
            raise WrongModelDeserializationError(model.semantic_analyzer_model)
        if hasattr(model, 'analyze_derivational_morphology'): # backwards compatibility
            analyze_derivational_morphology = model.analyze_derivational_morphology
        else:
            analyze_derivational_morphology = False
        if self._structural_matcher.analyze_derivational_morphology != \
                analyze_derivational_morphology:
            print(''.join((
                    'manager: ', str(self._structural_matcher.analyze_derivational_morphology),
                    '; model: ', str(analyze_derivational_morphology))))
            raise IncompatibleAnalyzeDerivationalMorphologyDeserializationError(''.join((
                    'manager: ', str(self._structural_matcher.analyze_derivational_morphology),
                    '; model: ', str(analyze_derivational_morphology))))
        self._structural_matcher.ontology = model.structural_matcher_ontology
        self._structural_matcher.populate_ontology_reverse_derivational_dict()
        self._phraselet_labels_to_search_phrases = \
                self._structural_matcher.create_search_phrases_from_phraselet_infos(
                        model.phraselet_infos)

    def parse_and_classify(self, text):
        """ Returns a list containing zero, one or many document classifications. Where more
            than one classifications are returned, the labels are ordered by decreasing
            probability.

            Parameter:

            text -- the text to parse and classify.
        """
        return self.classify(self._semantic_analyzer.parse(text))

    def classify(self, doc):
        """ Returns a list containing zero, one or many document classifications. Where more
            than one classifications are returned, the labels are ordered by decreasing
            probability.

            Parameter:

            doc -- the pre-parsed document to classify.
        """

        if self._model == None:
            raise RuntimeError('No model defined')
        new_document_matrix = dok_matrix((1, len(self._model.sorted_label_dict)))
        if not self._utils.record_matches(structural_matcher=self._structural_matcher,
                phraselet_labels_to_search_phrases=self._phraselet_labels_to_search_phrases,
                sorted_label_dict=self._model.sorted_label_dict,
                doc=doc,
                doc_label = '',
                matrix=new_document_matrix,
                row_index=0,
                verbose=self._verbose):
            return []
        else:
            classification_indexes = self._model.mlp.predict(new_document_matrix).nonzero()[1]
            if len(classification_indexes) > 1:
                probabilities = self._model.mlp.predict_proba(new_document_matrix)
                classification_indexes = sorted(classification_indexes, key=lambda index:
                        1-probabilities[0,index])
            return list(map(lambda index:self._model.classifications[index],
                    classification_indexes))

    def serialize_model(self):
        return jsonpickle.encode(self._model)
