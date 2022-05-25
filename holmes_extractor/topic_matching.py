from typing import List, Set, Dict, Union, Any, Tuple, Optional, cast
from spacy.compat import Literal
from spacy.tokens import Doc
from thinc.types import Floats1d

from .word_matching.general import WordMatch
from .structural_matching import Match, StructuralMatcher
from .word_matching.embedding import EmbeddingWordMatchingStrategy
from .word_matching.entity_embedding import EntityEmbeddingWordMatchingStrategy
from .word_matching.question import QuestionWordMatchingStrategy
from .parsing import Index, CorpusWordPosition, PhraseletInfo, SearchPhrase


class TopicMatch:
    """A topic match between some text and part of a document. Note that the end indexes refer
        to the token in question rather than to the following token.

    Properties:

    document_label -- the document label.
    index_within_document -- the index of the token within the document where 'score' was achieved.
    subword_index -- the index of the subword within the token within the document where 'score'
            was achieved, or *None* if the match involved the whole word.
    start_index -- the token start index of the topic match within the document.
    end_index -- the token end index of the topic match within the document.
    sentences_start_index -- the token start index within the document of the sentence that contains
        'start_index'
    sentences_end_index -- the token end index within the document of the sentence that contains
        'end_index'
    score -- the similarity score of the topic match
    text -- the text between 'sentences_start_index' and 'sentences_end_index'
    structural_matches -- a list of `Match` objects that were used to derive this object.
    """

    def __init__(
        self,
        document_label: str,
        index_within_document: int,
        subword_index: Optional[int],
        start_index: int,
        end_index: int,
        sentences_start_index: int,
        sentences_end_index: int,
        score: float,
        text: str,
        structural_matches: List[Match],
    ) -> None:
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
    def relative_start_index(self) -> int:
        return self.start_index - self.sentences_start_index

    @property
    def relative_end_index(self) -> int:
        return self.end_index - self.sentences_start_index


class PhraseletActivationTracker:
    """Tracks the activation for a specific phraselet - the most recent score
    and the position within the document at which that score was calculated.
    """

    def __init__(self, position: int, score: float) -> None:
        self.position = position
        self.score = score


class PhraseletWordMatchInfo:
    def __init__(self):
        self.single_word_match_corpus_words: Set[CorpusWordPosition] = set()
        # The indexes at which the single word phraselet for this word was matched.

        self.phraselet_labels_to_parent_match_corpus_words: Dict[
            str, List[CorpusWordPosition]
        ] = {}
        # Dictionary from phraselets with this word as the parent to indexes where the
        # phraselet was matched.

        self.phraselet_labels_to_child_match_corpus_words: Dict[
            str, List[CorpusWordPosition]
        ] = {}
        # Dictionary from phraselets with this word as the child to indexes where the
        # phraselet was matched.

        self.parent_match_corpus_words_to_matches: Dict[
            CorpusWordPosition, List[Match]
        ] = {}
        # Dictionary from indexes where phraselets with this word as the parent were matched
        # to the match objects.

        self.child_match_corpus_words_to_matches: Dict[
            CorpusWordPosition, List[Match]
        ] = {}
        # Dictionary from indexes where phraselets with this word as the child were matched
        # to the match objects.


class TopicMatcher:
    """A topic matcher object. See manager.py for details of the properties."""

    def __init__(
        self,
        *,
        structural_matcher: StructuralMatcher,
        document_labels_to_documents: Dict[str, Doc],
        reverse_dict: Dict[str, List[CorpusWordPosition]],
        text_to_match: str,
        phraselet_labels_to_phraselet_infos: Dict[str, PhraseletInfo],
        phraselet_labels_to_search_phrases: Dict[str, SearchPhrase],
        maximum_activation_distance: int,
        overall_similarity_threshold: float,
        initial_question_word_overall_similarity_threshold: float,
        relation_score: int,
        reverse_only_relation_score: int,
        single_word_score: int,
        single_word_any_tag_score: int,
        initial_question_word_answer_score: int,
        initial_question_word_behaviour: Literal["process", "exclusive", "ignore"],
        different_match_cutoff_score: int,
        overlapping_relation_multiplier: float,
        embedding_penalty: float,
        ontology_penalty: float,
        relation_matching_frequency_threshold: float,
        embedding_matching_frequency_threshold: float,
        sideways_match_extent: int,
        only_one_result_per_document: bool,
        number_of_results: int,
        document_label_filter: str,
        use_frequency_factor: bool,
        entity_label_to_vector_dict: Dict[str, Floats1d]
    ) -> None:
        self.structural_matcher = structural_matcher
        self.semantic_matching_helper = structural_matcher.semantic_matching_helper
        self.document_labels_to_documents = document_labels_to_documents
        self.reverse_dict = reverse_dict
        self.text_to_match = text_to_match
        self.phraselet_labels_to_phraselet_infos = phraselet_labels_to_phraselet_infos
        self.phraselet_labels_to_search_phrases = phraselet_labels_to_search_phrases
        self.maximum_activation_distance = maximum_activation_distance
        self.overall_similarity_threshold = overall_similarity_threshold
        self.initial_question_word_overall_similarity_threshold = (
            initial_question_word_overall_similarity_threshold
        )
        self.relation_score = relation_score
        self.reverse_only_relation_score = reverse_only_relation_score
        self.single_word_score = single_word_score
        self.single_word_any_tag_score = single_word_any_tag_score
        self.initial_question_word_answer_score = initial_question_word_answer_score
        self.initial_question_word_behaviour = initial_question_word_behaviour
        self.different_match_cutoff_score = different_match_cutoff_score
        self.overlapping_relation_multiplier = overlapping_relation_multiplier
        self.embedding_penalty = embedding_penalty
        self.ontology_penalty = ontology_penalty
        self.relation_matching_frequency_threshold = (
            relation_matching_frequency_threshold
        )
        self.relation_matching_frequency_threshold = (
            relation_matching_frequency_threshold
        )
        self.embedding_matching_frequency_threshold = (
            embedding_matching_frequency_threshold
        )
        self.sideways_match_extent = sideways_match_extent
        self.only_one_result_per_document = only_one_result_per_document
        self.number_of_results = number_of_results
        self.document_label_filter = document_label_filter
        self.use_frequency_factor = use_frequency_factor
        self.words_to_phraselet_word_match_infos: Dict[str, PhraseletWordMatchInfo] = {}

        process_initial_question_words = initial_question_word_behaviour in (
            "process",
            "exclusive",
        )

        word_matching_strategies = (
            self.semantic_matching_helper.main_word_matching_strategies
            + self.semantic_matching_helper.ontology_word_matching_strategies
        )[:]
        if overall_similarity_threshold < 1.0 or (
            process_initial_question_words
            and initial_question_word_overall_similarity_threshold < 1.0
        ):
            word_matching_strategies.append(
                EmbeddingWordMatchingStrategy(
                    self.semantic_matching_helper,
                    structural_matcher.perform_coreference_resolution,
                    overall_similarity_threshold,
                    initial_question_word_overall_similarity_threshold
                    if process_initial_question_words
                    else overall_similarity_threshold,
                )
            )
            word_matching_strategies.append(
                EntityEmbeddingWordMatchingStrategy(
                    self.semantic_matching_helper,
                    structural_matcher.perform_coreference_resolution,
                    overall_similarity_threshold,
                    initial_question_word_overall_similarity_threshold
                    if process_initial_question_words
                    else overall_similarity_threshold,
                    entity_label_to_vector_dict,
                )
            )
        if process_initial_question_words:
            word_matching_strategies.append(
                QuestionWordMatchingStrategy(
                    self.semantic_matching_helper,
                    structural_matcher.perform_coreference_resolution,
                    initial_question_word_overall_similarity_threshold,
                    entity_label_to_vector_dict,
                )
            )

        # First get single-word matches
        structural_matches = self.structural_matcher.match(
            word_matching_strategies=word_matching_strategies,
            document_labels_to_documents=self.document_labels_to_documents,
            reverse_dict=self.reverse_dict,
            search_phrases=phraselet_labels_to_search_phrases.values(),
            match_depending_on_single_words=True,
            compare_embeddings_on_root_words=False,
            compare_embeddings_on_non_root_words=False,
            reverse_matching_cwps=None,
            embedding_reverse_matching_cwps=None,
            process_initial_question_words=process_initial_question_words,
            overall_similarity_threshold=overall_similarity_threshold,
            initial_question_word_overall_similarity_threshold=initial_question_word_overall_similarity_threshold,
            document_label_filter=self.document_label_filter,
        )

        # Now get normally matched relations
        structural_matches.extend(
            self.structural_matcher.match(
                word_matching_strategies=word_matching_strategies,
                document_labels_to_documents=self.document_labels_to_documents,
                reverse_dict=self.reverse_dict,
                search_phrases=phraselet_labels_to_search_phrases.values(),
                match_depending_on_single_words=False,
                compare_embeddings_on_root_words=False,
                compare_embeddings_on_non_root_words=False,
                reverse_matching_cwps=None,
                embedding_reverse_matching_cwps=None,
                process_initial_question_words=process_initial_question_words,
                overall_similarity_threshold=overall_similarity_threshold,
                initial_question_word_overall_similarity_threshold=initial_question_word_overall_similarity_threshold,
                document_label_filter=self.document_label_filter,
            )
        )

        self.rebuild_document_info_dict(
            structural_matches, phraselet_labels_to_phraselet_infos
        )
        parent_direct_retry_corpus_word_positions: Set[CorpusWordPosition] = set()
        parent_embedding_retry_corpus_word_positions: Set[CorpusWordPosition] = set()
        child_embedding_retry_corpus_word_positions: Set[CorpusWordPosition] = set()
        for phraselet in (
            phraselet_labels_to_search_phrases[phraselet_info.label]
            for phraselet_info in phraselet_labels_to_phraselet_infos.values()
            if phraselet_info.child_lemma is not None
        ):
            self.add_indexes_for_reverse_matching(
                phraselet=phraselet,
                phraselet_info=phraselet_labels_to_phraselet_infos[phraselet.label],
                parent_direct_retry_corpus_word_positions=parent_direct_retry_corpus_word_positions,
                parent_embedding_retry_corpus_word_positions=parent_embedding_retry_corpus_word_positions,
                child_embedding_retry_corpus_word_positions=child_embedding_retry_corpus_word_positions,
            )
        if (
            len(parent_embedding_retry_corpus_word_positions) > 0
            or len(parent_direct_retry_corpus_word_positions) > 0
        ):
            # Perform reverse matching at selected indexes
            structural_matches.extend(
                self.structural_matcher.match(
                    word_matching_strategies=word_matching_strategies,
                    document_labels_to_documents=self.document_labels_to_documents,
                    reverse_dict=self.reverse_dict,
                    search_phrases=phraselet_labels_to_search_phrases.values(),
                    match_depending_on_single_words=False,
                    compare_embeddings_on_root_words=True,
                    compare_embeddings_on_non_root_words=False,
                    reverse_matching_cwps=parent_direct_retry_corpus_word_positions,
                    embedding_reverse_matching_cwps=parent_embedding_retry_corpus_word_positions,
                    process_initial_question_words=process_initial_question_words,
                    overall_similarity_threshold=overall_similarity_threshold,
                    initial_question_word_overall_similarity_threshold=initial_question_word_overall_similarity_threshold,
                    document_label_filter=self.document_label_filter,
                )
            )

        if len(child_embedding_retry_corpus_word_positions) > 0:
            # Retry normal matching at selected indexes with embedding-based matching on children
            structural_matches.extend(
                self.structural_matcher.match(
                    word_matching_strategies=word_matching_strategies,
                    document_labels_to_documents=self.document_labels_to_documents,
                    reverse_dict=self.reverse_dict,
                    search_phrases=phraselet_labels_to_search_phrases.values(),
                    match_depending_on_single_words=False,
                    compare_embeddings_on_root_words=False,
                    compare_embeddings_on_non_root_words=True,
                    reverse_matching_cwps=None,
                    embedding_reverse_matching_cwps=child_embedding_retry_corpus_word_positions,
                    process_initial_question_words=process_initial_question_words,
                    overall_similarity_threshold=overall_similarity_threshold,
                    initial_question_word_overall_similarity_threshold=initial_question_word_overall_similarity_threshold,
                    document_label_filter=self.document_label_filter,
                )
            )
        if (
            len(parent_direct_retry_corpus_word_positions) > 0
            or len(parent_embedding_retry_corpus_word_positions) > 0
            or len(child_embedding_retry_corpus_word_positions) > 0
        ):
            self.rebuild_document_info_dict(
                structural_matches, phraselet_labels_to_phraselet_infos
            )
        structural_matches = list(
            filter(self.filter_superfluous_matches, structural_matches)
        )
        phraselet_labels_to_frequency_factors = {
            info.label: info.frequency_factor
            for info in phraselet_labels_to_phraselet_infos.values()
        }
        position_sorted_structural_matches = sorted(
            structural_matches,
            key=lambda match: (
                match.document_label,
                match.index_within_document,
                match.get_subword_index_for_sorting(),
                len(
                    [
                        1
                        for wm in match.word_matches
                        if wm.search_phrase_token._.holmes.is_initial_question_word
                    ]
                )
                == 0,
                match.from_single_word_phraselet,
            ),
        )
        position_sorted_structural_matches = self.remove_duplicates(
            position_sorted_structural_matches
        )
        position_sorted_structural_matches = (
            self.remove_single_word_matches_made_superfluous_by_multiword_matches(
                position_sorted_structural_matches
            )
        )

        # Read through the documents measuring the activation based on where
        # in the document structural matches were found
        score_sorted_structural_matches = self.perform_activation_scoring(
            position_sorted_structural_matches,
            cast(Dict[str, float], phraselet_labels_to_frequency_factors),
        )
        self.topic_matches = self.generate_topic_matches(
            score_sorted_structural_matches, position_sorted_structural_matches
        )

    def get_phraselet_word_match_info(self, word: str) -> PhraseletWordMatchInfo:
        if word in self.words_to_phraselet_word_match_infos:
            return self.words_to_phraselet_word_match_infos[word]
        else:
            phraselet_word_match_info = PhraseletWordMatchInfo()
            self.words_to_phraselet_word_match_infos[word] = phraselet_word_match_info
            return phraselet_word_match_info

    def add_indexes_for_reverse_matching(
        self,
        *,
        phraselet: SearchPhrase,
        phraselet_info: PhraseletInfo,
        parent_direct_retry_corpus_word_positions: Set[CorpusWordPosition],
        parent_embedding_retry_corpus_word_positions: Set[CorpusWordPosition],
        child_embedding_retry_corpus_word_positions: Set[CorpusWordPosition]
    ) -> None:
        """
        parent_direct_retry_corpus_word_positions -- indexes where matching against a reverse
            matching phraselet should be attempted. These are ascertained by examining the child
            words.
        parent_embedding_retry_corpus_word_positions -- indexes where matching
            against a phraselet should be attempted with embedding-based matching on the
            parent (root) word. These are ascertained by examining the child words.
        child_embedding_retry_corpus_word_positions -- indexes where matching
            against a phraselet should be attempted with embedding-based matching on the
            child (non-root) word. These are ascertained by examining the parent words.
        """

        parent_token = phraselet.root_token
        parent_word = parent_token._.holmes.derived_lemma
        child_token = [
            token for token in phraselet.matchable_tokens if token.i != parent_token.i
        ][0]
        child_word = child_token._.holmes.derived_lemma
        if parent_word in self.words_to_phraselet_word_match_infos and (
            (
                not phraselet.reverse_only
                and not phraselet.treat_as_reverse_only_during_initial_relation_matching
            )
            or child_token._.holmes.has_initial_question_word_in_phrase
        ):
            parent_phraselet_word_match_info = self.words_to_phraselet_word_match_infos[
                parent_word
            ]
            parent_single_word_match_corpus_words = (
                parent_phraselet_word_match_info.single_word_match_corpus_words
            )
            if (
                phraselet.label
                in parent_phraselet_word_match_info.phraselet_labels_to_parent_match_corpus_words
            ):
                parent_relation_match_corpus_words = parent_phraselet_word_match_info.phraselet_labels_to_parent_match_corpus_words[
                    phraselet.label
                ]
            else:
                parent_relation_match_corpus_words = []
            if (
                cast(float, phraselet_info.parent_frequency_factor)
                >= self.embedding_matching_frequency_threshold
                or child_token._.holmes.has_initial_question_word_in_phrase
            ):
                child_embedding_retry_corpus_word_positions.update(
                    cwp
                    for cwp in parent_single_word_match_corpus_words.difference(
                        parent_relation_match_corpus_words
                    )
                )
        if child_word in self.words_to_phraselet_word_match_infos:
            child_phraselet_word_match_info = self.words_to_phraselet_word_match_infos[
                child_word
            ]
            child_single_word_match_corpus_words = (
                child_phraselet_word_match_info.single_word_match_corpus_words
            )
            if (
                phraselet.label
                in child_phraselet_word_match_info.phraselet_labels_to_child_match_corpus_words
            ):
                child_relation_match_corpus_words = child_phraselet_word_match_info.phraselet_labels_to_child_match_corpus_words[
                    phraselet.label
                ]
            else:
                child_relation_match_corpus_words = []

            if (
                cast(float, phraselet_info.child_frequency_factor)
                >= self.embedding_matching_frequency_threshold
                or parent_token._.holmes.has_initial_question_word_in_phrase
            ):
                set_to_add_to = parent_embedding_retry_corpus_word_positions
            elif cast(
                float, phraselet_info.child_frequency_factor
            ) >= self.relation_matching_frequency_threshold and (
                phraselet.reverse_only
                or phraselet.treat_as_reverse_only_during_initial_relation_matching
            ):
                set_to_add_to = parent_direct_retry_corpus_word_positions
            else:
                return
            linking_dependency = (
                parent_token._.holmes.get_label_of_dependency_with_child_index(
                    child_token.i
                )
            )
            for corpus_word_position in child_single_word_match_corpus_words.difference(
                child_relation_match_corpus_words
            ):
                doc = self.document_labels_to_documents[
                    corpus_word_position.document_label
                ]
                working_index = corpus_word_position.index
                working_token = doc[working_index.token_index]
                if (
                    not working_index.is_subword()
                    or working_token._.holmes.subwords[
                        working_index.subword_index
                    ].is_head
                ):
                    for (
                        parent_dependency
                    ) in working_token._.holmes.coreference_linked_parent_dependencies:
                        if self.semantic_matching_helper.dependency_labels_match(
                            search_phrase_dependency_label=linking_dependency,
                            document_dependency_label=parent_dependency[1],
                            inverse_polarity=False,
                        ):
                            working_index = Index(parent_dependency[0], None)
                            working_cwp = CorpusWordPosition(
                                corpus_word_position.document_label, working_index
                            )
                            set_to_add_to.add(working_cwp)
                    for (
                        child_dependency
                    ) in working_token._.holmes.coreference_linked_child_dependencies:
                        if (
                            self.structural_matcher.use_reverse_dependency_matching
                            and self.semantic_matching_helper.dependency_labels_match(
                                search_phrase_dependency_label=linking_dependency,
                                document_dependency_label=child_dependency[1],
                                inverse_polarity=True,
                            )
                        ):
                            working_index = Index(child_dependency[0], None)
                            working_cwp = CorpusWordPosition(
                                corpus_word_position.document_label, working_index
                            )
                            set_to_add_to.add(working_cwp)
                else:
                    working_subword = working_token._.holmes.subwords[
                        working_index.subword_index
                    ]
                    if self.semantic_matching_helper.dependency_labels_match(
                        search_phrase_dependency_label=linking_dependency,
                        document_dependency_label=working_subword.governing_dependency_label,
                        inverse_polarity=False,
                    ):
                        working_index = Index(
                            working_index.token_index, working_subword.governor_index
                        )
                        working_cwp = CorpusWordPosition(
                            corpus_word_position.document_label, working_index
                        )
                        set_to_add_to.add(working_cwp)
                    if (
                        self.structural_matcher.use_reverse_dependency_matching
                        and self.semantic_matching_helper.dependency_labels_match(
                            search_phrase_dependency_label=linking_dependency,
                            document_dependency_label=working_subword.dependency_label,
                            inverse_polarity=True,
                        )
                    ):
                        working_index = Index(
                            working_index.token_index, working_subword.dependent_index
                        )
                        working_cwp = CorpusWordPosition(
                            corpus_word_position.document_label, working_index
                        )
                        set_to_add_to.add(working_cwp)

    def rebuild_document_info_dict(
        self,
        matches: List[Match],
        phraselet_labels_to_phraselet_infos: Dict[str, PhraseletInfo],
    ) -> None:
        def process_word_match(
            match: Match, parent: bool  # 'True' -> parent, 'False' -> child
        ) -> None:
            word_match = self.get_word_match_from_match(match, parent)
            word = word_match.search_phrase_token._.holmes.derived_lemma
            phraselet_word_match_info = self.get_phraselet_word_match_info(word)
            corpus_word_position = CorpusWordPosition(
                match.document_label, word_match.get_document_index()
            )
            if parent:
                self.add_to_dict_list(
                    phraselet_word_match_info.parent_match_corpus_words_to_matches,
                    corpus_word_position,
                    match,
                )
                self.add_to_dict_list(
                    phraselet_word_match_info.phraselet_labels_to_parent_match_corpus_words,
                    match.search_phrase_label,
                    corpus_word_position,
                )
            else:
                self.add_to_dict_list(
                    phraselet_word_match_info.child_match_corpus_words_to_matches,
                    corpus_word_position,
                    match,
                )
                self.add_to_dict_list(
                    phraselet_word_match_info.phraselet_labels_to_child_match_corpus_words,
                    match.search_phrase_label,
                    corpus_word_position,
                )

        self.words_to_phraselet_word_match_infos = {}
        for match in matches:
            if match.from_single_word_phraselet:
                phraselet_info = phraselet_labels_to_phraselet_infos[
                    match.search_phrase_label
                ]
                word = phraselet_info.parent_derived_lemma
                phraselet_word_match_info = self.get_phraselet_word_match_info(word)
                word_match = match.word_matches[0]
                phraselet_word_match_info.single_word_match_corpus_words.add(
                    CorpusWordPosition(
                        match.document_label, word_match.get_document_index()
                    )
                )
            else:
                process_word_match(match, True)
                process_word_match(match, False)

    def filter_superfluous_matches(self, match: Match) -> bool:
        def get_other_matches_at_same_word(
            match: Match, parent: bool
        ) -> List[Match]:  # 'True' -> parent, 'False' -> child
            word_match = self.get_word_match_from_match(match, parent)
            word = word_match.search_phrase_token._.holmes.derived_lemma
            phraselet_word_match_info = self.get_phraselet_word_match_info(word)
            corpus_word_position = CorpusWordPosition(
                match.document_label, word_match.get_document_index()
            )
            if parent:
                match_dict = (
                    phraselet_word_match_info.parent_match_corpus_words_to_matches
                )
            else:
                match_dict = (
                    phraselet_word_match_info.child_match_corpus_words_to_matches
                )
            return match_dict[corpus_word_position]

        def check_for_sibling_match_with_higher_similarity(
            match: Match,
            other_match: Match,
            word_match: WordMatch,
            other_word_match: WordMatch,
        ) -> bool:
            # We do not want the same phraselet to match multiple siblings, so choose
            # the sibling that is most similar to the search phrase token.
            # Uses filter semantics, i.e. returns 'True' if 'match' should be retained.
            if self.overall_similarity_threshold == 1.0:
                return True
            if word_match.document_token.i == other_word_match.document_token.i:
                return True
            working_sibling = word_match.document_token.doc[
                word_match.document_token._.holmes.token_or_lefthand_sibling_index
            ]
            for sibling in working_sibling._.holmes.loop_token_and_righthand_siblings(
                word_match.document_token.doc
            ):
                if (
                    match.search_phrase_label == other_match.search_phrase_label
                    and other_word_match.document_token.i == sibling.i
                    and other_word_match.similarity_measure
                    > word_match.similarity_measure
                ):
                    return False
            return True

        def perform_checks_at_pole(
            match: Match, parent: bool
        ) -> bool:  # parent is 'True' -> parent, 'False' -> child
            # Uses filter semantics, i.e. returns 'True' if 'match' should be retained.

            this_this_pole_word_match = self.get_word_match_from_match(match, parent)
            this_pole_index = this_this_pole_word_match.document_token.i
            this_other_pole_word_match = self.get_word_match_from_match(
                match, not parent
            )
            for other_this_pole_match in get_other_matches_at_same_word(match, parent):
                other_other_pole_word_match = self.get_word_match_from_match(
                    other_this_pole_match, not parent
                )
                if this_other_pole_word_match.document_subword is not None:
                    this_other_pole_subword_index = (
                        this_other_pole_word_match.document_subword.index
                    )
                else:
                    this_other_pole_subword_index = None
                if other_other_pole_word_match.document_subword is not None:
                    other_other_pole_subword_index = (
                        other_other_pole_word_match.document_subword.index
                    )
                else:
                    other_other_pole_subword_index = None
                if (
                    this_other_pole_word_match.document_token.i
                    == other_other_pole_word_match.document_token.i
                    and this_other_pole_subword_index == other_other_pole_subword_index
                    and other_other_pole_word_match.similarity_measure
                    > this_other_pole_word_match.similarity_measure
                ):
                    # The other match has a higher similarity measure at the other pole than
                    # this match. The matched tokens are the same. The matching phraselets
                    # must be different.
                    return False
                if (
                    this_other_pole_word_match.document_token.i
                    == other_other_pole_word_match.document_token.i
                    and this_other_pole_subword_index is not None
                    and other_other_pole_subword_index is None
                ):
                    # This match is with a subword where the other match has matched the entire
                    # word, so this match should be removed.
                    return False
                    # Check unnecessary if parent==True as it has then already
                    # been carried out during structural matching.
                if (
                    not parent
                    and this_other_pole_word_match.document_token.i
                    != other_other_pole_word_match.document_token.i
                    and other_other_pole_word_match.document_token.i
                    in this_other_pole_word_match.document_token._.holmes.token_and_coreference_chain_indexes
                    and match.search_phrase_label
                    == other_this_pole_match.search_phrase_label
                    and (
                        (
                            abs(
                                this_pole_index
                                - this_other_pole_word_match.document_token.i
                            )
                            > abs(
                                this_pole_index
                                - other_other_pole_word_match.document_token.i
                            )
                        )
                        or (
                            abs(
                                this_pole_index
                                - this_other_pole_word_match.document_token.i
                            )
                            == abs(
                                this_pole_index
                                - other_other_pole_word_match.document_token.i
                            )
                            and this_other_pole_word_match.document_token.i
                            > other_other_pole_word_match.document_token.i
                        )
                    )
                ):
                    # The document tokens at the other poles corefer with each other and
                    # the other match's token is closer to the second document token (the
                    # one at this pole). Both matches are from the same phraselet.
                    # If the tokens from the two matches are the same distance from the document
                    # token at this pole but on opposite sides of it, the preceding one beats
                    # the succeeding one simply because we have to choose one or the other.
                    return False

                if not check_for_sibling_match_with_higher_similarity(
                    match,
                    other_this_pole_match,
                    this_other_pole_word_match,
                    other_other_pole_word_match,
                ):
                    return False
            return True

        if match.from_single_word_phraselet:
            return True
        if not perform_checks_at_pole(match, True):
            return False
        if not perform_checks_at_pole(match, False):
            return False
        return True

    def remove_single_word_matches_made_superfluous_by_multiword_matches(
        self, matches: List[Match]
    ) -> List[Match]:
        indexes_to_remove: Set[int] = set()
        for counter, match in enumerate(matches):
            match_word_match = match.word_matches[0]
            if (
                match.from_single_word_phraselet
                and match_word_match.first_document_token
                != match_word_match.last_document_token
            ):
                for inner_counter in range(counter + 1, len(matches)):
                    if matches[inner_counter].from_single_word_phraselet:
                        inner_match = matches[inner_counter]
                        inner_match_word_match = inner_match.word_matches[0]
                        if (
                            inner_match_word_match.first_document_token
                            > match_word_match.last_document_token
                            or inner_match_word_match.first_document_token
                            != inner_match_word_match.last_document_token
                        ):
                            break
                        indexes_to_remove.add(inner_counter)
                for inner_counter in range(counter - 1, -1, -1):
                    if matches[inner_counter].from_single_word_phraselet:
                        inner_match = matches[inner_counter]
                        inner_match_word_match = inner_match.word_matches[0]
                        if (
                            inner_match_word_match.last_document_token
                            < match_word_match.first_document_token
                            or inner_match_word_match.first_document_token
                            != inner_match_word_match.last_document_token
                        ):
                            break
                        indexes_to_remove.add(inner_counter)
        return [m for i, m in enumerate(matches) if i not in indexes_to_remove]

    def remove_duplicates(self, matches: List[Match]) -> List[Match]:
        # Situations where the same document tokens have been matched by multiple phraselets
        matches_to_return: List[Match] = []
        if len(matches) == 0:
            return matches_to_return
        else:
            matches_to_return.append(matches[0])
        if len(matches) > 1:
            previous_whole_word_single_word_match = None
            for counter in range(1, len(matches)):
                this_match = matches[counter]
                previous_match = matches[counter - 1]
                if this_match.document_label != previous_match.document_label:
                    matches_to_return.append(this_match)
                    continue
                if (
                    this_match.index_within_document
                    == previous_match.index_within_document
                ):
                    if (
                        previous_match.from_single_word_phraselet
                        and previous_match.get_subword_index() is None
                    ):
                        previous_whole_word_single_word_match = previous_match
                    if (
                        this_match.get_subword_index() is not None
                        and previous_whole_word_single_word_match is not None
                        and this_match.index_within_document
                        == previous_whole_word_single_word_match.index_within_document
                    ):
                        # This match is against a subword where the whole word has also been
                        # matched, so reject it
                        continue
                if len(this_match.word_matches) != len(previous_match.word_matches):
                    matches_to_return.append(this_match)
                else:
                    this_word_matches_indexes = [
                        word_match.get_document_index()
                        for word_match in this_match.word_matches
                    ]
                    previous_word_matches_indexes = [
                        word_match.get_document_index()
                        for word_match in previous_match.word_matches
                    ]
                    # In some circumstances the two phraselets may have matched the same
                    # tokens the opposite way round
                    if sorted(this_word_matches_indexes) != sorted(
                        previous_word_matches_indexes
                    ):
                        matches_to_return.append(this_match)
        return matches_to_return

    def get_word_match_from_match(self, match: Match, parent: bool) -> WordMatch:
        ## child if parent==False
        for word_match in match.word_matches:
            if parent == word_match.temp_is_parent:
                return word_match
        raise RuntimeError("".join(("Word match not found with parent==", str(parent))))

    def add_to_dict_list(self, dictionary: Dict, key: Any, value: Any) -> None:
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]

    def add_to_dict_set(self, dictionary: Dict, key: Any, value: Any) -> None:
        if not key in dictionary:
            dictionary[key] = set()
        dictionary[key].add(value)

    def perform_activation_scoring(
        self,
        position_sorted_structural_matches: List[Match],
        phraselet_labels_to_frequency_factors: Dict[str, float],
    ) -> List[Match]:
        """
        Read through the documents measuring the activation based on where
        in the document structural matches were found.
        """

        def get_set_from_dict(
            dictionary: Dict[CorpusWordPosition, Set[str]], key: CorpusWordPosition
        ) -> Set[str]:
            if key in dictionary:
                return dictionary[key]
            else:
                return set()

        def is_intcompound_match_within_same_document_word(match: Match) -> bool:
            # Where a relationship match involves subwords of the same word both on the
            # searched text and on the document side, it should receive the same activation as a
            # single-word match.
            return (
                match.search_phrase_label.startswith("intcompound")
                and len({wm.document_token.i for wm in match.word_matches}) == 1
            )

        def get_current_activation_for_phraselet(
            phraselet_activation_tracker: PhraseletActivationTracker, current_index: int
        ) -> float:
            distance_to_last_match = (
                current_index - phraselet_activation_tracker.position
            )
            tailoff_quotient = distance_to_last_match / self.maximum_activation_distance
            tailoff_quotient = min(tailoff_quotient, 1.0)
            return (1 - tailoff_quotient) * phraselet_activation_tracker.score

        document_labels_to_indexes_to_phraselet_labels: Dict[
            str, Dict[CorpusWordPosition, Set[str]]
        ] = {}
        for match in (
            match
            for match in position_sorted_structural_matches
            if not match.from_single_word_phraselet
            and not is_intcompound_match_within_same_document_word(match)
        ):
            if match.document_label in document_labels_to_indexes_to_phraselet_labels:
                inner_dict = document_labels_to_indexes_to_phraselet_labels[
                    match.document_label
                ]
            else:
                inner_dict = {}
                document_labels_to_indexes_to_phraselet_labels[
                    match.document_label
                ] = inner_dict
            parent_word_match = self.get_word_match_from_match(match, True)
            self.add_to_dict_set(
                inner_dict,
                parent_word_match.get_document_index(),
                match.search_phrase_label,
            )
            child_word_match = self.get_word_match_from_match(match, False)
            self.add_to_dict_set(
                inner_dict,
                child_word_match.get_document_index(),
                match.search_phrase_label,
            )
        current_document_label = None
        for pssm_index, match in enumerate(position_sorted_structural_matches):
            match.original_index_within_list = (  # type:ignore[attr-defined]
                pssm_index  # store for later use after resorting
            )
            if match.document_label != current_document_label or pssm_index == 0:
                current_document_label = match.document_label
                phraselet_labels_to_phraselet_activation_trackers: Dict[
                    str, PhraseletActivationTracker
                ] = {}
                indexes_to_phraselet_labels = (
                    document_labels_to_indexes_to_phraselet_labels.get(
                        current_document_label, {}
                    )
                )
            match.is_overlapping_relation = False  # type:ignore[attr-defined]
            if (
                match.from_single_word_phraselet
                or is_intcompound_match_within_same_document_word(match)
            ):
                this_match_score: float
                if match.from_topic_match_phraselet_created_without_matching_tags:
                    this_match_score = self.single_word_any_tag_score
                else:
                    this_match_score = self.single_word_score
            else:
                if match.from_reverse_only_topic_match_phraselet:
                    this_match_score = self.reverse_only_relation_score
                else:
                    this_match_score = self.relation_score
                for word_match in match.word_matches:
                    if (
                        word_match.search_phrase_token._.holmes.is_initial_question_word
                        or word_match.search_phrase_token._.holmes.has_initial_question_word_in_phrase
                    ) and not (
                        word_match.document_token._.holmes.is_initial_question_word
                        or word_match.document_token.tag_
                        in self.semantic_matching_helper.interrogative_pronoun_tags
                    ):
                        this_match_score = self.initial_question_word_answer_score
                this_match_parent_word_match = self.get_word_match_from_match(
                    match, True
                )
                this_match_parent_index = (
                    this_match_parent_word_match.get_document_index()
                )
                this_match_child_word_match = self.get_word_match_from_match(
                    match, False
                )
                this_match_child_index = (
                    this_match_child_word_match.get_document_index()
                )
                other_relevant_phraselet_labels = get_set_from_dict(
                    indexes_to_phraselet_labels, this_match_parent_index
                ) | get_set_from_dict(
                    indexes_to_phraselet_labels, this_match_child_index
                )
                other_relevant_phraselet_labels.remove(match.search_phrase_label)
                if len(other_relevant_phraselet_labels) > 0:
                    match.is_overlapping_relation = True  # type:ignore[attr-defined]
                    this_match_score *= self.overlapping_relation_multiplier

            if self.use_frequency_factor:
                # multiply the score by the frequency factor
                this_match_score *= phraselet_labels_to_frequency_factors[
                    match.search_phrase_label
                ]

            overall_similarity_measure = float(match.overall_similarity_measure)
            if overall_similarity_measure < 1.0:
                this_match_score *= self.embedding_penalty * overall_similarity_measure
            for word_match in (
                word_match
                for word_match in match.word_matches
                if word_match.word_match_type == "ontology"
            ):
                this_match_score *= self.ontology_penalty ** (abs(word_match.depth) + 1)

            if (
                match.search_phrase_label
                in phraselet_labels_to_phraselet_activation_trackers
            ):
                phraselet_activation_tracker = (
                    phraselet_labels_to_phraselet_activation_trackers[
                        match.search_phrase_label
                    ]
                )
                current_score = get_current_activation_for_phraselet(
                    phraselet_activation_tracker, match.index_within_document
                )
                if this_match_score > current_score:
                    phraselet_activation_tracker.score = this_match_score
                else:
                    phraselet_activation_tracker.score = current_score
                phraselet_activation_tracker.position = match.index_within_document
            else:
                phraselet_labels_to_phraselet_activation_trackers[
                    match.search_phrase_label
                ] = PhraseletActivationTracker(
                    match.index_within_document, this_match_score
                )
            match.topic_score = 0  # type:ignore[attr-defined]
            for phraselet_label in list(
                phraselet_labels_to_phraselet_activation_trackers
            ):
                phraselet_activation_tracker = (
                    phraselet_labels_to_phraselet_activation_trackers[phraselet_label]
                )
                current_activation = get_current_activation_for_phraselet(
                    phraselet_activation_tracker, match.index_within_document
                )
                if current_activation <= 0:
                    del phraselet_labels_to_phraselet_activation_trackers[
                        phraselet_label
                    ]
                else:
                    match.topic_score += current_activation  # type:ignore[attr-defined]
        return sorted(
            position_sorted_structural_matches,
            key=lambda match: 0 - match.topic_score,  # type:ignore[attr-defined]
        )

    def generate_topic_matches(
        self,
        score_sorted_structural_matches: List[Match],
        position_sorted_structural_matches: List[Match],
    ) -> List[TopicMatch]:
        """Resort the matches starting with the highest (most active) and
        create topic match objects with information about the surrounding sentences.
        """

        def match_contained_within_existing_topic_match(
            topic_matches: List[TopicMatch], match: Match
        ) -> bool:
            for topic_match in topic_matches:
                if (
                    match.document_label == topic_match.document_label
                    and match.index_within_document >= topic_match.start_index
                    and match.index_within_document <= topic_match.end_index
                ):
                    return True
            return False

        def alter_start_and_end_indexes_for_match(
            start_index: int, end_index: int, match: Match
        ) -> Tuple[int, int]:
            for word_match in match.word_matches:
                if word_match.first_document_token.i < start_index:
                    start_index = word_match.first_document_token.i
                if (
                    word_match.document_subword is not None
                    and word_match.document_subword.containing_token_index < start_index
                ):
                    start_index = word_match.document_subword.containing_token_index
                if word_match.last_document_token.i > end_index:
                    end_index = word_match.last_document_token.i
                if (
                    word_match.document_subword is not None
                    and word_match.document_subword.containing_token_index > end_index
                ):
                    end_index = word_match.document_subword.containing_token_index
            return start_index, end_index

        if self.only_one_result_per_document:
            existing_document_labels = []
        topic_matches: List[TopicMatch] = []
        counter = 0
        for score_sorted_match in score_sorted_structural_matches:
            if counter >= self.number_of_results:
                break
            if match_contained_within_existing_topic_match(
                topic_matches, score_sorted_match
            ):
                continue
            if (
                self.only_one_result_per_document
                and score_sorted_match.document_label in existing_document_labels
            ):
                continue
            start_index, end_index = alter_start_and_end_indexes_for_match(
                score_sorted_match.index_within_document,
                score_sorted_match.index_within_document,
                score_sorted_match,
            )
            previous_index_within_list = (
                score_sorted_match.original_index_within_list  # type:ignore[attr-defined]
            )
            while (
                previous_index_within_list > 0
                and position_sorted_structural_matches[
                    previous_index_within_list - 1
                ].document_label
                == score_sorted_match.document_label
                and position_sorted_structural_matches[
                    previous_index_within_list
                ].topic_score
                > self.different_match_cutoff_score
            ):
                # previous_index_within_list rather than previous_index_within_list -1 :
                # when a complex structure is matched, it will often begin with a single noun
                # that should be included within the topic match indexes
                if match_contained_within_existing_topic_match(
                    topic_matches,
                    position_sorted_structural_matches[previous_index_within_list - 1],
                ):
                    break
                if (
                    score_sorted_match.index_within_document
                    - position_sorted_structural_matches[
                        previous_index_within_list - 1
                    ].index_within_document
                    > self.sideways_match_extent
                ):
                    break
                previous_index_within_list -= 1
                start_index, end_index = alter_start_and_end_indexes_for_match(
                    start_index,
                    end_index,
                    position_sorted_structural_matches[previous_index_within_list],
                )
            next_index_within_list = (
                score_sorted_match.original_index_within_list  # type:ignore[attr-defined]
            )
            while (
                next_index_within_list + 1 < len(score_sorted_structural_matches)
                and position_sorted_structural_matches[
                    next_index_within_list + 1
                ].document_label
                == score_sorted_match.document_label
                and position_sorted_structural_matches[
                    next_index_within_list + 1
                ].topic_score
                >= self.different_match_cutoff_score
            ):
                if match_contained_within_existing_topic_match(
                    topic_matches,
                    position_sorted_structural_matches[next_index_within_list + 1],
                ):
                    break
                if (
                    position_sorted_structural_matches[
                        next_index_within_list + 1
                    ].index_within_document
                    - score_sorted_match.index_within_document
                    > self.sideways_match_extent
                ):
                    break
                next_index_within_list += 1
                start_index, end_index = alter_start_and_end_indexes_for_match(
                    start_index,
                    end_index,
                    position_sorted_structural_matches[next_index_within_list],
                )
            working_document = self.document_labels_to_documents[
                score_sorted_match.document_label
            ]
            relevant_sentences = [
                sentence
                for sentence in working_document.sents
                if sentence.end > start_index and sentence.start <= end_index
            ]
            sentences_start_index = relevant_sentences[0].start
            sentences_end_index = relevant_sentences[-1].end
            text = working_document[sentences_start_index:sentences_end_index].text
            topic_matches.append(
                TopicMatch(
                    score_sorted_match.document_label,
                    score_sorted_match.index_within_document,
                    score_sorted_match.get_subword_index(),
                    start_index,
                    end_index,
                    sentences_start_index,
                    sentences_end_index - 1,
                    score_sorted_match.topic_score,  # type:ignore[attr-defined]
                    text,
                    position_sorted_structural_matches[
                        previous_index_within_list : next_index_within_list + 1
                    ],
                )
            )
            if self.only_one_result_per_document:
                existing_document_labels.append(score_sorted_match.document_label)
            counter += 1
        # If two matches have the same score, order them by length
        return sorted(
            topic_matches,
            key=lambda topic_match: (
                0 - topic_match.score,
                topic_match.start_index - topic_match.end_index,
            ),
        )

    def get_topic_match_dictionaries(self):
        class WordInfo:
            def __init__(
                self,
                relative_start_index: int,
                relative_end_index: int,
                phraselet_match_type: Literal[
                    "derivation",
                    "direct",
                    "embedding",
                    "entity_embedding",
                    "entity",
                    "ontology",
                    "question",
                ],
                explanation: str,
            ):
                self.relative_start_index = relative_start_index
                self.relative_end_index = relative_end_index
                self.phraselet_match_type = phraselet_match_type
                self.explanation = explanation
                self.is_highest_activation = False

            def __eq__(self, other) -> bool:
                return (
                    isinstance(other, WordInfo)
                    and self.relative_start_index == other.relative_start_index
                    and self.relative_end_index == other.relative_end_index
                )

            def __hash__(self) -> int:
                return hash((self.relative_start_index, self.relative_end_index))

        def get_containing_word_info_key(
            word_infos_to_word_infos: Dict[WordInfo, WordInfo], this_word_info: WordInfo
        ) -> Optional[WordInfo]:
            for other_word_info in word_infos_to_word_infos:
                if (
                    this_word_info.relative_start_index
                    > other_word_info.relative_start_index
                    and this_word_info.relative_end_index
                    <= other_word_info.relative_end_index
                ):
                    return other_word_info
                if (
                    this_word_info.relative_start_index
                    >= other_word_info.relative_start_index
                    and this_word_info.relative_end_index
                    < other_word_info.relative_end_index
                ):
                    return other_word_info
            return None

        topic_match_dicts = []
        for topic_match_counter, topic_match in enumerate(self.topic_matches):
            doc = self.document_labels_to_documents[topic_match.document_label]
            sentences_character_start_index_in_document = doc[
                topic_match.sentences_start_index
            ].idx
            sentences_character_end_index_in_document = doc[
                topic_match.sentences_end_index
            ].idx + len(doc[topic_match.sentences_end_index].text)
            word_infos_to_word_infos = {}
            answers_set = set()
            for match in topic_match.structural_matches:
                for word_match in match.word_matches:
                    if word_match.document_subword is not None:
                        subword = word_match.document_subword
                        relative_start_index = (
                            doc[subword.containing_token_index].idx
                            + subword.char_start_index
                            - sentences_character_start_index_in_document
                        )
                        relative_end_index = relative_start_index + len(subword.text)
                    else:
                        relative_start_index = (
                            word_match.first_document_token.idx
                            - sentences_character_start_index_in_document
                        )
                        relative_end_index = (
                            word_match.last_document_token.idx
                            + len(word_match.last_document_token.text)
                            - sentences_character_start_index_in_document
                        )
                    if match.is_overlapping_relation:
                        word_info = WordInfo(
                            relative_start_index,
                            relative_end_index,
                            "overlapping_relation",
                            word_match.explanation,
                        )
                    elif match.from_single_word_phraselet:  # two subwords within word:
                        word_info = WordInfo(
                            relative_start_index,
                            relative_end_index,
                            "single",
                            word_match.explanation,
                        )
                    else:
                        word_info = WordInfo(
                            relative_start_index,
                            relative_end_index,
                            "relation",
                            word_match.explanation,
                        )
                    if (
                        word_match.search_phrase_token._.holmes.is_initial_question_word
                        or word_match.search_phrase_token._.holmes.has_initial_question_word_in_phrase
                    ) and not (
                        word_match.document_token._.holmes.is_initial_question_word
                        or word_match.document_token.tag_
                        in self.semantic_matching_helper.interrogative_pronoun_tags
                    ):
                        if word_match.document_subword is not None:
                            answer_relative_start_index = (
                                word_match.document_token.idx
                                - sentences_character_start_index_in_document
                            )
                            answer_relative_end_index = relative_end_index
                        else:
                            subtree_without_conjunction = self.semantic_matching_helper.get_subtree_list_for_question_answer(
                                word_match.document_token
                            )
                            answer_relative_start_index = (
                                subtree_without_conjunction[0].idx
                                - sentences_character_start_index_in_document
                            )
                            answer_relative_end_index = (
                                subtree_without_conjunction[-1].idx
                                + len(subtree_without_conjunction[-1].text)
                                - sentences_character_start_index_in_document
                            )
                        answers_set.add(
                            (answer_relative_start_index, answer_relative_end_index)
                        )
                    if word_info in word_infos_to_word_infos:
                        existing_word_info = word_infos_to_word_infos[word_info]
                        if (
                            not existing_word_info.phraselet_match_type
                            == "overlapping_relation"
                        ):
                            if match.is_overlapping_relation:
                                existing_word_info.phraselet_match_type = (
                                    "overlapping_relation"
                                )
                            elif not match.from_single_word_phraselet:
                                existing_word_info.phraselet_match_type = "relation"
                    else:
                        word_infos_to_word_infos[word_info] = word_info
            for word_info in list(word_infos_to_word_infos.keys()):
                if (
                    get_containing_word_info_key(word_infos_to_word_infos, word_info)
                    is not None
                ):
                    del word_infos_to_word_infos[word_info]
            if (
                self.initial_question_word_behaviour != "exclusive"
                or len(answers_set) > 0
            ):
                if topic_match.subword_index is not None:
                    subword = doc[topic_match.index_within_document]._.holmes.subwords[
                        topic_match.subword_index
                    ]
                    highest_activation_relative_start_index = (
                        doc[subword.containing_token_index].idx
                        + subword.char_start_index
                        - sentences_character_start_index_in_document
                    )
                    highest_activation_relative_end_index = (
                        highest_activation_relative_start_index + len(subword.text)
                    )
                else:
                    highest_activation_relative_start_index = (
                        doc[topic_match.index_within_document].idx
                        - sentences_character_start_index_in_document
                    )
                    highest_activation_relative_end_index = (
                        doc[topic_match.index_within_document].idx
                        + len(doc[topic_match.index_within_document].text)
                        - sentences_character_start_index_in_document
                    )
                highest_activation_word_info = WordInfo(
                    highest_activation_relative_start_index,
                    highest_activation_relative_end_index,
                    "temp",
                    "temp",
                )
                containing_word_info = get_containing_word_info_key(
                    word_infos_to_word_infos, highest_activation_word_info
                )
                if containing_word_info is not None:
                    highest_activation_word_info = containing_word_info
                word_infos_to_word_infos[
                    highest_activation_word_info
                ].is_highest_activation = True
                word_infos = sorted(
                    word_infos_to_word_infos.values(),
                    key=lambda word_info: (
                        word_info.relative_start_index,
                        word_info.relative_end_index,
                    ),
                )
                answers = list(answers_set)
                answers.sort(key=lambda answer: (answer[0], answer[1]))
                for answer in answers.copy():
                    if (
                        len(
                            [
                                1
                                for other_answer in answers
                                if other_answer[0] < answer[0]
                                and other_answer[1] >= answer[1]
                            ]
                        )
                        > 0
                    ):
                        answers.remove(answer)
                    elif (
                        len(
                            [
                                1
                                for other_answer in answers
                                if other_answer[0] == answer[0]
                                and other_answer[1] > answer[1]
                            ]
                        )
                        > 0
                    ):
                        answers.remove(answer)
                topic_match_dict = {
                    "document_label": topic_match.document_label,
                    "text": topic_match.text,
                    "text_to_match": self.text_to_match,
                    "rank": str(topic_match_counter + 1),  # ties are corrected by
                    # TopicMatchDictionaryOrderer
                    "index_within_document": topic_match.index_within_document,
                    "subword_index": topic_match.subword_index,
                    "start_index": topic_match.start_index,
                    "end_index": topic_match.end_index,
                    "sentences_start_index": topic_match.sentences_start_index,
                    "sentences_end_index": topic_match.sentences_end_index,
                    "sentences_character_start_index": sentences_character_start_index_in_document,
                    "sentences_character_end_index": sentences_character_end_index_in_document,
                    "score": topic_match.score,
                    "word_infos": [
                        [
                            word_info.relative_start_index,
                            word_info.relative_end_index,
                            word_info.phraselet_match_type,
                            word_info.is_highest_activation,
                            word_info.explanation,
                        ]
                        for word_info in word_infos
                    ],
                    # The word infos are labelled by array index alone to prevent the JSON from
                    # becoming too bloated,
                    "answers": [[answer[0], answer[1]] for answer in answers],
                }
                topic_match_dicts.append(topic_match_dict)
        return topic_match_dicts


class TopicMatchDictionaryOrderer:
    # in its own class as it is called from the main process rather than from the workers

    def order(
        self,
        topic_match_dicts: List[Dict],
        number_of_results: int,
        tied_result_quotient: float,
    ) -> List[Dict[str, Union[str, int, float]]]:

        topic_match_dicts = sorted(
            topic_match_dicts,
            key=lambda dict: (
                0 - dict["score"],
                0 - len(dict["text"].split()),
                dict["document_label"],
                dict["word_infos"][0][0],
            ),
        )
        topic_match_dicts = topic_match_dicts[0:number_of_results]
        topic_match_counter = 0
        while topic_match_counter < len(topic_match_dicts):
            topic_match_dicts[topic_match_counter]["rank"] = str(
                topic_match_counter + 1
            )
            following_topic_match_counter = topic_match_counter + 1
            while (
                following_topic_match_counter < len(topic_match_dicts)
                and topic_match_dicts[following_topic_match_counter]["score"]
                / topic_match_dicts[topic_match_counter]["score"]
                > tied_result_quotient
            ):
                working_rank = "".join((str(topic_match_counter + 1), "="))
                topic_match_dicts[topic_match_counter]["rank"] = working_rank
                topic_match_dicts[following_topic_match_counter]["rank"] = working_rank
                following_topic_match_counter += 1
            topic_match_counter = following_topic_match_counter
        return topic_match_dicts
