from typing import List, Dict, Set, Sequence, Optional, Any, ValuesView, Union
import sys
from spacy.tokens import Doc, Token
from .parsing import (
    CorpusWordPosition,
    Index,
    SearchPhrase,
    SemanticMatchingHelper,
)
from .word_matching.general import WordMatch, WordMatchingStrategy


class Match:
    """A match between a search phrase and a document.

    Properties:

    word_matches -- a list of *WordMatch* objects.
    is_negated -- *True* if this match is negated.
    is_uncertain -- *True* if this match is uncertain.
    involves_coreference -- *True* if this match was found using coreference resolution.
    search_phrase_label -- the label of the search phrase that matched.
    search_phrase_text -- the text of the search phrase that matched.
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

    def __init__(
        self,
        search_phrase_label: str,
        search_phrase_text: str,
        document_label: str,
        from_single_word_phraselet: bool,
        from_topic_match_phraselet_created_without_matching_tags: bool,
        from_reverse_only_topic_match_phraselet: bool,
    ) -> None:
        self.word_matches: List[WordMatch] = []
        self.is_negated: bool = False
        self.is_uncertain: bool = False
        self.search_phrase_label = search_phrase_label
        self.search_phrase_text = search_phrase_text
        self.document_label = document_label
        self.from_single_word_phraselet = from_single_word_phraselet
        self.from_topic_match_phraselet_created_without_matching_tags = (
            from_topic_match_phraselet_created_without_matching_tags
        )
        self.from_reverse_only_topic_match_phraselet = (
            from_reverse_only_topic_match_phraselet
        )
        self.index_within_document: int = -1
        self.overall_similarity_measure = 1.0

    @property
    def involves_coreference(self) -> bool:
        for word_match in self.word_matches:
            if word_match.involves_coreference:
                return True
        return False

    def __copy__(self):
        match_to_return = Match(
            self.search_phrase_label,
            self.search_phrase_text,
            self.document_label,
            self.from_single_word_phraselet,
            self.from_topic_match_phraselet_created_without_matching_tags,
            self.from_reverse_only_topic_match_phraselet,
        )
        match_to_return.word_matches = self.word_matches.copy()
        match_to_return.is_negated = self.is_negated
        match_to_return.is_uncertain = self.is_uncertain
        match_to_return.index_within_document = self.index_within_document
        match_to_return.overall_similarity_measure = self.overall_similarity_measure
        return match_to_return

    def get_subword_index(self) -> Optional[int]:
        """Returns the subword index of the root token."""
        for word_match in self.word_matches:
            if word_match.search_phrase_token.dep_ == "ROOT" or (
                hasattr(word_match, "temp_is_parent") and word_match.temp_is_parent
            ):
                if word_match.document_subword is None:
                    return None
                return word_match.document_subword.index
        raise RuntimeError(
            "No word match with search phrase token with root dependency"
        )

    def get_subword_index_for_sorting(self) -> int:
        """Returns *-1* rather than *None* in the absence of a subword."""
        subword_index = self.get_subword_index()
        return subword_index if subword_index is not None else -1


class StructuralMatcher:
    """The class responsible for matching search phrases with documents."""

    def __init__(
        self,
        semantic_matching_helper: SemanticMatchingHelper,
        embedding_based_matching_on_root_words: bool,
        analyze_derivational_morphology: bool,
        perform_coreference_resolution: bool,
        use_reverse_dependency_matching: bool,
    ):
        """Args:

        semantic_matching_helper -- the *SemanticMatchingHelper* object to use
        embedding_based_matching_on_root_words -- *True* if embedding-based matching should be
            attempted on search-phrase root tokens
        analyze_derivational_morphology -- *True* if matching should be attempted between different
            words from the same word family. Defaults to *True*.
        perform_coreference_resolution -- *True* if coreference resolution should be taken into
            account when matching.
        use_reverse_dependency_matching -- *True* if appropriate dependencies in documents can be
            matched to dependencies in search phrases where the two dependencies point in opposite
            directions.
        """
        self.semantic_matching_helper = semantic_matching_helper
        self.embedding_based_matching_on_root_words = (
            embedding_based_matching_on_root_words
        )
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.perform_coreference_resolution = perform_coreference_resolution
        self.use_reverse_dependency_matching = use_reverse_dependency_matching

    def match(
        self,
        *,
        word_matching_strategies: List[WordMatchingStrategy],
        document_labels_to_documents: Dict[str, Doc],
        reverse_dict: Dict[str, List[CorpusWordPosition]],
        search_phrases: Union[List[SearchPhrase], ValuesView[SearchPhrase]],
        match_depending_on_single_words: Optional[bool],
        compare_embeddings_on_root_words: bool,
        compare_embeddings_on_non_root_words: bool,
        reverse_matching_cwps: Optional[Set[CorpusWordPosition]],
        embedding_reverse_matching_cwps: Optional[Set[CorpusWordPosition]],
        process_initial_question_words: bool,
        overall_similarity_threshold: float,
        initial_question_word_overall_similarity_threshold: float,
        document_label_filter: Optional[str] = None
    ) -> List[Match]:
        """Finds and returns matches between search phrases and documents.
        match_depending_on_single_words -- 'True' to match only single word search phrases,
            'False' to match only non-single-word search phrases and 'None' to match both.
        compare_embeddings_on_root_words -- if 'True', embeddings on root words are compared.
        compare_embeddings_on_non_root_words -- if 'True', embeddings on non-root words are
            compared.
        reverse_matching_cwps -- corpus word positions for non-embedding
            reverse matching only.
        embedding_reverse_matching_cwps -- corpus word positions for embedding
            and non-embedding reverse matching.
        process_initial_question_words -- 'True' if interrogative pronouns in search phrases should
            be matched to answering phrases in documents. Only used with topic matching.
        overall_similarity_threshold -- the overall similarity threshold for embedding-based
            matching.
        initial_question_word_overall_similarity_threshold -- the overall similarity threshold for
            embedding-based matching where the search phrase word has a dependent initial question
            word.
        document_label_filter -- a string with which the label of a document must begin for that
            document to be considered for matching, or 'None' if no filter is in use.
        """

        if (
            overall_similarity_threshold == 1.0
            and initial_question_word_overall_similarity_threshold == 1.0
        ):
            compare_embeddings_on_root_words = False
            compare_embeddings_on_non_root_words = False
        match_specific_indexes = (
            reverse_matching_cwps is not None
            or embedding_reverse_matching_cwps is not None
        )
        if reverse_matching_cwps is None:
            reverse_matching_cwps = set()
        if embedding_reverse_matching_cwps is None:
            embedding_reverse_matching_cwps = set()

        matches: List[Match] = []
        # Dictionary used to improve performance when embedding-based matching for root tokens
        # is active and there are multiple search phrases with the same root token word: the
        # same corpus word positions will then match all the search phrase root tokens.
        root_lemma_to_cwps_to_match_dict: Dict[str, Set[CorpusWordPosition]] = {}

        for search_phrase in search_phrases:
            if (
                not search_phrase.has_single_matchable_word
                and match_depending_on_single_words
            ):
                continue
            if (
                search_phrase.has_single_matchable_word
                and match_depending_on_single_words is False
            ):
                continue
            if not match_specific_indexes and (
                search_phrase.reverse_only
                or search_phrase.treat_as_reverse_only_during_initial_relation_matching
            ):
                continue
            if (
                self.semantic_matching_helper.get_entity_placeholder(
                    search_phrase.root_token
                )
                == "ENTITYNOUN"
            ):
                for document_label, doc in document_labels_to_documents.items():
                    for token in doc:
                        if token.pos_ in self.semantic_matching_helper.noun_pos:
                            matches.extend(
                                self.get_matches_starting_at_root_word_match(
                                    word_matching_strategies,
                                    search_phrase,
                                    doc,
                                    token,
                                    None,
                                    document_label,
                                    compare_embeddings_on_non_root_words,
                                    process_initial_question_words,
                                )
                            )
                continue
            direct_matching_cwps: List[CorpusWordPosition] = []
            matched_cwps: Set[CorpusWordPosition] = set()
            entity_label = self.semantic_matching_helper.get_entity_placeholder(
                search_phrase.root_token
            )
            if entity_label is not None:
                if entity_label in reverse_dict.keys():
                    entity_matching_cwps = reverse_dict[entity_label]
                    if match_specific_indexes:
                        entity_matching_cwps = [
                            cwp
                            for cwp in entity_matching_cwps
                            if cwp in reverse_matching_cwps
                            or cwp in embedding_reverse_matching_cwps
                            and not cwp.index.is_subword()
                        ]
                    matched_cwps.update(entity_matching_cwps)
            else:
                for word_matching_root_token in search_phrase.words_matching_root_token:
                    if word_matching_root_token in reverse_dict.keys():
                        direct_matching_cwps = reverse_dict[word_matching_root_token]
                        if match_specific_indexes:
                            direct_matching_cwps = [
                                cwp
                                for cwp in direct_matching_cwps
                                if cwp in reverse_matching_cwps
                                or cwp in embedding_reverse_matching_cwps
                            ]
                        matched_cwps.update(direct_matching_cwps)
            if (
                compare_embeddings_on_root_words
                and self.semantic_matching_helper.get_entity_placeholder(
                    search_phrase.root_token
                )
                is None
                and not search_phrase.reverse_only
                and self.semantic_matching_helper.embedding_matching_permitted(
                    search_phrase.root_token
                )
            ):
                if (
                    not search_phrase.topic_match_phraselet
                    and len(search_phrase.root_token._.holmes.lemma.split()) > 1
                ):
                    root_token_lemma_to_use = search_phrase.root_token.lemma_
                else:
                    root_token_lemma_to_use = search_phrase.root_token._.holmes.lemma
                if root_token_lemma_to_use in root_lemma_to_cwps_to_match_dict:
                    matched_cwps.update(
                        root_lemma_to_cwps_to_match_dict[root_token_lemma_to_use]
                    )
                else:
                    working_cwps_to_match_for_cache = set()
                    for document_word in reverse_dict:
                        corpus_word_positions_to_match = reverse_dict[document_word]
                        if match_specific_indexes:
                            corpus_word_positions_to_match = [
                                cwp
                                for cwp in corpus_word_positions_to_match
                                if cwp in embedding_reverse_matching_cwps
                                and cwp not in direct_matching_cwps
                            ]
                            if len(corpus_word_positions_to_match) == 0:
                                continue
                        search_phrase_vector = (
                            search_phrase.matchable_non_entity_tokens_to_vectors[
                                search_phrase.root_token.i
                            ]
                        )
                        example_cwp = corpus_word_positions_to_match[0]
                        example_doc = document_labels_to_documents[
                            example_cwp.document_label
                        ]
                        example_index = example_cwp.index
                        example_document_token = example_doc[example_index.token_index]
                        if example_index.is_subword():
                            if not self.semantic_matching_helper.embedding_matching_permitted(
                                example_document_token._.holmes.subwords[
                                    example_index.subword_index
                                ]
                            ):
                                continue
                            document_vector = example_document_token._.holmes.subwords[
                                example_index.subword_index
                            ].vector
                        else:
                            if not self.semantic_matching_helper.embedding_matching_permitted(
                                example_document_token
                            ):
                                continue
                            document_vector = example_document_token._.holmes.vector
                        if (
                            search_phrase_vector is not None
                            and document_vector is not None
                        ):
                            similarity_measure = (
                                self.semantic_matching_helper.cosine_similarity(
                                    search_phrase_vector, document_vector
                                )
                            )
                            search_phrase_initial_question_word = (
                                process_initial_question_words
                                and search_phrase.root_token._.holmes.has_initial_question_word_in_phrase
                            )
                            single_token_similarity_threshold = (
                                initial_question_word_overall_similarity_threshold
                                if search_phrase_initial_question_word
                                else overall_similarity_threshold
                            ) ** len(
                                search_phrase.matchable_non_entity_tokens_to_vectors
                            )
                            if similarity_measure >= single_token_similarity_threshold:
                                matched_cwps.update(corpus_word_positions_to_match)
                                working_cwps_to_match_for_cache.update(
                                    corpus_word_positions_to_match
                                )
                    root_lemma_to_cwps_to_match_dict[
                        root_token_lemma_to_use
                    ] = working_cwps_to_match_for_cache
            for corpus_word_position in matched_cwps:
                if (
                    document_label_filter is not None
                    and corpus_word_position.document_label is not None
                    and not corpus_word_position.document_label.startswith(
                        document_label_filter
                    )
                ):
                    continue
                doc = document_labels_to_documents[corpus_word_position.document_label]
                matches.extend(
                    self.get_matches_starting_at_root_word_match(
                        word_matching_strategies,
                        search_phrase,
                        doc,
                        doc[corpus_word_position.index.token_index],
                        corpus_word_position.index.subword_index,
                        corpus_word_position.document_label,
                        compare_embeddings_on_non_root_words,
                        process_initial_question_words,
                    )
                )
        return sorted(
            matches,
            key=lambda match: (
                1 - float(match.overall_similarity_measure),
                match.document_label,
                match.index_within_document,
            ),
        )

    def get_matches_starting_at_root_word_match(
        self,
        word_matching_strategies: List[WordMatchingStrategy],
        search_phrase: SearchPhrase,
        document: Doc,
        document_token: Token,
        document_subword_index: Optional[int],
        document_label: str,
        compare_embeddings_on_non_root_words: bool,
        process_initial_question_words: bool,
    ) -> List[Match]:
        """Begin recursive matching where a search phrase root token has matched a document
        token.
        """
        # array of sets to guard against endless looping during recursion. Each set
        # corresponds to the search phrase token with its index and contains the Index objects
        # for the document words for which a match to that search phrase token has been attempted.
        search_phrase_and_document_visited_table: List[Set[Index]] = [
            set() for token in search_phrase.doc
        ]
        word_match_dicts = self.match_recursively(
            word_matching_strategies=word_matching_strategies,
            search_phrase=search_phrase,
            search_phrase_token=search_phrase.root_token,
            document=document,
            document_token=document_token,
            document_subword_index=document_subword_index,
            search_phrase_and_document_visited_table=search_phrase_and_document_visited_table,
            is_uncertain=document_token._.holmes.is_uncertain,
            structurally_matched_document_token=document_token,
            compare_embeddings_on_non_root_words=compare_embeddings_on_non_root_words,
            process_initial_question_words=process_initial_question_words,
        )
        if word_match_dicts is None:
            return []
        matches = []
        for word_match_dict in word_match_dicts:
            match = Match(
                search_phrase.label,
                search_phrase.doc_text,
                document_label,
                search_phrase.topic_match_phraselet
                and search_phrase.has_single_matchable_word,
                search_phrase.topic_match_phraselet_created_without_matching_tags,
                search_phrase.reverse_only,
            )
            not_normalized_overall_similarity_measure = 1.0
            for search_phrase_token in search_phrase.matchable_tokens:
                if search_phrase_token not in word_match_dict:
                    break
                word_match = word_match_dict[search_phrase_token]
                if (
                    word_match.document_subword is not None
                    and word_match.document_token.i
                    != word_match.document_subword.containing_token_index
                    and not self._subword_containing_token_is_within_match(
                        word_match, word_match_dict.values()
                    )
                ):
                    break
                match.word_matches.append(word_match)
                if word_match.is_negated:
                    match.is_negated = True
                if word_match.is_uncertain:
                    match.is_uncertain = True
                if word_match.search_phrase_token == search_phrase.root_token:
                    match.index_within_document = word_match.document_token.i
                not_normalized_overall_similarity_measure *= (
                    word_match.similarity_measure
                )
                if search_phrase.topic_match_phraselet:
                    word_match.temp_is_parent = (
                        word_match.search_phrase_token.i
                        == search_phrase.root_token_index
                    )
            if len(match.word_matches) < len(search_phrase.matchable_token_indexes):
                continue
            if not_normalized_overall_similarity_measure < 1.0:
                match.overall_similarity_measure = round(
                    not_normalized_overall_similarity_measure
                    ** (1 / len(search_phrase.matchable_non_entity_tokens_to_vectors)),
                    8,
                )
            matches.append(match)
        return matches

    def _subword_containing_token_is_within_match(
        self, word_match: WordMatch, other_word_matches: ValuesView[WordMatch]
    ) -> bool:
        """Where subwords are involved in conjunction, subwords whose meaning is distributed
        among multiple words are modelled as belonging to each of those words even if they
        only occur once in the document text. This method is used to filter out duplicate
        matches: wherever a potential match contains a subword A that is expressed on another word,
        it checks that the match also contains at least one subword B that is expressed on the
        word where A is modelled.
        """
        for other_word_match in other_word_matches:
            if (
                other_word_match.document_subword is not None
                and other_word_match.document_subword.containing_token_index
                == word_match.document_token.i
            ):
                return True
        return False

    def match_recursively(
        self,
        *,
        word_matching_strategies: List[WordMatchingStrategy],
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document: Doc,
        document_token: Token,
        document_subword_index: Optional[int],
        search_phrase_and_document_visited_table: List[Set[Index]],
        is_uncertain: bool,
        structurally_matched_document_token: Token,
        compare_embeddings_on_non_root_words: bool,
        process_initial_question_words: bool
    ) -> Optional[List[Dict[Token, WordMatch]]]:
        """Called whenever matching is attempted between a search phrase token and a document
        token."""
        index = Index(document_token.i, document_subword_index)
        if document_subword_index is None:
            for word_matching_strategy in word_matching_strategies:
                if document_token._.holmes.multiword_spans is not None:
                    potential_word_match = word_matching_strategy.match_multiwords(
                        search_phrase,
                        search_phrase_token,
                        document_token,
                        document_token._.holmes.multiword_spans,
                    )
                    if potential_word_match is not None:
                        break
                potential_word_match = word_matching_strategy.match_token(
                    search_phrase, search_phrase_token, document_token
                )
                if potential_word_match is not None:
                    break
            else:
                return None
        else:
            for word_matching_strategy in word_matching_strategies:
                potential_word_match = word_matching_strategy.match_subword(
                    search_phrase,
                    search_phrase_token,
                    document_token,
                    document_token._.holmes.subwords[document_subword_index],
                )
                if potential_word_match is not None:
                    break
            else:
                return None

        word_match_dicts_to_return = [{search_phrase_token: potential_word_match}]
        already_recursed = (
            index in search_phrase_and_document_visited_table[search_phrase_token.i]
        )
        search_phrase_and_document_visited_table[search_phrase_token.i].add(index)

        if not search_phrase.has_single_matchable_word and not already_recursed:
            for dependency in (
                dependency
                for dependency in search_phrase_token._.holmes.children
                if dependency.child_token(search_phrase_token.doc)._.holmes.is_matchable
                or (
                    search_phrase.topic_match_phraselet
                    and process_initial_question_words
                    and dependency.child_token(
                        search_phrase_token.doc
                    )._.holmes.is_initial_question_word
                )
            ):
                search_phrase_child_token = dependency.child_token(
                    search_phrase_token.doc
                )
                this_dependency_word_match_dicts = []
                # Loop through this token and any tokens linked to it by coreference
                working_document_parent_indexes = [
                    Index(document_token.i, document_subword_index)
                ]
                if self.perform_coreference_resolution and (
                    document_subword_index is None
                    or document_token._.holmes.subwords[document_subword_index].is_head
                ):
                    working_document_parent_indexes.extend(
                        [
                            Index(token_index, None)
                            for token_index in document_token._.holmes.token_and_coreference_chain_indexes
                            if token_index != document_token.i
                        ]
                    )
                    # Try coreferents closer to the structurally match token first. Once we've matched a document
                    # child from one of these coreferents, it shouldn't be matched again from elsewhere in the chain
                    working_document_parent_indexes.sort(
                        key=lambda index: (
                            abs(index.token_index - document_token.i),
                            index.token_index > document_token.i,
                        )
                    )
                matched_document_indexes_for_parent = []
                for working_document_parent_index in working_document_parent_indexes:
                    document_parent_token = document_token.doc[
                        working_document_parent_index.token_index
                    ]
                    if (
                        not working_document_parent_index.is_subword()
                        or document_parent_token._.holmes.subwords[
                            working_document_parent_index.subword_index
                        ].is_head
                    ):
                        # is_head: e.g. 'Polizeiinformation über Kriminelle' should match
                        # 'Information über Kriminelle'

                        # inverse_polarity_boolean: *True* in the special case where the
                        # dependency has been matched backwards
                        document_dependencies_to_inverse_polarity_booleans = {
                            document_dependency: False
                            for document_dependency in document_parent_token._.holmes.children
                            if self.semantic_matching_helper.dependency_labels_match(
                                search_phrase_dependency_label=dependency.label,
                                document_dependency_label=document_dependency.label,
                                inverse_polarity=False,
                            )
                        }
                        document_dependencies_to_inverse_polarity_booleans.update(
                            {
                                document_dependency: True
                                for document_dependency in document_parent_token._.holmes.parents
                                if self.use_reverse_dependency_matching
                                and self.semantic_matching_helper.dependency_labels_match(
                                    search_phrase_dependency_label=dependency.label,
                                    document_dependency_label=document_dependency.label,
                                    inverse_polarity=True,
                                )
                            }
                        )
                        for (
                            document_dependency,
                            inverse_polarity,
                        ) in document_dependencies_to_inverse_polarity_booleans.items():
                            if not inverse_polarity:
                                document_child = document_dependency.child_token(
                                    document_token.doc
                                )
                            else:
                                document_child = document_dependency.parent_token(
                                    document_token.doc
                                )
                            working_document_child_mentions = [[document_child.i]]
                            if (
                                self.perform_coreference_resolution
                                and document_child._.holmes.mentions is not None
                            ):
                                working_document_child_mentions.extend(
                                    [
                                        m.indexes
                                        for m in document_child._.holmes.mentions
                                        if document_child.i not in m.indexes
                                    ]
                                )
                            for (
                                working_document_child_mention
                            ) in working_document_child_mentions:
                                if (
                                    document_token.doc[
                                        working_document_child_mention[0]
                                    ].pos_
                                    == "PRON"
                                ):
                                    continue
                                working_document_child_indexes = []
                                for (
                                    working_document_child_token_index
                                ) in working_document_child_mention:
                                    working_document_child_indexes.append(
                                        Index(working_document_child_token_index, None)
                                    )
                                    working_document_child = document_token.doc[
                                        working_document_child_token_index
                                    ]
                                    for subword in (
                                        subword
                                        for subword in working_document_child._.holmes.subwords
                                        if subword.is_head
                                    ):
                                        working_document_child_indexes.append(
                                            Index(
                                                working_document_child.i, subword.index
                                            )
                                        )
                                at_least_one_match_within_mention = False
                                for (
                                    working_document_child_index
                                ) in working_document_child_indexes:
                                    if search_phrase.question_phraselet and document[
                                        working_document_parent_index.token_index
                                    ] in self.semantic_matching_helper.get_subtree_list_for_question_answer(
                                        document[
                                            working_document_child_index.token_index
                                        ]
                                    ):
                                        # e.g. 'Who did Richard see?' 'The person Richard saw was angry'
                                        continue
                                    if (
                                        working_document_child_index
                                        in matched_document_indexes_for_parent
                                    ):
                                        continue
                                    word_match_dicts = self.match_recursively(
                                        word_matching_strategies=word_matching_strategies,
                                        search_phrase=search_phrase,
                                        search_phrase_token=search_phrase_child_token,
                                        document=document,
                                        document_token=document[
                                            working_document_child_index.token_index
                                        ],
                                        document_subword_index=working_document_child_index.subword_index,
                                        search_phrase_and_document_visited_table=search_phrase_and_document_visited_table,
                                        is_uncertain=(
                                            (
                                                document_dependency.is_uncertain
                                                and not dependency.is_uncertain
                                            )
                                            or inverse_polarity
                                        ),
                                        structurally_matched_document_token=document_child,
                                        compare_embeddings_on_non_root_words=compare_embeddings_on_non_root_words,
                                        process_initial_question_words=process_initial_question_words,
                                    )
                                    if word_match_dicts is not None:
                                        at_least_one_match_within_mention = True
                                        this_dependency_word_match_dicts.extend(
                                            word_match_dicts
                                        )
                                        matched_document_indexes_for_parent.append(
                                            working_document_child_index
                                        )
                                if at_least_one_match_within_mention:
                                    break
                    if working_document_parent_index.is_subword():
                        # examine relationship to dependent subword in the same word
                        document_parent_subword = document_token.doc[
                            working_document_parent_index.token_index
                        ]._.holmes.subwords[working_document_parent_index.subword_index]
                        if (
                            document_parent_subword.dependent_index is not None
                            and self.semantic_matching_helper.dependency_labels_match(
                                search_phrase_dependency_label=dependency.label,
                                document_dependency_label=document_parent_subword.dependency_label,
                                inverse_polarity=False,
                            )
                        ):
                            word_match_dicts = self.match_recursively(
                                word_matching_strategies=word_matching_strategies,
                                search_phrase=search_phrase,
                                search_phrase_token=search_phrase_child_token,
                                document=document,
                                document_token=document_token,
                                document_subword_index=document_parent_subword.dependent_index,
                                search_phrase_and_document_visited_table=search_phrase_and_document_visited_table,
                                is_uncertain=False,
                                structurally_matched_document_token=document_token,
                                compare_embeddings_on_non_root_words=compare_embeddings_on_non_root_words,
                                process_initial_question_words=process_initial_question_words,
                            )
                            if word_match_dicts is not None:
                                this_dependency_word_match_dicts.extend(
                                    word_match_dicts
                                )
                        # examine relationship to governing subword in the same word
                        document_child_subword = document_token.doc[
                            working_document_parent_index.token_index
                        ]._.holmes.subwords[working_document_parent_index.subword_index]
                        if (
                            document_child_subword.governor_index is not None
                            and self.use_reverse_dependency_matching
                            and self.semantic_matching_helper.dependency_labels_match(
                                search_phrase_dependency_label=dependency.label,
                                document_dependency_label=document_parent_subword.governing_dependency_label,
                                inverse_polarity=True,
                            )
                        ):
                            word_match_dicts = self.match_recursively(
                                word_matching_strategies=word_matching_strategies,
                                search_phrase=search_phrase,
                                search_phrase_token=search_phrase_child_token,
                                document=document,
                                document_token=document_token,
                                document_subword_index=document_parent_subword.governor_index,
                                search_phrase_and_document_visited_table=search_phrase_and_document_visited_table,
                                is_uncertain=False,
                                structurally_matched_document_token=document_token,
                                compare_embeddings_on_non_root_words=compare_embeddings_on_non_root_words,
                                process_initial_question_words=process_initial_question_words,
                            )
                            if word_match_dicts is not None:
                                this_dependency_word_match_dicts.extend(
                                    word_match_dicts
                                )
                if len(this_dependency_word_match_dicts) == 0:
                    return None
                new_word_match_dicts_to_return = []
                for dependency_word_match_dict in this_dependency_word_match_dicts:
                    for existing_word_match_dict in (
                        w.copy() for w in word_match_dicts_to_return
                    ):
                        merged_word_match_dict = self.merge_word_match_dicts(
                            existing_word_match_dict, dependency_word_match_dict
                        )
                        if merged_word_match_dict is not None:
                            new_word_match_dicts_to_return.append(
                                merged_word_match_dict
                            )
                word_match_dicts_to_return = new_word_match_dicts_to_return
        potential_word_match.structurally_matched_document_token = (
            structurally_matched_document_token
        )
        potential_word_match.is_negated = document_token._.holmes.is_negated
        potential_word_match.is_uncertain = (
            is_uncertain or document_token._.holmes.is_uncertain
        )
        return word_match_dicts_to_return

    def merge_word_match_dicts(
        self, existing_word_match_dict, dependency_word_match_dict
    ):
        """Where the search phrase tokens form a closed net, we need to filter out
        document subgraph matches where the net structure is open.
        """
        for key in dependency_word_match_dict:
            if key not in existing_word_match_dict:
                existing_word_match_dict[key] = dependency_word_match_dict[key]
            elif (
                existing_word_match_dict[key].document_token
                != dependency_word_match_dict[key].document_token
            ):
                return None
        return existing_word_match_dict

    def build_match_dictionaries(self, matches: List[Match]) -> List[Dict]:
        """Builds and returns a sorted list of match dictionaries."""
        match_dicts: List[Dict[str, Any]] = []
        for match in matches:
            earliest_sentence_index = sys.maxsize
            latest_sentence_index = -1
            for word_match in match.word_matches:
                sentence_index = word_match.document_token.sent.start
                earliest_sentence_index = min(sentence_index, earliest_sentence_index)
                latest_sentence_index = max(sentence_index, latest_sentence_index)
            sentences_string = " ".join(
                sentence.text.strip()
                for sentence in match.word_matches[0].document_token.doc.sents
                if sentence.start >= earliest_sentence_index
                and sentence.start <= latest_sentence_index
            )
            match_dict: Dict[str, Any] = {
                "search_phrase_label": match.search_phrase_label,
                "search_phrase_text": match.search_phrase_text,
                "document": match.document_label,
                "index_within_document": match.index_within_document,
                "sentences_within_document": sentences_string,
                "negated": match.is_negated,
                "uncertain": match.is_uncertain,
                "involves_coreference": match.involves_coreference,
                "overall_similarity_measure": match.overall_similarity_measure,
            }
            text_word_matches: List[Dict[str, Any]] = []
            for word_match in match.word_matches:
                text_word_matches.append(
                    {
                        "search_phrase_token_index": word_match.search_phrase_token.i,
                        "search_phrase_word": word_match.search_phrase_word,
                        "document_token_index": word_match.document_token.i,
                        "first_document_token_index": word_match.first_document_token.i,
                        "last_document_token_index": word_match.last_document_token.i,
                        "structurally_matched_document_token_index": word_match.structurally_matched_document_token.i,
                        "document_subword_index": word_match.document_subword.index
                        if word_match.document_subword is not None
                        else None,
                        "document_subword_containing_token_index": word_match.document_subword.containing_token_index
                        if word_match.document_subword is not None
                        else None,
                        "document_word": word_match.document_word,
                        "document_phrase": self.semantic_matching_helper.get_dependent_phrase(
                            word_match.document_token, word_match.document_subword
                        ),
                        "match_type": word_match.word_match_type,
                        "negated": word_match.is_negated,
                        "uncertain": word_match.is_uncertain,
                        "similarity_measure": word_match.similarity_measure,
                        "involves_coreference": word_match.involves_coreference,
                        "extracted_word": word_match.extracted_word,
                        "depth": word_match.depth,
                        "explanation": word_match.explanation,
                    }
                )
            match_dict["word_matches"] = text_word_matches
            match_dicts.append(match_dict)
        return match_dicts
