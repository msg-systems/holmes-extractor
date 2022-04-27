from typing import Optional, List, Dict
from spacy.tokens import Token
from thinc.types import Floats1d
from .general import WordMatch, WordMatchingStrategy
from ..parsing import SearchPhrase, MultiwordSpan, SemanticMatchingHelper


class EntityEmbeddingWordMatchingStrategy(WordMatchingStrategy):

    WORD_MATCH_TYPE_LABEL = "entity_embedding"

    @staticmethod
    def _get_explanation(similarity: float, search_phrase_display_word: str) -> str:
        printable_similarity = str(int(similarity * 100))
        return "".join(
            (
                "Has an entity label that is ",
                printable_similarity,
                "% similar to the word embedding corresponding to ",
                search_phrase_display_word.upper(),
                ".",
            )
        )

    def __init__(
        self,
        semantic_matching_helper: SemanticMatchingHelper,
        perform_coreference_resolution: bool,
        overall_similarity_threshold: float,
        initial_question_word_overall_similarity_threshold: float,
        entity_label_to_vector_dict: Dict[str, Floats1d],
    ):
        self.overall_similarity_threshold = overall_similarity_threshold
        self.initial_question_word_overall_similarity_threshold = (
            initial_question_word_overall_similarity_threshold
        )
        self.entity_label_to_vector_dict = entity_label_to_vector_dict
        super().__init__(semantic_matching_helper, perform_coreference_resolution)

    def match_multiwords(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_multiwords: List[MultiwordSpan],
    ) -> Optional[WordMatch]:

        if (
            search_phrase_token.i
            in search_phrase.matchable_non_entity_tokens_to_vectors.keys()
            and self.semantic_matching_helper.embedding_matching_permitted(
                search_phrase_token
            )
        ):
            search_phrase_vector = search_phrase.matchable_non_entity_tokens_to_vectors[
                search_phrase_token.i
            ]
            if (
                search_phrase_vector is None
                or not self.semantic_matching_helper.embedding_matching_permitted(
                    document_token
                )
            ):
                return None
            for document_multiword in document_multiwords:
                if document_token.ent_type_ != "" and all(
                    document_token.doc[i].ent_type_ == document_token.ent_type_
                    for i in document_multiword.token_indexes
                ):
                    potential_word_match = self._check_for_word_match(
                        search_phrase=search_phrase,
                        search_phrase_token=search_phrase_token,
                        search_phrase_vector=search_phrase_vector,
                        document_token=document_token,
                        first_document_token=document_token.doc[
                            document_multiword.token_indexes[0]
                        ],
                        last_document_token=document_token.doc[
                            document_multiword.token_indexes[-1]
                        ],
                    )
                    if potential_word_match is not None:
                        return potential_word_match

        return None

    def match_token(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
    ) -> Optional[WordMatch]:

        if (
            search_phrase_token.i
            in search_phrase.matchable_non_entity_tokens_to_vectors.keys()
            and self.semantic_matching_helper.embedding_matching_permitted(
                search_phrase_token
            )
        ):
            search_phrase_vector = search_phrase.matchable_non_entity_tokens_to_vectors[
                search_phrase_token.i
            ]
            if (
                search_phrase_vector is None
                or not self.semantic_matching_helper.embedding_matching_permitted(
                    document_token
                )
            ):
                return None
            if document_token.ent_type_ != "":
                return self._check_for_word_match(
                    search_phrase=search_phrase,
                    search_phrase_token=search_phrase_token,
                    search_phrase_vector=search_phrase_vector,
                    document_token=document_token,
                    first_document_token=document_token,
                    last_document_token=document_token,
                )
        return None

    def _check_for_word_match(
        self,
        *,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        search_phrase_vector: Floats1d,
        document_token: Token,
        first_document_token: Token,
        last_document_token: Token,
    ) -> Optional[WordMatch]:
        if (
            search_phrase_token._.holmes.is_initial_question_word
            or search_phrase_token._.holmes.has_initial_question_word_in_phrase
        ) and self.initial_question_word_overall_similarity_threshold is not None:
            working_overall_similarity_threshold = (
                self.initial_question_word_overall_similarity_threshold
            )
        else:
            working_overall_similarity_threshold = self.overall_similarity_threshold
        single_token_similarity_threshold = working_overall_similarity_threshold ** len(
            search_phrase.matchable_non_entity_tokens_to_vectors
        )

        similarity_measure = self.semantic_matching_helper.token_matches_ent_type(
            search_phrase_vector,
            self.entity_label_to_vector_dict,
            (document_token.ent_type_,),
            single_token_similarity_threshold,
        )
        if similarity_measure > 0:
            if (
                not search_phrase.topic_match_phraselet
                and len(search_phrase_token._.holmes.lemma.split()) > 1
            ):
                search_phrase_display_word = search_phrase_token.lemma_
            else:
                search_phrase_display_word = search_phrase_token._.holmes.lemma
            word_match = WordMatch(
                search_phrase_token=search_phrase_token,
                search_phrase_word=search_phrase_display_word,
                document_token=document_token,
                first_document_token=first_document_token,
                last_document_token=last_document_token,
                document_subword=None,
                document_word=document_token.lemma_,
                word_match_type=self.WORD_MATCH_TYPE_LABEL,
                explanation=self._get_explanation(
                    similarity_measure, search_phrase_display_word
                ),
            )
            word_match.similarity_measure = similarity_measure
            return word_match
        return None
