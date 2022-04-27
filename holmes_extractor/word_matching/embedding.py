from typing import Optional
from spacy.tokens import Token
from .general import WordMatch, WordMatchingStrategy
from ..parsing import SemanticMatchingHelper, Subword, SearchPhrase


class EmbeddingWordMatchingStrategy(WordMatchingStrategy):

    WORD_MATCH_TYPE_LABEL = "embedding"

    @staticmethod
    def _get_explanation(similarity: float, search_phrase_display_word: str) -> str:
        printable_similarity = str(int(similarity * 100))
        return "".join(
            (
                "Has a word embedding that is ",
                printable_similarity,
                "% similar to ",
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
    ):
        self.overall_similarity_threshold = overall_similarity_threshold
        self.initial_question_word_overall_similarity_threshold = (
            initial_question_word_overall_similarity_threshold
        )
        super().__init__(semantic_matching_helper, perform_coreference_resolution)

    def match_token(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
    ) -> Optional[WordMatch]:

        return self._check_for_word_match(
            search_phrase, search_phrase_token, document_token, None
        )

    def match_subword(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_subword: Subword,
    ) -> Optional[WordMatch]:

        return self._check_for_word_match(
            search_phrase, search_phrase_token, document_token, document_subword
        )

    def _check_for_word_match(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_subword: Optional[Subword],
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
            if search_phrase_vector is None:
                return None
            if document_subword is not None:
                if not self.semantic_matching_helper.embedding_matching_permitted(
                    document_subword
                ):
                    return None
                document_vector = document_subword.vector
                document_word = document_subword.lemma
            else:
                if not self.semantic_matching_helper.embedding_matching_permitted(
                    document_token
                ):
                    return None
                document_vector = document_token._.holmes.vector
                document_word = document_token.lemma_
            if (
                (
                    search_phrase_token._.holmes.is_initial_question_word
                    or search_phrase_token._.holmes.has_initial_question_word_in_phrase
                )
                and self.initial_question_word_overall_similarity_threshold is not None
            ):
                working_overall_similarity_threshold = (
                    self.initial_question_word_overall_similarity_threshold
                )
            else:
                working_overall_similarity_threshold = self.overall_similarity_threshold
            single_token_similarity_threshold = (
                working_overall_similarity_threshold
                ** len(search_phrase.matchable_non_entity_tokens_to_vectors)
            )
            if document_vector is not None:
                similarity_measure = self.semantic_matching_helper.cosine_similarity(
                    search_phrase_vector, document_vector
                )
                if similarity_measure > single_token_similarity_threshold:
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
                        first_document_token=document_token,
                        last_document_token=document_token,
                        document_subword=document_subword,
                        document_word=document_word,
                        word_match_type=self.WORD_MATCH_TYPE_LABEL,
                        explanation=self._get_explanation(
                            similarity_measure, search_phrase_display_word
                        ),
                    )
                    word_match.similarity_measure = similarity_measure
                    return word_match
        return None
