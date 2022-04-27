from typing import Optional, Dict
from spacy.tokens import Token
from thinc.types import Floats1d
from .general import WordMatch, WordMatchingStrategy
from ..parsing import SearchPhrase, SemanticMatchingHelper, Subword


class QuestionWordMatchingStrategy(WordMatchingStrategy):

    WORD_MATCH_TYPE_LABEL = "question"

    @staticmethod
    def _get_explanation(search_phrase_display_word: str) -> str:
        return "".join(
            ("Matches the question word ", search_phrase_display_word.upper(), ".")
        )

    def __init__(
        self,
        semantic_matching_helper: SemanticMatchingHelper,
        perform_coreference_resolution: bool,
        initial_question_word_overall_similarity_threshold: float,
        entity_label_to_vector_dict: Dict[str, Floats1d],
    ):
        self.initial_question_word_overall_similarity_threshold = (
            initial_question_word_overall_similarity_threshold
        )
        self.entity_label_to_vector_dict = entity_label_to_vector_dict
        super().__init__(semantic_matching_helper, perform_coreference_resolution)

    def match_token(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
    ) -> Optional[WordMatch]:

        if search_phrase_token._.holmes.is_initial_question_word:
            document_vector = document_token._.holmes.vector
            if document_vector is not None:
                question_word_matches = (
                    self.semantic_matching_helper.question_word_matches(
                        search_phrase_token,
                        document_token,
                        None,
                        document_vector,
                        self.entity_label_to_vector_dict,
                        self.initial_question_word_overall_similarity_threshold
                        ** len(search_phrase.matchable_non_entity_tokens_to_vectors),
                    )
                )
            else:
                question_word_matches = (
                    self.semantic_matching_helper.question_word_matches(
                        search_phrase_token, document_token, None, None, None, None
                    )
                )
            if question_word_matches:
                first_document_token_index = (
                    last_document_token_index
                ) = document_token.i
                if (
                    document_token.pos_ in self.semantic_matching_helper.noun_pos
                    and len(document_token.ent_type_) > 0
                ):
                    while first_document_token_index >= 1:
                        if (
                            document_token.doc[first_document_token_index - 1].pos_
                            in self.semantic_matching_helper.noun_pos
                        ):
                            first_document_token_index = first_document_token_index - 1
                        else:
                            break
                    while last_document_token_index + 1 < len(document_token.doc):
                        if (
                            document_token.doc[last_document_token_index + 1].pos_
                            in self.semantic_matching_helper.noun_pos
                        ):
                            last_document_token_index = last_document_token_index + 1
                        else:
                            break
                return WordMatch(
                    search_phrase_token=search_phrase_token,
                    search_phrase_word=search_phrase_token._.holmes.lemma,
                    document_token=document_token,
                    first_document_token=document_token.doc[first_document_token_index],
                    last_document_token=document_token.doc[last_document_token_index],
                    document_subword=None,
                    document_word=document_token._.holmes.lemma,
                    word_match_type=self.WORD_MATCH_TYPE_LABEL,
                    explanation=self._get_explanation(
                        search_phrase_token._.holmes.lemma
                    ),
                )
        return None

    def match_subword(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_subword: Subword,
    ) -> Optional[WordMatch]:

        if (
            document_subword.is_head
        ):  # question words should not match a head subword but the whole word or multiword:
            return None
        if search_phrase_token._.holmes.is_initial_question_word:
            document_vector = document_subword.vector
            if document_vector is not None:
                question_word_matches = (
                    self.semantic_matching_helper.question_word_matches(
                        search_phrase_token,
                        document_token,
                        document_subword.index,
                        document_vector,
                        self.entity_label_to_vector_dict,
                        self.initial_question_word_overall_similarity_threshold
                        ** len(search_phrase.matchable_non_entity_tokens_to_vectors),
                    )
                )
            else:
                question_word_matches = (
                    self.semantic_matching_helper.question_word_matches(
                        search_phrase_token,
                        document_token,
                        document_subword.index,
                        None,
                        None,
                        None,
                    )
                )
            if question_word_matches:
                return WordMatch(
                    search_phrase_token=search_phrase_token,
                    search_phrase_word=search_phrase_token._.holmes.lemma,
                    document_token=document_token,
                    first_document_token=document_token,
                    last_document_token=document_token,
                    document_subword=document_subword,
                    document_word=document_subword.lemma,
                    word_match_type=self.WORD_MATCH_TYPE_LABEL,
                    explanation=self._get_explanation(document_subword.lemma),
                )
        return None
