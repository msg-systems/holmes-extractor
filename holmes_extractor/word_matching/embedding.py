from typing import Optional, List
from spacy.tokens import Token
from .general import WordMatch, WordMatchingStrategy
from ..parsing import Subword, SearchPhrase


class EmbeddingWordMatchingStrategy(WordMatchingStrategy):

    WORD_MATCH_TYPE_LABEL = "embedding"

    @staticmethod
    def _get_explanation(similarity: float, search_phrase_display_word: str) -> str:
        printable_similarity = str(int(similarity * 100))
        return ''.join((
            "Has a word embedding that is ", printable_similarity,
            "% similar to ", search_phrase_display_word, "."))


    def __init__(self, semantic_matching_helper, overall_similarity_threshold, initial_question_word_overall_similarity_threshold):
        self.overall_similarity_threshold = overall_similarity_threshold
        self.initial_question_word_overall_similarity_threshold = initial_question_word_overall_similarity_threshold
        super().__init__(semantic_matching_helper)

    def match_token(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
    ) -> Optional[WordMatch]:

        if search_phrase_token.i in \
                search_phrase.matchable_non_entity_tokens_to_vectors.keys() and \
                self.semantic_matching_helper.embedding_matching_permitted(search_phrase_token):
            search_phrase_vector = search_phrase.matchable_non_entity_tokens_to_vectors[
                search_phrase_token.i]
            if search_phrase_vector is None or not self.semantic_matching_helper.embedding_matching_permitted(document_token):
                return None
            document_vector = document_token._.holmes.vector
            if search_phrase.root_token._.holmes.has_initial_question_word_in_phrase and self.initial_question_word_overall_similarity_threshold is not None:
                working_overall_similarity_threshold = self.initial_question_word_overall_similarity_threshold
            else:
                working_overall_similarity_threshold = self.overall_similarity_threshold
            single_token_similarity_threshold = working_overall_similarity_threshold ** len(
                search_phrase.matchable_non_entity_tokens_to_vectors)
            if document_vector is not None:
                similarity = \
                    self.semantic_matching_helper.cosine_similarity(search_phrase_vector,
                    document_vector)
                if similarity > single_token_similarity_threshold:
                    if not search_phrase.topic_match_phraselet and \
                            len(search_phrase_token._.holmes.lemma.split()) > 1:
                        search_phrase_display_word = search_phrase_token.lemma_
                    else:
                        search_phrase_display_word = search_phrase_token._.holmes.lemma
                    return WordMatch(
                        search_phrase_token=search_phrase_token,
                        search_phrase_word=search_phrase_display_word,
                        document_token=document_token,
                        first_document_token=document_token,
                        last_document_token=document_token,
                        document_subword=None,
                        document_word=document_token.lemma_,
                        word_match_type=self.WORD_MATCH_TYPE_LABEL,
                        explanation=self._get_explanation(similarity, search_phrase_display_word),
                    )
        return None


    def match_subword(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_subword: Subword,
    ) -> Optional[WordMatch]:

        if search_phrase_token.i in \
                search_phrase.matchable_non_entity_tokens_to_vectors.keys() and \
                self.semantic_matching_helper.embedding_matching_permitted(search_phrase_token):
            search_phrase_vector = search_phrase.matchable_non_entity_tokens_to_vectors[
                search_phrase_token.i]
            if search_phrase_vector is None or not self.semantic_matching_helper.embedding_matching_permitted(document_subword):
                return None
            document_vector = document_subword.vector
            if search_phrase.root_token._.holmes.has_initial_question_word_in_phrase and self.initial_question_word_overall_similarity_threshold is not None:
                working_overall_similarity_threshold = self.initial_question_word_overall_similarity_threshold
            else:
                working_overall_similarity_threshold = self.overall_similarity_threshold
            single_token_similarity_threshold = working_overall_similarity_threshold ** len(
                search_phrase.matchable_non_entity_tokens_to_vectors)
            if document_vector is not None:
                similarity = \
                    self.semantic_matching_helper.cosine_similarity(search_phrase_vector,
                    document_vector)
                if similarity > single_token_similarity_threshold:
                    if not search_phrase.topic_match_phraselet and \
                            len(search_phrase_token._.holmes.lemma.split()) > 1:
                        search_phrase_display_word = search_phrase_token.lemma_
                    else:
                        search_phrase_display_word = search_phrase_token._.holmes.lemma
                        return WordMatch(
                        search_phrase_token=search_phrase_token,
                        search_phrase_word=search_phrase_display_word,
                        document_token=document_token,
                        first_document_token=document_token,
                        last_document_token=document_token,
                        document_subword=document_subword,
                        document_word=document_subword.lemma,
                        word_match_type=self.WORD_MATCH_TYPE_LABEL,
                        explanation=self._get_explanation(similarity, search_phrase_display_word),
                    )
        return None
