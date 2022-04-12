from typing import Optional, List
from spacy.tokens import Token
from .general import WordMatch, WordMatchingStrategy
from ..parsing import MultiwordSpan, SearchPhrase


class EntityWordMatchingStrategy(WordMatchingStrategy):

    WORD_MATCH_TYPE_LABEL = "entity"

    @staticmethod
    def _get_explanation(search_phrase_display_word: str) -> str:
        return ''.join(("Has an entity label matching ", search_phrase_display_word, "."))

    def match_multiwords(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_multiwords: List[MultiwordSpan],
    ) -> Optional[WordMatch]:

        entity_placeholder = self.semantic_matching_helper.get_entity_placeholder(search_phrase_token)
        if entity_placeholder is None:
            return None

        for multiword in document_multiwords:
            if any(1 for i in multiword.token_indexes if not self._entity_placeholder_matches(entity_placeholder, document_token.doc[i])):
                continue
            return WordMatch(
                search_phrase_token=search_phrase_token,
                search_phrase_word=entity_placeholder,
                document_token=document_token,
                first_document_token=document_token.doc[multiword.token_indexes[0]],
                last_document_token=document_token.doc[multiword.token_indexes[-1]],
                document_subword=None,
                document_word=multiword.text,
                word_match_type=self.WORD_MATCH_TYPE_LABEL,
                explanation=self._get_explanation(entity_placeholder),
            )
        return None

    def match_token(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
    ) -> Optional[WordMatch]:

        entity_placeholder = self.semantic_matching_helper.get_entity_placeholder(search_phrase_token)
        if entity_placeholder is None:
            return None

        if self._entity_placeholder_matches(entity_placeholder, document_token):
            return WordMatch(
                search_phrase_token=search_phrase_token,
                search_phrase_word=entity_placeholder,
                document_token=document_token,
                first_document_token=document_token,
                last_document_token=document_token,
                document_subword=None,
                document_word=document_token.text,
                word_match_type=self.WORD_MATCH_TYPE_LABEL,
                explanation=self._get_explanation(entity_placeholder),
            )
        return None

    def _entity_placeholder_matches(
            self, entity_placeholder, document_token):
        return (
            document_token.ent_type_ == entity_placeholder[6:] and
            len(document_token._.holmes.lemma.strip()) > 0) or (
                entity_placeholder == 'ENTITYNOUN' and
                document_token.pos_ in self.semantic_matching_helper.noun_pos)
                # len(document_token._.holmes.lemma.strip()) > 0: in German spaCy sometimes
                # classifies whitespace as entities.
