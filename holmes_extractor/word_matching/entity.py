from typing import Dict, Optional, List
from spacy.tokens import Token, Doc
from .general import WordMatch, WordMatchingStrategy
from ..parsing import MultiwordSpan, CorpusWordPosition, SearchPhrase


class EntityWordMatchingStrategy(WordMatchingStrategy):

    WORD_MATCH_TYPE_LABEL = "entity"

    @staticmethod
    def _get_explanation(search_phrase_display_word: str) -> str:
        return "".join(
            ("Has an entity label matching ", search_phrase_display_word.upper(), ".")
        )

    def match_multiwords(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_multiwords: List[MultiwordSpan],
    ) -> Optional[WordMatch]:

        entity_placeholder = self.semantic_matching_helper.get_entity_placeholder(
            search_phrase_token
        )
        if entity_placeholder is None:
            return None

        for multiword in document_multiwords:
            if any(
                1
                for i in multiword.token_indexes
                if not self._entity_placeholder_matches(
                    entity_placeholder, document_token.doc[i]
                )
            ):
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

        entity_placeholder = self.semantic_matching_helper.get_entity_placeholder(
            search_phrase_token
        )
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
                document_word=document_token.text.lower(),
                word_match_type=self.WORD_MATCH_TYPE_LABEL,
                explanation=self._get_explanation(entity_placeholder),
            )
        return None

    def add_reverse_dict_entries(
        self,
        reverse_dict: Dict[str, List[CorpusWordPosition]],
        doc: Doc,
        document_label: str,
    ) -> None:
        for token in doc:
            # parent check is necessary so we only find multiword entities once per
            # search phrase. sibling_marker_deps applies to siblings which would
            # otherwise be excluded because the main sibling would normally also match the
            # entity root word.
            if len(token.ent_type_) > 0 and (
                token.dep_ == "ROOT"
                or token.dep_ in self.semantic_matching_helper.sibling_marker_deps
                or token.ent_type_ != token.head.ent_type_
            ):
                entity_label = "".join(("ENTITY", token.ent_type_))
                self.add_reverse_dict_entry(
                    reverse_dict,
                    entity_label,
                    document_label,
                    token.i,
                    None,
                )
            entity_defined_multiword = (
                self.semantic_matching_helper.get_entity_defined_multiword(token)
            )
            if entity_defined_multiword is not None:
                self.add_reverse_dict_entry(
                    reverse_dict,
                    entity_defined_multiword.text.lower(),
                    document_label,
                    token.i,
                    None,
                )

    def _entity_placeholder_matches(
        self, entity_placeholder: str, document_token: Token
    ) -> bool:
        return (
            document_token.ent_type_ == entity_placeholder[6:]
            and len(document_token._.holmes.lemma.strip()) > 0
        ) or (
            entity_placeholder == "ENTITYNOUN"
            and document_token.pos_ in self.semantic_matching_helper.noun_pos
        )
        # len(document_token._.holmes.lemma.strip()) > 0: some German spaCy models sometimes
        # classifies whitespace as entities.
