from typing import Dict, Optional, List
from spacy.tokens import Token, Doc
from .general import WordMatch, WordMatchingStrategy
from ..parsing import CorpusWordPosition, MultiwordSpan, Subword, SearchPhrase


class DerivationWordMatchingStrategy(WordMatchingStrategy):

    WORD_MATCH_TYPE_LABEL = "derivation"

    @staticmethod
    def _get_explanation(search_phrase_display_word: str) -> str:
        return "".join(
            ("Has a common stem with ", search_phrase_display_word.upper(), ".")
        )

    def match_multiwords(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_multiwords: List[MultiwordSpan],
    ) -> Optional[WordMatch]:

        if len(search_phrase_token._.holmes.lemma.split()) == 1:
            return None
        if search_phrase_token._.holmes.derivation_matching_reprs is None and not any(
            m for m in document_multiwords if m.derivation_matching_reprs is not None
        ):
            return None

        for multiword in document_multiwords:

            search_phrase_reprs = []
            document_reprs = []

            if search_phrase_token._.holmes.derivation_matching_reprs is not None:
                search_phrase_reprs.extend(
                    search_phrase_token._.holmes.derivation_matching_reprs
                )
                document_reprs.extend(multiword.direct_matching_reprs)
            if multiword.derivation_matching_reprs is not None:
                document_reprs.extend(multiword.derivation_matching_reprs)
                search_phrase_reprs.extend(
                    search_phrase_token._.holmes.direct_matching_reprs
                )

            for search_phrase_representation in search_phrase_reprs:
                for document_representation in document_reprs:
                    if search_phrase_representation == document_representation:
                        search_phrase_display_word = search_phrase_token._.holmes.lemma
                        return WordMatch(
                            search_phrase_token=search_phrase_token,
                            search_phrase_word=search_phrase_representation,
                            document_token=document_token,
                            first_document_token=document_token.doc[
                                multiword.token_indexes[0]
                            ],
                            last_document_token=document_token.doc[
                                multiword.token_indexes[-1]
                            ],
                            document_subword=None,
                            document_word=document_representation,
                            word_match_type=self.WORD_MATCH_TYPE_LABEL,
                            explanation=self._get_explanation(
                                search_phrase_display_word
                            ),
                        )
        return None

    def match_token(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
    ) -> Optional[WordMatch]:

        search_phrase_reprs = []
        document_reprs = []

        if search_phrase_token._.holmes.derivation_matching_reprs is not None:
            search_phrase_reprs.extend(
                search_phrase_token._.holmes.derivation_matching_reprs
            )
            document_reprs.extend(document_token._.holmes.direct_matching_reprs)
        if document_token._.holmes.derivation_matching_reprs is not None:
            document_reprs.extend(document_token._.holmes.derivation_matching_reprs)
            search_phrase_reprs.extend(
                search_phrase_token._.holmes.direct_matching_reprs
            )

        for search_phrase_representation in search_phrase_reprs:
            for document_representation in document_reprs:
                if search_phrase_representation == document_representation:
                    search_phrase_display_word = search_phrase_token._.holmes.lemma
                    return WordMatch(
                        search_phrase_token=search_phrase_token,
                        search_phrase_word=search_phrase_representation,
                        document_token=document_token,
                        first_document_token=document_token,
                        last_document_token=document_token,
                        document_subword=None,
                        document_word=document_representation,
                        word_match_type=self.WORD_MATCH_TYPE_LABEL,
                        extracted_word=self.get_extracted_word_for_token(
                            document_token, document_representation
                        ),
                        explanation=self._get_explanation(search_phrase_display_word),
                    )
        return None

    def match_subword(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_subword: Subword,
    ) -> Optional[WordMatch]:

        search_phrase_reprs = []
        document_reprs = []

        if search_phrase_token._.holmes.derivation_matching_reprs is not None:
            search_phrase_reprs.extend(
                search_phrase_token._.holmes.derivation_matching_reprs
            )
            document_reprs.extend(document_subword.direct_matching_reprs)
        if document_subword.derivation_matching_reprs is not None:
            document_reprs.extend(document_subword.derivation_matching_reprs)
            search_phrase_reprs.extend(
                search_phrase_token._.holmes.direct_matching_reprs
            )

        for search_phrase_representation in search_phrase_reprs:
            for document_representation in document_reprs:
                if search_phrase_representation == document_representation:
                    search_phrase_display_word = search_phrase_token._.holmes.lemma
                    return WordMatch(
                        search_phrase_token=search_phrase_token,
                        search_phrase_word=search_phrase_representation,
                        document_token=document_token,
                        first_document_token=document_token,
                        last_document_token=document_token,
                        document_subword=document_subword,
                        document_word=document_representation,
                        word_match_type=self.WORD_MATCH_TYPE_LABEL,
                        explanation=self._get_explanation(search_phrase_display_word),
                    )
        return None

    def add_words_matching_search_phrase_root_token(
        self, search_phrase: SearchPhrase
    ) -> None:
        if (
            search_phrase.root_token._.holmes.derived_lemma
            != search_phrase.root_token._.holmes.lemma
        ):
            search_phrase.add_word_information(
                search_phrase.root_token._.holmes.derived_lemma,
            )

    def add_reverse_dict_entries(
        self,
        reverse_dict: Dict[str, List[CorpusWordPosition]],
        doc: Doc,
        document_label: str,
    ) -> None:
        for token in doc:
            if token._.holmes.derived_lemma != token._.holmes.lemma:
                self.add_reverse_dict_entry(
                    reverse_dict,
                    token._.holmes.derived_lemma.lower(),
                    document_label,
                    token.i,
                    None,
                )
            for subword in token._.holmes.subwords:
                if subword.derived_lemma != subword.lemma:
                    self.add_reverse_dict_entry(
                        reverse_dict,
                        subword.derived_lemma.lower(),
                        document_label,
                        token.i,
                        subword.index,
                    )
