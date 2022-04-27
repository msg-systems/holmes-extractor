from typing import Optional, List, Dict
from spacy.tokens import Token, Doc
from .general import WordMatch, WordMatchingStrategy
from ..parsing import (
    MultiwordSpan,
    CorpusWordPosition,
    Subword,
    SearchPhrase,
)


class DirectWordMatchingStrategy(WordMatchingStrategy):

    WORD_MATCH_TYPE_LABEL = "direct"

    @staticmethod
    def _get_explanation(search_phrase_display_word: str) -> str:
        return "".join(("Matches ", search_phrase_display_word.upper(), " directly."))

    def match_multiwords(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_multiwords: List[MultiwordSpan],
    ) -> Optional[WordMatch]:

        if len(search_phrase_token._.holmes.lemma.split()) == 1:
            return None
        for (
            search_phrase_representation
        ) in search_phrase_token._.holmes.direct_matching_reprs:
            for multiword in document_multiwords:
                for document_representation in multiword.direct_matching_reprs:
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

        for (
            search_phrase_representation
        ) in search_phrase_token._.holmes.direct_matching_reprs:
            for (
                document_representation
            ) in document_token._.holmes.direct_matching_reprs:
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

        for (
            search_phrase_representation
        ) in search_phrase_token._.holmes.direct_matching_reprs:
            for document_representation in document_subword.direct_matching_reprs:
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
        for word in search_phrase.root_token._.holmes.direct_matching_reprs:
            search_phrase.add_word_information(word)

    def add_reverse_dict_entries(
        self,
        reverse_dict: Dict[str, List[CorpusWordPosition]],
        doc: Doc,
        document_label: str,
    ) -> None:
        for token in doc:
            for representation in token._.holmes.direct_matching_reprs:
                self.add_reverse_dict_entry(
                    reverse_dict,
                    representation.lower(),
                    document_label,
                    token.i,
                    None,
                )
            for subword in token._.holmes.subwords:
                for representation in subword.direct_matching_reprs:
                    self.add_reverse_dict_entry(
                        reverse_dict,
                        representation.lower(),
                        document_label,
                        token.i,
                        subword.index,
                    )
