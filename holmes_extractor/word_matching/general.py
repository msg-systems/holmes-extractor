from typing import Optional, List, Dict
from spacy.tokens import Token, Doc
from ..parsing import (
    CorpusWordPosition,
    MultiwordSpan,
    SemanticMatchingHelper,
    Subword,
    Index,
    SearchPhrase,
)


class WordMatchingStrategy:
    """Parent class for all word matching strategies. Each strategy only implements those methods that are relevant to it."""

    def __init__(
        self,
        semantic_matching_helper: SemanticMatchingHelper,
        perform_coreference_resolution: bool,
    ):
        self.semantic_matching_helper = semantic_matching_helper
        self.perform_coreference_resolution = perform_coreference_resolution

    def match_multiwords(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_multiwords: List[MultiwordSpan],
    ) -> Optional["WordMatch"]:
        """Attempts to match a search phrase token to a list of multiwords headed by a document token and ordered by decreasing size."""
        pass

    def match_token(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
    ) -> Optional["WordMatch"]:
        """Attempts to match a search phrase token to a document token."""
        pass

    def match_subword(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_subword: Subword,
    ) -> Optional["WordMatch"]:
        """Attempts to match a search phrase token to a document subword (currently only relevant for German)."""
        pass

    def add_words_matching_search_phrase_root_token(
        self, search_phrase: SearchPhrase
    ) -> None:
        """Determines words that match a search phrase root token and notifies the *SearchPhrase* object of them."""
        pass

    def add_reverse_dict_entries(
        self, doc: Doc, document_label: str, reverse_dict: Dict[str, List[CorpusWordPosition]]
    ) -> None:
        """Determines words that match each token within a document and adds corresponding entries to the reverse dictionary."""
        pass

    @staticmethod
    def add_reverse_dict_entry(
        reverse_dict: Dict[str, List[CorpusWordPosition]],
        key_word: str,
        document_label: str,
        token_index: int,
        subword_index: int,
    ) -> None:
        """Adds a single entry to the reverse dictionary. Called by implementing classes."""
        index = Index(token_index, subword_index)
        corpus_word_position = CorpusWordPosition(document_label, index)
        if key_word in reverse_dict.keys():
            if corpus_word_position not in reverse_dict[key_word]:
                reverse_dict[key_word].append(corpus_word_position)
        else:
            reverse_dict[key_word] = [corpus_word_position]

    def get_extracted_word_for_token(self, token: Token, document_word: str) -> str:
        """Gets the extracted word for a token. If the token is part of a coreference chain, the extracted word is the most specific
        term within that chain; otherwise it is the same as the document word.
        """
        extracted_word = document_word
        if (
            self.perform_coreference_resolution
            and token._.holmes.most_specific_coreferring_term_index is not None
        ):
            most_specific_token = token.doc[
                token._.holmes.most_specific_coreferring_term_index
            ]
            if token._.holmes.lemma != most_specific_token._.holmes.lemma:
                if most_specific_token._.holmes.multiword_spans is not None:
                    for multiword_span in most_specific_token._.holmes.multiword_spans:
                        extracted_word = multiword_span.text
                        return extracted_word
                extracted_word = most_specific_token.text.lower()
        return extracted_word


class WordMatch:
    """A match between a searched phrase word and a document word.

    Properties:

    search_phrase_token -- the spaCy token from the search phrase.
    search_phrase_word -- the word that matched from the search phrase.
    document_token -- the spaCy token from the document.
    first_document_token -- the first token that matched from the document, which will equal
        *document_token* except with multiword matches.
    last_document_token -- the last token that matched from the document, which will equal
        *document_token* except with multiword matches.
    document_subword -- the subword from the token that matched, or *None* if the match was
        with the whole token.
    document_word -- the word or subword that matched structurally from the document.
    word_match_type -- *direct*, *entity*, *embedding*, or *derivation*.
    depth -- the vertical difference in the ontology from *search_phrase_word* to *document_word*
        (can be negative).
    extracted_word -- the most specific term that corresponded to *document_word* within the
        coreference chain.
    explanation -- a human-readable explanation of how the word match was determined designed
        e.g. for use as a tooltip.
    similarity_measure -- for type *embedding*, the similarity between the two tokens,
        otherwise 1.0.
    involves_coreference -- *True* if *document_token* and *structurally_matched_document_token*
        are different.
    """

    def __init__(
        self,
        *,
        search_phrase_token: Token,
        search_phrase_word: str,
        document_token: Token,
        first_document_token: Token,
        last_document_token: Token,
        document_subword: Subword,
        document_word: str,
        word_match_type: str,
        depth: int = 0,
        extracted_word: str = None,
        explanation: str
    ):

        self.search_phrase_token = search_phrase_token
        self.search_phrase_word = search_phrase_word
        self.document_token = document_token
        self.first_document_token = first_document_token
        self.last_document_token = last_document_token
        self.document_subword = document_subword
        self.document_word = document_word
        self.word_match_type = word_match_type
        self.is_negated = False  # will be set by StructuralMatcher
        self.is_uncertain = False  # will be set by StructuralMatcher
        self.structurally_matched_document_token = (
            None  # will be set by StructuralMatcher
        )
        self.extracted_word = (
            extracted_word if extracted_word is not None else document_word
        )
        self.depth = depth
        self.similarity_measure = 1.0
        self.explanation = explanation

    @property
    def involves_coreference(self) -> bool:
        return self.document_token != self.structurally_matched_document_token

    def get_document_index(self) -> Index:
        if self.document_subword is not None:
            subword_index = self.document_subword.index
        else:
            subword_index = None
        return Index(self.document_token.i, subword_index)
