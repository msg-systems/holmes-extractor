from typing import Optional, List
from spacy.tokens import Token
from ..parsing import MultiwordSpan, SemanticMatchingHelper, Subword, Index, SearchPhrase

class WordMatchingStrategy:

    def __init__(self, semantic_matching_helper: SemanticMatchingHelper):
        self.semantic_matching_helper == semantic_matching_helper

    def match_multiword(self, search_phrase: SearchPhrase, search_phrase_token: Token, document_token: Token, document_multiwords: List[MultiwordSpan]) -> Optional["WordMatch"]:
        pass

    def match_token(self, search_phrase: SearchPhrase, search_phrase_token: Token, document_token: Token) -> Optional["WordMatch"]:
        pass

    def match_subword(self, search_phrase: SearchPhrase, search_phrase_token: Token, document_token: Token, document_subword: Subword) -> Optional["WordMatch"]:
        pass

    def add_words_matching_search_phrase_root_token(self, search_phrase:SearchPhrase):
        pass

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
    similarity_measure -- for type *embedding*, the similarity between the two tokens,
        otherwise 1.0.
    involves_coreference -- *True* if *document_token* and *structurally_matched_document_token*
        are different.
    extracted_word -- the most specific term that corresponded to *document_word* within the
        coreference chain.
    depth -- currently unused (always 0).
    search_phrase_initial_question_word -- *True* if *search_phrase_token* is an initial question
        word or governs an initial question word.
    """

    def __init__(
            self, search_phrase_token, search_phrase_word, document_token,
            first_document_token, last_document_token, document_subword, document_word,
            word_match_type, explanation):

        self.search_phrase_token = search_phrase_token
        self.search_phrase_word = search_phrase_word
        self.document_token = document_token
        self.first_document_token = first_document_token
        self.last_document_token = last_document_token
        self.document_subword = document_subword
        self.document_word = document_word
        self.word_match_type = word_match_type
        self.is_negated = None # will be set by StructuralMatcher
        self.is_uncertain = None # will be set by StructuralMatcher
        self.structurally_matched_document_token = None # will be set by StructuralMatcher
        self.extracted_word = document_word
        self.depth = 0
        self.similarity_measure = 1.0
        self.explanation = explanation

    @property
    def involves_coreference(self):
        return self.document_token != self.structurally_matched_document_token

    def get_document_index(self):
        if self.document_subword is not None:
            subword_index = self.document_subword.index
        else:
            subword_index = None
        return Index(self.document_token.i, subword_index)