from typing import Optional, List, Dict, Tuple
from spacy.tokens import Token, Doc
from ..parsing import CorpusWordPosition, MultiwordSpan, ReverseIndexValue, SemanticMatchingHelper, Subword, Index, SearchPhrase

class WordMatchingStrategy:

    def __init__(self, semantic_matching_helper: SemanticMatchingHelper, perform_coreference_resolution: bool):
        self.semantic_matching_helper = semantic_matching_helper
        self.perform_coreference_resolution = perform_coreference_resolution

    def match_multiwords(self, search_phrase: SearchPhrase, search_phrase_token: Token, document_token: Token, document_multiwords: List[MultiwordSpan]) -> Optional["WordMatch"]:
        pass

    def match_token(self, search_phrase: SearchPhrase, search_phrase_token: Token, document_token: Token) -> Optional["WordMatch"]:
        pass

    def match_subword(self, search_phrase: SearchPhrase, search_phrase_token: Token, document_token: Token, document_subword: Subword) -> Optional["WordMatch"]:
        pass

    def add_words_matching_search_phrase_root_token(self, search_phrase:SearchPhrase) -> None:
        pass

    def add_reverse_dict_entries(self, doc:Doc, document_label:str, reverse_index: Dict[str, ReverseIndexValue]) -> None:
        pass

    @staticmethod
    def add_reverse_dict_entry(reverse_dict: Dict[str, ReverseIndexValue], document_label:str, key_word:str, value_word:str, token_index:int, subword_index:int, match_type:str) -> None:
        index = Index(token_index, subword_index)
        corpus_word_position = CorpusWordPosition(document_label, index)
        reverse_index_value = ReverseIndexValue(corpus_word_position, value_word, match_type)
        if key_word in reverse_dict.keys():
            if not any(1 for riv in reverse_dict[key_word] if riv.corpus_word_position == corpus_word_position):
                reverse_dict[key_word].append(reverse_index_value)
        else:
            reverse_dict[key_word] = [reverse_index_value]

    def get_extracted_word_for_token(self, token: Token, document_word:str) -> str:
        extracted_word = document_word
        if self.perform_coreference_resolution and token._.holmes.most_specific_coreferring_term_index is not None:
            most_specific_token = token.doc[token._.holmes.most_specific_coreferring_term_index]
            if token._.holmes.lemma != most_specific_token._.holmes.lemma:
                for multiword_span in most_specific_token._.holmes.multiword_spans:
                    extracted_word = multiword_span.text
                    break
                else:
                    extracted_word = most_specific_token.text
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
    similarity_measure -- for type *embedding*, the similarity between the two tokens,
        otherwise 1.0.
    involves_coreference -- *True* if *document_token* and *structurally_matched_document_token*
        are different.
    extracted_word -- the most specific term that corresponded to *document_word* within the
        coreference chain.
    depth -- the vertical difference in the ontology from *search_phrase_word* to *document_word*
        (can be negative).
    """

    def __init__(
            self, *, search_phrase_token, search_phrase_word, document_token,
            first_document_token, last_document_token, document_subword, document_word,
            word_match_type, depth=0, extracted_word=None, explanation):

        self.search_phrase_token = search_phrase_token
        self.search_phrase_word = search_phrase_word
        self.document_token = document_token
        self.first_document_token = first_document_token
        self.last_document_token = last_document_token
        self.document_subword = document_subword
        self.document_word = document_word
        self.word_match_type = word_match_type
        self.is_negated = False # will be set by StructuralMatcher
        self.is_uncertain = False # will be set by StructuralMatcher
        self.structurally_matched_document_token = None # will be set by StructuralMatcher
        self.extracted_word = extracted_word if extracted_word is not None else document_word
        self.depth = depth
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