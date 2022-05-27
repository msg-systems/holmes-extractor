from typing import Optional, List, Dict, Union
from holmes_extractor.ontology import Ontology
from spacy.tokens import Token, Doc
from .general import WordMatch, WordMatchingStrategy
from ..parsing import (
    HolmesDictionary,
    CorpusWordPosition,
    MultiwordSpan,
    SemanticMatchingHelper,
    Subword,
    SearchPhrase,
)


class OntologyWordMatchingStrategy(WordMatchingStrategy):
    """
    The patent US8155946 associated with this code has been made available under the MIT licence, 
    with kind permission from AstraZeneca.
    """

    WORD_MATCH_TYPE_LABEL = "ontology"

    ONTOLOGY_DEPTHS_TO_NAMES = {
        -4: "an ancestor",
        -3: "a great-grandparent",
        -2: "a grandparent",
        -1: "a parent",
        0: "a synonym",
        1: "a child",
        2: "a grandchild",
        3: "a great-grandchild",
        4: "a descendant",
    }

    def _get_explanation(self, search_phrase_display_word: str, depth: int) -> str:
        depth = min(depth, 4)
        depth = max(depth, -4)
        return "".join(
            (
                "Is ",
                self.ONTOLOGY_DEPTHS_TO_NAMES[depth],
                " of ",
                search_phrase_display_word.upper(),
                " in the ontology.",
            )
        )

    def __init__(
        self,
        semantic_matching_helper: SemanticMatchingHelper,
        perform_coreference_resolution: bool,
        ontology: Ontology,
        analyze_derivational_morphology: bool,
        ontology_reverse_derivational_dict: Optional[Dict[str, str]],
    ):
        self.ontology = ontology
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.ontology_reverse_derivational_dict = ontology_reverse_derivational_dict
        super().__init__(semantic_matching_helper, perform_coreference_resolution)

    def match_multiwords(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
        document_multiwords: List[MultiwordSpan],
    ) -> Optional[WordMatch]:

        for search_phrase_representation in self._get_reprs(
            search_phrase_token._.holmes
        ):
            for multiword in document_multiwords:
                entry = self.ontology.matches(
                    search_phrase_representation, self._get_reprs(multiword)
                )
                if entry is not None:
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
                        document_word=entry.word,
                        word_match_type=self.WORD_MATCH_TYPE_LABEL,
                        depth=entry.depth,
                        explanation=self._get_explanation(
                            search_phrase_display_word, entry.depth
                        ),
                    )
        return None

    def match_token(
        self,
        search_phrase: SearchPhrase,
        search_phrase_token: Token,
        document_token: Token,
    ) -> Optional[WordMatch]:

        for search_phrase_representation in self._get_reprs(
            search_phrase_token._.holmes
        ):
            entry = self.ontology.matches(
                search_phrase_representation, self._get_reprs(document_token._.holmes)
            )
            if entry is not None:
                search_phrase_display_word = search_phrase_token._.holmes.lemma
                return WordMatch(
                    search_phrase_token=search_phrase_token,
                    search_phrase_word=search_phrase_representation,
                    document_token=document_token,
                    first_document_token=document_token,
                    last_document_token=document_token,
                    document_subword=None,
                    document_word=entry.word,
                    word_match_type=self.WORD_MATCH_TYPE_LABEL,
                    extracted_word=self.get_extracted_word_for_token(
                        document_token, entry.word
                    ),
                    depth=entry.depth,
                    explanation=self._get_explanation(
                        search_phrase_display_word, entry.depth
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

        for search_phrase_representation in self._get_reprs(
            search_phrase_token._.holmes
        ):
            entry = self.ontology.matches(
                search_phrase_representation, self._get_reprs(document_subword)
            )
            if entry is not None:
                search_phrase_display_word = search_phrase_token._.holmes.lemma
                return WordMatch(
                    search_phrase_token=search_phrase_token,
                    search_phrase_word=search_phrase_representation,
                    document_token=document_token,
                    first_document_token=document_token,
                    last_document_token=document_token,
                    document_subword=document_subword,
                    document_word=entry.word,
                    word_match_type=self.WORD_MATCH_TYPE_LABEL,
                    depth=entry.depth,
                    explanation=self._get_explanation(
                        search_phrase_display_word, entry.depth
                    ),
                )
        return None

    def add_words_matching_search_phrase_root_token(
        self, search_phrase: SearchPhrase
    ) -> None:
        search_phrase_reprs = search_phrase.root_token._.holmes.direct_matching_reprs[:]
        if (
            self.analyze_derivational_morphology
            and search_phrase.root_token._.holmes.derivation_matching_reprs is not None
        ):
            search_phrase_reprs.extend(
                search_phrase.root_token._.holmes.derivation_matching_reprs
            )
        for word in search_phrase_reprs:
            for entry in self.ontology.get_matching_entries(word):
                for repr in entry.reprs:
                    search_phrase.add_word_information(repr)

    def add_reverse_dict_entries(
        self,
        reverse_dict: Dict[str, List[CorpusWordPosition]],
        doc: Doc,
        document_label: str,
    ) -> None:
        for token in doc:
            odw = self.semantic_matching_helper.get_ontology_defined_multiword(
                token, self.ontology
            )
            if odw is not None:
                for representation in odw.direct_matching_reprs:
                    self.add_reverse_dict_entry(
                        reverse_dict,
                        representation.lower(),
                        document_label,
                        token.i,
                        None,
                    )
                if (
                    self.analyze_derivational_morphology
                    and odw.derivation_matching_reprs is not None
                ):
                    for representation in odw.derivation_matching_reprs:
                        self.add_reverse_dict_entry(
                            reverse_dict,
                            representation.lower(),
                            document_label,
                            token.i,
                            None,
                        )

    def _get_reprs(
        self, repr_bearer: Union[HolmesDictionary, Subword, MultiwordSpan]
    ) -> List[str]:
        reprs = repr_bearer.direct_matching_reprs
        if (
            self.analyze_derivational_morphology
            and repr_bearer.derivation_matching_reprs is not None
        ):
            reprs.extend(repr_bearer.derivation_matching_reprs)
        return reprs
