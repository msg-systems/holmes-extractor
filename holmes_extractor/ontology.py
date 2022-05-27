from typing import List, Dict, Set, Union, Tuple, Optional
from holmes_extractor.errors import OntologyObjectSharedBetweenManagersError
from spacy.compat import Literal
import urllib
import rdflib


class Entry:
    """Args:

    word -- the entry word.
    depth -- depth -- the number of hyponym relationships linking this entry and the
        search phrase word in whose dictionary this entry exists.
    is_individual -- *True* if this entry is an individual, *False* if it is a hyponym
        or synonym.
    """

    def __init__(self, word: str, depth: int, is_individual: bool):
        self.word = word
        self.reprs = [word.lower()]
        self.depth = depth
        self.is_individual = is_individual

    def __eq__(self, other) -> bool:
        if not isinstance(other, Entry):
            return False
        return self.word == other.word

    def __hash__(self) -> int:
        return hash(self.word)


class Ontology:
    """Loads information from an existing ontology and manages ontology matching.

    The ontology must follow the W3C OWL 2 standard. Search phrase words are matched
    to hyponyms, synonyms and individuals from within documents being searched.

    This class is designed for small ontologies that have been constructed by hand
    for specific use cases.

    Holmes is not designed to support changes to a loaded ontology via direct
    calls to the methods of this class. It is also not permitted to share a single instance
    of this class between multiple Manager instances: instead, a separate Ontology instance
    pointing to the same path should be created for each Manager.

    Matching is case-insensitive.

    Args:

    ontology_path -- the path from where the ontology is to be loaded, or a list of
        several such paths. See https://github.com/RDFLib/rdflib/.
    owl_class_type -- optionally overrides the OWL 2 URL for types.
    owl_individual_type -- optionally overrides the OWL 2 URL for individuals.
    owl_type_link -- optionally overrides the RDF URL for types.
    owl_synonym_type -- optionally overrides the OWL 2 URL for synonyms.
    owl_hyponym_type -- optionally overrides the RDF URL for hyponyms.
    symmetric_matching -- if 'True' means relationships are also taken into account where
        a search phrase word is a hyponym of a document word. Defaults to 'False'
    """

    def __init__(
        self,
        ontology_path: Union[str, List[str]],
        owl_class_type: str = "http://www.w3.org/2002/07/owl#Class",
        owl_individual_type: str = "http://www.w3.org/2002/07/owl#NamedIndividual",
        owl_type_link: str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        owl_synonym_type: str = "http://www.w3.org/2002/07/owl#equivalentClass",
        owl_hyponym_type: str = "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        symmetric_matching: bool = False,
    ):
        self.status: int = 0
        self.path = ontology_path
        self._graph = rdflib.Graph()
        if isinstance(self.path, list):
            for entry in ontology_path:
                self._graph.load(entry)
        else:
            self._graph.load(ontology_path)
        self.owl_class_type = owl_class_type
        self.owl_individual_type = owl_individual_type
        self.owl_type_link = owl_type_link
        self.owl_synonym_type = owl_synonym_type
        self.owl_hyponym_type = owl_hyponym_type
        self.match_dict: Dict[str, Set[Entry]] = {}
        self.symmetric_matching = symmetric_matching
        self.populate_dictionary()
        self.refresh_words()
        self.status = 1

    def populate_dictionary(self) -> None:
        """Generates the dictionary from search phrase words to matching document words."""

        for class_id, _, _ in self._get_classes():
            entry_word = self._get_entry_word(class_id)
            if entry_word in self.match_dict:
                entry_set = self.match_dict[entry_word]
            else:
                self.match_dict[entry_word] = entry_set = set()
            self._recursive_add_to_dict(
                entry_set,
                entry_word,
                class_id,
                set(),
                0,
                False,
                False,
                self.symmetric_matching,
            )
        for class_id, _, _ in self._get_individuals():
            entry_word = self._get_entry_word(class_id)
            if entry_word in self.match_dict:
                entry_set = self.match_dict[entry_word]
            else:
                self.match_dict[entry_word] = entry_set = set()
            self._recursive_add_to_dict(
                entry_set,
                entry_word,
                class_id,
                set(),
                0,
                True,
                False,
                self.symmetric_matching,
            )

    def contains_word(self, word: str) -> bool:
        """Returns whether or not a multiword is present in the loaded ontology."""
        return word.lower() in self.words

    def contains_multiword(self, multiword: str) -> bool:
        """Returns whether or not a multiword is present in the loaded ontology."""
        return multiword.lower() in self._multiwords

    def matches(
        self, search_phrase_word: str, candidate_words: List[str]
    ) -> Optional[Entry]:
        """Returns whether or not any of *candidate_words* matches *search_phrase_word*.

        Matching is defined as *candidate_word* being a hyponym, synonym or individual instance
        of *search_phrase_word*. Where *symmetric_matching==True*, matching also encompasses
        *search_phrase_word* being a hyponym of *candidate_word*."""
        if search_phrase_word.lower() in self.match_dict:
            for entry in self.match_dict[search_phrase_word.lower()]:
                for candidate_word in candidate_words:
                    if candidate_word.lower() in entry.reprs:
                        return entry
        return None

    def get_matching_entries(self, search_phrase_word: str) -> Set[Entry]:
        """Returns entries for the synonyms, hyponyms and individual instances of
        *search_phrase_word*, as well as the hypernyms where *symmetric_matching==True*.
        All words are set to lower case.
        """
        if search_phrase_word.lower() in self.match_dict:
            return self.match_dict[search_phrase_word.lower()]
        else:
            return set()

    def refresh_words(self) -> None:
        self.words = []
        self._multiwords = []
        for key in self.match_dict:
            self.words.append(key)
            if " " in key:
                self._multiwords.append(key)
            for entry in self.match_dict[key]:
                for repr in entry.reprs:
                    self.words.append(repr)
                    if " " in repr:
                        self._multiwords.append(repr)
        self.status += 1

    def get_most_general_hypernym_ancestor(self, word: str) -> str:
        """Returns the most general hypernym ancestor of 'word', one of the most general ancestors
        if there are several, or 'word' if 'word' is not found in the ontology or has
        no hypernym. If there are several hypernym ancestors at the same level, the first one
        in the alphabet is returned.
        """
        matching_set = set()
        for clazz in (
            clazz
            for clazz, t, m in self._get_classes()
            if self._get_entry_word(clazz) == word.lower()
        ):
            this_class_set: Set[Entry] = set()
            self._recursive_add_to_dict(
                this_class_set, word, clazz, set(), 0, False, False, True
            )
            matching_set |= this_class_set
        for individual in (
            individual
            for individual, t, m in self._get_individuals()
            if self._get_entry_word(individual) == word.lower()
        ):
            this_individual_set: Set[Entry] = set()
            self._recursive_add_to_dict(
                this_individual_set, word, individual, set(), 0, True, False, True
            )
            matching_set |= this_individual_set
        matching_list = sorted(
            matching_set, key=lambda entry: (entry.depth, entry.word)
        )
        matching_list = list(filter(lambda entry: entry.depth < 0, matching_list))
        if len(matching_list) == 0:
            return word
        else:
            return matching_list[0].word

    def _get_entry_word(self, class_id: str, *, lower_case: bool = True) -> str:
        """Converts an OWL URL into an entry word

        The fragment is retrieved from the URL and underscores are replaced with spaces.
        """
        entry_word = str(
            urllib.parse.urlparse(class_id).fragment
        ).replace(  # type:ignore[attr-defined]
            "_", " "
        )
        if lower_case:
            entry_word = entry_word.lower()
        return entry_word

    def _recursive_add_to_dict(
        self,
        entry_set: Set[Entry],
        word: str,
        working_entry_url: str,
        visited: Set[str],
        depth: int,
        is_individual: bool,
        is_hypernym: bool,
        symmetric: bool,
    ):
        """Adds synonyms and hyponyms of a search phrase word to its dictionary.

        Keyword arguments:

        entry_set -- the set of matching synonyms, hyponyms and instances and optionally hypernyms.
        word -- the word whose dictionary entry is being built up.
        working_entry_url -- the URL that is to be added.
        visited -- a set of entry URLs that have already been processed.
        depth -- the number of hyponym relationships linking the working entry and the
            search phrase word
        is_individual -- *True* if the working entry is an individual, *False* if it
            is a hyponym or synonym
        is_hypernym -- *True* if the working entry is a hypernym, which can only happen
            if *symmetric==True*
        symmetric -- *True* if hypernyms should be matched as well as synonyms, hyponyms
            and individuals.
        """
        if working_entry_url not in visited:
            visited.add(working_entry_url)
            working_entry_word = self._get_entry_word(
                working_entry_url, lower_case=False
            )
            if word.lower() != working_entry_word.lower():
                entry_set.add(Entry(working_entry_word, depth, is_individual))
            if not is_hypernym:  # prevent recursive traversal of adjacent branches
                for entry, _, _ in self._graph.triples(
                    (
                        None,
                        rdflib.term.URIRef(self.owl_hyponym_type),
                        working_entry_url,
                    )  # type:ignore [arg-type]
                ):
                    self._recursive_add_to_dict(
                        entry_set,
                        word,
                        entry,
                        visited,
                        depth + 1,
                        False,
                        False,
                        symmetric,
                    )
                for entry, _, _ in self._graph.triples(
                    (
                        None,
                        rdflib.term.URIRef(self.owl_type_link),
                        working_entry_url,
                    )  # type:ignore [arg-type]
                ):
                    self._recursive_add_to_dict(
                        entry_set,
                        word,
                        entry,
                        visited,
                        depth + 1,
                        True,
                        False,
                        symmetric,
                    )
            for entry, _, _ in self._graph.triples(
                (
                    None,
                    rdflib.term.URIRef(self.owl_synonym_type),
                    working_entry_url,
                )  # type:ignore [arg-type]
            ):
                self._recursive_add_to_dict(
                    entry_set, word, entry, visited, depth, False, False, symmetric
                )
            for _, _, entry in self._graph.triples(
                (
                    working_entry_url,
                    rdflib.term.URIRef(self.owl_synonym_type),
                    None,
                )  # type:ignore [arg-type]
            ):
                self._recursive_add_to_dict(
                    entry_set, word, entry, visited, depth, False, False, symmetric
                )
            if symmetric and depth <= 0:
                for _, _, entry in self._graph.triples(
                    (
                        working_entry_url,
                        rdflib.term.URIRef(self.owl_hyponym_type),
                        None,
                    )  # type:ignore [arg-type]
                ):
                    self._recursive_add_to_dict(
                        entry_set,
                        word,
                        entry,
                        visited,
                        depth - 1,
                        False,
                        True,
                        symmetric,
                    )
                if is_individual:
                    for _, _, entry in self._graph.triples(
                        (  # type:ignore [arg-type]
                            working_entry_url,
                            rdflib.term.URIRef(self.owl_type_link),
                            None,
                        )
                    ):
                        if entry != rdflib.term.URIRef(self.owl_individual_type):
                            self._recursive_add_to_dict(
                                entry_set,
                                word,
                                entry,
                                visited,
                                depth - 1,
                                False,
                                True,
                                symmetric,
                            )
                # setting depth to a negative value ensures the hypernym
                # can never qualify as being equally or more specific than the original match.

    def _get_classes(self) -> Set[Tuple]:
        """Returns all classes from the loaded ontology."""
        return self._graph.triples(
            (
                None,
                rdflib.term.URIRef(self.owl_type_link),
                rdflib.term.URIRef(self.owl_class_type),
            )
        )

    def _get_individuals(self) -> Set[Tuple]:
        """Returns all classes from the loaded ontology."""
        return self._graph.triples(
            (
                None,
                rdflib.term.URIRef(self.owl_type_link),
                rdflib.term.URIRef(self.owl_individual_type),
            )
        )
