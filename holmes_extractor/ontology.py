import rdflib
import urllib
from itertools import chain

class Ontology:
    """Loads information from an existing ontology and manages ontology matching.

    The ontology must follow the W3C OWL 2 standard. Search phrase words are matched
    to hyponyms, synonyms and individuals from within documents being searched.

    This class is designed for small ontologies that have been constructed by hand
    for specific use cases.

    Matching is case-insensitive.

    Args:

    ontology_path -- the path from where the ontology is to be loaded.
        See https://github.com/RDFLib/rdflib/.
    owl_class_type -- optionally overrides the OWL 2 URL for types.
    owl_individual_type -- optionally overrides the OWL 2 URL for individuals.
    owl_type_link -- optionally overrides the RDF URL for types.
    owl_synonym_type -- optionally overrides the OWL 2 URL for synonyms.
    owl_hyponym_type -- optionally overrides the RDF URL for hyponyms.
    symmetric_matching -- if 'True' means relationships are also taken into account where
        a search phrase word is a hyponym of a document word.
    """
    def __init__(self, ontology_path,
                 owl_class_type='http://www.w3.org/2002/07/owl#Class',
                 owl_individual_type='http://www.w3.org/2002/07/owl#NamedIndividual',
                 owl_type_link='http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                 owl_synonym_type='http://www.w3.org/2002/07/owl#equivalentClass',
                 owl_hyponym_type='http://www.w3.org/2000/01/rdf-schema#subClassOf',
                 symmetric_matching=False):
        self.path = ontology_path
        self._graph = rdflib.Graph()
        self._graph.load(ontology_path)
        self._owl_class_type = owl_class_type
        self._owl_individual_type = owl_individual_type
        self._owl_type_link = owl_type_link
        self._owl_synonym_type = owl_synonym_type
        self._owl_hyponym_type = owl_hyponym_type
        self._words, self._multiwords = self._get_words()
        self._match_dict = {}
        self.symmetric_matching=symmetric_matching

    class Entry:
        """Args:

        word -- the entry word.
        depth -- depth -- the number of hyponym relationships linking this entry and the
            search phrase word in whose dictionary this entry exists.
        is_individual -- *True* if this entry is an individual, *False* if it is a hyponym
            or synonym.
        """
        def __init__(self, word, depth, is_individual):
            self.word = word
            self.depth = depth
            self.is_individual = is_individual

    def add_to_dictionary(self, search_phrase_word):
        """Generates the dictionary for a search_phrase word."""
        search_phrase_word = search_phrase_word.lower()
        if search_phrase_word not in self._match_dict:
            entry_set = set()
            self._match_dict[search_phrase_word] = entry_set
            for class_id, type_link, metaclass_id in self._get_classes():
                entry_word = self._get_entry_word(class_id).lower()
                if entry_word == search_phrase_word:
                    self._recursive_add_to_dict(
                            entry_set, entry_word, class_id, set(), 0, False, False,
                            self.symmetric_matching)
            for class_id, type_link, metaclass_id in self._get_individuals():
                entry_word = self._get_entry_word(class_id).lower()
                if entry_word == search_phrase_word:
                    self._recursive_add_to_dict(
                            entry_set, entry_word, class_id, set(), 0, True, False,
                            self.symmetric_matching)

    def contains(self, word):
        """Returns whether or not a word is present in the loaded ontology."""
        return word.lower() in self._words

    def contains_multiword(self, multiword):
        """Returns whether or not a multiword is present in the loaded ontology."""
        return multiword.lower() in self._multiwords

    def matches(self, search_phrase_word, candidate_word):
        """Returns whether or not *candidate_word* matches *search_phrase_word*.

        Matching is defined as *candidate_word* being a hyponym, synonym or individual instance
        of *search_phrase_word*. Where *symmetric_matching==True*, matching also encompasses
        *search_phrase_word* being a hyponym of *candidate_word*."""
        if search_phrase_word.lower() in self._match_dict.keys():
            for entry in self._match_dict[search_phrase_word.lower()]:
                if entry.word.lower() == candidate_word.lower():
                    return entry
        return None

    def get_words_matching(self, search_phrase_word):
        """Returns the synonyms, hyponyms and individual instances of *search_phrase_word*,
            as well as the hypernyms where *symmetric_matching==True*"""
        if search_phrase_word.lower() in self._match_dict.keys():
            return set(map(lambda entry: entry.word, self._match_dict[search_phrase_word]))
        else:
            return []

    def get_words_matching_lower_case(self, search_phrase_word):
        """Returns the synonyms, hyponyms and individual instances of *search_phrase_word*,
            as well as the hypernyms where *symmetric_matching==True*
            All words are set to lower case.
        """
        if search_phrase_word.lower() in self._match_dict.keys():
            return set(map(lambda entry: entry.word.lower(), self._match_dict[search_phrase_word]))
        else:
            return []

    def _get_classes(self):
        """Returns all classes from the loaded ontology."""
        return self._graph.triples((None, rdflib.term.URIRef(self._owl_type_link),
                rdflib.term.URIRef(self._owl_class_type)))

    def _get_individuals(self):
        """Returns all classes from the loaded ontology."""
        return self._graph.triples((None, rdflib.term.URIRef(self._owl_type_link),
                rdflib.term.URIRef(self._owl_individual_type)))

    def _get_words(self):
        """Finds all words in the loaded ontology and returns multiwords in a separate list."""
        words = []
        multiwords = []
        for class_id, type_link, metaclass_id in chain(
                self._get_classes(), self._get_individuals()):
            entry_word = self._get_entry_word(class_id)
            words.append(entry_word.lower())
            if ' ' in entry_word:
                multiwords.append(entry_word.lower())
        return words, multiwords

    def _recursive_add_to_dict(self, entry_set, word, working_entry_url, visited,
            depth, is_individual, is_hypernym, symmetric):
        """Adds synonyms and hyponyms of a search phrase word to its dictionary.

        Keyword arguments:

        entry_set -- the set of matching synonyms, hyponyms and instances and optionally hypernyms.
        word -- the word whose dictionary entry is being built up.
        working_entry_url -- the URL that is to be added.
        visited -- an array of entry URLs that have already been processed.
        depth -- the number of hyponym relationships linking the working entry and the
            search phrase word
        is_individual -- 'True' if the working entry is an individual, *False* if it
            is a hyponym or synonym
        is_hypernym -- 'True' if the working entry is a hypernym, which can only happen
            if 'symmetric==True'
        symmetric -- 'True' if hypernyms should be matched as well as synonyms, hyponyms
            and individuals.
        """
        if working_entry_url not in visited:
            visited.add(working_entry_url)
            working_entry_word = self._get_entry_word(working_entry_url)
            if word.lower() != working_entry_word.lower():
                entry_set.add(self.Entry(working_entry_word, depth, is_individual))
            if not is_hypernym:
                for entry, type_link, metaclass_id in self._graph.triples((None,
                        rdflib.term.URIRef(self._owl_hyponym_type), working_entry_url)):
                    self._recursive_add_to_dict(entry_set, word, entry, visited,
                            depth+1, False, False, symmetric)
                for entry, type_link, metaclass_id in self._graph.triples((None,
                        rdflib.term.URIRef(self._owl_type_link), working_entry_url)):
                    self._recursive_add_to_dict(entry_set, word, entry, visited,
                            depth+1, True, False, symmetric)
            for entry, type_link, metaclass_id in self._graph.triples((None,
                    rdflib.term.URIRef(self._owl_synonym_type), working_entry_url)):
                self._recursive_add_to_dict(entry_set, word, entry, visited,
                        depth, False, False, symmetric)
            for metaclass_id, type_link, entry in self._graph.triples((working_entry_url,
                    rdflib.term.URIRef(self._owl_synonym_type), None)):
                self._recursive_add_to_dict(entry_set, word, entry, visited,
                        depth, False, False, symmetric)
            if symmetric and depth <= 0:
                for metaclass_id, type_link, entry in self._graph.triples((working_entry_url,
                        rdflib.term.URIRef(self._owl_hyponym_type), None)):
                    self._recursive_add_to_dict(entry_set, word, entry, visited,
                            depth-1, False, True, symmetric)
                if is_individual:
                    for metaclass_id, type_link, entry in self._graph.triples((working_entry_url,
                            rdflib.term.URIRef(self._owl_type_link), None)):
                        if entry != rdflib.term.URIRef(self._owl_individual_type):
                            self._recursive_add_to_dict(entry_set, word, entry, visited,
                                    depth-1, False, True, symmetric)
                # setting depth to a negative value ensures the hypernym
                # can never qualify as being equally or more specific than the original match. It
                # also prevents recursive traversal down other hyponym branches.

    def _get_entry_word(self, class_id):
        """Converts an OWL URL into an entry word

        The fragment is retrieved from the URL and underscores are replaced with spaces.
        """
        return str(urllib.parse.urlparse(class_id).fragment).replace('_', ' ')

    def get_most_general_hypernym_ancestor(self, word):
        """Returns the most general hypernym ancestor of 'word', one of the most general ancestors
            if there are several, or 'word' if 'word' is not found in the ontology or has
            no hypernym. If there are several hypernym ancestors at the same level, the first one
            in the alphabet is returned.
        """
        matching_set = set()
        for clazz in (clazz for clazz, t, m in self._get_classes() if
                self._get_entry_word(clazz).lower()==word.lower()):
            this_class_set = set()
            self._recursive_add_to_dict(this_class_set, word, clazz, set(), 0, False,
                    False, True)
            matching_set |= this_class_set
        for individual in (individual for individual, t, m in self._get_individuals() if
                self._get_entry_word(individual).lower() == word.lower()):
            this_individual_set = set()
            self._recursive_add_to_dict(this_individual_set, word, individual, set(), 0,
                    True, False, True)
            matching_set |= this_individual_set
        matching_list = sorted(matching_set, key=lambda entry: (entry.depth, entry.word))
        matching_list = list(filter(lambda entry: entry.depth < 0, matching_list))
        if len(matching_list) == 0:
            return word
        else:
            return matching_list[0].word
