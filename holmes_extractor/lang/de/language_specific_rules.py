from typing import Optional, Dict, List, Tuple
import importlib
from string import punctuation
from wasabi import Printer  # type: ignore[import]
from spacy.language import Language
from spacy.tokens import Token, Doc
from ...parsing import (
    SemanticAnalyzer,
    SemanticMatchingHelper,
    MatchImplication,
    PhraseletTemplate,
    SemanticDependency,
    Subword,
)


class LanguageSpecificSemanticAnalyzer(SemanticAnalyzer):

    language_name = "German"

    noun_pos = ["NOUN", "PROPN", "ADJ"]

    matchable_pos = [
        "ADJ",
        "ADP",
        "ADV",
        "NOUN",
        "NUM",
        "PROPN",
        "VERB",
        "AUX",
        "X",
        "INTJ",
    ]

    predicate_head_pos = ["VERB", "AUX"]

    adjectival_predicate_head_pos = ["AUX"]

    adjectival_predicate_subject_pos = ["NOUN", "PROPN", "PRON"]

    adjectival_predicate_subject_dep = "sb"

    adjectival_predicate_predicate_pos = "ADV"

    adjectival_predicate_predicate_dep = "pd"

    modifier_dep = "nk"

    spacy_noun_to_preposition_dep = "mnr"

    spacy_verb_to_preposition_dep = "mo"

    holmes_noun_to_preposition_dep = "mnrposs"

    holmes_verb_to_preposition_dep = "moposs"

    conjunction_deps = ["cj", "cd", "punct", "app"]

    interrogative_pronoun_tags = ["PWAT", "PWAV", "PWS"]

    semantic_dependency_excluded_tags = ["ART"]

    generic_pronoun_lemmas = ["jemand", "etwas"]

    or_lemma = "oder"

    mark_child_dependencies_copied_to_siblings_as_uncertain = False

    maximum_mentions_in_coreference_chain = 3

    maximum_word_distance_in_coreference_chain = 300

    sibling_marker_deps = ["cj", "app"]

    entity_labels_to_corresponding_lexemes = {
        "PER": "person",
        "LOC": "ort",
        "ORG": "organisation",
        "MISC": "sache",
    }

    whose_lemma = "wessen"

    # Only words at least this long are examined for possible subwords
    minimum_length_for_subword_search = 10

    # Part-of-speech pos examined for subwords (additionally to part-of-speech tags below)
    pos_for_subword_search = "X"

    # Part-of-speech tags examined for subwords
    # Verbs are not examined because the separable parts that would typically be found as
    # subwords are too short to be found.
    tag_for_subword_search = ("NE", "NNE", "NN", "TRUNC", "ADJA", "ADJD", "$(")

    # Absolute minimum length of a subword.
    minimum_subword_length = 3

    # Subwords at least this long are preferred.
    minimum_normal_subword_length = 6

    # Subwords longer than this are likely not to be atomic and solutions that split them up are
    # preferred
    maximum_realistic_subword_length = 12

    # Scoring bonus where a Fugen-S follows a whitelisted ending
    # (one where a Fugen-S is normally expected)
    fugen_s_after_whitelisted_ending_bonus = 5

    # Scoring bonus where a Fugen-S follows an ending where it is neither expected nor disallowed
    fugen_s_after_non_whitelisted_non_blacklisted_ending_bonus = 3

    # Both words around a Fugen-S have to be at least this long for the scoring bonus to be applied
    fugen_s_whitelist_bonus_surrounding_word_minimum_length = 5

    # Endings after which a Fugen-S is normally expected
    fugen_s_ending_whitelist = (
        "tum",
        "ling",
        "ion",
        "tät",
        "heit",
        "keit",
        "schaft",
        "sicht",
        "ung",
    )

    # Endings after which a Fugen-S is normally disallowed
    fugen_s_ending_blacklist = (
        "a",
        "ä",
        "e",
        "i",
        "o",
        "ö",
        "u",
        "ü",
        "nt",
        "sch",
        "s",
        "ß",
        "st",
        "tz",
        "z",
    )

    # Whitelisted subwords
    subword_whitelist = "haltig"

    # Blacklisted subwords
    subword_blacklist = (
        "igkeit",
        "igkeiten",
        "digkeit",
        "digkeiten",
        "schaft",
        "schaften",
        "keit",
        "keiten",
        "lichkeit",
        "lichkeiten",
        "tigten",
        "tigung",
        "tigungen",
        "barkeit",
        "barkeiten",
        "heit",
        "heiten",
        "ung",
        "ungen",
        "aften",
        "erung",
        "erungen",
        "mungen",
        "tig",
    )

    # Bigraphs of two consonants that can occur at the start of a subword.
    subword_start_consonant_bigraph_whitelist = (
        "bl",
        "br",
        "ch",
        "cl",
        "cr",
        "dr",
        "fl",
        "fr",
        "gl",
        "gm",
        "gn",
        "gr",
        "kl",
        "kn",
        "kr",
        "kw",
        "pf",
        "ph",
        "pl",
        "pn",
        "pr",
        "ps",
        "rh",
        "sc",
        "sh",
        "sk",
        "sl",
        "sm",
        "sp",
        "st",
        "sw",
        "sz",
        "th",
        "tr",
        "vl",
        "vr",
        "wr",
        "zw",
    )

    # Bigraphs of two consonants that can occur at the end of a subword.
    # Bigraphs where the second consonant is 's' are always allowed.
    subword_end_consonant_bigraph_whitelist = (
        "bb",
        "bs",
        "bt",
        "ch",
        "ck",
        "ct",
        "dd",
        "ds",
        "dt",
        "ff",
        "fs",
        "ft",
        "gd",
        "gg",
        "gn",
        "gs",
        "gt",
        "hb",
        "hd",
        "hf",
        "hg",
        "hk",
        "hl",
        "hm",
        "hn",
        "hp",
        "hr",
        "hs",
        "ht",
        "ks",
        "kt",
        "lb",
        "lc",
        "ld",
        "lf",
        "lg",
        "lk",
        "ll",
        "lm",
        "ln",
        "lp",
        "ls",
        "lt",
        "lx",
        "lz",
        "mb",
        "md",
        "mk",
        "mm",
        "mp",
        "ms",
        "mt",
        "mx",
        "nb",
        "nd",
        "nf",
        "ng",
        "nk",
        "nn",
        "np",
        "ns",
        "nt",
        "nx",
        "nz",
        "pf",
        "ph",
        "pp",
        "ps",
        "pt",
        "rb",
        "rc",
        "rd",
        "rf",
        "rg",
        "rk",
        "rl",
        "rm",
        "rn",
        "rp",
        "rr",
        "rs",
        "rt",
        "rx",
        "rz",
        "sk",
        "sl",
        "sp",
        "ss",
        "st",
        "th",
        "ts",
        "tt",
        "tz",
        "xt",
        "zt",
        "ßt",
    )

    # Letters that can represent vowel sounds
    vowels = ("a", "e", "i", "o", "u", "ä", "ö", "ü", "y")

    # Subwords used in analysis but not recorded on the Holmes dictionary instances. At present
    # the code only supports these in word-final position; word-initial position would require
    # a code change.
    non_recorded_subword_list = ("lein", "chen")

    # Subword solutions that scored higher than this are regarded as probably wrong and so are
    # not recorded.
    maximum_acceptable_subword_score = 8

    def __init__(self, *, nlp: Language, vectors_nlp: Language) -> None:
        super().__init__(nlp=nlp, vectors_nlp=vectors_nlp)
        if "EditTreeLemmatizer" in str(type(nlp.get_pipe("lemmatizer"))):  # > spaCy 3.3
            if importlib.util.find_spec("spacy_lookups_data") is None:
                msg = Printer()
                msg.fail(
                    "To use Holmes with the loaded spaCy model, please install the spacy-lookups-data package with 'pip install spacy-lookups-data'"
                )
                raise ImportError("spacy_lookups_data")
            nlp.remove_pipe("lemmatizer")
            nlp.add_pipe("lemmatizer", before="ner")
            lemmatizer = nlp.get_pipe("lemmatizer")
            lemmatizer.initialize()

    def is_oov(self, word: str) -> bool:
        working_word = word.lower()
        if not self.vectors_nlp.vocab[working_word].is_oov:
            return False
        if len(word) == 1:
            return True
        working_word = "".join((working_word[0].upper(), working_word[1:]))
        return self.vectors_nlp.vocab[working_word].is_oov

    def is_separable_prefix(self, token: Token) -> bool:
        return token.dep_ == "svp" or (
            token.dep_ == "mo"
            and token.pos_ == "ADP"
            and len(list(token.children)) == 0
        )

    def add_subwords(
        self, token: Token, subword_cache: Dict[str, List[Subword]]
    ) -> None:
        """Adds any subwords to *token._.holmes*."""

        class PossibleSubword:
            """A subword within a possible solution.

            text -- the text
            char_start_index -- the character start index of the subword within the word.
            fugen_s_status --
            '1' if the preceding word has an ending that normally has a Fugen-S,
            '2' if the preceding word has an ending that precludes using a Fugen-S,
            '0' otherwise.
            """

            def __init__(
                self, text: str, char_start_index: int, fugen_s_status: int
            ) -> None:
                self.text = text
                self.char_start_index = char_start_index
                self.fugen_s_status = fugen_s_status

        def get_subword(lemma: str, initial_index: int, length: int) -> Optional[str]:
            # find the shortest subword longer than length.
            for end_index in range(initial_index + length, len(lemma) + 1):
                possible_word = lemma[initial_index:end_index]
                if (
                    (
                        not self.is_oov(possible_word)
                        or possible_word in self.subword_whitelist
                    )
                    and len(possible_word) >= 2
                    and (
                        possible_word[0] in self.vowels
                        or possible_word[1] in self.vowels
                        or possible_word[:2]
                        in self.subword_start_consonant_bigraph_whitelist
                    )
                    and (
                        possible_word[-1] in self.vowels
                        or possible_word[-2] in self.vowels
                        or possible_word[-2:]
                        in self.subword_end_consonant_bigraph_whitelist
                    )
                ):
                    return possible_word
            return None

        def score(possible_solution: List[PossibleSubword]) -> int:
            # Lower scores are better.
            number = 0
            for subword in possible_solution:
                # subwords shorter than minimum_normal_subword_length: penalty of 2
                if len(subword.text) < self.minimum_normal_subword_length:
                    number += 2 * (
                        self.minimum_normal_subword_length - len(subword.text)
                    )
                # subwords longer than 12: penalty of 2
                elif len(subword.text) > self.maximum_realistic_subword_length:
                    number += 2 * (
                        len(subword.text) - self.maximum_realistic_subword_length
                    )
                # fugen-s after a whitelist ending
                if subword.fugen_s_status == 2:
                    number -= self.fugen_s_after_whitelisted_ending_bonus
                # fugen-s after an ending that is neither whitelist nor blacklist
                elif subword.fugen_s_status == 1:
                    number -= (
                        self.fugen_s_after_non_whitelisted_non_blacklisted_ending_bonus
                    )
            return number

        def scan_recursively_for_subwords(
            lemma: str, initial_index: int = 0
        ) -> Optional[List[PossibleSubword]]:

            if initial_index == 0:  # only need to check on the initial (outermost) call
                for char in lemma:
                    if not char.isalpha() and char != "-":
                        return None
            if initial_index + 1 < len(lemma) and lemma[initial_index] == "-":
                return scan_recursively_for_subwords(lemma, initial_index + 1)
            lengths = list(
                range(self.minimum_subword_length, 1 + len(lemma) - initial_index)
            )
            possible_solutions = []
            working_subword = None
            for length in lengths:
                if working_subword is not None and len(working_subword) >= length:
                    # we are catching up with the length already returned by get_subword
                    continue
                working_subword = get_subword(lemma, initial_index, length)
                if (
                    working_subword is None
                    or working_subword in self.subword_blacklist
                    or "-" in working_subword
                ):
                    continue
                possible_solution = [PossibleSubword(working_subword, initial_index, 0)]
                if (
                    (initial_index + len(working_subword) == len(lemma))
                    or (
                        initial_index + len(working_subword) + 1 == len(lemma)
                        and lemma[-1] == "-"
                    )
                    or (
                        initial_index + len(working_subword) + 2 == len(lemma)
                        and lemma[-2:] == "s-"
                    )
                ):
                    # we have reached the end of the word
                    possible_solutions.append(possible_solution)
                    break
                following_subwords = scan_recursively_for_subwords(
                    lemma, initial_index + len(working_subword)
                )
                if following_subwords is not None:
                    possible_solution.extend(following_subwords)
                    possible_solutions.append(possible_solution)
                if (
                    initial_index + len(working_subword) + 2 < len(lemma)
                    and lemma[
                        initial_index
                        + len(working_subword) : initial_index
                        + len(working_subword)
                        + 2
                    ]
                    == "s-"
                ):
                    following_initial_index = initial_index + len(working_subword) + 2
                elif (
                    initial_index + len(working_subword) + 1 < len(lemma)
                    and lemma[initial_index + len(working_subword)] == "s"
                ):
                    following_initial_index = initial_index + len(working_subword) + 1
                else:
                    continue
                possible_solution = [PossibleSubword(working_subword, initial_index, 0)]
                following_subwords = scan_recursively_for_subwords(
                    lemma, following_initial_index
                )
                if following_subwords is not None:
                    for ending in self.fugen_s_ending_whitelist:
                        if working_subword.endswith(ending):
                            following_subwords[0].fugen_s_status = 2
                    if (
                        following_subwords[0].fugen_s_status == 0
                        and len(working_subword)
                        >= self.fugen_s_whitelist_bonus_surrounding_word_minimum_length
                        and len(following_subwords[0].text)
                        >= self.fugen_s_whitelist_bonus_surrounding_word_minimum_length
                    ):
                        # if the first does not have a whitelist ending and one of the words is
                        # short, do not give the score bonus
                        following_subwords[0].fugen_s_status = 1
                        for ending in self.fugen_s_ending_blacklist:
                            # blacklist ending: take the bonus away again
                            if working_subword.endswith(ending):
                                following_subwords[0].fugen_s_status = 0
                    possible_solution.extend(following_subwords)
                    possible_solutions.append(possible_solution)
            if len(possible_solutions) > 0:
                possible_solutions = sorted(possible_solutions, key=score)
                return possible_solutions[0]
            else:
                return None

        def get_lemmatization_doc(
            possible_subwords: List[PossibleSubword], pos: str
        ) -> Doc:
            # We retrieve the lemma for each subword by calling spaCy. To reduce the
            # overhead, we concatenate the subwords in the form:
            # Subword1. Subword2. Subword3
            entry_words = []
            for counter, _ in enumerate(possible_subwords):
                if counter + 1 == len(possible_subwords) and pos == "ADJ":
                    entry_words.append(possible_subwords[counter].text)
                else:
                    entry_words.append(possible_subwords[counter].text.capitalize())
            subword_lemmatization_string = " . ".join(entry_words)
            return self.spacy_parse_for_lemmas(subword_lemmatization_string)

        if (
            not (
                token.tag_ in self.tag_for_subword_search
                or token.pos_ in self.pos_for_subword_search
            )
            or (
                len(token._.holmes.lemma) < self.minimum_length_for_subword_search
                and "-" not in token._.holmes.lemma
            )
            or token._.holmes.lemma in punctuation
        ):
            return
        if token.text in subword_cache:
            cached_subwords = subword_cache[token.text]
            for cached_subword in cached_subwords:
                token._.holmes.subwords.append(
                    Subword(
                        token.i,
                        cached_subword.index,
                        cached_subword.text,
                        cached_subword.lemma,
                        cached_subword.derived_lemma,
                        self.get_vector(cached_subword.lemma),
                        cached_subword.char_start_index,
                        cached_subword.dependent_index,
                        cached_subword.dependency_label,
                        cached_subword.governor_index,
                        cached_subword.governing_dependency_label,
                    )
                )
        else:
            working_subwords = []
            possible_subwords = scan_recursively_for_subwords(token._.holmes.lemma)
            if (
                possible_subwords is None
                or score(possible_subwords) > self.maximum_acceptable_subword_score
            ):
                return
            if len(possible_subwords) == 1 and token._.holmes.lemma.isalpha():
                # not ... isalpha(): hyphenation
                subword_cache[token.text] = []
            else:
                index = 0
                if token._.holmes.lemma[0] == "-":
                    # with truncated nouns, the righthand siblings may actually occur to the left
                    # of the head noun
                    head_sibling = token.doc[
                        token._.holmes.token_or_lefthand_sibling_index
                    ]
                    if len(head_sibling._.holmes.righthand_siblings) > 0:
                        indexes = token._.holmes.get_sibling_indexes(token.doc)
                        first_sibling = token.doc[indexes[0]]
                        first_sibling_possible_subwords = scan_recursively_for_subwords(
                            first_sibling._.holmes.lemma
                        )
                        if first_sibling_possible_subwords is not None:
                            first_sibling_lemmatization_doc = get_lemmatization_doc(
                                first_sibling_possible_subwords, token.pos_
                            )
                            final_subword_counter = (
                                len(first_sibling_possible_subwords) - 1
                            )
                            if (
                                final_subword_counter > 0
                                and first_sibling_possible_subwords[
                                    final_subword_counter
                                ].text
                                in self.non_recorded_subword_list
                            ):
                                final_subword_counter -= 1
                            for counter in range(final_subword_counter):
                                first_sibling_possible_subword = (
                                    first_sibling_possible_subwords[counter]
                                )
                                if (
                                    first_sibling_possible_subword.text
                                    in self.non_recorded_subword_list
                                ):
                                    continue
                                text = first_sibling.text[
                                    first_sibling_possible_subword.char_start_index : first_sibling_possible_subword.char_start_index
                                    + len(first_sibling_possible_subword.text)
                                ]
                                lemma = first_sibling_lemmatization_doc[
                                    counter * 2
                                ].lemma_.lower()
                                derived_lemma = self.derived_holmes_lemma(None, lemma)
                                working_subwords.append(
                                    Subword(
                                        first_sibling.i,
                                        index,
                                        text,
                                        lemma,
                                        derived_lemma,
                                        self.get_vector(lemma),
                                        first_sibling_possible_subword.char_start_index,
                                        None,
                                        None,
                                        None,
                                        None,
                                    )
                                )
                                index += 1
                lemmatization_doc = get_lemmatization_doc(possible_subwords, token.pos_)
                for counter, possible_subword in enumerate(possible_subwords):
                    possible_subword = possible_subwords[counter]
                    if possible_subword.text in self.non_recorded_subword_list:
                        continue
                    text = token.text[
                        possible_subword.char_start_index : possible_subword.char_start_index
                        + len(possible_subword.text)
                    ]
                    lemma = lemmatization_doc[counter * 2].lemma_.lower()
                    derived_lemma = self.derived_holmes_lemma(None, lemma)
                    working_subwords.append(
                        Subword(
                            token.i,
                            index,
                            text.lower(),
                            lemma,
                            derived_lemma,
                            self.get_vector(lemma),
                            possible_subword.char_start_index,
                            None,
                            None,
                            None,
                            None,
                        )
                    )
                    index += 1
                if token._.holmes.lemma[-1] == "-":
                    # with truncated nouns, the righthand siblings may actually occur to the left
                    # of the head noun
                    head_sibling = token.doc[
                        token._.holmes.token_or_lefthand_sibling_index
                    ]
                    if len(head_sibling._.holmes.righthand_siblings) > 0:
                        indexes = token._.holmes.get_sibling_indexes(token.doc)
                        last_sibling_index = indexes[-1]
                        if token.i != last_sibling_index:
                            last_sibling = token.doc[last_sibling_index]
                            last_sibling_possible_subwords = (
                                scan_recursively_for_subwords(
                                    last_sibling._.holmes.lemma
                                )
                            )
                            if last_sibling_possible_subwords is not None:
                                last_sibling_lemmatization_doc = get_lemmatization_doc(
                                    last_sibling_possible_subwords, token.pos_
                                )
                                for counter in range(
                                    1, len(last_sibling_possible_subwords)
                                ):
                                    last_sibling_possible_subword = (
                                        last_sibling_possible_subwords[counter]
                                    )
                                    if (
                                        last_sibling_possible_subword.text
                                        in self.non_recorded_subword_list
                                    ):
                                        continue
                                    text = last_sibling.text[
                                        last_sibling_possible_subword.char_start_index : last_sibling_possible_subword.char_start_index
                                        + len(last_sibling_possible_subword.text)
                                    ]
                                    lemma = last_sibling_lemmatization_doc[
                                        counter * 2
                                    ].lemma_.lower()
                                    derived_lemma = self.derived_holmes_lemma(
                                        None, lemma
                                    )
                                    working_subwords.append(
                                        Subword(
                                            last_sibling.i,
                                            index,
                                            text.lower(),
                                            lemma,
                                            derived_lemma,
                                            self.get_vector(lemma),
                                            last_sibling_possible_subword.char_start_index,
                                            None,
                                            None,
                                            None,
                                            None,
                                        )
                                    )
                                    index += 1

                if (
                    index > 1
                ):  # if only one subword was found, no need to record it on ._.holmes
                    for counter, working_subword in enumerate(working_subwords):
                        if counter > 0:
                            dependency_label = "intcompound"
                            dependent_index = counter - 1
                        else:
                            dependency_label = None
                            dependent_index = None
                        if counter + 1 < len(working_subwords):
                            governing_dependency_label = "intcompound"
                            governor_index = counter + 1
                        else:
                            governing_dependency_label = None
                            governor_index = None
                        working_subword = working_subwords[counter]
                        token._.holmes.subwords.append(
                            Subword(
                                working_subword.containing_token_index,
                                working_subword.index,
                                working_subword.text.lower(),
                                working_subword.lemma,
                                working_subword.derived_lemma,
                                self.get_vector(working_subword.lemma),
                                working_subword.char_start_index,
                                dependent_index,
                                dependency_label,
                                governor_index,
                                governing_dependency_label,
                            )
                        )
                if token._.holmes.lemma.isalpha():  # caching only where no hyphenation
                    subword_cache[token.text] = token._.holmes.subwords
        if len(token._.holmes.subwords) > 1 and "nicht" in (
            subword.lemma for subword in token._.holmes.subwords
        ):
            token._.holmes.is_negated = True

    def set_negation(self, token: Token) -> None:
        """Marks any negation on *token*. A token is negative if it or one of its ancestors
        has a negation word as a syntactic (not semantic!) child.
        """
        if token._.holmes.is_negated is not None:
            return
        for child in token.children:
            if child._.holmes.lemma in (
                "nicht",
                "kein",
                "keine",
                "nie",
            ) or child._.holmes.lemma.startswith("nirgend"):
                token._.holmes.is_negated = True
                return
        if token.dep_ == "ROOT":
            token._.holmes.is_negated = False
            return
        if token.i != token.head.i:
            self.set_negation(token.head)
        token._.holmes.is_negated = token.head._.holmes.is_negated

    def correct_auxiliaries_and_passives(self, token: Token) -> None:
        """Wherever auxiliaries and passives are found, derive the semantic information
        from the syntactic information supplied by spaCy.
        """

        def correct_auxiliaries_and_passives_recursively(
            token, processed_auxiliary_indexes
        ):
            if token.i not in processed_auxiliary_indexes:
                processed_auxiliary_indexes.append(token.i)
                if (
                    token.pos_ == "AUX"
                    or token.tag_.startswith("VM")
                    or token.tag_.startswith("VA")
                ) and len(
                    [
                        dependency
                        for dependency in token._.holmes.children
                        if dependency.child_index >= 0
                        and self.is_separable_prefix(token.doc[dependency.child_index])
                    ]
                ) == 0:  # 'vorhaben'
                    for dependency in (
                        dependency
                        for dependency in token._.holmes.children
                        if token.doc[dependency.child_index].pos_ in ("VERB", "AUX")
                        and token.doc[dependency.child_index].dep_ in ("oc", "pd")
                    ):
                        token._.holmes.is_matchable = False
                        child = token.doc[dependency.child_index]
                        self.move_information_between_tokens(token, child)
                        # VM indicates a modal verb, which has to be marked as uncertain
                        if token.tag_.startswith("VM") or dependency.is_uncertain:
                            for child_dependency in child._.holmes.children:
                                child_dependency.is_uncertain = True
                        # 'er ist froh zu kommen' / 'er ist schwer zu erreichen'
                        # set dependency label to 'arg' because semantic role could be either
                        # subject or object
                        if token._.holmes.lemma == "sein" and (
                            len(
                                [
                                    child_dependency
                                    for child_dependency in child._.holmes.children
                                    if child_dependency.label == "pm"
                                    and child_dependency.child_token(token.doc).tag_
                                    == "PTKZU"
                                ]
                            )
                            > 0
                            or child.tag_ == "VVIZU"
                        ):
                            for new_dependency in (
                                new_dependency
                                for new_dependency in child._.holmes.children
                                if new_dependency.label == "sb"
                            ):
                                new_dependency.label = "arg"
                                new_dependency.is_uncertain = True
                        # passive construction
                        if (
                            token._.holmes.lemma == "werden"
                            and child.tag_
                            not in ("VVINF", "VAINF", "VAFIN", "VAINF", "VVFIN")
                        ) and len(
                            [
                                c
                                for c in token.children
                                if c.dep_ == "oc" and c.lemma_ == "haben"
                            ]
                        ) == 0:
                            for (
                                child_or_sib
                            ) in child._.holmes.loop_token_and_righthand_siblings(
                                token.doc
                            ):
                                # mark syntactic subject as semantic object
                                for grandchild_dependency in [
                                    grandchild_dependency
                                    for grandchild_dependency in child_or_sib._.holmes.children
                                    if grandchild_dependency.label == "sb"
                                ]:
                                    grandchild_dependency.label = "oa"
                                # mark syntactic object as synctactic subject, removing the
                                # preposition 'von' or 'durch' from the construction and marking
                                # it as non-matchable
                                for grandchild_dependency in (
                                    gd
                                    for gd in child_or_sib._.holmes.children
                                    if gd.child_index >= 0
                                ):
                                    grandchild = token.doc[
                                        grandchild_dependency.child_index
                                    ]
                                    if (
                                        grandchild_dependency.label == "sbp"
                                        and grandchild._.holmes.lemma in ("von", "vom")
                                    ) or (
                                        grandchild_dependency.label == "mo"
                                        and grandchild._.holmes.lemma
                                        in ("von", "vom", "durch")
                                    ):
                                        grandchild._.holmes.is_matchable = False
                                        for (
                                            great_grandchild_dependency
                                        ) in grandchild._.holmes.children:
                                            if (
                                                child_or_sib.i
                                                != great_grandchild_dependency.child_index
                                            ):
                                                child_or_sib._.holmes.children.append(
                                                    SemanticDependency(
                                                        child_or_sib.i,
                                                        great_grandchild_dependency.child_index,
                                                        "sb",
                                                        dependency.is_uncertain,
                                                    )
                                                )
                                        child_or_sib._.holmes.remove_dependency_with_child_index(
                                            grandchild_dependency.child_index
                                        )
            for syntactic_child in token.children:
                correct_auxiliaries_and_passives_recursively(
                    syntactic_child, processed_auxiliary_indexes
                )

        if token.dep_ == "ROOT":
            correct_auxiliaries_and_passives_recursively(token, [])

    def handle_relative_constructions(self, token: Token) -> None:
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.child_index >= 0
            and dependency.child_token(token.doc).tag_ in ("PRELS", "PRELAT")
            and dependency.child_token(token.doc).dep_ != "par"
        ):
            counter = dependency.child_index
            while counter > token.sent.start:
                # find the antecedent
                counter -= 1
                working_token = token.doc[counter]
                if (
                    working_token.pos_ in ("NOUN", "PROPN")
                    and working_token.dep_ not in self.sibling_marker_deps
                ):
                    working_dependency = None
                    for antecedent in (
                        antecedent
                        for antecedent in working_token._.holmes.loop_token_and_righthand_siblings(
                            token.doc
                        )
                        if antecedent.i != token.i
                    ):
                        # add new dependency from the verb to the antecedent
                        working_dependency = SemanticDependency(
                            token.i, antecedent.i, dependency.label, True
                        )
                        token._.holmes.children.append(working_dependency)
                    # the last antecedent before the pronoun is not uncertain, so reclassify it
                    if working_dependency is not None:
                        working_dependency.is_uncertain = False
                        # remove the dependency from the verb to the relative pronoun
                        token._.holmes.remove_dependency_with_child_index(
                            dependency.child_index
                        )
                        # label the relative pronoun as a grammatical token pointing to its
                        # direct antecedent
                        dependency.child_token(token.doc)._.holmes.children = [
                            SemanticDependency(
                                dependency.child_index,
                                0 - (working_dependency.child_index + 1),
                                None,
                            )
                        ]

    def holmes_lemma(self, token: Token) -> str:
        """Relabel the lemmas of separable verbs in sentences like 'er steht auf' to incorporate
        the entire separable verb to facilitate matching.
        """
        if token.pos_ in ("VERB", "AUX") and token.tag_ not in (
            "VAINF",
            "VMINF",
            "VVINF",
            "VVIZU",
        ):
            for child in token.children:
                if self.is_separable_prefix(child):
                    child_lemma = child.lemma_.lower()
                    if child_lemma == "einen":
                        child_lemma = "ein"
                    return "".join([child_lemma, token.lemma_.lower()])
        if token.tag_ == "APPRART":
            if token.lemma_.lower() in ("im", "ins"):
                return "in"
            if token.lemma_.lower() in ("am", "ans"):
                return "an"
            if token.lemma_.lower() == "beim":
                return "bei"
            if token.lemma_.lower() == "vom":
                return "von"
            if token.lemma_.lower() in ("zum", "zur"):
                return "zu"
        # sometimes adjectives retain their inflectional endings
        if token.tag_ in ("ADJA", "ADJD") and len(token.lemma_) > 5:
            if token.lemma_.lower().endswith("ten"):
                working_lemma = token.lemma_.lower()[:-2]
            elif token.lemma_.lower().endswith("tes"):
                working_lemma = token.lemma_.lower()[:-2]
            elif token.lemma_.lower().endswith("ter"):
                working_lemma = token.lemma_.lower()[:-2]
            elif token.lemma_.lower().endswith("te"):
                working_lemma = token.lemma_.lower()[:-1]
            else:
                working_lemma = token.lemma_.lower()
            # see if the adjective is a participle
            participle_test_doc = self.spacy_parse_for_lemmas(
                " ".join(("Jemand hat", working_lemma))
            )
            return participle_test_doc[2].lemma_.lower()
        return token.lemma_.lower()

    _ung_ending_blacklist = ("sprung", "schwung", "nibelung")

    def language_specific_derived_holmes_lemma(self, token: Token, lemma: str) -> str:
        """token is None where *lemma* belongs to a subword"""

        # verbs with 'ieren' -> 'ation'
        if (
            (token is None or token.pos_ == "VERB")
            and len(lemma) > 9
            and lemma.endswith("ieren")
        ):
            working_lemma = "".join((lemma[:-5], "ation"))
            if not self.is_oov(working_lemma):
                return working_lemma
        # nouns with 'ierung' -> 'ation'
        if (
            (token is None or token.pos_ == "NOUN")
            and len(lemma) > 10
            and lemma.endswith("ierung")
        ):
            working_lemma = "".join((lemma[:-6], "ation"))
            if not self.is_oov(working_lemma):
                return working_lemma
        # nominalization with 'ung'
        if (token is None or token.tag_ == "NN") and lemma.endswith("ung"):
            for word in self._ung_ending_blacklist:
                if lemma.endswith(word):
                    return lemma
            if (
                lemma.endswith("erung") and not lemma.endswith("ierung")
            ) or lemma.endswith("elung"):
                return "".join((lemma[:-3], "n"))
            elif (
                lemma.endswith("lung")
                and len(lemma) >= 5
                and lemma[-5] not in ("a", "e", "i", "o", "u", "ä", "ö", "ü", "h")
            ):
                return "".join((lemma[:-4], "eln"))
            return "".join((lemma[:-3], "en"))
        # nominalization with 'heit', 'keit'
        if (token is None or token.tag_ == "NN") and (
            lemma.endswith("keit") or lemma.endswith("heit")
        ):
            return lemma[:-4]
        if (
            (token is None or token.pos_ in ("NOUN", "PROPN"))
            and len(lemma) > 6
            and (lemma.endswith("chen") or lemma.endswith("lein"))
        ):
            # len > 6: because e.g. Dach and Loch have lemmas 'dachen' and 'lochen'
            working_lemma = lemma[-12:-4]
            # replace umlauts in the last 8 characters of the derived lemma
            working_lemma = (
                working_lemma.replace("ä", "a").replace("ö", "o").replace("ü", "u")
            )
            working_lemma = "".join((lemma[:-12], working_lemma))
            if not self.is_oov(working_lemma):
                return working_lemma
            if lemma[-4] == "l":  # 'lein' where original word ends in 'l'
                second_working_lemma = "".join((working_lemma, "l"))
                if not self.is_oov(working_lemma):
                    return second_working_lemma
            second_working_lemma = lemma[:-4]  # 'Löffelchen'
            if not self.is_oov(second_working_lemma):
                return second_working_lemma
            if lemma[-4] == "l":  # 'Schlüsselein'
                second_working_lemma = "".join((second_working_lemma, "l"))
                if not self.is_oov(second_working_lemma):
                    return second_working_lemma
            return working_lemma
        if (
            (token is None or token.tag_ == "NN")
            and lemma.endswith("e")
            and len(lemma) > 1
            and not lemma[-2] in self.vowels
        ):
            # for comparability with diminutive forms, e.g. äuglein <-> auge
            return lemma[:-1]
        return lemma

    def perform_language_specific_tasks(self, token: Token) -> None:

        # Because separable verbs are conflated into a single lemma, remove the dependency
        # from the verb to the preposition
        if (
            self.is_separable_prefix(token)
            and token.head.pos_ in ("VERB", "AUX")
            and token.head.tag_ not in ("VAINF", "VMINF", "VVINF", "VVIZU")
        ):
            token.head._.holmes.remove_dependency_with_child_index(token.i)
            token._.holmes.is_matchable = False

        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.label in ("mo", "mnr", "pg", "op")
        ):
            child = dependency.child_token(token.doc)
            for child_dependency in (
                child_dependency
                for child_dependency in child._.holmes.children
                if child_dependency.label == "nk"
                and token.i != child_dependency.child_index
                and child.pos_ == "ADP"
            ):
                if dependency.label in ("mnr", "pg", "op") and dependency.child_token(
                    token.doc
                )._.holmes.lemma in ("von", "vom"):
                    token._.holmes.children.append(
                        SemanticDependency(
                            token.i, child_dependency.child_index, "pobjo"
                        )
                    )
                    # pobjO from English 'of'
                    child._.holmes.is_matchable = False
                elif dependency.label in ("mnr") and dependency.child_token(
                    token.doc
                )._.holmes.lemma in ("durch"):
                    token._.holmes.children.append(
                        SemanticDependency(
                            token.i, child_dependency.child_index, "pobjb"
                        )
                    )
                    # pobjB from English 'by'
                else:
                    token._.holmes.children.append(
                        SemanticDependency(
                            token.i,
                            child_dependency.child_index,
                            "pobjp",
                            dependency.is_uncertain or child_dependency.is_uncertain,
                        )
                    )

        # # where a 'moposs' or 'mnrposs' dependency has been added and the preposition is not
        # 'von' or 'vom' add a corresponding uncertain 'pobjp'
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.label in ["moposs", "mnrposs"]
        ):
            child = dependency.child_token(token.doc)
            for child_dependency in (
                child_dependency
                for child_dependency in child._.holmes.children
                if child_dependency.label == "nk"
                and token.i != child_dependency.child_index
                and child._.holmes.is_matchable
            ):
                token._.holmes.children.append(
                    SemanticDependency(
                        token.i, child_dependency.child_index, "pobjp", True
                    )
                )

        # Loop through the structure around a dependent verb to find the lexical token at which
        # to add new dependencies, and find out whether it is active or passive so we know
        # whether to add an 'sb' or an 'oa'.
        def find_target_tokens_and_dependency_recursively(
            token: Token, visited: List[Token] = None
        ) -> Tuple[List[Token], str]:
            if visited is None:
                visited = []
            visited.append(token.i)
            tokens_to_return = []
            target_dependency = "sb"
            # Loop through grammatical tokens. 'dependency.child_index + token.i != -1' would mean
            # a grammatical token were pointing to itself (should never happen!)
            if (
                len(
                    [
                        dependency
                        for dependency in token._.holmes.children
                        if dependency.child_index < 0
                        and dependency.child_index + token.i != -1
                    ]
                )
                > 0
            ):
                for dependency in (
                    dependency
                    for dependency in token._.holmes.children
                    if dependency.child_index < 0
                    and dependency.child_index + token.i != -1
                ):
                    # resolve the grammatical token pointer
                    child_token = token.doc[0 - (dependency.child_index + 1)]
                    # passive construction
                    if token._.holmes.lemma == "werden" and child_token.tag_ not in (
                        "VVINF",
                        "VAINF",
                        "VAFIN",
                        "VAINF",
                        "VVFIN",
                    ):
                        target_dependency = "oa"
                    if child_token.i not in visited:
                        (
                            new_tokens,
                            new_target_dependency,
                        ) = find_target_tokens_and_dependency_recursively(
                            child_token, visited
                        )
                        tokens_to_return.extend(new_tokens)
                        if new_target_dependency == "oa":
                            target_dependency = "oa"
                    else:
                        tokens_to_return.append(token)
            else:
                # we have reached the target token
                tokens_to_return.append(token)
            return tokens_to_return, target_dependency

        def has_morphs(token, *morphs) -> bool:
            if len(morphs) == 0:
                raise ValueError("At least one morph is required.")
            for morph in morphs:
                if not morph in token.morph:
                    return False
            return True

        # 'Der Mann hat xxx, es zu yyy' and similar structures
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.label in ("oc", "oa", "mo", "re")
            and token.pos_ in ("VERB", "AUX")
            and dependency.child_token(token.doc).pos_ in ("VERB", "AUX")
        ):
            dependencies_to_add = []
            (
                target_tokens,
                target_dependency,
            ) = find_target_tokens_and_dependency_recursively(
                dependency.child_token(token.doc)
            )
            # with um ... zu structures the antecedent subject is always the subject of the
            # dependent clause, unlike with 'zu' structures without the 'um'
            if (
                len(
                    [
                        other_dependency
                        for other_dependency in target_tokens[0]._.holmes.children
                        if other_dependency.child_token(token.doc)._.holmes.lemma
                        == "um"
                        and other_dependency.child_token(token.doc).tag_ == "KOUI"
                    ]
                )
                == 0
            ):
                # er hat ihm vorgeschlagen, etwas zu tun
                for other_dependency in (
                    other_dependency
                    for other_dependency in token._.holmes.children
                    if other_dependency.label == "da"
                ):
                    dependencies_to_add.append(other_dependency)
                if len(dependencies_to_add) == 0:
                    # er hat ihn gezwungen, etwas zu tun
                    # We have to distinguish this type of 'oa' relationship from dependent
                    # clauses and reflexive pronouns ('er entschied sich, ...')
                    for other_dependency in (
                        other_dependency
                        for other_dependency in token._.holmes.children
                        if other_dependency.label == "oa"
                        and other_dependency.child_token(token.doc).pos_
                        not in ("VERB", "AUX")
                        and other_dependency.child_token(token.doc).tag_ != "PRF"
                    ):
                        dependencies_to_add.append(other_dependency)
            if len(dependencies_to_add) == 0:
                # We haven't found any object dependencies, so take the subject dependency
                for other_dependency in (
                    other_dependency
                    for other_dependency in token._.holmes.children
                    if other_dependency.label == "sb"
                ):
                    dependencies_to_add.append(other_dependency)
            for target_token in target_tokens:
                for other_dependency in (
                    other_dependency
                    for other_dependency in dependencies_to_add
                    if target_token.i != other_dependency.child_index
                    and len(
                        [
                            dep
                            for dep in target_token._.holmes.children
                            if dep.label == target_dependency and not dep.is_uncertain
                        ]
                    )
                    == 0
                ):
                    # these dependencies are always uncertain
                    target_token._.holmes.children.append(
                        SemanticDependency(
                            target_token.i,
                            other_dependency.child_index,
                            target_dependency,
                            True,
                        )
                    )

        # 'Der Löwe bat den Hund, die Katze zu jagen' and similar structures
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.label == "oc"
            and token.pos_ == "NOUN"
            and dependency.child_token(token.doc).pos_ in ("VERB", "AUX")
        ):
            (
                target_tokens,
                target_dependency,
            ) = find_target_tokens_and_dependency_recursively(
                dependency.child_token(token.doc)
            )
            for target_token in (
                target_token
                for target_token in target_tokens
                if target_token.i != token.i
                and len(
                    [
                        dep
                        for dep in target_token._.holmes.children
                        if dep.label == target_dependency and not dep.is_uncertain
                    ]
                )
                == 0
            ):
                target_token._.holmes.children.append(
                    SemanticDependency(target_token.i, token.i, target_dependency, True)
                )

        # 'er dachte darüber nach, es zu tun' and similar structures
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.label == "op"
            and dependency.child_token(token.doc).tag_ == "PROAV"
        ):
            child_token = dependency.child_token(token.doc)
            for child_dependency in (
                child_dependency
                for child_dependency in child_token._.holmes.children
                if child_dependency.label == "re"
                and child_dependency.child_token(token.doc).pos_ in ("VERB", "AUX")
            ):
                (
                    target_tokens,
                    target_dependency,
                ) = find_target_tokens_and_dependency_recursively(
                    child_dependency.child_token(token.doc)
                )
                for other_dependency in (
                    other_dependency
                    for other_dependency in token._.holmes.children
                    if other_dependency.label == "sb"
                ):
                    for target_token in (
                        target_token
                        for target_token in target_tokens
                        if len(
                            [
                                dep
                                for dep in target_token._.holmes.children
                                if dep.label == target_dependency
                                and not dep.is_uncertain
                            ]
                        )
                        == 0
                    ):
                        target_token._.holmes.children.append(
                            SemanticDependency(
                                target_token.i,
                                other_dependency.child_index,
                                target_dependency,
                                True,
                            )
                        )

        # 'er war froh, etwas zu tun'
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.label == "nk"
            and token.pos_ in ("NOUN", "PROPN")
            and token.dep_ == "sb"
            and dependency.child_token(token.doc).pos_
            == self.adjectival_predicate_predicate_pos
        ):
            child_token = dependency.child_token(token.doc)
            relevant_dependencies = [
                child_dependency
                for child_dependency in child_token._.holmes.children
                if child_dependency.label in ("oc", "re")
                and child_dependency.child_token(token.doc).pos_ in ("VERB", "AUX")
            ]
            for grandchild_token in (
                gd.child_token(token.doc)
                for gd in child_token._.holmes.children
                if gd.label in ("mo", "op")
                and gd.child_token(token.doc).tag_ == "PROAV"
            ):
                relevant_dependencies.extend(
                    [
                        grandchild_dependency
                        for grandchild_dependency in grandchild_token._.holmes.children
                        if grandchild_dependency.label in ("oc", "re")
                        and grandchild_dependency.child_token(token.doc).pos_
                        in ("VERB", "AUX")
                    ]
                )
            for relevant_dependency in relevant_dependencies:
                (
                    target_tokens,
                    target_dependency,
                ) = find_target_tokens_and_dependency_recursively(
                    relevant_dependency.child_token(token.doc)
                )
                for target_token in (
                    target_token
                    for target_token in target_tokens
                    if target_token.i != dependency.parent_index
                    and len(
                        [
                            dep
                            for dep in target_token._.holmes.children
                            if dep.label == target_dependency and not dep.is_uncertain
                        ]
                    )
                    == 0
                ):
                    # these dependencies are always uncertain
                    target_token._.holmes.children.append(
                        SemanticDependency(
                            target_token.i,
                            dependency.parent_index,
                            target_dependency,
                            True,
                        )
                    )

        # sometimes two verb arguments are interpreted as both subjects or both objects,
        # if this occurs reinterpret them

        def get_first_real_subject(
            dependencies: List[SemanticDependency],
        ) -> Optional[Token]:
            potential_first_real_subject = None
            for working_dep in dependencies:
                potential_first_real_subject = working_dep.child_token(token.doc)
                if has_morphs(
                    potential_first_real_subject,
                    "Case=Nom",
                    "Gender=Masc",
                    "Number=Sing",
                ):
                    # if there is a 'der' later on in the dependencies it must be the subject
                    return potential_first_real_subject
            for working_dep in dependencies:
                potential_first_real_subject = working_dep.child_token(token.doc)
                if not has_morphs(potential_first_real_subject, "Case=Acc"):
                    # a 'den' can never be the subject
                    return potential_first_real_subject
            return None

        # find first 'sb' dependency for verb
        dependencies = [
            dependency
            for dependency in token._.holmes.children
            if token.pos_ == "VERB"
            and dependency.label == "sb"
            and not dependency.is_uncertain
            and (
                dependency.child_token(token.doc).i == 0
                or token.doc[dependency.child_token(token.doc).i - 1].dep_
                not in self.conjunction_deps
            )
        ]
        if (
            len(dependencies) > 0
            and len(
                [
                    object_dependency
                    for object_dependency in dependencies
                    if object_dependency.label == "oa" and not dependency.is_uncertain
                ]
            )
            == 0
        ):
            dependencies.sort(key=lambda dependency: dependency.child_index)
            first_real_subject = get_first_real_subject(dependencies)
            if first_real_subject is not None:
                for (
                    real_subject_index
                ) in first_real_subject._.holmes.get_sibling_indexes(token.doc):
                    for dependency in dependencies:
                        if dependency.child_index == real_subject_index:
                            dependencies.remove(dependency)
                for dependency in (
                    other_dependency for other_dependency in dependencies
                ):
                    dependency.label = "oa"

        dependencies = [
            dependency
            for dependency in token._.holmes.children
            if token.pos_ in ("VERB", "AUX")
            and dependency.label in ("oa", "pd")
            and not dependency.is_uncertain
            and (
                dependency.child_token(token.doc).i == 0
                or token.doc[dependency.child_token(token.doc).i - 1].dep_
                not in self.conjunction_deps
            )
        ]
        if (
            len(dependencies) > 0
            and len(
                [
                    object_dependency
                    for object_dependency in dependencies
                    if object_dependency.label == "sb" and not dependency.is_uncertain
                ]
            )
            == 0
        ):
            dependencies.sort(key=lambda dependency: dependency.child_index)
            first_real_subject = get_first_real_subject(dependencies)
            if first_real_subject is not None:
                real_subject_indexes = first_real_subject._.holmes.get_sibling_indexes(
                    token.doc
                )
                if len(dependencies) > len(real_subject_indexes):
                    for dependency in (
                        dependency
                        for dependency in dependencies
                        if dependency.child_index in real_subject_indexes
                    ):
                        dependency.label = "sb"

    def normalize_hyphens(self, word: str) -> str:
        """Normalizes hyphens in a multiword. Depending on the language,
        this may involve replacing them with spaces (English) or deleting them entirely
        (German).
        """
        if word.strip().startswith("-") or word.endswith("-"):
            return word
        else:
            return word.replace("-", "")


class LanguageSpecificSemanticMatchingHelper(SemanticMatchingHelper):

    noun_pos = ["NOUN", "PROPN", "ADJ"]

    preposition_deps = ["prep"]

    permissible_embedding_pos = ["NOUN", "PROPN", "ADJ", "ADV"]

    noun_kernel_dep = ["nk", "pnc"]

    minimum_embedding_match_word_length = 4

    interrogative_pronoun_tags = ["PWAT", "PWAV", "PWS"]

    topic_matching_phraselet_stop_lemmas = ["dann", "danach", "so", "ich", "mein"]

    topic_matching_reverse_only_parent_lemmas = [
        ("sein", "AUX"),
        ("werden", "AUX"),
        ("haben", "AUX"),
        ("sagen", "VERB"),
        ("machen", "VERB"),
        ("tun", "VERB"),
        ("haben", "VERB"),
        ("werden", "VERB"),
    ]

    topic_matching_phraselet_stop_tags = ["PPER", "PDS", "PRF"]

    supervised_document_classification_phraselet_stop_lemmas = ["sein", "haben"]

    preferred_phraselet_pos = ["NOUN", "PROPN"]

    entity_defined_multiword_pos = ["NOUN", "PROPN"]

    entity_defined_multiword_entity_types = ["PER", "LOC"]

    sibling_marker_deps = ["cj", "app"]

    question_answer_blacklist_deps = ["cj", "cd", "punct", "app", "punct"]

    question_answer_final_blacklist_deps: List[str] = []

    match_implication_dict = {
        "sb": MatchImplication(
            search_phrase_dependency="sb",
            document_dependencies=["pobjb", "ag", "arg", "intcompound"],
            reverse_document_dependencies=["nk"],
        ),
        "ag": MatchImplication(
            search_phrase_dependency="ag",
            document_dependencies=["nk", "pobjo", "intcompound"],
            reverse_document_dependencies=["nk"],
        ),
        "oa": MatchImplication(
            search_phrase_dependency="oa",
            document_dependencies=["pobjo", "ag", "arg", "intcompound", "og", "oc"],
            reverse_document_dependencies=["nk"],
        ),
        "arg": MatchImplication(
            search_phrase_dependency="arg",
            document_dependencies=["sb", "oa", "ag", "intcompound", "pobjb", "pobjo"],
            reverse_document_dependencies=["nk"],
        ),
        "mo": MatchImplication(
            search_phrase_dependency="mo",
            document_dependencies=["moposs", "mnr", "mnrposs", "nk", "oc", "pd"],
        ),
        "mnr": MatchImplication(
            search_phrase_dependency="mnr",
            document_dependencies=["mnrposs", "mo", "moposs", "nk", "oc"],
        ),
        "nk": MatchImplication(
            search_phrase_dependency="nk",
            document_dependencies=["ag", "pobjo", "intcompound", "oc", "mo"],
            reverse_document_dependencies=[
                "sb",
                "ag",
                "oa",
                "arg",
                "pobjo",
                "intcompound",
            ],
        ),
        "oc": MatchImplication(
            search_phrase_dependency="oc",
            document_dependencies=["pobjo", "ag", "arg", "intcompound", "og", "oa"],
            reverse_document_dependencies=["nk"],
        ),
        "pd": MatchImplication(
            search_phrase_dependency="pd", document_dependencies=["moposs", "mo"]
        ),
        "pobjo": MatchImplication(
            search_phrase_dependency="pobjo",
            document_dependencies=["ag", "intcompound"],
            reverse_document_dependencies=["nk"],
        ),
        "pobjp": MatchImplication(
            search_phrase_dependency="pobjp", document_dependencies=["intcompound"]
        ),
        "wh_sb": MatchImplication(
            search_phrase_dependency="wh_sb",
            document_dependencies=["pobjb", "ag", "arg", "intcompound", "sb", "pd"],
            reverse_document_dependencies=["nk"],
        ),
        "wh_wildcard": MatchImplication(
            search_phrase_dependency="wh_wildcard",
            document_dependencies=["mo", "oc", "mnr", "mnrposs", "prep"],
        ),
        # intcompound is only used within extensive matching because it is not assigned
        # in the context of registering search phrases.
        "intcompound": MatchImplication(
            search_phrase_dependency="intcompound",
            document_dependencies=[
                "sb",
                "oa",
                "ag",
                "og",
                "nk",
                "mo",
                "pobjo",
                "pobjp",
            ],
            reverse_document_dependencies=["nk"],
        ),
    }

    phraselet_templates = [
        PhraseletTemplate(
            "verb-nom",
            "Eine Sache tut",
            2,
            1,
            ["sb", "pobjb"],
            [
                "VMFIN",
                "VMINF",
                "VMPP",
                "VVFIN",
                "VVIMP",
                "VVINF",
                "VVIZU",
                "VVPP",
                "VAFIN",
                "VAIMP",
                "VAINF",
                "VAPP",
                "FM",
                "NE",
                "NNE",
                "NN",
            ],
            ["FM", "NE", "NNE", "NN"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "verb-acc",
            "Jemand tut eine Sache",
            1,
            3,
            ["oa", "pobjo", "ag", "og", "oc"],
            [
                "VMFIN",
                "VMINF",
                "VMPP",
                "VVFIN",
                "VVIMP",
                "VVINF",
                "VVIZU",
                "VVPP",
                "VAFIN",
                "VAIMP",
                "VAINF",
                "VAPP",
                "FM",
                "NE",
                "NNE",
                "NN",
            ],
            ["FM", "NE", "NNE", "NN"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "verb-dat",
            "Jemand gibt einer Sache etwas",
            1,
            3,
            ["da"],
            [
                "VMFIN",
                "VMINF",
                "VMPP",
                "VVFIN",
                "VVIMP",
                "VVINF",
                "VVIZU",
                "VVPP",
                "VAFIN",
                "VAIMP",
                "VAINF",
                "VAPP",
            ],
            ["FM", "NE", "NNE", "NN"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "verb-pd",
            "Jemand ist eine Sache",
            1,
            3,
            ["pd"],
            [
                "VMFIN",
                "VMINF",
                "VMPP",
                "VVFIN",
                "VVIMP",
                "VVINF",
                "VVIZU",
                "VVPP",
                "VAFIN",
                "VAIMP",
                "VAINF",
                "VAPP",
            ],
            ["FM", "NE", "NNE", "NN"],
            reverse_only=True,
            question=False,
        ),
        PhraseletTemplate(
            "noun-dependent",
            "Eine große Sache",
            2,
            1,
            ["nk"],
            ["FM", "NE", "NNE", "NN"],
            ["FM", "NE", "NNE", "NN", "ADJA", "ADJD", "ADV", "CARD"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "verb-adverb",
            "schnell machen",
            1,
            0,
            ["mo", "moposs", "oc"],
            [
                "VMFIN",
                "VMINF",
                "VMPP",
                "VVFIN",
                "VVIMP",
                "VVINF",
                "VVIZU",
                "VVPP",
                "VAFIN",
                "VAIMP",
                "VAINF",
                "VAPP",
            ],
            ["ADJA", "ADJD", "ADV"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "prepgovernor-noun",
            "Eine Sache in einer Sache",
            1,
            4,
            ["pobjp"],
            [
                "VMFIN",
                "VMINF",
                "VMPP",
                "VVFIN",
                "VVIMP",
                "VVINF",
                "VVIZU",
                "VVPP",
                "VAFIN",
                "VAIMP",
                "VAINF",
                "VAPP",
                "FM",
                "NE",
                "NNE",
                "NN",
            ],
            ["FM", "NE", "NNE", "NN"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "prep-noun",
            "in einer Sache",
            0,
            2,
            ["nk"],
            ["APPO", "APPR", "APPRART", "APZR"],
            ["FM", "NE", "NNE", "NN"],
            reverse_only=True,
            question=False,
        ),
        PhraseletTemplate(
            "verb-toughmovedargument",
            "Eine Sache ist schwer zu tun",
            5,
            1,
            ["arg"],
            [
                "VMFIN",
                "VMINF",
                "VMPP",
                "VVFIN",
                "VVIMP",
                "VVINF",
                "VVIZU",
                "VVPP",
                "VAFIN",
                "VAIMP",
                "VAINF",
                "VAPP",
            ],
            ["FM", "NE", "NNE", "NN"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "intcompound",
            "Eine Sache in einer Sache",
            1,
            4,
            ["intcompound"],
            ["NE", "NNE", "NN", "TRUNC", "ADJA", "ADJD", "TRUNC"],
            ["NE", "NNE", "NN", "TRUNC", "ADJA", "ADJD", "TRUNC"],
            reverse_only=False,
            question=False,
            assigned_dependency_label="intcompound",
        ),
        PhraseletTemplate(
            "head-WHnom",
            "wer kam?",
            1,
            0,
            ["sb"],
            [
                "VMFIN",
                "VMINF",
                "VMPP",
                "VVFIN",
                "VVIMP",
                "VVINF",
                "VVIZU",
                "VVPP",
                "VAFIN",
                "VAINF",
            ],
            ["PWS"],
            reverse_only=False,
            question=True,
            assigned_dependency_label="wh_sb",
        ),
        PhraseletTemplate(
            "head-WHacc",
            "wen siehst du?",
            1,
            0,
            ["oa"],
            ["VMFIN", "VMINF", "VMPP", "VVFIN", "VVIMP", "VVINF", "VVIZU", "VVPP"],
            ["PWS"],
            reverse_only=False,
            question=True,
        ),
        PhraseletTemplate(
            "head-WHdat",
            "wem gibst du es?",
            1,
            0,
            ["da"],
            ["VMFIN", "VMINF", "VMPP", "VVFIN", "VVIMP", "VVINF", "VVIZU", "VVPP"],
            ["PWS"],
            reverse_only=False,
            question=True,
        ),
        PhraseletTemplate(
            "head-WHadv",
            "womit machst du es?",
            1,
            0,
            ["mo"],
            ["VMFIN", "VMINF", "VMPP", "VVFIN", "VVIMP", "VVINF", "VVIZU", "VVPP"],
            ["PWAV"],
            reverse_only=False,
            question=True,
            assigned_dependency_label="wh_wildcard",
        ),
        PhraseletTemplate(
            "word",
            "Sache",
            0,
            None,
            None,
            ["FM", "NE", "NNE", "NN"],
            None,
            reverse_only=False,
            question=False,
        ),
    ]

    def question_word_matches(
        self,
        search_phrase_token: Token,
        document_token: Token,
        document_subword_index: Optional[int],
        document_vector,
        entity_label_to_vector_dict: dict,
        initial_question_word_embedding_match_threshold: float,
    ) -> bool:
        if document_subword_index is not None:
            return search_phrase_token._.holmes.lemma in ("was")
        if search_phrase_token._.holmes.lemma in ("wer", "wen", "wem"):
            ent_types = ("PER", "ORG")
            return (
                document_token.ent_type_ in ent_types
                or self.token_matches_ent_type(
                    document_vector,
                    entity_label_to_vector_dict,
                    ent_types,
                    initial_question_word_embedding_match_threshold,
                )
                > 0
            )
        if search_phrase_token._.holmes.lemma in ("was"):
            return True
        if search_phrase_token._.holmes.lemma == "wo":
            # spaCy model does not handle postpositions
            return (
                document_token.tag_ in ("APPR", "APPRART")
                and document_token._.holmes.lemma
                in (
                    "an",
                    "auf",
                    "aus",
                    "bei",
                    "gegenüber",
                    "hinter",
                    "in",
                    "neben",
                    "über",
                    "unter",
                    "vor",
                    "zu",
                    "zwischen",
                )
                and len(
                    [
                        1
                        for c in document_token._.holmes.children
                        if "Case=Dat" in c.child_token(document_token.doc).morph
                    ]
                )
                > 0
            )
        if search_phrase_token._.holmes.lemma == "wohin":
            return (
                document_token.tag_ in ("APPR", "APPRART")
                and document_token._.holmes.lemma
                in (
                    "an",
                    "auf",
                    "hinter",
                    "in",
                    "neben",
                    "über",
                    "unter",
                    "vor",
                    "zwischen",
                )
                and len(
                    [
                        1
                        for c in document_token._.holmes.children
                        if "Case=Acc" in c.child_token(document_token.doc).morph
                    ]
                )
                > 0
            )
        if search_phrase_token._.holmes.lemma == "wann":
            if document_token.tag_ in ("APPR", "APPRART"):
                return document_token._.holmes.lemma in (
                    "ab",
                    "an",
                    "bis",
                    "für",
                    "in",
                    "nach",
                    "seit",
                    "vor",
                    "um",
                )
            return document_token.dep_ == "mo" and document_token.pos_ in (
                "NOUN",
                "PROPN",
                "ADV",
                "VERB",
                "AUX",
            )
        if search_phrase_token._.holmes.lemma == "wie":
            if document_token.tag_ in ("APPR", "APPRART"):
                return document_token._.holmes.lemma in ("mit", "mittels")
            if document_token.dep_ == ("mo") and document_token.tag_ in ("ADJD"):
                return True
            return (
                document_token.dep_ in ("mo", "oc")
                and len(
                    [
                        1
                        for c in document_token._.holmes.children
                        if c.child_token(document_token.doc)._.holmes.lemma == "indem"
                    ]
                )
                > 0
            )
        if search_phrase_token._.holmes.lemma == "woher":
            if document_token.tag_ in ("APPR", "APPRART"):
                return document_token._.holmes.lemma in ("aus", "von", "wegen")
            return (
                document_token.dep_ in ("mo", "oc")
                and len(
                    [
                        1
                        for c in document_token._.holmes.children
                        if c.child_token(document_token.doc)._.holmes.lemma == "weil"
                    ]
                )
                > 0
            )
        if search_phrase_token._.holmes.lemma in ("warum", "wieso", "weshalb"):
            if document_token.tag_ in ("APPR", "APPRART"):
                return document_token._.holmes.lemma in ("wegen")
            return (
                document_token.dep_ in ("mo", "oc", "cd")
                and len(
                    [
                        1
                        for c in document_token.children
                        if c._.holmes.lemma in ("weil", "damit")
                    ]
                )
                > 0
            )
            # syntactic not semantic children to handle subject-predicate phrases correctly
        if (
            search_phrase_token._.holmes.lemma.startswith("wo")
            and document_token.tag_ == "APPR"
            and search_phrase_token._.holmes.lemma.endswith(
                document_token._.holmes.lemma
            )
        ):
            return True
        # 'wessen' is not correctly recognized by the current _lg model
        return False
