from typing import Optional, Dict, List
from spacy.tokens import Token
from ...parsing import (
    SemanticAnalyzer,
    SemanticMatchingHelper,
    MatchImplication,
    PhraseletTemplate,
    SemanticDependency,
    Subword,
)


class LanguageSpecificSemanticAnalyzer(SemanticAnalyzer):

    language_name = "English"

    # The part of speech tags that can refer to nouns
    noun_pos = ["NOUN", "PROPN"]

    # The part of speech tags that can refer to predicate heads
    predicate_head_pos = ["VERB", "AUX"]

    # The part of speech tags that require a match in the search sentence when they occur within a
    # search_phrase
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

    # The part of speech tags that can refer to the head of an adjectival predicate phrase
    # ("is" in "The dog is tired")
    adjectival_predicate_head_pos = ["VERB", "AUX"]

    # The part of speech tags that can refer to the subject of a adjectival predicate
    # ("dog" in "The dog is tired")
    adjectival_predicate_subject_pos = ["NOUN", "PROPN", "PRON"]

    # Dependency label that marks the subject of an adjectival predicate
    adjectival_predicate_subject_dep = "nsubj"

    # Dependency label that marks the predicate of an adjectival predicate
    adjectival_predicate_predicate_dep = "acomp"

    # Part of speech that marks the predicate of an adjectival predicate
    adjectival_predicate_predicate_pos = "ADJ"

    # Dependency label that marks a modifying adjective
    modifier_dep = "amod"

    # Original dependency label from nouns to prepositions
    spacy_noun_to_preposition_dep = "prep"

    # Original dependency label from verbs to prepositions
    spacy_verb_to_preposition_dep = "prep"

    # Added possible dependency label from nouns to prepositions
    holmes_noun_to_preposition_dep = "prepposs"

    # Added possible dependency label from verbs to prepositions
    holmes_verb_to_preposition_dep = "prepposs"

    # Dependency labels that occur in a conjunction phrase (righthand siblings and conjunctions)
    conjunction_deps = ["conj", "appos", "cc"]

    # Syntactic tags that can mark interrogative pronouns
    interrogative_pronoun_tags = ["WDT", "WP", "WRB"]

    # Syntactic tags that exclude a token from being the child token within a semantic dependency
    semantic_dependency_excluded_tags = ["DT"]

    # Generic pronouns
    generic_pronoun_lemmas = ["something", "somebody", "someone"]

    # The word for 'or' in this language
    or_lemma = "or"

    # Where dependencies from a parent to a child are copied to the parent's righthand siblings,
    # it can make sense to mark the dependency as uncertain depending on the underlying spaCy
    # representations for the individual language
    mark_child_dependencies_copied_to_siblings_as_uncertain = True

    # Coreference chains are only processed up to this number of mentions away from the currently
    # matched document location
    maximum_mentions_in_coreference_chain = 3

    # Coreference chains are only processed up to this number of words away from the currently
    # matched document location
    maximum_word_distance_in_coreference_chain = 300

    # Dependency labels that can mark righthand siblings
    sibling_marker_deps = ["conj", "appos"]

    # Map from entity labels to words that correspond to their meaning
    entity_labels_to_corresponding_lexemes = {
        "PERSON": "person",
        "NORP": "group",
        "FAC": "building",
        "ORG": "organization",
        "GPE": "place",
        "LOC": "place",
        "PRODUCT": "product",
        "EVENT": "event",
        "WORK_OF_ART": "artwork",
        "LAW": "law",
        "LANGUAGE": "language",
        "DATE": "date",
        "TIME": "time",
        "PERCENT": "percent",
        "MONEY": "money",
        "QUANTITY": "quantity",
        "ORDINAL": "number",
        "CARDINAL": "number",
    }

    whose_lemma = "whose"

    def add_subwords(
        self, token: Token, subword_cache: Dict[str, List[Subword]]
    ) -> None:
        pass
        """Analyses the internal structure of the word to find atomic semantic elements. Is
        relevant for German but not implemented for English.
        """
        pass

    def set_negation(self, token: Token) -> None:
        """Marks the negation on the token. A token is negative if it or one of its ancestors
        has a negation word as a syntactic (not semantic!) child.
        """
        if token._.holmes.is_negated is not None:
            return
        for child in token.children:
            if (
                child._.holmes.lemma
                in (
                    "nobody",
                    "nothing",
                    "nowhere",
                    "noone",
                    "neither",
                    "nor",
                    "no",
                    "not",
                )
                or child.dep_ == "neg"
            ):
                token._.holmes.is_negated = True
                return
            if child._.holmes.lemma in ("more", "longer"):
                for grandchild in child.children:
                    if grandchild._.holmes.lemma == "no":
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
        # 'auxpass' means an auxiliary used in a passive context. We mark its subject with
        # a new dependency label 'nsubjpass'.
        if (
            len(
                [
                    dependency
                    for dependency in token._.holmes.children
                    if dependency.label == "auxpass"
                ]
            )
            > 0
        ):
            for dependency in token._.holmes.children:
                if dependency.label == "nsubj":
                    dependency.label = "nsubjpass"

        # Structures like 'he used to' and 'he is going to'
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.label == "xcomp"
        ):
            child = dependency.child_token(token.doc)
            # distinguish 'he used to ...' from 'he used it to ...'
            if (
                token._.holmes.lemma == "use"
                and token.tag_ == "VBD"
                and len(
                    [
                        element
                        for element in token._.holmes.children
                        if element.label == "dobj"
                    ]
                )
                == 0
            ):
                self.move_information_between_tokens(token, child)
            elif token._.holmes.lemma == "go":
                # 'was going to' is marked as uncertain, 'is going to' is not marked as uncertain
                uncertainty_flag = False
                for other_dependency in (
                    other_dependency
                    for other_dependency in token._.holmes.children
                    if other_dependency.label == "aux"
                ):
                    other_dependency_token = other_dependency.child_token(token.doc)
                    if (
                        other_dependency_token._.holmes.lemma == "be"
                        and other_dependency_token.tag_ == "VBD"
                    ):  # 'was going to'
                        uncertainty_flag = True
                self.move_information_between_tokens(token, child)
                if uncertainty_flag:
                    for child_dependency in child._.holmes.children:
                        child_dependency.is_uncertain = True
            else:
                # constructions like:
                #
                #'she told him to close the contract'
                #'he decided to close the contract'
                for other_dependency in token._.holmes.children:
                    if other_dependency.label in ("dobj", "nsubjpass") or (
                        other_dependency.label == "nsubj"
                        and len(
                            [
                                element
                                for element in token._.holmes.children
                                if element.label == "dobj"
                            ]
                        )
                        == 0
                    ):
                        if (
                            len(
                                [
                                    element
                                    for element in child._.holmes.children
                                    if element.label == "auxpass"
                                ]
                            )
                            > 0
                        ):
                            if (
                                not child._.holmes.has_dependency_with_child_index(
                                    other_dependency.child_index
                                )
                                and dependency.child_index
                                > other_dependency.child_index
                            ):
                                child._.holmes.children.append(
                                    SemanticDependency(
                                        dependency.child_index,
                                        other_dependency.child_index,
                                        "nsubjpass",
                                        True,
                                    )
                                )
                        else:
                            if (
                                not child._.holmes.has_dependency_with_child_index(
                                    other_dependency.child_index
                                )
                                and dependency.child_index
                                > other_dependency.child_index
                            ):
                                child._.holmes.children.append(
                                    SemanticDependency(
                                        dependency.child_index,
                                        other_dependency.child_index,
                                        "nsubj",
                                        True,
                                    )
                                )

    def handle_relative_constructions(self, token: Token) -> None:
        if token.dep_ == "relcl":
            for dependency in token._.holmes.children:
                child = dependency.child_token(token.doc)
                # handle 'whose' clauses
                for child_dependency in (
                    child_dependency
                    for child_dependency in child._.holmes.children
                    if child_dependency.child_index >= 0
                    and child_dependency.label == "poss"
                    and child_dependency.child_token(token.doc).tag_ == "WP$"
                ):
                    whose_pronoun_token = child_dependency.child_token(token.doc)
                    working_index = whose_pronoun_token.i
                    while working_index >= token.sent.start:
                        # find the antecedent (possessed entity)
                        if (
                            len(
                                [
                                    1
                                    for working_dependency in whose_pronoun_token.doc[
                                        working_index
                                    ]._.holmes.children
                                    if working_dependency.label == "relcl"
                                ]
                            )
                            > 0
                        ):
                            working_token = child.doc[working_index]
                            working_token = working_token.doc[
                                working_token._.holmes.token_or_lefthand_sibling_index
                            ]
                            for (
                                lefthand_sibling_of_antecedent
                            ) in working_token._.holmes.loop_token_and_righthand_siblings(
                                token.doc
                            ):
                                # find the possessing noun
                                for possessing_noun in (
                                    possessing_noun
                                    for possessing_noun in child._.holmes.loop_token_and_righthand_siblings(
                                        token.doc
                                    )
                                    if possessing_noun.i
                                    != lefthand_sibling_of_antecedent.i
                                ):
                                    # add the semantic dependency
                                    possessing_noun._.holmes.children.append(
                                        SemanticDependency(
                                            possessing_noun.i,
                                            lefthand_sibling_of_antecedent.i,
                                            "poss",
                                            lefthand_sibling_of_antecedent.i
                                            != working_index,
                                        )
                                    )
                                    # remove the syntactic dependency
                                    possessing_noun._.holmes.remove_dependency_with_child_index(
                                        whose_pronoun_token.i
                                    )
                                whose_pronoun_token._.holmes.children = [
                                    SemanticDependency(
                                        whose_pronoun_token.i,
                                        0 - (working_index + 1),
                                        None,
                                    )
                                ]
                            return
                        working_index -= 1
                    return
                if child.tag_ in ("WP", "WRB", "WDT"):  # 'that' or 'which'
                    working_dependency_label = dependency.label
                    child._.holmes.children = [
                        SemanticDependency(child.i, 0 - (token.head.i + 1), None)
                    ]
                else:
                    # relative antecedent, new dependency tag, 'the man I saw yesterday'
                    working_dependency_label = "relant"
                last_righthand_sibling_of_predicate = list(
                    token._.holmes.loop_token_and_righthand_siblings(token.doc)
                )[-1]
                for preposition_dependency in (
                    dep
                    for dep in last_righthand_sibling_of_predicate._.holmes.children
                    if dep.label == "prep"
                    and dep.child_token(token.doc)._.holmes.is_matchable
                ):
                    preposition = preposition_dependency.child_token(token.doc)
                    for grandchild_dependency in (
                        dep
                        for dep in preposition._.holmes.children
                        if dep.child_token(token.doc).tag_ in ("WP", "WRB", "WDT")
                        and dep.child_token(token.doc).i >= 0
                    ):
                        # 'that' or 'which'
                        complementizer = grandchild_dependency.child_token(token.doc)
                        preposition._.holmes.remove_dependency_with_child_index(
                            grandchild_dependency.child_index
                        )
                        # a new relation pointing directly to the antecedent noun
                        # will be added in the section below
                        complementizer._.holmes.children = [
                            SemanticDependency(
                                grandchild_dependency.child_index,
                                0 - (token.head.i + 1),
                                None,
                            )
                        ]
                displaced_preposition_dependencies = [
                    dep
                    for dep in last_righthand_sibling_of_predicate._.holmes.children
                    if dep.label == "prep"
                    and len(dep.child_token(token.doc)._.holmes.children) == 0
                    and dep.child_token(token.doc)._.holmes.is_matchable
                ]
                antecedent = token.doc[
                    token.head._.holmes.token_or_lefthand_sibling_index
                ]
                if len(displaced_preposition_dependencies) > 0:
                    displaced_preposition = displaced_preposition_dependencies[
                        0
                    ].child_token(token.doc)
                    for lefthand_sibling_of_antecedent in (
                        lefthand_sibling_of_antecedent
                        for lefthand_sibling_of_antecedent in antecedent._.holmes.loop_token_and_righthand_siblings(
                            token.doc
                        )
                        if displaced_preposition.i != lefthand_sibling_of_antecedent.i
                    ):
                        displaced_preposition._.holmes.children.append(
                            SemanticDependency(
                                displaced_preposition.i,
                                lefthand_sibling_of_antecedent.i,
                                "pobj",
                                lefthand_sibling_of_antecedent.i != token.head.i,
                            )
                        )
                        # Where the antecedent is not the final one before the relative
                        # clause, mark the dependency as uncertain
                    for (
                        sibling_of_pred
                    ) in token._.holmes.loop_token_and_righthand_siblings(token.doc):
                        if (
                            not sibling_of_pred._.holmes.has_dependency_with_child_index(
                                displaced_preposition.i
                            )
                            and sibling_of_pred.i != displaced_preposition.i
                        ):
                            sibling_of_pred._.holmes.children.append(
                                SemanticDependency(
                                    sibling_of_pred.i,
                                    displaced_preposition.i,
                                    "prep",
                                    True,
                                )
                            )
                        if working_dependency_label != "relant":
                            # if 'that' or 'which', remove it
                            sibling_of_pred._.holmes.remove_dependency_with_child_index(
                                child.i
                            )
                else:
                    for (
                        lefthand_sibling_of_antecedent
                    ) in antecedent._.holmes.loop_token_and_righthand_siblings(
                        token.doc
                    ):
                        for sibling_of_predicate in (
                            sibling_of_predicate
                            for sibling_of_predicate in token._.holmes.loop_token_and_righthand_siblings(
                                token.doc
                            )
                            if sibling_of_predicate.i
                            != lefthand_sibling_of_antecedent.i
                        ):
                            sibling_of_predicate._.holmes.children.append(
                                SemanticDependency(
                                    sibling_of_predicate.i,
                                    lefthand_sibling_of_antecedent.i,
                                    working_dependency_label,
                                    lefthand_sibling_of_antecedent.i != token.head.i,
                                )
                            )
                            # Where the antecedent is not the final one before the relative
                            # clause, mark the dependency as uncertain
                            if working_dependency_label != "relant":
                                sibling_of_predicate._.holmes.remove_dependency_with_child_index(
                                    child.i
                                )
                break

    def holmes_lemma(self, token: Token) -> str:
        """Relabel the lemmas of phrasal verbs in sentences like 'he gets up' to incorporate
        the entire phrasal verb to facilitate matching.
        """
        if token.pos_ == "VERB":
            for child in token.children:
                if child.tag_ == "RP":
                    return " ".join([token.lemma_.lower(), child.lemma_.lower()])
        if token.pos_ == "ADJ":
            # see if the adjective is a participle
            participle_test_doc = self.spacy_parse_for_lemmas(
                " ".join(("Somebody has", token.lemma_.lower()))
            )
            return participle_test_doc[2].lemma_.lower()
        return token.lemma_.lower()

    def language_specific_derived_holmes_lemma(self, token: Token, lemma: str) -> str:
        """Generates and returns a derived lemma where appropriate, otherwise returns *lemma*."""
        if (token is None or token.pos_ == "NOUN") and len(lemma) >= 10:
            possible_lemma = None
            if lemma.endswith("isation") or lemma.endswith("ization"):
                possible_lemma = "".join(
                    (lemma[:-5], "e")
                )  # 'isation', 'ization' -> 'ise', 'ize'
                if possible_lemma.endswith("ise"):
                    lemma_to_test_in_vocab = "".join((possible_lemma[:-3], "ize"))
                    # only American spellings in vocab
                else:
                    lemma_to_test_in_vocab = possible_lemma
            elif lemma.endswith("ication"):
                possible_lemma = "".join((lemma[:-7], "y"))  # implication -> imply
                lemma_to_test_in_vocab = possible_lemma
            if (
                possible_lemma is None
                or self.vectors_nlp.vocab[lemma_to_test_in_vocab].is_oov
            ) and lemma.endswith("ation"):
                possible_lemma = "".join(
                    (lemma[:-3], "e")
                )  # manipulation -> manipulate
                lemma_to_test_in_vocab = possible_lemma
            if (
                possible_lemma is not None
                and not self.vectors_nlp.vocab[lemma_to_test_in_vocab].is_oov
            ):
                return possible_lemma
        # deadjectival nouns in -ness
        if (
            (token is None or token.pos_ == "NOUN")
            and len(lemma) >= 7
            and lemma.endswith("ness")
        ):
            working_possible_lemma = lemma[:-4]
            # 'bawdiness'
            if working_possible_lemma[-1] == "i":
                working_possible_lemma = "".join((working_possible_lemma[:-1], "y"))
            if not self.vectors_nlp.vocab[working_possible_lemma].is_oov:
                return working_possible_lemma
            else:
                return lemma
        # adverb with 'ly' -> adjective without 'ly'
        if token is None or token.tag_ == "RB":
            # domestically -> domestic
            if lemma.endswith("ically"):
                return lemma[:-4]
            # 'regrettably', 'horribly' -> 'regrettable', 'horrible'
            if lemma.endswith("ably") or lemma.endswith("ibly"):
                return "".join((lemma[:-1], "e"))
            if lemma.endswith("ly"):
                derived_lemma = lemma[:-2]
                # 'happily' -> 'happy'
                if derived_lemma[-1] == "i":
                    derived_lemma = "".join((derived_lemma[:-1], "y"))
                return derived_lemma
        # singing -> sing
        if (token is None or token.tag_ == "NN") and lemma.endswith("ing"):
            lemmatization_sentence = " ".join(("it is", lemma))
            lemmatization_doc = self.spacy_parse_for_lemmas(lemmatization_sentence)
            return lemmatization_doc[2].lemma_.lower()
        return lemma

    def perform_language_specific_tasks(self, token: Token) -> None:

        # Because phrasal verbs are conflated into a single lemma, remove the dependency
        # from the verb to the preposition and mark the preposition is unmatchable
        if token.tag_ == "RP":
            token.head._.holmes.remove_dependency_with_child_index(token.i)
            token._.holmes.is_matchable = False

        # mark modal verb dependencies as uncertain
        if token.pos_ == "VERB":
            for dependency in (
                dependency
                for dependency in token._.holmes.children
                if dependency.label == "aux"
            ):
                child = dependency.child_token(token.doc)
                if child.pos_ in ("VERB", "AUX") and child._.holmes.lemma not in (
                    "be",
                    "have",
                    "do",
                    "go",
                    "use",
                    "will",
                    "shall",
                ):
                    for other_dependency in (
                        other_dependency
                        for other_dependency in token._.holmes.children
                        if other_dependency.label != "aux"
                    ):
                        other_dependency.is_uncertain = True

        # set auxiliaries as not matchable
        if token.dep_ in ("aux", "auxpass"):
            token._.holmes.is_matchable = False

        # Add new dependencies to phrases with 'by', 'of' and 'to' to enable the matching
        # of deverbal nominal phrases with verb phrases; add 'dative' dependency to
        # nouns within dative 'to' phrases; add new dependency spanning other prepositions
        # to facilitate topic matching and supervised document classification
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.label in ("prep", "agent", "dative")
        ):
            child = dependency.child_token(token.doc)
            if child._.holmes.lemma == "by":
                working_dependency_label = "pobjb"
            elif child._.holmes.lemma == "of":
                working_dependency_label = "pobjo"
            elif child._.holmes.lemma == "to":
                if dependency.label == "dative":
                    working_dependency_label = "dative"
                else:
                    working_dependency_label = "pobjt"
            else:
                working_dependency_label = "pobjp"
            # for 'by', 'of' and 'to' the preposition is marked as not matchable
            if working_dependency_label != "pobjp":
                child._.holmes.is_matchable = False
            for child_dependency in (
                child_dependency
                for child_dependency in child._.holmes.children
                if child_dependency.label == "pobj"
                and token.i != child_dependency.child_index
            ):
                token._.holmes.children.append(
                    SemanticDependency(
                        token.i,
                        child_dependency.child_index,
                        working_dependency_label,
                        dependency.is_uncertain or child_dependency.is_uncertain,
                    )
                )

        # where a 'prepposs' dependency has been added and the preposition is not 'by', 'of' or
        #'to', add a corresponding uncertain 'pobjp'
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.label == "prepposs"
        ):
            child = dependency.child_token(token.doc)
            for child_dependency in (
                child_dependency
                for child_dependency in child._.holmes.children
                if child_dependency.label == "pobj"
                and token.i != child_dependency.child_index
                and child._.holmes.is_matchable
            ):
                token._.holmes.children.append(
                    SemanticDependency(
                        token.i, child_dependency.child_index, "pobjp", True
                    )
                )

        # handle present active participles
        if token.dep_ == "acl" and token.tag_ == "VBG":
            lefthand_sibling = token.doc[
                token.head._.holmes.token_or_lefthand_sibling_index
            ]
            for (
                antecedent
            ) in lefthand_sibling._.holmes.loop_token_and_righthand_siblings(token.doc):
                if token.i != antecedent.i:
                    token._.holmes.children.append(
                        SemanticDependency(token.i, antecedent.i, "nsubj")
                    )

        # handle past passive participles
        if token.dep_ == "acl" and token.tag_ == "VBN":
            lefthand_sibling = token.doc[
                token.head._.holmes.token_or_lefthand_sibling_index
            ]
            for (
                antecedent
            ) in lefthand_sibling._.holmes.loop_token_and_righthand_siblings(token.doc):
                if token.i != antecedent.i:
                    token._.holmes.children.append(
                        SemanticDependency(token.i, antecedent.i, "dobj")
                    )

        # handle phrases like 'cat-eating dog' and 'dog-eaten cat', adding new dependencies
        if token.dep_ == "amod" and token.pos_ == "VERB":
            for dependency in (
                dependency
                for dependency in token._.holmes.children
                if dependency.label == "npadvmod"
            ):
                if token.tag_ == "VBG":
                    dependency.label = "advmodobj"
                    noun_dependency = "advmodsubj"
                elif token.tag_ == "VBN":
                    dependency.label = "advmodsubj"
                    noun_dependency = "advmodobj"
                else:
                    break
                for noun in token.head._.holmes.loop_token_and_righthand_siblings(
                    token.doc
                ):
                    if token.i != noun.i:
                        token._.holmes.children.append(
                            SemanticDependency(
                                token.i, noun.i, noun_dependency, noun.i != token.head.i
                            )
                        )
                break  # we only handle one antecedent, spaCy never seems to produce more anyway

        # handle phrases like 'he is thinking about singing', 'he keeps on singing'
        # find governed verb
        if token.pos_ == "VERB" and token.dep_ == "pcomp":
            # choose correct noun dependency for passive or active structure
            if (
                len(
                    [
                        dependency
                        for dependency in token._.holmes.children
                        if dependency.label == "auxpass"
                    ]
                )
                > 0
            ):
                new_dependency_label = "nsubjpass"
            else:
                new_dependency_label = "nsubj"
            # check that governed verb does not already have a dependency with the same label
            if (
                len(
                    [
                        target_token_dependency
                        for target_token_dependency in token._.holmes.children
                        if target_token_dependency.label == new_dependency_label
                    ]
                )
                == 0
            ):
                # Go back in the sentence to find the first subject phrase
                counter = token.i
                while True:
                    counter -= 1
                    if counter < token.sent.start:
                        return
                    if token.doc[counter].dep_ in ("nsubj", "nsubjpass"):
                        break
                # From the subject phrase loop up through the syntactic parents
                # to handle relative constructions
                working_token = token.doc[counter]
                while True:
                    if (
                        working_token.tag_.startswith("NN")
                        or working_token._.holmes.is_involved_in_coreference()
                    ):
                        for (
                            source_token
                        ) in working_token._.holmes.loop_token_and_righthand_siblings(
                            token.doc
                        ):
                            for (
                                target_token
                            ) in token._.holmes.loop_token_and_righthand_siblings(
                                token.doc
                            ):
                                if target_token.i != source_token.i:
                                    # such dependencies are always uncertain
                                    target_token._.holmes.children.append(
                                        SemanticDependency(
                                            target_token.i,
                                            source_token.i,
                                            new_dependency_label,
                                            True,
                                        )
                                    )
                        return
                    if working_token.dep_ != "ROOT":
                        working_token = working_token.head
                    else:
                        return

        # handle phrases like 'he is easy to find', 'he is ready to go'
        # There is no way of knowing from the syntax whether the noun is a semantic
        # subject or object of the verb, so the new dependency label 'arg' is added.
        if token.tag_.startswith("NN") or token._.holmes.is_involved_in_coreference():
            for adjective_dep in (
                dep
                for dep in token._.holmes.children
                if dep.label == self.modifier_dep
                and dep.child_token(token.doc).pos_
                == self.adjectival_predicate_predicate_pos
            ):
                adj_token = adjective_dep.child_token(token.doc)
                for verb_dep in (
                    dep
                    for dep in adj_token._.holmes.children
                    if dep.label == "xcomp"
                    and dep.child_token(token.doc).pos_ == "VERB"
                ):
                    verb_token = verb_dep.child_token(token.doc)
                    verb_token._.holmes.children.append(
                        SemanticDependency(verb_token.i, token.i, "arg", True)
                    )

    def normalize_hyphens(self, word: str) -> str:
        """Normalizes hyphens in a multiword. Depending on the language,
        this may involve replacing them with spaces (English) or deleting them entirely
        (German).
        """
        if word.strip().startswith("-") or word.endswith("-"):
            return word
        else:
            return word.replace("-", " ")


class LanguageSpecificSemanticMatchingHelper(SemanticMatchingHelper):

    # The part of speech tags that can refer to nouns
    noun_pos = ["NOUN", "PROPN"]

    # Dependency labels between a head and a preposition
    preposition_deps = ["prep"]

    # Parts of speech for which embedding matching is attempted
    permissible_embedding_pos = ["NOUN", "PROPN", "ADJ", "ADV"]

    # Dependency labels that mark noun kernel elements that are not the head noun
    noun_kernel_dep = ["nmod", "compound", "appos", "nummod"]

    # Minimum length of a word taking part in an embedding-based match.
    # Necessary because of the proliferation of short nonsense strings in the vocabularies.
    minimum_embedding_match_word_length = 3

    # Lemmas that should be suppressed within relation phraselets or as words of
    # single-word phraselets during topic matching.
    topic_matching_phraselet_stop_lemmas = ["then", "therefore", "so"]

    # Parent lemma / part-of-speech combinations that should lead to phraselets being
    # reverse-matched only during topic matching.
    topic_matching_reverse_only_parent_lemmas = [
        ("be", "VERB"),
        ("be", "AUX"),
        ("have", "VERB"),
        ("have", "AUX"),
        ("do", "VERB"),
        ("say", "VERB"),
        ("go", "VERB"),
        ("get", "VERB"),
        ("make", "VERB"),
    ]

    # Tags of tokens that should be ignored during topic matching (normally pronouns).
    topic_matching_phraselet_stop_tags = ["PRP", "PRP$"]

    # Lemmas that should be suppressed within relation phraselets or as words of
    # single-word phraselets during supervised document classification.
    supervised_document_classification_phraselet_stop_lemmas = ["be", "have"]

    # Parts of speech that are preferred as lemmas within phraselets
    preferred_phraselet_pos = ["NOUN", "PROPN"]

    # The part-o'f-speech labels permitted for elements of an entity-defined multiword.
    entity_defined_multiword_pos = ["NOUN", "PROPN"]

    # The entity labels permitted for elements of an entity-defined multiword.
    entity_defined_multiword_entity_types = ["PERSON", "ORG", "GPE", "WORK_OF_ART"]

    # Dependency labels that can mark righthand siblings
    sibling_marker_deps = ["conj", "appos"]

    # Syntactic tags that can mark interrogative pronouns
    interrogative_pronoun_tags = ["WDT", "WP", "WRB"]

    # Dependency labels from a token's subtree that are not included in a question answer
    question_answer_blacklist_deps = ["conj", "appos", "cc", "punct"]

    # Dependency labels from a token's subtree that are not included in a question answer if in
    # final position.
    question_answer_final_blacklist_deps = ["case"]

    # Maps from dependency tags as occurring within search phrases to corresponding implication
    # definitions. This is the main source of the asymmetry in matching from search phrases to
    # documents versus from documents to search phrases.
    match_implication_dict = {
        "nsubj": MatchImplication(
            search_phrase_dependency="nsubj",
            document_dependencies=[
                "csubj",
                "poss",
                "pobjb",
                "pobjo",
                "advmodsubj",
                "arg",
            ],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "acomp": MatchImplication(
            search_phrase_dependency="acomp",
            document_dependencies=["amod", "advmod", "npmod", "advcl"],
            reverse_document_dependencies=[
                "nsubj",
                "csubj",
                "poss",
                "pobjb",
                "advmodsubj",
                "dobj",
                "pobjo",
                "relant",
                "nsubjpass",
                "csubjpass",
                "compound",
                "advmodobj",
                "dative",
                "arg",
            ],
        ),
        "advcl": MatchImplication(
            search_phrase_dependency="advcl",
            document_dependencies=[
                "pobjo",
                "poss",
                "relant",
                "nsubjpass",
                "csubjpass",
                "compound",
                "advmodobj",
                "arg",
                "dobj",
                "xcomp",
            ],
        ),
        "amod": MatchImplication(
            search_phrase_dependency="amod",
            document_dependencies=["acomp", "advmod", "npmod", "advcl", "compound"],
            reverse_document_dependencies=[
                "nsubj",
                "csubj",
                "poss",
                "pobjb",
                "advmodsubj",
                "dobj",
                "pobjo",
                "relant",
                "nsubjpass",
                "csubjpass",
                "compound",
                "advmodobj",
                "dative",
                "arg",
            ],
        ),
        "advmod": MatchImplication(
            search_phrase_dependency="advmod",
            document_dependencies=["acomp", "amod", "npmod", "advcl"],
        ),
        "arg": MatchImplication(
            search_phrase_dependency="arg",
            document_dependencies=[
                "nsubj",
                "csubj",
                "poss",
                "pobjb",
                "advmodsubj",
                "dobj",
                "pobjo",
                "relant",
                "nsubjpass",
                "csubjpass",
                "compound",
                "advmodobj",
                "dative",
                "pobjp",
            ],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "compound": MatchImplication(
            search_phrase_dependency="compound",
            document_dependencies=[
                "nmod",
                "appos",
                "nounmod",
                "nsubj",
                "csubj",
                "poss",
                "pobjb",
                "advmodsubj",
                "dobj",
                "pobjo",
                "relant",
                "pobjp",
                "nsubjpass",
                "csubjpass",
                "arg",
                "advmodobj",
                "dative",
                "amod",
            ],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "dative": MatchImplication(
            search_phrase_dependency="dative",
            document_dependencies=["pobjt", "relant", "nsubjpass"],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "pobjt": MatchImplication(
            search_phrase_dependency="pobjt",
            document_dependencies=["dative", "relant"],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "nsubjpass": MatchImplication(
            search_phrase_dependency="nsubjpass",
            document_dependencies=[
                "dobj",
                "pobjo",
                "poss",
                "relant",
                "csubjpass",
                "compound",
                "advmodobj",
                "arg",
                "dative",
            ],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "dobj": MatchImplication(
            search_phrase_dependency="dobj",
            document_dependencies=[
                "pobjo",
                "poss",
                "relant",
                "nsubjpass",
                "csubjpass",
                "compound",
                "advmodobj",
                "arg",
                "xcomp",
                "advcl",
            ],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "nmod": MatchImplication(
            search_phrase_dependency="nmod",
            document_dependencies=["appos", "compound", "nummod"],
        ),
        "poss": MatchImplication(
            search_phrase_dependency="poss",
            document_dependencies=[
                "pobjo",
                "nsubj",
                "csubj",
                "pobjb",
                "advmodsubj",
                "arg",
                "relant",
                "nsubjpass",
                "csubjpass",
                "compound",
                "advmodobj",
                "det",
            ],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "pobjo": MatchImplication(
            search_phrase_dependency="pobjo",
            document_dependencies=[
                "poss",
                "dobj",
                "relant",
                "nsubjpass",
                "csubjpass",
                "compound",
                "advmodobj",
                "arg",
                "xcomp",
                "nsubj",
                "csubj",
                "advmodsubj",
            ],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "pobjb": MatchImplication(
            search_phrase_dependency="pobjb",
            document_dependencies=["nsubj", "csubj", "poss", "advmodsubj", "arg"],
            reverse_document_dependencies=["acomp", "amod"],
        ),
        "pobjp": MatchImplication(
            search_phrase_dependency="pobjp", document_dependencies=["compound"]
        ),
        "pobj": MatchImplication(
            search_phrase_dependency="pobj", document_dependencies=["pcomp"]
        ),
        "pcomp": MatchImplication(
            search_phrase_dependency="pcomp", document_dependencies=["pobj"]
        ),
        "prep": MatchImplication(
            search_phrase_dependency="prep", document_dependencies=["prepposs"]
        ),
        "wh_wildcard": MatchImplication(
            search_phrase_dependency="wh_wildcard",
            document_dependencies=["advmod", "advcl", "npadvmod", "prep", "pobjp"],
        ),
        "xcomp": MatchImplication(
            search_phrase_dependency="xcomp",
            document_dependencies=[
                "pobjo",
                "poss",
                "relant",
                "nsubjpass",
                "csubjpass",
                "compound",
                "advmodobj",
                "arg",
                "dobj",
                "advcl",
            ],
        ),
    }

    # The templates used to generate topic matching phraselets.
    phraselet_templates = [
        PhraseletTemplate(
            "predicate-actor",
            "A thing does",
            2,
            1,
            ["nsubj", "csubj", "pobjb", "advmodsubj"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "predicate-patient",
            "Somebody does a thing",
            1,
            3,
            ["dobj", "relant", "advmodobj", "xcomp"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "word-ofword",
            "A thing of a thing",
            1,
            4,
            ["pobjo", "poss"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "predicate-toughmovedargument",
            "A thing is easy to do",
            5,
            1,
            ["arg"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "predicate-passivesubject",
            "A thing is done",
            3,
            1,
            ["nsubjpass", "csubjpass"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "be-attribute",
            "Something is a thing",
            1,
            3,
            ["attr"],
            ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            reverse_only=True,
            question=False,
        ),
        PhraseletTemplate(
            "predicate-recipient",
            "Somebody gives a thing something",
            1,
            3,
            ["dative", "pobjt"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "governor-adjective",
            "A big thing",
            2,
            1,
            ["acomp", "amod", "advmod", "npmod", "advcl", "dobj"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["JJ", "JJR", "JJS", "VBN", "RB", "RBR", "RBS"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "noun-noun",
            "A thing thing",
            2,
            1,
            ["nmod", "appos", "compound", "nounmod"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "number-noun",
            "Seven things",
            1,
            0,
            ["nummod"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            ["CD"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "prepgovernor-noun",
            "A thing in a thing",
            1,
            4,
            ["pobjp"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            reverse_only=False,
            question=False,
        ),
        PhraseletTemplate(
            "prep-noun",
            "in a thing",
            0,
            2,
            ["pobj", "pcomp"],
            ["IN"],
            ["FW", "NN", "NNP", "NNPS", "NNS"],
            reverse_only=True,
            question=False,
        ),
        PhraseletTemplate(
            "head-WHattr",
            "what is this?",
            1,
            0,
            ["attr"],
            ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["WP"],
            reverse_only=False,
            question=True,
        ),
        PhraseletTemplate(
            "head-WHsubj",
            "who came?",
            1,
            0,
            ["nsubj", "nsubjpass", "pobjb"],
            ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["WP"],
            reverse_only=False,
            question=True,
        ),
        PhraseletTemplate(
            "head-WHobj",
            "who did you see?",
            3,
            0,
            ["dobj", "pobjo"],
            ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["WP"],
            reverse_only=False,
            question=True,
        ),
        PhraseletTemplate(
            "head-WHadv",
            "where did you go?",
            3,
            0,
            ["advmod"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["WRB"],
            reverse_only=False,
            question=True,
            assigned_dependency_label="wh_wildcard",
        ),
        PhraseletTemplate(
            "headprep-WH",
            "what did you put it in?",
            3,
            0,
            ["pobjp"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["WP"],
            reverse_only=False,
            question=True,
        ),
        PhraseletTemplate(
            "headprepto-WH",
            "who did you say it to?",
            3,
            0,
            ["pobjt"],
            ["FW", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            ["WP"],
            reverse_only=False,
            question=True,
        ),
        PhraseletTemplate(
            "word",
            "thing",
            0,
            None,
            None,
            ["FW", "NN", "NNP", "NNPS", "NNS"],
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
        """Checks whether *search_phrase_token* is a question word matching *document_token*."""

        if search_phrase_token._.holmes.lemma.startswith("who"):
            ent_types = ("PERSON", "NORP", "ORG", "GPE")
            if document_token.ent_type_ in ent_types:
                return True
            if (
                self.token_matches_ent_type(
                    document_vector,
                    entity_label_to_vector_dict,
                    ent_types,
                    initial_question_word_embedding_match_threshold,
                )
                > 0
            ):
                return True
            return (
                len(
                    [
                        1
                        for i in document_token._.holmes.token_and_coreference_chain_indexes
                        if len(document_token.doc[i].morph.get("Gender")) > 0
                        and document_token.doc[i].morph.get("Gender")[0]
                        in ("Masc", "Fem")
                    ]
                )
                > 0
            )
        if search_phrase_token._.holmes.lemma == "what":
            return True
        if search_phrase_token._.holmes.lemma == "where":
            return (
                document_token.ent_type_ not in ("DATE", "TIME")
                and len(
                    [
                        1
                        for c in document_token._.holmes.children
                        if c.child_token(document_token.doc).ent_type_
                        in ("DATE", "TIME")
                    ]
                )
                == 0
                and document_token.tag_ == "IN"
                and document_token._.holmes.lemma
                in (
                    "above",
                    "across",
                    "against",
                    "along",
                    "among",
                    "amongst",
                    "around",
                    "at",
                    "behind",
                    "below",
                    "beneath",
                    "beside",
                    "between",
                    "beyond",
                    "by",
                    "close",
                    "down",
                    "in",
                    "into",
                    "near",
                    "next",
                    "off",
                    "on",
                    "onto",
                    "opposite",
                    "out",
                    "outside",
                    "round",
                    "through",
                    "under",
                    "underneath",
                    "up",
                )
            )
        if search_phrase_token._.holmes.lemma == "when":
            if document_token.tag_ == "IN":
                return document_token._.holmes.lemma in (
                    "after",
                    "before",
                    "by",
                    "for",
                    "since",
                    "till",
                    "until",
                )
            return document_token.ent_type_ in ("DATE", "TIME")
        if search_phrase_token._.holmes.lemma == "how":
            return document_token.tag_ == "IN" and document_token._.holmes.lemma in (
                "by",
                "with",
            )
        if search_phrase_token._.holmes.lemma == "why":
            if document_token.tag_ == "IN":
                if document_token._.holmes.lemma == "in":
                    return (
                        len(
                            [
                                1
                                for c in document_token._.holmes.children
                                if c.child_token(document_token.doc)._.holmes.lemma
                                == "order"
                            ]
                        )
                        > 0
                    )
                return document_token._.holmes.lemma in ("because")
            if (
                document_token.dep_ in ("advcl", "prep")
                and document_token.text.lower() == "owing"
            ):
                return True
            if (
                document_token.dep_ == "npadvmod"
                and document_token.text.lower() == "thanks"
            ):
                return True

            return (
                document_token.dep_ in ("advmod", "advcl", "acomp")
                and len(
                    [
                        1
                        for c in document_token.children
                        if c._.holmes.lemma in ("because") or c.tag_ == "TO"
                    ]
                )
                > 0
            )
            # syntactic not semantic children to handle subject-predicate phrases correctly
        return False
