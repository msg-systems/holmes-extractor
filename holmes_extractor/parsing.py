from typing import List, Dict, Optional, Tuple, Generator, cast, Set, Union
import math
import pickle
import importlib
from abc import ABC, abstractmethod
from copy import copy
from functools import total_ordering
import srsly
import pkg_resources
from spacy.language import Language
from spacy.vocab import Vocab
from spacy.tokens import Token, Doc
from thinc.api import get_current_ops
from thinc.types import Floats1d
from holmes_extractor.ontology import Ontology
from .errors import (
    DocumentTooBigError,
    SearchPhraseContainsNegationError,
    SearchPhraseContainsConjunctionError,
    SearchPhraseWithoutMatchableWordsError,
    SearchPhraseContainsMultipleClausesError,
    SearchPhraseContainsCoreferringPronounError,
)

SERIALIZED_DOCUMENT_VERSION = "4.0"


class SemanticDependency:
    """A labelled semantic dependency between two tokens."""

    def __init__(
        self,
        parent_index: int,
        child_index: int,
        label: str = None,
        is_uncertain: bool = False,
    ) -> None:
        """Args:

        parent_index -- the index of the parent token within the document.
        child_index -- the index of the child token within the document, or one less than zero
            minus the index of the child token within the document for a grammatical dependency. A
            grammatical dependency is always in a non-final position within a chain of dependencies
            ending in one or more non-grammatical (lexical / normal) dependencies. When creating
            both Holmes semantic structures and search phrases, grammatical dependencies are
            sometimes replaced by the lexical dependencies at the end of their chains.
        label -- the label of the semantic dependency, which must be *None* for grammatical
            dependencies.
        is_uncertain -- if *True*, any match involving this dependency will itself be uncertain.
        """
        if child_index < 0 and label is not None:
            raise RuntimeError(
                "Semantic dependency with negative child index may not have a label."
            )
        if parent_index == child_index:
            raise RuntimeError(
                " ".join(
                    (
                        "Attempt to create self-referring semantic dependency with index",
                        str(parent_index),
                    )
                )
            )
        self.parent_index = parent_index
        self.child_index = child_index
        self.label = label
        self.is_uncertain = is_uncertain

    def parent_token(self, doc: Doc) -> Token:
        """Convenience method to return the parent token of this dependency.

        doc -- the document containing the token.
        """
        index = self.parent_index
        if index < 0:
            index = -1 - index
        return doc[index]

    def child_token(self, doc: Doc) -> Token:
        """Convenience method to return the child token of this dependency.

        doc -- the document containing the token.
        """
        index = self.child_index
        if index < 0:
            index = -1 - index
        return doc[index]

    def __str__(self) -> str:
        """e.g. *2:nsubj* or *2:nsubj(U)* to represent uncertainty."""
        working_label = str(self.label)
        if self.is_uncertain:
            working_label = "".join((working_label, "(U)"))
        return ":".join((str(self.child_index), working_label))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, SemanticDependency)
            and self.parent_index == other.parent_index
            and self.child_index == other.child_index
            and self.label == other.label
            and self.is_uncertain == other.is_uncertain
        )

    def __hash__(self) -> int:
        return hash(
            (self.parent_index, self.child_index, self.label, self.is_uncertain)
        )


class Mention:
    """Simplified information about a coreference mention with respect to a specific token."""

    def __init__(self, root_index: int, indexes: List[int]) -> None:
        """
        root_index -- the index of the member of *indexes* that forms the syntactic head of any
            coordinated phrase, or *indexes[0]* if *len(indexes) == 1*.
        indexes -- the indexes of the tokens that make up the mention. If there is more than one
            token, they must form a coordinated phrase.
        """
        self.root_index = root_index
        self.indexes = indexes

    def __str__(self) -> str:
        return "".join(("[", str(self.root_index), "; ", str(self.indexes), "]"))


class Subword:
    """A semantically atomic part of a word. Currently only used for German.

    containing_token_index -- the index of the containing token within the document.
    index -- the index of the subword within the word.
    text -- the original subword string.
    lemma -- the model-normalized representation of the subword string.
    derived_lemma -- where relevant, another lemma with which *lemma* is derivationally related
    and which can also be useful for matching in some usecases; otherwise *None*
    vector -- the vector representation of *lemma*, or *None* if there is none available.
    char_start_index -- the character index of the subword within the containing word.
    dependent_index -- the index of a subword that is dependent on this subword, or *None*
        if there is no such subword.
    dependency_label -- the label of the dependency between this subword and its dependent,
        or *None* if it has no dependent.
    governor_index -- the index of a subword on which this subword is dependent, or *None*
        if there is no such subword.
    governing_dependency_label -- the label of the dependency between this subword and its
        governor, or *None* if it has no governor.
    """

    def __init__(
        self,
        containing_token_index: int,
        index: int,
        text: str,
        lemma: str,
        derived_lemma: str,
        vector: Floats1d,
        char_start_index: int,
        dependent_index: Optional[int],
        dependency_label: Optional[str],
        governor_index: Optional[int],
        governing_dependency_label: Optional[str],
    ):
        self.containing_token_index = containing_token_index
        self.index = index
        self.text = text
        self.lemma = lemma
        self.direct_matching_reprs = [lemma]
        if text != lemma:
            self.direct_matching_reprs.append(text)
        self.derived_lemma = derived_lemma
        if derived_lemma != lemma:
            self.derivation_matching_reprs: Optional[List[str]] = [derived_lemma]
        else:
            self.derivation_matching_reprs = None
        self.vector = vector
        self.char_start_index = char_start_index
        self.dependent_index = dependent_index
        self.dependency_label = dependency_label
        self.governor_index = governor_index
        self.governing_dependency_label = governing_dependency_label

    @property
    def is_head(self) -> bool:
        return self.governor_index is None

    def __str__(self) -> str:
        if self.derived_lemma is not None:
            lemma_string = "".join((self.lemma, "(", self.derived_lemma, ")"))
        else:
            lemma_string = self.lemma
        return "/".join((self.text, lemma_string))


@total_ordering
class Index:
    """The position of a multiword, word or subword within a document."""

    def __init__(self, token_index: int, subword_index: Optional[int]) -> None:
        self.token_index = token_index
        self.subword_index = subword_index

    def is_subword(self) -> bool:
        return self.subword_index is not None

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Index)
            and self.token_index == other.token_index
            and self.subword_index == other.subword_index
        )

    def __lt__(self, other) -> bool:
        if not isinstance(other, Index):
            raise RuntimeError("Comparison between Index and another type.")
        if self.token_index < other.token_index:
            return True
        if not self.is_subword() and other.is_subword():
            return True
        if (
            self.is_subword()
            and other.is_subword()
            and self.subword_index < other.subword_index  # type:ignore[operator]
        ):
            return True
        return False

    def __hash__(self) -> int:
        return hash((self.token_index, self.subword_index))


class CorpusWordPosition:
    """A reference to a word or subword within a corpus of one or more documents."""

    def __init__(self, document_label: str, index: Index) -> None:
        if document_label is None:
            raise RuntimeError("CorpusWordPosition.document_label must have a value.")
        self.document_label = document_label
        self.index = index

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, CorpusWordPosition)
            and self.document_label == other.document_label
            and self.index == other.index
        )

    def __hash__(self) -> int:
        return hash((self.document_label, self.index))

    def __str__(self) -> str:
        return ":".join((self.document_label, str(self.index)))


class MultiwordSpan:
    def __init__(
        self,
        text: str,
        hyphen_normalized_lemma: Optional[str],
        lemma: str,
        derived_lemma: Optional[str],
        token_indexes: List[int],
    ) -> None:
        """Args:

        text -- the raw text representation of the multiword span
        lemma - the lemma representation of the multiword span
        hyphen_normalized_lemma -- a hyphen-normalized representation of *lemmaÜ
        derived_lemma - the lemma representation with individual words that have derived
            lemmas replaced by those derived lemmas
        token_indexes -- a list of token indexes that make up the multiword span
        """
        text = text
        self.text = text
        self.lemma = lemma
        self.derived_lemma = derived_lemma
        self.direct_matching_reprs = [lemma]
        if hyphen_normalized_lemma != lemma and hyphen_normalized_lemma is not None:
            self.direct_matching_reprs.append(hyphen_normalized_lemma)
        if lemma != text.lower():
            self.direct_matching_reprs.append(text.lower())
        if derived_lemma != lemma and derived_lemma is not None:
            self.derivation_matching_reprs: Optional[List[str]] = [derived_lemma]
        else:
            self.derivation_matching_reprs = None
        self.token_indexes = token_indexes


class MatchImplication:
    """Entry describing which document dependencies match a given search phrase dependency.

    Parameters:

    search_phrase_dependency -- the search phrase dependency.
    document_dependencies -- the matching document dependencies.
    reverse_document_dependencies -- document dependencies that match when the polarity is
        opposite to the polarity of *search_phrase_dependency*.
    """

    def __init__(
        self,
        *,
        search_phrase_dependency: str,
        document_dependencies: List[str],
        reverse_document_dependencies: Optional[List[str]] = None
    ):
        self.search_phrase_dependency = search_phrase_dependency
        self.document_dependencies = document_dependencies
        if reverse_document_dependencies is None:
            reverse_document_dependencies = []
        self.reverse_document_dependencies: List[str] = reverse_document_dependencies


class HolmesDocumentInfo:
    def __init__(self, semantic_analyzer: "SemanticAnalyzer"):
        self.model = semantic_analyzer.get_model_name()
        self.serialized_document_version = SERIALIZED_DOCUMENT_VERSION

    @srsly.msgpack_encoders("holmes_document_info_holder")
    def serialize_obj(obj, chain=None):
        if isinstance(obj, HolmesDocumentInfo):
            return {"__holmes_document_info_holder__": pickle.dumps(obj)}
        return obj if chain is None else chain(obj)

    @srsly.msgpack_decoders("holmes_document_info_holder")
    def deserialize_obj(obj, chain=None):
        if "__holmes_document_info_holder__" in obj:
            return pickle.loads(obj["__holmes_document_info_holder__"])
        return obj if chain is None else chain(obj)


class HolmesDictionary:
    """The holder object for token-level semantic information managed by Holmes

    Holmes dictionaries are accessed using the syntax *token._.holmes*.

    index -- the index of the token
    lemma -- the value returned from *._.holmes.lemma* for the token.
    hyphen_normalized_lemma -- a hyphen-normalized version of *lemma*.
    derived_lemma -- the value returned from *._.holmes.derived_lemma for the token; where relevant,
        another lemma with which *lemma* is derivationally related and which can also be useful for
        matching in some usecases; otherwise the same value as *._.holmes.lemma*.
    direct_matching_reprs -- a list of representations of the token that can be used for direct
        matching, consisting of *token.text* and optionally a hyphen-normalized version of *token.text*
        and *token.lemma_* if these are different from *token.text*.
    derivation_matching_reprs -- if *lemma != derived_lemma*, a list of representations of the token
        that can be used for derivation matching, consisting of *derived_lema*, *token.text*
        and optionally a hyphen-normalized version of *token.text* and *token.lemma_* if these
        are different from *token.text*; otherwise *None*.
    vector -- the vector representation of *lemma*, unless *lemma* is a multiword, in which case
        the vector representation of *token.lemma_* is used instead. *None* where there is no
        vector for the lexeme.
    multiword_spans -- where relevant, a list of multiword spans, otherwise *None*. Set after initialization.
    """

    def __init__(
        self,
        index: int,
        lemma: str,
        hyphen_normalized_lemma: str,
        derived_lemma: str,
        direct_matching_reprs: List[str],
        derivation_matching_reprs: Optional[List[str]],
        vector: Floats1d,
    ):
        self.index = index
        self.lemma = lemma
        self.hyphen_normalized_lemma = hyphen_normalized_lemma
        self.derived_lemma = derived_lemma
        self.direct_matching_reprs = direct_matching_reprs
        self.derivation_matching_reprs = derivation_matching_reprs
        self.multiword_spans: List[MultiwordSpan] = []
        self.vector = vector
        self.children: List[
            SemanticDependency
        ] = []  # list of *SemanticDependency* objects where this token is the parent.
        self.parents: List[
            SemanticDependency
        ] = []  # list of *SemanticDependency* objects where this token is the child.
        self.righthand_siblings: List[
            Token
        ] = []  # list of tokens to the right of this token that stand in a
        # conjunction relationship to this token and that share its semantic parents.
        self.token_or_lefthand_sibling_index = (
            None  # the index of this token's lefthand sibling,
        )
        # or this token's own index if this token has no lefthand sibling.
        self.is_involved_in_or_conjunction = False
        self.is_negated: Optional[bool] = None
        self.is_matchable: Optional[bool] = None
        self.is_initial_question_word = False
        self.has_initial_question_word_in_phrase = False
        self.coreference_linked_child_dependencies: List[
            Tuple[Index, str]
        ] = []  # list of [index, label] specifications of
        # dependencies where this token is the parent, taking any coreference resolution into
        # account. Used in topic matching.
        self.coreference_linked_parent_dependencies: List[
            Tuple[Index, str]
        ] = []  # list of [index, label] specifications of
        # dependencies where this token is the child, taking any coreference resolution into
        # account. Used in topic matching.
        self.token_and_coreference_chain_indexes: Optional[
            List[int]
        ] = None  # where no coreference, only the token
        # index; where coreference, the token index followed by the indexes of coreferring tokens
        self.mentions: List[int] = []
        self.subwords: List[int] = []

    @property
    def is_uncertain(self) -> bool:
        """if *True*, a match involving this token will itself be uncertain."""
        return self.is_involved_in_or_conjunction

    def loop_token_and_righthand_siblings(
        self, doc: Doc
    ) -> Generator[Token, None, None]:
        """Convenience generator to loop through this token and any righthand siblings."""
        indexes = [self.index]
        indexes.extend(self.righthand_siblings)
        indexes = sorted(
            indexes
        )  # in rare cases involving truncated nouns in German, righthand
        # siblings can actually end up to the left of the head word.
        for index in indexes:
            yield doc[index]

    def get_sibling_indexes(self, doc: Doc) -> List[Token]:
        """Returns the indexes of this token and any siblings, ordered from left to right."""
        # with truncated nouns in German, the righthand siblings may occasionally occur to the left
        # of the head noun
        head_sibling = doc[self.token_or_lefthand_sibling_index]
        indexes = [self.token_or_lefthand_sibling_index]
        indexes.extend(head_sibling._.holmes.righthand_siblings)
        return sorted(indexes)

    def has_dependency_with_child_index(self, index: int) -> bool:
        for dependency in self.children:
            if dependency.child_index == index:
                return True
        return False

    def get_label_of_dependency_with_child_index(self, index: int) -> Optional[str]:
        for dependency in self.children:
            if dependency.child_index == index:
                return dependency.label
        return None

    def has_dependency_with_label(self, label: str) -> bool:
        for dependency in self.children:
            if dependency.label == label:
                return True
        return False

    def has_dependency_with_child_index_and_label(self, index: int, label: str) -> bool:
        for dependency in self.children:
            if dependency.child_index == index and dependency.label == label:
                return True
        return False

    def remove_dependency_with_child_index(self, index: int) -> None:
        self.children = [dep for dep in self.children if dep.child_index != index]

    def string_representation_of_children(self) -> str:
        children = sorted(self.children, key=lambda dependency: dependency.child_index)
        return "; ".join(str(child) for child in children)

    def string_representation_of_parents(self) -> str:
        parents = sorted(self.parents, key=lambda dependency: dependency.parent_index)
        return "; ".join(
            ":".join((str(parent.parent_index), cast(str, parent.label)))
            for parent in parents
        )

    def is_involved_in_coreference(self) -> bool:
        return len(self.mentions) > 0

    @srsly.msgpack_encoders("holmes_dictionary_holder")
    def serialize_obj(obj, chain=None):
        if isinstance(obj, HolmesDictionary):
            return {"__holmes_dictionary_holder__": pickle.dumps(obj)}
        return obj if chain is None else chain(obj)

    @srsly.msgpack_decoders("holmes_dictionary_holder")
    def deserialize_obj(obj, chain=None):
        if "__holmes_dictionary_holder__" in obj:
            return pickle.loads(obj["__holmes_dictionary_holder__"])
        return obj if chain is None else chain(obj)


class PhraseletTemplate:
    """A template for a phraselet used in topic matching.

    Properties:

    label -- a label for the relation which will be used to form part of the labels of phraselets
        derived from this template.
    template_sentence -- a sentence with the target grammatical structure for phraselets derived
        from this template.
    template_doc -- a spacy Doc representing *template_sentence* (set by the *Manager* object)
    parent_index -- the index within *template_sentence* of the parent participant in the dependency
        (for relation phraselets) or of the word (for single-word phraselets).
    child_index -- the index within *template_sentence* of the child participant in the dependency
        (for relation phraselets) or 'None' for single-word phraselets.
    dependency_labels -- the labels of dependencies that match the template
        (for relation phraselets) or 'None' for single-word phraselets.
    parent_tags -- the tag_ values of parent participants in the dependency (for parent phraselets)
        of of the word (for single-word phraselets) that match the template.
    child_tags -- the tag_ values of child participants in the dependency (for parent phraselets)
        that match the template, or 'None' for single-word phraselets.
    reverse_only -- specifies that relation phraselets derived from this template should only be
        reverse-matched, e.g. matching should only be attempted during topic matching when the
        possible child token has already been matched to a single-word phraselet. This
        is used for performance reasons when the parent tag belongs to a closed word class like
        prepositions. Reverse-only phraselets are ignored in supervised document classification.
    question -- a question template involves an interrogative pronoun in the child position and is
        always matched regardless of corpus frequencies.
    assigned_dependency_label -- if a value other than 'None', specifies a dependency label that
        should be used to relabel the relationship between the parent and child participants.
        Has no effect if child_index is None.
    """

    def __init__(
        self,
        label: str,
        template_sentence: str,
        parent_index: int,
        child_index: Optional[int],
        dependency_labels: Optional[List[str]],
        parent_tags: List[str],
        child_tags: Optional[List[str]],
        *,
        reverse_only: bool,
        question: bool,
        assigned_dependency_label: Optional[str] = None
    ):
        self.label = label
        self.template_sentence = template_sentence
        self.parent_index = parent_index
        self.child_index = child_index
        self.dependency_labels = dependency_labels
        self.parent_tags = parent_tags
        self.child_tags = child_tags
        self.reverse_only = reverse_only
        self.question = question
        self.assigned_dependency_label = assigned_dependency_label
        self.template_doc: Doc

    def single_word(self) -> bool:
        """'True' if this is a template for single-word phraselets, otherwise 'False'."""
        return self.child_index is None


class PhraseletInfo:
    """Information describing a topic matching phraselet.

    Parameters:

    label -- the phraselet label, e.g. 'predicate-patient: open-door'
    template_label -- the value of 'PhraseletTemplate.label', e.g. 'predicate-patient'
    parent_lemma -- the parent lemma, or the lemma for single-word phraselets.
    parent_hyphen_normalized_lemma -- a hyphen-normalized version of *parent_lemma*-
    parent_derived_lemma -- the parent derived lemma, or the derived lemma for single-word
        phraselets.
    parent_pos -- the part of speech tag of the token that supplied the parent word.
    parent_ent_type -- the parent entity label, or the entity label for single-word
        phraselets. '' if there is none.
    parent_is_initial_question_word -- 'True' or 'False'
    parent_has_initial_question_word_in_phrase -- 'True' or 'False'
    child_lemma -- the child lemma, or 'None' for single-word phraselets.
    child_hyphen_normalized_lemma -- a hyphen-normalized version of *child_lemma*-
    child_derived_lemma -- the child derived lemma, or 'None' for single-word phraselets.
    child_pos -- the part of speech tag of the token that supplied the child word, or 'None'
        for single-word phraselets.
    child_ent_type -- the child entity label. '' if there is none; 'None' for single-word
        phraselets.
    child_is_initial_question_word -- 'True' or 'False'
    child_has_initial_question_word_in_phrase -- 'True' or 'False'
    created_without_matching_tags -- 'True' if created without matching tags.
    reverse_only_parent_lemma -- 'True' if the parent lemma is in the reverse matching list.
    frequency_factor -- a multiplication factor between 0.0 and 1.0 which is lower the more
        frequently words occur in the corpus, relating to the whole phraselet.
    parent_frequency_factor -- a multiplication factor between 0.0 and 1.0 which is lower the
        more frequently words occur in the corpus, relating to the parent token.
    child_frequency_factor -- a multiplication factor between 0.0 and 1.0 which is lower the
        more frequently words occur in the corpus, relating to the child token.
    """

    def __init__(
        self,
        label: str,
        template_label: str,
        parent_lemma: str,
        parent_hyphen_normalized_lemma: str,
        parent_derived_lemma: str,
        parent_pos: str,
        parent_ent_type: str,
        parent_is_initial_question_word: bool,
        parent_has_initial_question_word_in_phrase: bool,
        child_lemma: Optional[str],
        child_hyphen_normalized_lemma: Optional[str],
        child_derived_lemma: Optional[str],
        child_pos: Optional[str],
        child_ent_type: Optional[str],
        child_is_initial_question_word: Optional[bool],
        child_has_initial_question_word_in_phrase: Optional[bool],
        created_without_matching_tags: bool,
        reverse_only_parent_lemma: Optional[bool],
        frequency_factor: Optional[float],
        parent_frequency_factor: Optional[float],
        child_frequency_factor: Optional[float],
    ):
        self.label = label
        self.template_label = template_label
        self.parent_lemma = parent_lemma
        self.parent_hyphen_normalized_lemma = parent_hyphen_normalized_lemma
        self.parent_derived_lemma = parent_derived_lemma
        self.parent_pos = parent_pos
        self.parent_ent_type = parent_ent_type
        self.parent_is_initial_question_word = parent_is_initial_question_word
        self.parent_has_initial_question_word_in_phrase = (
            parent_has_initial_question_word_in_phrase
        )
        self.child_lemma = child_lemma
        self.child_hyphen_normalized_lemma = child_hyphen_normalized_lemma
        self.child_derived_lemma = child_derived_lemma
        self.child_pos = child_pos
        self.child_ent_type = child_ent_type
        self.child_is_initial_question_word = child_is_initial_question_word
        self.child_has_initial_question_word_in_phrase = (
            child_has_initial_question_word_in_phrase
        )
        self.created_without_matching_tags = created_without_matching_tags
        self.reverse_only_parent_lemma = reverse_only_parent_lemma
        self.frequency_factor = frequency_factor
        self.parent_frequency_factor = parent_frequency_factor
        self.child_frequency_factor = child_frequency_factor
        self.set_parent_reprs()
        self.set_child_reprs()

    def set_parent_reprs(self) -> None:
        self.parent_direct_matching_reprs = [self.parent_lemma]
        if self.parent_lemma != self.parent_hyphen_normalized_lemma:
            self.parent_direct_matching_reprs.append(
                self.parent_hyphen_normalized_lemma
            )
        if self.parent_lemma != self.parent_derived_lemma:
            self.parent_derivation_matching_reprs: Optional[List[str]] = [
                self.parent_derived_lemma
            ]
        else:
            self.parent_derivation_matching_reprs = None

    def set_child_reprs(self) -> None:
        if self.child_lemma is not None:
            self.child_direct_matching_reprs: Optional[List[str]] = [self.child_lemma]
            if self.child_lemma != self.child_hyphen_normalized_lemma:
                self.child_direct_matching_reprs.append(
                    cast(str, self.child_hyphen_normalized_lemma)
                )
            if self.child_lemma != self.child_derived_lemma:
                self.child_derivation_matching_reprs: Optional[List[str]] = [
                    cast(str, self.child_derived_lemma)
                ]
            else:
                self.child_derivation_matching_reprs = None
        else:
            self.child_direct_matching_reprs = None
            self.child_derivation_matching_reprs = None

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, PhraseletInfo)
            and self.label == other.label
            and self.template_label == other.template_label
            and self.parent_lemma == other.parent_lemma
            and self.parent_hyphen_normalized_lemma
            == other.parent_hyphen_normalized_lemma
            and self.parent_derived_lemma == other.parent_derived_lemma
            and self.parent_pos == other.parent_pos
            and self.parent_ent_type == other.parent_ent_type
            and self.parent_is_initial_question_word
            == other.parent_is_initial_question_word
            and self.parent_has_initial_question_word_in_phrase
            == other.parent_has_initial_question_word_in_phrase
            and self.child_lemma == other.child_lemma
            and self.child_hyphen_normalized_lemma
            == other.child_hyphen_normalized_lemma
            and self.child_derived_lemma == other.child_derived_lemma
            and self.child_pos == other.child_pos
            and self.child_ent_type == other.child_ent_type
            and self.child_is_initial_question_word
            == other.child_is_initial_question_word
            and self.child_has_initial_question_word_in_phrase
            == other.child_has_initial_question_word_in_phrase
            and self.created_without_matching_tags
            == other.created_without_matching_tags
            and self.reverse_only_parent_lemma == other.reverse_only_parent_lemma
            and str(self.frequency_factor) == str(other.frequency_factor)
            and str(self.parent_frequency_factor) == str(other.parent_frequency_factor)
            and str(self.child_frequency_factor) == str(other.child_frequency_factor)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.label,
                self.template_label,
                self.parent_lemma,
                self.parent_derived_lemma,
                self.parent_hyphen_normalized_lemma,
                self.parent_pos,
                self.parent_ent_type,
                self.parent_is_initial_question_word,
                self.parent_has_initial_question_word_in_phrase,
                self.child_lemma,
                self.child_derived_lemma,
                self.child_pos,
                self.child_ent_type,
                self.child_hyphen_normalized_lemma,
                self.child_is_initial_question_word,
                self.child_has_initial_question_word_in_phrase,
                self.created_without_matching_tags,
                self.reverse_only_parent_lemma,
                str(self.frequency_factor),
                str(self.parent_frequency_factor),
                str(self.child_frequency_factor),
            )
        )


class SearchPhrase:
    def __init__(
        self,
        doc: Doc,
        matchable_token_indexes: List[int],
        root_token_index: int,
        matchable_non_entity_tokens_to_vectors: Dict[Token, Floats1d],
        label: str,
        topic_match_phraselet: bool,
        topic_match_phraselet_created_without_matching_tags: bool,
        question_phraselet: bool,
        reverse_only: bool,
        treat_as_reverse_only_during_initial_relation_matching: bool,
        has_single_matchable_word: bool,
    ):
        """Args:

        doc -- the Holmes document created for the search phrase
        matchable_token_indexes -- a list of indexes of tokens all of which must have counterparts
            in the document to produce a match
        root_token_index -- the index of the token at which recursive matching starts
        matchable_non_entity_tokens_to_vectors -- dictionary from token indexes to vectors.
            Only used when embedding matching is active.
        label -- a label for the search phrase.
        topic_match_phraselet -- 'True' if a topic match phraselet, otherwise 'False'.
        topic_match_phraselet_created_without_matching_tags -- 'True' if a topic match
        phraselet created without matching tags (match_all_words), otherwise 'False'.
        question_phraselet -- 'True' if a topic match phraselet where the child member is
        an initial question word, otherwise 'False'
        reverse_only -- 'True' if a phraselet that should only be reverse-matched.
        treat_as_reverse_only_during_initial_relation_matching -- phraselets are
            set to *True* in the context of topic matching to prevent them from being taken into
            account during initial relation matching because the parent relation occurs too
            frequently during the corpus. *reverse_only* cannot be used instead because it
            has an effect on scoring.
        has_single_matchable_word -- **True** or **False**.
        """
        self.doc = doc
        self.doc_text = doc.text
        self.matchable_token_indexes = matchable_token_indexes
        self.root_token_index = root_token_index
        self.matchable_non_entity_tokens_to_vectors = (
            matchable_non_entity_tokens_to_vectors
        )
        self.label = label
        self.topic_match_phraselet = topic_match_phraselet
        self.topic_match_phraselet_created_without_matching_tags = (
            topic_match_phraselet_created_without_matching_tags
        )
        self.question_phraselet = question_phraselet
        self.reverse_only = reverse_only
        self.treat_as_reverse_only_during_initial_relation_matching = (
            treat_as_reverse_only_during_initial_relation_matching
        )
        self.words_matching_root_token: List[str] = []
        self.has_single_matchable_word = (
            has_single_matchable_word  # len(matchable_token_indexes) == 1
        )

    @property
    def matchable_tokens(self) -> List[Token]:
        return [self.doc[index] for index in self.matchable_token_indexes]

    @property
    def root_token(self) -> Token:
        return self.doc[self.root_token_index]

    def add_word_information(self, word: str) -> None:
        if word not in self.words_matching_root_token:
            self.words_matching_root_token.append(word)

    def pack(self) -> None:
        """Prepares the search phrase for serialization."""
        self.serialized_doc = self.doc.to_bytes()
        self.doc = None

    def unpack(self, vocab: Vocab) -> None:
        """Restores the search phrase following deserialization."""
        self.doc = Doc(vocab).from_bytes(self.serialized_doc)
        self.serialized_doc = None


class SemanticAnalyzerFactory:
    """Returns the correct *SemanticAnalyzer* for the model language.
    This class must be added to if additional implementations are added for new languages.
    """

    def semantic_analyzer(self, *, nlp: Language, vectors_nlp: Language):
        language = nlp.meta["lang"]
        try:
            language_specific_rules_module = importlib.import_module(
                ".".join((".lang", language, "language_specific_rules")),
                "holmes_extractor",
            )
        except ModuleNotFoundError:
            raise ValueError(" ".join(("Language", language, "not supported")))
        return language_specific_rules_module.LanguageSpecificSemanticAnalyzer(
            nlp=nlp, vectors_nlp=vectors_nlp
        )


class SemanticAnalyzer(ABC):
    """Abstract *SemanticAnalyzer* parent class. A *SemanticAnalyzer* is responsible for adding the
    *token._.holmes* dictionaries to each token within a spaCy document. It requires full access to
    the spaCy *Language* object, cannot be serialized and so may only be called within main
    processes, not from worker processes.

    Functionality is placed here that is common to all
    current implementations. It follows that some functionality will probably have to be moved
    out to specific implementations whenever an implementation for a new language is added.

    For explanations of the abstract variables and methods, see the *EnglishSemanticAnalyzer*
    implementation where they can be illustrated with direct examples.
    """

    language_name: str = NotImplemented

    noun_pos: List[str] = NotImplemented

    predicate_head_pos: List[str] = NotImplemented

    matchable_pos: List[str] = NotImplemented

    adjectival_predicate_head_pos: List[str] = NotImplemented

    adjectival_predicate_subject_pos: List[str] = NotImplemented

    adjectival_predicate_subject_dep: str = NotImplemented

    adjectival_predicate_predicate_dep: str = NotImplemented

    adjectival_predicate_predicate_pos: str = NotImplemented

    modifier_dep: str = NotImplemented

    spacy_noun_to_preposition_dep: str = NotImplemented

    spacy_verb_to_preposition_dep: str = NotImplemented

    holmes_noun_to_preposition_dep: str = NotImplemented

    holmes_verb_to_preposition_dep: str = NotImplemented

    conjunction_deps: List[str] = NotImplemented

    interrogative_pronoun_tags: List[str] = NotImplemented

    semantic_dependency_excluded_tags: List[str] = NotImplemented

    generic_pronoun_lemmas: List[str] = NotImplemented

    or_lemma: str = NotImplemented

    mark_child_dependencies_copied_to_siblings_as_uncertain: bool = NotImplemented

    maximum_mentions_in_coreference_chain: int = NotImplemented

    maximum_word_distance_in_coreference_chain: int = NotImplemented

    sibling_marker_deps: List[str] = NotImplemented

    entity_labels_to_corresponding_lexemes: Dict[str, str] = NotImplemented

    whose_lemma: str = NotImplemented

    @abstractmethod
    def add_subwords(
        self, token: Token, subword_cache: Dict[str, List[Subword]]
    ) -> None:
        pass

    @abstractmethod
    def set_negation(self, token: Token) -> None:
        pass

    @abstractmethod
    def correct_auxiliaries_and_passives(self, token: Token) -> None:
        pass

    @abstractmethod
    def perform_language_specific_tasks(self, token: Token) -> None:
        pass

    @abstractmethod
    def handle_relative_constructions(self, token: Token) -> None:
        pass

    @abstractmethod
    def holmes_lemma(self, token: Token) -> str:
        pass

    @abstractmethod
    def language_specific_derived_holmes_lemma(self, token: Token, lemma: str) -> str:
        pass

    def __init__(self, *, nlp: Language, vectors_nlp: Language) -> None:
        """Args:

        nlp -- the spaCy model
        vectors_nlp -- the spaCy model to use for vocabularies and vectors
        """
        self.nlp = nlp
        self.vectors_nlp = vectors_nlp
        self.model = "_".join((self.nlp.meta["lang"], self.nlp.meta["name"]))
        self.derivational_dictionary = self.load_derivational_dictionary()
        self.serialized_document_version = SERIALIZED_DOCUMENT_VERSION

    def load_derivational_dictionary(self) -> Dict[str, str]:
        in_package_filename = "".join(
            ("lang/", self.nlp.meta["lang"], "/data/derivation.csv")
        )
        absolute_filename = pkg_resources.resource_filename(
            __name__, in_package_filename
        )
        dictionary: Dict[str, str] = {}
        with open(absolute_filename, "r", encoding="utf-8") as file:
            for line in file.readlines():
                words = [word.strip() for word in line.split(",")]
                for index, _ in enumerate(words):
                    dictionary[words[index]] = words[0]
        return dictionary

    _maximum_document_size = 1000000

    def spacy_parse_for_lemmas(self, text: str) -> Doc:
        """Performs a standard spaCy parse on a string in order to obtain lemmas. The 'parser' and 'ner' components
        are disabled as they are not required to perform this task."""
        if len(text) > self._maximum_document_size:
            raise DocumentTooBigError(
                " ".join(
                    ("size:", str(len(text)), "max:", str(self._maximum_document_size))
                )
            )
        return self.nlp(text, disable=["parser", "ner", "coreferee", "holmes"])

    def parse(self, text: str) -> Doc:
        return self.nlp(text)

    def get_vector(self, lemma: str) -> Floats1d:
        """Returns a vector representation of *lemma*, or *None* if none is available."""
        lexeme = self.vectors_nlp.vocab[lemma]
        return lexeme.vector if lexeme.has_vector and lexeme.vector_norm > 0 else None

    def holmes_parse(self, spacy_doc: Doc) -> Doc:
        """Adds the Holmes-specific information to each token within a spaCy document."""
        spacy_doc._.set("holmes_document_info", HolmesDocumentInfo(self))
        for token in spacy_doc:
            lemma = self.holmes_lemma(token)
            derived_lemma = self.derived_holmes_lemma(token, lemma)
            direct_matching_reprs = [lemma]
            hyphen_normalized_lemma = self.normalize_hyphens(lemma)
            if lemma != hyphen_normalized_lemma:
                direct_matching_reprs.append(hyphen_normalized_lemma)
            if token.text.lower() != lemma:
                direct_matching_reprs.append(token.text.lower())
            if derived_lemma != lemma:
                derivation_matching_reprs = [derived_lemma]
            else:
                derivation_matching_reprs = None
            lexeme = self.vectors_nlp.vocab[
                token.lemma_ if len(lemma.split()) > 1 else lemma
            ]
            vector = (
                lexeme.vector if lexeme.has_vector and lexeme.vector_norm > 0 else None
            )
            token._.set(
                "holmes",
                HolmesDictionary(
                    token.i,
                    lemma,
                    hyphen_normalized_lemma,
                    derived_lemma,
                    direct_matching_reprs,
                    derivation_matching_reprs,
                    vector,
                ),
            )
        for token in spacy_doc:
            self.set_negation(token)
        for token in spacy_doc:
            self.initialize_semantic_dependencies(token)
        self.set_initial_question_words(spacy_doc)
        for token in spacy_doc:
            self.mark_if_righthand_sibling(token)
            token._.holmes.token_or_lefthand_sibling_index = (
                self._lefthand_sibling_recursively(token)
            )
        for token in spacy_doc:
            self.copy_any_sibling_info(token)
        subword_cache: Dict[str, List[Subword]] = {}
        for token in spacy_doc:
            self.add_subwords(token, subword_cache)
        for token in spacy_doc:
            self.set_coreference_information(token)
        for token in spacy_doc:
            self.set_matchability(token)
            token._.holmes.multiword_spans = self.multiword_spans_with_head_token(token)
        for token in spacy_doc:
            self.correct_auxiliaries_and_passives(token)
        for token in spacy_doc:
            self.copy_any_sibling_info(token)
        for token in spacy_doc:
            self.handle_relative_constructions(token)
        for token in spacy_doc:
            self.normalize_predicative_adjectives(token)
        for token in spacy_doc:
            self.create_additional_preposition_phrase_semantic_dependencies(token)
        for token in spacy_doc:
            self.perform_language_specific_tasks(token)
        for token in spacy_doc:
            self.create_convenience_dependencies(token)
        return spacy_doc

    def _lefthand_sibling_recursively(self, token: Token) -> int:
        """If *token* is a righthand sibling, return the index of the token that has a sibling
        reference to it, otherwise return the index of *token* itself.
        """
        if token.dep_ not in self.conjunction_deps:
            return token.i
        else:
            return self._lefthand_sibling_recursively(token.head)

    def debug_structures(self, doc: Doc) -> None:
        for token in doc:
            if token._.holmes.derived_lemma is not None:
                lemma_string = "".join(
                    (token._.holmes.lemma, "(", token._.holmes.derived_lemma, ")")
                )
            else:
                lemma_string = token._.holmes.lemma
            subwords_strings = ";".join(
                str(subword) for subword in token._.holmes.subwords
            )
            subwords_strings = "".join(("[", subwords_strings, "]"))
            negation_string = "negative" if token._.holmes.is_negated else "positive"
            uncertainty_string = (
                "uncertain" if token._.holmes.is_uncertain else "certain"
            )
            matchability_string = (
                "matchable" if token._.holmes.is_matchable else "unmatchable"
            )
            if token._.holmes.is_involved_in_coreference():
                coreference_string = "; ".join(
                    str(mention) for mention in token._.holmes.mentions
                )
            else:
                coreference_string = ""
            print(
                token.i,
                token.text,
                lemma_string,
                subwords_strings,
                token.pos_,
                token.tag_,
                token.dep_,
                token.ent_type_,
                token.head.i,
                token._.holmes.string_representation_of_children(),
                token._.holmes.righthand_siblings,
                negation_string,
                uncertainty_string,
                matchability_string,
                coreference_string,
            )

    def set_coreference_information(self, token: Token) -> None:
        token._.holmes.token_and_coreference_chain_indexes = [token.i]
        token._.holmes.most_specific_coreferring_term_index = None
        for chain in token._.coref_chains:
            this_token_mention_index = -1
            for mention_index, mention in enumerate(chain):
                if token.i in mention.token_indexes:
                    this_token_mention_index = mention_index
                    break
            if this_token_mention_index > -1:
                for mention_index, mention in enumerate(chain):
                    if (
                        this_token_mention_index - mention_index
                        > self.maximum_mentions_in_coreference_chain
                        or abs(mention.root_index - token.i)
                        > self.maximum_word_distance_in_coreference_chain
                    ):
                        continue
                    if (
                        mention_index - this_token_mention_index
                        > self.maximum_mentions_in_coreference_chain
                    ):
                        break
                    token._.holmes.mentions.append(
                        Mention(
                            mention.root_index,
                            [token.i]
                            if token.i in mention.token_indexes
                            else mention.token_indexes,
                        )
                    )
            if (
                len(chain[0]) == 1
            ):  # chains with coordinated mentions are not relevant to
                # most specific mentions
                token._.holmes.most_specific_coreferring_term_index = chain[
                    chain.most_specific_mention_index
                ][0]
        working_set: Set[int] = set()
        for mention in (m for m in token._.holmes.mentions if token.i not in m.indexes):
            working_set.update(mention.indexes)
        token._.holmes.token_and_coreference_chain_indexes.extend(sorted(working_set))

    def model_supports_embeddings(self) -> bool:
        return self.vectors_nlp.meta["vectors"]["vectors"] > 0

    def is_interrogative_pronoun(self, token: Token) -> bool:
        return token.tag_ in self.interrogative_pronoun_tags

    def potential_derived_holmes_lemma(self, lemma: str) -> str:
        return_words = []
        for word in lemma.split():
            derived_word = self.derived_holmes_lemma(None, word)
            return_words.append(derived_word if derived_word is not None else word)
        return " ".join(return_words)

    def derived_holmes_lemma(self, token: Token, lemma: str) -> str:
        if lemma in self.derivational_dictionary:
            derived_lemma = self.derivational_dictionary[lemma]
            return derived_lemma
        else:
            return self.language_specific_derived_holmes_lemma(token, lemma)

    def initialize_semantic_dependencies(self, token: Token) -> None:
        for child in (
            child
            for child in token.children
            if child.dep_ != "punct"
            and child.tag_ not in self.semantic_dependency_excluded_tags
        ):
            token._.holmes.children.append(
                SemanticDependency(token.i, child.i, child.dep_)
            )

    def set_initial_question_words(self, doc: Doc) -> None:
        """is_initial_question_word -- True on a token that represents an interrogative pronoun
            within an initial phrase.
        has_initial_question_word_in_phrase -- True on a token within an initial phrase that
            governs an interrogative pronoun.
        """
        initial_sentence = next(doc.sents, None)
        if initial_sentence is not None:
            visited = set()
            working_first_phrase_head = doc[0]
            while (
                working_first_phrase_head.head is not None
                and not working_first_phrase_head.head.pos_ in ("VERB", "AUX")
                and not working_first_phrase_head.head in visited
            ):
                visited.add(working_first_phrase_head)
                working_first_phrase_head = working_first_phrase_head.head
            for token in initial_sentence:
                if (
                    self.is_interrogative_pronoun(token)
                    and token in working_first_phrase_head.subtree
                ):
                    token._.holmes.is_initial_question_word = True
            for token in initial_sentence:
                if (
                    token.pos_ in self.noun_pos
                    and len(
                        [
                            1
                            for c in token._.holmes.children
                            if self.is_interrogative_pronoun(c.child_token(token.doc))
                            and c.child_token(token.doc)._.holmes.lemma
                            != self.whose_lemma
                        ]
                    )
                    > 0
                ):
                    token._.holmes.has_initial_question_word_in_phrase = True

    def mark_if_righthand_sibling(self, token: Token) -> None:
        if token.dep_ in self.sibling_marker_deps:  # i.e. is righthand sibling
            working_token = token
            working_or_conjunction_flag = False
            # work up through the tree until the lefthandmost sibling element with the
            # semantic relationships to the rest of the sentence is reached
            while working_token.dep_ in self.conjunction_deps:
                working_token = working_token.head
                for working_child in working_token.children:
                    if working_child.lemma_ == self.or_lemma:
                        working_or_conjunction_flag = True
            # add this element to the lefthandmost sibling as a righthand sibling
            working_token._.holmes.righthand_siblings.append(token.i)
            if working_or_conjunction_flag:
                working_token._.holmes.is_involved_in_or_conjunction = True

    def copy_any_sibling_info(self, token: Token) -> None:
        # Copy the or conjunction flag to righthand siblings
        if token._.holmes.is_involved_in_or_conjunction:
            for righthand_sibling in token._.holmes.righthand_siblings:
                token.doc[
                    righthand_sibling
                ]._.holmes.is_involved_in_or_conjunction = True
        for dependency in (
            dependency
            for dependency in token._.holmes.children
            if dependency.child_index >= 0
        ):
            # where a token has a dependent token and the dependent token has righthand siblings,
            # add dependencies from the parent token to the siblings
            for child_righthand_sibling in token.doc[
                dependency.child_index
            ]._.holmes.righthand_siblings:
                # Check this token does not already have the dependency
                if (
                    len(
                        [
                            dependency
                            for dependency in token._.holmes.children
                            if dependency.child_index == child_righthand_sibling
                        ]
                    )
                    == 0
                ):
                    child_index_to_add = child_righthand_sibling
                    # If this token is a grammatical element, it needs to point to new
                    # child dependencies as a grammatical element as well
                    if dependency.child_index < 0:
                        child_index_to_add = 0 - (child_index_to_add + 1)
                    # Check adding the new dependency will not result in a loop and that
                    # this token still does not have the dependency now its index has
                    # possibly been changed
                    if (
                        token.i != child_index_to_add
                        and not token._.holmes.has_dependency_with_child_index(
                            child_index_to_add
                        )
                    ):
                        token._.holmes.children.append(
                            SemanticDependency(
                                token.i,
                                child_index_to_add,
                                dependency.label,
                                dependency.is_uncertain,
                            )
                        )
            # where a token has a dependent token and the parent token has righthand siblings,
            # add dependencies from the siblings to the dependent token, unless the dependent
            # token is to the right of the parent token but to the left of the sibling.
            for righthand_sibling in (
                righthand_sibling
                for righthand_sibling in token._.holmes.righthand_siblings
                if righthand_sibling != dependency.child_index
                and (
                    righthand_sibling < dependency.child_index
                    or dependency.child_index < token.i
                )
            ):
                # unless the sibling already contains a dependency with the same label
                # or the sibling has this token as a dependent child
                righthand_sibling_token = token.doc[righthand_sibling]
                if (
                    len(
                        [
                            sibling_dependency
                            for sibling_dependency in righthand_sibling_token._.holmes.children
                            if sibling_dependency.label == dependency.label
                            and not token._.holmes.has_dependency_with_child_index(
                                sibling_dependency.child_index
                            )
                        ]
                    )
                    == 0
                    and dependency.label not in self.conjunction_deps
                    and not righthand_sibling_token._.holmes.has_dependency_with_child_index(
                        dependency.child_index
                    )
                    and righthand_sibling != dependency.child_index
                ):
                    righthand_sibling_token._.holmes.children.append(
                        SemanticDependency(
                            righthand_sibling,
                            dependency.child_index,
                            dependency.label,
                            self.mark_child_dependencies_copied_to_siblings_as_uncertain
                            or dependency.is_uncertain,
                        )
                    )

    def normalize_predicative_adjectives(self, token: Token) -> None:
        """Change phrases like *the town is old* and *the man is poor* so their
        semantic structure is equivalent to *the old town* and *the poor man*.
        """
        if token.pos_ in self.adjectival_predicate_head_pos:
            altered = False
            for predicative_adjective_index in (
                dependency.child_index
                for dependency in token._.holmes.children
                if dependency.label == self.adjectival_predicate_predicate_dep
                and token.doc[dependency.child_index].pos_
                == self.adjectival_predicate_predicate_pos
                and dependency.child_index >= 0
            ):
                for subject_index in (
                    dependency.child_index
                    for dependency in token._.holmes.children
                    if dependency.label == self.adjectival_predicate_subject_dep
                    and (
                        dependency.child_token(token.doc).pos_
                        in self.adjectival_predicate_subject_pos
                        or dependency.child_token(
                            token.doc
                        )._.holmes.is_involved_in_coreference()
                        and dependency.child_index >= 0
                        and dependency.child_index != predicative_adjective_index
                    )
                ):
                    token.doc[subject_index]._.holmes.children.append(
                        SemanticDependency(
                            subject_index,
                            predicative_adjective_index,
                            self.modifier_dep,
                        )
                    )
                    altered = True
            if altered:
                token._.holmes.children = [
                    SemanticDependency(token.i, 0 - (subject_index + 1), None)
                ]

    def create_additional_preposition_phrase_semantic_dependencies(
        self, token: Token
    ) -> None:
        """In structures like 'Somebody needs insurance for a period' it seems to be
        mainly language-dependent whether the preposition phrase is analysed as being
        dependent on the preceding noun or the preceding verb. We add an additional, new
        dependency to whichever of the noun or the verb does not already have one. In English,
        the new label is defined in *match_implication_dict* in such a way that original
        dependencies in search phrases match new dependencies in documents but not vice versa.
        This restriction is not applied in German because the fact the verb can be placed in
        different positions within the sentence means there is considerable variation around
        how prepositional phrases are analyzed by spaCy.
        """

        def add_dependencies_pointing_to_preposition_and_siblings(parent, label):
            for working_preposition in token._.holmes.loop_token_and_righthand_siblings(
                token.doc
            ):
                if parent.i != working_preposition.i:
                    parent._.holmes.children.append(
                        SemanticDependency(parent.i, working_preposition.i, label, True)
                    )

        # token is a preposition ...
        if token.pos_ == "ADP":
            # directly preceded by a noun
            if (
                token.i > 0
                and token.doc[token.i - 1].sent == token.sent
                and token.doc[token.i - 1].pos_ in ("NOUN", "PROPN", "PRON")
            ):
                preceding_noun = token.doc[token.i - 1]
                # and the noun is governed by at least one verb
                governing_verbs = [
                    working_token
                    for working_token in token.sent
                    if working_token.pos_ == "VERB"
                    and working_token._.holmes.has_dependency_with_child_index(
                        preceding_noun.i
                    )
                ]
                if len(governing_verbs) == 0:
                    return
                # if the noun governs the preposition, add new possible dependencies
                # from the verb(s)
                for governing_verb in governing_verbs:
                    if preceding_noun._.holmes.has_dependency_with_child_index_and_label(
                        token.i, self.spacy_noun_to_preposition_dep
                    ) and not governing_verb._.holmes.has_dependency_with_child_index_and_label(
                        token.i, self.spacy_verb_to_preposition_dep
                    ):
                        add_dependencies_pointing_to_preposition_and_siblings(
                            governing_verb, self.holmes_verb_to_preposition_dep
                        )
                # if the verb(s) governs the preposition, add new possible dependencies
                # from the noun
                if governing_verbs[
                    0
                ]._.holmes.has_dependency_with_child_index_and_label(
                    token.i, self.spacy_verb_to_preposition_dep
                ) and not preceding_noun._.holmes.has_dependency_with_child_index_and_label(
                    token.i, self.spacy_noun_to_preposition_dep
                ):
                    # check the preposition is not pointing back to a relative clause
                    for preposition_dep_index in (
                        dep.child_index
                        for dep in token._.holmes.children
                        if dep.child_index >= 0
                    ):
                        if token.doc[
                            preposition_dep_index
                        ]._.holmes.has_dependency_with_label("relcl"):
                            return
                    add_dependencies_pointing_to_preposition_and_siblings(
                        preceding_noun, self.holmes_noun_to_preposition_dep
                    )

    def set_matchability(self, token: Token) -> None:
        """Marks whether this token, if it appears in a search phrase, should require a counterpart
        in a document being matched.
        """
        token._.holmes.is_matchable = (
            (
                token.pos_ in self.matchable_pos
                or token._.holmes.is_involved_in_coreference()
                or len(token._.holmes.subwords) > 0
            )
            and not self.is_interrogative_pronoun(token)
            and token._.holmes.lemma not in self.generic_pronoun_lemmas
        )

    def move_information_between_tokens(
        self, from_token: Token, to_token: Token
    ) -> None:
        """Moves semantic child and sibling information from one token to another.

        Args:

        from_token -- the source token, which will be marked as a grammatical token
        pointing to *to_token*.
        to_token -- the destination token.
        """
        linking_dependencies = [
            dependency
            for dependency in from_token._.holmes.children
            if dependency.child_index == to_token.i
        ]
        if len(linking_dependencies) == 0:
            return  # should only happen if there is a problem with the spaCy structure
        # only loop dependencies whose label or index are not already present at the destination
        for dependency in (
            dependency
            for dependency in from_token._.holmes.children
            if not to_token._.holmes.has_dependency_with_child_index(
                dependency.child_index
            )
            and to_token.i != dependency.child_index
            and to_token.i
            not in to_token.doc[dependency.child_index]._.holmes.righthand_siblings
            and dependency.child_index not in to_token._.holmes.righthand_siblings
        ):
            to_token._.holmes.children.append(
                SemanticDependency(
                    to_token.i,
                    dependency.child_index,
                    dependency.label,
                    dependency.is_uncertain,
                )
            )
        from_token._.holmes.children = [
            SemanticDependency(from_token.i, 0 - (to_token.i + 1))
        ]
        to_token._.holmes.righthand_siblings.extend(
            from_token._.holmes.righthand_siblings
        )
        from_token._.holmes.righthand_siblings = []
        if from_token._.holmes.is_involved_in_or_conjunction:
            to_token._.holmes.is_involved_in_or_conjunction = True
        if from_token._.holmes.is_negated:
            to_token._.holmes.is_negated = True
        # If from_token is the righthand sibling of some other token within the same sentence,
        # replace that token's reference with a reference to to_token
        for token in from_token.sent:
            if from_token.i in token._.holmes.righthand_siblings:
                token._.holmes.righthand_siblings.remove(from_token.i)
                if token.i != to_token.i:
                    token._.holmes.righthand_siblings.append(to_token.i)

    def create_convenience_dependencies(self, token: Token) -> None:
        for child_dependency in (
            child_dependency
            for child_dependency in token._.holmes.children
            if child_dependency.child_index >= 0
        ):
            child_token = child_dependency.child_token(token.doc)
            child_token._.holmes.parents.append(child_dependency)
        for linked_parent_index in token._.holmes.token_and_coreference_chain_indexes:
            linked_parent = token.doc[linked_parent_index]
            for child_dependency in (
                child_dependency
                for child_dependency in linked_parent._.holmes.children
                if child_dependency.child_index >= 0
            ):
                child_token = child_dependency.child_token(token.doc)
                for (
                    linked_child_index
                ) in child_token._.holmes.token_and_coreference_chain_indexes:
                    linked_child = token.doc[linked_child_index]
                    token._.holmes.coreference_linked_child_dependencies.append(
                        [linked_child.i, child_dependency.label]
                    )
                    linked_child._.holmes.coreference_linked_parent_dependencies.append(
                        [token.i, child_dependency.label]
                    )

    def multiword_spans_with_head_token(
        self, token: Token
    ) -> Optional[List[MultiwordSpan]]:
        """Returns a list of *MultiwordSpan* objects with *token* at their head if *token* is a noun,
        otherwise a *None*.
        """

        if not token.pos_ in self.noun_pos:
            return None
        return_list: List[MultiwordSpan] = []
        pointer = token.left_edge.i
        while pointer <= token.right_edge.i:
            working_text = ""
            working_hyphen_normalized_lemma = ""
            working_lemma = ""
            working_derived_lemma = ""
            working_tokens = []
            inner_pointer = pointer
            while inner_pointer <= token.right_edge.i and (
                token.doc[inner_pointer]._.holmes.is_matchable
                or token.doc[inner_pointer].text == "-"
            ):
                if token.doc[inner_pointer].text != "-":
                    working_text = " ".join(
                        (working_text, token.doc[inner_pointer].text)
                    )
                    working_hyphen_normalized_lemma = " ".join(
                        (
                            working_hyphen_normalized_lemma,
                            token.doc[inner_pointer]._.holmes.hyphen_normalized_lemma,
                        )
                    )
                    working_lemma = " ".join(
                        (working_lemma, token.doc[inner_pointer]._.holmes.lemma)
                    )
                    this_token_derived_lemma = token.doc[
                        inner_pointer
                    ]._.holmes.derived_lemma
                    working_derived_lemma = " ".join(
                        (working_derived_lemma, this_token_derived_lemma)
                    )
                    working_tokens.append(token.doc[inner_pointer])
                inner_pointer += 1
            if pointer + 1 < inner_pointer and token in working_tokens:
                return_list.append(
                    MultiwordSpan(
                        working_text.strip().lower(),
                        working_hyphen_normalized_lemma.strip(),
                        working_lemma.strip(),
                        working_derived_lemma.strip(),
                        [t.i for t in working_tokens],
                    )
                )
            pointer += 1
        return return_list if len(return_list) > 0 else None

    def get_entity_label_to_vector_dict(self) -> Dict[str, List[Floats1d]]:
        return {
            label: self.vectors_nlp.vocab[
                self.entity_labels_to_corresponding_lexemes[label]
            ].vector
            for label in self.entity_labels_to_corresponding_lexemes
        }

    @abstractmethod
    def normalize_hyphens(self, word: str) -> str:
        pass

    def update_ontology(self, ontology: Ontology) -> None:
        """Update the ontology with derived lemmas retrieved from the spaCy model in use."""
        for entries in ontology.match_dict.values():
            for entry in entries:
                assert len(entry.reprs) == 1
                derived_repr = self.potential_derived_holmes_lemma(entry.reprs[0])
                if derived_repr != entry.reprs[0]:
                    entry.reprs.append(derived_repr)
        for key in ontology.match_dict.copy():
            derived_key = self.potential_derived_holmes_lemma(key)
            if derived_key not in ontology.match_dict:
                ontology.match_dict[derived_key] = ontology.match_dict[key]
        ontology.refresh_words()

    def get_ontology_reverse_derivational_dict(
        self, ontology: Ontology
    ) -> Dict[str, List[str]]:
        """During structural matching, a lemma or derived lemma matches any words in the ontology
        that yield the same word as their derived lemmas. This method generates a dictionary
        from derived lemmas to ontology words that yield them to facilitate such matching.
        """
        ontology_reverse_derivational_dict: Dict[str, List[str]] = {}
        for ontology_word in ontology.words:
            derived_lemmas = []
            normalized_ontology_word = self.normalize_hyphens(ontology_word)
            for textual_word in normalized_ontology_word.split():
                derived_lemma = self.derived_holmes_lemma(None, textual_word.lower())
                if derived_lemma is None:
                    derived_lemma = textual_word
                derived_lemmas.append(derived_lemma)
            derived_ontology_word = " ".join(derived_lemmas)
            if derived_ontology_word != ontology_word:
                if derived_ontology_word in ontology_reverse_derivational_dict:
                    ontology_reverse_derivational_dict[derived_ontology_word].append(
                        ontology_word
                    )
                else:
                    ontology_reverse_derivational_dict[derived_ontology_word] = [
                        ontology_word
                    ]
        # sort entry lists to ensure deterministic behaviour
        for derived_ontology_word in ontology_reverse_derivational_dict:
            ontology_reverse_derivational_dict[derived_ontology_word] = sorted(
                ontology_reverse_derivational_dict[derived_ontology_word]
            )
        return ontology_reverse_derivational_dict

    def get_model_name(self) -> str:
        return (
            self.nlp.meta["lang"]
            + "_"
            + self.nlp.meta["name"]
            + " v"
            + self.nlp.meta["version"]
        )


class LinguisticObjectFactory:
    """Factory for search phrases and topic matching phraselets."""

    def __init__(
        self,
        semantic_analyzer,
        semantic_matching_helper,
        overall_similarity_threshold,
        embedding_based_matching_on_root_words,
        analyze_derivational_morphology,
        perform_coreference_resolution,
        ontology,
        ontology_reverse_derivational_dict,
    ):
        """Args:

        semantic_analyzer -- the *SemanticAnalyzer* object to use
        semantic_matching_helper -- the *SemanticMatchingHelper* object to use
        overall_similarity_threshold -- if embedding-based matching is to be activated, a float
            value between 0 and 1. A match between a search phrase and a document is then valid
            if the geometric mean of all the similarities between search phrase tokens and
            document tokens is this value or greater. If this value is set to 1.0,
            embedding-based matching is deactivated.
        embedding_based_matching_on_root_words -- determines whether or not embedding-based
            matching should be attempted on search-phrase root tokens, which has a considerable
            performance hit. Defaults to *False*.
        analyze_derivational_morphology -- *True* if matching should be attempted between different
            words from the same word family. Defaults to *True*.
        perform_coreference_resolution -- *True* if coreference resolution should be performed.
        ontology -- an *Ontology* object, or *None* if no ontology is in use.
        ontology_reverse_derivational_dict -- a dictionary from derived lemmas to ontology words
            that yield them.
        """
        self.semantic_analyzer = semantic_analyzer
        self.semantic_matching_helper = semantic_matching_helper
        self.overall_similarity_threshold = overall_similarity_threshold
        self.embedding_based_matching_on_root_words = (
            embedding_based_matching_on_root_words
        )
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.perform_coreference_resolution = perform_coreference_resolution
        self.ontology = ontology
        self.ontology_reverse_derivational_dict = ontology_reverse_derivational_dict

    def add_phraselets_to_dict(
        self,
        doc,
        *,
        phraselet_labels_to_phraselet_infos,
        replace_with_hypernym_ancestors,
        match_all_words,
        ignore_relation_phraselets,
        include_reverse_only,
        stop_lemmas,
        stop_tags,
        reverse_only_parent_lemmas,
        words_to_corpus_frequencies,
        maximum_corpus_frequency,
        process_initial_question_words
    ):
        """Creates topic matching phraselets extracted from a text to match against.

        Properties:

        doc -- the Holmes-parsed document
        phraselet_labels_to_phraselet_infos -- a dictionary from labels to phraselet info objects
            that are used to generate phraselet search phrases.
        match_all_words -- if 'True', word phraselets are generated for all matchable words
            rather than just for words whose tags match the phraselet template; multiwords
            are not taken into account when processing single-word phraselets; and single-word
            phraselets are generated for subwords.
        ignore_relation_phraselets -- if 'True', only single-word phraselets are processed.
        include_reverse_only -- whether to generate phraselets that are only reverse-matched.
            Reverse matching is used in topic matching but not in supervised document
            classification.
        stop_lemmas -- lemmas that should prevent all types of phraselet production.
        stop_tags -- tags that should prevent all types of phraselet production.
        reverse_only_parent_lemmas -- lemma / part-of-speech combinations that, when present at
            the parent pole of a relation phraselet, should cause that phraselet to be
            reverse-matched.
        words_to_corpus_frequencies -- a dictionary from words to the number of times each
            word occurs in the indexed documents.
        maximum_corpus_frequency -- the maximum value within *words_to_corpus_frequencies*.
        process_initial_question_words -- *True* if interrogative pronouns are permitted within
            phraselets.
        """
        index_to_lemmas_cache = {}

        def get_lemmas_from_index(index: Index) -> Tuple[str, str]:
            """Returns the strings to use as the lemma and derived lemma for an index within the text."""
            if index in index_to_lemmas_cache:
                return index_to_lemmas_cache[index]
            token = doc[index.token_index]
            if self.semantic_matching_helper.get_entity_placeholder(token) is not None:
                # False in order to get text rather than lemma
                index_to_lemmas_cache[index] = token.text, token.text
                return token.text, token.text
                # keep the text, because the lemma will be lowercase
            if index.is_subword():
                lemma = token._.holmes.subwords[index.subword_index].lemma
                if self.analyze_derivational_morphology:
                    derived_lemma = token._.holmes.subwords[
                        index.subword_index
                    ].derived_lemma
                else:
                    derived_lemma = lemma
                if (
                    self.ontology is not None
                    and not self.ontology.contains_word(lemma)
                    and self.ontology.contains_word(
                        token._.holmes.subwords[index.subword_index].text.lower()
                    )
                ):
                    lemma = derived_lemma = token._.holmes.subwords[
                        index.subword_index
                    ].text.lower()
            else:
                lemma = token._.holmes.lemma
                if self.analyze_derivational_morphology:
                    derived_lemma = token._.holmes.derived_lemma
                else:
                    derived_lemma = lemma
                if (
                    self.ontology is not None
                    and not self.ontology.contains_word(lemma)
                    and self.ontology.contains_word(token.text.lower())
                ):
                    lemma = derived_lemma = token.text.lower()
            if (
                self.ontology is not None
                and self.analyze_derivational_morphology
                and derived_lemma in self.ontology_reverse_derivational_dict
            ):
                derived_lemma = self.ontology_reverse_derivational_dict[derived_lemma][
                    0
                ]
            index_to_lemmas_cache[index] = lemma, derived_lemma
            return lemma, derived_lemma

        def replace_lemmas_with_most_general_ancestor(
            lemma: str, derived_lemma: str
        ) -> Tuple[str, str]:
            new_derived_lemma = self.ontology.get_most_general_hypernym_ancestor(
                derived_lemma
            ).lower()
            if derived_lemma != new_derived_lemma:
                lemma = derived_lemma = new_derived_lemma
            return lemma, derived_lemma

        def lemma_replacement_indicated(
            existing_lemma: str,
            existing_pos: str,
            new_lemma: Optional[str],
            new_pos: Optional[str],
        ) -> bool:
            # The aim is that the same phraselet should be produced from different
            # grammatical constructions sharing the same underlying semantics. For
            # consistency, nominal elements are preferred over verbal elements, and
            # it can be necessary to change lemmas within an existing phraselet if its
            # meaning is encounteted within the search text with a different POS. Note
            # that the derived lemmas (which determine the label) are not affected.
            if existing_lemma is None:
                return False
            if (
                not existing_pos
                in self.semantic_matching_helper.preferred_phraselet_pos
                and new_pos in self.semantic_matching_helper.preferred_phraselet_pos
            ):
                return True
            if (
                existing_pos in self.semantic_matching_helper.preferred_phraselet_pos
                and new_pos not in self.semantic_matching_helper.preferred_phraselet_pos
            ):
                return False
            return len(cast(str, new_lemma)) < len(existing_lemma)

        def add_new_phraselet_info(
            phraselet_template: PhraseletTemplate,
            created_without_matching_tags: bool,
            is_reverse_only_parent_lemma: Optional[bool],
            parent_lemma: str,
            parent_derived_lemma: str,
            parent_pos: str,
            parent_ent_type: str,
            parent_is_initial_question_word: bool,
            parent_has_initial_question_word_in_phrase: bool,
            child_lemma: Optional[str],
            child_derived_lemma: Optional[str],
            child_pos: Optional[str],
            child_ent_type: Optional[str],
            child_is_initial_question_word: Optional[bool],
            child_has_initial_question_word_in_phrase: Optional[bool],
        ) -> None:
            def get_frequency_factor_for_pole(
                parent: bool,
            ) -> float:  # pole is 'True' -> parent, 'False' -> child
                original_word_set = (
                    {parent_lemma, parent_derived_lemma}
                    if parent
                    else {cast(str, child_lemma), cast(str, child_derived_lemma)}
                )
                word_set = original_word_set.copy()
                if self.ontology is not None:
                    for word in original_word_set:
                        for entry in self.ontology.get_matching_entries(word):
                            word_set.update(entry.reprs)
                frequencies = []
                for word in word_set:
                    if word in words_to_corpus_frequencies:
                        frequencies.append(float(words_to_corpus_frequencies[word]))
                if len(frequencies) == 0:
                    return 1.0
                adjusted_max_frequency = max(frequencies) - 1.0
                if adjusted_max_frequency <= 0.0:
                    return 1.0
                return 1 - (
                    math.log(adjusted_max_frequency)
                    / math.log(maximum_corpus_frequency)
                )

            frequency_factor = parent_frequency_factor = child_frequency_factor = None
            if words_to_corpus_frequencies is not None:
                parent_frequency_factor = get_frequency_factor_for_pole(True)
                frequency_factor = parent_frequency_factor
                if child_lemma is not None:
                    child_frequency_factor = get_frequency_factor_for_pole(False)
                    frequency_factor *= child_frequency_factor
            parent_hyphen_normalized_lemma = self.semantic_analyzer.normalize_hyphens(
                parent_lemma
            )
            if child_lemma is not None:
                child_hyphen_normalized_lemma = (
                    self.semantic_analyzer.normalize_hyphens(child_lemma)
                )
            else:
                child_hyphen_normalized_lemma = None
            phraselet_label = "".join(
                (
                    phraselet_template.label,
                    ": ",
                    parent_derived_lemma,
                    "-" if child_derived_lemma is not None else "",
                    child_derived_lemma if child_derived_lemma is not None else "",
                )
            )
            if phraselet_label not in phraselet_labels_to_phraselet_infos:
                phraselet_labels_to_phraselet_infos[phraselet_label] = PhraseletInfo(
                    phraselet_label,
                    phraselet_template.label,
                    parent_lemma,
                    parent_hyphen_normalized_lemma,
                    parent_derived_lemma,
                    parent_pos,
                    parent_ent_type,
                    parent_is_initial_question_word,
                    parent_has_initial_question_word_in_phrase,
                    child_lemma,
                    child_hyphen_normalized_lemma,
                    child_derived_lemma,
                    child_pos,
                    child_ent_type,
                    child_is_initial_question_word,
                    child_has_initial_question_word_in_phrase,
                    created_without_matching_tags,
                    is_reverse_only_parent_lemma,
                    frequency_factor,
                    parent_frequency_factor,
                    child_frequency_factor,
                )
            else:
                existing_phraselet = phraselet_labels_to_phraselet_infos[
                    phraselet_label
                ]
                if lemma_replacement_indicated(
                    existing_phraselet.parent_lemma,
                    existing_phraselet.parent_pos,
                    parent_lemma,
                    parent_pos,
                ):
                    existing_phraselet.parent_lemma = parent_lemma
                    existing_phraselet.parent_hyphen_normalized_lemma = (
                        parent_hyphen_normalized_lemma
                    )
                    existing_phraselet.parent_pos = parent_pos
                    existing_phraselet.set_parent_reprs()
                if lemma_replacement_indicated(
                    existing_phraselet.child_lemma,
                    existing_phraselet.child_pos,
                    child_lemma,
                    child_pos,
                ):
                    existing_phraselet.child_lemma = child_lemma
                    existing_phraselet.child_hyphen_normalized_lemma = (
                        child_hyphen_normalized_lemma
                    )
                    existing_phraselet.child_pos = child_pos
                    existing_phraselet.set_child_reprs()

        def process_single_word_phraselet_templates(
            token: Token,
            subword_index: int,
            checking_tags: bool,
            token_indexes_to_multiwords: Dict[int, str],
        ):
            for phraselet_template in (
                phraselet_template
                for phraselet_template in self.semantic_matching_helper.local_phraselet_templates
                if phraselet_template.single_word()
                and (token._.holmes.is_matchable or subword_index is not None)
            ):
                if (
                    not checking_tags or token.tag_ in phraselet_template.parent_tags
                ) and token.tag_ not in stop_tags:
                    if token.i in token_indexes_to_multiwords and not match_all_words:
                        lemma = derived_lemma = token_indexes_to_multiwords[token.i]
                    else:
                        lemma, derived_lemma = get_lemmas_from_index(
                            Index(token.i, subword_index)
                        )
                    if self.ontology is not None and replace_with_hypernym_ancestors:
                        (
                            lemma,
                            derived_lemma,
                        ) = replace_lemmas_with_most_general_ancestor(
                            lemma, derived_lemma
                        )
                    if (
                        derived_lemma not in stop_lemmas
                        and derived_lemma != "ENTITYNOUN"
                    ):
                        # ENTITYNOUN has to be excluded as single word although it is still
                        # permitted within relation phraselet templates
                        add_new_phraselet_info(
                            phraselet_template,
                            not checking_tags,
                            None,
                            lemma,
                            derived_lemma,
                            token.pos_,
                            token.ent_type_,
                            token._.holmes.is_initial_question_word,
                            token._.holmes.has_initial_question_word_in_phrase,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )

        def add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(
            index_list: List[Index],
        ) -> None:
            # for each token in the list, find out whether it has subwords and if so add the
            # head subword to the list
            for index in index_list.copy():
                token = doc[index.token_index]
                for subword in (
                    subword
                    for subword in token._.holmes.subwords
                    if subword.is_head and subword.containing_token_index == token.i
                ):
                    index_list.append(Index(token.i, subword.index))
                # if one or more subwords do not belong to this token, it is a hyphenated word
                # within conjunction and the whole word should not be used to build relation phraselets.
                if (
                    len(
                        [
                            subword
                            for subword in token._.holmes.subwords
                            if subword.containing_token_index != token.i
                        ]
                    )
                    > 0
                ):
                    index_list.remove(index)

        token_indexes_to_multiwords = {}
        token_indexes_within_multiwords_to_ignore = []
        for token in (token for token in doc if len(token._.holmes.lemma.split()) == 1):
            odm = None
            if self.ontology is not None:
                odm = self.semantic_matching_helper.get_ontology_defined_multiword(
                    token, self.ontology
                )
                if odm is not None:
                    for index in odm.token_indexes:
                        if index == token.i:
                            multiword_to_use = odm.text.lower()
                            if (
                                multiword_to_use
                                in self.ontology_reverse_derivational_dict
                            ):
                                multiword_to_use = (
                                    self.ontology_reverse_derivational_dict[
                                        multiword_to_use
                                    ][0]
                                )
                            token_indexes_to_multiwords[index] = multiword_to_use
                        else:
                            token_indexes_within_multiwords_to_ignore.append(index)
            if odm is None:
                edm = self.semantic_matching_helper.get_entity_defined_multiword(token)
                if edm is not None:
                    for index in edm.token_indexes:
                        if index == token.i:
                            token_indexes_to_multiwords[index] = edm.text.lower()
                        else:
                            token_indexes_within_multiwords_to_ignore.append(index)
        for token in doc:
            if token.i in token_indexes_within_multiwords_to_ignore:
                if match_all_words:
                    process_single_word_phraselet_templates(
                        token, None, False, token_indexes_to_multiwords
                    )
                continue
            if (
                len(
                    [
                        subword
                        for subword in token._.holmes.subwords
                        if subword.containing_token_index != token.i
                    ]
                )
                == 0
            ):
                # whole single words involved in subword conjunction should not be included as
                # these are partial words including hyphens.
                process_single_word_phraselet_templates(
                    token, None, not match_all_words, token_indexes_to_multiwords
                )
            if match_all_words:
                for subword in (
                    subword
                    for subword in token._.holmes.subwords
                    if token.i == subword.containing_token_index
                ):
                    process_single_word_phraselet_templates(
                        token, subword.index, False, token_indexes_to_multiwords
                    )
            if ignore_relation_phraselets:
                continue
            if self.perform_coreference_resolution:
                parents = [
                    Index(token_index, None)
                    for token_index in token._.holmes.token_and_coreference_chain_indexes
                ]
            else:
                parents = [Index(token.i, None)]
            add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(
                parents
            )
            for parent in parents:
                for dependency in (
                    dependency
                    for dependency in doc[parent.token_index]._.holmes.children
                    if dependency.child_index
                    not in token_indexes_within_multiwords_to_ignore
                ):
                    if self.perform_coreference_resolution:
                        children = [
                            Index(token_index, None)
                            for token_index in dependency.child_token(
                                doc
                            )._.holmes.token_and_coreference_chain_indexes
                        ]
                    else:
                        children = [Index(dependency.child_token(doc).i, None)]
                    add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(
                        children
                    )
                    for child in children:
                        for phraselet_template in (
                            phraselet_template
                            for phraselet_template in self.semantic_matching_helper.local_phraselet_templates
                            if not phraselet_template.single_word()
                            and (
                                not phraselet_template.reverse_only
                                or include_reverse_only
                            )
                        ):
                            if (
                                dependency.label in phraselet_template.dependency_labels
                                and doc[parent.token_index].tag_
                                in phraselet_template.parent_tags
                                and doc[child.token_index].tag_
                                in phraselet_template.child_tags
                                and doc[parent.token_index]._.holmes.is_matchable
                                and (
                                    doc[child.token_index]._.holmes.is_matchable
                                    or (
                                        process_initial_question_words
                                        and doc[
                                            child.token_index
                                        ]._.holmes.is_initial_question_word
                                    )
                                )
                            ):
                                if parent.token_index in token_indexes_to_multiwords:
                                    parent_lemma = (
                                        parent_derived_lemma
                                    ) = token_indexes_to_multiwords[parent.token_index]
                                else:
                                    (
                                        parent_lemma,
                                        parent_derived_lemma,
                                    ) = get_lemmas_from_index(parent)
                                if (
                                    self.ontology is not None
                                    and replace_with_hypernym_ancestors
                                ):
                                    (
                                        parent_lemma,
                                        parent_derived_lemma,
                                    ) = replace_lemmas_with_most_general_ancestor(
                                        parent_lemma, parent_derived_lemma
                                    )
                                if child.token_index in token_indexes_to_multiwords:
                                    child_lemma = (
                                        child_derived_lemma
                                    ) = token_indexes_to_multiwords[child.token_index]
                                else:
                                    (
                                        child_lemma,
                                        child_derived_lemma,
                                    ) = get_lemmas_from_index(child)
                                if (
                                    self.ontology is not None
                                    and replace_with_hypernym_ancestors
                                ):
                                    (
                                        child_lemma,
                                        child_derived_lemma,
                                    ) = replace_lemmas_with_most_general_ancestor(
                                        child_lemma, child_derived_lemma
                                    )
                                is_reverse_only_parent_lemma = False
                                if reverse_only_parent_lemmas is not None:
                                    for entry in reverse_only_parent_lemmas:
                                        if (
                                            entry[0]
                                            == doc[parent.token_index]._.holmes.lemma
                                            and entry[1] == doc[parent.token_index].pos_
                                        ):
                                            is_reverse_only_parent_lemma = True
                                if (
                                    parent_lemma not in stop_lemmas
                                    and child_lemma not in stop_lemmas
                                    and not (
                                        is_reverse_only_parent_lemma
                                        and not include_reverse_only
                                    )
                                ):
                                    add_new_phraselet_info(
                                        phraselet_template,
                                        match_all_words,
                                        is_reverse_only_parent_lemma,
                                        parent_lemma,
                                        parent_derived_lemma,
                                        doc[parent.token_index].pos_,
                                        doc[parent.token_index].ent_type_,
                                        doc[
                                            parent.token_index
                                        ]._.holmes.is_initial_question_word,
                                        doc[
                                            parent.token_index
                                        ]._.holmes.has_initial_question_word_in_phrase,
                                        child_lemma,
                                        child_derived_lemma,
                                        doc[child.token_index].pos_,
                                        doc[child.token_index].ent_type_,
                                        doc[
                                            child.token_index
                                        ]._.holmes.is_initial_question_word,
                                        doc[
                                            child.token_index
                                        ]._.holmes.has_initial_question_word_in_phrase,
                                    )

            # We do not check for matchability in order to catch pos_='X', tag_='TRUNC'. This
            # is not a problem as only a limited range of parts of speech receive subwords in
            # the first place.
            for subword in (
                subword
                for subword in token._.holmes.subwords
                if subword.dependent_index is not None
            ):
                parent_subword_index = subword.index
                child_subword_index = subword.dependent_index
                if (
                    token._.holmes.subwords[parent_subword_index].containing_token_index
                    != token.i
                    and token._.holmes.subwords[
                        child_subword_index
                    ].containing_token_index
                    != token.i
                ):
                    continue
                for phraselet_template in (
                    phraselet_template
                    for phraselet_template in self.semantic_matching_helper.local_phraselet_templates
                    if not phraselet_template.single_word()
                    and (not phraselet_template.reverse_only or include_reverse_only)
                    and subword.dependency_label in phraselet_template.dependency_labels
                    and token.tag_ in phraselet_template.parent_tags
                ):
                    parent_lemma, parent_derived_lemma = get_lemmas_from_index(
                        Index(token.i, parent_subword_index)
                    )
                    if self.ontology is not None and replace_with_hypernym_ancestors:
                        (
                            parent_lemma,
                            parent_derived_lemma,
                        ) = replace_lemmas_with_most_general_ancestor(
                            parent_lemma, parent_derived_lemma
                        )
                    child_lemma, child_derived_lemma = get_lemmas_from_index(
                        Index(token.i, child_subword_index)
                    )
                    if self.ontology is not None and replace_with_hypernym_ancestors:
                        (
                            child_lemma,
                            child_derived_lemma,
                        ) = replace_lemmas_with_most_general_ancestor(
                            child_lemma, child_derived_lemma
                        )
                    add_new_phraselet_info(
                        phraselet_template,
                        match_all_words,
                        False,
                        parent_lemma,
                        parent_derived_lemma,
                        token.pos_,
                        token.ent_type_,
                        token._.holmes.is_initial_question_word,
                        token._.holmes.has_initial_question_word_in_phrase,
                        child_lemma,
                        child_derived_lemma,
                        token.pos_,
                        token.ent_type_,
                        token._.holmes.is_initial_question_word,
                        token._.holmes.has_initial_question_word_in_phrase,
                    )
        if len(phraselet_labels_to_phraselet_infos) == 0 and not match_all_words:
            for token in doc:
                process_single_word_phraselet_templates(
                    token, None, False, token_indexes_to_multiwords
                )

    def create_search_phrases_from_phraselet_infos(
        self,
        phraselet_infos: List[PhraseletInfo],
        reverse_matching_frequency_threshold: Optional[float] = None,
    ) -> Dict[str, SearchPhrase]:
        """Creates search phrases from phraselet info objects, returning a dictionary from
        phraselet labels to the created search phrases.

        reverse_matching_frequency_threshold: an optional threshold between 0.0 and 1.0.
            Where the parent word in a phraselet has a frequency factor below the threshold,
            the search phrase will be set to
            *treat_as_reverse_only_during_initial_relation_matching=True*.
        """

        def create_search_phrase_from_phraselet(
            phraselet_info: PhraseletInfo,
        ) -> SearchPhrase:
            for (
                phraselet_template
            ) in self.semantic_matching_helper.local_phraselet_templates:
                if phraselet_info.template_label == phraselet_template.label:
                    phraselet_doc = phraselet_template.template_doc.copy()
                    phraselet_doc[
                        phraselet_template.parent_index
                    ]._.holmes.lemma = phraselet_info.parent_lemma
                    phraselet_doc[
                        phraselet_template.parent_index
                    ]._.holmes.direct_matching_reprs = (
                        phraselet_info.parent_direct_matching_reprs
                    )
                    phraselet_doc[
                        phraselet_template.parent_index
                    ]._.holmes.derived_lemma = phraselet_info.parent_derived_lemma
                    phraselet_doc[
                        phraselet_template.parent_index
                    ]._.holmes.derivation_matching_reprs = (
                        phraselet_info.parent_derivation_matching_reprs
                    )
                    phraselet_doc[
                        phraselet_template.parent_index
                    ]._.holmes.ent_type = phraselet_info.parent_ent_type
                    phraselet_doc[
                        phraselet_template.parent_index
                    ]._.holmes.is_initial_question_word = (
                        phraselet_info.parent_is_initial_question_word
                    )
                    phraselet_doc[
                        phraselet_template.parent_index
                    ]._.holmes.has_initial_question_word_in_phrase = (
                        phraselet_info.parent_has_initial_question_word_in_phrase
                    )
                    if phraselet_info.child_lemma is not None:
                        phraselet_doc[
                            phraselet_template.child_index
                        ]._.holmes.lemma = phraselet_info.child_lemma
                        phraselet_doc[
                            phraselet_template.child_index
                        ]._.holmes.direct_matching_reprs = (
                            phraselet_info.child_direct_matching_reprs
                        )
                        phraselet_doc[
                            phraselet_template.child_index
                        ]._.holmes.derived_lemma = phraselet_info.child_derived_lemma
                        phraselet_doc[
                            phraselet_template.child_index
                        ]._.holmes.derivation_matching_reprs = (
                            phraselet_info.child_derivation_matching_reprs
                        )
                        phraselet_doc[
                            phraselet_template.child_index
                        ]._.holmes.ent_type = phraselet_info.child_ent_type
                        phraselet_doc[
                            phraselet_template.child_index
                        ]._.holmes.is_initial_question_word = (
                            phraselet_info.child_is_initial_question_word
                        )
                        phraselet_doc[
                            phraselet_template.child_index
                        ]._.holmes.has_initial_question_word_in_phrase = (
                            phraselet_info.child_has_initial_question_word_in_phrase
                        )
                    return self.create_search_phrase(
                        "topic match phraselet",
                        phraselet_doc,
                        phraselet_info.label,
                        phraselet_template,
                        phraselet_info.created_without_matching_tags,
                        (
                            reverse_matching_frequency_threshold is not None
                            and cast(float, phraselet_info.parent_frequency_factor)
                            < reverse_matching_frequency_threshold
                            and phraselet_info.child_lemma is not None
                            and not phraselet_template.question
                        )
                        or phraselet_info.parent_lemma == "ENTITYNOUN",
                        phraselet_info.reverse_only_parent_lemma,
                        True,
                        root_token_index=phraselet_template.parent_index,
                    )
            raise RuntimeError(
                "".join(
                    ("Phraselet template", phraselet_info.template_label, "not found.")
                )
            )

        return {
            phraselet_info.label: create_search_phrase_from_phraselet(phraselet_info)
            for phraselet_info in phraselet_infos
        }

    def get_phraselet_labels_to_phraselet_infos(
        self,
        *,
        text_to_match_doc: Doc,
        words_to_corpus_frequencies: Dict[str, int],
        maximum_corpus_frequency: int,
        process_initial_question_words: bool
    ) -> Dict[str, PhraseletInfo]:
        phraselet_labels_to_phraselet_infos: Dict[str, PhraseletInfo] = {}
        self.add_phraselets_to_dict(
            text_to_match_doc,
            phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors=False,
            match_all_words=False,
            ignore_relation_phraselets=False,
            include_reverse_only=True,
            stop_tags=self.semantic_matching_helper.topic_matching_phraselet_stop_tags,
            stop_lemmas=self.semantic_matching_helper.topic_matching_phraselet_stop_lemmas,
            reverse_only_parent_lemmas=self.semantic_matching_helper.topic_matching_reverse_only_parent_lemmas,
            words_to_corpus_frequencies=words_to_corpus_frequencies,
            maximum_corpus_frequency=maximum_corpus_frequency,
            process_initial_question_words=process_initial_question_words,
        )

        # now add the single word phraselets whose tags did not match.
        self.add_phraselets_to_dict(
            text_to_match_doc,
            phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors=False,
            match_all_words=True,
            ignore_relation_phraselets=True,
            include_reverse_only=False,  # value is irrelevant with
            # ignore_relation_phraselets == True
            stop_lemmas=self.semantic_matching_helper.topic_matching_phraselet_stop_lemmas,
            stop_tags=self.semantic_matching_helper.topic_matching_phraselet_stop_tags,
            reverse_only_parent_lemmas=self.semantic_matching_helper.topic_matching_reverse_only_parent_lemmas,
            words_to_corpus_frequencies=words_to_corpus_frequencies,
            maximum_corpus_frequency=maximum_corpus_frequency,
            process_initial_question_words=False,
        )
        return phraselet_labels_to_phraselet_infos

    def create_search_phrase(
        self,
        search_phrase_text: str,
        search_phrase_doc: Doc,
        label: str,
        phraselet_template: Optional[PhraseletTemplate],
        topic_match_phraselet_created_without_matching_tags: bool,
        treat_as_reverse_only_during_initial_relation_matching: bool,
        is_reverse_only_parent_lemma: Optional[bool],
        process_initial_question_words: bool,
        *,
        root_token_index=None
    ) -> SearchPhrase:
        """phraselet_template -- 'None' if this search phrase is not a topic match phraselet"""

        def replace_grammatical_root_token_recursively(token: Token) -> Token:
            """Where the syntactic root of a search phrase document is a grammatical token or is
            marked as non-matchable, loop through the semantic dependencies to find the
            semantic root.
            """
            for dependency in token._.holmes.children:
                if dependency.child_index < 0:
                    return replace_grammatical_root_token_recursively(
                        token.doc[(0 - dependency.child_index) - 1]
                    )
            if not token._.holmes.is_matchable:
                for dependency in token._.holmes.children:
                    if (
                        dependency.child_index >= 0
                        and dependency.child_token(token.doc)._.holmes.is_matchable
                    ):
                        return replace_grammatical_root_token_recursively(
                            token.doc[dependency.child_index]
                        )
            return token

        for token in search_phrase_doc:
            if len(token._.holmes.righthand_siblings) > 0:
                # SearchPhrases may not themselves contain conjunctions like 'and'
                # because then the matching becomes too complicated
                raise SearchPhraseContainsConjunctionError(search_phrase_text)
            if (
                self.perform_coreference_resolution
                and token.pos_ == "PRON"
                and token._.holmes.is_involved_in_coreference()
            ):
                # SearchPhrases may not themselves contain coreferring pronouns
                # because then the matching becomes too complicated
                raise SearchPhraseContainsCoreferringPronounError(search_phrase_text)
            if token._.holmes.is_negated and phraselet_template is None:
                # SearchPhrases may not themselves contain negation
                # because then the matching becomes too complicated.
                # Not relevant for phraselets to enable callers to add additional
                # phraselets that find negation.
                raise SearchPhraseContainsNegationError(search_phrase_text)

        root_tokens = []
        tokens_to_match = []
        matchable_non_entity_tokens_to_vectors = {}
        token_indexes_within_multiwords_to_ignore: List[int] = []
        if (
            self.ontology is not None
            and self.analyze_derivational_morphology
            and phraselet_template is None
        ):
            for token in search_phrase_doc:
                odm = self.semantic_matching_helper.get_ontology_defined_multiword(
                    token, self.ontology
                )
                if odm is not None:
                    for index in odm.token_indexes:
                        if index == token.i:
                            token._.holmes.lemma = odm.lemma
                            token._.holmes.derived_lemma = odm.derived_lemma
                            token._.holmes.direct_matching_reprs = (
                                odm.direct_matching_reprs
                            )
                            token._.holmes.derivation_matching_reprs = (
                                odm.derivation_matching_reprs
                            )
                        else:
                            token_indexes_within_multiwords_to_ignore.append(index)

        for token in search_phrase_doc:
            if token.i in token_indexes_within_multiwords_to_ignore:
                token._.holmes.is_matchable = False
                continue
            # check whether grammatical token
            if (
                phraselet_template is not None
                and phraselet_template.parent_index != token.i
                and phraselet_template.child_index != token.i
            ):
                token._.holmes.is_matchable = False
            if (
                phraselet_template is not None
                and phraselet_template.parent_index == token.i
                and not phraselet_template.single_word()
                and phraselet_template.assigned_dependency_label is not None
            ):
                for dependency in (
                    dependency
                    for dependency in token._.holmes.children
                    if dependency.child_index == phraselet_template.child_index
                ):
                    dependency.label = phraselet_template.assigned_dependency_label
            if token._.holmes.is_matchable and not (
                len(token._.holmes.children) > 0
                and token._.holmes.children[0].child_index < 0
            ):
                tokens_to_match.append(token)
                if self.semantic_matching_helper.get_entity_placeholder(token) is None:
                    if (
                        phraselet_template is None
                        and len(token._.holmes.lemma.split()) > 1
                    ):
                        working_lexeme = self.semantic_analyzer.vectors_nlp.vocab[
                            token.lemma_
                        ]
                    else:
                        working_lexeme = self.semantic_analyzer.vectors_nlp.vocab[
                            token._.holmes.lemma
                        ]
                    if working_lexeme.has_vector and working_lexeme.vector_norm > 0:
                        matchable_non_entity_tokens_to_vectors[
                            token.i
                        ] = working_lexeme.vector
                    else:
                        matchable_non_entity_tokens_to_vectors[token.i] = None
            if (
                process_initial_question_words
                and self.semantic_analyzer.is_interrogative_pronoun(token)
            ):
                tokens_to_match.append(token)
                matchable_non_entity_tokens_to_vectors[token.i] = None
            if token.dep_ == "ROOT":  # syntactic root
                root_tokens.append(replace_grammatical_root_token_recursively(token))
        if len(tokens_to_match) == 0:
            raise SearchPhraseWithoutMatchableWordsError(search_phrase_text)
        if len(root_tokens) > 1:
            raise SearchPhraseContainsMultipleClausesError(search_phrase_text)
        root_token = root_tokens[0]
        if phraselet_template is None:
            reverse_only = False
        else:
            reverse_only = not phraselet_template.question and (
                is_reverse_only_parent_lemma or phraselet_template.reverse_only
            )

        search_phrase = SearchPhrase(
            search_phrase_doc,
            [token.i for token in tokens_to_match],
            root_token.i if root_token_index is None else root_token_index,
            matchable_non_entity_tokens_to_vectors,
            label,
            phraselet_template is not None,
            topic_match_phraselet_created_without_matching_tags,
            (phraselet_template is not None and phraselet_template.question),
            reverse_only,
            treat_as_reverse_only_during_initial_relation_matching,
            len(tokens_to_match) == 1
            and not (phraselet_template is not None and phraselet_template.question),
        )
        for word_matching_strategy in (
            self.semantic_matching_helper.main_word_matching_strategies
            + self.semantic_matching_helper.ontology_word_matching_strategies
        ):
            word_matching_strategy.add_words_matching_search_phrase_root_token(
                search_phrase
            )
        search_phrase.words_matching_root_token.sort(key=lambda word: 0 - len(word))
        # process longer entries first so that multiwords are considered before their constituent parts
        return search_phrase


class SemanticMatchingHelperFactory:
    """Returns the correct *SemanticMatchingHelperFactory* for the language in use."""

    def semantic_matching_helper(self, *, language: str) -> "SemanticMatchingHelper":
        language_specific_rules_module = importlib.import_module(
            ".".join((".lang", language, "language_specific_rules")), "holmes_extractor"
        )
        return language_specific_rules_module.LanguageSpecificSemanticMatchingHelper()


class SemanticMatchingHelper(ABC):
    """Abstract *SemanticMatchingHelper* parent class containing language-specific properties and
    methods that are required for matching and can be successfully and efficiently serialized.
    Functionality is placed here that is common to all current implementations. It follows that
    some functionality will probably have to be moved out to specific implementations whenever
    an implementation for a new language is added.

    For explanations of the abstract variables and methods, see the
    *EnglishSemanticMatchingHelper* implementation where they can be illustrated with direct
    examples.
    """

    noun_pos: List[str] = NotImplemented

    permissible_embedding_pos: List[str] = NotImplemented

    noun_kernel_dep: List[str] = NotImplemented

    minimum_embedding_match_word_length: int = NotImplemented

    topic_matching_phraselet_stop_lemmas: List[str] = NotImplemented

    topic_matching_reverse_only_parent_lemmas: List[Tuple[str, str]] = NotImplemented

    topic_matching_phraselet_stop_tags: List[str] = NotImplemented

    supervised_document_classification_phraselet_stop_lemmas: List[str] = NotImplemented

    preferred_phraselet_pos: List[str] = NotImplemented

    entity_defined_multiword_pos: List[str] = NotImplemented

    entity_defined_multiword_entity_types: List[str] = NotImplemented

    sibling_marker_deps: List[str] = NotImplemented

    preposition_deps: List[str] = NotImplemented

    interrogative_pronoun_tags: List[str] = NotImplemented

    question_answer_blacklist_deps: List[str] = NotImplemented

    question_answer_final_blacklist_deps: List[str] = NotImplemented

    match_implication_dict: Dict[str, MatchImplication] = NotImplemented

    phraselet_templates: List[PhraseletTemplate] = NotImplemented

    @abstractmethod
    def question_word_matches(
        self,
        search_phrase_token: Token,
        document_token: Token,
        document_subword_index: Optional[int],
        document_vector,
        entity_label_to_vector_dict: dict,
        initial_question_word_embedding_match_threshold: float,
    ) -> bool:
        pass

    def __init__(self) -> None:
        self.local_phraselet_templates = [copy(t) for t in self.phraselet_templates]
        for key, match_implication in self.match_implication_dict.items():
            assert key == match_implication.search_phrase_dependency
            assert key not in match_implication.document_dependencies
            assert (
                len(
                    [
                        dep
                        for dep in match_implication.document_dependencies
                        if match_implication.document_dependencies.count(dep) > 1
                    ]
                )
                == 0
            )
            assert key not in match_implication.reverse_document_dependencies
            assert (
                len(
                    [
                        dep
                        for dep in match_implication.reverse_document_dependencies
                        if match_implication.reverse_document_dependencies.count(dep)
                        > 1
                    ]
                )
                == 0
            )
        self.main_word_matching_strategies: List = []
        self.ontology_word_matching_strategies: List = []
        self.embedding_word_matching_strategies: List = []

    def get_subtree_list_for_question_answer(self, token: Token) -> List[Token]:
        """Returns the part of the subtree of a token that has matched a question word
        that is analysed as answering the question. Essentially, this is all the subtree but
        excluding any areas that are in a conjunction relationship with *token*; these will be
        returned as separate answers in their own right.
        """
        list_to_return = []
        for working_token in token.subtree:
            if (
                token == working_token
                or working_token.dep_ not in self.question_answer_blacklist_deps
                or working_token.text == "-"
            ):
                list_to_return.append(working_token)
            else:
                return [token] if len(list_to_return) == 0 else list_to_return
        if (
            len(list_to_return) > 1
            and list_to_return[-1].dep_ in self.question_answer_final_blacklist_deps
        ):
            list_to_return = list_to_return[:-1]
        return list_to_return

    def cosine_similarity(self, vector1: Floats1d, vector2: Floats1d) -> float:
        ops = get_current_ops()
        return ops.xp.dot(vector1, vector2) / (
            ops.xp.linalg.norm(vector1) * ops.xp.linalg.norm(vector2)
        )

    def token_matches_ent_type(
        self,
        token_vector,
        entity_label_to_vector_dict: dict,
        entity_labels: tuple,
        initial_question_word_embedding_match_threshold: float,
    ) -> float:
        """Checks if the vector of a token lexeme has a similarity to a lexeme regarded as typical
            for one of a group of entity labels above a threshold. If so, returns the similarity;
            if not, returns *0.0*.

        Parameters:

        token_vector -- the document token vector.
        entity_label_to_vector_dict -- a dictionary from entity labels to vectors for lexemes
            regarded as typical for those entity labels.
        entity_labels -- the entity labels to check for similarity.
        initial_question_word_embedding_match_threshold -- the threshold above which a similarity
            is regarded as significant.
        """

        if token_vector is not None:
            for ent_type in entity_labels:
                cosine_similarity = self.cosine_similarity(
                    entity_label_to_vector_dict[ent_type], token_vector
                )
                if cosine_similarity > initial_question_word_embedding_match_threshold:
                    return cosine_similarity
        return 0.0

    def add_to_reverse_dict(
        self,
        reverse_dict: Dict[str, List[CorpusWordPosition]],
        parsed_document: Doc,
        document_label: str,
    ) -> None:
        """Indexes a parsed document."""
        for word_matching_strategy in (
            self.main_word_matching_strategies + self.ontology_word_matching_strategies
        ):
            word_matching_strategy.add_reverse_dict_entries(
                reverse_dict, parsed_document, document_label
            )

    def get_reverse_dict_removing_document(
        self, reverse_dict: Dict[str, List[CorpusWordPosition]], document_label: str
    ) -> Dict[str, List[CorpusWordPosition]]:
        new_reverse_dict = {}
        for entry in reverse_dict:
            new_value = [
                cwp
                for cwp in reverse_dict[entry]
                if cwp.document_label != document_label
            ]
            if len(new_value) > 0:
                new_reverse_dict[entry] = new_value
        return new_reverse_dict

    def dependency_labels_match(
        self,
        *,
        search_phrase_dependency_label: str,
        document_dependency_label: str,
        inverse_polarity: bool
    ) -> bool:
        """Determines whether a dependency label in a search phrase matches a dependency label in
        a document being searched.
        inverse_polarity: *True* if the matching dependencies have to point in opposite
        directions.
        """
        if not inverse_polarity:
            if search_phrase_dependency_label == document_dependency_label:
                return True
            if search_phrase_dependency_label not in self.match_implication_dict.keys():
                return False
            return (
                document_dependency_label
                in self.match_implication_dict[
                    search_phrase_dependency_label
                ].document_dependencies
            )
        else:
            return (
                search_phrase_dependency_label in self.match_implication_dict.keys()
                and document_dependency_label
                in self.match_implication_dict[
                    search_phrase_dependency_label
                ].reverse_document_dependencies
            )

    def get_entity_placeholder(self, search_phrase_token: Token) -> Optional[str]:
        if (
            search_phrase_token.text[:6] == "ENTITY"
            and len(search_phrase_token.text) > 6
        ):
            return search_phrase_token.text
        if (
            search_phrase_token._.holmes.lemma[:6] == "ENTITY"
            and len(search_phrase_token._.holmes.lemma) > 6
        ):
            return search_phrase_token._.holmes.lemma
        return None

    def embedding_matching_permitted(self, obj: Union[Token, Subword]):
        """Embedding matching is suppressed for some parts of speech as well as for very short
        words."""
        if isinstance(obj, Token):
            if len(obj._.holmes.lemma.split()) > 1:
                working_lemma = obj.lemma_
            else:
                working_lemma = obj._.holmes.lemma
            return (
                obj.pos_ in self.permissible_embedding_pos
                and len(working_lemma) >= self.minimum_embedding_match_word_length
            )
        elif isinstance(obj, Subword):
            return len(obj.lemma) >= self.minimum_embedding_match_word_length
        else:
            raise RuntimeError("'obj' must be either a Token or a Subword")

    def belongs_to_entity_defined_multiword(self, token: Token) -> bool:
        return (
            token.pos_ in self.entity_defined_multiword_pos
            and token.ent_type_ in self.entity_defined_multiword_entity_types
        )

    def get_entity_defined_multiword(self, token: Token) -> Optional[MultiwordSpan]:
        """If this token is at the head of a multiword recognized by spaCy named entity processing,
        returns the multiword span, otherwise *None*.
        """
        if (
            not self.belongs_to_entity_defined_multiword(token)
            or (
                token.dep_ != "ROOT"
                and self.belongs_to_entity_defined_multiword(token.head)
            )
            or token.ent_type_ == ""
            or token.left_edge.i == token.right_edge.i
        ):
            return None
        working_ent = token.ent_type_
        working_texts: List[str] = []
        working_indexes: List[int] = []
        for counter in range(token.left_edge.i, token.right_edge.i + 1):
            multiword_token = token.doc[counter]
            if (
                not self.belongs_to_entity_defined_multiword(multiword_token)
                or multiword_token.ent_type_ != working_ent
            ):
                if len(working_texts) > 0:
                    return None
                else:
                    continue
            working_texts.append(multiword_token.text)
            working_indexes.append(multiword_token.i)
        if len(working_texts) > 1:
            multiword_text = " ".join(working_texts)
            return MultiwordSpan(
                multiword_text, None, multiword_text, None, working_indexes
            )
        else:
            return None

    def get_ontology_defined_multiword(
        self, token: Token, ontology: Ontology
    ) -> Optional[MultiwordSpan]:
        if token._.holmes.multiword_spans is None:
            return None
        for multiword_span in token._.holmes.multiword_spans:
            reprs = multiword_span.direct_matching_reprs[:]
            if multiword_span.derivation_matching_reprs is not None:
                reprs.extend(multiword_span.derivation_matching_reprs)
            for repr in reprs:
                if ontology.contains_multiword(repr):
                    return multiword_span
        return None

    def get_dependent_phrase(self, token: Token, subword: Subword) -> str:
        """Return the dependent phrase of a token, with an optional subword reference. Used in
        building match dictionaries."""
        if subword is not None:
            return subword.text
        if not token.pos_ in self.noun_pos:
            return token.text
        return_string = ""
        pointer = token.left_edge.i - 1
        while True:
            pointer += 1
            if (
                token.doc[pointer].pos_ not in self.noun_pos
                and token.doc[pointer].dep_ not in self.noun_kernel_dep
                and pointer > token.i
            ):
                return return_string.strip()
            if return_string == "":
                return_string = token.doc[pointer].text
            else:
                return_string = " ".join((return_string, token.doc[pointer].text))
            if token.right_edge.i <= pointer:
                return return_string
