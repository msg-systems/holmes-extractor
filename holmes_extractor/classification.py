from typing import List, Tuple, Dict, Callable, Optional, cast, Any, Set
from collections import OrderedDict
import uuid
import statistics
import pickle
from holmes_extractor.word_matching.ontology import OntologyWordMatchingStrategy
from tqdm import tqdm
from spacy.tokens import Doc
from thinc.api import Model
from thinc.backends import get_current_ops, Ops
from thinc.loss import SequenceCategoricalCrossentropy
from thinc.layers import chain, Relu, Softmax
from thinc.optimizers import Adam
from thinc.types import Floats2d
from .errors import (
    WrongModelDeserializationError,
    FewerThanTwoClassificationsError,
    DuplicateDocumentError,
    NoPhraseletsAfterFilteringError,
    IncompatibleAnalyzeDerivationalMorphologyDeserializationError,
)
from .structural_matching import Match, StructuralMatcher
from .ontology import Ontology
from .parsing import (
    CorpusWordPosition,
    LinguisticObjectFactory,
    SearchPhrase,
    SemanticAnalyzer,
    SemanticMatchingHelper,
)
from .parsing import PhraseletInfo


class SupervisedTopicTrainingUtils:
    def __init__(self, overlap_memory_size, one_hot):
        self.overlap_memory_size = overlap_memory_size
        self.one_hot = one_hot

    def get_labels_to_classification_frequencies_dict(
        self,
        *,
        matches: List[Match],
        labels_to_classifications_dict: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Builds a dictionary from search phrase (phraselet) labels to classification
        frequencies. Depending on the training phase, which is signalled by the parameters, the
        dictionary tracks either raw frequencies for each search phrase label or points to a
        second dictionary from classification labels to frequencies.

        Parameters:

        matches -- the structural matches from which to build the dictionary
        labels_to_classifications_dict -- a dictionary from document labels to document
            classifications, or 'None' if the target dictionary should contain raw frequencies.
        """

        def increment(search_phrase_label, document_label):
            if labels_to_classifications_dict is not None:
                if search_phrase_label not in labels_to_frequencies_dict:
                    classification_frequency_dict = {}
                    labels_to_frequencies_dict[
                        search_phrase_label
                    ] = classification_frequency_dict
                else:
                    classification_frequency_dict = labels_to_frequencies_dict[
                        search_phrase_label
                    ]
                classification = labels_to_classifications_dict[document_label]
                if classification in classification_frequency_dict:
                    classification_frequency_dict[classification] += 1
                else:
                    classification_frequency_dict[classification] = 1
            else:
                if search_phrase_label not in labels_to_frequencies_dict:
                    labels_to_frequencies_dict[search_phrase_label] = 1
                else:
                    labels_to_frequencies_dict[search_phrase_label] += 1

        def relation_match_involves_whole_word_containing_subwords(match):
            # Where there are subwords, we suppress relation matches with the
            # entire word. The same rule is not applied to single-word matches because
            # it still makes sense to track words with more than three subwords.
            return (
                len(match.word_matches) > 1
                and len(
                    [
                        word_match
                        for word_match in match.word_matches
                        if len(word_match.document_token._.holmes.subwords) > 0
                        and word_match.document_subword is None
                    ]
                )
                > 0
            )

        labels_to_frequencies_dict: Dict[str, Any] = {}
        matches = [
            match
            for match in matches
            if not relation_match_involves_whole_word_containing_subwords(match)
        ]
        matches = sorted(
            matches,
            key=lambda match: (
                match.document_label,
                match.index_within_document,
                match.get_subword_index_for_sorting(),
            ),
        )
        for index, match in enumerate(matches):
            this_document_label: Optional[str]
            if self.one_hot:
                if (
                    "this_document_label" not in locals()
                ) or this_document_label != match.document_label:
                    this_document_label = match.document_label
                    search_phrases_added_for_this_document = set()
                if (
                    match.search_phrase_label
                    not in search_phrases_added_for_this_document
                ):
                    increment(match.search_phrase_label, match.document_label)
                    search_phrases_added_for_this_document.add(
                        match.search_phrase_label
                    )
            else:
                increment(match.search_phrase_label, match.document_label)
            if not match.from_single_word_phraselet:
                previous_match_index = index
                number_of_analyzed_matches_counter = 0
                while (
                    previous_match_index > 0
                    and number_of_analyzed_matches_counter <= self.overlap_memory_size
                ):
                    previous_match_index -= 1
                    previous_match = matches[previous_match_index]
                    if previous_match.document_label != match.document_label:
                        break
                    if previous_match.from_single_word_phraselet:
                        continue
                    if previous_match.search_phrase_label == match.search_phrase_label:
                        continue  # otherwise coreference resolution leads to phrases being
                        # combined with themselves
                    number_of_analyzed_matches_counter += 1
                    previous_word_match_doc_indexes = [
                        word_match.get_document_index()
                        for word_match in previous_match.word_matches
                    ]
                    for word_match in match.word_matches:
                        if (
                            word_match.get_document_index()
                            in previous_word_match_doc_indexes
                        ):
                            # the same word is involved in both matches, so combine them
                            # into a new label
                            label_parts = sorted(
                                (
                                    previous_match.search_phrase_label,
                                    match.search_phrase_label,
                                )
                            )
                            combined_label = "/".join((label_parts[0], label_parts[1]))
                            if self.one_hot:
                                if (
                                    combined_label
                                    not in search_phrases_added_for_this_document
                                ):
                                    increment(combined_label, match.document_label)
                                    search_phrases_added_for_this_document.add(
                                        combined_label
                                    )
                            else:
                                increment(combined_label, match.document_label)
        return labels_to_frequencies_dict

    def get_occurrence_dicts(
        self,
        *,
        phraselet_labels_to_search_phrases: Dict[str, SearchPhrase],
        semantic_matching_helper: SemanticMatchingHelper,
        structural_matcher: StructuralMatcher,
        sorted_label_dict: Dict[str, int],
        overall_similarity_threshold: float,
        training_document_labels_to_documents: Dict[str, Doc]
    ) -> List[Dict[int, int]]:
        """Matches documents against the currently stored phraselets and records the matches
        in a custom sparse format.

        Parameters:

        phraselet_labels_to_search_phrases -- a dictionary from search phrase (phraselet)
            labels to search phrase objects.
        semantic_matching_helper -- the semantic matching helper to use.
        structural_matcher -- the structural matcher to use for comparisons.
        sorted_label_dict -- a dictionary from search phrase (phraselet) labels to their own
            alphabetic sorting indexes.
        overall_similarity_threshold -- the threshold for embedding-based matching.
        training_document_labels_to_documents -- a dictionary.
        """
        return_dicts: List[Dict[int, int]] = []
        for doc_label in sorted(training_document_labels_to_documents.keys()):
            this_document_dict: Dict[int, int] = {}
            doc = training_document_labels_to_documents[doc_label]
            document_labels_to_documents = {doc_label: doc}
            reverse_dict: Dict[str, List[CorpusWordPosition]] = {}
            semantic_matching_helper.add_to_reverse_dict(reverse_dict, doc, doc_label)
            for (
                label,
                occurrences,
            ) in self.get_labels_to_classification_frequencies_dict(
                matches=structural_matcher.match(
                    word_matching_strategies=semantic_matching_helper.main_word_matching_strategies
                    + semantic_matching_helper.ontology_word_matching_strategies
                    + semantic_matching_helper.embedding_word_matching_strategies,
                    document_labels_to_documents=document_labels_to_documents,
                    reverse_dict=reverse_dict,
                    search_phrases=phraselet_labels_to_search_phrases.values(),
                    match_depending_on_single_words=None,
                    compare_embeddings_on_root_words=False,
                    compare_embeddings_on_non_root_words=True,
                    reverse_matching_cwps=None,
                    embedding_reverse_matching_cwps=None,
                    process_initial_question_words=False,
                    overall_similarity_threshold=overall_similarity_threshold,
                    initial_question_word_overall_similarity_threshold=1.0,
                ),
                labels_to_classifications_dict=None,
            ).items():
                if self.one_hot:
                    occurrences = 1
                if (
                    label in sorted_label_dict
                ):  # may not be the case for compound labels
                    label_index = sorted_label_dict[label]
                    this_document_dict[label_index] = occurrences
            return_dicts.append(this_document_dict)
        return return_dicts

    def get_thinc_model(
        self, *, hidden_layer_sizes: List[int], input_width: int, output_width: int
    ) -> Model[List[Dict[int, int]], Floats2d]:
        """Generates the structure — without weights — of the Thinc neural network.

        Parameters:

        hidden_layer_sizes -- a list containing the number of neurons in each hidden layer.
        input_width -- the input neuron width, which corresponds to the number of phraselets.
        output_width -- the output neuron width, which corresponds to the number of classifications.
        """

        def get_doc_infos(
            input_len,
        ) -> Model[List[Dict[int, int]], Floats2d]:
            model: Model[List[Dict[int, int]], Floats2d] = Model(
                "doc_infos_forward", doc_infos_forward
            )
            model.attrs["input_len"] = input_len
            return model

        def doc_infos_forward(
            model: Model[List[Dict[int, int]], Floats2d],
            occurrence_dicts: List[Dict[int, int]],
            is_train: bool,
        ) -> Tuple[Floats2d, Callable]:
            def backprop(sparse_infos: Floats2d) -> List[Dict[int, int]]:
                return []

            input_len = model.attrs["input_len"]
            return_matrix = model.ops.alloc2f(len(occurrence_dicts), input_len) + 0.0

            for index, occurrence_dict in enumerate(occurrence_dicts):
                for key, value in occurrence_dict.items():
                    return_matrix[index, key] = value

            return return_matrix, backprop

        hidden_layers: Model[Floats2d, Floats2d]
        if len(hidden_layer_sizes) == 1:
            hidden_layers = Relu(hidden_layer_sizes[0])
        else:
            hidden_layers = chain(*(Relu(size) for size in hidden_layer_sizes))
        model: Model[List[Dict[int, int]], Floats2d] = chain(
            get_doc_infos(input_width),
            hidden_layers,
            Softmax(output_width),
        )
        return model


class SupervisedTopicTrainingBasis:
    """Holder object for training documents and their classifications from which one or more
    'SupervisedTopicModelTrainer' objects can be derived. This class is *NOT* threadsafe.
    """

    def __init__(
        self,
        *,
        linguistic_object_factory: LinguisticObjectFactory,
        structural_matcher: StructuralMatcher,
        classification_ontology: Optional[Ontology],
        overlap_memory_size: int,
        one_hot: bool,
        match_all_words: bool,
        overall_similarity_threshold: float,
        verbose: bool
    ):
        """Parameters:

        linguistic_object_factory -- the linguistic object factory to use
        structural_matcher -- the structural matcher to use.
        classification_ontology -- an Ontology object incorporating relationships between
            classification labels.
        overlap_memory_size -- how many non-word phraselet matches to the left should be
            checked for words in common with a current match.
        one_hot -- whether the same word or relationship matched multiple times should be
            counted once only (value 'True') or multiple times (value 'False')
        match_all_words -- whether all single words should be taken into account
            (value 'True') or only single words with noun tags (value 'False')
        overall_similarity_threshold -- the overall similarity threshold for embedding-based
            matching. Defaults to *1.0*, which deactivates embedding-based matching.
        verbose -- if 'True', information about training progress is outputted to the console.
        """
        self.linguistic_object_factory = linguistic_object_factory
        self.structural_matcher = structural_matcher
        self.semantic_analyzer = linguistic_object_factory.semantic_analyzer
        self.semantic_matching_helper = (
            linguistic_object_factory.semantic_matching_helper
        )
        self.overall_similarity_threshold = overall_similarity_threshold
        self.classification_ontology = classification_ontology
        self.utils = SupervisedTopicTrainingUtils(overlap_memory_size, one_hot)
        self.match_all_words = match_all_words
        self.verbose = verbose

        self.training_document_labels_to_documents: Dict[str, Doc] = {}
        self.reverse_dict: Dict[str, List[CorpusWordPosition]] = {}
        self.training_documents_labels_to_classifications_dict: Dict[str, str] = {}
        self.additional_classification_labels: Set[str] = set()
        self.classification_implication_dict: Dict[str, List[str]] = {}
        self.labels_to_classification_frequencies: Optional[Dict[str, Any]] = None
        self.phraselet_labels_to_phraselet_infos: Dict[str, PhraseletInfo] = {}
        self.classifications: Optional[List[str]] = None

    def parse_and_register_training_document(
        self, text: str, classification: str, label: Optional[str] = None
    ) -> None:
        """Parses and registers a document to use for training.

        Parameters:

        text -- the document text
        classification -- the classification label
        label -- a label with which to identify the document in verbose training output,
            or 'None' if a random label should be assigned.
        """
        self.register_training_document(
            self.semantic_analyzer.parse(text), classification, label
        )

    def register_training_document(
        self, doc: Doc, classification: str, label: Optional[str]
    ) -> None:
        """Registers a pre-parsed document to use for training.

        Parameters:

        doc -- the document
        classification -- the classification label
        label -- a label with which to identify the document in verbose training output,
            or 'None' if a random label should be assigned.
        """
        if self.labels_to_classification_frequencies is not None:
            raise RuntimeError(
                "register_training_document() may not be called once prepare() has been called"
            )
        if label is None:
            label = str(uuid.uuid4())
        if label in self.training_document_labels_to_documents:
            raise DuplicateDocumentError(label)
        if self.verbose:
            print("Registering document", label)
        self.training_document_labels_to_documents[label] = doc
        self.semantic_matching_helper.add_to_reverse_dict(self.reverse_dict, doc, label)
        self.linguistic_object_factory.add_phraselets_to_dict(
            doc,
            phraselet_labels_to_phraselet_infos=self.phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors=True,
            match_all_words=self.match_all_words,
            ignore_relation_phraselets=False,
            include_reverse_only=False,
            stop_lemmas=self.semantic_matching_helper.supervised_document_classification_phraselet_stop_lemmas,
            stop_tags=self.semantic_matching_helper.topic_matching_phraselet_stop_tags,
            reverse_only_parent_lemmas=None,
            words_to_corpus_frequencies=None,
            maximum_corpus_frequency=None,
            process_initial_question_words=False,
        )
        self.training_documents_labels_to_classifications_dict[label] = classification

    def register_additional_classification_label(self, label: str) -> None:
        """Register an additional classification label which no training document has explicitly
        but that should be assigned to documents whose explicit labels are related to the
        additional classification label via the classification ontology.
        """
        if self.labels_to_classification_frequencies is not None:
            raise RuntimeError(
                "register_additional_classification_label() may not be called once prepare() has "
                " been called"
            )
        if (
            self.classification_ontology is not None
            and self.classification_ontology.contains_word(label)
        ):
            self.additional_classification_labels.add(label)

    def prepare(self) -> None:
        """Matches the phraselets derived from the training documents against the training
        documents to generate frequencies that also include combined labels, and examines the
        explicit classification labels, the additional classification labels and the
        classification ontology to derive classification implications.

        Once this method has been called, the instance no longer accepts new training documents
        or additional classification labels.
        """
        if self.labels_to_classification_frequencies is not None:
            raise RuntimeError("prepare() may only be called once")
        if self.verbose:
            print("Matching documents against all phraselets")
        search_phrases = (
            self.linguistic_object_factory.create_search_phrases_from_phraselet_infos(
                list(self.phraselet_labels_to_phraselet_infos.values())
            ).values()
        )
        self.labels_to_classification_frequencies = cast(
            Dict[str, Dict[str, int]],
            self.utils.get_labels_to_classification_frequencies_dict(
                matches=self.structural_matcher.match(
                    word_matching_strategies=self.semantic_matching_helper.main_word_matching_strategies
                    + self.semantic_matching_helper.ontology_word_matching_strategies
                    + self.semantic_matching_helper.embedding_word_matching_strategies,
                    document_labels_to_documents=self.training_document_labels_to_documents,
                    reverse_dict=self.reverse_dict,
                    search_phrases=search_phrases,
                    match_depending_on_single_words=None,
                    compare_embeddings_on_root_words=False,
                    compare_embeddings_on_non_root_words=True,
                    reverse_matching_cwps=None,
                    embedding_reverse_matching_cwps=None,
                    process_initial_question_words=False,
                    overall_similarity_threshold=self.overall_similarity_threshold,
                    initial_question_word_overall_similarity_threshold=1.0,
                ),
                labels_to_classifications_dict=self.training_documents_labels_to_classifications_dict,
            ),
        )
        self.classifications = sorted(
            set(self.training_documents_labels_to_classifications_dict.values()).union(
                self.additional_classification_labels
            )
        )
        if len(self.classifications) < 2:
            raise FewerThanTwoClassificationsError(len(self.classifications))
        if self.classification_ontology is not None:
            for parent in self.classifications:
                for child in self.classifications:
                    if (
                        self.classification_ontology.matches(parent, [child])
                        is not None
                    ):
                        if child in self.classification_implication_dict.keys():
                            self.classification_implication_dict[child].append(parent)
                        else:
                            self.classification_implication_dict[child] = [parent]

    def train(
        self,
        *,
        minimum_occurrences: int = 4,
        cv_threshold: float = 1.0,
        learning_rate: float = 0.001,
        batch_size: int = 5,
        max_epochs: int = 200,
        convergence_threshold: float = 0.0001,
        hidden_layer_sizes: Optional[List[int]] = None,
        shuffle: bool = True,
        normalize: bool = True
    ) -> "SupervisedTopicModelTrainer":
        """Trains a model based on the prepared state.

        Parameters:

        minimum_occurrences -- the minimum number of times a word or relationship has to
            occur in the context of at least one single classification for the phraselet
            to be accepted into the final model.
        cv_threshold -- the minimum coefficient of variation a word or relationship has
            to occur with respect to explicit classification labels for the phraselet to be
            accepted into the final model.
        learning_rate -- the learning rate for the Adam optimizer.
        batch_size -- the number of documents in each training batch.
        max_epochs -- the maximum number of training epochs.
        convergence_threshold -- the threshold below which loss measurements after consecutive
            epochs are regarded as equivalent. Training stops before *max_epochs* is reached
            if equivalent results are achieved after four consecutive epochs.
        hidden_layer_sizes -- a list containing the number of neurons in each hidden layer, or
            'None' if the topology should be determined automatically.
        shuffle -- *True* if documents should be shuffled during batching.
        normalize -- *True* if normalization should be applied to the loss function.
        """

        if self.labels_to_classification_frequencies is None:
            raise RuntimeError(
                "train() may only be called after prepare() has been called"
            )
        return SupervisedTopicModelTrainer(
            training_basis=self,
            linguistic_object_factory=self.linguistic_object_factory,
            structural_matcher=self.structural_matcher,
            labels_to_classification_frequencies=self.labels_to_classification_frequencies,
            phraselet_infos=list(self.phraselet_labels_to_phraselet_infos.values()),
            minimum_occurrences=minimum_occurrences,
            cv_threshold=cv_threshold,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold,
            hidden_layer_sizes=hidden_layer_sizes,
            shuffle=shuffle,
            normalize=normalize,
            utils=self.utils,
        )


class SupervisedTopicModelTrainer:
    """Worker object used to train and generate models. This object could be removed from the public interface
    (`SupervisedTopicTrainingBasis.train()` could return a `SupervisedTopicClassifier` directly) but has
    been retained to facilitate testability.

    This class is NOT threadsafe.
    """

    def __init__(
        self,
        *,
        training_basis: SupervisedTopicTrainingBasis,
        linguistic_object_factory: LinguisticObjectFactory,
        structural_matcher: StructuralMatcher,
        labels_to_classification_frequencies: Dict[str, Dict[str, int]],
        phraselet_infos: List[PhraseletInfo],
        minimum_occurrences: int,
        cv_threshold: float,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        convergence_threshold: float,
        hidden_layer_sizes: Optional[List[int]],
        shuffle: bool,
        normalize: bool,
        utils: SupervisedTopicTrainingUtils
    ):

        self.utils = utils
        self.semantic_analyzer = linguistic_object_factory.semantic_analyzer
        self.linguistic_object_factory = linguistic_object_factory
        self.semantic_matching_helper = (
            linguistic_object_factory.semantic_matching_helper
        )
        self.structural_matcher = structural_matcher
        self.training_basis = training_basis
        self.minimum_occurrences = minimum_occurrences
        self.cv_threshold = cv_threshold
        self.labels_to_classification_frequencies, self.phraselet_infos = self.filter(
            labels_to_classification_frequencies, phraselet_infos
        )

        if len(self.phraselet_infos) == 0:
            raise NoPhraseletsAfterFilteringError(
                "".join(
                    (
                        "minimum_occurrences: ",
                        str(minimum_occurrences),
                        "; cv_threshold: ",
                        str(cv_threshold),
                    )
                )
            )

        phraselet_labels_to_search_phrases = (
            self.linguistic_object_factory.create_search_phrases_from_phraselet_infos(
                self.phraselet_infos
            )
        )
        self.sorted_label_dict = {}
        for index, label in enumerate(
            sorted(self.labels_to_classification_frequencies.keys())
        ):
            self.sorted_label_dict[label] = index
        if self.training_basis.verbose:
            print("Matching documents against filtered phraselets")
        self.occurrence_dicts = self.utils.get_occurrence_dicts(
            phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
            semantic_matching_helper=self.semantic_matching_helper,
            structural_matcher=self.structural_matcher,
            sorted_label_dict=self.sorted_label_dict,
            overall_similarity_threshold=self.training_basis.overall_similarity_threshold,
            training_document_labels_to_documents=self.training_basis.training_document_labels_to_documents,
        )
        self.output_matrix = self.record_classifications_for_training()

        self._hidden_layer_sizes = hidden_layer_sizes
        if self._hidden_layer_sizes is None or len(self._hidden_layer_sizes) == 0:
            start = len(self.sorted_label_dict)
            step = (
                len(self.training_basis.classifications)  # type:ignore[arg-type]
                - len(self.sorted_label_dict)
            ) / 3
            self._hidden_layer_sizes = [
                start,
                int(start + step),
                int(start + (2 * step)),
            ]
        if self.training_basis.verbose:
            print("Hidden layer sizes:", self._hidden_layer_sizes)

        self._thinc_model = self.utils.get_thinc_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            input_width=len(self.sorted_label_dict),
            output_width=len(
                self.training_basis.classifications  # type:ignore[arg-type]
            ),
        )
        optimizer = Adam(learning_rate)
        average_losses: List[float] = []
        initialized = False
        for epoch in range(1, max_epochs):
            if self.training_basis.verbose:
                print("Epoch", epoch)
                batches = tqdm(
                    self._thinc_model.ops.multibatch(
                        batch_size,
                        self.occurrence_dicts,
                        self.output_matrix,
                        shuffle=shuffle,
                    )
                )
            else:
                batches = self._thinc_model.ops.multibatch(
                    batch_size,
                    self.occurrence_dicts,
                    self.output_matrix,
                    shuffle=shuffle,
                )
            loss_calc = SequenceCategoricalCrossentropy(normalize=normalize)
            losses = []
            for X, Y in batches:
                if not initialized:
                    self._thinc_model.initialize(X, Y)
                    initialized = True
                Yh, backprop = cast(
                    Tuple[Floats2d, Callable], self._thinc_model.begin_update(X)
                )
                grads, loss = loss_calc(Yh, Y)  # type:ignore[arg-type]
                losses.append(loss.tolist())  # type: ignore[attr-defined]
                backprop(
                    self._thinc_model.ops.asarray2f(grads)  # type:ignore[arg-type]
                )
                self._thinc_model.finish_update(optimizer)
            average_loss = round(sum(losses) / len(losses), 6)
            if self.training_basis.verbose:
                print("Average absolute loss:", average_loss)
                print()
            average_losses.append(average_loss)
            if (
                len(average_losses) >= 4
                and abs(average_losses[-1] - average_losses[-2]) < convergence_threshold
                and abs(average_losses[-2] - average_losses[-3]) < convergence_threshold
                and abs(average_losses[-3] - average_losses[-4]) < convergence_threshold
            ):
                if self.training_basis.verbose:
                    print("Neural network converged after", epoch, "epochs.")
                break

    def filter(
        self,
        labels_to_classification_frequencies: Dict[str, Dict[str, int]],
        phraselet_infos: List[PhraseletInfo],
    ) -> Tuple[Dict[str, Dict[str, int]], List[PhraseletInfo]]:
        """Filters the phraselets in memory based on minimum_occurrences and cv_threshold."""

        accepted = 0
        underminimum_occurrences = 0
        under_minimum_cv = 0
        new_labels_to_classification_frequencies = {}
        for (
            label,
            classification_frequencies,
        ) in labels_to_classification_frequencies.items():
            at_least_minimum = False
            working_classification_frequencies = classification_frequencies.copy()
            for classification in working_classification_frequencies:
                if (
                    working_classification_frequencies[classification]
                    >= self.minimum_occurrences
                ):
                    at_least_minimum = True
            if not at_least_minimum:
                underminimum_occurrences += 1
                continue
            frequency_list = list(working_classification_frequencies.values())
            # We only want to take explicit classification labels into account, i.e. ignore the
            # classification ontology.
            number_of_classification_labels = len(
                set(
                    self.training_basis.training_documents_labels_to_classifications_dict.values()
                )
            )
            frequency_list.extend([0] * number_of_classification_labels)
            frequency_list = frequency_list[:number_of_classification_labels]
            if (
                statistics.pstdev(frequency_list) / statistics.mean(frequency_list)
                >= self.cv_threshold
            ):
                accepted += 1
                new_labels_to_classification_frequencies[
                    label
                ] = classification_frequencies
            else:
                under_minimum_cv += 1
        if self.training_basis.verbose:
            print(
                "Filtered: accepted",
                accepted,
                "; removed minimum occurrences",
                underminimum_occurrences,
                "; removed cv threshold",
                under_minimum_cv,
            )
        new_phraselet_infos = [
            phraselet_info
            for phraselet_info in phraselet_infos
            if phraselet_info.label in new_labels_to_classification_frequencies.keys()
        ]
        return new_labels_to_classification_frequencies, new_phraselet_infos

    def record_classifications_for_training(self) -> Floats2d:
        ops: Ops = get_current_ops()
        output_matrix = (
            ops.alloc2f(
                len(
                    self.training_basis.training_documents_labels_to_classifications_dict
                ),
                len(self.training_basis.classifications),  # type:ignore[arg-type]
            )
            + 0.0
        )

        for index, training_document_label in enumerate(
            sorted(
                self.training_basis.training_documents_labels_to_classifications_dict.keys()
            )
        ):
            classification = (
                self.training_basis.training_documents_labels_to_classifications_dict[
                    training_document_label
                ]
            )
            classification_index = (
                self.training_basis.classifications.index(  # type:ignore[union-attr]
                    classification
                )
            )
            output_matrix[index, classification_index] = 1.0
            if classification in self.training_basis.classification_implication_dict:
                for (
                    implied_classification
                ) in self.training_basis.classification_implication_dict[
                    classification
                ]:
                    implied_classification_index = self.training_basis.classifications.index(  # type:ignore[union-attr]
                        implied_classification
                    )
                    output_matrix[index, implied_classification_index] = 1.0
        return output_matrix

    def classifier(self):
        """Returns a supervised topic classifier which contains no explicit references to the
        training data and that can be serialized.
        """
        model = SupervisedTopicClassifierModel(
            semantic_analyzer_model=self.semantic_analyzer.model,
            structural_matching_ontology=self.linguistic_object_factory.ontology,
            phraselet_infos=self.phraselet_infos,
            sorted_label_dict=self.sorted_label_dict,
            classifications=self.training_basis.classifications,
            overlap_memory_size=self.utils.overlap_memory_size,
            one_hot=self.utils.one_hot,
            analyze_derivational_morphology=self.structural_matcher.analyze_derivational_morphology,
            hidden_layer_sizes=self._hidden_layer_sizes,
            serialized_thinc_model=self._thinc_model.to_dict(),
        )
        return SupervisedTopicClassifier(
            self.semantic_analyzer,
            self.linguistic_object_factory,
            self.structural_matcher,
            model,
            self.training_basis.overall_similarity_threshold,
            self.training_basis.verbose,
        )


class SupervisedTopicClassifierModel:
    """A serializable classifier model.
    Parameters:
    semantic_analyzer_model -- a string specifying the spaCy model with which this instance
        was generated and with which it must be used.
    structural_matching_ontology -- the ontology used for matching documents against this model
            (not the classification ontology!)
    phraselet_infos -- the phraselets used for structural matching
    sorted_label_dict -- a dictionary from search phrase (phraselet) labels to their own
        alphabetic sorting indexes.
    classifications -- an ordered list of classification labels corresponding to the
        neural network outputs
    overlap_memory_size -- how many non-word phraselet matches to the left should be
        checked for words in common with a current match.
    one_hot -- whether the same word or relationship matched multiple times should be
        counted once only (value 'True') or multiple times (value 'False')
    analyze_derivational_morphology -- the value of this manager parameter that was in force
        when the model was built. The same value has to be in force when the model is
        deserialized and reused.
    hidden_layer_sizes -- the definition of the topology of the neural-network hidden layers
    serialized_thinc_model -- the serialized neural-network weights
    """

    def __init__(
        self,
        *,
        semantic_analyzer_model: str,
        structural_matching_ontology: Ontology,
        phraselet_infos: List[PhraseletInfo],
        sorted_label_dict: Dict[str, int],
        classifications: List[str],
        overlap_memory_size: int,
        one_hot: bool,
        analyze_derivational_morphology: bool,
        hidden_layer_sizes: List[int],
        serialized_thinc_model: Dict
    ):
        self.semantic_analyzer_model = semantic_analyzer_model
        self.structural_matching_ontology = structural_matching_ontology
        self.phraselet_infos = phraselet_infos
        self.sorted_label_dict = sorted_label_dict
        self.classifications = classifications
        self.overlap_memory_size = overlap_memory_size
        self.one_hot = one_hot
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.hidden_layer_sizes = hidden_layer_sizes
        self.serialized_thinc_model = serialized_thinc_model
        self.version = "1.0"


class SupervisedTopicClassifier:
    """Classifies new documents based on a pre-trained model."""

    def __init__(
        self,
        semantic_analyzer: SemanticAnalyzer,
        linguistic_object_factory: LinguisticObjectFactory,
        structural_matcher: StructuralMatcher,
        model: SupervisedTopicClassifierModel,
        overall_similarity_threshold: float,
        verbose: bool,
    ):
        self.model = model
        self.semantic_analyzer = semantic_analyzer
        self.linguistic_object_factory = linguistic_object_factory
        self.semantic_matching_helper = (
            linguistic_object_factory.semantic_matching_helper
        )
        self.structural_matcher = structural_matcher
        self.overall_similarity_threshold = overall_similarity_threshold
        self.verbose = verbose
        self.utils = SupervisedTopicTrainingUtils(
            model.overlap_memory_size, model.one_hot
        )
        if self.semantic_analyzer.model != model.semantic_analyzer_model:
            raise WrongModelDeserializationError(model.semantic_analyzer_model)
        if (
            self.structural_matcher.analyze_derivational_morphology
            != model.analyze_derivational_morphology
        ):
            raise IncompatibleAnalyzeDerivationalMorphologyDeserializationError(
                "".join(
                    (
                        "manager: ",
                        str(self.structural_matcher.analyze_derivational_morphology),
                        "; model: ",
                        str(model.analyze_derivational_morphology),
                    )
                )
            )
        self.linguistic_object_factory.ontology = model.structural_matching_ontology
        self.semantic_matching_helper = self.structural_matcher.semantic_matching_helper
        if model.structural_matching_ontology is not None:
            self.linguistic_object_factory.ontology_reverse_derivational_dict = (
                self.semantic_analyzer.get_ontology_reverse_derivational_dict(
                    model.structural_matching_ontology
                )
            )
            self.semantic_matching_helper.ontology_word_matching_strategies = [
                OntologyWordMatchingStrategy(
                    self.semantic_matching_helper,
                    perform_coreference_resolution=self.structural_matcher.perform_coreference_resolution,
                    ontology=model.structural_matching_ontology,
                    analyze_derivational_morphology=model.analyze_derivational_morphology,
                    ontology_reverse_derivational_dict=self.linguistic_object_factory.ontology_reverse_derivational_dict,
                )
            ]
        self.phraselet_labels_to_search_phrases = (
            self.linguistic_object_factory.create_search_phrases_from_phraselet_infos(
                model.phraselet_infos
            )
        )

        self.thinc_model = self.utils.get_thinc_model(
            hidden_layer_sizes=model.hidden_layer_sizes,
            input_width=len(model.sorted_label_dict),
            output_width=len(model.classifications),
        )
        self.thinc_model.from_dict(model.serialized_thinc_model)

    def parse_and_classify(self, text: str) -> Optional[OrderedDict]:
        """Returns a dictionary from classification labels to probabilities
        ordered starting with the most probable, or *None* if the text did
        not contain any words recognised by the model.

        Parameter:

        text -- the text to parse and classify.
        """
        return self.classify(self.semantic_analyzer.parse(text))

    def classify(self, doc: Doc) -> Optional[OrderedDict]:
        """Returns a dictionary from classification labels to probabilities
        ordered starting with the most probable, or *None* if the text did
        not contain any words recognised by the model.

        Parameter:

        doc -- the pre-parsed document to classify.
        """

        if self.thinc_model is None:
            raise RuntimeError("No model defined")
        occurrence_dicts = self.utils.get_occurrence_dicts(
            semantic_matching_helper=self.semantic_matching_helper,
            structural_matcher=self.structural_matcher,
            phraselet_labels_to_search_phrases=self.phraselet_labels_to_search_phrases,
            sorted_label_dict=self.model.sorted_label_dict,
            overall_similarity_threshold=self.overall_similarity_threshold,
            training_document_labels_to_documents={"": doc},
        )
        if len(occurrence_dicts[0]) == 0:
            return None
        else:
            return_dict = OrderedDict()
            predictions = self.thinc_model.predict(occurrence_dicts)[0]
            for i in (-predictions).argsort():  # type:ignore[attr-defined]
                return_dict[self.model.classifications[i.item()]] = predictions[
                    i
                ].item()

            return return_dict

    def serialize_model(self) -> bytes:
        return pickle.dumps(self.model)
