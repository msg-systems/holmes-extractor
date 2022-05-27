from typing import List, Dict, Optional, Any
from multiprocessing import Process, Queue, Manager as MultiprocessingManager, cpu_count
from threading import Lock
from string import punctuation
from math import sqrt
import traceback
import sys
import os
import pickle
import pkg_resources
import spacy
import coreferee
from spacy import Language
from spacy.compat import Literal
from spacy.tokens import Doc, Token
from wasabi import Printer  # type: ignore[import]
from thinc.api import Config
from .errors import *
from .structural_matching import StructuralMatcher
from .ontology import Ontology
from .parsing import (
    SemanticAnalyzerFactory,
    SemanticAnalyzer,
    SemanticMatchingHelperFactory,
    LinguisticObjectFactory,
    SearchPhrase,
    SERIALIZED_DOCUMENT_VERSION,
)
from .classification import SupervisedTopicTrainingBasis, SupervisedTopicClassifier
from .topic_matching import TopicMatcher, TopicMatchDictionaryOrderer
from .consoles import HolmesConsoles
from .word_matching.derivation import DerivationWordMatchingStrategy
from .word_matching.direct import DirectWordMatchingStrategy
from .word_matching.embedding import EmbeddingWordMatchingStrategy
from .word_matching.entity import EntityWordMatchingStrategy
from .word_matching.entity_embedding import EntityEmbeddingWordMatchingStrategy
from .word_matching.general import WordMatchingStrategy
from .word_matching.ontology import OntologyWordMatchingStrategy

TIMEOUT_SECONDS = 180

absolute_config_filename = pkg_resources.resource_filename(__name__, "config.cfg")
config = Config().from_disk(absolute_config_filename)
vector_nlps_config_dict = config["vector_nlps"]
model_names_to_nlps = {}
MODEL_NAMES_TO_SEMANTIC_ANALYZERS = {}
nlp_lock = Lock()
pipeline_components_lock = Lock()


def get_nlp(model_name: str) -> Language:
    with nlp_lock:
        if model_name not in model_names_to_nlps:
            if model_name.endswith("_trf"):
                model_names_to_nlps[model_name] = spacy.load(
                    model_name,
                    config={
                        "components.transformer.model.tokenizer_config.use_fast": False
                    },
                )
            else:
                model_names_to_nlps[model_name] = spacy.load(model_name)
        return model_names_to_nlps[model_name]


def get_semantic_analyzer(nlp: Language) -> SemanticAnalyzer:
    model_name = "_".join((nlp.meta["lang"], nlp.meta["name"]))
    vectors_nlp = (
        get_nlp(vector_nlps_config_dict[model_name])
        if model_name in vector_nlps_config_dict
        else nlp
    )
    with nlp_lock:
        if model_name not in MODEL_NAMES_TO_SEMANTIC_ANALYZERS:
            MODEL_NAMES_TO_SEMANTIC_ANALYZERS[
                model_name
            ] = SemanticAnalyzerFactory().semantic_analyzer(
                nlp=nlp, vectors_nlp=vectors_nlp
            )
        return MODEL_NAMES_TO_SEMANTIC_ANALYZERS[model_name]


class Manager:
    """The facade class for the Holmes library.

    Parameters:

    model -- the name of the spaCy model, e.g. *en_core_web_trf*
    overall_similarity_threshold -- the overall similarity threshold for embedding-based
        matching. Defaults to *1.0*, which deactivates embedding-based matching. Note that this
        parameter is not relevant for topic matching, where the thresholds for embedding-based
        matching are set on the call to *topic_match_documents_against*.
    embedding_based_matching_on_root_words -- determines whether or not embedding-based
        matching should be attempted on search-phrase root tokens, which has a considerable
        performance hit. Defaults to *False*. Note that this parameter is not relevant for topic
        matching.
    ontology -- an *Ontology* object. Defaults to *None* (no ontology).
    analyze_derivational_morphology -- *True* if matching should be attempted between different
        words from the same word family. Defaults to *True*.
    perform_coreference_resolution -- *True* if coreference resolution should be taken into account
        when matching. Defaults to *True*.
    use_reverse_dependency_matching -- *True* if appropriate dependencies in documents can be
        matched to dependencies in search phrases where the two dependencies point in opposite
        directions. Defaults to *True*.
    number_of_workers -- the number of worker processes to use, or *None* if the number of worker
        processes should depend on the number of available cores. Defaults to *None*
    verbose -- a boolean value specifying whether multiprocessing messages should be outputted to
        the console. Defaults to *False*
    """

    def __init__(
        self,
        model: str,
        *,
        overall_similarity_threshold: float = 1.0,
        embedding_based_matching_on_root_words: bool = False,
        ontology=None,
        analyze_derivational_morphology: bool = True,
        perform_coreference_resolution: bool = True,
        use_reverse_dependency_matching: bool = True,
        number_of_workers: int = None,
        verbose: bool = False
    ):
        self.verbose = verbose
        self.nlp = get_nlp(model)
        with pipeline_components_lock:
            if not self.nlp.has_pipe("coreferee"):
                self.nlp.add_pipe("coreferee")
            if not self.nlp.has_pipe("holmes"):
                self.nlp.add_pipe("holmes")
        HolmesBroker.set_extensions()
        self.semantic_analyzer = get_semantic_analyzer(self.nlp)
        if not self.semantic_analyzer.model_supports_embeddings():
            overall_similarity_threshold = 1.0
        if overall_similarity_threshold < 0.0 or overall_similarity_threshold > 1.0:
            raise ValueError("overall_similarity_threshold must be between 0.0 and 1.0")
        if (
            overall_similarity_threshold == 1.0
            and embedding_based_matching_on_root_words
        ):
            raise ValueError(
                "overall_similarity_threshold is 1.0; embedding_based_matching_on_root_words must "
                "be False"
            )
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.entity_label_to_vector_dict = (
            self.semantic_analyzer.get_entity_label_to_vector_dict()
            if self.semantic_analyzer.model_supports_embeddings()
            else {}
        )
        self.perform_coreference_resolution = perform_coreference_resolution
        self.semantic_matching_helper = (
            SemanticMatchingHelperFactory().semantic_matching_helper(
                language=self.nlp.meta["lang"]
            )
        )
        if analyze_derivational_morphology and ontology is not None:
            ontology_reverse_derivational_dict = (
                self.semantic_analyzer.get_ontology_reverse_derivational_dict(ontology)
            )
        else:
            ontology_reverse_derivational_dict = None
        self.semantic_matching_helper.main_word_matching_strategies.append(
            DirectWordMatchingStrategy(
                self.semantic_matching_helper, self.perform_coreference_resolution
            )
        )
        if analyze_derivational_morphology:
            self.semantic_matching_helper.main_word_matching_strategies.append(
                DerivationWordMatchingStrategy(
                    self.semantic_matching_helper, self.perform_coreference_resolution
                )
            )
        self.semantic_matching_helper.main_word_matching_strategies.append(
            EntityWordMatchingStrategy(
                self.semantic_matching_helper, self.perform_coreference_resolution
            )
        )
        if ontology is not None:
            if analyze_derivational_morphology:
                if ontology.status == 2:
                    msg = Printer()
                    msg.fail(
                        "Since Holmes v4.0, Ontology instances may no longer be shared between Manager instances. Please instantiate a separate Ontology instance for each Manager instance."
                    )
                    raise OntologyObjectSharedBetweenManagersError("status == 2")
                self.semantic_analyzer.update_ontology(ontology)
            self.semantic_matching_helper.ontology_word_matching_strategies.append(
                OntologyWordMatchingStrategy(
                    self.semantic_matching_helper,
                    self.perform_coreference_resolution,
                    ontology,
                    analyze_derivational_morphology,
                    ontology_reverse_derivational_dict,
                )
            )
        if overall_similarity_threshold < 1.0:
            self.semantic_matching_helper.embedding_word_matching_strategies.append(
                EmbeddingWordMatchingStrategy(
                    self.semantic_matching_helper,
                    self.perform_coreference_resolution,
                    overall_similarity_threshold,
                    overall_similarity_threshold,
                )
            )
            self.semantic_matching_helper.embedding_word_matching_strategies.append(
                EntityEmbeddingWordMatchingStrategy(
                    self.semantic_matching_helper,
                    self.perform_coreference_resolution,
                    overall_similarity_threshold,
                    overall_similarity_threshold,
                    self.entity_label_to_vector_dict,
                )
            )

        self.overall_similarity_threshold = overall_similarity_threshold
        self.use_reverse_dependency_matching = use_reverse_dependency_matching
        self.linguistic_object_factory = LinguisticObjectFactory(
            self.semantic_analyzer,
            self.semantic_matching_helper,
            overall_similarity_threshold,
            embedding_based_matching_on_root_words,
            analyze_derivational_morphology,
            perform_coreference_resolution,
            ontology,
            ontology_reverse_derivational_dict,
        )
        self.structural_matcher = StructuralMatcher(
            self.semantic_matching_helper,
            embedding_based_matching_on_root_words,
            analyze_derivational_morphology,
            perform_coreference_resolution,
            use_reverse_dependency_matching,
        )
        self.document_labels_to_worker_queues: Dict[str, int] = {}
        self.search_phrases: List[SearchPhrase] = []
        for (
            phraselet_template
        ) in self.semantic_matching_helper.local_phraselet_templates:
            phraselet_template.template_doc = self.semantic_analyzer.parse(
                phraselet_template.template_sentence
            )
            if (
                next(phraselet_template.template_doc.sents).root.i
                == phraselet_template.child_index
            ):
                raise RuntimeError(
                    "Child index of phraselet template may not point to syntactic root."
                )
        if number_of_workers is None:
            number_of_workers = cpu_count()
        elif number_of_workers <= 0:
            raise ValueError("number_of_workers must be a positive integer.")
        self.number_of_workers = number_of_workers
        self.next_worker_to_use = 0
        self.multiprocessing_manager = MultiprocessingManager()
        self.worker = (
            Worker()
        )  # will be copied to worker processes by value (Windows) or
        # by reference (Linux)
        self.workers: List[Process] = []
        self.input_queues: List[Queue] = []
        self.word_dictionaries_need_rebuilding = False
        self.words_to_corpus_frequencies: Dict[str, int] = {}
        self.maximum_corpus_frequency = 0

        for counter in range(0, self.number_of_workers):
            input_queue: Queue = Queue()
            self.input_queues.append(input_queue)
            worker_label = " ".join(("Worker", str(counter)))
            this_worker = Process(
                target=self.worker.listen,
                args=(
                    self.structural_matcher,
                    self.overall_similarity_threshold,
                    self.entity_label_to_vector_dict,
                    self.nlp.vocab,
                    self.semantic_analyzer.get_model_name(),
                    SERIALIZED_DOCUMENT_VERSION,
                    input_queue,
                    worker_label,
                ),
                daemon=True,
            )
            self.workers.append(this_worker)
            this_worker.start()
        self.lock = Lock()

    def _next_worker_queue_number(self) -> int:
        """Must be called with 'self.lock'."""
        self.next_worker_to_use += 1
        if self.next_worker_to_use == self.number_of_workers:
            self.next_worker_to_use = 0
        return self.next_worker_to_use

    def _handle_response(
        self, reply_queue, number_of_messages: int, method_name: str
    ) -> List[Any]:
        return_values = []
        exception_worker_label = None
        for _ in range(number_of_messages):
            worker_label, return_value, return_info = reply_queue.get(
                timeout=TIMEOUT_SECONDS
            )
            if isinstance(
                return_info,
                (WrongModelDeserializationError, WrongVersionDeserializationError),
            ):
                raise return_info
            elif isinstance(return_info, Exception):
                if exception_worker_label is None:
                    exception_worker_label = worker_label
            else:
                return_values.append(return_value)
                if self.verbose:
                    with self.lock:
                        print(return_info)
        if exception_worker_label is not None:
            with self.lock:
                print(
                    "".join(
                        (
                            "ERROR executing ",
                            method_name,
                            "() on ",
                            exception_worker_label,
                            ". Please examine the output from the worker processes to identify the problem.",
                        )
                    )
                )
        return return_values

    def register_serialized_documents(
        self, document_dictionary: Dict[str, bytes]
    ) -> None:
        """Note that this function is the most efficient way of loading documents.

        Parameters:

        document_dictionary -- a dictionary from labels to serialized documents.
        """
        reply_queue = self.multiprocessing_manager.Queue()
        with self.lock:
            for label, serialized_doc in document_dictionary.items():
                if label in self.document_labels_to_worker_queues:
                    raise DuplicateDocumentError(label)
                else:
                    worker_queue_number = self._next_worker_queue_number()
                    self.document_labels_to_worker_queues[label] = worker_queue_number
                    self.word_dictionaries_need_rebuilding = True
                    self.input_queues[worker_queue_number].put(
                        (
                            self.worker.register_serialized_document,
                            (serialized_doc, label),
                            reply_queue,
                        ),
                        timeout=TIMEOUT_SECONDS,
                    )
        self._handle_response(
            reply_queue, len(document_dictionary), "register_serialized_documents"
        )

    def register_serialized_document(
        self, serialized_document: bytes, label: str
    ) -> None:
        """
        Parameters:

        document -- a preparsed Holmes document.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases involving single documents (typically user entries).
        """
        self.register_serialized_documents({label: serialized_document})

    def parse_and_register_document(self, document_text: str, label: str = "") -> None:
        """Parameters:

        document_text -- the raw document text.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases involving single documents (typically user entries).
        """

        doc = self.nlp(document_text)
        self.register_serialized_document(doc.to_bytes(), label)

    def remove_document(self, label: str) -> None:
        """Parameters:

        label -- the label of the document to be removed.
        """
        reply_queue = self.multiprocessing_manager.Queue()
        with self.lock:
            if label in self.document_labels_to_worker_queues:
                self.input_queues[self.document_labels_to_worker_queues[label]].put(
                    (self.worker.remove_document, (label,), reply_queue),
                    timeout=TIMEOUT_SECONDS,
                )
                del self.document_labels_to_worker_queues[label]
                self.word_dictionaries_need_rebuilding = True
            else:
                return
        self._handle_response(reply_queue, 1, "remove_document")

    def remove_all_documents(self, labels_starting: str = None) -> None:
        """
        Parameters:

        labels_starting -- a string starting the labels of documents to be removed,
        or *None* if all documents are to be removed.
        """
        if labels_starting is None:
            labels_starting = ""
        reply_queue = self.multiprocessing_manager.Queue()
        with self.lock:
            for worker_index in range(self.number_of_workers):
                self.input_queues[worker_index].put(
                    (self.worker.remove_all_documents, (labels_starting,), reply_queue),
                    timeout=TIMEOUT_SECONDS,
                )
            self.word_dictionaries_need_rebuilding = True
            self.document_labels_to_worker_queues = {
                key: value
                for key, value in self.document_labels_to_worker_queues.items()
                if not key.startswith(labels_starting)
            }
        self._handle_response(
            reply_queue, self.number_of_workers, "remove_all_documents"
        )

    def list_document_labels(self) -> List[str]:
        """Returns a list of the labels of the currently registered documents."""
        with self.lock:
            unsorted_labels = self.document_labels_to_worker_queues.keys()
        return sorted(unsorted_labels)

    def serialize_document(self, label: str) -> Optional[bytes]:
        """Returns a serialized representation of a Holmes document that can be persisted to
            a file. If *label* is not the label of a registered document, *None* is returned
            instead.

        Parameters:

        label -- the label of the document to be serialized.
        """
        reply_queue = self.multiprocessing_manager.Queue()
        with self.lock:
            if label in self.document_labels_to_worker_queues:
                self.input_queues[self.document_labels_to_worker_queues[label]].put(
                    (self.worker.get_serialized_document, (label,), reply_queue),
                    timeout=TIMEOUT_SECONDS,
                )
            else:
                return None
        return self._handle_response(reply_queue, 1, "serialize_document")[0]

    def get_document(self, label: str = "") -> Optional[Doc]:
        """Returns a Holmes document. If *label* is not the label of a registered document, *None*
            is returned instead.

        Parameters:

        label -- the label of the document to be serialized.
        """
        serialized_document = self.serialize_document(label)
        return (
            None
            if serialized_document is None
            else Doc(self.nlp.vocab).from_bytes(serialized_document)
        )

    def debug_document(self, label: str = "") -> None:
        """Outputs a debug representation for a loaded document."""
        serialized_document = self.serialize_document(label)
        if serialized_document is not None:
            doc = Doc(self.nlp.vocab).from_bytes(serialized_document)
            self.semantic_analyzer.debug_structures(doc)
        else:
            print("No document with label", label)

    def _create_search_phrase(self, search_phrase_text: str, label: Optional[str]):
        if label is None:
            label = search_phrase_text
        search_phrase_doc = self.nlp(search_phrase_text)
        search_phrase = self.linguistic_object_factory.create_search_phrase(
            search_phrase_text,
            search_phrase_doc,
            label,
            None,
            False,
            False,
            False,
            False,
        )
        return search_phrase

    def register_search_phrase(
        self, search_phrase_text: str, label: str = None
    ) -> SearchPhrase:
        """Registers and returns a new search phrase.

        Parameters:

        search_phrase_text -- the raw search phrase text.
        label -- a label for the search phrase which need *not* be unique. Defaults to the raw
            search phrase text.
        """
        search_phrase = self._create_search_phrase(search_phrase_text, label)
        search_phrase.pack()
        reply_queue = self.multiprocessing_manager.Queue()
        with self.lock:
            for worker_index in range(self.number_of_workers):
                self.input_queues[worker_index].put(
                    (self.worker.register_search_phrase, (search_phrase,), reply_queue),
                    timeout=TIMEOUT_SECONDS,
                )
            self.search_phrases.append(search_phrase)
        self._handle_response(
            reply_queue, self.number_of_workers, "register_search_phrase"
        )
        return search_phrase

    def remove_all_search_phrases_with_label(self, label: str) -> None:
        reply_queue = self.multiprocessing_manager.Queue()
        with self.lock:
            for worker_index in range(self.number_of_workers):
                self.input_queues[worker_index].put(
                    (
                        self.worker.remove_all_search_phrases_with_label,
                        (label,),
                        reply_queue,
                    ),
                    timeout=TIMEOUT_SECONDS,
                )
            self.search_phrases = [
                search_phrase
                for search_phrase in self.search_phrases
                if search_phrase.label != label
            ]
        self._handle_response(
            reply_queue, self.number_of_workers, "remove_all_search_phrases_with_label"
        )

    def remove_all_search_phrases(self) -> None:
        reply_queue = self.multiprocessing_manager.Queue()
        with self.lock:
            for worker_index in range(self.number_of_workers):
                self.input_queues[worker_index].put(
                    (self.worker.remove_all_search_phrases, None, reply_queue),
                    timeout=TIMEOUT_SECONDS,
                )
            self.search_phrases = []
        self._handle_response(
            reply_queue, self.number_of_workers, "remove_all_search_phrases"
        )

    def list_search_phrase_labels(self) -> List[str]:
        with self.lock:
            return sorted(
                list({search_phrase.label for search_phrase in self.search_phrases})
            )

    def match(
        self, search_phrase_text: str = None, document_text: str = None
    ) -> List[Dict]:
        """Matches search phrases to documents and returns the result as match dictionaries.

        Parameters:

        search_phrase_text -- a text from which to generate a search phrase, or *None* if the
            preloaded search phrases should be used for matching.
        document_text -- a text from which to generate a document, or *None* if the preloaded
            documents should be used for matching.
        """

        if search_phrase_text is not None:
            search_phrase = self._create_search_phrase(search_phrase_text, "")
        elif len(self.list_search_phrase_labels()) == 0:
            raise NoSearchPhraseError(
                "At least one search phrase is required for matching."
            )
        else:
            search_phrase = None
        if document_text is not None:
            serialized_document = self.nlp(document_text).to_bytes()
            with self.lock:
                worker_indexes = {self._next_worker_queue_number()}
        else:
            with self.lock:
                if len(self.document_labels_to_worker_queues) == 0:
                    raise NoDocumentError(
                        "At least one document is required for matching."
                    )
                worker_indexes = set(self.document_labels_to_worker_queues.values())
            serialized_document = None
        reply_queue = self.multiprocessing_manager.Queue()
        for worker_index in worker_indexes:
            self.input_queues[worker_index].put(
                (self.worker.match, (serialized_document, search_phrase), reply_queue),
                timeout=TIMEOUT_SECONDS,
            )
        worker_match_dicts_lists = self._handle_response(
            reply_queue, len(worker_indexes), "match"
        )
        match_dicts = []
        for worker_match_dicts in worker_match_dicts_lists:
            match_dicts.extend(worker_match_dicts)
        return sorted(
            match_dicts,
            key=lambda match_dict: (
                1 - float(match_dict["overall_similarity_measure"]),
                match_dict["document"],
            ),
        )

    def get_corpus_frequency_information(self):
        def merge_dicts_adding_common_values(dict1, dict2):
            dict_to_return = {**dict1, **dict2}
            for key in dict_to_return:
                if key in dict1 and key in dict2:
                    dict_to_return[key] = dict1[key] + dict2[key]
            return dict_to_return

        with self.lock:
            if self.word_dictionaries_need_rebuilding:
                reply_queue = self.multiprocessing_manager.Queue()
                worker_frequency_dict = {}
                worker_indexes = set(self.document_labels_to_worker_queues.values())
                for worker_index in worker_indexes:
                    self.input_queues[worker_index].put(
                        (
                            self.worker.get_words_to_corpus_frequencies,
                            None,
                            reply_queue,
                        ),
                        timeout=TIMEOUT_SECONDS,
                    )
                exception_worker_label = None
                for _ in range(len(worker_indexes)):
                    worker_label, return_value, return_info = reply_queue.get(
                        timeout=TIMEOUT_SECONDS
                    )
                    if isinstance(return_info, Exception):
                        if exception_worker_label is None:
                            exception_worker_label = worker_label
                    else:
                        worker_frequency_dict = merge_dicts_adding_common_values(
                            worker_frequency_dict, return_value
                        )
                        if self.verbose:
                            print(return_info)
                    if exception_worker_label is not None:
                        print(
                            "".join(
                                (
                                    "ERROR executing get_words_to_corpus_frequencies() on ",
                                    exception_worker_label,
                                    ". Please examine the output from the worker processes to identify the problem.",
                                )
                            )
                        )
                self.words_to_corpus_frequencies = {}
                for word in worker_frequency_dict:
                    if word in self.words_to_corpus_frequencies:
                        self.words_to_corpus_frequencies[word] += worker_frequency_dict[
                            word
                        ]
                    else:
                        self.words_to_corpus_frequencies[word] = worker_frequency_dict[
                            word
                        ]
                self.maximum_corpus_frequency = max(
                    self.words_to_corpus_frequencies.values()
                )
                self.word_dictionaries_need_rebuilding = False
            return self.words_to_corpus_frequencies, self.maximum_corpus_frequency

    def topic_match_documents_against(
        self,
        text_to_match: str,
        *,
        use_frequency_factor: bool = True,
        maximum_activation_distance: int = 75,
        word_embedding_match_threshold: float = 0.8,
        initial_question_word_embedding_match_threshold: float = 0.7,
        relation_score: int = 300,
        reverse_only_relation_score: int = 200,
        single_word_score: int = 50,
        single_word_any_tag_score: int = 20,
        initial_question_word_answer_score: int = 600,
        initial_question_word_behaviour: Literal["process", "exclusive", "ignore"] = "process",
        different_match_cutoff_score: int = 15,
        overlapping_relation_multiplier: float = 1.5,
        embedding_penalty: float = 0.6,
        ontology_penalty: float = 0.9,
        relation_matching_frequency_threshold: float = 0.25,
        embedding_matching_frequency_threshold: float = 0.5,
        sideways_match_extent: int = 100,
        only_one_result_per_document: bool = False,
        number_of_results: int = 10,
        document_label_filter: str = None,
        tied_result_quotient: float = 0.9
    ) -> List[Dict]:

        """Returns a list of dictionaries representing the results of a topic match between an
        entered text and the loaded documents.

        Properties:

        text_to_match -- the text to match against the loaded documents.
        use_frequency_factor -- *True* if scores should be multiplied by a factor between 0 and 1
            expressing how rare the words matching each phraselet are in the corpus. Note that,
            even if this parameter is set to *False*, the factors are still calculated as they 
            are required for determining which relation and embedding matches should be attempted.
        maximum_activation_distance -- the number of words it takes for a previous phraselet
            activation to reduce to zero when the library is reading through a document.
        word_embedding_match_threshold -- the cosine similarity above which two words match where
          the search phrase word does not govern an interrogative pronoun..
        initial_question_word_embedding_match_threshold -- the cosine similarity above which two
            words match where the search phrase word governs an interrogative pronoun.
        relation_score -- the activation score added when a normal two-word
            relation is matched.
        reverse_only_relation_score -- the activation score added when a two-word relation
                is matched using a search phrase that can only be reverse-matched.
        single_word_score -- the activation score added when a single noun is matched.
        single_word_any_tag_score -- the activation score added when a single word is matched
            that is not a noun.
        initial_question_word_answer_score -- the activation score added when a question word is
            matched to an answering phrase.
        initial_question_word_behaviour -- 'process' if a question word in the sentence
            constituent at the beginning of *text_to_match* is to be matched to document phrases
            that answer it and to matching question words; 'exclusive' if only topic matches that 
            answer questions are to be permitted; 'ignore' if question words are to be ignored.
        different_match_cutoff_score -- the activation threshold under which topic matches are
            separated from one another. Note that the default value will probably be too low if
            *use_frequency_factor* is set to *False*.
        overlapping_relation_multiplier -- the value by which the activation score is multiplied
            when two relations were matched and the matches involved a common document word.
        embedding_penalty -- a value between 0 and 1 with which scores are multiplied when the
            match involved an embedding. The result is additionally multiplied by the overall
            similarity measure of the match.
        ontology_penalty -- a value between 0 and 1 with which scores are multiplied for each
            word match within a match that involved the ontology. For each such word match,
            the score is multiplied by the value (abs(depth) + 1) times, so that the penalty is
            higher for hyponyms and hypernyms than for synonyms and increases with the
            depth distance.
        relation_matching_frequency_threshold -- the frequency threshold above which single
            word matches are used as the basis for attempting relation matches.
        embedding_matching_frequency_threshold -- the frequency threshold above which single
            word matches are used as the basis for attempting relation matches with
            embedding-based matching on the second word.
        sideways_match_extent -- the maximum number of words that may be incorporated into a
            topic match either side of the word where the activation peaked.
        only_one_result_per_document -- if 'True', prevents multiple results from being returned
            for the same document.
        number_of_results -- the number of topic match objects to return.
        document_label_filter -- optionally, a string with which document labels must start to
            be considered for inclusion in the results.
        tied_result_quotient -- the quotient between a result and following results above which
            the results are interpreted as tied.
        """
        if word_embedding_match_threshold < 0.0 or word_embedding_match_threshold > 1.0:
            raise ValueError("word_embedding_match_threshold must be between 0 and 1")
        if (
            initial_question_word_embedding_match_threshold < 0.0
            or initial_question_word_embedding_match_threshold > 1.0
        ):
            raise ValueError(
                "initial_question_word_embedding_match_threshold must be between 0 and 1"
            )

        if not self.semantic_analyzer.model_supports_embeddings():
            word_embedding_match_threshold = (
                initial_question_word_embedding_match_threshold
            ) = 1.0

        overall_similarity_threshold = sqrt(word_embedding_match_threshold)
        initial_question_word_overall_similarity_threshold = sqrt(
            initial_question_word_embedding_match_threshold
        )

        if initial_question_word_behaviour not in ("process", "exclusive", "ignore"):
            raise ValueError(
                ": ".join(
                    ("initial_question_word_behaviour", initial_question_word_behaviour)
                )
            )
        if (
            embedding_matching_frequency_threshold < 0.0
            or embedding_matching_frequency_threshold > 1.0
        ):
            raise ValueError(
                ": ".join(
                    (
                        "embedding_matching_frequency_threshold",
                        str(embedding_matching_frequency_threshold),
                    )
                )
            )
        if (
            relation_matching_frequency_threshold < 0.0
            or relation_matching_frequency_threshold > 1.0
        ):
            raise ValueError(
                ": ".join(
                    (
                        "relation_matching_frequency_threshold",
                        str(relation_matching_frequency_threshold),
                    )
                )
            )
        if (
            embedding_matching_frequency_threshold
            < relation_matching_frequency_threshold
        ):
            raise EmbeddingThresholdLessThanRelationThresholdError(
                " ".join(
                    (
                        "embedding",
                        str(embedding_matching_frequency_threshold),
                        "relation",
                        str(relation_matching_frequency_threshold),
                    )
                )
            )
        with self.lock:
            if len(self.document_labels_to_worker_queues) == 0:
                raise NoDocumentError("At least one document is required for matching.")
        (
            words_to_corpus_frequencies,
            maximum_corpus_frequency,
        ) = self.get_corpus_frequency_information()

        reply_queue = self.multiprocessing_manager.Queue()
        text_to_match_doc = self.semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_phraselet_infos = (
            self.linguistic_object_factory.get_phraselet_labels_to_phraselet_infos(
                text_to_match_doc=text_to_match_doc,
                words_to_corpus_frequencies=words_to_corpus_frequencies,
                maximum_corpus_frequency=maximum_corpus_frequency,
                process_initial_question_words=initial_question_word_behaviour
                in ("process", "exclusive"),
            )
        )
        if len(phraselet_labels_to_phraselet_infos) == 0:
            return []
        phraselet_labels_to_search_phrases = (
            self.linguistic_object_factory.create_search_phrases_from_phraselet_infos(
                list(phraselet_labels_to_phraselet_infos.values()),
                relation_matching_frequency_threshold,
            )
        )
        for search_phrase in phraselet_labels_to_search_phrases.values():
            search_phrase.pack()

        worker_indexes = set(self.document_labels_to_worker_queues.values())
        for worker_index in worker_indexes:
            self.input_queues[worker_index].put(
                (
                    self.worker.get_topic_matches,
                    (
                        text_to_match,
                        phraselet_labels_to_phraselet_infos,
                        phraselet_labels_to_search_phrases,
                        maximum_activation_distance,
                        overall_similarity_threshold,
                        initial_question_word_overall_similarity_threshold,
                        relation_score,
                        reverse_only_relation_score,
                        single_word_score,
                        single_word_any_tag_score,
                        initial_question_word_answer_score,
                        initial_question_word_behaviour,
                        different_match_cutoff_score,
                        overlapping_relation_multiplier,
                        embedding_penalty,
                        ontology_penalty,
                        relation_matching_frequency_threshold,
                        embedding_matching_frequency_threshold,
                        sideways_match_extent,
                        only_one_result_per_document,
                        number_of_results,
                        document_label_filter,
                        use_frequency_factor,
                    ),
                    reply_queue,
                ),
                timeout=TIMEOUT_SECONDS,
            )
        worker_topic_match_dictss = self._handle_response(
            reply_queue, len(worker_indexes), "match"
        )
        topic_match_dicts = []
        for worker_topic_match_dicts in worker_topic_match_dictss:
            if worker_topic_match_dicts is not None:
                topic_match_dicts.extend(worker_topic_match_dicts)
        return TopicMatchDictionaryOrderer().order(
            topic_match_dicts, number_of_results, tied_result_quotient
        )

    def get_supervised_topic_training_basis(
        self,
        *,
        classification_ontology: Ontology = None,
        overlap_memory_size: int = 10,
        one_hot: bool = True,
        match_all_words: bool = False,
        verbose: bool = True
    ) -> SupervisedTopicTrainingBasis:
        """Returns an object that is used to train and generate a model for the supervised
        document classification use case.

        Parameters:

        classification_ontology -- an Ontology object incorporating relationships between
            classification labels, or *None* if no such ontology is to be used.
        overlap_memory_size -- how many non-word phraselet matches to the left should be
            checked for words in common with a current match.
        one_hot -- whether the same word or relationship matched multiple times should be
            counted once only (value 'True') or multiple times (value 'False')
        match_all_words -- whether all single words should be taken into account
            (value 'True') or only single words with noun tags (value 'False')
        verbose -- if 'True', information about training progress is outputted to the console.
        """
        return SupervisedTopicTrainingBasis(
            linguistic_object_factory=self.linguistic_object_factory,
            structural_matcher=self.structural_matcher,
            classification_ontology=classification_ontology,
            overlap_memory_size=overlap_memory_size,
            one_hot=one_hot,
            match_all_words=match_all_words,
            overall_similarity_threshold=self.overall_similarity_threshold,
            verbose=verbose,
        )

    def deserialize_supervised_topic_classifier(
        self, serialized_model: bytes, verbose: bool = False
    ) -> SupervisedTopicClassifier:
        """Returns a classifier for the supervised document classification use case
        that will use a supplied pre-trained model.

        Parameters:

        serialized_model -- the pre-trained model, which will correspond to a
            'SupervisedTopicClassifierModel' instance.
        verbose -- if 'True', information about matching is outputted to the console.
        """
        model = pickle.loads(serialized_model)
        return SupervisedTopicClassifier(
            self.semantic_analyzer,
            self.linguistic_object_factory,
            self.structural_matcher,
            model,
            self.overall_similarity_threshold,
            verbose,
        )

    def start_chatbot_mode_console(self):
        """Starts a chatbot mode console enabling the matching of pre-registered
        search phrases to documents (chatbot entries) ad-hoc by the user.
        """
        holmes_consoles = HolmesConsoles(self)
        holmes_consoles.start_chatbot_mode()

    def start_structural_extraction_mode_console(self):
        """Starts a structural extraction mode console enabling the matching of pre-registered
        documents to search phrases entered ad-hoc by the user.
        """
        holmes_consoles = HolmesConsoles(self)
        holmes_consoles.start_structural_extraction_mode()

    def start_topic_matching_search_mode_console(
        self,
        only_one_result_per_document: bool = False,
        word_embedding_match_threshold: float = 0.8,
        initial_question_word_embedding_match_threshold: float = 0.7,
    ):
        """Starts a topic matching search mode console enabling the matching of pre-registered
        documents to query phrases entered ad-hoc by the user.

        Parameters:

        only_one_result_per_document -- if 'True', prevents multiple topic match
            results from being returned for the same document.
        word_embedding_match_threshold -- the cosine similarity above which two words match
            where the search phrase word does not govern an interrogative pronoun.
        initial_question_word_embedding_match_threshold -- the cosine similarity above which two
            words match where the search phrase word governs an interrogative pronoun.
        """
        holmes_consoles = HolmesConsoles(self)
        holmes_consoles.start_topic_matching_search_mode(
            only_one_result_per_document,
            word_embedding_match_threshold,
            initial_question_word_embedding_match_threshold,
        )

    def close(self) -> None:
        """Terminates the worker processes."""
        for worker in self.workers:
            worker.terminate()


class Worker:
    """Worker implementation used by *Manager*."""

    def listen(
        self,
        structural_matcher,
        overall_similarity_threshold,
        entity_label_to_vector_dict,
        vocab,
        model_name,
        serialized_document_version,
        input_queue,
        worker_label,
    ):
        state = {
            "structural_matcher": structural_matcher,
            "word_matching_strategies": structural_matcher.semantic_matching_helper.main_word_matching_strategies
            + structural_matcher.semantic_matching_helper.ontology_word_matching_strategies
            + structural_matcher.semantic_matching_helper.embedding_word_matching_strategies,
            "overall_similarity_threshold": overall_similarity_threshold,
            "entity_label_to_vector_dict": entity_label_to_vector_dict,
            "vocab": vocab,
            "model_name": model_name,
            "serialized_document_version": serialized_document_version,
            "document_labels_to_documents": {},
            "reverse_dict": {},
            "search_phrases": [],
        }
        HolmesBroker.set_extensions()
        while True:
            method, args, reply_queue = input_queue.get()
            try:
                if args is not None:
                    return_value, return_info = method(state, *args)
                else:
                    return_value, return_info = method(state)
                reply_queue.put(
                    (worker_label, return_value, return_info), timeout=TIMEOUT_SECONDS
                )
            except Exception as err:
                print("Exception calling", method)
                print("String arguments:", [arg for arg in args if type(arg) == str])
                print(worker_label, " - error:")
                print(traceback.format_exc())
                reply_queue.put((worker_label, None, err), timeout=TIMEOUT_SECONDS)
            except:
                print("Exception calling", str(method))
                print("String arguments:", [arg for arg in args if type(arg) == str])
                print(worker_label, " - error:")
                print(traceback.format_exc())
                err_identifier = str(sys.exc_info()[0])
                reply_queue.put(
                    (worker_label, None, err_identifier), timeout=TIMEOUT_SECONDS
                )

    def load_document(self, state, serialized_doc, document_label, reverse_dict):
        doc = Doc(state["vocab"]).from_bytes(serialized_doc)
        if doc._.holmes_document_info.model != state["model_name"]:
            raise WrongModelDeserializationError(
                "; ".join((state["model_name"], doc._.holmes_document_info.model))
            )
        if (
            doc._.holmes_document_info.serialized_document_version
            != state["serialized_document_version"]
        ):
            raise WrongVersionDeserializationError(
                "; ".join(
                    (
                        str(state["serialized_document_version"]),
                        str(doc._.holmes_document_info.serialized_document_version),
                    )
                )
            )
        state["document_labels_to_documents"][document_label] = doc
        state["structural_matcher"].semantic_matching_helper.add_to_reverse_dict(
            reverse_dict, doc, document_label
        )
        return doc

    def register_serialized_document(self, state, serialized_doc, document_label):
        self.load_document(state, serialized_doc, document_label, state["reverse_dict"])
        return None, " ".join(("Registered document", document_label))

    def remove_document(self, state, document_label):
        state["document_labels_to_documents"].pop(document_label)
        state["reverse_dict"] = state[
            "structural_matcher"
        ].semantic_matching_helper.get_reverse_dict_removing_document(
            state["reverse_dict"], document_label
        )
        return None, " ".join(("Removed document", document_label))

    def remove_all_documents(self, state, labels_starting):
        if len(labels_starting) == 0:
            state["document_labels_to_documents"] = {}
            state["reverse_dict"] = {}
            return None, "Removed all documents"
        else:
            labels_to_remove = [
                label
                for label in state["document_labels_to_documents"].keys()
                if label.startswith(labels_starting)
            ]
            state["document_labels_to_documents"] = {
                key: value
                for key, value in state["document_labels_to_documents"].items()
                if key not in labels_to_remove
            }
            for label_to_remove in labels_to_remove:
                state["reverse_dict"] = state[
                    "structural_matcher"
                ].semantic_matching_helper.get_reverse_dict_removing_document(
                    state["reverse_dict"], label_to_remove
                )
            return None, " ".join(
                ("Removed all documents with labels beginning", labels_starting)
            )

    def get_serialized_document(self, state, label):
        if label in state["document_labels_to_documents"]:
            return state["document_labels_to_documents"][label].to_bytes(), " ".join(
                ("Returned serialized document with label", label)
            )
        else:
            return None, " ".join(("No document found with label", label))

    def register_search_phrase(self, state, search_phrase):
        search_phrase.unpack(state["vocab"])
        state["search_phrases"].append(search_phrase)
        return None, " ".join(
            ("Registered search phrase with label", search_phrase.label)
        )

    def remove_all_search_phrases_with_label(self, state, label):
        state["search_phrases"] = [
            search_phrase
            for search_phrase in state["search_phrases"]
            if search_phrase.label != label
        ]
        return None, " ".join(("Removed all search phrases with label '", label, "'"))

    def remove_all_search_phrases(self, state):
        state["search_phrases"] = []
        return None, "Removed all search phrases"

    def get_words_to_corpus_frequencies(self, state):
        words_to_corpus_frequencies = {}
        for word, cwps in state["reverse_dict"].items():
            if word in punctuation:
                continue
            if word in words_to_corpus_frequencies:
                words_to_corpus_frequencies[word] += len(set(cwps))
            else:
                words_to_corpus_frequencies[word] = len(set(cwps))
        return words_to_corpus_frequencies, "Retrieved words to corpus frequencies"

    def match(self, state, serialized_doc, search_phrase):
        if serialized_doc is not None:
            reverse_dict = {}
            doc = self.load_document(state, serialized_doc, "", reverse_dict)
            document_labels_to_documents = {"": doc}
        else:
            reverse_dict = state["reverse_dict"]
            document_labels_to_documents = state["document_labels_to_documents"]
        search_phrases = (
            [search_phrase] if search_phrase is not None else state["search_phrases"]
        )
        if len(document_labels_to_documents) > 0 and len(search_phrases) > 0:
            matches = state["structural_matcher"].match(
                word_matching_strategies=state["word_matching_strategies"],
                document_labels_to_documents=document_labels_to_documents,
                reverse_dict=reverse_dict,
                search_phrases=search_phrases,
                match_depending_on_single_words=None,
                compare_embeddings_on_root_words=state[
                    "structural_matcher"
                ].embedding_based_matching_on_root_words,
                compare_embeddings_on_non_root_words=True,
                reverse_matching_cwps=None,
                embedding_reverse_matching_cwps=None,
                process_initial_question_words=False,
                overall_similarity_threshold=state["overall_similarity_threshold"],
                initial_question_word_overall_similarity_threshold=1.0,
            )
            return (
                state["structural_matcher"].build_match_dictionaries(matches),
                "Returned matches",
            )
        else:
            return [], "No stored objects to match against"

    def get_topic_matches(
        self,
        state,
        text_to_match,
        phraselet_labels_to_phraselet_infos,
        phraselet_labels_to_search_phrases,
        maximum_activation_distance,
        overall_similarity_threshold,
        initial_question_word_overall_similarity_threshold,
        relation_score,
        reverse_only_relation_score,
        single_word_score,
        single_word_any_tag_score,
        initial_question_word_answer_score,
        initial_question_word_behaviour,
        different_match_cutoff_score,
        overlapping_relation_multiplier,
        embedding_penalty,
        ontology_penalty,
        relation_matching_frequency_threshold,
        embedding_matching_frequency_threshold,
        sideways_match_extent,
        only_one_result_per_document,
        number_of_results,
        document_label_filter,
        use_frequency_factor,
    ):
        if len(state["document_labels_to_documents"]) == 0:
            return [], "No stored documents to match against"
        for search_phrase in phraselet_labels_to_search_phrases.values():
            search_phrase.unpack(state["vocab"])
        topic_matcher = TopicMatcher(
            structural_matcher=state["structural_matcher"],
            document_labels_to_documents=state["document_labels_to_documents"],
            reverse_dict=state["reverse_dict"],
            text_to_match=text_to_match,
            phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
            phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
            maximum_activation_distance=maximum_activation_distance,
            overall_similarity_threshold=overall_similarity_threshold,
            initial_question_word_overall_similarity_threshold=initial_question_word_overall_similarity_threshold,
            relation_score=relation_score,
            reverse_only_relation_score=reverse_only_relation_score,
            single_word_score=single_word_score,
            single_word_any_tag_score=single_word_any_tag_score,
            initial_question_word_answer_score=initial_question_word_answer_score,
            initial_question_word_behaviour=initial_question_word_behaviour,
            different_match_cutoff_score=different_match_cutoff_score,
            overlapping_relation_multiplier=overlapping_relation_multiplier,
            embedding_penalty=embedding_penalty,
            ontology_penalty=ontology_penalty,
            relation_matching_frequency_threshold=relation_matching_frequency_threshold,
            embedding_matching_frequency_threshold=embedding_matching_frequency_threshold,
            sideways_match_extent=sideways_match_extent,
            only_one_result_per_document=only_one_result_per_document,
            number_of_results=number_of_results,
            document_label_filter=document_label_filter,
            use_frequency_factor=use_frequency_factor,
            entity_label_to_vector_dict=state["entity_label_to_vector_dict"],
        )
        return (
            topic_matcher.get_topic_match_dictionaries(),
            "Returned topic match dictionaries",
        )


@Language.factory("holmes")
class HolmesBroker:
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.pid = os.getpid()
        self.semantic_analyzer = get_semantic_analyzer(nlp)
        self.set_extensions()

    def __call__(self, doc: Doc) -> Doc:
        try:
            self.semantic_analyzer.holmes_parse(doc)
        except:
            print("Unexpected error annotating document, skipping ....")
            exception_info_parts = sys.exc_info()
            print(exception_info_parts[0])
            print(exception_info_parts[1])
            traceback.print_tb(exception_info_parts[2])
        return doc

    def __getstate__(self):
        return self.nlp.meta

    def __setstate__(self, meta):
        nlp_name = "_".join((meta["lang"], meta["name"]))
        self.nlp = spacy.load(nlp_name)
        self.semantic_analyzer = get_semantic_analyzer(self.nlp)
        self.pid = os.getpid()
        HolmesBroker.set_extensions()

    @staticmethod
    def set_extensions():
        if not Doc.has_extension("coref_chains"):
            Doc.set_extension("coref_chains", default=None)
        if not Token.has_extension("coref_chains"):
            Token.set_extension("coref_chains", default=None)
        if not Doc.has_extension("holmes_document_info"):
            Doc.set_extension("holmes_document_info", default=None)
        if not Token.has_extension("holmes"):
            Token.set_extension("holmes", default=None)
