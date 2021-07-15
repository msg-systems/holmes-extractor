from multiprocessing import Process, Queue, Manager as Multiprocessing_manager, cpu_count
from threading import Lock
from string import punctuation
import traceback
import sys
import os
import jsonpickle
import pkg_resources
import spacy
import coreferee
from spacy import Language
from spacy.tokens import Doc, Token
from thinc.api import Config
from .errors import *
from .matching import StructuralMatcher
from .parsing import SemanticAnalyzerFactory, SemanticAnalyzer, SemanticMatchingHelperFactory,\
    LinguisticObjectFactory, SERIALIZED_DOCUMENT_VERSION
from .classification import SupervisedTopicTrainingBasis, SupervisedTopicClassifier
from .topic_matching import TopicMatcher, TopicMatchDictionaryOrderer
from .consoles import HolmesConsoles

TIMEOUT_SECONDS = 300

absolute_config_filename = pkg_resources.resource_filename(__name__, 'config.cfg')
config = Config().from_disk(absolute_config_filename)
vector_nlps_config_dict = config['vector_nlps']
model_names_to_nlps = {}
model_names_to_semantic_analyzers = {}
nlp_lock = Lock()
pipeline_components_lock = Lock()

def get_nlp(model_name:str) -> Language:
    with nlp_lock:
        if model_name not in model_names_to_nlps:
            if model_name.endswith('_trf'):
                model_names_to_nlps[model_name] = spacy.load(model_name,
                    config={'components.transformer.model.tokenizer_config.use_fast': False})
            else:
                model_names_to_nlps[model_name] = spacy.load(model_name)
        return model_names_to_nlps[model_name]

def get_semantic_analyzer(nlp:Language) -> SemanticAnalyzer:
    global model_names_to_semantic_analyzers
    model_name = '_'.join((nlp.meta['lang'], nlp.meta['name']))
    vectors_nlp = get_nlp(vector_nlps_config_dict[model_name]) \
        if model_name in vector_nlps_config_dict else nlp
    with nlp_lock:
        if model_name not in model_names_to_semantic_analyzers:
            model_names_to_semantic_analyzers[model_name] = \
                SemanticAnalyzerFactory().semantic_analyzer(nlp=nlp, vectors_nlp=vectors_nlp)
        return model_names_to_semantic_analyzers[model_name]

class Manager:
    """The facade class for the Holmes library.

    Parameters:

    model -- the name of the spaCy model, e.g. *en_core_web_trf*
    overall_similarity_threshold -- the overall similarity threshold for embedding-based
        matching. Defaults to *1.0*, which deactivates embedding-based matching.
    embedding_based_matching_on_root_words -- determines whether or not embedding-based
        matching should be attempted on search-phrase root tokens, which has a considerable
        performance hit. Defaults to *False*.
    ontology -- an *Ontology* object. Defaults to *None* (no ontology).
    analyze_derivational_morphology -- *True* if matching should be attempted between different
        words from the same word family. Defaults to *True*.
    perform_coreference_resolution -- *True*, *False*, or *None* if coreference resolution
        should be taken into account when matching. Defaults to *True*.
    use_reverse_dependency_matching -- *True* if appropriate dependencies in documents can be
        matched to dependencies in search phrases where the two dependencies point in opposite
        directions. Defaults to *True*.
    number_of_workers -- the number of worker processes to use, or *None* if the number of worker
        processes should depend on the number of available cores. Defaults to *None*
    verbose -- a boolean value specifying whether multiprocessing messages should be outputted to
        the console. Defaults to *False*
    """

    def __init__(
            self, model, *, overall_similarity_threshold=1.0,
            embedding_based_matching_on_root_words=False, ontology=None,
            analyze_derivational_morphology=True, perform_coreference_resolution=True,
            use_reverse_dependency_matching=True, verbose=False,
            number_of_workers:int=None):
        self.verbose = verbose
        self.nlp = get_nlp(model)
        with pipeline_components_lock:
            if not self.nlp.has_pipe('coreferee'):
                self.nlp.add_pipe('coreferee')
            if not self.nlp.has_pipe('holmes'):
                self.nlp.add_pipe('holmes')
        self.semantic_analyzer = get_semantic_analyzer(self.nlp)
        if overall_similarity_threshold < 0.0 or overall_similarity_threshold > 1.0:
            raise ValueError(
                'overall_similarity_threshold must be between 0 and 1')
        if overall_similarity_threshold != 1.0 and not \
                self.semantic_analyzer.model_supports_embeddings():
            raise ValueError(
                'Model has no embeddings: overall_similarity_threshold must be 1.')
        if overall_similarity_threshold == 1.0 and embedding_based_matching_on_root_words:
            raise ValueError(
                'overall_similarity_threshold is 1; embedding_based_matching_on_root_words must '\
                'be False')
        self.ontology = ontology
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.semantic_matching_helper = SemanticMatchingHelperFactory().semantic_matching_helper(
            language=self.nlp.meta['lang'], ontology=ontology,
            analyze_derivational_morphology=analyze_derivational_morphology)
        self.overall_similarity_threshold = overall_similarity_threshold
        self.embedding_based_matching_on_root_words = embedding_based_matching_on_root_words
        self.perform_coreference_resolution = perform_coreference_resolution
        self.use_reverse_dependency_matching = use_reverse_dependency_matching
        self.linguistic_object_factory = LinguisticObjectFactory(
            self.semantic_analyzer, self.semantic_matching_helper, ontology,
            overall_similarity_threshold, embedding_based_matching_on_root_words,
            analyze_derivational_morphology, perform_coreference_resolution)
        self.semantic_matching_helper.ontology_reverse_derivational_dict = \
            self.linguistic_object_factory.get_ontology_reverse_derivational_dict()
        self.structural_matcher = StructuralMatcher(
            self.semantic_matching_helper, ontology, overall_similarity_threshold,
            embedding_based_matching_on_root_words,
            analyze_derivational_morphology, perform_coreference_resolution,
            use_reverse_dependency_matching)
        self.document_labels_to_worker_queues = {}
        self.search_phrases = []
        HolmesBroker.set_extensions()
        for phraselet_template in self.semantic_matching_helper.phraselet_templates:
            phraselet_template.template_doc = self.semantic_analyzer.parse(
                phraselet_template.template_sentence)
        if number_of_workers is None:
            number_of_workers = cpu_count()
        elif not 0 < number_of_workers:
            raise ValueError('number_of_workers must be a postitive integer.')
        self.number_of_workers = number_of_workers
        self.next_worker_to_use = 0
        self.multiprocessor_manager = Multiprocessing_manager()
        self.worker = Worker() # will be copied to worker processes by value (Windows) or
                                # by reference (Linux)
        self.workers = []
        self.input_queues = []
        self.word_dictionaries_need_rebuilding = False
        self.words_to_corpus_frequencies = {}
        self.maximum_corpus_frequency = 0

        for counter in range(0, self.number_of_workers):
            input_queue = Queue()
            self.input_queues.append(input_queue)
            worker_label = ' '.join(('Worker', str(counter)))
            this_worker = Process(
                target=self.worker.listen, args=(
                self.structural_matcher, self.nlp.vocab, model,
                SERIALIZED_DOCUMENT_VERSION, input_queue, worker_label),
                daemon=True)
            self.workers.append(this_worker)
            this_worker.start()
        self.lock = Lock()

    def next_worker_queue_number(self):
        self.next_worker_to_use += 1
        if self.next_worker_to_use == self.number_of_workers:
            self.next_worker_to_use = 0
        return self.next_worker_to_use

    def handle_response(self, reply_queue, number_of_messages, method_name):
        return_values = []
        exception_worker_label = None
        for _ in range(number_of_messages):
            worker_label, return_value, return_info = reply_queue.get(timeout=TIMEOUT_SECONDS)
            if isinstance(return_info, WrongModelDeserializationError) or \
                    isinstance(return_info, WrongVersionDeserializationError):
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
                print(''.join(('ERROR executing ', method_name, '() on ',
                exception_worker_label,
                '. Please examine the output from the worker processes to identify the problem.')))
        return return_values

    def register_serialized_documents(self, document_dictionary):
        """Parameters:

        document -- a preparsed Holmes document.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases involving single documents (typically user entries).
        """
        reply_queue = self.multiprocessor_manager.Queue()
        with self.lock:
            for label, serialized_doc in document_dictionary.items():
                if label in self.document_labels_to_worker_queues:
                    raise DuplicateDocumentError(label)
                else:
                    worker_queue_number = self.next_worker_queue_number()
                    self.document_labels_to_worker_queues[label] = worker_queue_number
                    self.word_dictionaries_need_rebuilding = True
                    self.input_queues[worker_queue_number].put((
                        self.worker.register_serialized_document,
                        (serialized_doc, label), reply_queue), TIMEOUT_SECONDS)
        self.handle_response(reply_queue, len(document_dictionary), 'register_serialized_documents')

    def register_serialized_document(self, serialized_document, label):
        """Parameters:

        document -- a preparsed Holmes document.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases involving single documents (typically user entries).
        """
        self.register_serialized_documents({label: serialized_document})

    def parse_and_register_document(self, document_text, label=''):
        """Parameters:

        document_text -- the raw document text.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases involving single documents (typically user entries).
        """

        doc = self.nlp(document_text)
        self.register_serialized_document(doc.to_bytes(), label)

    def remove_document(self, label):
        """Parameters:

        label -- the label of the document to be removed.
        """
        reply_queue = self.multiprocessor_manager.Queue()
        with self.lock:
            if label in self.document_labels_to_worker_queues:
                self.input_queues[self.document_labels_to_worker_queues[label]].put((
                    self.worker.remove_document, (label,), reply_queue), TIMEOUT_SECONDS)
                del self.document_labels_to_worker_queues[label]
                self.word_dictionaries_need_rebuilding = True
            else:
                return
        self.handle_response(reply_queue, 1, 'remove_document')

    def remove_all_documents(self):
        reply_queue = self.multiprocessor_manager.Queue()
        with self.lock:
            for worker_index in range(self.number_of_workers):
                self.input_queues[worker_index].put((
                    self.worker.remove_all_documents, None, reply_queue), TIMEOUT_SECONDS)
            self.word_dictionaries_need_rebuilding = True
            self.document_labels_to_worker_queues = {}
        self.handle_response(reply_queue, self.number_of_workers, 'remove_all_documents')

    def document_labels(self):
        """Returns a list of the labels of the currently registered documents."""
        with self.lock:
            unsorted_labels = self.document_labels_to_worker_queues.keys()
        return sorted(unsorted_labels)

    def serialize_document(self, label):
        """Returns a serialized representation of a Holmes document that can be persisted to
            a file. If *label* is not the label of a registered document, *None* is returned
            instead.

        Parameters:

        label -- the label of the document to be serialized.
        """
        reply_queue = self.multiprocessor_manager.Queue()
        with self.lock:
            if label in self.document_labels_to_worker_queues:
                self.input_queues[self.document_labels_to_worker_queues[label]].put((
                    self.worker.get_serialized_document, (label,), reply_queue), TIMEOUT_SECONDS)
            else:
                return None
        return self.handle_response(reply_queue, 1, 'serialize_document')[0]

    def get_document(self, label):
        serialized_document = self.serialize_document(label)
        return None if serialized_document is None else \
            Doc(self.nlp.vocab).from_bytes(serialized_document)

    def debug_document(self, label):
        serialized_document = self.serialize_document(label)
        if serialized_document is not None:
            doc = Doc(self.nlp.vocab).from_bytes(serialized_document)
            self.semantic_analyzer.debug_structures(doc)
        else:
            print('No document with label', label)

    def internal_get_search_phrase(self, search_phrase_text, label):
        if label is None:
            label = search_phrase_text
        search_phrase_doc = self.nlp(search_phrase_text)
        search_phrase = self.linguistic_object_factory.create_search_phrase(
            search_phrase_text, search_phrase_doc, label, None, False)
        return search_phrase

    def register_search_phrase(self, search_phrase_text, label=None):
        """Returns the new search phrase.

        Parameters:

        search_phrase_text -- the raw search phrase text.
        label -- a label for the search phrase which need *not* be unique. Defaults to the raw
            search phrase text.
        """
        search_phrase = self.internal_get_search_phrase(search_phrase_text, label)
        search_phrase.pack()
        reply_queue = self.multiprocessor_manager.Queue()
        with self.lock:
            for worker_index in range(self.number_of_workers):
                self.input_queues[worker_index].put((
                    self.worker.register_search_phrase,
                    (search_phrase,), reply_queue), TIMEOUT_SECONDS)
            self.search_phrases.append(search_phrase)
        self.handle_response(reply_queue, self.number_of_workers, 'register_search_phrase')
        return search_phrase

    def remove_all_search_phrases_with_label(self, label):
        reply_queue = self.multiprocessor_manager.Queue()
        with self.lock:
            for worker_index in range(self.number_of_workers):
                self.input_queues[worker_index].put((
                    self.worker.remove_all_search_phrases_with_label,
                    (label,), reply_queue), TIMEOUT_SECONDS)
            self.search_phrases = [search_phrase for search_phrase in self.search_phrases
                if search_phrase.label != label]
        self.handle_response(reply_queue, self.number_of_workers,
            'remove_all_search_phrases_with_label')

    def remove_all_search_phrases(self):
        reply_queue = self.multiprocessor_manager.Queue()
        with self.lock:
            for worker_index in range(self.number_of_workers):
                self.input_queues[worker_index].put((
                    self.worker.remove_all_search_phrases, None, reply_queue), TIMEOUT_SECONDS)
            self.search_phrases = []
        self.handle_response(reply_queue, self.number_of_workers, 'remove_all_search_phrases')

    def list_search_phrase_labels(self):
        with self.lock:
            return sorted(list(set([search_phrase.label for search_phrase in self.search_phrases])))

    def match(self, search_phrase_text=None, document_text=None):
        if search_phrase_text is not None:
            search_phrase = self.internal_get_search_phrase(search_phrase_text, '')
        elif len(self.list_search_phrase_labels()) == 0:
            raise NoSearchPhraseError('At least one search phrase is required for matching.')
        else:
            search_phrase = None
        if document_text is not None:
            serialized_document = self.nlp(document_text).to_bytes()
            with self.lock:
                worker_queue_number = self.next_worker_queue_number()
            worker_range = range(worker_queue_number, worker_queue_number + 1)
            number_of_workers = 1
        else:
            with self.lock:
                if len(self.document_labels_to_worker_queues) == 0:
                    raise NoDocumentError('At least one document is required for matching.')
            serialized_document = None
            number_of_workers = self.number_of_workers
            worker_range = range(number_of_workers)
        reply_queue = self.multiprocessor_manager.Queue()
        for worker_index in worker_range:
            self.input_queues[worker_index].put((
                self.worker.match, (serialized_document, search_phrase), reply_queue),
                TIMEOUT_SECONDS)
        worker_match_dictss = self.handle_response(reply_queue, number_of_workers,
            'match')
        match_dicts = []
        for worker_match_dicts in worker_match_dictss:
            match_dicts.extend(worker_match_dicts)
        return self.structural_matcher.sort_match_dictionaries(match_dicts)

    def get_corpus_frequency_information(self):

        def merge_dicts_adding_common_values(dict1, dict2):
           dict_to_return = {**dict1, **dict2}
           for key, value in dict_to_return.items():
               if key in dict1 and key in dict2:
                   dict_to_return[key] = dict1[key] + dict2[key]
           return dict_to_return

        with self.lock:
            if self.word_dictionaries_need_rebuilding:
                reply_queue = self.multiprocessor_manager.Queue()
                worker_frequency_dict = {}
                for worker_index in range(self.number_of_workers):
                    self.input_queues[worker_index].put((
                        self.worker.get_words_to_corpus_frequencies, None, reply_queue),
                        TIMEOUT_SECONDS)
                exception_worker_label = None
                for _ in range(self.number_of_workers):
                    worker_label, return_value, return_info = reply_queue.get(
                        timeout=TIMEOUT_SECONDS)
                    if isinstance(return_info, Exception):
                        if exception_worker_label is None:
                            exception_worker_label = worker_label
                    else:
                        worker_frequency_dict = merge_dicts_adding_common_values(
                            worker_frequency_dict, return_value)
                        if self.verbose:
                            print(return_info)
                    if exception_worker_label is not None:
                        print(''.join(('ERROR executing ', method_name, '() on ',
                        exception_worker_label,
                        '. Please examine the output from the worker processes to identify the problem.')))
                self.words_to_corpus_frequencies = {}
                for word in worker_frequency_dict:
                    if word in self.words_to_corpus_frequencies:
                        self.words_to_corpus_frequencies[word] += \
                            worker_frequency_dict[word]
                    else:
                        self.words_to_corpus_frequencies[word] = \
                            worker_frequency_dict[word]
                self.maximum_corpus_frequency = max(self.words_to_corpus_frequencies.values())
                self.word_dictionaries_need_rebuilding = False
            return self.words_to_corpus_frequencies, self.maximum_corpus_frequency

    def topic_match_documents_against(
            self, text_to_match, *, use_frequency_factor=True, maximum_activation_distance=75,
            relation_score=300, reverse_only_relation_score=200,
            single_word_score=50, single_word_any_tag_score=20, different_match_cutoff_score=15,
            overlapping_relation_multiplier=1.5, embedding_penalty=0.6,
            ontology_penalty=0.9,
            maximum_number_of_single_word_matches_for_relation_matching=500,
            maximum_number_of_single_word_matches_for_embedding_matching=100,
            sideways_match_extent=100, only_one_result_per_document=False, number_of_results=10,
            document_label_filter=None, tied_result_quotient=0.9):

        """Returns the results of a topic match between an entered text and the loaded documents.

        Properties:

        text_to_match -- the text to match against the loaded documents.
        use_frequency_factor -- *True* if scores should be multiplied by a factor between 0 and 1
            expressing how rare the words matching each phraselet are in the corpus,
            otherwise *False*
        maximum_activation_distance -- the number of words it takes for a previous phraselet
            activation to reduce to zero when the library is reading through a document.
        relation_score -- the activation score added when a normal two-word
            relation is matched.
        reverse_only_relation_score -- the activation score added when a two-word relation
                is matched using a search phrase that can only be reverse-matched.
        single_word_score -- the activation score added when a normal single
            word is matched.
        single_word_any_tag_score -- the activation score added when a single word is matched
            whose tag did not correspond to the template specification.
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
        maximum_number_of_single_word_matches_for_relation_matching -- the maximum number
                of single word matches that are used as the basis for matching relations. If more
                document words than this value correspond to each of the two words within a
                relation phraselet, matching on the phraselet is not attempted.
        maximum_number_of_single_word_matches_for_embedding_matching = the maximum number
          of single word matches that are used as the basis for matching with
          embeddings at the other word. If more than this value exist, matching with
          embeddings is not attempted because the performance hit would be too great.
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
        if maximum_number_of_single_word_matches_for_embedding_matching > \
                maximum_number_of_single_word_matches_for_relation_matching:
            raise EmbeddingThresholdGreaterThanRelationThresholdError(' '.join((
                'embedding',
                str(maximum_number_of_single_word_matches_for_embedding_matching),
                'relation',
                str(maximum_number_of_single_word_matches_for_relation_matching))))
        with self.lock:
            if len(self.document_labels_to_worker_queues) == 0:
                raise NoDocumentError('At least one document is required for matching.')
        if use_frequency_factor:
            words_to_corpus_frequencies, maximum_corpus_frequency = \
                self.get_corpus_frequency_information()
        else:
            words_to_corpus_frequencies = None
            maximum_corpus_frequency = None

        reply_queue = self.multiprocessor_manager.Queue()
        text_to_match_doc = self.semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_phraselet_infos = \
            self.linguistic_object_factory.get_phraselet_labels_to_phraselet_infos(
            text_to_match_doc=text_to_match_doc,
            words_to_corpus_frequencies=words_to_corpus_frequencies,
            maximum_corpus_frequency=maximum_corpus_frequency)
        if len(phraselet_labels_to_phraselet_infos) == 0:
            return []
        phraselet_labels_to_search_phrases = \
            self.linguistic_object_factory.create_search_phrases_from_phraselet_infos(
                phraselet_labels_to_phraselet_infos.values())
        for search_phrase in phraselet_labels_to_search_phrases.values():
            search_phrase.pack()

        for worker_index in range(self.number_of_workers):
            self.input_queues[worker_index].put((
                self.worker.get_topic_matches,
                (text_to_match, phraselet_labels_to_phraselet_infos,
                phraselet_labels_to_search_phrases,
                self.embedding_based_matching_on_root_words, maximum_activation_distance,
                relation_score, reverse_only_relation_score, single_word_score,
                single_word_any_tag_score, different_match_cutoff_score,
                overlapping_relation_multiplier, embedding_penalty, ontology_penalty,
                maximum_number_of_single_word_matches_for_relation_matching,
                maximum_number_of_single_word_matches_for_embedding_matching,
                sideways_match_extent, only_one_result_per_document, number_of_results,
                document_label_filter), reply_queue), TIMEOUT_SECONDS)
        worker_topic_match_dictss = self.handle_response(reply_queue,
            self.number_of_workers, 'match')
        topic_match_dicts = []
        for worker_topic_match_dicts in worker_topic_match_dictss:
            if worker_topic_match_dicts is not None:
                topic_match_dicts.extend(worker_topic_match_dicts)
        return TopicMatchDictionaryOrderer().order(
            topic_match_dicts, number_of_results, tied_result_quotient)

    def get_supervised_topic_training_basis(
            self, *, classification_ontology=None,
            overlap_memory_size=10, oneshot=True, match_all_words=False, verbose=True):
        """ Returns an object that is used to train and generate a document model.

            Parameters:

            classification_ontology -- an Ontology object incorporating relationships between
                classification labels, or 'None' if no such ontology is to be used.
            overlap_memory_size -- how many non-word phraselet matches to the left should be
                checked for words in common with a current match.
            oneshot -- whether the same word or relationship matched multiple times should be
                counted once only (value 'True') or multiple times (value 'False')
            match_all_words -- whether all single words should be taken into account
                (value 'True') or only single words with noun tags (value 'False')
            verbose -- if 'True', information about training progress is outputted to the console.
        """
        return SupervisedTopicTrainingBasis(
            linguistic_object_factory=self.linguistic_object_factory,
            structural_matcher=self.structural_matcher,
            classification_ontology=classification_ontology,
            overlap_memory_size=overlap_memory_size, oneshot=oneshot,
            match_all_words=match_all_words, verbose=verbose)

    def deserialize_supervised_topic_classifier(self, serialized_model, verbose=False):
        """ Returns a document classifier that will use a pre-trained model.

            Parameters:

            serialized_model -- the pre-trained model, which will correspond to a
                'SupervisedTopicClassifierModel' instance.
            verbose -- if 'True', information about matching is outputted to the console.
        """
        model = jsonpickle.decode(serialized_model)
        return SupervisedTopicClassifier(
            self.semantic_analyzer, self.linguistic_object_factory, self.structural_matcher,
            model, verbose)

    def start_chatbot_mode_console(self):
        """Starts a chatbot mode console enabling the matching of pre-registered search phrases
            to documents (chatbot entries) entered ad-hoc by the user.
        """
        holmes_consoles = HolmesConsoles(self)
        holmes_consoles.start_chatbot_mode()

    def start_structural_search_mode_console(self):
        """Starts a structural search mode console enabling the matching of pre-registered documents
            to search phrases entered ad-hoc by the user.
        """
        holmes_consoles = HolmesConsoles(self)
        holmes_consoles.start_structural_search_mode()

    def start_topic_matching_search_mode_console(
            self, only_one_result_per_document=False,
            maximum_number_of_single_word_matches_for_relation_matching=500,
            maximum_number_of_single_word_matches_for_embedding_matching=100):
        """Starts a topic matching search mode console enabling the matching of pre-registered
            documents to search texts entered ad-hoc by the user.

            Parameters:

            only_one_result_per_document -- if 'True', prevents multiple topic match
                results from being returned for the same document.
            maximum_number_of_single_word_matches_for_relation_matching -- the maximum number
                of single word matches that are used as the basis for matching relations. If more
                document words than this value correspond to each of the two words within a
                relation phraselet, matching on the phraselet is not attempted.
            maximum_number_of_single_word_matches_for_embedding_matching = the maximum
                number of single word matches that are used as the basis for reverse matching with
                embeddings at the parent word. If more than this value exist, reverse matching with
                embeddings is not attempted because the performance hit would be too great.
        """
        holmes_consoles = HolmesConsoles(self)
        holmes_consoles.start_topic_matching_search_mode(
            only_one_result_per_document,
            maximum_number_of_single_word_matches_for_relation_matching=
            maximum_number_of_single_word_matches_for_relation_matching,
            maximum_number_of_single_word_matches_for_embedding_matching=
            maximum_number_of_single_word_matches_for_embedding_matching)

    def close(self):
        for worker in self.workers:
            worker.terminate()

class Worker:
    """Worker implementation used by *Manager*.
    """

    def error_header(self, method, args, worker_label):
        return ''.join((
            worker_label,
            ' - error:'))

    def listen(self, structural_matcher, vocab, model_name, serialized_document_version,
            input_queue, worker_label):
        state = {
            'structural_matcher': structural_matcher,
            'vocab': vocab,
            'model_name': model_name,
            'serialized_document_version': serialized_document_version,
            'indexed_documents': {},
            'search_phrases': [],
        }
        HolmesBroker.set_extensions()
        while True:
            method, args, reply_queue = input_queue.get()
            try:
                if args is not None:
                    return_value, return_info = method(state, *args)
                else:
                    return_value, return_info = method(state)
                reply_queue.put((worker_label, return_value, return_info), TIMEOUT_SECONDS)
            except Exception as err:
                print(self.error_header(method, args, worker_label))
                print(traceback.format_exc())
                reply_queue.put((worker_label, None, err), TIMEOUT_SECONDS)
            except:
                print(self.error_header(method, args, worker_label))
                print(traceback.format_exc())
                err_identifier = str(sys.exc_info()[0])
                reply_queue.put((worker_label, None, err_identifier), TIMEOUT_SECONDS)

    def get_indexed_document(self, state, serialized_doc):
        doc = Doc(state['vocab']).from_bytes(serialized_doc)
        if doc._.holmes_document_info.model != state['model_name']:
            raise WrongModelDeserializationError('; '.join((
                state['model_name'], doc._.holmes_document_info.model)))
        if doc._.holmes_document_info.serialized_document_version != \
                state['serialized_document_version']:
            raise WrongVersionDeserializationError('; '.join((
                str(state['serialized_document_version']),
                str(doc._.holmes_document_info.serialized_document_version))))
        return state['structural_matcher'].semantic_matching_helper.index_document(doc)

    def register_serialized_document(self, state, serialized_doc, label):
        state['indexed_documents'][label] = self.get_indexed_document(state, serialized_doc)
        return None, ' '.join(('Registered document', label))

    def remove_document(self, state, label):
        state['indexed_documents'].pop(label)
        return None, ' '.join(('Removed document', label))

    def remove_all_documents(self, state):
        state['indexed_documents'] = {}
        return None, 'Removed all documents'

    def get_serialized_document(self, state, label):
        if label in state['indexed_documents']:
            return state['indexed_documents'][label].doc.to_bytes(), \
                ' '.join(('Returned serialized document with label', label))
        else:
            return None, ' '.join(('No document found with label', label))

    def register_search_phrase(self, state, search_phrase):
        search_phrase.unpack(state['vocab'])
        state['search_phrases'].append(search_phrase)
        return None, ' '.join(('Registered search phrase with label', search_phrase.label))

    def remove_all_search_phrases_with_label(self, state, label):
        state['search_phrases'] = [search_phrase for search_phrase in state['search_phrases']
            if search_phrase.label != label]
        return None, ' '.join(("Removed all search phrases with label '", label, "'"))

    def remove_all_search_phrases(self, state):
        state['search_phrases'] = []
        return None, 'Removed all search phrases'

    def get_words_to_corpus_frequencies(self, state):
        words_to_corpus_frequencies = {}
        for indexed_document in state['indexed_documents'].values():
            words_to_token_info_dict = indexed_document.words_to_token_info_dict
            for word, token_info_tuples in words_to_token_info_dict.items():
                if word in punctuation:
                    continue
                indexes = [index for index, _, _ in token_info_tuples]
                if word in words_to_corpus_frequencies:
                    words_to_corpus_frequencies[word] += len(set(indexes))
                else:
                    words_to_corpus_frequencies[word] = len(set(indexes))
        return words_to_corpus_frequencies, 'Retrieved words to corpus frequencies'

    def match(self, state, serialized_doc, search_phrase):
        indexed_documents = {'': self.get_indexed_document(state, serialized_doc)} \
            if serialized_doc is not None else state['indexed_documents']
        search_phrases = [search_phrase] if search_phrase is not None \
            else state['search_phrases']
        if len(indexed_documents) > 0 and len(search_phrases) > 0:
            matches = state['structural_matcher'].match(
                indexed_documents=indexed_documents,
                search_phrases=search_phrases,
                output_document_matching_message_to_console=False,
                match_depending_on_single_words=None,
                compare_embeddings_on_root_words=False,
                compare_embeddings_on_non_root_words=True,
                document_labels_to_indexes_for_reverse_matching_sets=None,
                document_labels_to_indexes_for_embedding_reverse_matching_sets=None)
            return state['structural_matcher'].build_match_dictionaries(matches), \
                'Returned matches'
        else:
            return [], 'No stored objects to match against'

    def get_topic_matches(self, state, text_to_match,
            phraselet_labels_to_phraselet_infos, phraselet_labels_to_search_phrases,
            embedding_based_matching_on_root_words, maximum_activation_distance,
            relation_score, reverse_only_relation_score, single_word_score,
            single_word_any_tag_score, different_match_cutoff_score,
            overlapping_relation_multiplier, embedding_penalty, ontology_penalty,
            maximum_number_of_single_word_matches_for_relation_matching,
            maximum_number_of_single_word_matches_for_embedding_matching,
            sideways_match_extent, only_one_result_per_document, number_of_results,
            document_label_filter):
        if len(state['indexed_documents']) == 0:
            return [], 'No stored documents to match against'
        for search_phrase in phraselet_labels_to_search_phrases.values():
            search_phrase.unpack(state['vocab'])
        topic_matcher = TopicMatcher(
            structural_matcher=state['structural_matcher'],
            indexed_documents=state['indexed_documents'],
            text_to_match=text_to_match,
            phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
            phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
            embedding_based_matching_on_root_words=embedding_based_matching_on_root_words,
            maximum_activation_distance=maximum_activation_distance,
            relation_score=relation_score,
            reverse_only_relation_score=reverse_only_relation_score,
            single_word_score=single_word_score,
            single_word_any_tag_score=single_word_any_tag_score,
            different_match_cutoff_score=different_match_cutoff_score,
            overlapping_relation_multiplier=overlapping_relation_multiplier,
            embedding_penalty=embedding_penalty,
            ontology_penalty=ontology_penalty,
            maximum_number_of_single_word_matches_for_relation_matching=
            maximum_number_of_single_word_matches_for_relation_matching,
            maximum_number_of_single_word_matches_for_embedding_matching=
            maximum_number_of_single_word_matches_for_embedding_matching,
            sideways_match_extent=sideways_match_extent,
            only_one_result_per_document=only_one_result_per_document,
            number_of_results=number_of_results,
            document_label_filter=document_label_filter)
        return topic_matcher.get_topic_match_dictionaries(), \
            'Returned topic match dictionaries'

@Language.factory("holmes")
class HolmesBroker:
    def __init__(self, nlp:Language, name:str):
        self.nlp = nlp
        self.pid = os.getpid()
        self.semantic_analyzer = get_semantic_analyzer(nlp)
        self.set_extensions()

    def __call__(self, doc:Doc) -> Doc:
        if os.getpid() != self.pid:
            raise MultiprocessingParsingNotSupportedError(
                'Unfortunately at present parsing cannot be shared between forked processes.')
        try:
            self.semantic_analyzer.holmes_parse(doc)
        except:
            print('Unexpected error annotating document, skipping ....')
            exception_info_parts = sys.exc_info()
            print(exception_info_parts[0])
            print(exception_info_parts[1])
            traceback.print_tb(exception_info_parts[2])
        return doc

    def __getstate__(self):
        return self.nlp.meta

    def __setstate__(self, meta):
        nlp_name = '_'.join((meta['lang'], meta['name']))
        self.nlp = spacy.load(nlp_name)
        self.semantic_analyzer = get_semantic_analyzer(self.nlp)
        self.pid = os.getpid()
        HolmesBroker.set_extensions()

    @staticmethod
    def set_extensions():
        if not Doc.has_extension('coref_chains'):
            Doc.set_extension('coref_chains', default=None)
        if not Token.has_extension('coref_chains'):
            Token.set_extension('coref_chains', default=None)
        if not Doc.has_extension('holmes_document_info'):
            Doc.set_extension('holmes_document_info', default=None)
        if not Token.has_extension('holmes'):
            Token.set_extension('holmes', default=None)
