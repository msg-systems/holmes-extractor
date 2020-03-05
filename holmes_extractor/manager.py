import copy
import sys
import thinc
from .errors import *
from .structural_matching import StructuralMatcher, ThreadsafeContainer
from .semantics import SemanticAnalyzerFactory
from .extensive_matching import *
from .consoles import HolmesConsoles
from multiprocessing import Process, Queue, Manager as Multiprocessing_manager, cpu_count
from threading import Lock

def validate_options(semantic_analyzer, overall_similarity_threshold,
        embedding_based_matching_on_root_words, perform_coreference_resolution):
    if overall_similarity_threshold < 0.0 or overall_similarity_threshold > 1.0:
        raise ValueError(
                'overall_similarity_threshold must be between 0 and 1')
    if overall_similarity_threshold != 1.0 and not \
            semantic_analyzer.model_supports_embeddings():
        raise ValueError(
                'Model has no embeddings: overall_similarity_threshold must be 1.')
    if overall_similarity_threshold == 1.0 and embedding_based_matching_on_root_words:
        raise ValueError(
                'overall_similarity_threshold is 1; embedding_based_matching_on_root_words must be False')
    if perform_coreference_resolution and not \
            semantic_analyzer.model_supports_coreference_resolution():
        raise ValueError(
                'Model does not support coreference resolution: perform_coreference_resolution may not be True')


class Manager:
    """The facade class for the Holmes library.

    Parameters:

    model -- the name of the spaCy model, e.g. *en_core_web_lg*
    overall_similarity_threshold -- the overall similarity threshold for embedding-based
        matching. Defaults to *1.0*, which deactivates embedding-based matching.
    embedding_based_matching_on_root_words -- determines whether or not embedding-based
        matching should be attempted on search-phrase root tokens, which has a considerable
        performance hit. Defaults to *False*.
    ontology -- an *Ontology* object. Defaults to *None* (no ontology).
    analyze_derivational_morphology -- *True* if matching should be attempted between different
        words from the same word family. Defaults to *True*.
    perform_coreference_resolution -- *True*, *False*, or *None* if coreference resolution
        should be performed depending on whether the model supports it. Defaults to *None*.
    debug -- a boolean value specifying whether debug representations should be outputted
        for parsed sentences. Defaults to *False*.
    """

    def __init__(self, model, *, overall_similarity_threshold=1.0,
            embedding_based_matching_on_root_words=False, ontology=None,
            analyze_derivational_morphology=True, perform_coreference_resolution=None, debug=False):
        self.semantic_analyzer = SemanticAnalyzerFactory().semantic_analyzer(model=model,
                perform_coreference_resolution=perform_coreference_resolution, debug=debug)
        if perform_coreference_resolution == None:
            perform_coreference_resolution = \
                    self.semantic_analyzer.model_supports_coreference_resolution()
        validate_options(self.semantic_analyzer, overall_similarity_threshold,
                embedding_based_matching_on_root_words, perform_coreference_resolution)
        self.ontology = ontology
        self.debug = debug
        self.overall_similarity_threshold = overall_similarity_threshold
        self.embedding_based_matching_on_root_words = embedding_based_matching_on_root_words
        self.perform_coreference_resolution = perform_coreference_resolution
        self.structural_matcher = StructuralMatcher(self.semantic_analyzer, ontology,
                overall_similarity_threshold, embedding_based_matching_on_root_words,
                analyze_derivational_morphology, perform_coreference_resolution)
        self.threadsafe_container = ThreadsafeContainer()

    def parse_and_register_document(self, document_text, label=''):
        """Parameters:

        document_text -- the raw document text.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases involving single documents (typically user entries).
        """

        doc = self.semantic_analyzer.parse(document_text)
        self.register_parsed_document(doc, label)

    def register_parsed_document(self, doc, label=''):
        """Parameters:

        document -- a preparsed Holmes document.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases involving single documents (typically user entries).
        """
        indexed_document = self.structural_matcher.index_document(doc)
        self.threadsafe_container.register_document(indexed_document, label)

    def deserialize_and_register_document(self, document, label=''):
        """Parameters:

        document -- a Holmes document serialized using the *serialize_document()* function.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases involving single documents (typically user entries).
        """
        if self.perform_coreference_resolution:
            raise SerializationNotSupportedError(self.semantic_analyzer.model)
        doc = self.semantic_analyzer.from_serialized_string(document)
        self.semantic_analyzer.debug_structures(doc) # only has effect when debug=True
        indexed_document = self.structural_matcher.index_document(doc)
        self.threadsafe_container.register_document(indexed_document, label)

    def remove_document(self, label):
        """Parameters:

        label -- the label of the document to be removed.
        """
        self.threadsafe_container.remove_document(label)

    def remove_all_documents(self):
        self.threadsafe_container.remove_all_documents()

    def document_labels(self):
        """Returns a list of the labels of the currently registered documents."""
        return self.threadsafe_container.document_labels()

    def serialize_document(self, label):
        """Returns a serialized representation of a Holmes document that can be persisted to
            a file. If *label* is not the label of a registered document, *None* is returned
            instead.

        Parameters:

        label -- the label of the document to be serialized.
        """

        if self.perform_coreference_resolution:
            raise SerializationNotSupportedError(self.semantic_analyzer.model)
        doc = self.threadsafe_container.get_document(label)
        if doc != None:
            return self.semantic_analyzer.to_serialized_string(doc)
        else:
            return None

    def register_search_phrase(self, search_phrase_text, label=None):
        """Parameters:

        search_phrase_text -- the raw search phrase text.
        label -- a label for the search phrase which need *not* be unique. Defaults to the raw
            search phrase text.
        """
        if label==None:
            label=search_phrase_text
        search_phrase_doc = self.semantic_analyzer.parse(search_phrase_text)
        search_phrase = self.structural_matcher.create_search_phrase(search_phrase_text,
                search_phrase_doc, label, None, False)
        self.threadsafe_container.register_search_phrase(search_phrase)

    def remove_all_search_phrases(self):
        self.threadsafe_container.remove_all_search_phrases()

    def remove_all_search_phrases_with_label(self, label):
        self.threadsafe_container.remove_all_search_phrases_with_label(label)

    def match(self):
        """Matches the registered search phrases to the registered documents. Returns a list
            of *Match* objects sorted by their overall similarity measures in descending order.
            Should be called by applications wishing to retain references to the spaCy and
            Holmes information that was used to derive the matches.
        """
        indexed_documents = self.threadsafe_container.get_indexed_documents()
        search_phrases = self.threadsafe_container.get_search_phrases()
        return self.structural_matcher.match(indexed_documents = indexed_documents,
                search_phrases = search_phrases,
                output_document_matching_message_to_console = False,
                match_depending_on_single_words = None,
                compare_embeddings_on_root_words = False,
                compare_embeddings_on_non_root_words = True,
                document_labels_to_indexes_for_reverse_matching_sets = None,
                document_labels_to_indexes_for_embedding_reverse_matching_sets = None)

    def _build_match_dictionaries(self, matches):
        """Builds and returns a list of dictionaries describing matches."""
        match_dicts = []
        for match in matches:
            earliest_sentence_index = sys.maxsize
            latest_sentence_index = -1
            for word_match in match.word_matches:
                sentence_index = word_match.document_token.sent.start
                if sentence_index < earliest_sentence_index:
                    earliest_sentence_index = sentence_index
                if sentence_index > latest_sentence_index:
                    latest_sentence_index = sentence_index
            sentences_string = ' '.join(sentence.text.strip() for sentence in
                    match.word_matches[0].document_token.doc.sents if sentence.start >=
                    earliest_sentence_index and sentence.start <= latest_sentence_index)

            match_dict = {
                    'search_phrase': match.search_phrase_label,
                    'document': match.document_label,
                    'index_within_document': match.index_within_document,
                    'sentences_within_document': sentences_string,
                    'negated': match.is_negated,
                    'uncertain': match.is_uncertain,
                    'involves_coreference': match.involves_coreference,
                    'overall_similarity_measure': match.overall_similarity_measure}
            text_word_matches = []
            for word_match in match.word_matches:
                text_word_matches.append({
                        'search_phrase_word': word_match.search_phrase_word,
                        'document_word': word_match.document_word,
                        'document_phrase': self.semantic_analyzer.get_dependent_phrase(
                                word_match.document_token, word_match.document_subword),
                        'match_type': word_match.type,
                        'similarity_measure': str(word_match.similarity_measure),
                        'involves_coreference': word_match.involves_coreference,
                        'extracted_word': word_match.extracted_word})
            match_dict['word_matches']=text_word_matches
            match_dicts.append(match_dict)
        return match_dicts

    def match_returning_dictionaries(self):
        """Matches the registered search phrases to the registered documents. Returns a list
            of dictionaries describing any matches, sorted by their overall similarity measures in
            descending order. Callers of this method do not have to manage any further
            dependencies on spaCy or Holmes.
        """
        return self._build_match_dictionaries(self.match())

    def match_search_phrases_against(self, entry):
        """Matches the registered search phrases against a single document
            supplied to the method and returns dictionaries describing any matches.
        """
        search_phrases = self.threadsafe_container.get_search_phrases()
        doc = self.semantic_analyzer.parse(entry)
        indexed_documents = {'':self.structural_matcher.index_document(doc)}
        matches = self.structural_matcher.match(indexed_documents = indexed_documents,
                        search_phrases = search_phrases,
                        output_document_matching_message_to_console = False,
                        match_depending_on_single_words = None,
                        compare_embeddings_on_root_words = False,
                        compare_embeddings_on_non_root_words = True,
                        document_labels_to_indexes_for_reverse_matching_sets = None,
                        document_labels_to_indexes_for_embedding_reverse_matching_sets = None)
        return self._build_match_dictionaries(matches)

    def match_documents_against(self, search_phrase_text):
        """Matches the registered documents against a single search phrase
            supplied to the method and returns dictionaries describing any matches.
        """
        indexed_documents = self.threadsafe_container.get_indexed_documents()
        search_phrase_doc = self.semantic_analyzer.parse(search_phrase_text)
        search_phrases = [self.structural_matcher.create_search_phrase(search_phrase_text,
                search_phrase_doc, search_phrase_text, None, False)]
        matches = self.structural_matcher.match(indexed_documents = indexed_documents,
                        search_phrases = search_phrases,
                        output_document_matching_message_to_console = False,
                        match_depending_on_single_words = None,
                        compare_embeddings_on_root_words = False,
                        compare_embeddings_on_non_root_words = True,
                        document_labels_to_indexes_for_reverse_matching_sets = None,
                        document_labels_to_indexes_for_embedding_reverse_matching_sets = None)
        return self._build_match_dictionaries(matches)

    def topic_match_documents_against(self, text_to_match, *, maximum_activation_distance=75,
            relation_score=30, reverse_only_relation_score = 20,
            single_word_score=5, single_word_any_tag_score=2,
            overlapping_relation_multiplier=1.5, embedding_penalty=0.6,
            maximum_activation_value=1000,
            maximum_number_of_single_word_matches_for_relation_matching = 500,
            maximum_number_of_single_word_matches_for_embedding_matching = 100,
            sideways_match_extent=100, only_one_result_per_document=False, number_of_results=10,
            document_label_filter=None):
        """Returns the results of a topic match between an entered text and the loaded documents.

        Properties:

        text_to_match -- the text to match against the loaded documents.
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
        overlapping_relation_multiplier -- the value by which the activation score is multiplied
            when two relations were matched and the matches involved a common document word.
        embedding_penalty -- a value between 0 and 1 with which scores are multiplied when the
            match involved an embedding. The result is additionally multiplied by the overall
            similarity measure of the match.
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
        """
        maximum_number_of_single_word_matches_for_relation_matching,
        maximum_number_of_single_word_matches_for_embedding_matching,
        topic_matcher = TopicMatcher(semantic_analyzer = self.semantic_analyzer,
                structural_matcher = self.structural_matcher,
                indexed_documents = self.threadsafe_container.get_indexed_documents(),
                maximum_activation_distance=maximum_activation_distance,
                relation_score=relation_score,
                reverse_only_relation_score=reverse_only_relation_score,
                single_word_score=single_word_score,
                single_word_any_tag_score=single_word_any_tag_score,
                overlapping_relation_multiplier=overlapping_relation_multiplier,
                embedding_penalty=embedding_penalty,
                maximum_number_of_single_word_matches_for_relation_matching =
                maximum_number_of_single_word_matches_for_relation_matching,
                maximum_number_of_single_word_matches_for_embedding_matching =
                maximum_number_of_single_word_matches_for_embedding_matching,
                sideways_match_extent=sideways_match_extent,
                only_one_result_per_document=only_one_result_per_document,
                number_of_results=number_of_results,
                document_label_filter=document_label_filter)
        return topic_matcher.topic_match_documents_against(text_to_match)

    def topic_match_documents_returning_dictionaries_against(self, text_to_match, *,
            maximum_activation_distance=75, relation_score=30, reverse_only_relation_score = 20,
            single_word_score=5, single_word_any_tag_score=2, overlapping_relation_multiplier=1.5,
            embedding_penalty=0.6,
            maximum_number_of_single_word_matches_for_relation_matching = 500,
            maximum_number_of_single_word_matches_for_embedding_matching = 100,
            sideways_match_extent=100, only_one_result_per_document=False, number_of_results=10,
            document_label_filter=None, tied_result_quotient=0.9):
        """Returns a list of dictionaries representing the results of a topic match between an
            entered text and the loaded documents. Callers of this method do not have to manage any
            further dependencies on spaCy or Holmes.

        Properties:

        text_to_match -- the text to match against the loaded documents.
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
        overlapping_relation_multiplier -- the value by which the activation score is multiplied
            when two relations were matched and the matches involved a common document word.
        embedding_penalty -- a value between 0 and 1 with which scores are multiplied when the
            match involved an embedding. The result is additionally multiplied by the overall
            similarity measure of the match.
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
        tied_result_quotient -- the quotient between a result and following results above which
            the results are interpreted as tied.
        document_label_filter -- optionally, a string with which document labels must start to
            be considered for inclusion in the results.
        """

        topic_matcher = TopicMatcher(semantic_analyzer = self.semantic_analyzer,
                structural_matcher = self.structural_matcher,
                indexed_documents = self.threadsafe_container.get_indexed_documents(),
                maximum_activation_distance=maximum_activation_distance,
                relation_score=relation_score,
                reverse_only_relation_score=reverse_only_relation_score,
                single_word_score=single_word_score,
                single_word_any_tag_score=single_word_any_tag_score,
                overlapping_relation_multiplier=overlapping_relation_multiplier,
                embedding_penalty=embedding_penalty,
                maximum_number_of_single_word_matches_for_relation_matching =
                maximum_number_of_single_word_matches_for_relation_matching,
                maximum_number_of_single_word_matches_for_embedding_matching =
                maximum_number_of_single_word_matches_for_embedding_matching,
                sideways_match_extent=sideways_match_extent,
                only_one_result_per_document=only_one_result_per_document,
                number_of_results=number_of_results,
                document_label_filter=document_label_filter)
        return topic_matcher.topic_match_documents_returning_dictionaries_against(text_to_match,
                tied_result_quotient=tied_result_quotient)

    def get_supervised_topic_training_basis(self, *, classification_ontology=None,
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
        return SupervisedTopicClassifier(self.semantic_analyzer,
                self.structural_matcher,
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

    def start_topic_matching_search_mode_console(self, only_one_result_per_document=False,
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
        holmes_consoles.start_topic_matching_search_mode(only_one_result_per_document,
                maximum_number_of_single_word_matches_for_relation_matching=
                maximum_number_of_single_word_matches_for_relation_matching,
                maximum_number_of_single_word_matches_for_embedding_matching=
                maximum_number_of_single_word_matches_for_embedding_matching)

class MultiprocessingManager:
    """The facade class for the Holmes library used in a multiprocessing environment.
        This class is threadsafe.

    Parameters:

    model -- the name of the spaCy model, e.g. *en_core_web_lg*
    overall_similarity_threshold -- the overall similarity threshold for embedding-based
        matching. Defaults to *1.0*, which deactivates embedding-based matching.
    embedding_based_matching_on_root_words -- determines whether or not embedding-based
        matching should be attempted on root (parent) tokens, which has a considerable
        performance hit. Defaults to *False*.
    ontology -- an *Ontology* object. Defaults to *None* (no ontology).
    analyze_derivational_morphology -- *True* if matching should be attempted between different
        words from the same word family. Defaults to *True*.
    perform_coreference_resolution -- *True*, *False* or *None* if coreference resolution
        should be performed depending on whether the model supports it. Defaults to *None*.
    debug -- a boolean value specifying whether debug representations should be outputted
        for parsed sentences. Defaults to *False*.
    verbose -- a boolean value specifying whether status messages should be outputted to the
        console. Defaults to *True*
    number_of_workers -- the number of worker processes to use, or *None* if the number of worker
        processes should depend on the number of available cores. Defaults to *None*
    """
    def __init__(self, model, *, overall_similarity_threshold=1.0,
            embedding_based_matching_on_root_words=False, ontology=None,
            analyze_derivational_morphology=True, perform_coreference_resolution=None,
            debug=False, verbose=True, number_of_workers=None):
        self.semantic_analyzer = SemanticAnalyzerFactory().semantic_analyzer(model=model,
                perform_coreference_resolution=perform_coreference_resolution, debug=debug)
        if perform_coreference_resolution == None:
            perform_coreference_resolution = \
                    self.semantic_analyzer.model_supports_coreference_resolution()
        validate_options(self.semantic_analyzer, overall_similarity_threshold,
                embedding_based_matching_on_root_words, perform_coreference_resolution)
        self.structural_matcher = StructuralMatcher(self.semantic_analyzer, ontology,
                overall_similarity_threshold, embedding_based_matching_on_root_words,
                analyze_derivational_morphology, perform_coreference_resolution)
        self._perform_coreference_resolution = perform_coreference_resolution

        self._verbose = verbose
        self._document_labels = []
        self._input_queues = []
        if number_of_workers == None:
            number_of_workers = cpu_count()
        self._number_of_workers = number_of_workers
        self._next_worker_to_use = 0
        self._multiprocessor_manager = Multiprocessing_manager()
        self._worker = Worker() # will be copied to worker processes by value (Windows) or
                                # by reference (Linux)
        self._workers = []
        for counter in range(0, self._number_of_workers):
            input_queue = Queue()
            self._input_queues.append(input_queue)
            worker_label = ' '.join(('Worker', str(counter)))
            this_worker = Process(target=self._worker.listen, args=(self.semantic_analyzer,
                    self.structural_matcher, input_queue, worker_label), daemon=True)
            self._workers.append(this_worker)
            this_worker.start()
        self._lock = Lock()

    def _add_document_label(self, label):
        with self._lock:
            if label in self._document_labels:
                raise DuplicateDocumentError(label)
            else:
                self._document_labels.append(label)

    def _handle_reply(self, worker_label, return_value):
        """ If 'return_value' is an exception, return it, otherwise return 'None'. """
        if isinstance(return_value, Exception):
            with self._lock:
                print('Exception', type(return_value), worker_label, return_value)
            return return_value
        elif self._verbose:
            if not type(return_value) is list:
                with self._lock:
                    print(': '.join((worker_label, return_value)))
            return None

    def _internal_register_documents(self, dict, worker_method):
        reply_queue = self._multiprocessor_manager.Queue()
        for label, value in dict.items():
            self._add_document_label(label)
            with(self._lock):
                self._input_queues[self._next_worker_to_use].put\
                        ((worker_method, (value, label), reply_queue))
                self._next_worker_to_use += 1
                if self._next_worker_to_use == self._number_of_workers:
                    self._next_worker_to_use = 0
        for _ in range(0, len(dict)):
            possible_exception = self._handle_reply(*reply_queue.get())
            if possible_exception != None:
                self.close()

    def parse_and_register_documents(self, document_dictionary):
        """Parameters:

        document_dictionary -- a dictionary from unique document labels to raw document texts.
        """
        self._internal_register_documents(document_dictionary,
                self._worker.worker_parse_and_register_document)

    def deserialize_and_register_documents(self, serialized_document_dictionary):
        """Parameters:

        serialized_document_dictionary -- a dictionary from unique document labels to
        documents serialized using the *Manager.serialize_document()* method.
        """
        if self._perform_coreference_resolution:
            raise SerializationNotSupportedError(self.semantic_analyzer.model)
        self._internal_register_documents(serialized_document_dictionary,
                self._worker.worker_deserialize_and_register_document)

    def document_labels(self):
        with self._lock:
            document_labels = self._document_labels
        return sorted(document_labels)

    def topic_match_documents_returning_dictionaries_against(self, text_to_match, *,
            maximum_activation_distance=75, relation_score=30, reverse_only_relation_score = 20,
            single_word_score=5, single_word_any_tag_score=2, overlapping_relation_multiplier=1.5,
            embedding_penalty=0.6,maximum_number_of_single_word_matches_for_relation_matching = 500,
            maximum_number_of_single_word_matches_for_embedding_matching = 100,
            sideways_match_extent=100, only_one_result_per_document=False, number_of_results=10,
            document_label_filter=None, tied_result_quotient=0.9):
        """Returns the results of a topic match between an entered text and the loaded documents.

        Properties:

        text_to_match -- the text to match against the loaded documents.
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
        overlapping_relation_multiplier -- the value by which the activation score is multiplied
            when two relations were matched and the matches involved a common document word.
        embedding_penalty -- a value between 0 and 1 with which scores are multiplied when the
            match involved an embedding. The result is additionally multiplied by the overall
            similarity measure of the match.
        maximum_number_of_single_word_matches_for_relation_matching -- the maximum number
                of single word matches that are used as the basis for matching relations. If more
                document words than this value correspond to each of the two words within a
                relation phraselet, matching on the phraselet is not attempted.
        maximum_number_of_single_word_matches_for_embedding_matching = the maximum number
                of single word matches that are used as the basis for reverse matching with
                embeddings at the parent word. If more than this value exist, reverse matching with
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
        reply_queue = self._multiprocessor_manager.Queue()
        for counter in range(0, self._number_of_workers):
            self._input_queues[counter].put((
                    self._worker.worker_topic_match_documents_returning_dictionaries_against,
                    (text_to_match, maximum_activation_distance, relation_score,
                    reverse_only_relation_score, single_word_score, single_word_any_tag_score,
                    overlapping_relation_multiplier, embedding_penalty,
                    maximum_number_of_single_word_matches_for_relation_matching,
                    maximum_number_of_single_word_matches_for_embedding_matching,
                    sideways_match_extent, only_one_result_per_document, number_of_results,
                    document_label_filter, tied_result_quotient), reply_queue))
        topic_match_dicts = []
        for _ in range(0, self._number_of_workers):
            worker_label, worker_topic_match_dicts = reply_queue.get()
            possible_exception = self._handle_reply(worker_label, worker_topic_match_dicts)
            if possible_exception != None:
                self.close()
                break
            topic_match_dicts.extend(worker_topic_match_dicts)
        else:
            return TopicMatchDictionaryOrderer().order(topic_match_dicts, number_of_results,
                    tied_result_quotient)

    def start_topic_matching_search_mode_console(self, only_one_result_per_document=False,
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
            maximum_number_of_single_word_matches_for_embedding_matching = the maximum number
              of single word matches that are used as the basis for matching with
              embeddings at the other word. If more than this value exist, matching with
              embeddings is not attempted because the performance hit would be too great.
        """
        holmes_consoles = HolmesConsoles(self)
        holmes_consoles.start_topic_matching_search_mode(only_one_result_per_document,
                maximum_number_of_single_word_matches_for_relation_matching=
                maximum_number_of_single_word_matches_for_relation_matching,
                maximum_number_of_single_word_matches_for_embedding_matching=
                maximum_number_of_single_word_matches_for_embedding_matching)

    def close(self):
        for worker in self._workers:
            worker.terminate()

class Worker:
    """Worker implementation used by *MultiprocessingManager*.
    """

    def listen(self, semantic_analyzer, structural_matcher, input_queue, worker_label):
        semantic_analyzer.reload_model() # necessary to avoid neuralcoref MemoryError on Linux
        indexed_documents = {}
        while(True):
            method, args, reply_queue = input_queue.get()
            try:
                reply = method(semantic_analyzer, structural_matcher, indexed_documents, *args)
            except Exception as err:
                reply_queue.put((worker_label, err))
                break
            except:
                err_identifier = str(sys.exc_info()[0])
                reply_queue.put((worker_label, err_identifier))
                break
            reply_queue.put((worker_label, reply))

    def worker_parse_and_register_document(self, semantic_analyzer, structural_matcher,
            indexed_documents, document_text, label):
        doc = semantic_analyzer.parse(document_text)
        indexed_document = structural_matcher.index_document(doc)
        indexed_documents[label] = indexed_document
        return ' '.join(('Parsed and registered document', label))

    def worker_deserialize_and_register_document(self, semantic_analyzer, structural_matcher,
            indexed_documents, document, label):
        doc = semantic_analyzer.from_serialized_string(document)
        indexed_document = structural_matcher.index_document(doc)
        indexed_documents[label] = indexed_document
        return ' '.join(('Deserialized and registered document', label))

    def worker_topic_match_documents_returning_dictionaries_against(self, semantic_analyzer,
            structural_matcher, indexed_documents, text_to_match,
            maximum_activation_distance, relation_score, reverse_only_relation_score,
            single_word_score, single_word_any_tag_score, overlapping_relation_multiplier,
            embedding_penalty,maximum_number_of_single_word_matches_for_relation_matching,
            maximum_number_of_single_word_matches_for_embedding_matching,
            sideways_match_extent, only_one_result_per_document, number_of_results,
            document_label_filter, tied_result_quotient):
        if len(indexed_documents) == 0:
            return []
        topic_matcher = TopicMatcher(semantic_analyzer = semantic_analyzer,
                structural_matcher = structural_matcher,
                indexed_documents = indexed_documents,
                maximum_activation_distance=maximum_activation_distance,
                relation_score=relation_score,
                reverse_only_relation_score=reverse_only_relation_score,
                single_word_score=single_word_score,
                single_word_any_tag_score=single_word_any_tag_score,
                overlapping_relation_multiplier=overlapping_relation_multiplier,
                embedding_penalty=embedding_penalty,
                maximum_number_of_single_word_matches_for_relation_matching =
                maximum_number_of_single_word_matches_for_relation_matching,
                maximum_number_of_single_word_matches_for_embedding_matching =
                maximum_number_of_single_word_matches_for_embedding_matching,
                sideways_match_extent=sideways_match_extent,
                only_one_result_per_document=only_one_result_per_document,
                number_of_results=number_of_results,
                document_label_filter=document_label_filter)
        topic_match_dicts = \
                topic_matcher.topic_match_documents_returning_dictionaries_against(text_to_match,
                tied_result_quotient=tied_result_quotient)
        return topic_match_dicts
