import copy
import sys
from .errors import *
from .structural_matching import StructuralMatcher
from .semantics import SemanticAnalyzerFactory
from .extensive_matching import *
from .consoles import HolmesConsoles

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
    perform_coreference_resolution -- *True*, *False* or *None* if coreference resolution
        should be performed depending on whether the model supports it. Defaults to *None*.
    debug -- a boolean value specifying whether debug representations should be outputted
        for parsed sentences. Defaults to *False*.
    """

    def __init__(self, model, *, overall_similarity_threshold=1.0,
            embedding_based_matching_on_root_words=False, ontology=None,
            perform_coreference_resolution=None, debug=False):
        self.semantic_analyzer = SemanticAnalyzerFactory().semantic_analyzer(model=model,
                perform_coreference_resolution=perform_coreference_resolution, debug=debug)
        if perform_coreference_resolution == None:
            perform_coreference_resolution = \
                    self.semantic_analyzer.model_supports_coreference_resolution()
        self._validate_options(overall_similarity_threshold,
                embedding_based_matching_on_root_words, perform_coreference_resolution)
        self.ontology = ontology
        self.debug = debug
        self.overall_similarity_threshold = overall_similarity_threshold
        self.embedding_based_matching_on_root_words = embedding_based_matching_on_root_words
        self.perform_coreference_resolution = perform_coreference_resolution
        self.structural_matcher = StructuralMatcher(self.semantic_analyzer, ontology,
                overall_similarity_threshold, embedding_based_matching_on_root_words,
                perform_coreference_resolution)
        self.documents = {}

    def _validate_options(self, overall_similarity_threshold,
            embedding_based_matching_on_root_words, perform_coreference_resolution):
        if overall_similarity_threshold < 0.0 or overall_similarity_threshold > 1.0:
            raise ValueError(
                    'overall_similarity_threshold must be between 0 and 1')
        if overall_similarity_threshold != 1.0 and not \
                self.semantic_analyzer.model_supports_enbeddings():
            raise ValueError(
                    'Model has no embeddings: overall_similarity_threshold must be 1.')
        if overall_similarity_threshold == 1.0 and embedding_based_matching_on_root_words:
            raise ValueError(
                    'overall_similarity_threshold is 1; embedding_based_matching_on_root_words must be False')
        if perform_coreference_resolution and not \
                self.semantic_analyzer.model_supports_coreference_resolution():
            raise ValueError(
                    'Model does not support coreference resolution: perform_coreference_resolution may not be True')

    def parse_and_register_document(self, document_text, label=''):
        """Parameters:

        document_text -- the raw document text.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases where single documents (user entries) are
            matched to predefined search phrases.
        """

        self.structural_matcher.register_document(self.semantic_analyzer.parse(document_text),
                label)

    def register_parsed_document(self, document, label=''):
        """Parameters:

        document -- a preparsed Holmes document.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases where single documents (user entries) are
            matched to predefined search phrases.
        """
        self.structural_matcher.register_document(document, label)

    def deserialize_and_register_document(self, document, label=''):
        """Parameters:

        document -- a Holmes document serialized using the *serialize_document()* function.
        label -- a label for the document which must be unique. Defaults to the empty string,
            which is intended for use cases where single documents (user entries) are
            matched to predefined search phrases.
        """
        if self.perform_coreference_resolution:
            raise SerializationNotSupportedError(self.semantic_analyzer.model)
        doc = self.semantic_analyzer.from_serialized_string(document)
        self.semantic_analyzer.debug_structures(doc) # only has effect when debug=True
        self.structural_matcher.register_document(doc, label)

    def remove_document(self, label):
        """Parameters:

        label -- the label of the document to be removed.
        """
        self.structural_matcher.remove_document(label)

    def remove_all_documents(self):
        self.structural_matcher.remove_all_documents()

    def document_labels(self):
        """Returns a list of the labels of the currently registered documents."""
        return self.structural_matcher.document_labels()

    def serialize_document(self, label):
        """Returns a serialized representation of a Holmes document that can be persisted to
            a file. If *label* is not the label of a registered document, *None* is returned
            instead.

        Parameters:

        label -- the label of the document to be serialized.
        """

        if self.perform_coreference_resolution:
            raise SerializationNotSupportedError(self.semantic_analyzer.model)
        doc = self.structural_matcher.get_document(label)
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
        self.structural_matcher.register_search_phrase(search_phrase_text, label)

    def remove_all_search_phrases(self):
        self.structural_matcher.remove_all_search_phrases()

    def remove_all_search_phrases_with_label(self, label):
        self.structural_matcher.remove_all_search_phrases_with_label(label)

    def match(self):
        """Matches the registered search phrases to the registered documents. Returns a list
            of *Match* objects sorted by their overall similarity measures in descending order.
            Should be called by applications wishing to retain references to the spaCy and
            Holmes information that was used to derive the matches.
        """
        return self.structural_matcher.match()

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
                                word_match.document_token),
                        'match_type': word_match.type,
                        'similarity_measure': word_match.similarity_measure,
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
        doc = self.semantic_analyzer.parse(entry)
        return self._build_match_dictionaries(
                self.structural_matcher.match_search_phrases_against(doc))

    def match_documents_against(self, search_phrase):
        """Matches the registered documents against a single search phrase
            supplied to the method and returns dictionaries describing any matches.
        """
        return self._build_match_dictionaries(
                self.structural_matcher.match_documents_against(search_phrase))

    def topic_match_documents_against(self, text_to_match, *, maximum_activation_distance=75,
            relation_score=30, single_word_score=5, overlapping_relation_multiplier=1.5,
                overlap_memory_size=10, maximum_activation_value=1000, sideways_match_extent=100,
                only_one_result_per_document=False, number_of_results=10):
        """Returns the results of a topic match between an entered text and the loaded documents.

        Properties:

        text_to_match -- the text to match against the loaded documents.
        maximum_activation_distance -- the number of words it takes for a previous activation to
            reduce to zero when the library is reading through a document.
        relation_score -- the activation score added when a two-word relation is matched.
        single_word_score -- the activation score added when a single word is matched.
        overlapping_relation_multiplier -- the value by which the activation score is multiplied
            when two relations were matched and the matches involved a common document word.
        overlap_memory_size -- the size of the memory for previous matches that is taken into
            consideration when searching for overlaps (matches are sorted according to the head
            word, and the dependent word that overlaps may be removed from the head word by
            some distance within the document text).
        maximum_activation_value -- the maximum permissible activation value.
        sideways_match_extent -- the maximum number of words that may be incorporated into a
            topic match either side of the word where the activation peaked.
        only_one_result_per_document -- if 'True', prevents multiple results from being returned
            for the same document.
        number_of_results -- the number of topic match objects to return.
        """
        topic_matcher = TopicMatcher(self,
                maximum_activation_distance=maximum_activation_distance,
                relation_score=relation_score,
                single_word_score=single_word_score,
                overlapping_relation_multiplier=overlapping_relation_multiplier,
                overlap_memory_size=overlap_memory_size,
                maximum_activation_value=maximum_activation_value,
                sideways_match_extent=sideways_match_extent,
                only_one_result_per_document=only_one_result_per_document,
                number_of_results=number_of_results)
        return topic_matcher.topic_match_documents_against(text_to_match)

    def _new_supervised_topic_structural_matcher(self):
        return StructuralMatcher(self.semantic_analyzer, self.ontology,
                overall_similarity_threshold = self.overall_similarity_threshold,
                embedding_based_matching_on_root_words =
                self.embedding_based_matching_on_root_words,
                perform_coreference_resolution = self.perform_coreference_resolution)
        # a private structural matcher is required to prevent the supervised topic
        # classification from matching against previously registered documents

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
                structural_matcher=self._new_supervised_topic_structural_matcher(),
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
        classifier = SupervisedTopicClassifier(self.semantic_analyzer,
                self._new_supervised_topic_structural_matcher(),
                None, verbose)
        classifier.deserialize_model(serialized_model)
        return classifier

    def start_chatbot_mode_console(self):
        """Starts a chatbot mode console enabling the matching of pre-registered search phrases
            to documents (chatbot entries) entered ad-hoc by the user.
        """
        holmes_consoles = HolmesConsoles(self)
        holmes_consoles.start_chatbot_mode()

    def start_search_mode_console(self, only_one_topic_match_per_document=False):
        """Starts a search mode console enabling the matching of pre-registered documents
            to search phrases and topic-matching phrases entered ad-hoc by the user.

            Parameters:

            only_one_topic_match_per_document -- if 'True', prevents multiple topic match
            results from being returned for the same document.
        """
        holmes_consoles = HolmesConsoles(self)
        holmes_consoles.start_search_mode(only_one_topic_match_per_document)
