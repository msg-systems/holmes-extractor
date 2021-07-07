import uuid
import statistics
import jsonpickle
from scipy.sparse import dok_matrix
from sklearn.neural_network import MLPClassifier
from .parsing import SemanticMatchingHelperFactory
from .errors import WrongModelDeserializationError, FewerThanTwoClassificationsError, \
        DuplicateDocumentError, NoPhraseletsAfterFilteringError, \
        EmbeddingThresholdGreaterThanRelationThresholdError, \
        IncompatibleAnalyzeDerivationalMorphologyDeserializationError

class SupervisedTopicTrainingUtils:

    def __init__(self, overlap_memory_size, oneshot):
        self.overlap_memory_size = overlap_memory_size
        self.oneshot = oneshot

    def get_labels_to_classification_frequencies_dict(
            self, *, matches, labels_to_classifications_dict):
        """ Builds a dictionary from search phrase (phraselet) labels to classification
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
                    labels_to_frequencies_dict[search_phrase_label] = classification_frequency_dict
                else:
                    classification_frequency_dict = labels_to_frequencies_dict[search_phrase_label]
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
            return len(match.word_matches) > 1 and \
                len(
                    [
                        word_match for word_match in match.word_matches if
                        len(word_match.document_token._.holmes.subwords) > 0 and
                        word_match.document_subword is None]
                ) > 0

        labels_to_frequencies_dict = {}
        matches = [
            match for match in matches if not
            relation_match_involves_whole_word_containing_subwords(match)]
        matches = sorted(
            matches, key=lambda match: (
                match.document_label, match.index_within_document,
                match.get_subword_index_for_sorting()))
        for index, match in enumerate(matches):
            if self.oneshot:
                if ('this_document_label' not in locals()) or \
                        this_document_label != match.document_label:
                    this_document_label = match.document_label
                    search_phrases_added_for_this_document = set()
                if match.search_phrase_label not in search_phrases_added_for_this_document:
                    increment(match.search_phrase_label, match.document_label)
                    search_phrases_added_for_this_document.add(match.search_phrase_label)
            else:
                increment(match.search_phrase_label, match.document_label)
            if not match.from_single_word_phraselet:
                previous_match_index = index
                number_of_analyzed_matches_counter = 0
                while previous_match_index > 0 and number_of_analyzed_matches_counter \
                        <= self.overlap_memory_size:
                    previous_match_index -= 1
                    previous_match = matches[previous_match_index]
                    if previous_match.document_label != match.document_label:
                        break
                    if previous_match.from_single_word_phraselet:
                        continue
                    if previous_match.search_phrase_label == match.search_phrase_label:
                        continue # otherwise coreference resolution leads to phrases being
                                 # combined with themselves
                    number_of_analyzed_matches_counter += 1
                    previous_word_match_doc_indexes = [
                        word_match.get_document_index() for word_match in
                        previous_match.word_matches]
                    for word_match in match.word_matches:
                        if word_match.get_document_index() in previous_word_match_doc_indexes:
                            # the same word is involved in both matches, so combine them
                            # into a new label
                            label_parts = sorted((
                                previous_match.search_phrase_label, match.search_phrase_label))
                            combined_label = '/'.join((label_parts[0], label_parts[1]))
                            if self.oneshot:
                                if combined_label not in search_phrases_added_for_this_document:
                                    increment(combined_label, match.document_label)
                                    search_phrases_added_for_this_document.add(combined_label)
                            else:
                                increment(combined_label, match.document_label)
        return labels_to_frequencies_dict

    def record_matches(
            self, *, phraselet_labels_to_search_phrases, linguistic_object_factory,
            structural_matcher, sorted_label_dict, doc_label, doc, matrix, row_index, verbose):
        """ Matches a document against the currently stored phraselets and records the matches
            in a matrix.

            Parameters:

            phraselet_labels_to_search_phrases -- a dictionary from search phrase (phraselet)
                labels to search phrase objects.
            linguistic_object_factory -- the linguistic object factory to use.
            structural_matcher -- the structural matcher to use for comparisons.
            sorted_label_dict -- a dictionary from search phrase (phraselet) labels to their own
                alphabetic sorting indexes.
            doc_label -- the document label, or 'None' if there is none.
            doc -- the document to be matched.
            matrix -- the matrix within which to record the matches.
            row_index -- the row number within the matrix corresponding to the document.
            verbose -- if 'True', matching information is outputted to the console.
        """
        indexed_document = linguistic_object_factory.index_document(doc)
        indexed_documents = {doc_label:indexed_document}
        found = False
        for label, occurrences in \
                self.get_labels_to_classification_frequencies_dict(
                    matches=structural_matcher.match(
                        indexed_documents=indexed_documents,
                        search_phrases=phraselet_labels_to_search_phrases.values(),
                        output_document_matching_message_to_console=verbose,
                        match_depending_on_single_words=None,
                        compare_embeddings_on_root_words=False,
                        compare_embeddings_on_non_root_words=True,
                        document_labels_to_indexes_for_reverse_matching_sets=None,
                        document_labels_to_indexes_for_embedding_reverse_matching_sets=None),
                    labels_to_classifications_dict=None
                ).items():
            if self.oneshot:
                occurrences = 1
            if label in sorted_label_dict: # may not be the case for compound labels
                label_index = sorted_label_dict[label]
                matrix[row_index, label_index] = occurrences
                found = True
        return found

class SupervisedTopicTrainingBasis:
    """ Holder object for training documents and their classifications from which one or more
        'SupervisedTopicModelTrainer' objects can be derived. This class is *NOT* threadsafe.
    """
    def __init__(
            self, *, linguistic_object_factory, structural_matcher, classification_ontology,
            overlap_memory_size, oneshot, match_all_words, verbose):
        """ Parameters:

            linguistic_object_factory -- the linguistic object factory to use
            structural_matcher -- the structural matcher to use.
            classification_ontology -- an Ontology object incorporating relationships between
                classification labels.
            overlap_memory_size -- how many non-word phraselet matches to the left should be
                checked for words in common with a current match.
            oneshot -- whether the same word or relationship matched multiple times should be
                counted once only (value 'True') or multiple times (value 'False')
            match_all_words -- whether all single words should be taken into account
                (value 'True') or only single words with noun tags (value 'False')
            verbose -- if 'True', information about training progress is outputted to the console.
        """
        self.linguistic_object_factory = linguistic_object_factory
        self.structural_matcher = structural_matcher
        self.semantic_analyzer = linguistic_object_factory.semantic_analyzer
        self.semantic_matching_helper = linguistic_object_factory.semantic_matching_helper
        self.classification_ontology = classification_ontology
        self._utils = SupervisedTopicTrainingUtils(overlap_memory_size, oneshot)
        self._match_all_words = match_all_words
        self.verbose = verbose

        self.training_documents = {}
        self.training_documents_labels_to_classifications_dict = {}
        self.additional_classification_labels = set()
        self.classification_implication_dict = {}
        self.labels_to_classification_frequencies = None
        self.phraselet_labels_to_phraselet_infos = {}
        self.classifications = None

    def parse_and_register_training_document(self, text, classification, label=None):
        """ Parses and registers a document to use for training.

            Parameters:

            text -- the document text
            classification -- the classification label
            label -- a label with which to identify the document in verbose training output,
                or 'None' if a random label should be assigned.
        """
        self.register_training_document(self.semantic_analyzer.parse(text), classification, label)

    def register_training_document(self, doc, classification, label):
        """ Registers a pre-parsed document to use for training.

            Parameters:

            doc -- the document
            classification -- the classification label
            label -- a label with which to identify the document in verbose training output,
                or 'None' if a random label should be assigned.
        """
        if self.labels_to_classification_frequencies is not None:
            raise RuntimeError(
                "register_training_document() may not be called once prepare() has been called")
        if label is None:
            label = str(uuid.uuid4())
        if label in self.training_documents:
            raise DuplicateDocumentError(label)
        if self.verbose:
            print('Registering document', label)
        indexed_document = self.linguistic_object_factory.index_document(doc)
        self.training_documents[label] = indexed_document
        self.linguistic_object_factory.add_phraselets_to_dict(
            doc,
            phraselet_labels_to_phraselet_infos=
            self.phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors=True,
            match_all_words=self._match_all_words,
            ignore_relation_phraselets=False,
            include_reverse_only=False,
            stop_lemmas=self.semantic_matching_helper.\
            supervised_document_classification_phraselet_stop_lemmas,
            stop_tags=self.semantic_matching_helper.topic_matching_phraselet_stop_tags,
            reverse_only_parent_lemmas=None,
            words_to_corpus_frequencies=None,
            maximum_corpus_frequency=None)
        self.training_documents_labels_to_classifications_dict[label] = classification

    def register_additional_classification_label(self, label):
        """ Register an additional classification label which no training document has explicitly
            but that should be assigned to documents whose explicit labels are related to the
            additional classification label via the classification ontology.
        """
        if self.labels_to_classification_frequencies is not None:
            raise RuntimeError(
                "register_additional_classification_label() may not be called once prepare() has "\
                " been called")
        if self.classification_ontology is not None and \
                self.classification_ontology.contains(label):
            self.additional_classification_labels.add(label)

    def prepare(self):
        """ Matches the phraselets derived from the training documents against the training
            documents to generate frequencies that also include combined labels, and examines the
            explicit classification labels, the additional classification labels and the
            classification ontology to derive classification implications.

            Once this method has been called, the instance no longer accepts new training documents
            or additional classification labels.
        """
        if self.labels_to_classification_frequencies is not None:
            raise RuntimeError(
                "prepare() may only be called once")
        if self.verbose:
            print('Matching documents against all phraselets')
        search_phrases = self.linguistic_object_factory.create_search_phrases_from_phraselet_infos(
            self.phraselet_labels_to_phraselet_infos.values()).values()
        self.labels_to_classification_frequencies = self._utils.\
            get_labels_to_classification_frequencies_dict(
                matches=self.structural_matcher.match(
                    indexed_documents=self.training_documents,
                    search_phrases=search_phrases,
                    output_document_matching_message_to_console=self.verbose,
                    match_depending_on_single_words=None,
                    compare_embeddings_on_root_words=False,
                    compare_embeddings_on_non_root_words=True,
                    document_labels_to_indexes_for_reverse_matching_sets=None,
                    document_labels_to_indexes_for_embedding_reverse_matching_sets=None),
                labels_to_classifications_dict=
                self.training_documents_labels_to_classifications_dict)
        self.classifications = sorted(set(
            self.training_documents_labels_to_classifications_dict.values()
            ).union(self.additional_classification_labels))
        if len(self.classifications) < 2:
            raise FewerThanTwoClassificationsError(len(self.classifications))
        if self.classification_ontology is not None:
            for parent in self.classifications:
                for child in self.classifications:
                    if self.classification_ontology.matches(parent, child):
                        if child in self.classification_implication_dict.keys():
                            self.classification_implication_dict[child].append(parent)
                        else:
                            self.classification_implication_dict[child] = [parent]

    def train(
            self, *, minimum_occurrences=4, cv_threshold=1.0, mlp_activation='relu',
            mlp_solver='adam', mlp_learning_rate='constant', mlp_learning_rate_init=0.001,
            mlp_max_iter=200, mlp_shuffle=True, mlp_random_state=42, overlap_memory_size=10,
            hidden_layer_sizes=None):
        """ Trains a model based on the prepared state.

            Parameters:

            minimum_occurrences -- the minimum number of times a word or relationship has to
                occur in the context of at least one single classification for the phraselet
                to be accepted into the final model.
            cv_threshold -- the minimum coefficient of variation a word or relationship has
                to occur with respect to explicit classification labels for the phraselet to be
                accepted into the final model.
            mlp_* -- see https://scikit-learn.org/stable/modules/generated/
            sklearn.neural_network.MLPClassifier.html.
            overlap_memory_size -- No longer has any effect - the value defined in __init__()
                is used instead. Retained for backwards compatibility.
            hidden_layer_sizes -- a tuple containing the number of neurons in each hidden layer, or
                'None' if the topology should be determined automatically.
        """

        if self.labels_to_classification_frequencies is None:
            raise RuntimeError("train() may only be called after prepare() has been called")
        return SupervisedTopicModelTrainer(
            training_basis=self,
            semantic_analyzer=self.semantic_analyzer,
            linguistic_object_factory=self.linguistic_object_factory,
            structural_matcher=self.structural_matcher,
            labels_to_classification_frequencies=self.labels_to_classification_frequencies,
            phraselet_infos=self.phraselet_labels_to_phraselet_infos.values(),
            minimum_occurrences=minimum_occurrences,
            cv_threshold=cv_threshold,
            mlp_activation=mlp_activation,
            mlp_solver=mlp_solver,
            mlp_learning_rate=mlp_learning_rate,
            mlp_learning_rate_init=mlp_learning_rate_init,
            mlp_max_iter=mlp_max_iter,
            mlp_shuffle=mlp_shuffle,
            mlp_random_state=mlp_random_state,
            hidden_layer_sizes=hidden_layer_sizes,
            utils=self._utils
        )

class SupervisedTopicModelTrainer:
    """ Worker object used to train and generate models. This class is *NOT* threadsafe."""

    def __init__(
            self, *, training_basis, semantic_analyzer,
            linguistic_object_factory, structural_matcher, labels_to_classification_frequencies,
            phraselet_infos, minimum_occurrences, cv_threshold, mlp_activation, mlp_solver,
            mlp_learning_rate, mlp_learning_rate_init, mlp_max_iter, mlp_shuffle, mlp_random_state,
            hidden_layer_sizes, utils):

        self._utils = utils
        self._semantic_analyzer = linguistic_object_factory.semantic_analyzer
        self._linguistic_object_factory = linguistic_object_factory
        self._structural_matcher = structural_matcher
        self._training_basis = training_basis
        self._minimum_occurrences = minimum_occurrences
        self._cv_threshold = cv_threshold
        self._labels_to_classification_frequencies, self._phraselet_infos = self._filter(
            labels_to_classification_frequencies, phraselet_infos)

        if len(self._phraselet_infos) == 0:
            raise NoPhraseletsAfterFilteringError(
                ''.join((
                    'minimum_occurrences: ', str(minimum_occurrences), '; cv_threshold: ',
                    str(cv_threshold)))
                )

        phraselet_labels_to_search_phrases = \
            self._linguistic_object_factory.create_search_phrases_from_phraselet_infos(
                self._phraselet_infos)
        self._sorted_label_dict = {}
        for index, label in enumerate(sorted(self._labels_to_classification_frequencies.keys())):
            self._sorted_label_dict[label] = index
        self._input_matrix = dok_matrix((
            len(self._training_basis.training_documents), len(self._sorted_label_dict)))
        self._output_matrix = dok_matrix((
            len(self._training_basis.training_documents),
            len(self._training_basis.classifications)))

        if self._training_basis.verbose:
            print('Matching documents against filtered phraselets')
        for index, document_label in enumerate(
                sorted(self._training_basis.training_documents.keys())):
            self._utils.record_matches(
                linguistic_object_factory=self._linguistic_object_factory,
                structural_matcher=self._structural_matcher,
                phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                sorted_label_dict=self._sorted_label_dict,
                doc_label=document_label,
                doc=self._training_basis.training_documents[document_label].doc,
                matrix=self._input_matrix,
                row_index=index,
                verbose=self._training_basis.verbose)
            self._record_classifications_for_training(document_label, index)
        self._hidden_layer_sizes = hidden_layer_sizes
        if self._hidden_layer_sizes is None:
            start = len(self._sorted_label_dict)
            step = (len(self._training_basis.classifications) - len(self._sorted_label_dict)) / 3
            self._hidden_layer_sizes = (start, int(start+step), int(start+(2*step)))
        if self._training_basis.verbose:
            print('Hidden layer sizes:', self._hidden_layer_sizes)
        self._mlp = MLPClassifier(
            activation=mlp_activation,
            solver=mlp_solver,
            hidden_layer_sizes=self._hidden_layer_sizes,
            learning_rate=mlp_learning_rate,
            learning_rate_init=mlp_learning_rate_init,
            max_iter=mlp_max_iter,
            shuffle=mlp_shuffle,
            verbose=self._training_basis.verbose,
            random_state=mlp_random_state)
        self._mlp.fit(self._input_matrix, self._output_matrix)
        if self._training_basis.verbose and self._mlp.n_iter_ < mlp_max_iter:
            print('MLP neural network converged after', self._mlp.n_iter_, 'iterations.')

    def _filter(self, labels_to_classification_frequencies, phraselet_infos):
        """ Filters the phraselets in memory based on minimum_occurrences and cv_threshold. """

        accepted = 0
        under_minimum_occurrences = 0
        under_minimum_cv = 0
        new_labels_to_classification_frequencies = {}
        for label, classification_frequencies in labels_to_classification_frequencies.items():
            at_least_minimum = False
            working_classification_frequencies = classification_frequencies.copy()
            for classification in working_classification_frequencies:
                if working_classification_frequencies[classification] >= self._minimum_occurrences:
                    at_least_minimum = True
            if not at_least_minimum:
                under_minimum_occurrences += 1
                continue
            frequency_list = list(working_classification_frequencies.values())
            # We only want to take explicit classification labels into account, i.e. ignore the
            # classification ontology.
            number_of_classification_labels = \
                len(set(
                    self._training_basis.training_documents_labels_to_classifications_dict.values())
                    )
            frequency_list.extend([0] * number_of_classification_labels)
            frequency_list = frequency_list[:number_of_classification_labels]
            if statistics.pstdev(frequency_list) / statistics.mean(frequency_list) >= \
                    self._cv_threshold:
                accepted += 1
                new_labels_to_classification_frequencies[label] = classification_frequencies
            else:
                under_minimum_cv += 1
        if self._training_basis.verbose:
            print(
                'Filtered: accepted', accepted, '; removed minimum occurrences',
                under_minimum_occurrences, '; removed cv threshold',
                under_minimum_cv)
        new_phraselet_infos = [
            phraselet_info for phraselet_info in phraselet_infos if
            phraselet_info.label in new_labels_to_classification_frequencies.keys()]
        return new_labels_to_classification_frequencies, new_phraselet_infos

    def _record_classifications_for_training(self, document_label, index):
        classification = self._training_basis.training_documents_labels_to_classifications_dict[
            document_label]
        classification_index = self._training_basis.classifications.index(classification)
        self._output_matrix[index, classification_index] = 1
        if classification in self._training_basis.classification_implication_dict:
            for implied_classification in \
                    self._training_basis.classification_implication_dict[classification]:
                implied_classification_index = self._training_basis.classifications.index(
                    implied_classification)
                self._output_matrix[index, implied_classification_index] = 1

    def classifier(self):
        """ Returns a supervised topic classifier which contains no explicit references to the
            training data and that can be serialized.
        """
        self._mlp.verbose = False # we no longer require output once we are using the model
                                # to classify new documents
        model = SupervisedTopicClassifierModel(
            semantic_analyzer_model=self._semantic_analyzer.model,
            structural_matcher_ontology=self._structural_matcher.ontology,
            phraselet_infos=self._phraselet_infos,
            mlp=self._mlp,
            sorted_label_dict=self._sorted_label_dict,
            classifications=self._training_basis.classifications,
            overlap_memory_size=self._utils.overlap_memory_size,
            oneshot=self._utils.oneshot,
            analyze_derivational_morphology=
            self._structural_matcher.analyze_derivational_morphology)
        return SupervisedTopicClassifier(
            self._semantic_analyzer, self._linguistic_object_factory,
            self._structural_matcher, model, self._training_basis.verbose)

class SupervisedTopicClassifierModel:
    """ A serializable classifier model.

        Parameters:

        semantic_analyzer_model -- a string specifying the spaCy model with which this instance
            was generated and with which it must be used.
        structural_matcher_ontology -- the ontology used for matching documents against this model
            (not the classification ontology!)
        phraselet_infos -- the phraselets used for structural matching
        mlp -- the neural network
        sorted_label_dict -- a dictionary from search phrase (phraselet) labels to their own
            alphabetic sorting indexes.
        classifications -- an ordered list of classification labels corresponding to the
            neural network outputs
        overlap_memory_size -- how many non-word phraselet matches to the left should be
            checked for words in common with a current match.
        oneshot -- whether the same word or relationship matched multiple times should be
            counted once only (value 'True') or multiple times (value 'False')
        analyze_derivational_morphology -- the value of this manager parameter that was in force
            when the model was built. The same value has to be in force when the model is
            deserialized and reused.
    """

    def __init__(
            self, semantic_analyzer_model, structural_matcher_ontology,
            phraselet_infos, mlp, sorted_label_dict, classifications, overlap_memory_size,
            oneshot, analyze_derivational_morphology):
        self.semantic_analyzer_model = semantic_analyzer_model
        self.structural_matcher_ontology = structural_matcher_ontology
        self.phraselet_infos = phraselet_infos
        self.mlp = mlp
        self.sorted_label_dict = sorted_label_dict
        self.classifications = classifications
        self.overlap_memory_size = overlap_memory_size
        self.oneshot = oneshot
        self.analyze_derivational_morphology = analyze_derivational_morphology

class SupervisedTopicClassifier:
    """Classifies new documents based on a pre-trained model."""

    def __init__(self, semantic_analyzer, linguistic_object_factory, structural_matcher, model,
            verbose):
        self.semantic_analyzer = semantic_analyzer
        self.linguistic_object_factory = linguistic_object_factory
        self.structural_matcher = structural_matcher
        self.model = model
        self.verbose = verbose
        self.utils = SupervisedTopicTrainingUtils(model.overlap_memory_size, model.oneshot)
        if self.semantic_analyzer.model != model.semantic_analyzer_model:
            raise WrongModelDeserializationError(model.semantic_analyzer_model)
        if hasattr(model, 'analyze_derivational_morphology'): # backwards compatibility
            analyze_derivational_morphology = model.analyze_derivational_morphology
        else:
            analyze_derivational_morphology = False
        if self.structural_matcher.analyze_derivational_morphology != \
                analyze_derivational_morphology:
            print(
                ''.join((
                    'manager: ', str(self.structural_matcher.analyze_derivational_morphology),
                    '; model: ', str(analyze_derivational_morphology))))
            raise IncompatibleAnalyzeDerivationalMorphologyDeserializationError(
                ''.join((
                    'manager: ', str(self.structural_matcher.analyze_derivational_morphology),
                    '; model: ', str(analyze_derivational_morphology))))
        self.structural_matcher.ontology = model.structural_matcher_ontology
        self.linguistic_object_factory.ontology = model.structural_matcher_ontology
        self.semantic_matching_helper = self.structural_matcher.semantic_matching_helper
        self.semantic_matching_helper.ontology = model.structural_matcher_ontology
        self.semantic_matching_helper.ontology_reverse_derivational_dict = \
            self.linguistic_object_factory.get_ontology_reverse_derivational_dict()
        self.phraselet_labels_to_search_phrases = \
            self.linguistic_object_factory.create_search_phrases_from_phraselet_infos(
                model.phraselet_infos)

    def parse_and_classify(self, text):
        """ Returns a list containing zero, one or many document classifications. Where more
            than one classifications are returned, the labels are ordered by decreasing
            probability.

            Parameter:

            text -- the text to parse and classify.
        """
        return self.classify(self.semantic_analyzer.parse(text))

    def classify(self, doc):
        """ Returns a list containing zero, one or many document classifications. Where more
            than one classifications are returned, the labels are ordered by decreasing
            probability.

            Parameter:

            doc -- the pre-parsed document to classify.
        """

        if self.model is None:
            raise RuntimeError('No model defined')
        new_document_matrix = dok_matrix((1, len(self.model.sorted_label_dict)))
        if not self.utils.record_matches(
                linguistic_object_factory=self.linguistic_object_factory,
                structural_matcher=self.structural_matcher,
                phraselet_labels_to_search_phrases=self.phraselet_labels_to_search_phrases,
                sorted_label_dict=self.model.sorted_label_dict,
                doc=doc,
                doc_label='',
                matrix=new_document_matrix,
                row_index=0,
                verbose=self.verbose):
            return []
        else:
            classification_indexes = self.model.mlp.predict(new_document_matrix).nonzero()[1]
            if len(classification_indexes) > 1:
                probabilities = self.model.mlp.predict_proba(new_document_matrix)
                classification_indexes = sorted(
                    classification_indexes, key=lambda index: 1-probabilities[0, index])
            return list(map(
                lambda index: self.model.classifications[index], classification_indexes))

    def serialize_model(self):
        return jsonpickle.encode(self.model)
