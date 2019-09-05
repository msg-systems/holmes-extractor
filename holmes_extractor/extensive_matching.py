import collections
import jsonpickle
import uuid
import statistics
from scipy.sparse import dok_matrix
from sklearn.neural_network import MLPClassifier
from .errors import WrongModelDeserializationError, FewerThanTwoClassificationsError, \
        DuplicateDocumentError, NoPhraseletsAfterFilteringError

class TopicMatch:
    """A topic match between some text and part of a document.

    Properties:

    document_label -- the document label.
    start_index -- the start index of the topic match within the document.
    end_index -- the end index of the topic match within the document.
    sentences_start_index -- the start index within the document of the sentence that contains
        'start_index'
    sentences_end_index -- the end index within the document of the sentence that contains
        'end_index'
    relative_start_index -- the start index of the topic match relative to 'sentences_start_index'
    relative_end_index -- the end index of the topic match relative to 'sentences_start_index'
    score -- the similarity score of the topic match
    text -- the text between 'sentences_start_index' and 'sentences_end_index'
    """

    def __init__(self, document_label, start_index, end_index, sentences_start_index,
            sentences_end_index, score, text):
        self.document_label = document_label
        self.start_index = start_index
        self.end_index = end_index
        self.sentences_start_index = sentences_start_index
        self.sentences_end_index = sentences_end_index
        self.score = score
        self.text = text

    @property
    def relative_start_index(self):
        return self.start_index - self.sentences_start_index

    @property
    def relative_end_index(self):
        return self.end_index - self.sentences_start_index

class TopicMatcher:
    """A topic matcher object. See manager.py for details of the properties."""

    def __init__(self, holmes, *, maximum_activation_distance, relation_score,
            single_word_score, overlapping_relation_multiplier,
            overlap_memory_size, maximum_activation_value, sideways_match_extent,
            only_one_result_per_document, number_of_results):
        self._holmes = holmes
        self._semantic_analyzer = holmes.semantic_analyzer
        self.structural_matcher = holmes.structural_matcher
        self._ontology = holmes.structural_matcher.ontology
        self.maximum_activation_distance = maximum_activation_distance
        self.relation_score = relation_score
        self.single_word_score = single_word_score
        self.overlapping_relation_multiplier = overlapping_relation_multiplier
        self.overlap_memory_size = overlap_memory_size
        self.maximum_activation_value = maximum_activation_value
        self.sideways_match_extent = sideways_match_extent
        self.only_one_result_per_document = only_one_result_per_document
        self.number_of_results = number_of_results

    def topic_match_documents_against(self, text_to_match):
        """ Performs a topic match against the loaded documents.

        Property:

        text_to_match -- the text to match against the documents.
        """
        doc = self._semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_search_phrases = {}
        self.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                replace_with_hypernym_ancestors=False,
                match_all_words = False,
                returning_serialized_phraselets = False)
        if len(phraselet_labels_to_search_phrases) == 0:
            return []
        # First get the structural matches sorted by position
        position_sorted_structural_matches = \
                sorted(self.structural_matcher.match_documents_against_search_phrase_list(
                        phraselet_labels_to_search_phrases.values(), False), key=lambda match:
                (match.document_label, match.index_within_document))
        if len(position_sorted_structural_matches) == 0:
            # We found nothing, so we try again with all single words (not just nouns)
            self.structural_matcher.add_phraselets_to_dict(doc,
                    phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                    replace_with_hypernym_ancestors=False,
                    match_all_words = True,
                    returning_serialized_phraselets = False)
            position_sorted_structural_matches = \
                    sorted(self.structural_matcher.match_documents_against_search_phrase_list(
                            phraselet_labels_to_search_phrases.values(), False), key=lambda match:
                    (match.document_label, match.index_within_document))
        # Read through the documents measuring the activation based on where
        # in the document structural matches were found
        score_sorted_structural_matches = self.perform_activation_scoring(
                position_sorted_structural_matches)
        return self.get_topic_matches(score_sorted_structural_matches,
                position_sorted_structural_matches)

    def perform_activation_scoring(self, position_sorted_structural_matches):
        """
        Read through the documents measuring the activation based on where
        in the document structural matches were found.
        """
        current_document_label = None
        last_search_phrase_label = None
        for index, match in enumerate(position_sorted_structural_matches):
            match.original_index_within_list = index # store for later use after resorting
            if match.document_label != current_document_label or index == 0:
                current_document_label = match.document_label
                current_activation_score = 0
                current_unconstrained_activation_score = 0
                last_search_phrase_label = None
                # The deque has to be twice the size of the topic matches to track because
                # each topic match under consideration involves two tokens
                previous_topic_matches = collections.deque(maxlen=self.overlap_memory_size * 2)
            else:
                distance_to_last_match = match.index_within_document - \
                        position_sorted_structural_matches[index-1].index_within_document
                tailoff_quotient = distance_to_last_match / self.maximum_activation_distance
                if tailoff_quotient > 1.0:
                    tailoff_quotient = 1.0
                current_activation_score = current_activation_score * (1 - tailoff_quotient)
                current_unconstrained_activation_score = current_unconstrained_activation_score * \
                        (1 - tailoff_quotient)
            adjusted_relation_score = self.relation_score * float(match.overall_similarity_measure)
            adjusted_single_word_score = self.single_word_score * \
                    float(match.overall_similarity_measure)
            if not match.from_single_word_phraselet:
                current_activation_score += adjusted_relation_score
                current_unconstrained_activation_score += adjusted_relation_score
            elif last_search_phrase_label != match.search_phrase_label:
                current_activation_score += adjusted_single_word_score
                current_unconstrained_activation_score += adjusted_single_word_score
            else:
                # If the same single word match occurs repeatedly, we do not allow the activation to
                # increase further
                current_activation_score = max(current_activation_score, adjusted_single_word_score)
                current_unconstrained_activation_score += \
                        max(current_unconstrained_activation_score,
                        adjusted_single_word_score)
            if not match.from_single_word_phraselet:
                for word_match in match.word_matches:
                    if word_match.document_token.i in previous_topic_matches:
                        # We have matched a larger structure from the text to match
                        current_activation_score *= self.overlapping_relation_multiplier
                        current_unconstrained_activation_score *= \
                                self.overlapping_relation_multiplier
                    previous_topic_matches.append(word_match.document_token.i)
            if current_activation_score > self.maximum_activation_value:
                # We do not allow the activation to get too large. At the same time, we store
                # 'unconstrained_topic_score' to ensure the correct sorting of topic matches
                # where there are several with the maximum score.
                current_activation_score = self.maximum_activation_value
            last_search_phrase_label = match.search_phrase_label
            match.topic_score = current_activation_score
            match.unconstrained_topic_score = current_unconstrained_activation_score
        return sorted(position_sorted_structural_matches, key=lambda match: (0-match.topic_score,
                0-match.unconstrained_topic_score))

    def get_topic_matches(self, score_sorted_structural_matches,
            position_sorted_structural_matches):
        """Resort the matches starting with the highest (most active) and
            create topic match objects with information about the surrounding sentences.
        """

        def match_contained_within_existing_topic_match(topic_matches, match):
            for topic_match in topic_matches:
                if match.document_label == topic_match.document_label and \
                match.index_within_document >= topic_match.start_index and \
                match.index_within_document <= topic_match.end_index:
                    return True
            return False

        def alter_start_and_end_indexes_for_match(start_index, end_index, match,
                reference_match_index_within_document):
            if match.index_within_document < start_index and \
                    reference_match_index_within_document - match.index_within_document < \
                    self.sideways_match_extent:
                start_index = match.index_within_document
            for word_match in match.word_matches:
                if word_match.document_token.i < start_index and \
                        reference_match_index_within_document - word_match.document_token.i \
                        < self.sideways_match_extent:
                    start_index = word_match.document_token.i
            if match.index_within_document > end_index and match.index_within_document - \
                    reference_match_index_within_document < self.sideways_match_extent:
                end_index = match.index_within_document
            for word_match in match.word_matches:
                if word_match.document_token.i > end_index and \
                        word_match.document_token.i - reference_match_index_within_document \
                        < self.sideways_match_extent:
                    end_index = word_match.document_token.i
            return start_index, end_index

        if self.only_one_result_per_document:
            existing_document_labels = []
        topic_matches = []
        counter = 0
        for score_sorted_match in score_sorted_structural_matches:
            if counter >= self.number_of_results:
                break
            if match_contained_within_existing_topic_match(topic_matches,
                    score_sorted_match):
                continue
            if self.only_one_result_per_document and score_sorted_match.document_label \
                    in existing_document_labels:
                continue
            start_index, end_index = alter_start_and_end_indexes_for_match(
                    score_sorted_match.index_within_document,
                    score_sorted_match.index_within_document,
                    score_sorted_match, score_sorted_match.index_within_document)
            index_within_list = score_sorted_match.original_index_within_list - 1
            while index_within_list >= 0 and position_sorted_structural_matches[
                    index_within_list].document_label == score_sorted_match.document_label \
                    and position_sorted_structural_matches[index_within_list + 1].topic_score > \
                    self.single_word_score:
                    # index_within_list + 1: when a complex structure is matched, it will often
                    # begin with a single noun that should be included within the topic match
                    # indexes
                if match_contained_within_existing_topic_match(topic_matches,
                        position_sorted_structural_matches[
                        index_within_list]):
                    break
                if score_sorted_match.index_within_document - position_sorted_structural_matches[
                        index_within_list].index_within_document > self.sideways_match_extent:
                    break
                start_index, end_index = alter_start_and_end_indexes_for_match(
                        start_index, end_index,
                        position_sorted_structural_matches[index_within_list],
                        score_sorted_match.index_within_document
                )
                index_within_list -= 1
            index_within_list = score_sorted_match.original_index_within_list + 1
            while index_within_list + 1 <= len(score_sorted_structural_matches) and \
                    position_sorted_structural_matches[index_within_list].document_label == \
                    score_sorted_match.document_label and \
                    position_sorted_structural_matches[index_within_list].topic_score > \
                    self.single_word_score:
                if match_contained_within_existing_topic_match(topic_matches,
                        position_sorted_structural_matches[
                        index_within_list]):
                    break
                if position_sorted_structural_matches[
                        index_within_list].index_within_document - \
                        score_sorted_match.index_within_document > self.sideways_match_extent:
                    break
                start_index, end_index = alter_start_and_end_indexes_for_match(
                        start_index, end_index,
                        position_sorted_structural_matches[index_within_list],
                        score_sorted_match.index_within_document
                )
                index_within_list += 1
            relevant_sentences = [sentence for sentence in
                    self.structural_matcher.get_document(score_sorted_match.document_label).sents
                    if sentence.end >= start_index and sentence.start <= end_index]
            sentences_start_index = relevant_sentences[0].start
            sentences_end_index = relevant_sentences[-1].end - 1
            sentences_string = ' '.join(sentence.text.strip() for sentence in relevant_sentences)
            topic_matches.append(TopicMatch(score_sorted_match.document_label,
                    start_index, end_index, sentences_start_index, sentences_end_index,
                    score_sorted_match.topic_score, sentences_string))
            if self.only_one_result_per_document:
                existing_document_labels.append(score_sorted_match.document_label)
            counter += 1
        # If two matches have the same score, order them by length
        return sorted(topic_matches, key=lambda topic_match: (0-topic_match.score,
                topic_match.start_index - topic_match.end_index))

class SupervisedTopicTrainingUtils:

    def __init__(self, overlap_memory_size, oneshot):
        self.overlap_memory_size = overlap_memory_size
        self.oneshot = oneshot

    def get_labels_to_classification_frequencies_dict(self, *, matches,
            labels_to_classifications_dict):
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
            if labels_to_classifications_dict != None:
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

        labels_to_frequencies_dict = {}
        matches = sorted(matches,
                key=lambda match:(match.document_label, match.index_within_document))
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
                    previous_word_match_doc_indexes = list(map(lambda word_match:
                            word_match.document_token.i, previous_match.word_matches))
                    for word_match in match.word_matches:
                        if word_match.document_token.i in previous_word_match_doc_indexes:
                            # the same word is involved in both matches, so combine them
                            # into a new label
                            label_parts = sorted((previous_match.search_phrase_label,
                                    match.search_phrase_label))
                            combined_label = '/'.join((label_parts[0],
                                    label_parts[1]))
                            if self.oneshot:
                                if combined_label not in search_phrases_added_for_this_document:
                                    increment(combined_label, match.document_label)
                                    search_phrases_added_for_this_document.add(combined_label)
                            else:
                                increment(combined_label, match.document_label)
        return labels_to_frequencies_dict

    def record_matches(self, *, phraselet_labels_to_search_phrases, structural_matcher,
            sorted_label_dict, doc, doc_label, matrix, row_index, verbose):
        """ Matches a document against the currently stored phraselets and records the matches
            in a matrix.

            Parameters:

            structural_matcher -- the structural matcher object on which the phraselets are stored.
            phraselet_labels_to_search_phrases -- a dictionary from search phrase (phraselet)
                labels to search phrase objects.
            sorted_label_dict -- a dictionary from search phrase (phraselet) labels to their own
                alphabetic sorting indexes.
            doc -- the document to be matched.
            doc_label - the label of 'doc'.
            matrix -- the matrix within which to record the matches.
            row_index -- the row number within the matrix corresponding to the document.
            verbose -- if 'True', matching information is outputted to the console.
        """

        structural_matcher.remove_all_documents()
        structural_matcher.register_document(doc, doc_label)
        found = False
        for label, occurrences in \
                self.get_labels_to_classification_frequencies_dict(
                matches=structural_matcher.match_documents_against_search_phrase_list(
                        phraselet_labels_to_search_phrases.values(), verbose
                ),
                labels_to_classifications_dict=None).items():
            if self.oneshot:
                occurrences = 1
            if label in sorted_label_dict: # may not be the case for compound labels
                label_index = sorted_label_dict[label]
                matrix[row_index, label_index] = occurrences
                found = True
        return found

class SupervisedTopicTrainingBasis:
    """ Holder object for training documents and their classifications from which one or more
        'SupervisedTopicModelTrainer' objects can be derived. This class is not threadsafe.
    """
    def __init__(self, *, structural_matcher, classification_ontology, overlap_memory_size,
            oneshot, match_all_words, verbose):
        """ Parameters:

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
        self.semantic_analyzer = structural_matcher.semantic_analyzer
        self.structural_matcher = structural_matcher
        self.classification_ontology = classification_ontology
        self._utils = SupervisedTopicTrainingUtils(overlap_memory_size, oneshot)
        self._match_all_words = match_all_words
        self.verbose = verbose

        self.training_documents = {}
        self.training_documents_labels_to_classifications_dict = {}
        self.additional_classification_labels = set()
        self.classification_implication_dict = {}
        self.labels_to_classification_frequencies = None
        self.phraselet_labels_to_search_phrases = {}
        self.serialized_phraselets = []

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
        if self.labels_to_classification_frequencies != None:
            raise RuntimeError(
                    "register_training_document() may not be called once prepare() has been called")
        if label == None:
            label = str(uuid.uuid4())
        if label in self.training_documents:
            raise DuplicateDocumentError(label)
        if self.verbose:
            print('Registering document', label)
        self.training_documents[label] = doc
        self.serialized_phraselets.extend(
                self.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_search_phrases=
                self.phraselet_labels_to_search_phrases,
                replace_with_hypernym_ancestors=True,
                match_all_words=self._match_all_words,
                returning_serialized_phraselets = True))
        self.structural_matcher.register_document(doc, label)
        self.training_documents_labels_to_classifications_dict[label] = classification

    def register_additional_classification_label(self, label):
        """ Register an additional classification label which no training document has explicitly
            but that should be assigned to documents whose explicit labels are related to the
            additional classification label via the classification ontology.
        """
        if self.labels_to_classification_frequencies != None:
            raise RuntimeError(
                    "register_additional_classification_label() may not be called once prepare() has been called")
        if self.classification_ontology != None and self.classification_ontology.contains(label):
            self.additional_classification_labels.add(label)

    def prepare(self):
        """ Matches the phraselets derived from the training documents against the training
            documents to generate frequencies that also include combined labels, and examines the
            explicit classification labels, the additional classification labels and the
            classification ontology to derive classification implications.

            Once this method has been called, the instance no longer accepts new training documents
            or additional classification labels.
        """
        if self.labels_to_classification_frequencies != None:
            raise RuntimeError(
                    "prepare() may only be called once")
        if self.verbose:
            print('Matching documents against all phraselets')
        self.labels_to_classification_frequencies = self._utils.\
                get_labels_to_classification_frequencies_dict(
                matches=self.structural_matcher.match_documents_against_search_phrase_list(
                self.phraselet_labels_to_search_phrases.values(), self.verbose),
                labels_to_classifications_dict=
                self.training_documents_labels_to_classifications_dict)
        self.classifications = \
                sorted(set(self.training_documents_labels_to_classifications_dict.values()
                        ).union(self.additional_classification_labels))
        if len(self.classifications) < 2:
            raise FewerThanTwoClassificationsError(len(self.classifications))
        if self.classification_ontology != None:
            for classification in self.classifications:
                self.classification_ontology.add_to_dictionary(classification)
            for parent in self.classifications:
                for child in self.classifications:
                    if self.classification_ontology.matches(parent, child):
                        if child in self.classification_implication_dict.keys():
                            self.classification_implication_dict[child].append(parent)
                        else:
                            self.classification_implication_dict[child] = [parent]

    def train(self, *, minimum_occurrences=4, cv_threshold=1.0, mlp_activation='relu',
            mlp_solver='adam', mlp_learning_rate='constant', mlp_learning_rate_init=0.001,
            mlp_max_iter=200, mlp_shuffle=True, mlp_random_state=42, oneshot=True,
            overlap_memory_size=10, hidden_layer_sizes=None):
        """ Trains a model based on the prepared state.

            Parameters:

            minimum_occurrences -- the minimum number of times a word or relationship has to
                occur in the context of at least one single classification for the phraselet
                to be accepted into the final model.
            cv_threshold -- the minimum coefficient of variation a word or relationship has
                to occur with respect to explicit classification labels for the phraselet to be
                accepted into the final model.
            mlp_* -- see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html.
            oneshot -- whether the same word or relationship matched multiple times should be
                counted once only (value 'True') or multiple times (value 'False')
            overlap_memory_size -- how many non-word phraselet matches to the left should be
                checked for words in common with a current match.
            hidden_layer_sizes -- a list of the number of neurons in each hidden layer, or 'None'
                if the topology should be determined automatically.
        """

        if self.labels_to_classification_frequencies == None:
            raise RuntimeError(
                    "train() may only be called after prepare() has been called")
        return SupervisedTopicModelTrainer(
                training_basis = self,
                semantic_analyzer = self.semantic_analyzer,
                structural_matcher = self.structural_matcher,
                labels_to_classification_frequencies = self.labels_to_classification_frequencies,
                serialized_phraselets = self.serialized_phraselets,
                minimum_occurrences = minimum_occurrences,
                cv_threshold = cv_threshold,
                mlp_activation = mlp_activation,
                mlp_solver = mlp_solver,
                mlp_learning_rate = mlp_learning_rate,
                mlp_learning_rate_init = mlp_learning_rate_init,
                mlp_max_iter = mlp_max_iter,
                mlp_shuffle = mlp_shuffle,
                mlp_random_state = mlp_random_state,
                hidden_layer_sizes = hidden_layer_sizes,
                utils = self._utils
        )

class SupervisedTopicModelTrainer:
    """ Worker object used to train and generate models. This class is not threadsafe."""

    def __init__(self, *, training_basis, semantic_analyzer, structural_matcher,
            labels_to_classification_frequencies, serialized_phraselets, minimum_occurrences,
            cv_threshold, mlp_activation, mlp_solver, mlp_learning_rate, mlp_learning_rate_init,
            mlp_max_iter, mlp_shuffle, mlp_random_state, hidden_layer_sizes, utils):

        self._utils = utils
        self._semantic_analyzer = semantic_analyzer
        self._structural_matcher = structural_matcher
        self._training_basis = training_basis
        self._minimum_occurrences = minimum_occurrences
        self._cv_threshold = cv_threshold
        self._labels_to_classification_frequencies, self._serialized_phraselets = self._filter(
                labels_to_classification_frequencies, serialized_phraselets)

        if len(self._serialized_phraselets) == 0:
            raise NoPhraseletsAfterFilteringError(''.join(('minimum_occurrences: ',
                    str(minimum_occurrences), '; cv_threshold: ', str(cv_threshold))))

        phraselet_labels_to_search_phrases=self._structural_matcher.deserialize_phraselets(
                self._serialized_phraselets)
        self._sorted_label_dict = {}
        for index, label in enumerate(sorted(self._labels_to_classification_frequencies.keys())):
            self._sorted_label_dict[label] = index
        self._input_matrix = dok_matrix((len(self._training_basis.training_documents),
                len(self._sorted_label_dict)))
        self._output_matrix = dok_matrix((len(self._training_basis.training_documents),
                len(self._training_basis.classifications)))

        if self._training_basis.verbose:
            print('Matching documents against filtered phraselets')
        for index, document_label in enumerate(sorted(self._training_basis.
                training_documents.keys())):
            self._utils.record_matches(
                    structural_matcher = self._structural_matcher,
                    phraselet_labels_to_search_phrases =
                            phraselet_labels_to_search_phrases,
                    sorted_label_dict = self._sorted_label_dict,
                    doc=self._training_basis.training_documents[document_label],
                    doc_label=document_label,
                    matrix=self._input_matrix,
                    row_index=index,
                    verbose=True)
            self._record_classifications_for_training(document_label, index)
        self._hidden_layer_sizes = hidden_layer_sizes
        if self._hidden_layer_sizes == None:
            start = len(self._sorted_label_dict)
            step = (len(self._training_basis.classifications) - len(self._sorted_label_dict)) / 3
            self._hidden_layer_sizes = [start, int(start+step), int(start+(2*step))]
        if self._training_basis.verbose:
            print('Hidden layer sizes:', self._hidden_layer_sizes)
        self._mlp = MLPClassifier(
                activation=mlp_activation,
                solver=mlp_solver,
                hidden_layer_sizes= self._hidden_layer_sizes,
                learning_rate = mlp_learning_rate,
                learning_rate_init = mlp_learning_rate_init,
                max_iter = mlp_max_iter,
                shuffle = mlp_shuffle,
                verbose = self._training_basis.verbose,
                random_state = mlp_random_state)
        self._mlp.fit(self._input_matrix, self._output_matrix)
        if self._training_basis.verbose and self._mlp.n_iter_ < mlp_max_iter:
            print('MLP neural network converged after', self._mlp.n_iter_, 'iterations.')

    def _filter(self, labels_to_classification_frequencies, serialized_phraselets):
        """ Filters the phraselets in memory based on minimum_occurrences and cv_threshold. """

        def increment_totals(classification_frequencies):
            for classification, frequency in classification_frequencies.items():
                if classification in totals:
                    totals[classification] += frequency
                else:
                    totals[classification] = frequency

        totals = {}
        accepted = 0
        under_minimum_occurrences = 0
        under_minimum_cv = 0
        for classification_frequencies in labels_to_classification_frequencies.values():
            increment_totals(classification_frequencies)
        new_labels_to_classification_frequencies = {}
        new_serialized_phraselets = set()
        for label, classification_frequencies in labels_to_classification_frequencies.items():
            at_least_minimum = False
            working_classification_frequencies = classification_frequencies.copy()
            for classification in working_classification_frequencies:
                if working_classification_frequencies[classification] >= self._minimum_occurrences:
                    at_least_minimum = True
                working_classification_frequencies[classification] *= totals[classification]
            if not at_least_minimum:
                under_minimum_occurrences += 1
                continue
            frequency_list = list(working_classification_frequencies.values())
            # We only want to take explicit classification labels into account, i.e. ignore the
            # classification ontology.
            number_of_classification_labels = \
                len(set(
                self._training_basis.training_documents_labels_to_classifications_dict.values()))
            frequency_list.extend([0] * number_of_classification_labels)
            frequency_list = frequency_list[:number_of_classification_labels]
            if statistics.pstdev(frequency_list) / statistics.mean(frequency_list) >= \
                    self._cv_threshold:
                accepted += 1
                new_labels_to_classification_frequencies[label] = classification_frequencies
            else:
                under_minimum_cv += 1
        if self._training_basis.verbose:
            print('Filtered: accepted', accepted, '; removed minimum occurrences',
                under_minimum_occurrences, '; removed cv threshold',
                under_minimum_cv)
        for serialized_phraselet in serialized_phraselets:
            if serialized_phraselet.label in \
                    new_labels_to_classification_frequencies.keys():
                new_serialized_phraselets.add(serialized_phraselet)
        return new_labels_to_classification_frequencies, new_serialized_phraselets

    def _record_classifications_for_training(self, document_label, index):
        classification = \
                self._training_basis.training_documents_labels_to_classifications_dict[
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
        self._structural_matcher.output_document_matching_message_to_console = False
        self._mlp.verbose=False # we no longer require output once we are using the model
                                # to classify new documents
        model = SupervisedTopicClassifierModel(
                semantic_analyzer_model = self._semantic_analyzer.model,
                structural_matcher_ontology = self._structural_matcher.ontology,
                serialized_phraselets = self._serialized_phraselets,
                mlp = self._mlp,
                sorted_label_dict = self._sorted_label_dict,
                classifications = self._training_basis.classifications,
                overlap_memory_size = self._utils.overlap_memory_size,
                oneshot = self._utils.oneshot)
        return SupervisedTopicClassifier(self._semantic_analyzer, self._structural_matcher,
                model, self._training_basis.verbose)

class SupervisedTopicClassifierModel:
    """ A serializable classifier model.

        Parameters:

        semantic_analyzer_model -- a string specifying the spaCy model with which this instance
            was generated and with which it must be used.
        structural_matcher_ontology -- the ontology used for matching documents against this model
            (not the classification ontology!)
        serialized_phraselets -- the phraselets used for structural matching
        mlp -- the neural network
        sorted_label_dict -- a dictionary from search phrase (phraselet) labels to their own
            alphabetic sorting indexes.
        classifications -- an ordered list of classification labels corresponding to the
            neural network outputs
        overlap_memory_size -- how many non-word phraselet matches to the left should be
            checked for words in common with a current match.
        oneshot -- whether the same word or relationship matched multiple times should be
            counted once only (value 'True') or multiple times (value 'False')
    """

    def __init__(self, semantic_analyzer_model, structural_matcher_ontology,
            serialized_phraselets, mlp, sorted_label_dict, classifications, overlap_memory_size,
            oneshot):
        self.semantic_analyzer_model = semantic_analyzer_model
        self.structural_matcher_ontology = structural_matcher_ontology
        self.serialized_phraselets = serialized_phraselets
        self.mlp = mlp
        self.sorted_label_dict = sorted_label_dict
        self.classifications = classifications
        self.overlap_memory_size = overlap_memory_size
        self.oneshot = oneshot

class SupervisedTopicClassifier:
    """ Classifies new documents based on a pre-trained model."""

    def __init__(self, semantic_analyzer, structural_matcher, model, verbose):
        self._semantic_analyzer = semantic_analyzer
        self._structural_matcher = structural_matcher
        self._model = model
        self._verbose = verbose
        self._utils = SupervisedTopicTrainingUtils(model.overlap_memory_size, model.oneshot)
        if self._semantic_analyzer.model != model.semantic_analyzer_model:
            raise WrongModelDeserializationError(model.semantic_analyzer_model)
        self._structural_matcher.ontology = model.structural_matcher_ontology
        self._phraselet_labels_to_search_phrases = self._structural_matcher.deserialize_phraselets(
                model.serialized_phraselets)

    def parse_and_classify(self, text):
        """ Returns a list containing zero, one or many document classifications. Where more
            than one classifications are returned, the labels are ordered by decreasing
            probability.

            Parameter:

            text -- the text to parse and classify.
        """
        return self.classify(self._semantic_analyzer.parse(text))

    def classify(self, doc):
        """ Returns a list containing zero, one or many document classifications. Where more
            than one classifications are returned, the labels are ordered by decreasing
            probability.

            Parameter:

            doc -- the pre-parsed document to classify.
        """

        if self._model == None:
            raise RuntimeError('No model defined')
        new_document_matrix = dok_matrix((1, len(self._model.sorted_label_dict)))
        if not self._utils.record_matches(structural_matcher=self._structural_matcher,
                phraselet_labels_to_search_phrases=self._phraselet_labels_to_search_phrases,
                sorted_label_dict=self._model.sorted_label_dict,
                doc=doc,
                doc_label=None,
                matrix=new_document_matrix,
                row_index=0,
                verbose=self._verbose):
            return []
        else:
            classification_indexes = self._model.mlp.predict(new_document_matrix).nonzero()[1]
            if len(classification_indexes) > 1:
                probabilities = self._model.mlp.predict_proba(new_document_matrix)
                classification_indexes = sorted(classification_indexes, key=lambda index:
                        1-probabilities[0,index])
            return list(map(lambda index:self._model.classifications[index],
                    classification_indexes))

    def serialize_model(self):
        return jsonpickle.encode(self._model)
