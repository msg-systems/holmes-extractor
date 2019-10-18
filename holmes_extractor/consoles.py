from .errors import *

class HolmesConsoles:
    """Manages the consoles."""

    def __init__(self, holmes):
        self._holmes = holmes
        self._semantic_analyzer = holmes.semantic_analyzer
        self._structural_matcher = holmes.structural_matcher

    def _match_description(self, match_dict):
        """Returns a user-readable representation of a match dictionary."""
        match_description_to_return = ''
        if match_dict['negated']:
            match_description_to_return = '; negated'
        if match_dict['uncertain']:
            match_description_to_return = ''.join((match_description_to_return, '; uncertain'))
        if match_dict['involves_coreference']:
            match_description_to_return = ''.join((match_description_to_return,
                    '; involves coreference'))
        overall_similarity_measure = float(match_dict['overall_similarity_measure'])
        if overall_similarity_measure < 1.0:
            match_description_to_return = ''.join((match_description_to_return,
                    '; overall similarity measure=', str(overall_similarity_measure)))
        return match_description_to_return

    def _string_representation_of_word_match(self, word_match):
        """Returns a user-readable representation of a word match."""
        if word_match['document_word'] != word_match['extracted_word']:
            extracted_word = ''.join(("(refers to '", word_match['extracted_word'], "')"))
        else:
            extracted_word = ''
        string = ''.join(("'", word_match['document_phrase'], "'", extracted_word, "->'",
                word_match['search_phrase_word'], "' (", word_match['match_type']))
        if float(word_match['similarity_measure']) < 1.0:
            string = ''.join((string, ': ', word_match['similarity_measure']))
        string = ''.join((string, ")"))
        return string

    def _common(self):
        """Contains functionality common to both consoles."""
        print("Holmes version 2.1 written by richard.hudson@msg.group")
        print("Language is", self._semantic_analyzer.language_name)
        print("Model is", self._semantic_analyzer.model)
        if self._structural_matcher.ontology == None:
            print("No ontology is being used")
        else:
            print("Ontology is", self._structural_matcher.ontology.path)
            if self._structural_matcher.ontology.symmetric_matching:
                print("Symmetric matching is ON")
            else:
                print("Symmetric matching is OFF")
        if self._structural_matcher.perform_coreference_resolution:
            print("Coreference resolution is ON")
        else:
            print("Coreference resolution is OFF")
        print("Overall similarity threshold is", str(
                self._structural_matcher.overall_similarity_threshold))
        if self._structural_matcher.overall_similarity_threshold < 1.0:
            if self._structural_matcher.embedding_based_matching_on_root_words:
                print("Embedding-based matching on root words is ON")
            else:
                print("Embedding-based matching on root words is OFF")


    def start_chatbot_mode(self):
        """Starts a chatbot mode console enabling the matching of pre-registered search phrases
            to documents (chatbot entries) entered ad-hoc by the user.
        """
        self._common()
        print('Chatbot mode')
        print()
        if len(self._holmes.threadsafe_container._search_phrases) == 0:
            raise RuntimeError('No search_phrases registered.')
        # Display search phrases
        for search_phrase in self._holmes.threadsafe_container._search_phrases:
            print(''.join(("Search phrase '", search_phrase.doc.text, "'")))
            # only has an effect when debug==True
            self._semantic_analyzer.debug_structures(search_phrase.doc)
            if self._structural_matcher.ontology != None:
                for token in search_phrase.matchable_tokens:
                    lemma = token._.holmes.lemma
                    matching_terms = self._structural_matcher.ontology.get_words_matching(
                            lemma)
                    if len(matching_terms) > 0:
                        print(lemma, 'also matches', matching_terms)
            print()

        print()
        print('Ready for input')
        while True:
            print()
            search_sentence = input()
            print()
            if search_sentence in ('exit', 'exit()', 'bye'):
                break
            match_dicts = self._holmes.match_search_phrases_against(entry=search_sentence)
            for match_dict in match_dicts:
                print()
                print(''.join(("Matched search phrase '",
                        match_dict['search_phrase'], "'", self._match_description(match_dict),
                        ":")))
                word_matches_string = '; '.join(
                    map(self._string_representation_of_word_match, match_dict['word_matches']))
                print(word_matches_string)

    def start_structural_search_mode(self):
        """Starts a structural search mode console enabling the matching of pre-registered documents
            to search phrases entered ad-hoc by the user.
        """
        self._common()
        print('Structural search mode')
        print()
        if len(self._holmes.document_labels()) == 0:
            raise RuntimeError('No documents registered.')
        document_labels = '; '.join(self._holmes.document_labels())
        print(': '.join(('Documents', document_labels)))
        print()
        while True:
            print('Ready for phrases')
            print()
            search_phrase = input()
            # removing question marks seems to lead to better results
            search_phrase = search_phrase.strip(' ').strip('?')
            if search_phrase == '':
                continue
            if search_phrase in ('exit', 'exit()', 'bye'):
                break
            print()
            match_dicts=[]
            try:
                match_dicts = self._holmes.match_documents_against(search_phrase_text=search_phrase)
                if len(match_dicts) == 0:
                    print('No structural matching results were returned.')
                else:
                    print('Structural matching results:')
            except SearchPhraseContainsNegationError:
                print('Structural matching was not attempted because the search phrase contained negation (not, never).')
                print()
            except SearchPhraseContainsConjunctionError:
                print('Structural matching was not attempted because the search phrase contained conjunction (and, or).')
                print()
            except SearchPhraseContainsCoreferringPronounError:
                print('Structural matching was not attempted because the search phrase contained a pronoun that referred back to a noun.')
                print()
            except SearchPhraseWithoutMatchableWordsError:
                print('Structural matching was not attempted because the search phrase did not contain any words that could be matched.')
                print()
            except SearchPhraseContainsMultipleClausesError:
                print('Structural matching was not attempted because the search phrase contained multiple clauses.')
                print()
            print()
            for match_dict in match_dicts:
                print()
                print(''.join(("Matched document '", match_dict['document'],
                        "' at index ", str(match_dict['index_within_document']),
                        self._match_description(match_dict), ":")))
                print(''.join(('"', match_dict['sentences_within_document'], '"')))
                word_matches_string = '; '.join(map(self._string_representation_of_word_match,
                        match_dict['word_matches']))
                print(word_matches_string)

    def start_topic_matching_search_mode(self, only_one_result_per_document,
            maximum_number_of_single_word_matches_for_relation_matching,
            maximum_number_of_single_word_matches_for_embedding_reverse_matching):
        """Starts a topic matching search mode console enabling the matching of pre-registered
            documents to search texts entered ad-hoc by the user.

            Parameters:

            only_one_result_per_document -- if 'True', prevents multiple topic match
            results from being returned for the same document.
        """
        self._common()
        print('Topic matching search mode')
        print('maximum_number_of_single_word_matches_for_relation_matching = ',
                maximum_number_of_single_word_matches_for_relation_matching)
        print('maximum_number_of_single_word_matches_for_embedding_reverse_matching = ',
                maximum_number_of_single_word_matches_for_embedding_reverse_matching)
        print()
        if len(self._holmes.document_labels()) == 0:
            raise RuntimeError('No documents registered.')
        document_labels = '; '.join(self._holmes.document_labels())
        print(': '.join(('Documents', document_labels)))
        print()
        while True:
            print('Ready for search texts')
            print()
            search_text = input()
            # removing question marks seems to lead to better results
            search_text = search_text.strip(' ').strip('?')
            if search_text == '':
                continue
            if search_text in ('exit', 'exit()', 'bye'):
                break
            print()
            print('Performing topic matching ...')
            topic_matches = {}
            try:
                print()
                topic_match_dicts = \
                        self._holmes.topic_match_documents_returning_dictionaries_against(
                        search_text,
                        number_of_results = 5,
                        only_one_result_per_document=only_one_result_per_document,
                        maximum_number_of_single_word_matches_for_relation_matching =
                        maximum_number_of_single_word_matches_for_relation_matching,
                        maximum_number_of_single_word_matches_for_embedding_reverse_matching =
                        maximum_number_of_single_word_matches_for_embedding_reverse_matching)
            except NoSearchPhraseError:
                pass
            if topic_match_dicts == None or len(topic_match_dicts) == 0:
                print('No topic match results were returned.')
                print()
                continue
            elif only_one_result_per_document:
                print('Topic matching results (maximum one per document):')
            else:
                print('Topic matching results:')
            print()
            for index, topic_match_dict in enumerate(topic_match_dicts):
                output = ''.join((
                    topic_match_dict['rank'],
                    '. Document ',
                    topic_match_dict['document_label'],
                    '; sentences at character indexes ',
                    str(topic_match_dict['sentences_character_start_index_in_document']),
                    '-',
                    str(topic_match_dict['sentences_character_end_index_in_document']),
                    '; score ',
                    str(topic_match_dict['score']),
                    ':'
                ))
                print (output)
                print()
                print (topic_match_dict['text'])
                print()
            print()
