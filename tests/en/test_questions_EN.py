import unittest
import holmes_extractor as holmes
from holmes_extractor.topic_matching import TopicMatcher
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory, 'test_ontology.owl')),
                           symmetric_matching=True)
manager = holmes.Manager(model='en_core_web_trf', ontology=ontology,
                                      number_of_workers=1)

class EnglishInitialQuestionsTest(unittest.TestCase):

    def _check_equals(self, text_to_match, document_text, highest_score, answer_start, answer_end,
        word_embedding_match_threshold=0.42, initial_question_word_embedding_match_threshold=0.42,
        use_frequency_factor=True, initial_question_word_answer_score=40, relation_matching_frequency_threshold=0.0, embedding_matching_frequency_threshold=0.0):
        manager.remove_all_documents()
        manager.parse_and_register_document(document_text)
        topic_matches = manager.topic_match_documents_against(text_to_match,
                                                              word_embedding_match_threshold=
                                                              word_embedding_match_threshold,
                                                              initial_question_word_embedding_match_threshold=initial_question_word_embedding_match_threshold,
                                                              initial_question_word_answer_score=initial_question_word_answer_score,
                                                              relation_score=20,
                                                              reverse_only_relation_score=15, single_word_score=10, single_word_any_tag_score=5,
                                                              different_match_cutoff_score=10,
                                                              relation_matching_frequency_threshold=relation_matching_frequency_threshold,
                                                              embedding_matching_frequency_threshold=embedding_matching_frequency_threshold,
                                                              use_frequency_factor=use_frequency_factor)
        self.assertEqual(int(topic_matches[0]['score']), highest_score)
        if answer_start is not None:
            self.assertEqual(topic_matches[0]['answers'][0][0], answer_start)
            self.assertEqual(topic_matches[0]['answers'][0][1], answer_end)
        else:
            self.assertEqual(len(topic_matches[0]['answers']), 0)

    def test_basic_matching(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("Richard and Peter sang a duet.", 'q')
        manager.parse_and_register_document("A book sings an elogy", 'n')
        topic_matches = manager.topic_match_documents_against("Who sings?")
        self.assertEqual([{'document_label': 'q', 'text': 'Richard and Peter sang a duet.', 'text_to_match': 'Who sings?', 'rank': '1', 'index_within_document': 3, 'subword_index': None, 'start_index': 0, 'end_index': 3, 'sentences_start_index': 0, 'sentences_end_index': 6, 'sentences_character_start_index': 0, 'sentences_character_end_index': 30, 'score': 620.0, 'word_infos': [[0, 7, 'relation', False, 'Matches the question word WHO.'], [12, 17, 'relation', False, 'Matches the question word WHO.'], [18, 22, 'relation', True, 'Matches SING directly.']], 'answers': [[0, 7], [12, 17]]}, {'document_label': 'n', 'text': 'A book sings an elogy', 'text_to_match': 'Who sings?', 'rank': '2', 'index_within_document': 2, 'subword_index': None, 'start_index': 2, 'end_index': 2, 'sentences_start_index': 0, 'sentences_end_index': 4, 'sentences_character_start_index': 0, 'sentences_character_end_index': 21, 'score': 20.0, 'word_infos': [[7, 12, 'single', True, 'Matches SING directly.']], 'answers': []}], topic_matches)

    def test_ignore_questions(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("Richard and Peter sang a duet.", 'q')
        manager.parse_and_register_document("A book sings an elogy", 'n')
        topic_matches = manager.topic_match_documents_against("Who sings?", initial_question_word_behaviour='ignore')
        self.assertEqual([{'document_label': 'q', 'text': 'Richard and Peter sang a duet.', 'text_to_match': 'Who sings?', 'rank': '1=', 'index_within_document': 3, 'subword_index': None, 'start_index': 3, 'end_index': 3, 'sentences_start_index': 0, 'sentences_end_index': 6, 'sentences_character_start_index': 0, 'sentences_character_end_index': 30, 'score': 20.0, 'word_infos': [[18, 22, 'single', True, 'Matches SING directly.']], 'answers': []}, {'document_label': 'n', 'text': 'A book sings an elogy', 'text_to_match': 'Who sings?', 'rank': '1=', 'index_within_document': 2, 'subword_index': None, 'start_index': 2, 'end_index': 2, 'sentences_start_index': 0, 'sentences_end_index': 4, 'sentences_character_start_index': 0, 'sentences_character_end_index': 21, 'score': 20.0, 'word_infos': [[7, 12, 'single', True, 'Matches SING directly.']], 'answers': []}], topic_matches)

    def test_exclusive_questions(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("Richard and Peter sang a duet.", 'q')
        manager.parse_and_register_document("A book sings an elogy", 'n')
        topic_matches = manager.topic_match_documents_against("Who sings?", initial_question_word_behaviour='exclusive')
        self.assertEqual(len(topic_matches), 1)
        self.assertEqual(topic_matches[0]['document_label'], 'q')

    def test_governed_interrogative_pronoun_matching_common_noun(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("The man sang a duet.", 'q')
        topic_matches = manager.topic_match_documents_against("Which person sings?")
        self.assertEqual([{'document_label': 'q', 'text': 'The man sang a duet.', 'text_to_match': 'Which person sings?', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 1, 'end_index': 2, 'sentences_start_index': 0, 'sentences_end_index': 5, 'sentences_character_start_index': 0, 'sentences_character_end_index': 20, 'score': 288.3671696, 'word_infos': [[4, 7, 'relation', False, 'Has a word embedding that is 55% similar to PERSON.'], [8, 12, 'relation', True, 'Matches SING directly.']], 'answers': [[0, 7]]}], topic_matches)
        topic_matches = manager.topic_match_documents_against("A person sings", word_embedding_match_threshold=0.42)
        self.assertEqual([{'document_label': 'q', 'text': 'The man sang a duet.', 'text_to_match': 'A person sings', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 1, 'end_index': 2, 'sentences_start_index': 0, 'sentences_end_index': 5, 'sentences_character_start_index': 0, 'sentences_character_end_index': 20, 'score': 154.1835848, 'word_infos': [[4, 7, 'relation', False, 'Has a word embedding that is 55% similar to PERSON.'], [8, 12, 'relation', True, 'Matches SING directly.']], 'answers': []}], topic_matches)

    def test_governed_interrogative_pronoun_matching_proper_noun(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("Richard Hudson sang a duet.", 'q')
        topic_matches = manager.topic_match_documents_against("Which person sings?")
        self.assertEqual([{'document_label': 'q', 'text': 'Richard Hudson sang a duet.', 'text_to_match': 'Which person sings?', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 1, 'end_index': 2, 'sentences_start_index': 0, 'sentences_end_index': 5, 'sentences_character_start_index': 0, 'sentences_character_end_index': 27, 'score': 620.0, 'word_infos': [[8, 14, 'relation', False, 'Has an entity label that is 100% similar to the word embedding corresponding to PERSON.'], [15, 19, 'relation', True, 'Matches SING directly.']], 'answers': [[0, 14]]}], topic_matches)
        topic_matches = manager.topic_match_documents_against("A person sings")
        self.assertEqual([{'document_label': 'q', 'text': 'Richard Hudson sang a duet.', 'text_to_match': 'A person sings', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 1, 'end_index': 2, 'sentences_start_index': 0, 'sentences_end_index': 5, 'sentences_character_start_index': 0, 'sentences_character_end_index': 27, 'score': 320.0, 'word_infos': [[8, 14, 'relation', False, 'Has an entity label that is 100% similar to the word embedding corresponding to PERSON.'], [15, 19, 'relation', True, 'Matches SING directly.']], 'answers': []}], topic_matches)

    def test_basic_matching_with_coreference(self):
        self._check_equals("Who came home?", 'I spoke to Richard. He came home', 98, 11, 18)

    def test_basic_matching_with_coreference_and_coordination(self):
        self._check_equals("Who came home?", 'I spoke to Richard Hudson and Peter Hudson. They came home', 98, 11, 25)

    def test_governed_interrogative_pronoun_matching_direct(self):
        self._check_equals('Which politician lied?', 'The politician lied', 54, 0, 14)

    def test_governed_interrogative_pronoun_matching_direct_control(self):
        self._check_equals('A politician lies', 'The politician lied', 34, None, None)

    def test_governed_interrogative_pronoun_matching_derivation(self):
        self._check_equals('Which performance by the boys was important?', 'The boys performed', 59, 0, 18)

    def test_governed_interrogative_pronoun_matching_derivation_control(self):
        self._check_equals('A performance by the boys is important', 'The boys performed', 39, None, None)

    def test_governed_interrogative_pronoun_matching_ontology(self):
        self._check_equals('Which animal woke up?', 'The cat woke up', 45, 0, 7)

    def test_governed_interrogative_pronoun_matching_ontology_control(self):
        self._check_equals('An animal woke up', 'The cat woke up', 29, None, None)

    def test_governed_interrogative_pronoun_reverse_dependency(self):
        self._check_equals('Which child did its parents adopt?', 'The adopted child', 54, 0, 17)

    def test_governed_interrogative_pronoun_reverse_dependency_control(self):
        self._check_equals('A child is adopted by its parents', 'The adopted child', 34, None, None)

    def test_governed_interrogative_pronoun_with_coreference(self):
        self._check_equals("Which person came home?", 'I spoke to Richard. He came home', 98, 11, 18)

    def test_separate_embedding_threshold_for_question_words_normal_threshold_1(self):
         self._check_equals("Which man came home?", 'The person came home', 52, 0, 10,
            word_embedding_match_threshold=1.0, initial_question_word_answer_score=20)

    def test_separate_embedding_threshold_for_question_words_normal_threshold_1_control(self):
         self._check_equals("A man comes home", 'The person came home', 29, None, None,
            word_embedding_match_threshold=1.0, initial_question_word_answer_score=20)

    def test_separate_embedding_threshold_for_question_words_normal_threshold_below_1(self):
         self._check_equals("Which man came home?", 'The person came home', 52, 0, 10,
            word_embedding_match_threshold=0.9, initial_question_word_answer_score=20)

    def test_separate_embedding_threshold_for_question_words_normal_threshold_below_1_control(self):
         self._check_equals("A man comes home", 'The person came home', 29, None, None,
            word_embedding_match_threshold=0.9, initial_question_word_answer_score=20)

    def test_single_word_match_does_not_recognize_dependent_question_word(self):
         self._check_equals("Which man?", 'The man', 10, None, None)

    def test_single_word_match_with_dependent_question_word_control(self):
         self._check_equals("The man?", 'The man', 10, None, None)

    def test_check_who_positive_case(self):
        self._check_equals('Who looked into the sun?', 'the man looked into the sun', 127, 0, 7)

    def test_check_who_wrong_syntax(self):
        self._check_equals('Who looked into the sun?', 'the sun looked into the man', 19, None, None)

    def test_check_who_wrong_noun(self):
        self._check_equals('Who looked into the sun?', 'the dog looked into the sun', 70, None, None)

    def test_no_relation_frequency_threshold_for_direct_question_words(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("Richard came. Come. Come.", 'q')
        topic_matches = manager.topic_match_documents_against("What came?", relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0)
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'Richard came. Come. Come.', 'text_to_match': 'What came?', 'rank': '1', 'index_within_document': 1, 'subword_index': None, 'start_index': 0, 'end_index': 5, 'sentences_start_index': 0, 'sentences_end_index': 6, 'sentences_character_start_index': 0, 'sentences_character_end_index': 25, 'score': 228.8235527856964, 'word_infos': [[0, 7, 'relation', False, 'Matches the question word WHAT.'], [8, 12, 'relation', True, 'Matches COME directly.'], [14, 18, 'single', False, 'Matches COME directly.'], [20, 24, 'single', False, 'Matches COME directly.']], 'answers': [[0, 7]]}])

    def test_no_relation_frequency_threshold_for_direct_question_words_control(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("Richard came. Come. Come.", 'd')
        topic_matches = manager.topic_match_documents_against("Richard came?", relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0)
        self.assertEqual(topic_matches, [{'document_label': 'd', 'text': 'Richard came. Come. Come.', 'text_to_match': 'Richard came?', 'rank': '1', 'index_within_document': 1, 'subword_index': None, 'start_index': 0, 'end_index': 5, 'sentences_start_index': 0, 'sentences_end_index': 6, 'sentences_character_start_index': 0, 'sentences_character_end_index': 25, 'score': 167.43581219046695, 'word_infos': [[0, 7, 'relation', False, 'Matches RICHARD directly.'], [8, 12, 'relation', True, 'Matches COME directly.'], [14, 18, 'single', False, 'Matches COME directly.'], [20, 24, 'single', False, 'Matches COME directly.']], 'answers': []}])

    def test_no_relation_frequency_threshold_for_governed_question_words(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("The dog barked. The dog barked. The dog barked.", 'q')
        topic_matches = manager.topic_match_documents_against("Which dog barked?",
        relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0)
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'The dog barked. The dog barked. The dog barked.', 'text_to_match': 'Which dog barked?', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 1, 'end_index': 10, 'sentences_start_index': 0, 'sentences_end_index': 11, 'sentences_character_start_index': 0, 'sentences_character_end_index': 47, 'score': 107.3165784983407, 'word_infos': [[4, 7, 'relation', False, 'Matches DOG directly.'], [8, 14, 'relation', True, 'Matches BARK directly.'], [20, 23, 'relation', False, 'Matches DOG directly.'], [24, 30, 'relation', False, 'Matches BARK directly.'], [36, 39, 'relation', False, 'Matches DOG directly.'], [40, 46, 'relation', False, 'Matches BARK directly.']], 'answers': [[0, 7], [16, 23], [32, 39]]}])

    def test_no_relation_frequency_threshold_for_governed_question_words_control(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("The dog barked. The dog barked. The dog barked.", 'q')
        topic_matches = manager.topic_match_documents_against("The dog barked?",
        relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0)
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'The dog barked. The dog barked. The dog barked.', 'text_to_match': 'The dog barked?', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 1, 'end_index': 10, 'sentences_start_index': 0, 'sentences_end_index': 11, 'sentences_character_start_index': 0, 'sentences_character_end_index': 47, 'score': 25.58887041904562, 'word_infos': [[4, 7, 'single', False, 'Matches DOG directly.'], [8, 14, 'single', True, 'Matches BARK directly.'], [20, 23, 'single', False, 'Matches DOG directly.'], [24, 30, 'single', False, 'Matches BARK directly.'], [36, 39, 'single', False, 'Matches DOG directly.'], [40, 46, 'single', False, 'Matches BARK directly.']], 'answers': []}])

    def test_no_reverse_relation_frequency_threshold_for_governed_question_words(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("in a house. in a house. in a house.", 'q')
        topic_matches = manager.topic_match_documents_against("In which house?",
        relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0)
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'in a house. in a house. in a house.', 'text_to_match': 'In which house?', 'rank': '1', 'index_within_document': 4, 'subword_index': None, 'start_index': 0, 'end_index': 10, 'sentences_start_index': 0, 'sentences_end_index': 11, 'sentences_character_start_index': 0, 'sentences_character_end_index': 35, 'score': 107.07053166738835, 'word_infos': [[0, 2, 'relation', False, 'Matches IN directly.'], [5, 10, 'relation', False, 'Matches HOUSE directly.'], [12, 14, 'relation', True, 'Matches IN directly.'], [17, 22, 'relation', False, 'Matches HOUSE directly.'], [24, 26, 'relation', False, 'Matches IN directly.'], [29, 34, 'relation', False, 'Matches HOUSE directly.']], 'answers': [[3, 10], [15, 22], [27, 34]]}])

    def test_no_reverse_relation_frequency_threshold_for_governed_question_words_control(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("in a house. in a house. in a house.", 'q')
        topic_matches = manager.topic_match_documents_against("In a house",
        relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0)
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'in a house. in a house. in a house.', 'text_to_match': 'In a house', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 0, 'end_index': 10, 'sentences_start_index': 0, 'sentences_end_index': 11, 'sentences_character_start_index': 0, 'sentences_character_end_index': 35, 'score': 25.638079785236094, 'word_infos': [[0, 2, 'single', False, 'Matches IN directly.'], [5, 10, 'single', True, 'Matches HOUSE directly.'], [12, 14, 'single', False, 'Matches IN directly.'], [17, 22, 'single', False, 'Matches HOUSE directly.'], [24, 26, 'single', False, 'Matches IN directly.'], [29, 34, 'single', False, 'Matches HOUSE directly.']], 'answers': []}])

    def test_no_embedding_frequency_threshold_for_governed_question_words_on_child(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("The dog barked. The dog barked. The dog barked.", 'q')
        topic_matches = manager.topic_match_documents_against("Which cat barked?",
        relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0,
        initial_question_word_embedding_match_threshold=0.2, word_embedding_match_threshold=0.2)
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'The dog barked. The dog barked. The dog barked.', 'text_to_match': 'Which cat barked?', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 1, 'end_index': 10, 'sentences_start_index': 0, 'sentences_end_index': 11, 'sentences_character_start_index': 0, 'sentences_character_end_index': 47, 'score': 126.34484701824243, 'word_infos': [[4, 7, 'relation', False, 'Has a word embedding that is 80% similar to CAT.'], [8, 14, 'relation', True, 'Matches BARK directly.'], [20, 23, 'relation', False, 'Has a word embedding that is 80% similar to CAT.'], [24, 30, 'relation', False, 'Matches BARK directly.'], [36, 39, 'relation', False, 'Has a word embedding that is 80% similar to CAT.'], [40, 46, 'relation', False, 'Matches BARK directly.']], 'answers': [[0, 7], [16, 23], [32, 39]]}])

    def test_no_embedding_frequency_threshold_for_governed_question_words_on_child_control(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("The dog barked. The dog barked. The dog barked.", 'q')
        topic_matches = manager.topic_match_documents_against("The cat barked?",
        relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0,
        initial_question_word_embedding_match_threshold=0.2, word_embedding_match_threshold=0.2)
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'The dog barked.', 'text_to_match': 'The cat barked?', 'rank': '1=', 'index_within_document': 2, 'subword_index': None, 'start_index': 2, 'end_index': 2, 'sentences_start_index': 0, 'sentences_end_index': 3, 'sentences_character_start_index': 0, 'sentences_character_end_index': 15, 'score': 7.381404928570852, 'word_infos': [[8, 14, 'single', True, 'Matches BARK directly.']], 'answers': []}, {'document_label': 'q', 'text': 'The dog barked.', 'text_to_match': 'The cat barked?', 'rank': '1=', 'index_within_document': 6, 'subword_index': None, 'start_index': 6, 'end_index': 6, 'sentences_start_index': 4, 'sentences_end_index': 7, 'sentences_character_start_index': 16, 'sentences_character_end_index': 31, 'score': 7.381404928570852, 'word_infos': [[8, 14, 'single', True, 'Matches BARK directly.']], 'answers': []}, {'document_label': 'q', 'text': 'The dog barked.', 'text_to_match': 'The cat barked?', 'rank': '1=', 'index_within_document': 10, 'subword_index': None, 'start_index': 10, 'end_index': 10, 'sentences_start_index': 8, 'sentences_end_index': 11, 'sentences_character_start_index': 32, 'sentences_character_end_index': 47, 'score': 7.381404928570852, 'word_infos': [[8, 14, 'single', True, 'Matches BARK directly.']], 'answers': []}])

    def test_no_embedding_frequency_threshold_for_governed_question_words_on_parent(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("A big dog. A big dog. A big dog.", 'q')
        topic_matches = manager.topic_match_documents_against("Which big cat?",
        relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0,
        initial_question_word_embedding_match_threshold=0.2, word_embedding_match_threshold=0.2)
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'A big dog. A big dog. A big dog.', 'text_to_match': 'Which big cat?', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 1, 'end_index': 10, 'sentences_start_index': 0, 'sentences_end_index': 11, 'sentences_character_start_index': 0, 'sentences_character_end_index': 32, 'score': 126.24642828586148, 'word_infos': [[2, 5, 'relation', False, 'Matches BIG directly.'], [6, 9, 'relation', True, 'Has a word embedding that is 80% similar to CAT.'], [13, 16, 'relation', False, 'Matches BIG directly.'], [17, 20, 'relation', False, 'Has a word embedding that is 80% similar to CAT.'], [24, 27, 'relation', False, 'Matches BIG directly.'], [28, 31, 'relation', False, 'Has a word embedding that is 80% similar to CAT.']], 'answers': [[0, 10], [11, 21], [22, 32]]}])

    def test_no_embedding_frequency_threshold_for_governed_question_words_on_parent_control(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("A big dog. A big dog. A big dog.", 'q')
        topic_matches = manager.topic_match_documents_against("The big cat?",
        relation_matching_frequency_threshold=1.0, embedding_matching_frequency_threshold=1.0,
        initial_question_word_embedding_match_threshold=0.2, word_embedding_match_threshold=0.2)
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'A big dog.', 'text_to_match': 'The big cat?', 'rank': '1=', 'index_within_document': 1, 'subword_index': None, 'start_index': 1, 'end_index': 1, 'sentences_start_index': 0, 'sentences_end_index': 3, 'sentences_character_start_index': 0, 'sentences_character_end_index': 10, 'score': 7.381404928570852, 'word_infos': [[2, 5, 'single', True, 'Matches BIG directly.']], 'answers': []}, {'document_label': 'q', 'text': 'A big dog.', 'text_to_match': 'The big cat?', 'rank': '1=', 'index_within_document': 5, 'subword_index': None, 'start_index': 5, 'end_index': 5, 'sentences_start_index': 4, 'sentences_end_index': 7, 'sentences_character_start_index': 11, 'sentences_character_end_index': 21, 'score': 7.381404928570852, 'word_infos': [[2, 5, 'single', True, 'Matches BIG directly.']], 'answers': []}, {'document_label': 'q', 'text': 'A big dog.', 'text_to_match': 'The big cat?', 'rank': '1=', 'index_within_document': 9, 'subword_index': None, 'start_index': 9, 'end_index': 9, 'sentences_start_index': 8, 'sentences_end_index': 11, 'sentences_character_start_index': 22, 'sentences_character_end_index': 32, 'score': 7.381404928570852, 'word_infos': [[2, 5, 'single', True, 'Matches BIG directly.']], 'answers': []}])
