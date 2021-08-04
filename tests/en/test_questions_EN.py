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
        use_frequency_factor=True):
        manager.remove_all_documents()
        manager.parse_and_register_document(document_text)
        topic_matches = manager.topic_match_documents_against(text_to_match,
                                                              word_embedding_match_threshold=
                                                              word_embedding_match_threshold,
                                                              initial_question_word_embedding_match_threshold=initial_question_word_embedding_match_threshold,
                                                              relation_score=20,
                                                              reverse_only_relation_score=15, single_word_score=10, single_word_any_tag_score=5,
                                                              different_match_cutoff_score=10,
                                                              relation_matching_frequency_threshold=0.0,
                                                              embedding_matching_frequency_threshold=0.0,
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

    def test_governed_interrogative_pronoun_matching_proper_noun(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("Richard Hudson sang a duet.", 'q')
        topic_matches = manager.topic_match_documents_against("Which person sings?")
        self.assertEqual([{'document_label': 'q', 'text': 'Richard Hudson sang a duet.', 'text_to_match': 'Which person sings?', 'rank': '1', 'index_within_document': 2, 'subword_index': None, 'start_index': 1, 'end_index': 2, 'sentences_start_index': 0, 'sentences_end_index': 5, 'sentences_character_start_index': 0, 'sentences_character_end_index': 27, 'score': 620.0, 'word_infos': [[8, 14, 'relation', False, 'Has an entity label that is 100% similar to the word embedding of PERSON.'], [15, 19, 'relation', True, 'Matches SING directly.']], 'answers': [[0, 14]]}], topic_matches)

    def test_check_who_positive_case(self):
        self._check_equals('Who looked into the sun?', 'the man looked into the sun', 950, 0, 7)

    def test_check_who_wrong_syntax(self):
        self._check_equals('Who looked into the sun?', 'the sun looked into the man', 19, None, None)

    def test_check_who_wrong_noun(self):
        self._check_equals('Who looked into the sun?', 'the dog looked into the sun', 70, None, None)
