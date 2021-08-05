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
        print(topic_matches)
        self.assertEqual(int(topic_matches[0]['score']), highest_score)
        if answer_start is not None:
            self.assertEqual(topic_matches[0]['answers'][0][0], answer_start)
            self.assertEqual(topic_matches[0]['answers'][0][1], answer_end)
        else:
            self.assertEqual(len(topic_matches[0]['answers']), 0)

    def test_check_when_entity(self):
        self._check_equals('When did the meeting take place?', 'the meeting took place at 3pm', 143, 26, 29)
