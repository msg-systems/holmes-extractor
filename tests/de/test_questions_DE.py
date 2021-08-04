import unittest
import holmes_extractor as holmes
from holmes_extractor.topic_matching import TopicMatcher

manager = holmes.Manager(model='de_core_news_lg', number_of_workers=1)

class GermanInitialQuestionsTest(unittest.TestCase):

    def _check_equals(self, text_to_match, document_text, highest_score, answer_start, answer_end,
        word_embedding_match_threshold=0.42, initial_question_word_embedding_match_threshold=0.42,
        use_frequency_factor=True):
        manager.remove_all_documents()
        manager.parse_and_register_document(document_text)
        topic_matches = manager.topic_match_documents_against(text_to_match,
                                                              word_embedding_match_threshold=
                                                              word_embedding_match_threshold,
                                                              initial_question_word_embedding_match_threshold=initial_question_word_embedding_match_threshold,
                                                              initial_question_word_answer_score=40,
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

    def test_basic_matching_with_subword(self):
        self._check_equals("Was betrachtet man?", 'Informationsbetrachtung', 45, 0, 11)

    def test_governed_interrogative_pronoun_with_subword(self):
        self._check_equals("Welche Information betrachtet man?", 'Informationsbetrachtung', 55, 0, 11)

    def test_governed_interrogative_pronoun_with_subword_control(self):
        self._check_equals("Die Information betrachtet man.", 'Informationsbetrachtung', 35, None, None)

    def test_governed_interrogative_pronoun_with_complex_subword(self):
        self._check_equals("Welche Information betrachtet man?",
            'Extraktionsinformationsbetrachtung', 55, 0, 22)

    def test_governed_interrogative_pronoun_with_complex_subword_control(self):
        self._check_equals("Die Information betrachtet man.",
            'Extraktionsinformationsbetrachtung', 35, None, None)

    def test_governed_interrogative_pronoun_with_subword_and_coreference(self):
        self._check_equals("Welchen Löwen betrachten wir.", 'Es gab einen Extraktionslöwen. Leute haben ihn betrachtet', 54, 13, 29)

    def test_governed_interrogative_pronoun_with_subword_and_coreference_control(self):
        self._check_equals("Den Löwen betrachten wir.", 'Es gab einen Extraktionslöwen. Leute haben ihn betrachtet', 34, None, None)

    def test_governed_interrogative_pronoun_with_subword_and_embedding_matching(self):
        self._check_equals("Welchen Hund betrachten wir?", 'Leute betrachteten die Informationskatze', 25, 23, 40)

    def test_governed_interrogative_pronoun_with_subword_and_embedding_matching_control(self):
        self._check_equals("Den Hund betrachten wir.", 'Leute betrachteten den Informationskatze', 15, None, None)
