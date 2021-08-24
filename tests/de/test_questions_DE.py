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

    def test_check_was_predicate_positive_case(self):
        manager.remove_all_documents()
        manager.parse_and_register_document("Das ist ein Haus.", 'q')
        topic_matches = manager.topic_match_documents_against("Was ist das?")
        self.assertEqual(topic_matches, [{'document_label': 'q', 'text': 'Das ist ein Haus.', 'text_to_match': 'Was ist das?', 'rank': '1', 'index_within_document': 1, 'subword_index': None, 'start_index': 0, 'end_index': 3, 'sentences_start_index': 0, 'sentences_end_index': 4, 'sentences_character_start_index': 0, 'sentences_character_end_index': 17, 'score': 620.0, 'word_infos': [[0, 3, 'relation', False, 'Matches the question word WAS.'], [4, 7, 'relation', True, 'Matches SEIN directly.'], [12, 16, 'relation', False, 'Matches the question word WAS.']], 'answers': [[0, 3], [8, 16]]}])

    def test_check_wer_positive_case(self):
        self._check_equals('Wer schaute in die Sonne?', 'Die Person schaute in die Sonne', 127, 0, 10)

    def test_check_wer_wrong_syntax(self):
        self._check_equals('Wer schaute in die Sonne?', 'Die Sonne schaute in den Mann', 19, None, None)

    def test_check_wer_wrong_noun(self):
        self._check_equals('Wer schaute in die Sonne?', 'Das Gebäude schaute in die Sonne', 70, None, None)

    def test_check_wen_positive_case(self):
        self._check_equals('Wen sah das Gebäude?', 'Das Gebäude sah die Person', 54, 16, 26)

    def test_check_wen_wrong_syntax(self):
        self._check_equals('Wen sah das Gebäude?', 'Das Gebäude sah das Gebäude', 34, None, None)

    def test_check_was_acc(self):
        self._check_equals('Was sah das Gebäude?', 'Das Gebäude sah das Gebäude', 104, 16, 27)

    def test_check_wem_positive_case(self):
        self._check_equals('Wem hilfst du?', 'Ich helfe der Person', 45, 10, 20)

    def test_check_wem_non_personal_noun(self):
        self._check_equals('Wem hilfst du?', 'Ich helfe dem Gebäude', 45, 10, 21)

    def test_check_wo_positive_case(self):
        self._check_equals('Wo wohnst du?', 'Ich wohne in einem Haus', 45, 10, 23)

    def test_check_wo_positive_case_definite_preposition(self):
        self._check_equals('Wo wohnst du?', 'Ich wohne im Haus', 45, 10, 17)

    def test_check_wo_wrong_case_definite_preposition(self):
        self._check_equals('Wo wohnst du?', 'Ich wohne ins Haus', 5, None, None)

    def test_check_wo_wrong_case(self):
        self._check_equals('Wo wohnst du?', 'Ich wohne in ein Haus', 5, None, None)

    def test_check_wohin_positive_case(self):
        self._check_equals('Wohin fährst du?', 'Ich fahre in ein Haus', 45, 10, 21)

    def test_check_wohin_positive_case_definite_preposition(self):
        self._check_equals('Wohin fährst du?', 'Ich fahre ins Haus', 45, 10, 18)

    def test_check_wohin_wrong_case_definite_preposition(self):
        self._check_equals('Wohin fährst du?', 'Ich fahre im Haus', 5, None, None)

    def test_check_womit_positive_case(self):
        self._check_equals('Womit fährst du?', 'Ich fahre mit meinem Auto', 45, 10, 25)

    def test_check_womit_other_preposition(self):
        self._check_equals('Womit fährst du?', 'Ich fahre ohne mein Auto', 5, None, None)

    def test_check_wann_noun(self):
        self._check_equals('Wann fährst du?', 'Ich fahre nächste Woche', 45, 10, 23)

    def test_check_wann_preposition(self):
        self._check_equals('Wann fährst du?', 'Ich fahre in zwei Wochen', 45, 10, 24)

    def test_check_wann_wrong_preposition(self):
        self._check_equals('Wann fährst du?', 'Ich fahre wegen des Problems', 5, None, None)

    def test_check_wann_adverb(self):
        self._check_equals('Wann fährst du?', 'Ich fahre morgen', 45, 10, 16)

    def test_check_wann_verb_phrase(self):
        self._check_equals('Wann fährst du?', 'Ich fahre, wenn du mitkommst.', 45, 11, 28)

    def test_check_wie_preposition(self):
        self._check_equals('Wie fährst du?', 'Ich fahre mit dem Auto', 45, 10, 22)

    def test_check_wie_wrong_preposition(self):
        self._check_equals('Wie fährst du?', 'Ich fahre wegen des Problems', 5, None, None)

    def test_check_wie_adverb(self):
        self._check_equals('Wie fährst du?', 'Ich fahre langsam', 45, 10, 17)

    def test_check_wie_indem_phrase(self):
        self._check_equals('Wie fährst du?', 'Ich fahre, indem ich per Anhalter fahre', 45, 11, 39)

    def test_check_wie_other_phrase(self):
        self._check_equals('Wie fährst du?', 'Ich fahre, weil ich per Anhalter fahre', 5, None, None)

    def test_check_woher_preposition(self):
        self._check_equals('Woher denkst Du es?', 'Ich denke es wegen der Evidenz', 45, 13, 30)

    def test_check_woher_wrong_preposition(self):
        self._check_equals('Woher denkst Du es?', 'Ich denke es trotz der Evidenz', 5, None, None)

    def test_check_woher_weil(self):
        self._check_equals('Woher denkst Du es?', 'Ich denke es, weil es stimmt', 45, 14, 28)

    def test_check_woher_wrong_conjunction(self):
        self._check_equals('Woher denkst Du es?', 'Ich denke es, obwohl es nicht stimmt', 5, None, None)

    def test_check_warum_preposition(self):
        self._check_equals('Warum machst Du es?', 'Ich mache es wegen der Evidenz', 45, 13, 30)

    def test_check_warum_wrong_preposition(self):
        self._check_equals('Warum machst Du es?', 'Ich mache es trotz der Evidenz', 5, None, None)

    def test_check_warum_weil(self):
        self._check_equals('Warum machst Du es?', 'Ich mache es, weil es stimmt', 45, 14, 28)

    def test_check_warum_weil_sein(self):
        self._check_equals('Warum machst Du es?', 'Ich mache es, weil es gut ist', 45, 14, 29)

    def test_check_warum_damit(self):
        self._check_equals('Wieso machst Du es?', 'Ich mache es, damit Du kommst', 45, 14, 29)

    def test_check_warum_wrong_conjunction(self):
        self._check_equals('Woher machst Du es?', 'Ich mache es, obwohl es nicht stimmt', 5, None, None)
