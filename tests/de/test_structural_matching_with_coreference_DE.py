import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
coref_holmes_manager = holmes.Manager(model='de_core_news_lg',
                                      number_of_workers=2)
coref_holmes_manager.register_search_phrase("Ein Hund jagt eine Katze")
coref_holmes_manager.register_search_phrase("Ein großes Pferd jagt eine Katze")
coref_holmes_manager.register_search_phrase("Ein Tiger jagt eine kleine Katze")
coref_holmes_manager.register_search_phrase("Ein großer Löwe jagt eine Katze")
coref_holmes_manager.register_search_phrase("Ein ENTITYPER braucht Versicherung")
coref_holmes_manager.register_search_phrase("A big company makes a loss")
coref_holmes_manager.register_search_phrase(
    "A dog who chases rats chases mice")
coref_holmes_manager.register_search_phrase("ein müder Hund")
coref_holmes_manager.register_search_phrase("Ein Gepard jagt einen Gepard")
coref_holmes_manager.register_search_phrase("Ein Leopard jagt einen Leopard")

coref_holmes_manager.register_search_phrase("Ein Urlaub ist schwer zu finden")
coref_holmes_manager.register_search_phrase("A man sings")
coref_holmes_manager.register_search_phrase("Somebody finds a policy")
coref_holmes_manager.register_search_phrase(
    "Somebody writes a book about an animal")
coref_holmes_manager.register_search_phrase("Hermione breaks")
coref_holmes_manager.register_search_phrase("Jemand versucht, zu erklären")
coref_holmes_manager.register_search_phrase("An adopted boy")
coref_holmes_manager.register_search_phrase("A running boy")
nocoref_holmes_manager = holmes.Manager(model='de_core_news_lg',
                                        perform_coreference_resolution=False,
                                        number_of_workers=1)
nocoref_holmes_manager.register_search_phrase("Ein Hund jagt eine Katze")


class CoreferenceEnglishMatchingTest(unittest.TestCase):

    def _check_word_match(self, match, word_match_index, document_token_index, extracted_word):
        word_match = match['word_matches'][word_match_index]
        self.assertEqual(word_match['document_token_index'], document_token_index)
        self.assertEqual(word_match['extracted_word'], extracted_word)

    def test_simple_pronoun_coreference_same_sentence(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah einen Hund, und er jagte eine Katze.")
        matches = coref_holmes_manager.match()
        self._check_word_match(matches[0], 0, 3, 'Hund')
        self._check_word_match(matches[0], 1, 7, 'jagen')
        self._check_word_match(matches[0], 2, 9, 'Katze')

    def test_perform_coreference_resolution_false(self):
        nocoref_holmes_manager.remove_all_documents()
        nocoref_holmes_manager.parse_and_register_document(
            "Ich sah einen Hund, und er jagte eine Katze.")
        matches = nocoref_holmes_manager.match()
        self.assertEqual(len(matches), 0)

    def test_simple_pronoun_coreference_same_sentence_wrong_structure(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah einen Hund und er wurde von einer Katze gejagt.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 0)

    def test_simple_pronoun_coreference_same_sentence_plural_antecedent(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah Hunde und sie jagten eine Katze.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 2, 'hund')

    def test_simple_pronoun_coreference_same_sentence_conjunction_in_antecedent_both_match(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah einen Hund und einen Hund, und die jagten eine Katze.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 2)
        self._check_word_match(matches[0], 0, 3, 'Hund')
        self._check_word_match(matches[1], 0, 6, 'Hund')

    def test_simple_pronoun_coreference_same_sentence_conjunction_in_antecedent_left_matches(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah einen Hund und ein Pferd, und sie jagten eine Katze.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 3, 'Hund')

    def test_simple_pronoun_coreference_same_sentence_conjunction_in_antecedent_right_matches(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah ein Pferd und einen Hund, und die jagten eine Katze.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 6, 'Hund')

    def test_simple_pronoun_coreference_same_sentence_conjunction_pronouns_both_match(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich redete mit Peter Müller und Jana Müller, während er und sie Versicherung brauchten.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 2)
        self._check_word_match(matches[0], 0, 4, 'Peter Müller')
        self._check_word_match(matches[1], 0, 7, 'Jana Müller')

    def test_simple_pronoun_coreference_same_sentence_conjunction_lefthand_is_pronoun(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich redete mit Peter Müller, während er und Jana Müller Versicherung brauchten.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 2)
        self._check_word_match(matches[0], 0, 4, 'Peter Müller')
        self._check_word_match(matches[1], 0, 10, 'Jana Müller')

    def test_simple_pronoun_coreference_same_sentence_conjunction_righthand_is_pronoun(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "I redete mit Jana Müller, während Peter Müller und sie Versicherung brauchten.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 2)
        self._check_word_match(matches[0], 0, 8, 'Peter Müller')
        self._check_word_match(matches[1], 0, 4, 'Jana Müller')

    def test_simple_pronoun_coreference_same_sentence_conjunction_righthand_noun_not_match(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich redete mit Peter Müller, während er und ein Pferd Versicherung brauchten.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 4, 'Peter Müller')

    def test_simple_pronoun_coreference_diff_sentence(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah eine Katze. Ein Hund jagte sie.")
        matches = coref_holmes_manager.match()
        self._check_word_match(matches[0], 0, 6, 'Hund')
        self._check_word_match(matches[0], 1, 7, 'jagen')
        self._check_word_match(matches[0], 2, 3, 'Katze')

    def test_simple_pronoun_coreference_diff_sentence_wrong_structure(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah einen Hund. Er wurde durch eine Katze gejagt.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 0)

    def test_simple_pronoun_coreference_diff_sentence_plural_antecedent(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah Katzen. Sie wurden durch einen Hund gejagt.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 2, 2, 'katze')

    def test_simple_pronoun_coreference_diff_sentence_conjunction_in_antecedent_both_match(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah eine Katze und eine Katze. Ein Hund hat sie gejagt.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 2)
        self._check_word_match(matches[0], 2, 3, 'Katze')
        self._check_word_match(matches[1], 2, 6, 'Katze')

    def test_simple_pronoun_coreference_diff_sentence_conjunction_in_antecedent_left_matches(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah eine Katze und ein Pferd. Ein Hund hat sie gejagt.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 2, 3, 'Katze')

    def test_simple_pronoun_coreference_diff_sentence_conjunction_in_antecedent_right_matches(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah ein Pferd und eine Katze. Ein Hund hat sie gejagt")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 2, 6, 'Katze')

    def test_pronoun_coreferent_has_dependency_same_sentence(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah ein großes Pferd und es jagte eine Katze.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 3, 'groß')
        self._check_word_match(matches[0], 1, 4, 'Pferd')

    def test_pronoun_coreferents_with_dependency_conjunction_same_sentence_both_match(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah ein großes Pferd und ein großes Pferd und sie jagten eine Katze.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 2)
        self._check_word_match(matches[0], 0, 3, 'groß')
        self._check_word_match(matches[0], 1, 4, 'Pferd')
        self._check_word_match(matches[1], 0, 7, 'groß')
        self._check_word_match(matches[1], 1, 8, 'Pferd')

    def test_noun_coreferent_has_dependency_same_sentence(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah ein großes Pferd, und das Pferd jagte eine Katze.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 3, 'groß')
        self._check_word_match(matches[0], 1, 8, 'Pferd')

    def test_pronoun_coreferent_has_dependency_three_sentences(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ich sah ein Pferd. Es jagte eine Katze. Es war groß")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 12, 'groß')
        self._check_word_match(matches[0], 1, 3, 'Pferd')

    def test_reflexive_pronoun_coreferent(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Der Gepard jagte sich")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 1, 'Gepard')
        self._check_word_match(matches[0], 2, 1, 'Gepard')

    def test_reflexive_pronoun_coreferents_with_conjunction_same_noun(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Der Gepard und der Gepard jagten sich")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 4)
        self._check_word_match(matches[0], 0, 1, 'Gepard')
        self._check_word_match(matches[0], 2, 1, 'Gepard')
        self._check_word_match(matches[1], 0, 4, 'Gepard')
        self._check_word_match(matches[1], 2, 1, 'Gepard')
        self._check_word_match(matches[2], 0, 1, 'Gepard')
        self._check_word_match(matches[2], 2, 4, 'Gepard')
        self._check_word_match(matches[3], 0, 4, 'Gepard')
        self._check_word_match(matches[3], 2, 4, 'Gepard')

    def test_reflexive_pronoun_coreferents_with_conjunction_diff_noun(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Der Gepard und der Leopard jagten sich")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 2)
        self._check_word_match(matches[0], 0, 1, 'Gepard')
        self._check_word_match(matches[0], 2, 1, 'Gepard')
        self._check_word_match(matches[1], 0, 4, 'Leopard')
        self._check_word_match(matches[1], 0, 4, 'Leopard')

    def test_repeated_noun(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Wir sahen einen großes Hund. Der Hund jagte eine Katze")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 7, 'Hund')

    def test_repeated_noun_match_both_mentions(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Wir sahen einen müden Hund. Der Hund jagte einen Esel")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 2)
        self._check_word_match(matches[0], 0, 3, 'müde')
        self._check_word_match(matches[0], 1, 4, 'Hund')
        self._check_word_match(matches[1], 0, 3, 'müde')
        self._check_word_match(matches[1], 1, 7, 'Hund')

    def test_mentions_following_structural_match(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "Ein großes Pferd jagte eine Katze. Das Pferd war glücklich.")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self._check_word_match(matches[0], 0, 1, 'groß')
        self._check_word_match(matches[0], 1, 2, 'Pferd')

    def test_adjective_verb_phrase_as_search_phrase_matches_simple(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            """Wir besprachen einen Urlaub. Er war sehr schwer zu finden.""")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0]['uncertain'])

    def test_coreference_and_derivation(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            """Ich wollte eine Erklärung. Der Nachbar hat sie versucht.""")
        matches = coref_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['word_matches'][1]['match_type'], 'derivation')
