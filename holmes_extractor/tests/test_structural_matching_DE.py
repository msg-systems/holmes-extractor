import unittest
import holmes_extractor as holmes

holmes_manager = holmes.Manager(model='de_core_news_sm')
holmes_manager.register_search_phrase("Ein Hund jagt eine Katze")
holmes_manager.register_search_phrase("Ein Hund jagt einen Bären")
holmes_manager.register_search_phrase("Ein Hund frisst einen Knochen")
holmes_manager.register_search_phrase("Ein Mann ist schlau")
holmes_manager.register_search_phrase("Der reiche Mann")
holmes_manager.register_search_phrase("Jemand hat einen Berg gesehen")
holmes_manager.register_search_phrase("Ein Student geht aus", "excursion")
holmes_manager.register_search_phrase("Der Abschluss einer Versicherung")
holmes_manager.register_search_phrase("Die Kündigung von einer Versicherung")
holmes_manager.register_search_phrase("Jemand schließt eine Versicherung ab")
holmes_manager.register_search_phrase("Wer war traurig?")
holmes_manager.register_search_phrase("Das Fahrzeug hat einen Fehler")
holmes_manager.register_search_phrase("Jemand braucht eine Versicherung für fünf Jahre")
holmes_manager.register_search_phrase("Jemand braucht etwas für fünf Jahre")
holmes_manager.register_search_phrase("Jemand braucht für fünf Jahre")
holmes_manager_with_variable_search_phrases = holmes.Manager(model='de_core_news_sm')

class GermanStructuralMatchingTest(unittest.TestCase):

    def _get_matches(self, holmes_manager, text):
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(document_text=text)
        return holmes_manager.match()

    def test_direct_matching(self):
        matches = self._get_matches(holmes_manager, "Der Hund jagte die Katze")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_negated)
        self.assertFalse(matches[0].is_uncertain)
        self.assertEqual(matches[0].search_phrase_label, "Ein Hund jagt eine Katze")

    def test_matching_within_large_sentence_with_negation(self):
        matches = self._get_matches(holmes_manager, "Wir haben über Verschiedenes geredet. Obwohl es nie behauptet wurde, dass ein Hund jeweils eine Katze gejagt hätte, entsprach dies dennoch der Wahrheit. Dies war doch immer ein schwieriges Thema.")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_negated)
        self.assertFalse(matches[0].is_uncertain)

    def test_nouns_inverted(self):
        matches = self._get_matches(holmes_manager, "Die Katze jagte den Hund")
        self.assertEqual(len(matches), 0)

    def test_different_object(self):
        matches = self._get_matches(holmes_manager, "Der Hund jagte das Pferd")
        self.assertEqual(len(matches), 0)

    def test_verb_negation(self):
        matches = self._get_matches(holmes_manager, "Der Hund jagte die Katze nicht")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_negated)
        self.assertFalse(matches[0].is_uncertain)

    def test_noun_phrase_negation(self):
        matches = self._get_matches(holmes_manager, "Kein Hund jagte keine Katze")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_negated)
        self.assertFalse(matches[0].is_uncertain)

    def test_irrelevant_negation(self):
        matches = self._get_matches(holmes_manager, "Der nicht alte Hund jagte die Katze")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_negated)
        self.assertFalse(matches[0].is_uncertain)

    def test_adjective_swapping(self):
        matches = self._get_matches(holmes_manager, "Der schlaue Mann")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        matches = self._get_matches(holmes_manager, "Der Mann war reich")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_adjective_swapping_with_conjunction(self):
        matches = self._get_matches(holmes_manager, "Der schlaue und schlaue Mann")
        self.assertEqual(len(matches), 2)
        self.assertFalse(matches[0].is_uncertain)
        self.assertFalse(matches[1].is_uncertain)

        matches = self._get_matches(holmes_manager, "Der Mann war reich und reich")
        self.assertEqual(len(matches), 2)
        self.assertFalse(matches[0].is_uncertain)
        self.assertFalse(matches[1].is_uncertain)

    def test_conjunction_with_and(self):
        matches = self._get_matches(holmes_manager,
                "Der Hund und der Hund jagten die Katze und eine Katze")
        self.assertEqual(len(matches), 4)
        for text_match in matches:
            self.assertFalse(text_match.is_uncertain)

    def test_conjunction_with_or(self):
        matches = self._get_matches(holmes_manager,
                "Der Hund oder der Hund jagten die Katze und eine Katze")
        self.assertEqual(len(matches), 4)
        for text_match in matches:
            self.assertTrue(text_match.is_uncertain)

    def test_threeway_conjunction_with_or(self):
        matches = self._get_matches(holmes_manager,
                "Der Hund, der Hund oder der Hund jagten die Katze und eine Katze")
        self.assertEqual(len(matches), 6)
        for text_match in matches:
            self.assertTrue(text_match.is_uncertain)

    def test_generic_pronoun(self):
        matches = self._get_matches(holmes_manager, "Ein Berg wurde gesehen")
        self.assertEqual(len(matches), 1)

    def test_active(self):
        matches = self._get_matches(holmes_manager, "Der Hund wird die Katze jagen")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        matches = self._get_matches(holmes_manager, "Der Hund hatte die Katze gejagt")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_passive_with_von(self):
        matches = self._get_matches(holmes_manager, "Die Katze wird vom Hund gejagt")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        matches = self._get_matches(holmes_manager, "Die Katze wird vom Hund gejagt werden")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        matches = self._get_matches(holmes_manager, "Die Katze war vom Hund gejagt worden")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        matches = self._get_matches(holmes_manager, "Die Katze wird vom Hund gejagt worden sein")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_passive_with_durch(self):
        matches = self._get_matches(holmes_manager, "Die Katze wird durch den Hund gejagt")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        matches = self._get_matches(holmes_manager, "Die Katze wird durch den Hund gejagt werden")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        matches = self._get_matches(holmes_manager, "Die Katze war durch den Hund gejagt worden")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        matches = self._get_matches(holmes_manager,
                "Die Katze wird durch den Hund gejagt worden sein")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_modal(self):
        matches = self._get_matches(holmes_manager, "Der Hund könnte eine Katze jagen")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_tricky_passive(self):
        matches = self._get_matches(holmes_manager, "Warum der Berg gesehen wurde, ist unklar")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_relative_pronoun_nominative(self):
        matches = self._get_matches(holmes_manager, "Der Hund, der die Katze jagte, war müde")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_relative_pronoun_nominative_inverted(self):
        matches = self._get_matches(holmes_manager, "Die Katze, die den Hund jagte, war müde")
        self.assertEqual(len(matches), 0)

    def test_relative_pronoun_nominative_with_conjunction(self):
        matches = self._get_matches(holmes_manager,
                "Der Hund, der die Katze und die Katze jagte, war müde")
        self.assertEqual(len(matches), 2)
        self.assertFalse(matches[0].is_uncertain)
        self.assertFalse(matches[1].is_uncertain)

    def test_relative_pronoun_nominative_with_passive(self):
        matches = self._get_matches(holmes_manager,
                "Die Katze, die vom Hund gejagt wurde, war müde")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_relative_pronoun_accusative(self):
        matches = self._get_matches(holmes_manager, "Der Bär, den der Hund jagte, war müde")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_separable_verb(self):
        matches = self._get_matches(holmes_manager, "Die Studenten werden ausgehen")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        self.assertEqual(matches[0].search_phrase_label, "excursion")

    def test_von_phrase_matches_genitive_phrase(self):
        matches = self._get_matches(holmes_manager, "Der Abschluss von einer Versicherung")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_von_phrase_matches_genitive_phrase_with_coordination(self):
        matches = self._get_matches(holmes_manager,
                "Der Abschluss und der Abschluss von einer Versicherung und einer Versicherung")
        self.assertEqual(len(matches), 4)
        for text_match in matches:
            self.assertFalse(text_match.is_uncertain)

    def test_genitive_phrase_matches_von_phrase(self):
        matches = self._get_matches(holmes_manager, "Die Kündigung einer Versicherung")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_genitive_phrase_matches_von_phrase_with_coordination(self):
        matches = self._get_matches(holmes_manager,
                "Die Kündigung einer Versicherung und einer Versicherung")
        self.assertEqual(len(matches), 2)
        for text_match in matches:
            self.assertFalse(text_match.is_uncertain)

    def test_subjective_zu_clause_complement_with_conjunction_active(self):
        matches = self._get_matches(holmes_manager,
                "Der Hund und der Löwe entschlossen sich, eine Katze und eine Maus zu jagen")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_adjective_complement_with_conjunction_active(self):
        matches = self._get_matches(holmes_manager,
                "Der Hund war darüber besorgt, eine Katze und eine Maus zu jagen")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_passive_governing_clause_zu_clause_complement_with_conjunction_active(self):
        matches = self._get_matches(holmes_manager,
                "Dem Hund und dem Löwen wurde vorgeschlagen, eine Katze und eine Maus zu jagen")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_verb_complement_simple_passive(self):
        matches = self._get_matches(holmes_manager,
                "Die Katze dachte darüber nach, von einem Hund gejagt zu werden")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_subjective_zu_clause_complement_simple_passive(self):
        matches = self._get_matches(holmes_manager,
                "Die Katze entschied, vom Hund gejagt zu werden")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_um_zu_clause_complement_with_conjunction_passive(self):
        matches = self._get_matches(holmes_manager,
                "Die Katze benutzte den Elefant und die Maus, um vom Hund und Löwen gejagt zu werden")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_passive_search_phrase_with_active_searched_sentence(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Eine Katze wurde von einem Hund gejagt")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Der Hund wird die Katze jagen")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_passive_search_phrase_with_active_conjunction_searched_sentence(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Eine Katze wurde von einem Hund gejagt")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Der Hund und der Hund haben die Katze und die Katze gejagt")
        self.assertEqual(len(matches), 4)
        for text_match in matches:
            self.assertFalse(text_match.is_uncertain)

    def test_passive_search_phrase_with_passive_conjunction_searched_sentence(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Eine Katze wurde von einem Hund gejagt")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Die Katze und die Katze werden von einem Hund und einem Hund gejagt werden")
        self.assertEqual(len(matches), 4)
        for text_match in matches:
            self.assertFalse(text_match.is_uncertain)

    def test_passive_search_phrase_with_negated_searched_sentence(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Eine Katze wurde von einem Hund gejagt")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Der Hund jagte die Katze nie")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        self.assertTrue(matches[0].is_negated)

    def test_question_search_phrase_with_active_searched_sentence(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Welche Hunde fressen Knochen?")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Der Hund wird den Knochen fressen")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_question_search_phrase_with_active_conjunction_searched_sentence(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Welche Hunde fressen Knochen?")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Der Hund und der Hund haben einen Knochen und einen Knochen gefressen")
        self.assertEqual(len(matches), 4)
        for text_match in matches:
            self.assertFalse(text_match.is_uncertain)

    def test_question_search_phrase_with_passive_conjunction_searched_sentence(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Welche Hunde fressen Knochen?")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Der Knochen und der Knochen werden von einem Hund und einem Hund gefressen werden")
        self.assertEqual(len(matches), 4)
        for text_match in matches:
            self.assertFalse(text_match.is_uncertain)

    def test_question_search_phrase_with_negated_searched_sentence(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Welche Hunde fressen Knochen?")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Der Hund fraß den Knochen nie")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)
        self.assertTrue(matches[0].is_negated)

    def test_original_search_phrase_root_not_matchable(self):
        matches = self._get_matches(holmes_manager, "Der Mann war sehr traurig.")
        self.assertEqual(len(matches), 1)

    def test_non_grammatical_auxiliary(self):
        matches = self._get_matches(holmes_manager, "Das Fahrzeug hat einen Fehler.")
        self.assertEqual(len(matches), 1)

    def test_entitynoun_as_root_node(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Ein ENTITYNOUN")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Hunde, Katzen, Löwen und Elefanten")
        self.assertEqual(len(matches), 4)

    def test_entitynoun_as_non_root_node(self):
        matches = self._get_matches(holmes_manager, "Das Fahrzeug hat einen Fehler.")
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Ich sah ein ENTITYNOUN")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Ich sah einen Hund und eine Katze")
        self.assertEqual(len(matches), 2)

    def test_separable_verb_in_main_and_dependent_clauses(self):
        matches = self._get_matches(holmes_manager,
            "Der Mitarbeiter hatte vor, eine Versicherung abzuschließen.")
        self.assertEqual(len(matches), 1)

    def test_matching_additional_preposition_dependency_on_verb(self):
        matches = self._get_matches(holmes_manager,
            "Der Mitarbeiter braucht eine Versicherung für die nächsten fünf Jahre")
        self.assertEqual(len(matches), 3)
        for match in matches:
            if len(match.word_matches) == 5:
                self.assertFalse(match.is_uncertain)
            else:
                self.assertTrue(match.is_uncertain)
                self.assertEqual(len(match.word_matches), 4)

    def test_involves_coreference_false(self):
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(
                "Ein Hund jagte eine Katze.")
        matches = holmes_manager.match()
        self.assertFalse(matches[0].involves_coreference)
        self.assertFalse(matches[0].word_matches[0].involves_coreference)
        self.assertFalse(matches[0].word_matches[0].involves_coreference)
        self.assertFalse(matches[0].word_matches[0].involves_coreference)

    def test_empty_string_does_not_match_entity_search_phrase_token(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase("ENTITYMISC")
        holmes_manager_with_variable_search_phrases.remove_all_documents()
        holmes_manager_with_variable_search_phrases.parse_and_register_document(
                """
                Hier wird in einem Satz etwas besprochen.
                Und hier wird in einem zweiten Satz etwas anderes besprochen.
                """)
        matches = holmes_manager_with_variable_search_phrases.match()
        self.assertEqual(len(matches), 0)

    def test_capital_entity_is_not_analysed_as_entity_search_phrase_token(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase("ENTITY")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Richard Hudson")
        self.assertEqual(len(matches), 0)
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Wir haben eine Entity und eine zweite ENTITY besprochen.")
        self.assertEqual(len(matches), 2)
