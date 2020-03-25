import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
holmes_manager = holmes.Manager(model='de_core_news_md', ontology=ontology)
holmes_manager.register_search_phrase("Ein Hund jagt eine Katze")
holmes_manager.register_search_phrase("Ein Hund jagt einen Bären")
holmes_manager.register_search_phrase("Ein Hund frisst einen Knochen")
holmes_manager.register_search_phrase("Ein Mann ist schlau")
holmes_manager.register_search_phrase("Der reiche Mann")
holmes_manager.register_search_phrase("Jemand hat einen Berg gesehen")
holmes_manager.register_search_phrase("Jemand soll einen Fluss sehen")
holmes_manager.register_search_phrase("Ein Student geht aus", "excursion")
holmes_manager.register_search_phrase("Der Abschluss einer Versicherung")
holmes_manager.register_search_phrase("Die Kündigung von einer Versicherung")
holmes_manager.register_search_phrase("Jemand schließt eine Versicherung ab")
holmes_manager.register_search_phrase("Jemand findet eine Versicherung")
holmes_manager.register_search_phrase("Wer war traurig?")
holmes_manager.register_search_phrase("Das Fahrzeug hat einen Fehler")
holmes_manager.register_search_phrase("Jemand braucht eine Versicherung für fünf Jahre")
holmes_manager.register_search_phrase("Jemand braucht etwas für fünf Jahre")
holmes_manager.register_search_phrase("Jemand braucht für fünf Jahre")
holmes_manager.register_search_phrase("Ein Urlaub ist schwer zu buchen")
holmes_manager.register_search_phrase("Ein Mann geht aus")
holmes_manager.register_search_phrase("Ein Mann singt")
holmes_manager.register_search_phrase("Eine Party in den Bergen")
holmes_manager.register_search_phrase("Jemand wandert in den Bergen")
holmes_manager.register_search_phrase("Jemand eröffnet ein Konto für ein Kind")
holmes_manager.register_search_phrase("Extraktion der Information")
holmes_manager.register_search_phrase("Maßnahmen der Beschaffung der Information")
holmes_manager.register_search_phrase("Die Linguistik")
holmes_manager.register_search_phrase("Das große Interesse")
holmes_manager.register_search_phrase("Knochenmark wird extrahiert")
holmes_manager.register_search_phrase("Ein großes Wort-Mit-Bindestrich")
holmes_manager.register_search_phrase("Ein kleines Wortmitbindestrich")
holmes_manager.register_search_phrase("Ein großes Wort-Ohne-Bindestrich")
holmes_manager.register_search_phrase("Ein kleines Wortohnebindestrich")
holmes_manager.register_search_phrase("Einfach-Wort-Mit-Bindestrich")
holmes_manager.register_search_phrase("Einfachwortohnebindestrich")
holmes_manager.register_search_phrase("Wort-Mit-Bindestrich-Nicht-In-Ontologie")
holmes_manager.register_search_phrase("Wortohnebindestrichnichtinontologie")
holmes_manager_with_variable_search_phrases = holmes.Manager(model='de_core_news_md')
holmes_manager_with_embeddings = holmes.Manager(model='de_core_news_md',
        overall_similarity_threshold=0.7, perform_coreference_resolution=False,
        embedding_based_matching_on_root_words=True)
holmes_manager_with_embeddings.register_search_phrase("Ein Mann sieht einen großen Hund")
holmes_manager_with_embeddings.register_search_phrase("Der Himmel ist grün")
holmes_manager_with_embeddings.register_search_phrase("Ein König tritt zurück")
holmes_manager_with_embeddings.register_search_phrase("Die Abdankung eines Königs")
holmes_manager_with_embeddings.register_search_phrase("Informationskönig")

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

    def test_matching_with_negation_in_subordinate_clause(self):
        matches = self._get_matches(holmes_manager,
                "Es wurde nie behauptet, dass ein Hund eine Katze gejagt hatte.")
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

    def test_generic_pronoun_with_auxiliary(self):
        matches = self._get_matches(holmes_manager, "Ein Berg wurde gesehen")
        self.assertEqual(len(matches), 1)

    def test_generic_pronoun_with_modal(self):
        matches = self._get_matches(holmes_manager, "Ein Fluss wurde gesehen")
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
        self.assertEqual(len(matches), 2)
        self.assertFalse(matches[0].is_uncertain)

    def test_von_phrase_matches_genitive_phrase_with_coordination(self):
        matches = self._get_matches(holmes_manager,
                "Der Abschluss und der Abschluss von einer Versicherung und einer Versicherung")
        self.assertEqual(len(matches), 4)
        for match in matches:
            self.assertFalse(match.is_uncertain)

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
                "Hunde, Katzen, Löwen und Elefantenelefanten")
        self.assertEqual(len(matches), 4)

    def test_entitynoun_as_non_root_node(self):
        matches = self._get_matches(holmes_manager, "Das Fahrzeug hat einen Fehler.")
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Ich sah ein ENTITYNOUN")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Ich sah einen Hund und eine Elefantenkatze")
        self.assertEqual(len(matches), 2)

    def test_entity_token_does_not_match_subwords(self):
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Ein ENTITYMISC")

    def test_entitynoun_as_non_root_node(self):
        matches = self._get_matches(holmes_manager, "Das Fahrzeug hat einen Fehler.")
        holmes_manager_with_variable_search_phrases.remove_all_search_phrases()
        holmes_manager_with_variable_search_phrases.register_search_phrase(
                "Ich sah ein ENTITYNOUN")
        matches = self._get_matches(holmes_manager_with_variable_search_phrases,
                "Ich sah einen Hund und eine Elefantenkatze")
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

    def test_adjective_verb_phrase_as_search_phrase_matches_simple(self):
        matches = self._get_matches(holmes_manager,
                "Der Urlaub war sehr schwer zu buchen")
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_uncertain)

    def test_adjective_verb_phrase_as_search_phrase_no_match_with_normal_phrase(self):
        matches = self._get_matches(holmes_manager,
                "Der Urlaub wurde gebucht")
        self.assertEqual(len(matches), 0)

    def test_adjective_verb_phrase_as_search_phrase_matches_compound(self):
        matches = self._get_matches(holmes_manager,
                "Der Urlaub und der Urlaub waren sehr schwer und schwer zu buchen und zu buchen")
        self.assertEqual(len(matches), 8)
        for match in matches:
            self.assertFalse(match.is_uncertain)

    def test_objective_adjective_verb_phrase_separate_zu_matches_normal_search_phrase_simple(self):
        matches = self._get_matches(holmes_manager,
                "Die Versicherung war sehr schwer zu finden")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_objective_adjective_verb_phrase_separate_zu_matches_normal_search_phrase_compound(self):
        matches = self._get_matches(holmes_manager,
                "Die Versicherung und die Versicherung waren sehr schwer und schwer zu finden und zu finden")
        self.assertEqual(len(matches), 4)
        for match in matches:
            self.assertTrue(match.is_uncertain)

    def test_objective_adjective_verb_phrase_integrated_zu_matches_normal_search_phrase_simple(self):
        matches = self._get_matches(holmes_manager,
                "Die Versicherung war sehr schwer abzuschließen")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_objective_adjective_verb_phrase_integrated_zu_matches_normal_search_phrase_compound(self):
        matches = self._get_matches(holmes_manager,
                "Die Versicherung und die Versicherung waren sehr schwer und schwer abzuschließen und abzuschließen")
        self.assertEqual(len(matches), 4)
        for match in matches:
            self.assertTrue(match.is_uncertain)

    def test_subjective_adjective_verb_phrase_separate_zu_matches_normal_search_phrase_simple(self):
        matches = self._get_matches(holmes_manager,
                "Der Mann war sehr froh zu singen")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_subjective_adjective_verb_phrase_separate_zu_matches_normal_search_phrase_compound(self):
        matches = self._get_matches(holmes_manager,
                "Der Mann und der Mann waren sehr froh zu singen und zu singen")
        self.assertEqual(len(matches), 4)
        for match in matches:
            self.assertTrue(match.is_uncertain)

    def test_subjective_adjective_verb_phrase_integrated_zu_matches_normal_search_phrase_simple(self):
        matches = self._get_matches(holmes_manager,
                "Der Mann war sehr froh auszugehen")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_subjective_adjective_verb_phrase_integrated_zu_matches_normal_search_phrase_compound(self):
        matches = self._get_matches(holmes_manager,
                "Der Mann und der Mann waren sehr froh auszugehen")
        self.assertEqual(len(matches), 2)
        for match in matches:
            self.assertTrue(match.is_uncertain)

    def test_german_embeddings(self):
        matches = self._get_matches(holmes_manager_with_embeddings,
                "Der Mann sah eine große Katze")
        self.assertEqual(len(matches), 1)

    def test_german_embeddings_inflected_adjective(self):
        matches = self._get_matches(holmes_manager_with_embeddings,
                "Ich wohne im blauen Himmel")
        self.assertEqual(len(matches), 1)

    def test_prepositional_phrase_dependent_on_noun_no_conjunction(self):
        matches = self._get_matches(holmes_manager,
                "Eine Party in den Bergen")
        self.assertEqual(len(matches), 1)

    def test_prepositional_phrase_dependent_on_noun_with_conjunction(self):
        matches = self._get_matches(holmes_manager,
                "Eine Party in den Bergen und den Bergen")
        self.assertEqual(len(matches), 2)

    def test_prepositional_phrase_dependent_on_verb_no_conjunction(self):
        matches = self._get_matches(holmes_manager,
                "Mein Freund wandert in den Bergen")
        self.assertEqual(len(matches), 1)

    def test_prepositional_phrase_dependent_on_verb_with_conjunction(self):
        matches = self._get_matches(holmes_manager,
                "Mein Freund wandert in den Bergen und den Bergen")
        self.assertEqual(len(matches), 2)

    def test_moposs_before_governing_verb(self):
        matches = self._get_matches(holmes_manager,
                "Richard Hudson möchte ein Konto für sein Kind eröffnen")
        self.assertEqual(len(matches), 1)

    def test_separable_verbs_with_embeddings(self):
        matches = self._get_matches(holmes_manager_with_embeddings,
                "Der König dankt ab")
        self.assertEqual(len(matches), 1)

    def test_objective_deverbal_subword_phrase_with_durch_no_conjunction(self):
        matches = self._get_matches(holmes_manager,
                "Die Katzenjagd durch den Hund")
        self.assertEqual(len(matches), 1)

    def test_objective_deverbal_subword_phrase_with_durch_conjunction_within_subwords(self):
        matches = self._get_matches(holmes_manager,
                "Die Katzen- und Katzenjagd durch den Hund")
        self.assertEqual(len(matches), 2)

    def test_objective_deverbal_subword_phrase_with_durch(self):
        matches = self._get_matches(holmes_manager,
                "Die Katzenjagd durch den Hund und den Hund")
        self.assertEqual(len(matches), 2)

    def test_subjective_deverbal_subword_phrase_with_durch(self):
        matches = self._get_matches(holmes_manager,
                "Die Hundenjagd durch die Katze")
        self.assertEqual(len(matches), 0)

    def test_subjective_deverbal_subword_phrase_with_von(self):
        matches = self._get_matches(holmes_manager,
                "Die Hundenjagd von der Katze und der Katze")
        self.assertEqual(len(matches), 2)

    def test_adjectival_subword(self):
        matches = self._get_matches(holmes_manager,
                "Das Großinteresse")
        self.assertEqual(len(matches), 1)

    def test_two_subwords_filling_same_word(self):
        matches = self._get_matches(holmes_manager,
                "Informationsextraktion")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 0)

    def test_two_subwords_at_beginning_of_same_word(self):
        matches = self._get_matches(holmes_manager,
                "Informationsextraktionsmaßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 0)

    def test_two_subwords_at_end_of_same_word(self):
        matches = self._get_matches(holmes_manager,
                "Maßnahmeninformationsextraktion")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)

    def test_two_subwords_in_different_words(self):
        matches = self._get_matches(holmes_manager,
                "Maßnahmenextraktion der Maßnahmeninformation")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)

    def test_two_subwords_two_word_conjunction_first_element(self):
        matches = self._get_matches(holmes_manager,
                "Informationsentnahme und -extraktion")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 0)

    def test_two_subwords_three_word_conjunction_first_element(self):
        matches = self._get_matches(holmes_manager,
                "Informationsentnahme, -extraktion und -freude")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 0)

    def test_two_subwords_two_word_conjunction_last_element(self):
        matches = self._get_matches(holmes_manager,
                "Informations- und Entnahmeextraktion")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 0)

    def test_two_subwords_three_word_conjunction_last_element(self):
        matches = self._get_matches(holmes_manager,
                "Freude-, Informations- und Entnahmeextraktion")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 4)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 2)

    def test_two_subwords_in_middle_element(self):
        matches = self._get_matches(holmes_manager,
                "Freudeverwaltungs--, -informationsextraktions- und -entnahmeverwaltung")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 2)

    def test_three_subwords_filling_same_word_initial_position(self):
        matches = self._get_matches(holmes_manager,
                "Informationsbeschaffungsmaßnahmen waren das, worüber wir sprachen.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)

    def test_three_subwords_filling_same_word_later_position(self):
        matches = self._get_matches(holmes_manager,
                "Wir redeten über Informationsbeschaffungsmaßnahmen.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 3)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 3)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 3)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)

    def test_three_subwords_filling_same_word_beginning_of_word(self):
        matches = self._get_matches(holmes_manager,
                "Informationsbeschaffungsmaßnahmenextraktion.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)

    def test_three_subwords_filling_same_word_end_of_word(self):
        matches = self._get_matches(holmes_manager,
                "Extraktionsinformationsbeschaffungsmaßnahmen.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 3)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 1)

    def test_three_subwords_split_two_one(self):
        matches = self._get_matches(holmes_manager,
                "Maßnahmen der Informationsbeschaffung")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[0].document_subword, None)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 2)

    def test_three_subwords_split_two_one_with_more_subwords(self):
        matches = self._get_matches(holmes_manager,
                "Extraktionsmaßnahmen der Extraktionsinformationsbeschaffung")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 0)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 2)

    def test_three_subwords_split_one_two(self):
        matches = self._get_matches(holmes_manager,
                "Beschaffungsmaßnahmen der Information")
        self.assertEqual(len(matches), 0)

    def test_three_subwords_split_one_two_with_more_subwords(self):
        matches = self._get_matches(holmes_manager,
                "Extraktionsbeschaffungsmaßnahmen der Extraktionsinformation")
        self.assertEqual(len(matches), 0)

    def test_three_subwords_two_word_conjunction_first_elements_two_one(self):
        matches = self._get_matches(holmes_manager,
                "Informationsbeschaffungsprobleme und -maßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 0)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)

    def test_three_subwords_three_word_conjunction_first_elements_two_one(self):
        matches = self._get_matches(holmes_manager,
                "Informationsbeschaffungsprobleme, -maßnahmen und -interessen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 0)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)

    def test_three_subwords_two_word_conjunction_first_elements_one_two(self):
        matches = self._get_matches(holmes_manager,
                "Informationsprobleme und -beschaffungsmaßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)

    def test_three_subwords_three_word_conjunction_first_elements_one_two(self):
        matches = self._get_matches(holmes_manager,
                "Informationsprobleme, -interessen und -beschaffungsmaßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 4)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 4)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 4)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 4)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 4)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)

    def test_three_subwords_two_word_conjunction_last_elements_one_two(self):
        matches = self._get_matches(holmes_manager,
                "Informations- und Interessenbeschaffungsmaßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)

    def test_three_subwords_three_word_conjunction_last_elements_one_two(self):
        matches = self._get_matches(holmes_manager,
                "Informations-, Problem- und Interessenbeschaffungsmaßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 4)
        self.assertEqual(matches[0].word_matches[0].document_word, 'maßnahme')
        self.assertEqual(matches[0].word_matches[1].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 4)
        self.assertEqual(matches[0].word_matches[1].document_word, 'beschaffung')
        self.assertEqual(matches[0].word_matches[2].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)
        self.assertEqual(matches[0].word_matches[2].document_word, 'Information')

    def test_three_subwords_two_word_conjunction_last_elements_two_one(self):
        matches = self._get_matches(holmes_manager,
                "Informationsbeschaffungs- und Interessenmaßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 0)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)

    def test_three_subwords_three_word_conjunction_last_elements_two_one(self):
        matches = self._get_matches(holmes_manager,
                "Informationsbeschaffungs-, Problem- und Interessenmaßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 4)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 0)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)

    def test_three_subwords_three_word_conjunction(self):
        matches = self._get_matches(holmes_manager,
                "Informationsinteressen, -beschaffungs- und Problemmaßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 4)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 2)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 2)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)

    def test_three_subwords_three_word_conjunction_with_other_words(self):
        matches = self._get_matches(holmes_manager,
                "Informationsinteressen, -interessen-, -beschaffungs-, -interessen- und Problemmaßnahmen")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word_matches[0].document_token.i, 4)
        self.assertEqual(matches[0].word_matches[0].document_subword.index, 2)
        self.assertEqual(matches[0].word_matches[0].document_subword.containing_token_index, 8)
        self.assertEqual(matches[0].word_matches[1].document_token.i, 4)
        self.assertEqual(matches[0].word_matches[1].document_subword.index, 1)
        self.assertEqual(matches[0].word_matches[1].document_subword.containing_token_index, 4)
        self.assertEqual(matches[0].word_matches[2].document_token.i, 4)
        self.assertEqual(matches[0].word_matches[2].document_subword.index, 0)
        self.assertEqual(matches[0].word_matches[2].document_subword.containing_token_index, 0)

    def test_uncertain_subword_match_with_or_conjunction(self):
        matches = self._get_matches(holmes_manager,
                "Informationsinteressen oder -extraktion")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_uncertain)

    def test_embedding_match_on_root_subword(self):
        matches = self._get_matches(holmes_manager_with_embeddings,
                "Ein Informationskönig")
        self.assertEqual(len(matches), 1)

    def test_embedding_match_on_non_root_subword(self):
        matches = self._get_matches(holmes_manager_with_embeddings,
                "Die Prinzenabdankung")
        self.assertEqual(len(matches), 1)

    def test_ontology_matching_with_subwords(self):
        matches = self._get_matches(holmes_manager,
                "Die Literaturlinguistik")
        self.assertEqual(len(matches), 1)

    def test_ontology_matching_with_whole_word_containing_subwords(self):
        matches = self._get_matches(holmes_manager,
                "Die Sprachwissenschaft")
        self.assertEqual(len(matches), 1)

    def test_ontology_matching_with_whole_word_and_subword(self):
        matches = self._get_matches(holmes_manager,
                "Die Komputerlinguistik")
        self.assertEqual(len(matches), 2)

    def test_derivation_matching_with_subwords(self):
        matches = self._get_matches(holmes_manager,
                "Knochenmarkextraktion")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_1(self):
        matches = self._get_matches(holmes_manager,
                "Ein großes Wort-Mit-Bindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_2(self):
        matches = self._get_matches(holmes_manager,
                "Ein großes Wortmitbindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_3(self):
        matches = self._get_matches(holmes_manager,
                "Ein kleines Wort-Mit-Bindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_4(self):
        matches = self._get_matches(holmes_manager,
                "Ein kleines Wortmitbindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_5(self):
        matches = self._get_matches(holmes_manager,
                "Ein großes Wort-Ohne-Bindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_6(self):
        matches = self._get_matches(holmes_manager,
                "Ein großes Wortohnebindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_7(self):
        matches = self._get_matches(holmes_manager,
                "Ein kleines Wort-Ohne-Bindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_8(self):
        matches = self._get_matches(holmes_manager,
                "Ein kleines Wortohnebindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_9(self):
        matches = self._get_matches(holmes_manager,
                "Einfachwortmitbindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_10(self):
        matches = self._get_matches(holmes_manager,
                "Einfach-Wort-Ohne-Bindestrich")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_11(self):
        matches = self._get_matches(holmes_manager,
                "Wortmitbindestrichnichtinontologie")
        self.assertEqual(len(matches), 1)

    def test_hyphenation_10(self):
        matches = self._get_matches(holmes_manager,
                "Wort-Ohne-Bindestrich-Nicht-In-Ontologie")
        self.assertEqual(len(matches), 1)
