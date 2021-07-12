import unittest
import spacy
import coreferee
import holmes_extractor

nlp = spacy.load('de_core_news_lg')
nlp.add_pipe('coreferee')
nlp.add_pipe('holmes')

class GermanSemanticAnalyzerTest(unittest.TestCase):

    def test_initialize_semantic_dependencies(self):
        doc = nlp("Der Hund jagte die Katze.")
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '1:sb; 4:oa')
        self.assertEqual(
            doc[0]._.holmes.string_representation_of_children(), '')
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '')
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '')

    def test_one_righthand_sibling_with_and_conjunction(self):
        doc = nlp("Der Hund und der Löwe jagten die Katze")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4])
        self.assertFalse(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertFalse(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])

    def test_many_righthand_siblings_with_and_conjunction(self):
        doc = nlp(
            "Der Hund, der Hund und der Löwe jagten die Katze")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4, 7])
        self.assertFalse(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertFalse(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertFalse(doc[7]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])
        self.assertEqual(doc[7]._.holmes.righthand_siblings, [])

    def test_one_righthand_sibling_with_or_conjunction(self):
        doc = nlp("Der Hund oder der Löwe jagten die Katze")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4])
        self.assertTrue(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertTrue(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])

    def test_many_righthand_siblings_with_or_conjunction(self):
        doc = nlp(
            "Die Maus, der Hund oder der Löwe jagten die Katze")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4, 7])
        self.assertTrue(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertTrue(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertTrue(doc[7]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])
        self.assertEqual(doc[7]._.holmes.righthand_siblings, [])

    def test_righthand_siblings_of_semantic_children_two(self):
        doc = nlp("Der große und starke Hund kam heim")
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '1:nk; 3:nk')
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [3])

    def test_righthand_siblings_of_semantic_children_many(self):
        doc = nlp("Der große, starke und scharfe Hund kam heim")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:nk; 3:nk; 5:nk')
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [3, 5])
        self.assertEqual(doc[3]._.holmes.righthand_siblings, [])

    def test_semantic_children_of_righthand_siblings_two(self):
        doc = nlp("Der große Hund und Löwe")
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '1:nk; 3:cd')
        self.assertEqual(doc[2]._.holmes.righthand_siblings, [4])
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '1:nk')

    def test_semantic_children_of_righthand_siblings_many(self):
        doc = nlp("Der große Hund, Löwe und Elefant")
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '1:nk; 4:cj')
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '1:nk; 5:cd')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:nk')

    def test_predicative_adjective(self):
        doc = nlp("Der Hund war groß")
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '3:nk')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')
        self.assertTrue(doc[2]._.holmes.is_matchable)

    def test_predicative_adjective_with_conjunction(self):
        doc = nlp("Der Hund und die Katze waren groß und stark")
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '2:cd; 6:nk; 8:nk')
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '6:nk; 8:nk')

    def test_negator_negation_within_clause(self):
        doc = nlp("Der Hund jagte die Katze nicht")
        self.assertEqual(doc[2]._.holmes.is_negated, True)

    def test_operator_negation_within_clause(self):
        doc = nlp("Kein Hund hat irgendeine Katze gejagt")
        self.assertEqual(doc[1]._.holmes.is_negated, True)
        self.assertEqual(doc[2]._.holmes.is_negated, False)
        self.assertFalse(doc[2]._.holmes.is_matchable)

    def test_negator_negation_within_parent_clause(self):
        doc = nlp(
            "Er meinte nicht, dass der Hund die Katze gejagt hätte")
        self.assertEqual(doc[9]._.holmes.is_negated, True)
        self.assertFalse(doc[10]._.holmes.is_matchable)

    def test_operator_negation_within_parent_clause(self):
        doc = nlp(
            "Keiner behauptete, dass der Hund die Katze jagte")
        self.assertEqual(doc[5]._.holmes.is_negated, True)

    def test_negator_negation_within_child_clause(self):
        doc = nlp(
            "Der Hund jagte die Katze, die nicht glücklich war")
        self.assertEqual(doc[2]._.holmes.is_negated, False)

    def test_operator_negation_within_child_clause(self):
        doc = nlp("Der Hund jagte die Katze die es keinem erzählte")
        self.assertEqual(doc[2]._.holmes.is_negated, False)

    def test_dass_clause(self):
        doc = nlp(
            "Er ist zuversichtlich, dass der Hund die Katze jagen wird")
        self.assertEqual(
            doc[9]._.holmes.string_representation_of_children(), '4:cp; 6:sb; 8:oa')

    def test_active_perfect(self):
        doc = nlp("Der Hund hat die Katze gejagt")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:sb; 4:oa')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-6:None')

    def test_active_pluperfect(self):
        doc = nlp("Der Hund hatte die Katze gejagt")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:sb; 4:oa')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-6:None')

    def test_active_future(self):
        doc = nlp("Der Hund wird die Katze jagen")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:sb; 4:oa')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-6:None')

    def test_active_future_perfect(self):
        doc = nlp("Der Hund wird die Katze gejagt haben")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:sb; 4:oa')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-7:None')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '-6:None')
        self.assertFalse(doc[2]._.holmes.is_matchable)
        self.assertFalse(doc[6]._.holmes.is_matchable)

    def test_von_passive_perfect(self):
        doc = nlp("Die Katze ist vom Hund gejagt worden")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:oa; 4:sb')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-7:None')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '-6:None')

    def test_von_passive_pluperfect(self):
        doc = nlp("Die Katze war vom Hund gejagt worden")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:oa; 4:sb')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-7:None')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '-6:None')

    def test_von_passive_future(self):
        doc = nlp("Die Katze wird vom Hund gejagt werden")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:oa; 4:sb')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-7:None')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '-6:None')

    def test_von_passive_future_perfect(self):
        doc = nlp("Die Katze wird vom Hund gejagt worden sein")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:oa; 4:sb')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-8:None')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '-6:None')
        self.assertEqual(
            doc[7]._.holmes.string_representation_of_children(), '-7:None')

    def test_complex_tense_noun_conjunction_active(self):
        doc = nlp(
            "Der Hund und der Löwe haben die Katze und die Maus gejagt")
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '1:sb; 4:sb; 7:oa; 10:oa')

    def test_complex_tense_noun_conjunction_passive(self):
        doc = nlp(
            "Die Katze und die Maus werden vom Hund und Löwen gejagt werden")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                         '1:oa; 4:oa; 7:sb; 9:sb')

    def test_complex_tense_verb_conjunction_active(self):
        doc = nlp(
            "Der Hund wird die Katze gejagt und gefressen haben")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:sb; 4:oa; 6:cd; 8:oc')
        self.assertEqual(
            doc[7]._.holmes.string_representation_of_children(), '1:sb; 4:oa; 8:oc')

    def test_complex_tense_verb_conjunction_passive(self):
        doc = nlp(
            "Die Katze wird vom Hund gejagt und gefressen werden")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:oa; 4:sb; 6:cd')
        self.assertEqual(
            doc[7]._.holmes.string_representation_of_children(), '1:oa; 4:sb')

    def test_conjunction_everywhere_active(self):
        doc = nlp(
            "Der Hund und der Löwe werden die Katze und die Maus jagen und fressen")
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '1:sb; 4:sb; 7:oa; 10:oa; 12:cd')
        self.assertEqual(doc[13]._.holmes.string_representation_of_children(),
                         '1:sb; 4:sb; 7:oa; 10:oa')

    def test_conjunction_everywhere_passive(self):
        doc = nlp(
            "Die Katze und die Maus werden durch den Hund und den Löwen gejagt und gefressen werden")
        self.assertEqual(doc[12]._.holmes.string_representation_of_children(),
                         '1:oa; 4:oa; 8:sb; 11:sb; 13:cd; 15:oc')
        self.assertEqual(doc[14]._.holmes.string_representation_of_children(),
                         '1:oa; 4:oa; 8:sb; 11:sb; 15:oc')

    def test_simple_modal_verb_active(self):
        doc = nlp("Der Hund soll die Katze jagen")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:sb(U); 4:oa(U)')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-6:None')
        self.assertFalse(doc[2]._.holmes.is_matchable)

    def test_simple_modal_verb_passive(self):
        doc = nlp("Die Katze kann vom Hund gejagt werden")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:oa(U); 4:sb(U)')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-7:None')

    def test_negated_modal_verb(self):
        doc = nlp("Der Hund soll die Katze nicht jagen")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:oa(U); 5:ng(U)')
        self.assertTrue(doc[6]._.holmes.is_negated)

    def test_modal_verb_with_conjunction(self):
        doc = nlp(
            "Der Hund und der Löwe können die Katze und die Maus jagen")
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 7:oa(U); 10:oa(U)')
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '-12:None')

    def test_relative_pronoun_nominative(self):
        doc = nlp("Der Hund, der die Katze jagte, war müde")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:sb; 5:oa')

    def test_relative_pronoun_nominative_with_conjunction(self):
        doc = nlp(
            "Der Hund, der die Katze und die Maus jagte, war müde")
        self.assertEqual(
            doc[9]._.holmes.string_representation_of_children(), '1:sb; 5:oa; 8:oa')

    def test_relative_pronoun_nominative_with_passive(self):
        doc = nlp("Die Katze, die vom Hund gejagt wurde, war müde")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:oa; 5:sb')

    def test_relative_pronoun_accusative(self):
        doc = nlp("Der Bär, den der Hund jagte, war müde")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:oa; 5:sb')

    def test_relative_pronoun_conjunction_everywhere_active(self):
        doc = nlp(
            "Der Hund, der Elefant und der Bär, die die Katze und die Maus gejagt und gefressen haben, waren müde")
        self.assertEqual(doc[15]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 7:sb; 11:oa; 14:oa; 16:cd')
        self.assertEqual(doc[17]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 7:sb; 11:oa; 14:oa')

    def test_relative_pronoun_conjunction_everywhere_passive(self):
        doc = nlp(
            "Die Katze, die Maus und der Vogel, die vom Bären, Löwen und Hund gejagt und gefressen worden sind, waren tot")
        self.assertEqual(doc[16]._.holmes.string_representation_of_children(),
                         '1:oa(U); 4:oa(U); 7:oa; 11:sb; 13:sb; 15:sb; 17:cd')
        self.assertEqual(doc[18]._.holmes.string_representation_of_children(),
                         '1:oa(U); 4:oa(U); 7:oa; 11:sb; 13:sb; 15:sb')

    def test_separable_verb(self):
        doc = nlp("Er nimmt die Situation auf")
        self.assertEqual(doc[1]._.holmes.lemma, 'aufnehmen')
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '0:sb; 3:oa')

    def test_separable_verb_in_main_clause_but_infinitive_in_dependent_clause(self):
        doc = nlp("Der Mitarbeiter hatte vor, dies zu tun")
        self.assertEqual(doc[7]._.holmes.lemma, 'tun')

    def test_separable_verb_in_main_clause_but_separable_infinitive_in_dependent_clause(self):
        doc = nlp(
            "Der Mitarbeiter hatte vor, eine Versicherung abzuschließen")
        self.assertEqual(doc[7]._.holmes.lemma, 'abschließen')

    def test_apprart(self):
        doc = nlp("Er geht zur Party")
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '0:sb; 2:op; 3:pobjp')
        self.assertEqual(doc[2].lemma_, 'zur')
        self.assertEqual(doc[2]._.holmes.lemma, 'zu')

    def test_von_phrase(self):
        doc = nlp("Der Abschluss von einer Versicherung")
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '2:mnr; 4:pobjo')

    def test_von_phrase_with_conjunction(self):
        doc = nlp(
            "Der Abschluss und Aufrechterhaltung von einer Versicherung und einem Vertrag")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                         '2:cd; 4:mnr; 6:pobjo; 9:pobjo')
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '4:mnr; 6:pobjo; 9:pobjo')

    def test_von_and_durch_phrase(self):
        doc = nlp(
            "Der Abschluss von einer Versicherung durch einen Makler")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                         '2:mnr; 4:pobjo; 5:mnr; 7:pobjb')

    def test_genitive_and_durch_phrase(self):
        doc = nlp(
            "Der Abschluss einer Versicherung durch einen Makler")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                         '3:ag; 4:mnr; 6:pobjb')

    def test_subjective_zu_clause_complement_simple_active(self):
        doc = nlp("Der Hund überlegte, eine Katze zu jagen")
        self.assertEqual(
            doc[7]._.holmes.string_representation_of_children(), '1:sb(U); 5:oa; 6:pm')

    def test_subjective_zu_clause_complement_with_conjunction_active(self):
        doc = nlp(
            "Der Hund und der Löwe entschlossen sich, eine Katze und eine Maus zu jagen")
        self.assertEqual(doc[14]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 9:oa; 12:oa; 13:pm')

    def test_subjective_zu_clause_complement_with_relative_clause_active(self):
        doc = nlp(
            "Der Hund, der überlegte, eine Katze zu jagen, kam nach Hause")
        self.assertEqual(
            doc[9]._.holmes.string_representation_of_children(), '1:sb(U); 7:oa; 8:pm')

    def test_adjective_complement_simple_active(self):
        doc = nlp("Der Hund war darüber froh, eine Katze zu jagen")
        self.assertEqual(
            doc[9]._.holmes.string_representation_of_children(), '1:sb(U); 7:oa; 8:pm')

    def test_adjective_complement_with_conjunction_active(self):
        doc = nlp(
            "Der Hund war darüber froh, eine Katze und eine Maus zu jagen")
        self.assertEqual(doc[12]._.holmes.string_representation_of_children(),
                         '1:sb(U); 7:oa; 10:oa; 11:pm')

    def test_objective_zu_clause_complement_simple_active(self):
        doc = nlp("Der Löwe bat den Hund, eine Katze zu jagen")
        self.assertEqual(
            doc[9]._.holmes.string_representation_of_children(), '4:sb(U); 7:oa; 8:pm')

    def test_objective_zu_clause_complement_with_conjunction_active(self):
        doc = nlp(
            "Der Elefant schlag dem Hund und dem Löwen vor, eine Katze und eine Maus zu jagen")
        self.assertEqual(doc[16]._.holmes.string_representation_of_children(),
                         '4:sb(U); 7:sb(U); 11:oa; 14:oa; 15:pm')

    def test_passive_governing_clause_zu_clause_complement_simple_active(self):
        doc = nlp("Der Hund wurde gebeten, eine Katze zu jagen")
        self.assertEqual(
            doc[8]._.holmes.string_representation_of_children(), '1:sb(U); 6:oa; 7:pm')

    def test_passive_governing_clause_zu_clause_complement_with_conjunction_active(self):
        doc = nlp(
            "Dem Hund und dem Löwen wurde vorgeschlagen, eine Katze und eine Maus zu jagen")
        self.assertEqual(doc[14]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 9:oa; 12:oa; 13:pm')

    def test_um_zu_clause_complement_simple_active(self):
        doc = nlp(
            "Der Löwe benutzte den Hund, um eine Katze zu jagen")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                         '1:sb(U); 6:cp; 8:oa; 9:pm')

    def test_um_zu_clause_complement_with_conjunction_active(self):
        doc = nlp(
            "Der Elefant benutzte den Hund und den Löwen, um eine Katze und eine Maus zu jagen")
        self.assertEqual(doc[16]._.holmes.string_representation_of_children(),
                         '1:sb(U); 9:cp; 11:oa; 14:oa; 15:pm')

    def test_verb_complement_simple_passive(self):
        doc = nlp(
            "Die Katze dachte darüber nach, von einem Hund gejagt zu werden")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:oa(U); 8:sb; 10:pm')

    def test_verb_complement_with_conjunction_passive(self):
        doc = nlp(
            "Die Katze und die Maus dachten darüber nach, von einem Hund und einem Löwen gejagt zu werden")
        self.assertEqual(doc[15]._.holmes.string_representation_of_children(),
                         '1:oa(U); 4:oa(U); 11:sb; 14:sb; 16:pm')

    def test_verb_complement_with_conjunction_passive_second_pronominal_adverb(self):
        doc = nlp(
            "Die Katze und die Maus dachten darüber und darüber nach, von einem Hund und einem Löwen gejagt zu werden")
        self.assertEqual(doc[17]._.holmes.string_representation_of_children(),
                         '1:oa(U); 4:oa(U); 13:sb; 16:sb; 18:pm')

    def test_verb_complement_with_conjunction_passive_second_dependent_clause(self):
        doc = nlp(
            "Die Katze und die Maus dachten darüber nach, von einem Hund gejagt zu werden und von einem Löwen gejagt zu werden")
        self.assertEqual(doc[12]._.holmes.string_representation_of_children(),
                         '1:oa(U); 4:oa(U); 11:sb; 13:pm; 15:cd')
        self.assertEqual(doc[19]._.holmes.string_representation_of_children(),
                         '1:oa(U); 4:oa(U); 18:sb; 20:pm')

    def test_adjective_complement_simple_passive(self):
        doc = nlp(
            "Die Katze war darüber froh, von einem Hund gejagt zu werden")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:oa(U); 8:sb; 10:pm')

    def test_adjective_complement_with_conjunction_passive(self):
        doc = nlp(
            "Die Katze war darüber froh, von einem Hund und einem Löwen gejagt zu werden")
        self.assertEqual(doc[12]._.holmes.string_representation_of_children(),
                         '1:oa(U); 8:sb; 11:sb; 13:pm')

    def test_subjective_zu_clause_complement_simple_passive(self):
        doc = nlp("Die Katze entschied, vom Hund gejagt zu werden")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:oa(U); 5:sb; 7:pm')

    def test_subjective_zu_clause_complement_with_conjunction_passive(self):
        doc = nlp(
            "Die Katze und die Maus entschlossen sich, vom Hund und Löwen gejagt zu werden")
        self.assertEqual(doc[12]._.holmes.string_representation_of_children(),
                         '1:oa(U); 4:oa(U); 9:sb; 11:sb; 13:pm')

    def test_objective_zu_clause_complement_simple_passive(self):
        doc = nlp(
            "Der Löwe bat die Katze, vom Hund gejagt zu werden")
        self.assertEqual(
            doc[8]._.holmes.string_representation_of_children(), '4:oa(U); 7:sb; 9:pm')

    def test_objective_zu_clause_complement_with_conjunction_passive(self):
        doc = nlp(
            "Der Elefant schlag der Katze und der Maus vor, vom Hund und Löwen gejagt zu werden")
        self.assertEqual(doc[14]._.holmes.string_representation_of_children(),
                         '4:oa(U); 7:oa(U); 11:sb; 13:sb; 15:pm')

    def test_passive_governing_clause_zu_clause_complement_simple_passive(self):
        doc = nlp(
            "Die Katze wurde gebeten, von einem Hund gejagt zu werden")
        self.assertEqual(
            doc[8]._.holmes.string_representation_of_children(), '1:oa(U); 7:sb; 9:pm')

    def test_passive_governing_clause_zu_clause_complement_with_conjunction_passive(self):
        doc = nlp(
            "Der Katze und der Maus wurde vorgeschlagen, von einem Löwen gejagt zu werden")
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '1:oa(U); 4:oa(U); 10:sb; 12:pm')

    def test_um_zu_clause_complement_simple_passive(self):
        doc = nlp(
            "Der Löwe benutzte die Katze, um vom Hund gejagt zu werden")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:oa(U); 6:cp; 8:sb; 10:pm')

    def test_um_zu_clause_complement_with_conjunction_passive(self):
        doc = nlp(
            "Der Elefant benutzte die Katze und die Maus, um vom Hund und Löwen gejagt zu werden")
        self.assertEqual(doc[14]._.holmes.string_representation_of_children(),
                         '1:oa(U); 9:cp; 11:sb; 13:sb; 15:pm')

    def test_verb_complement_with_conjunction_of_dependent_verb(self):
        doc = nlp(
            "Die Katze und die Maus haben entschieden, zu singen und zu schreien")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 8:pm; 10:cd')
        self.assertEqual(doc[12]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 11:pm')

    def test_subjective_zu_clause_complement_with_conjunction_of_dependent_verb(self):
        doc = nlp(
            "Die Katze und die Maus entschlossen sich, zu singen und zu schreien")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 8:pm; 10:cd')
        self.assertEqual(doc[12]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 11:pm')

    def test_objective_zu_clause_complement_with_conjunction_of_dependent_verb(self):
        doc = nlp(
            "Die Katze und die Maus baten den Löwen, zu singen und zu schreien")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                         '7:sb(U); 9:pm; 11:cd')
        self.assertEqual(doc[13]._.holmes.string_representation_of_children(),
                         '7:sb(U); 12:pm')

    def test_um_zu_clause_complement_with_conjunction_of_dependent_verb(self):
        doc = nlp(
            "Die Katze und die Maus benutzen den Löwen, um zu singen und zu schreien")
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 9:cp; 10:pm; 12:cd')
        self.assertEqual(doc[14]._.holmes.string_representation_of_children(),
                         '1:sb(U); 4:sb(U); 9:cp; 13:pm')

    def test_single_preposition_dependency_added_to_verb(self):
        doc = nlp(
            "Der Mitarbeiter braucht eine Versicherung für die nächsten fünf Jahre")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                         '1:sb; 4:oa; 5:moposs(U); 9:pobjp(U)')
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '5:mnr; 9:pobjp')

    def test_multiple_preposition_dependencies_added_to_noun(self):
        doc = nlp(
            "Der Mitarbeiter braucht eine Versicherung für die nächsten fünf Jahre und in Europa")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                         '1:sb; 4:oa; 5:moposs(U); 9:pobjp(U); 11:moposs(U); 12:pobjp(U)')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(
        ), '5:mnr; 9:pobjp; 11:mnr; 12:pobjp')

    def test_no_exception_thrown_when_preposition_dependency_is_righthand_sibling(self):
        doc = nlp(
            "Diese Funktionalität erreichen Sie über Datei/Konfiguration für C")

    def test_phrase_in_parentheses_no_exception_thrown(self):
        doc = nlp(
            "Die Tilgung beginnt in der Auszahlungsphase (d.h. mit der zweiten Auszahlung)")

    def test_von_preposition_in_von_clause_unmatchable(self):
        doc = nlp(
            "Die Kündigung von einer Versicherung")
        self.assertFalse(doc[2]._.holmes.is_matchable)

    def test_self_referring_dependencies_no_exception_thrown_1(self):
        doc = nlp(
            "Die Version ist dabei mit der dieser Bug bereits gefixt sein sollte und nur noch nicht produktiv eingespielt.")

    def test_self_referring_dependencies_no_exception_thrown_2(self):
        doc = nlp(
            "Es sind Papiere, von denen SCD in den Simulationen dann eines auswählt.")

    def test_stripping_adjectival_inflections(self):
        doc = nlp(
            "Eine interessante Überlegung über gesunde Mittagessen.")
        self.assertEqual(doc[1].lemma_, 'interessante')
        self.assertEqual(doc[1]._.holmes.lemma, 'interessant')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '1:nk; 3:op; 5:pobjp')
        self.assertEqual(doc[4].lemma_, 'gesunden')
        self.assertEqual(doc[4]._.holmes.lemma, 'gesund')

    def test_adjective_complement_proper_name(self):
        doc = nlp("Richard war froh, es zu verstehen.")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '0:sb(U); 4:oa; 5:pm')

    def test_adjective_verb_clause_with_zu_subjective_zu_separate_simple(self):
        doc = nlp("Richard war froh zu verstehen.")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:mo; 3:pm')

    def test_adjective_verb_clause_with_zu_subjective_zu_separate_compound(self):
        doc = nlp(
            "Richard und Thomas waren froh und erleichtert zu verstehen und zu begreifen.")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 4:pd; 6:pd; 7:pm; 9:cd')
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 4:pd; 6:pd; 10:pm')

    def test_adjective_verb_clause_with_zu_objective_zu_separate_simple(self):
        doc = nlp("Richard war schwer zu erreichen.")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:mo; 3:pm')

    def test_adjective_verb_clause_with_zu_objective_zu_separate_compound(self):
        doc = nlp(
            "Richard und Thomas war schwer und schwierig zu erreichen und zu bekommen.")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 4:mo; 6:mo; 7:pm; 9:cd')
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 4:mo; 6:mo; 10:pm')

    def test_adjective_verb_clause_with_zu_subjective_zu_integrated_simple(self):
        doc = nlp("Richard war froh hineinzugehen.")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:mo')

    def test_adjective_verb_clause_with_zu_subjective_zu_integrated_compound(self):
        doc = nlp(
            "Richard und Thomas waren froh hineinzugehen und hinzugehen.")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 4:mo; 6:cd')
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 4:mo')

    def test_adjective_verb_clause_with_zu_objective_zu_integrated_simple(self):
        doc = nlp("Richard war leicht einzubinden.")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:mo')

    def test_adjective_verb_clause_with_zu_objective_zu_integrated_compound(self):
        doc = nlp(
            "Richard und Thomas waren leicht einzubinden und aufzugleisen.")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 4:mo; 6:cd')
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 4:mo')

    def test_ungrammatical_two_nominatives(self):
        doc = nlp("Der Hund jagt der Hund")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                         '1:sb; 4:oa')

    def test_ungrammatical_two_nominatives_with_noun_phrase_conjunction(self):
        doc = nlp(
            "Der Hund und der Hund jagen der Hund und der Hund")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:sb; 4:sb; 7:oa; 10:oa')

    def test_ungrammatical_two_nominatives_with_noun_phrase_and_verb_conjunction(self):
        doc = nlp(
            "Der Hund und der Hund jagen und fressen der Hund und der Hund")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:sb; 4:sb; 6:cd')
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '9:sb; 12:sb')

    def test_ungrammatical_two_accusatives(self):
        doc = nlp("Den Hund jagt den Hund")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                         '1:sb; 4:oa')

    def test_ungrammatical_two_accusatives_with_noun_phrase_conjunction(self):
        doc = nlp(
            "Den Hund und den Hund jagen den Hund und den Hund")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:sb; 4:sb; 7:oa; 10:oa')

    def test_ungrammatical_two_accusatives_with_noun_phrase_and_verb_conjunction(self):
        doc = nlp(
            "Den Hund und den Hund jagen und fressen den Hund und den Hund")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:oa; 4:oa; 6:cd')
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '9:oa; 12:oa')

    def test_uncertain_subject_and_subject(self):
        doc = nlp("Ich glaube, dass eine Pflanze wächst")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '0:sb(U); 3:cp; 5:sb')

    def test_moposs_before_governing_verb(self):
        doc = nlp("Ich möchte ein Konto für mein Kind eröffnen")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '0:sb(U); 3:oa(U); 4:moposs(U); 6:pobjp(U)')

    def test_hat_vor_clause(self):
        doc = nlp("Ich habe vor, ein Konto zu eröffnen")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '0:sb(U); 5:oa; 6:pm')

    def test_simple_relative_prepositional_phrase(self):
        doc = nlp("Der Tisch, von welchem wir aßen.")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '-2:None')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:pobjp; 3:mo; 5:sb')

    def test_conjunction_relative_prepositional_phrase(self):
        doc = nlp(
            "Der Tisch und der Stuhl, von denen du und ich aßen und tranken.")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '-5:None')
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '1:pobjp(U); 4:pobjp; 6:mo; 8:sb; 9:cd; 10:sb; 12:cd')
        self.assertEqual(doc[13]._.holmes.string_representation_of_children(),
                         '1:pobjp(U); 4:pobjp; 6:mo; 8:sb; 10:sb')

    def test_conjunction_with_subject_object_and_verb_further_right(self):
        doc = nlp("Der Mann aß das Fleisch und trank.")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:sb')

    def test_conjunction_with_subject_object_modal_and_verb_further_Right(self):
        doc = nlp(
            "Der Mann hat das Fleisch gegessen und getrunken.")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '1:sb; 4:oa')

    def test_conjunction_with_prepositional_phrase_and_noun_further_right(self):
        doc = nlp(
            "Eine Versicherung für die nächsten fünf Jahre und eine Police")
        self.assertEqual(
            doc[9]._.holmes.string_representation_of_children(), '')

    def test_parent_token_indexes(self):
        doc = nlp("Häuser im Dorf.")
        self.assertEqual(doc[2]._.holmes.coreference_linked_parent_dependencies, [
                         [0, 'pobjp'], [1, 'nk']])
        self.assertEqual(doc[2]._.holmes.string_representation_of_parents(),
                         '0:pobjp; 1:nk')

    def test_von_phrase(self):
        doc = nlp("Die Verwandlung von einem Mädchen")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                         '2:pg; 4:pobjo')

    def test_subwords_without_fugen_s(self):
        doc = nlp("Telefaxnummer.")
        self.assertEqual(len(doc[0]._.holmes.subwords), 2)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Telefax')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'telefax')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'nummer')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'nummer')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 7)

    def test_subwords_with_fugen_s(self):
        doc = nlp("Widerrufsbelehrung")
        self.assertEqual(len(doc[0]._.holmes.subwords), 2)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Widerruf')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'widerruf')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'belehrung')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'belehrung')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 9)

    def test_no_subwords_without_s(self):
        doc = nlp("Lappalie")
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_no_subwords_with_s(self):
        doc = nlp("Datenschutz")
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_no_subwords_because_of_extra_letter_after_valid_subwords(self):
        doc = nlp("ZahlungsverkehrX")
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_durch_phrase_simple(self):
        doc = nlp("Die Jagd durch den Hund")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                         '2:mnr; 4:pobjb')

    def test_durch_phrase_with_conjunction(self):
        doc = nlp("Die Jagd durch den Hund und die Katze")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                         '2:mnr; 4:pobjb; 7:pobjb')

    def test_subwords_word_twice_in_document(self):
        doc = nlp(
            "Widerrufsbelehrung und die widerrufsbelehrung waren interessant")
        self.assertEqual(len(doc[0]._.holmes.subwords), 2)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Widerruf')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'widerruf')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'belehrung')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'belehrung')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 9)

        self.assertEqual(len(doc[3]._.holmes.subwords), 2)

        self.assertEqual(doc[3]._.holmes.subwords[0].text, 'widerruf')
        self.assertEqual(doc[3]._.holmes.subwords[0].lemma, 'widerruf')
        self.assertEqual(doc[3]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[3]._.holmes.subwords[0].containing_token_index, 3)
        self.assertEqual(doc[3]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[3]._.holmes.subwords[1].text, 'belehrung')
        self.assertEqual(doc[3]._.holmes.subwords[1].lemma, 'belehrung')
        self.assertEqual(doc[3]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[3]._.holmes.subwords[1].containing_token_index, 3)
        self.assertEqual(doc[3]._.holmes.subwords[1].char_start_index, 9)

    def test_three_subwords_with_non_whitelisted_fugen_s(self):

        doc = nlp("Inhaltsverzeichnisanlage")
        self.assertEqual(len(doc[0]._.holmes.subwords), 3)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Inhalt')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'inhalt')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'verzeichnis')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'verzeichnis')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 7)

        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'anlage')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'anlage')
        self.assertEqual(doc[0]._.holmes.subwords[2].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[2].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[2].char_start_index, 18)

    def test_three_subwords_with_non_whitelisted_fugen_s(self):

        doc = nlp("Inhaltsverzeichnisanlage")
        self.assertEqual(len(doc[0]._.holmes.subwords), 3)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Inhalt')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'inhalt')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'verzeichnis')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'verzeichnis')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 7)

        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'anlage')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'anlage')
        self.assertEqual(doc[0]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[2].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[2].char_start_index, 18)

    def test_four_subwords_with_whitelisted_fugen_s(self):

        doc = nlp("Finanzdienstleistungsaufsicht")
        self.assertEqual(len(doc[0]._.holmes.subwords), 4)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Finanz')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'finanz')
        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'dienst')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'dienst')
        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'leistung')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'leistung')
        self.assertEqual(doc[0]._.holmes.subwords[3].text, 'aufsicht')
        self.assertEqual(doc[0]._.holmes.subwords[3].lemma, 'aufsicht')

    def test_inflected_main_word(self):

        doc = nlp("Verbraucherstreitbeilegungsgesetzes")
        self.assertEqual(len(doc[0]._.holmes.subwords), 4)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Verbraucher')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'verbraucher')
        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'streit')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'streit')
        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'beilegung')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'beilegung')
        self.assertEqual(doc[0]._.holmes.subwords[3].text, 'gesetzes')
        self.assertEqual(doc[0]._.holmes.subwords[3].lemma, 'gesetz')

    def test_inflected_subword_other_than_fugen_s(self):

        doc = nlp("Bundesoberbehörde")
        self.assertEqual(len(doc[0]._.holmes.subwords), 2)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Bundes')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'bund')
        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'oberbehörde')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'oberbehörde')

    def test_initial_short_word(self):

        doc = nlp("Vorversicherung")
        self.assertEqual(len(doc[0]._.holmes.subwords), 2)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Vor')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'vor')
        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'versicherung')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'versicherung')

    def test_subwords_score_too_high(self):

        doc = nlp("Requalifizierung")
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_final_blacklisted_subword(self):

        doc = nlp("Gemütlichkeit")
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_subword_rejected_because_of_bigraphs(self):

        doc = nlp("Verantwortung")
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_nonsense_word(self):

        doc = nlp("WiderrufsbelehrungWiderrufsrechtSie")
        self.assertEqual(len(doc[0]._.holmes.subwords), 5)

        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Widerruf')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'widerruf')
        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'belehrung')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'belehrung')
        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'Widerruf')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'widerruf')
        self.assertEqual(doc[0]._.holmes.subwords[3].text, 'recht')
        self.assertEqual(doc[0]._.holmes.subwords[3].lemma, 'recht')
        self.assertEqual(doc[0]._.holmes.subwords[4].text, 'Sie')
        self.assertEqual(doc[0]._.holmes.subwords[4].lemma, 'ich')

    def test_nonsense_word_with_number(self):

        doc = nlp("Widerrufs3belehrungWiderrufsrechtSie")
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_nonsense_word_with_underscore(self):

        doc = nlp("Widerrufs_belehrungWiderrufsrechtSie")
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_negated_subword_with_caching(self):

        doc = nlp(
            "Die Nichtbeachtung der Regeln. Die Nichtbeachtung der Regeln")
        self.assertTrue(doc[1]._.holmes.is_negated)
        self.assertFalse(doc[0]._.holmes.is_negated)
        self.assertFalse(doc[2]._.holmes.is_negated)
        self.assertFalse(doc[3]._.holmes.is_negated)
        self.assertFalse(doc[4]._.holmes.is_negated)

        self.assertTrue(doc[6]._.holmes.is_negated)
        self.assertFalse(doc[5]._.holmes.is_negated)
        self.assertFalse(doc[7]._.holmes.is_negated)
        self.assertFalse(doc[8]._.holmes.is_negated)

    def test_subword_conjunction_two_words_single_subwords_first_word_hyphenated(self):

        doc = nlp("Die Haupt- und Seiteneingänge")
        self.assertEqual(doc[1]._.holmes.subwords[0].text, 'Haupt')
        self.assertEqual(doc[1]._.holmes.subwords[0].lemma, 'haupt')
        self.assertEqual(doc[1]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[1]._.holmes.subwords[0].containing_token_index, 1)
        self.assertEqual(doc[1]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[1]._.holmes.subwords[1].text, 'eingänge')
        self.assertEqual(doc[1]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[1]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[1]._.holmes.subwords[1].containing_token_index, 3)
        self.assertEqual(doc[1]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[3]._.holmes.subwords[0].text, 'Seiten')
        self.assertEqual(doc[3]._.holmes.subwords[0].lemma, 'seite')
        self.assertEqual(doc[3]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[3]._.holmes.subwords[0].containing_token_index, 3)
        self.assertEqual(doc[3]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[3]._.holmes.subwords[1].text, 'eingänge')
        self.assertEqual(doc[3]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[3]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[3]._.holmes.subwords[1].containing_token_index, 3)
        self.assertEqual(doc[3]._.holmes.subwords[1].char_start_index, 6)

    def test_caching(self):

        doc = nlp(
            "Die Haupt- und Seiteneingänge. Die Haupt- und Seiteneingänge")
        self.assertEqual(doc[6]._.holmes.subwords[0].text, 'Haupt')
        self.assertEqual(doc[6]._.holmes.subwords[0].lemma, 'haupt')
        self.assertEqual(doc[6]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[6]._.holmes.subwords[0].containing_token_index, 6)
        self.assertEqual(doc[6]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[6]._.holmes.subwords[1].text, 'eingänge')
        self.assertEqual(doc[6]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[6]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[6]._.holmes.subwords[1].containing_token_index, 8)
        self.assertEqual(doc[6]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[8]._.holmes.subwords[0].text, 'Seiten')
        self.assertEqual(doc[8]._.holmes.subwords[0].lemma, 'seite')
        self.assertEqual(doc[8]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[8]._.holmes.subwords[0].containing_token_index, 8)
        self.assertEqual(doc[8]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[8]._.holmes.subwords[1].text, 'eingänge')
        self.assertEqual(doc[8]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[8]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[8]._.holmes.subwords[1].containing_token_index, 8)
        self.assertEqual(doc[8]._.holmes.subwords[1].char_start_index, 6)

    def test_subword_conjunction_three_words_single_subwords_first_word_hyphenated(self):

        doc = nlp("Die Haupt-, Neben- und Seiteneingänge")
        self.assertEqual(doc[1]._.holmes.subwords[0].text, 'Haupt')
        self.assertEqual(doc[1]._.holmes.subwords[0].lemma, 'haupt')
        self.assertEqual(doc[1]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[1]._.holmes.subwords[0].containing_token_index, 1)
        self.assertEqual(doc[1]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[1]._.holmes.subwords[1].text, 'eingänge')
        self.assertEqual(doc[1]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[1]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[1]._.holmes.subwords[1].containing_token_index, 5)
        self.assertEqual(doc[1]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[3]._.holmes.subwords[0].text, 'Neben')
        self.assertEqual(doc[3]._.holmes.subwords[0].lemma, 'neben')
        self.assertEqual(doc[3]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[3]._.holmes.subwords[0].containing_token_index, 3)
        self.assertEqual(doc[3]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[3]._.holmes.subwords[1].text, 'eingänge')
        self.assertEqual(doc[3]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[3]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[3]._.holmes.subwords[1].containing_token_index, 5)
        self.assertEqual(doc[3]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[5]._.holmes.subwords[0].text, 'Seiten')
        self.assertEqual(doc[5]._.holmes.subwords[0].lemma, 'seite')
        self.assertEqual(doc[5]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[5]._.holmes.subwords[0].containing_token_index, 5)
        self.assertEqual(doc[5]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[5]._.holmes.subwords[1].text, 'eingänge')
        self.assertEqual(doc[5]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[5]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[5]._.holmes.subwords[1].containing_token_index, 5)
        self.assertEqual(doc[5]._.holmes.subwords[1].char_start_index, 6)

    def test_subword_conjunction_two_words_multiple_subwords_first_word_hyphenated(self):

        doc = nlp("Die Haupt- und Seiteneingangsbeschränkungen")
        self.assertEqual(doc[1]._.holmes.subwords[0].text, 'Haupt')
        self.assertEqual(doc[1]._.holmes.subwords[0].lemma, 'haupt')
        self.assertEqual(doc[1]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[1]._.holmes.subwords[0].containing_token_index, 1)
        self.assertEqual(doc[1]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[1]._.holmes.subwords[1].text, 'eingang')
        self.assertEqual(doc[1]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[1]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[1]._.holmes.subwords[1].containing_token_index, 3)
        self.assertEqual(doc[1]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[1]._.holmes.subwords[2].text, 'beschränkungen')
        self.assertEqual(doc[1]._.holmes.subwords[2].lemma, 'beschränkung')
        self.assertEqual(doc[1]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[1]._.holmes.subwords[2].containing_token_index, 3)
        self.assertEqual(doc[1]._.holmes.subwords[2].char_start_index, 14)

        self.assertEqual(doc[3]._.holmes.subwords[0].text, 'Seiten')
        self.assertEqual(doc[3]._.holmes.subwords[0].lemma, 'seite')
        self.assertEqual(doc[3]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[3]._.holmes.subwords[0].containing_token_index, 3)
        self.assertEqual(doc[3]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[3]._.holmes.subwords[1].text, 'eingang')
        self.assertEqual(doc[3]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[3]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[3]._.holmes.subwords[1].containing_token_index, 3)
        self.assertEqual(doc[3]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[1]._.holmes.subwords[2].text, 'beschränkungen')
        self.assertEqual(doc[1]._.holmes.subwords[2].lemma, 'beschränkung')
        self.assertEqual(doc[1]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[1]._.holmes.subwords[2].containing_token_index, 3)
        self.assertEqual(doc[1]._.holmes.subwords[2].char_start_index, 14)

    def test_subword_conjunction_three_words_multiple_subwords_first_word_hyphenated(self):

        doc = nlp(
            "Die Haupt-, Neben- und Seiteneingangsbeschränkungen")
        self.assertEqual(doc[1]._.holmes.subwords[0].text, 'Haupt')
        self.assertEqual(doc[1]._.holmes.subwords[0].lemma, 'haupt')
        self.assertEqual(doc[1]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[1]._.holmes.subwords[0].containing_token_index, 1)
        self.assertEqual(doc[1]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[1]._.holmes.subwords[1].text, 'eingang')
        self.assertEqual(doc[1]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[1]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[1]._.holmes.subwords[1].containing_token_index, 5)
        self.assertEqual(doc[1]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[1]._.holmes.subwords[2].text, 'beschränkungen')
        self.assertEqual(doc[1]._.holmes.subwords[2].lemma, 'beschränkung')
        self.assertEqual(doc[1]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[1]._.holmes.subwords[2].containing_token_index, 5)
        self.assertEqual(doc[1]._.holmes.subwords[2].char_start_index, 14)

        self.assertEqual(doc[3]._.holmes.subwords[0].text, 'Neben')
        self.assertEqual(doc[3]._.holmes.subwords[0].lemma, 'neben')
        self.assertEqual(doc[3]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[3]._.holmes.subwords[0].containing_token_index, 3)
        self.assertEqual(doc[3]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[3]._.holmes.subwords[1].text, 'eingang')
        self.assertEqual(doc[3]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[3]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[3]._.holmes.subwords[1].containing_token_index, 5)
        self.assertEqual(doc[3]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[3]._.holmes.subwords[2].text, 'beschränkungen')
        self.assertEqual(doc[3]._.holmes.subwords[2].lemma, 'beschränkung')
        self.assertEqual(doc[3]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[3]._.holmes.subwords[2].containing_token_index, 5)
        self.assertEqual(doc[3]._.holmes.subwords[2].char_start_index, 14)

        self.assertEqual(doc[5]._.holmes.subwords[0].text, 'Seiten')
        self.assertEqual(doc[5]._.holmes.subwords[0].lemma, 'seite')
        self.assertEqual(doc[5]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[5]._.holmes.subwords[0].containing_token_index, 5)
        self.assertEqual(doc[5]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[5]._.holmes.subwords[1].text, 'eingang')
        self.assertEqual(doc[5]._.holmes.subwords[1].lemma, 'eingang')
        self.assertEqual(doc[5]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[5]._.holmes.subwords[1].containing_token_index, 5)
        self.assertEqual(doc[5]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[5]._.holmes.subwords[2].text, 'beschränkungen')
        self.assertEqual(doc[5]._.holmes.subwords[2].lemma, 'beschränkung')
        self.assertEqual(doc[5]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[5]._.holmes.subwords[2].containing_token_index, 5)
        self.assertEqual(doc[5]._.holmes.subwords[2].char_start_index, 14)

    def test_subword_conjunction_adjectives(self):

        doc = nlp("Das Essen war vitamin- und eiweißhaltig")
        self.assertEqual(doc[3]._.holmes.subwords[0].text, 'vitamin')
        self.assertEqual(doc[3]._.holmes.subwords[0].lemma, 'vitamin')
        self.assertEqual(doc[3]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[3]._.holmes.subwords[0].containing_token_index, 3)
        self.assertEqual(doc[3]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[3]._.holmes.subwords[1].text, 'haltig')
        self.assertEqual(doc[3]._.holmes.subwords[1].lemma, 'haltig')
        self.assertEqual(doc[3]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[3]._.holmes.subwords[1].containing_token_index, 5)
        self.assertEqual(doc[3]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[5]._.holmes.subwords[0].text, 'eiweiß')
        self.assertEqual(doc[5]._.holmes.subwords[0].lemma, 'eiweiß')
        self.assertEqual(doc[5]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[5]._.holmes.subwords[0].containing_token_index, 5)
        self.assertEqual(doc[5]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[5]._.holmes.subwords[1].text, 'haltig')
        self.assertEqual(doc[5]._.holmes.subwords[1].lemma, 'haltig')
        self.assertEqual(doc[5]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[5]._.holmes.subwords[1].containing_token_index, 5)
        self.assertEqual(doc[5]._.holmes.subwords[1].char_start_index, 6)

    def test_subword_conjunction_two_words_single_subwords_last_word_hyphenated(self):

        doc = nlp("Verkehrslenkung und -überwachung")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'lenkung')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'lenkung')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 8)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'überwachung')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'überwachung')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 1)

    def test_subword_conjunction_three_words_single_subwords_last_word_hyphenated(self):

        doc = nlp("Verkehrslenkung, -überwachung und -betrachtung")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'lenkung')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'lenkung')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 8)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'überwachung')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'überwachung')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 1)

        self.assertEqual(doc[4]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[4]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[4]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[4]._.holmes.subwords[1].text, 'betrachtung')
        self.assertEqual(doc[4]._.holmes.subwords[1].lemma, 'betrachtung')
        self.assertEqual(doc[4]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[4]._.holmes.subwords[1].containing_token_index, 4)
        self.assertEqual(doc[4]._.holmes.subwords[1].char_start_index, 1)

    def test_subword_conjunction_two_words_multiple_subwords_last_word_hyphenated(self):

        doc = nlp(
            "Verkehrskontrolllenkung und -überwachungsprinzipien")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'kontroll')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'kontroll')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 8)

        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'lenkung')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'lenkung')
        self.assertEqual(doc[0]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[2].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[2].char_start_index, 16)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'kontroll')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'kontroll')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 8)

        self.assertEqual(doc[2]._.holmes.subwords[2].text, 'überwachung')
        self.assertEqual(doc[2]._.holmes.subwords[2].lemma, 'überwachung')
        self.assertEqual(doc[2]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].char_start_index, 1)

        self.assertEqual(doc[2]._.holmes.subwords[3].text, 'prinzipien')
        self.assertEqual(doc[2]._.holmes.subwords[3].lemma, 'prinzip')
        self.assertEqual(doc[2]._.holmes.subwords[3].index, 3)
        self.assertEqual(doc[2]._.holmes.subwords[3].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[3].char_start_index, 13)

    def test_subword_conjunction_three_words_multiple_subwords_last_word_hyphenated(self):

        doc = nlp(
            "Verkehrskontrolllenkung, -überwachungsprinzipien und -betrachtung")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'kontroll')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'kontroll')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 8)

        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'lenkung')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'lenkung')
        self.assertEqual(doc[0]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[2].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[2].char_start_index, 16)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'kontroll')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'kontroll')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 8)

        self.assertEqual(doc[2]._.holmes.subwords[2].text, 'überwachung')
        self.assertEqual(doc[2]._.holmes.subwords[2].lemma, 'überwachung')
        self.assertEqual(doc[2]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].char_start_index, 1)

        self.assertEqual(doc[2]._.holmes.subwords[3].text, 'prinzipien')
        self.assertEqual(doc[2]._.holmes.subwords[3].lemma, 'prinzip')
        self.assertEqual(doc[2]._.holmes.subwords[3].index, 3)
        self.assertEqual(doc[2]._.holmes.subwords[3].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[3].char_start_index, 13)

        self.assertEqual(doc[4]._.holmes.subwords[0].text, 'Verkehr')
        self.assertEqual(doc[4]._.holmes.subwords[0].lemma, 'verkehr')
        self.assertEqual(doc[4]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[4]._.holmes.subwords[1].text, 'kontroll')
        self.assertEqual(doc[4]._.holmes.subwords[1].lemma, 'kontroll')
        self.assertEqual(doc[4]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[4]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[1].char_start_index, 8)

        self.assertEqual(doc[4]._.holmes.subwords[2].text, 'betrachtung')
        self.assertEqual(doc[4]._.holmes.subwords[2].lemma, 'betrachtung')
        self.assertEqual(doc[4]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[4]._.holmes.subwords[2].containing_token_index, 4)
        self.assertEqual(doc[4]._.holmes.subwords[2].char_start_index, 1)

    def test_subword_conjunction_two_words_single_subwords_first_and_last_words_hyphenated(self):

        doc = nlp("Textilgroß- und -einzelhandel")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Textil')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'textil')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'groß')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'groß')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'handel')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'handel')
        self.assertEqual(doc[0]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[2].containing_token_index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[2].char_start_index, 7)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Textil')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'textil')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'einzel')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'einzel')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 1)

        self.assertEqual(doc[2]._.holmes.subwords[2].text, 'handel')
        self.assertEqual(doc[2]._.holmes.subwords[2].lemma, 'handel')
        self.assertEqual(doc[2]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].char_start_index, 7)

    def test_subword_conjunction_two_words_multiple_subwords_first_and_last_words_hyphenated(self):

        doc = nlp("Feintextilgroß- und -einzeldetailhandel")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Fein')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'fein')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'textil')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'textil')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 4)

        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'groß')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'groß')
        self.assertEqual(doc[0]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[2].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[2].char_start_index, 10)

        self.assertEqual(doc[0]._.holmes.subwords[3].text, 'detail')
        self.assertEqual(doc[0]._.holmes.subwords[3].lemma, 'detail')
        self.assertEqual(doc[0]._.holmes.subwords[3].index, 3)
        self.assertEqual(doc[0]._.holmes.subwords[3].containing_token_index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[3].char_start_index, 7)

        self.assertEqual(doc[0]._.holmes.subwords[4].text, 'handel')
        self.assertEqual(doc[0]._.holmes.subwords[4].lemma, 'handel')
        self.assertEqual(doc[0]._.holmes.subwords[4].index, 4)
        self.assertEqual(doc[0]._.holmes.subwords[4].containing_token_index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[4].char_start_index, 13)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Fein')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'fein')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'textil')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'textil')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 4)

        self.assertEqual(doc[2]._.holmes.subwords[2].text, 'einzel')
        self.assertEqual(doc[2]._.holmes.subwords[2].lemma, 'einzel')
        self.assertEqual(doc[2]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].char_start_index, 1)

        self.assertEqual(doc[2]._.holmes.subwords[3].text, 'detail')
        self.assertEqual(doc[2]._.holmes.subwords[3].lemma, 'detail')
        self.assertEqual(doc[2]._.holmes.subwords[3].index, 3)
        self.assertEqual(doc[2]._.holmes.subwords[3].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[3].char_start_index, 7)

        self.assertEqual(doc[2]._.holmes.subwords[4].text, 'handel')
        self.assertEqual(doc[2]._.holmes.subwords[4].lemma, 'handel')
        self.assertEqual(doc[2]._.holmes.subwords[4].index, 4)
        self.assertEqual(doc[2]._.holmes.subwords[4].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[4].char_start_index, 13)

    def test_subword_conjunction_three_words_single_subwords_first_and_last_words_hyphenated(self):

        doc = nlp("Textilgroß-, -klein- und -einzelhandel")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Textil')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'textil')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'groß')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'groß')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 6)

        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'handel')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'handel')
        self.assertEqual(doc[0]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[2].containing_token_index, 4)
        self.assertEqual(doc[0]._.holmes.subwords[2].char_start_index, 7)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Textil')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'textil')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'klein')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'klein')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 1)

        self.assertEqual(doc[2]._.holmes.subwords[2].text, 'handel')
        self.assertEqual(doc[2]._.holmes.subwords[2].lemma, 'handel')
        self.assertEqual(doc[2]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].containing_token_index, 4)
        self.assertEqual(doc[2]._.holmes.subwords[2].char_start_index, 7)

        self.assertEqual(doc[4]._.holmes.subwords[0].text, 'Textil')
        self.assertEqual(doc[4]._.holmes.subwords[0].lemma, 'textil')
        self.assertEqual(doc[4]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[4]._.holmes.subwords[1].text, 'einzel')
        self.assertEqual(doc[4]._.holmes.subwords[1].lemma, 'einzel')
        self.assertEqual(doc[4]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[4]._.holmes.subwords[1].containing_token_index, 4)
        self.assertEqual(doc[4]._.holmes.subwords[1].char_start_index, 1)

        self.assertEqual(doc[4]._.holmes.subwords[2].text, 'handel')
        self.assertEqual(doc[4]._.holmes.subwords[2].lemma, 'handel')
        self.assertEqual(doc[4]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[4]._.holmes.subwords[2].containing_token_index, 4)
        self.assertEqual(doc[4]._.holmes.subwords[2].char_start_index, 7)

    def test_subword_conjunction_4_words_multiple_subwords_first_and_last_words_hyphenated(self):

        doc = nlp(
            "Feintextilgroß-, -klein-, -mittel- und -einzeldetailhandel")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Fein')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'fein')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].is_head, False)
        self.assertEqual(doc[0]._.holmes.subwords[0].dependent_index, None)
        self.assertEqual(doc[0]._.holmes.subwords[0].dependency_label, None)
        self.assertEqual(doc[0]._.holmes.subwords[0].governor_index, 1)
        self.assertEqual(
            doc[0]._.holmes.subwords[0].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'textil')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'textil')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 4)
        self.assertEqual(doc[0]._.holmes.subwords[1].is_head, False)
        self.assertEqual(doc[0]._.holmes.subwords[1].dependent_index, 0)
        self.assertEqual(
            doc[0]._.holmes.subwords[1].dependency_label, 'intcompound')
        self.assertEqual(doc[0]._.holmes.subwords[1].governor_index, 2)
        self.assertEqual(
            doc[0]._.holmes.subwords[1].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[0]._.holmes.subwords[2].text, 'groß')
        self.assertEqual(doc[0]._.holmes.subwords[2].lemma, 'groß')
        self.assertEqual(doc[0]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[2].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[2].char_start_index, 10)
        self.assertEqual(doc[0]._.holmes.subwords[2].is_head, False)
        self.assertEqual(doc[0]._.holmes.subwords[2].dependent_index, 1)
        self.assertEqual(
            doc[0]._.holmes.subwords[2].dependency_label, 'intcompound')
        self.assertEqual(doc[0]._.holmes.subwords[2].governor_index, 3)
        self.assertEqual(
            doc[0]._.holmes.subwords[2].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[0]._.holmes.subwords[3].text, 'detail')
        self.assertEqual(doc[0]._.holmes.subwords[3].lemma, 'detail')
        self.assertEqual(doc[0]._.holmes.subwords[3].index, 3)
        self.assertEqual(doc[0]._.holmes.subwords[3].containing_token_index, 6)
        self.assertEqual(doc[0]._.holmes.subwords[3].char_start_index, 7)
        self.assertEqual(doc[0]._.holmes.subwords[3].is_head, False)
        self.assertEqual(doc[0]._.holmes.subwords[3].dependent_index, 2)
        self.assertEqual(
            doc[0]._.holmes.subwords[3].dependency_label, 'intcompound')
        self.assertEqual(doc[0]._.holmes.subwords[3].governor_index, 4)
        self.assertEqual(
            doc[0]._.holmes.subwords[3].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[0]._.holmes.subwords[4].text, 'handel')
        self.assertEqual(doc[0]._.holmes.subwords[4].lemma, 'handel')
        self.assertEqual(doc[0]._.holmes.subwords[4].index, 4)
        self.assertEqual(doc[0]._.holmes.subwords[4].containing_token_index, 6)
        self.assertEqual(doc[0]._.holmes.subwords[4].char_start_index, 13)
        self.assertEqual(doc[0]._.holmes.subwords[4].is_head, True)
        self.assertEqual(doc[0]._.holmes.subwords[4].dependent_index, 3)
        self.assertEqual(
            doc[0]._.holmes.subwords[4].dependency_label, 'intcompound')
        self.assertEqual(doc[0]._.holmes.subwords[4].governor_index, None)
        self.assertEqual(
            doc[0]._.holmes.subwords[4].governing_dependency_label, None)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Fein')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'fein')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].is_head, False)
        self.assertEqual(doc[2]._.holmes.subwords[0].dependent_index, None)
        self.assertEqual(doc[2]._.holmes.subwords[0].dependency_label, None)
        self.assertEqual(doc[2]._.holmes.subwords[0].governor_index, 1)
        self.assertEqual(
            doc[2]._.holmes.subwords[0].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'textil')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'textil')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 4)
        self.assertEqual(doc[2]._.holmes.subwords[1].is_head, False)
        self.assertEqual(doc[2]._.holmes.subwords[1].dependent_index, 0)
        self.assertEqual(
            doc[2]._.holmes.subwords[1].dependency_label, 'intcompound')
        self.assertEqual(doc[2]._.holmes.subwords[1].governor_index, 2)
        self.assertEqual(
            doc[2]._.holmes.subwords[1].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[2]._.holmes.subwords[2].text, 'klein')
        self.assertEqual(doc[2]._.holmes.subwords[2].lemma, 'klein')
        self.assertEqual(doc[2]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[2].char_start_index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[2].is_head, False)
        self.assertEqual(doc[2]._.holmes.subwords[2].dependent_index, 1)
        self.assertEqual(
            doc[2]._.holmes.subwords[2].dependency_label, 'intcompound')
        self.assertEqual(doc[2]._.holmes.subwords[2].governor_index, 3)
        self.assertEqual(
            doc[2]._.holmes.subwords[2].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[2]._.holmes.subwords[3].text, 'detail')
        self.assertEqual(doc[2]._.holmes.subwords[3].lemma, 'detail')
        self.assertEqual(doc[2]._.holmes.subwords[3].index, 3)
        self.assertEqual(doc[2]._.holmes.subwords[3].containing_token_index, 6)
        self.assertEqual(doc[2]._.holmes.subwords[3].char_start_index, 7)
        self.assertEqual(doc[2]._.holmes.subwords[3].is_head, False)
        self.assertEqual(doc[2]._.holmes.subwords[3].dependent_index, 2)
        self.assertEqual(
            doc[2]._.holmes.subwords[3].dependency_label, 'intcompound')
        self.assertEqual(doc[2]._.holmes.subwords[3].governor_index, 4)
        self.assertEqual(
            doc[2]._.holmes.subwords[3].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[2]._.holmes.subwords[4].text, 'handel')
        self.assertEqual(doc[2]._.holmes.subwords[4].lemma, 'handel')
        self.assertEqual(doc[2]._.holmes.subwords[4].index, 4)
        self.assertEqual(doc[2]._.holmes.subwords[4].containing_token_index, 6)
        self.assertEqual(doc[2]._.holmes.subwords[4].char_start_index, 13)
        self.assertEqual(doc[2]._.holmes.subwords[4].is_head, True)
        self.assertEqual(doc[2]._.holmes.subwords[4].dependent_index, 3)
        self.assertEqual(
            doc[2]._.holmes.subwords[4].dependency_label, 'intcompound')
        self.assertEqual(doc[2]._.holmes.subwords[4].governor_index, None)
        self.assertEqual(
            doc[2]._.holmes.subwords[4].governing_dependency_label, None)

        self.assertEqual(doc[4]._.holmes.subwords[0].text, 'Fein')
        self.assertEqual(doc[4]._.holmes.subwords[0].lemma, 'fein')
        self.assertEqual(doc[4]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].char_start_index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].is_head, False)
        self.assertEqual(doc[4]._.holmes.subwords[0].dependent_index, None)
        self.assertEqual(doc[4]._.holmes.subwords[0].dependency_label, None)
        self.assertEqual(doc[4]._.holmes.subwords[0].governor_index, 1)
        self.assertEqual(
            doc[4]._.holmes.subwords[0].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[4]._.holmes.subwords[1].text, 'textil')
        self.assertEqual(doc[4]._.holmes.subwords[1].lemma, 'textil')
        self.assertEqual(doc[4]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[4]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[1].char_start_index, 4)
        self.assertEqual(doc[4]._.holmes.subwords[1].is_head, False)
        self.assertEqual(doc[4]._.holmes.subwords[1].dependent_index, 0)
        self.assertEqual(
            doc[4]._.holmes.subwords[1].dependency_label, 'intcompound')
        self.assertEqual(doc[4]._.holmes.subwords[1].governor_index, 2)
        self.assertEqual(
            doc[4]._.holmes.subwords[1].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[4]._.holmes.subwords[2].text, 'mittel')
        self.assertEqual(doc[4]._.holmes.subwords[2].lemma, 'mittel')
        self.assertEqual(doc[4]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[4]._.holmes.subwords[2].containing_token_index, 4)
        self.assertEqual(doc[4]._.holmes.subwords[2].char_start_index, 1)
        self.assertEqual(doc[4]._.holmes.subwords[2].is_head, False)
        self.assertEqual(doc[4]._.holmes.subwords[2].dependent_index, 1)
        self.assertEqual(
            doc[4]._.holmes.subwords[2].dependency_label, 'intcompound')
        self.assertEqual(doc[4]._.holmes.subwords[2].governor_index, 3)
        self.assertEqual(
            doc[4]._.holmes.subwords[2].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[4]._.holmes.subwords[3].text, 'detail')
        self.assertEqual(doc[4]._.holmes.subwords[3].lemma, 'detail')
        self.assertEqual(doc[4]._.holmes.subwords[3].index, 3)
        self.assertEqual(doc[4]._.holmes.subwords[3].containing_token_index, 6)
        self.assertEqual(doc[4]._.holmes.subwords[3].char_start_index, 7)
        self.assertEqual(doc[4]._.holmes.subwords[3].is_head, False)
        self.assertEqual(doc[4]._.holmes.subwords[3].dependent_index, 2)
        self.assertEqual(
            doc[4]._.holmes.subwords[3].dependency_label, 'intcompound')
        self.assertEqual(doc[4]._.holmes.subwords[3].governor_index, 4)
        self.assertEqual(
            doc[4]._.holmes.subwords[3].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[4]._.holmes.subwords[4].text, 'handel')
        self.assertEqual(doc[4]._.holmes.subwords[4].lemma, 'handel')
        self.assertEqual(doc[4]._.holmes.subwords[4].index, 4)
        self.assertEqual(doc[4]._.holmes.subwords[4].containing_token_index, 6)
        self.assertEqual(doc[4]._.holmes.subwords[4].char_start_index, 13)
        self.assertEqual(doc[4]._.holmes.subwords[4].is_head, True)
        self.assertEqual(doc[4]._.holmes.subwords[4].dependent_index, 3)
        self.assertEqual(
            doc[4]._.holmes.subwords[4].dependency_label, 'intcompound')
        self.assertEqual(doc[4]._.holmes.subwords[4].governor_index, None)
        self.assertEqual(
            doc[4]._.holmes.subwords[4].governing_dependency_label, None)

        self.assertEqual(doc[6]._.holmes.subwords[0].text, 'Fein')
        self.assertEqual(doc[6]._.holmes.subwords[0].lemma, 'fein')
        self.assertEqual(doc[6]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[6]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[6]._.holmes.subwords[0].char_start_index, 0)
        self.assertEqual(doc[6]._.holmes.subwords[0].is_head, False)
        self.assertEqual(doc[6]._.holmes.subwords[0].dependent_index, None)
        self.assertEqual(doc[6]._.holmes.subwords[0].dependency_label, None)
        self.assertEqual(doc[6]._.holmes.subwords[0].governor_index, 1)
        self.assertEqual(
            doc[6]._.holmes.subwords[0].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[6]._.holmes.subwords[1].text, 'textil')
        self.assertEqual(doc[6]._.holmes.subwords[1].lemma, 'textil')
        self.assertEqual(doc[6]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[6]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[6]._.holmes.subwords[1].char_start_index, 4)
        self.assertEqual(doc[6]._.holmes.subwords[1].is_head, False)
        self.assertEqual(doc[6]._.holmes.subwords[1].dependent_index, 0)
        self.assertEqual(
            doc[6]._.holmes.subwords[1].dependency_label, 'intcompound')
        self.assertEqual(doc[6]._.holmes.subwords[1].governor_index, 2)
        self.assertEqual(
            doc[6]._.holmes.subwords[1].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[6]._.holmes.subwords[2].text, 'einzel')
        self.assertEqual(doc[6]._.holmes.subwords[2].lemma, 'einzel')
        self.assertEqual(doc[6]._.holmes.subwords[2].index, 2)
        self.assertEqual(doc[6]._.holmes.subwords[2].containing_token_index, 6)
        self.assertEqual(doc[6]._.holmes.subwords[2].char_start_index, 1)
        self.assertEqual(doc[6]._.holmes.subwords[2].is_head, False)
        self.assertEqual(doc[6]._.holmes.subwords[2].dependent_index, 1)
        self.assertEqual(
            doc[6]._.holmes.subwords[2].dependency_label, 'intcompound')
        self.assertEqual(doc[6]._.holmes.subwords[2].governor_index, 3)
        self.assertEqual(
            doc[6]._.holmes.subwords[2].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[6]._.holmes.subwords[3].text, 'detail')
        self.assertEqual(doc[6]._.holmes.subwords[3].lemma, 'detail')
        self.assertEqual(doc[6]._.holmes.subwords[3].index, 3)
        self.assertEqual(doc[6]._.holmes.subwords[3].containing_token_index, 6)
        self.assertEqual(doc[6]._.holmes.subwords[3].char_start_index, 7)
        self.assertEqual(doc[6]._.holmes.subwords[3].is_head, False)
        self.assertEqual(doc[6]._.holmes.subwords[3].dependent_index, 2)
        self.assertEqual(
            doc[6]._.holmes.subwords[3].dependency_label, 'intcompound')
        self.assertEqual(doc[6]._.holmes.subwords[3].governor_index, 4)
        self.assertEqual(
            doc[6]._.holmes.subwords[3].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[6]._.holmes.subwords[4].text, 'handel')
        self.assertEqual(doc[6]._.holmes.subwords[4].lemma, 'handel')
        self.assertEqual(doc[6]._.holmes.subwords[4].index, 4)
        self.assertEqual(doc[6]._.holmes.subwords[4].containing_token_index, 6)
        self.assertEqual(doc[6]._.holmes.subwords[4].char_start_index, 13)
        self.assertEqual(doc[6]._.holmes.subwords[4].is_head, True)
        self.assertEqual(doc[6]._.holmes.subwords[4].dependent_index, 3)
        self.assertEqual(
            doc[6]._.holmes.subwords[4].dependency_label, 'intcompound')
        self.assertEqual(doc[6]._.holmes.subwords[4].governor_index, None)
        self.assertEqual(
            doc[6]._.holmes.subwords[4].governing_dependency_label, None)

    def test_inner_hyphens_single_word(self):

        doc = nlp("Mozart-Symphonien")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Mozart')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'mozart')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].is_head, False)
        self.assertEqual(doc[0]._.holmes.subwords[0].dependent_index, None)
        self.assertEqual(doc[0]._.holmes.subwords[0].dependency_label, None)
        self.assertEqual(doc[0]._.holmes.subwords[0].governor_index, 1)
        self.assertEqual(
            doc[0]._.holmes.subwords[0].governing_dependency_label, 'intcompound')

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'Symphonien')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'symphonie')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 7)
        self.assertEqual(doc[0]._.holmes.subwords[1].is_head, True)
        self.assertEqual(doc[0]._.holmes.subwords[1].dependent_index, 0)
        self.assertEqual(
            doc[0]._.holmes.subwords[1].dependency_label, 'intcompound')
        self.assertEqual(doc[0]._.holmes.subwords[1].governor_index, None)
        self.assertEqual(
            doc[0]._.holmes.subwords[1].governing_dependency_label, None)

    def test_inner_hyphens_single_word_fugen_s(self):

        doc = nlp("Informations-Extraktion")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Information')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'information')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'Extraktion')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'extraktion')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 13)

    def test_extraneous_final_hyphen(self):

        doc = nlp("Mozart- und Leute")
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_extraneous_initial_hyphen(self):

        doc = nlp("Mozart und -Leute")
        self.assertEqual(len(doc[2]._.holmes.subwords), 0)

    def test_hyphen_alone(self):

        doc = nlp("Mozart und - Leute")
        self.assertEqual(len(doc[2]._.holmes.subwords), 0)
        self.assertEqual(doc[2].text, '-')
        self.assertEqual(doc[2]._.holmes.lemma, '-')

    def test_inner_hyphens_last_word_hyphenated(self):

        doc = nlp("Mozart-Symphonien und -Sonaten")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Mozart')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'mozart')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'Symphonien')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'symphonie')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 7)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Mozart')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'mozart')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'Sonaten')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'sonaten')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 1)

    def test_inner_hyphens_last_word_hyphenated_fugen_s(self):

        doc = nlp("Informations-Extraktion und -beurteilung")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Information')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'information')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'Extraktion')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'extraktion')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 13)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Information')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'information')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'beurteilung')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'beurteilung')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 1)

    def test_inner_hyphens_first_word_hyphenated(self):

        doc = nlp("Mozart-, Mahler- und Wagner-Symphonien")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Mozart')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'mozart')
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'Symphonien')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'symphonie')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 4)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 7)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Mahler')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'mahler')
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'Symphonien')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'symphonie')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 4)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 7)

        self.assertEqual(doc[4]._.holmes.subwords[0].text, 'Wagner')
        self.assertEqual(doc[4]._.holmes.subwords[0].lemma, 'wagner')
        self.assertEqual(doc[4]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[4]._.holmes.subwords[0].containing_token_index, 4)
        self.assertEqual(doc[4]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[4]._.holmes.subwords[1].text, 'Symphonien')
        self.assertEqual(doc[4]._.holmes.subwords[1].lemma, 'symphonie')
        self.assertEqual(doc[4]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[4]._.holmes.subwords[1].containing_token_index, 4)
        self.assertEqual(doc[4]._.holmes.subwords[1].char_start_index, 7)

    def test_inner_hyphens_first_word_hyphenated_fugen_s(self):

        doc = nlp("Informations- und Extraktions-Beurteilung")
        self.assertEqual(doc[0]._.holmes.subwords[0].text, 'Information')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'information')
        self.assertEqual(doc[0]._.holmes.subwords[0].derived_lemma, None)
        self.assertEqual(doc[0]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].containing_token_index, 0)
        self.assertEqual(doc[0]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[0]._.holmes.subwords[1].text, 'Beurteilung')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'beurteilung')
        self.assertEqual(
            doc[0]._.holmes.subwords[1].derived_lemma, 'beurteilen')
        self.assertEqual(doc[0]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[0]._.holmes.subwords[1].containing_token_index, 2)
        self.assertEqual(doc[0]._.holmes.subwords[1].char_start_index, 12)

        self.assertEqual(doc[2]._.holmes.subwords[0].text, 'Extraktion')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'extraktion')
        self.assertEqual(doc[2]._.holmes.subwords[0].derived_lemma, None)
        self.assertEqual(doc[2]._.holmes.subwords[0].index, 0)
        self.assertEqual(doc[2]._.holmes.subwords[0].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[0].char_start_index, 0)

        self.assertEqual(doc[2]._.holmes.subwords[1].text, 'Beurteilung')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'beurteilung')
        self.assertEqual(
            doc[2]._.holmes.subwords[1].derived_lemma, 'beurteilen')
        self.assertEqual(doc[2]._.holmes.subwords[1].index, 1)
        self.assertEqual(doc[2]._.holmes.subwords[1].containing_token_index, 2)
        self.assertEqual(doc[2]._.holmes.subwords[1].char_start_index, 12)

    def test_conjunction_switched_round_with_hyphenated_subword_expression(self):

        doc = nlp(
            "Ein Informationsextraktions- und Besprechungspaket wird aufgelöst")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:oa; 3:oa')

    def test_conjunction_switched_round_with_hyphenated_subword_expression_and_relative_clause(self):

        doc = nlp(
            "Das Informationsextraktions- und Besprechungspaket, welches aufgelöst wurde")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:oa(U); 3:oa')

    def test_subword_is_abbreviation_no_error_thrown(self):

        doc = nlp("Briljanten")

    def test_derived_lemma_from_dictionary(self):
        doc = nlp("Er schießt.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'schuss')

    def test_derived_lemma_root_word_from_dictionary(self):
        doc = nlp("Der Schuss war laut.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, None)

    def test_derived_lemma_ung(self):
        doc = nlp("Eine hohe Regung.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'regen')

    def test_derived_lemma_lung(self):
        doc = nlp("Die Drosselung.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'drosseln')

    def test_derived_lemma_ierung(self):
        doc = nlp("Die Validierung.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'validation')

    def test_derived_lemma_ieren(self):
        doc = nlp("Wir validieren das.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'validation')

    def test_derived_lemma_rung(self):
        doc = nlp("Eine Behinderung.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'behindern')

    def test_derived_lemma_ung_blacklist_direct(self):
        doc = nlp("Der Nibelung.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, None)

    def test_derived_lemma_heit(self):
        doc = nlp("Die ganze Schönheit.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'schön')

    def test_derived_lemma_keit(self):
        doc = nlp("Seine Langlebigkeit.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'langlebig')

    def test_derived_lemma_chen_no_change(self):
        doc = nlp("Das Tischchen.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'tisch')

    def test_derived_lemma_lein_no_change(self):
        doc = nlp("Das Tischlein.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'tisch')

    def test_derived_lemma_chen_umlaut(self):
        doc = nlp("Das kleine Bäuchchen.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'bauch')

    def test_derived_lemma_four_letter_ending_ch(self):
        doc = nlp("Das Dach.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, None)

    def test_derived_lemma_lein_umlaut(self):
        doc = nlp("Das kleine Bäuchlein.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'bauch')

    def test_derived_lemma_chen_5_chars(self):
        doc = nlp("Das kleine Öchen.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, None)

    def test_derived_lemma_chen_4_chars(self):
        doc = nlp("Das kleine Chen.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, None)

    def test_derived_lemma_chen_no_umlaut_change(self):
        doc = nlp("Das kleine Löffelchen.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'löffel')

    def test_derived_lemma_lein_no_umlaut_change_l_ending(self):
        doc = nlp("Das kleine Löffelein.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'löffel')

    def test_derived_lemma_lein_l_ending(self):
        doc = nlp("Das kleine Schakalein.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'schakal')

    def test_derived_lemma_e(self):
        doc = nlp("Das große Auge.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'aug')

    def test_derived_lemma_e_with_preceding_vowel(self):
        doc = nlp("Die große Lappalie.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, None)

    def test_derived_lemma_e_1_char(self):
        doc = nlp("Das große E.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, None)

    def test_derived_lemma_subword_positive_case(self):
        doc = nlp("Informierensextraktion.")
        self.assertEqual(
            doc[0]._.holmes.subwords[0].derived_lemma, 'information')

    def test_derived_lemma_subword_negative_case(self):
        doc = nlp("Elefantenschau.")
        self.assertEqual(doc[0]._.holmes.subwords[0].derived_lemma, None)

    def test_derived_lemma_subword_conjunction_first_word(self):
        doc = nlp("Fitness- und Freizeitsjogging.")
        self.assertEqual(doc[0]._.holmes.subwords[1].derived_lemma, 'joggen')

    def test_derived_lemma_subword_conjunction_last_word(self):
        doc = nlp("Investitionsanfänge und -auswirkungen.")
        self.assertEqual(
            doc[0]._.holmes.subwords[0].derived_lemma, 'investieren')

    def test_derived_lemma_lung_after_consonant(self):
        doc = nlp("Verwandlung.")
        self.assertEqual(doc[0]._.holmes.derived_lemma, 'verwandeln')

    def test_derived_lemma_ierung_without_ation(self):
        doc = nlp("Bilanzierung.")
        self.assertEqual(doc[0]._.holmes.derived_lemma, 'bilanzieren')

    def test_derived_lemma_lung_after_vowel_sound(self):
        doc = nlp("Erzählung.")
        self.assertEqual(doc[0]._.holmes.derived_lemma, 'erzählen')

    def test_participle_lemma_adja(self):
        doc = nlp("Für die studierten Kinder.")
        self.assertEqual(doc[2]._.holmes.lemma, 'studieren')

    def test_participle_lemma_adjd(self):
        doc = nlp("Das Kind war studiert.")
        self.assertEqual(doc[3]._.holmes.lemma, 'studieren')

    def test_non_recorded_subword_alone(self):
        doc = nlp('Messerlein.')
        self.assertEqual(len(doc[0]._.holmes.subwords), 0)

    def test_non_recorded_subword_at_end(self):
        doc = nlp('Informationsmesserlein.')
        self.assertEqual(len(doc[0]._.holmes.subwords), 2)
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'information')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'messer')

    def test_non_recorded_subword_in_middle(self):
        doc = nlp('Messerleininformation.')
        self.assertEqual(len(doc[0]._.holmes.subwords), 2)
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'messer')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'information')

    def test_non_recorded_subword_at_beginning(self):
        doc = nlp('Leinmesserinformation.')
        self.assertEqual(len(doc[0]._.holmes.subwords), 2)
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'messer')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'information')

    def test_non_recorded_subword_as_first_member_of_compound(self):
        doc = nlp('Messerlein- und Tellerleingespräche.')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'messer')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'gespräch')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'teller')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'gespräch')

    def test_non_recorded_subword_as_second_member_of_compound(self):
        doc = nlp('Nahrungsmesserlein und -tellerlein.')
        self.assertEqual(doc[0]._.holmes.subwords[0].lemma, 'nahrung')
        self.assertEqual(doc[0]._.holmes.subwords[1].lemma, 'messer')
        self.assertEqual(doc[2]._.holmes.subwords[0].lemma, 'nahrung')
        self.assertEqual(doc[2]._.holmes.subwords[1].lemma, 'teller')
