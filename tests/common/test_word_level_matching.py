import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
holmes_manager_coref = holmes.Manager(model='en_core_web_trf', overall_similarity_threshold=0.82,
        embedding_based_matching_on_root_words=True, ontology=ontology,
        perform_coreference_resolution=False)
holmes_manager_coref.register_search_phrase('A dog chases a cat')
holmes_manager_coref.register_search_phrase('An ENTITYPERSON chases a horse')
holmes_manager_coref.register_search_phrase('A king wakes up')
holmes_manager_coref.register_search_phrase('A cat creature jumps')
holmes_manager_coref.register_search_phrase('cat creature')
holmes_manager_coref.register_search_phrase('An industrious king loved by all.')
holmes_manager_coref.register_search_phrase('A narcissistic king')
holmes_manager_coref.register_search_phrase('A splendid king')
holmes_manager_coref.register_search_phrase('A kind king')
holmes_manager_coref.register_search_phrase("An ENTITYGPE")
holmes_manager_coref.register_search_phrase("Somebody believes strongly")
holmes_manager_coref.register_search_phrase("A strong attraction")
symmetric_ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')),
        symmetric_matching=True)
second_holmes_manager_coref = holmes.Manager(model='en_core_web_trf', overall_similarity_threshold=0.82,
        embedding_based_matching_on_root_words=False, ontology=symmetric_ontology,
        perform_coreference_resolution=False)
second_holmes_manager_coref.register_search_phrase('A narcissistic king')
second_holmes_manager_coref.register_search_phrase('A king wakes up')
second_holmes_manager_coref.register_search_phrase('A kitten goes to bed')
second_holmes_manager_coref.register_search_phrase('Mimi Momo goes to bed')
second_holmes_manager_coref.register_search_phrase('A dog goes to bed')
second_holmes_manager_coref.register_search_phrase('unouno')
second_holmes_manager_coref.register_search_phrase('sześć')

class WordMatchingTest(unittest.TestCase):

    def test_direct_matching(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry='The dog chased the cat')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['search_phrase'], 'A dog chases a cat')
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'direct')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Matches DOG directly.")
        self.assertEqual(text_matches[0]['word_matches'][1]['match_type'], 'direct')
        self.assertEqual(text_matches[0]['word_matches'][2]['match_type'], 'direct')

    def test_entity_matching(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry='Richard Hudson chased the horse')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'entity')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Matches the ENTITYPERSON placeholder.")

    def test_ontology_matching(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry='The dog chased the kitten')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['word_matches'][2]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][2]['explanation'],
                "Is a child of CAT in the ontology.")

    def test_embedding_matching(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry='The queen woke up')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'embedding')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Has a word embedding that is 72% similar to KING.")
        self.assertEqual(text_matches[0]['word_matches'][1]['explanation'],
                "Matches WAKE UP directly.")

    def test_embedding_matching_on_root_node(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry='An industrious queen loved by all')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][1]['match_type'], 'embedding')

    def test_embedding_matching_on_root_node_with_multiple_templates(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document('A narcissistic queen',
                label='narcissistic queen')
        holmes_manager_coref.parse_and_register_document('A splendid queen', label='splendid queen')
        holmes_manager_coref.parse_and_register_document('A kind queen', label='kind queen')
        holmes_manager_coref.parse_and_register_document('A narcissistic toolbox',
                label='narcissistic toolbox')
        holmes_manager_coref.parse_and_register_document('A splendid toolbox', label='splendid toolbox')
        holmes_manager_coref.parse_and_register_document('A kind toolbox', label='kind toolbox')
        text_matches = holmes_manager_coref.match_returning_dictionaries()
        self.assertEqual(len(text_matches), 3)
        for text_match in text_matches:
            self.assertTrue(text_match['document'].endswith('queen'))

    def test_multiword_matching_multiword_in_document(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry='Fido chased Mimi Momo')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['word_matches'][2]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][2]['document_word'], 'Mimi Momo')

    def test_multiword_matching_multiword_in_search_phrase(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry='The cat jumped')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['document_word'], 'cat')
        self.assertEqual(text_matches[0]['word_matches'][0]['search_phrase_word'], 'cat creature')
        self.assertEqual(text_matches[1]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[1]['word_matches'][0]['document_word'], 'cat')
        self.assertEqual(text_matches[1]['word_matches'][0]['search_phrase_word'], 'cat creature')

    def test_multiword_matching_multiword_in_document_and_search_phrase(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry='Mimi Momo jumped')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['document_word'], 'Mimi Momo')
        self.assertEqual(text_matches[0]['word_matches'][0]['search_phrase_word'], 'cat creature')
        self.assertEqual(text_matches[1]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[1]['word_matches'][0]['document_word'], 'Mimi Momo')
        self.assertEqual(text_matches[1]['word_matches'][0]['search_phrase_word'], 'cat creature')

    def test_search_phrase_with_entity_root_single_word(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry=
                'Mallorca is a large municipality.')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'entity')
        self.assertEqual(text_matches[0]['word_matches'][0]['document_word'], 'Mallorca')

    def test_search_phrase_with_entity_root_multiword(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry=
                'New York is a large municipality.')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'entity')
        self.assertEqual(text_matches[0]['word_matches'][0]['document_word'], 'New York')

    def test_ontology_multiword_matches_exactly(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(entry='a cat creature')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['word_matches'][0]['document_word'], 'cat')
        self.assertEqual(text_matches[1]['word_matches'][0]['document_word'], 'cat creature')

    def test_embedding_matching_on_root_node_when_inactive(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='A narcissistic queen')
        self.assertEqual(len(text_matches), 0)

    def test_embedding_matching_when_embedding_root_node_inactive(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(entry='The queen woke up')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'embedding')

    def test_symmetric_ontology_single_word_match(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='an animal goes to bed')
        self.assertEqual(len(text_matches), 3)
        self.assertEqual(text_matches[0]['search_phrase'], 'A kitten goes to bed')
        self.assertEqual(text_matches[1]['search_phrase'], 'Mimi Momo goes to bed')
        self.assertEqual(text_matches[2]['search_phrase'], 'A dog goes to bed')

    def test_symmetric_ontology_multiword_word_match(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='a cat creature goes to bed')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['search_phrase'], 'A kitten goes to bed')
        self.assertEqual(text_matches[1]['search_phrase'], 'Mimi Momo goes to bed')

    def test_symmetric_ontology_same_word_match_on_normal_word(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='a kitten goes to bed')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['search_phrase'], 'A kitten goes to bed')
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'direct')
        self.assertEqual(text_matches[1]['search_phrase'], 'A dog goes to bed')
        self.assertEqual(text_matches[1]['word_matches'][0]['match_type'], 'embedding')

    def test_symmetric_ontology_same_word_match_on_individual(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='Mimi Momo goes to bed')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['search_phrase'], 'Mimi Momo goes to bed')

    def test_symmetric_ontology_hyponym_match_on_normal_word(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='A puppy goes to bed')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['search_phrase'], 'A dog goes to bed')
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[1]['search_phrase'], 'A kitten goes to bed')
        self.assertEqual(text_matches[1]['word_matches'][0]['match_type'], 'embedding')

    def test_symmetric_ontology_hyponym_match_on_individual(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='Fido goes to bed')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['search_phrase'], 'A dog goes to bed')

    def test_index_within_document(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(
                entry='Last week a dog chased a cat')
        self.assertEqual(len(text_matches), 2)
        self.assertEqual(text_matches[0]['index_within_document'], 4)
        self.assertEqual(text_matches[1]['index_within_document'], 6)

    def test_derivation_matching_1(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(
                entry='A strong belief')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'derivation')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Has a common stem with BELIEVE.")
        self.assertEqual(text_matches[0]['word_matches'][1]['match_type'], 'derivation')

    def test_derivation_matching_2(self):
        text_matches = holmes_manager_coref.match_search_phrases_against(
                entry='Someone is strongly attracted')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'derivation')
        self.assertEqual(text_matches[0]['word_matches'][1]['match_type'], 'derivation')

    def test_ontology_matching_depth_0(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='oans')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is a synonym of UNOUNO in the ontology.")

    def test_ontology_matching_depth_1(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='dos')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is a child of UNOUNO in the ontology.")

    def test_ontology_matching_depth_2(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='tres')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is a grandchild of UNOUNO in the ontology.")

    def test_ontology_matching_depth_3(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='cuatro')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is a great-grandchild of UNOUNO in the ontology.")

    def test_ontology_matching_depth_4(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='cinco')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is a descendant of UNOUNO in the ontology.")

    def test_ontology_matching_depth_5(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='seis')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is a descendant of UNOUNO in the ontology.")

    def test_ontology_matching_depth_minus_1(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='pięć')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is a parent of SZEŚĆ in the ontology.")

    def test_ontology_matching_depth_minus_2(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='cztery')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is a grandparent of SZEŚĆ in the ontology.")

    def test_ontology_matching_depth_minus_3(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='trzy')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is a great-grandparent of SZEŚĆ in the ontology.")

    def test_ontology_matching_depth_minus_4(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='dwa')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is an ancestor of SZEŚĆ in the ontology.")

    def test_ontology_matching_depth_minus_5(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='jeden')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is an ancestor of SZEŚĆ in the ontology.")

    def test_ontology_matching_depth_minus_5(self):
        text_matches = second_holmes_manager_coref.match_search_phrases_against(
                entry='jeden')
        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0]['word_matches'][0]['match_type'], 'ontology')
        self.assertEqual(text_matches[0]['word_matches'][0]['explanation'],
                "Is an ancestor of SZEŚĆ in the ontology.")

    def test_get_word_match_with_search_phrase_word_index(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document('The dog chased the cat')
        matches = holmes_manager_coref.match()
        self.assertEqual(matches[0].get_word_match_with_search_phrase_word_index(0), None)
        self.assertEqual(matches[0].get_word_match_with_search_phrase_word_index(3), None)
        self.assertEqual(matches[0].get_word_match_with_search_phrase_word_index(1),
            matches[0].word_matches[0])
        self.assertEqual(matches[0].get_word_match_with_search_phrase_word_index(2),
            matches[0].word_matches[1])
        self.assertEqual(matches[0].get_word_match_with_search_phrase_word_index(4),
            matches[0].word_matches[2])
