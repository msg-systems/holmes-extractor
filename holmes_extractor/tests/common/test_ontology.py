import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
symmetric_ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')),
        symmetric_matching=True)
for term in ['horse', 'football', 'gymnastics equipment', 'dog', 'cat', 'hound', 'pussy', 'animal',
        'foal', 'fido', 'mimi momo', 'absent', 'cat creature']:
    ontology.add_to_dictionary(term)
    symmetric_ontology.add_to_dictionary(term)
symmetric_ontology.add_to_dictionary('vaulting horse')

class OntologyTest(unittest.TestCase):

    def test_multiwords(self):
        self.assertTrue(ontology.contains_multiword('gymnastics equipment'))
        self.assertTrue(ontology.contains_multiword('German Shepherd dog'))
        self.assertTrue(ontology.contains_multiword('MIMI MOMO'))
        self.assertFalse(ontology.contains_multiword('horse'))
        self.assertFalse(ontology.contains_multiword('economic development'))
        self.assertFalse(ontology.contains_multiword('Fido'))

    def test_word_does_not_match_itself(self):
        self.assertEqual(len(ontology.get_words_matching('football')), 0)
        self.assertEqual(len(ontology.get_words_matching('fido')), 0)
        self.assertEqual(len(ontology.get_words_matching('mimi momo')), 0)

    def test_word_matches_subclasses_and_synonyms(self):
        self.assertEqual(ontology.get_words_matching('dog'),
                {'german shepherd dog', 'puppy', 'hound', 'fido'})
        self.assertEqual(ontology.get_words_matching('cat'),
                {'kitten', 'pussy', 'mimi momo', 'cat creature'})
        self.assertEqual(ontology.get_words_matching('hound'),
                {'german shepherd dog', 'puppy', 'dog', 'fido'})
        self.assertEqual(ontology.get_words_matching('pussy'),
                {'kitten', 'cat', 'mimi momo', 'cat creature'})
        self.assertEqual(ontology.get_words_matching('cat creature'),
                {'kitten', 'cat', 'mimi momo', 'pussy'})

    def test_matching_normal_term(self):
        entry = ontology.matches('animal', 'foal')
        self.assertEqual(entry.depth, 2)
        self.assertEqual(entry.is_individual, False)
        self.assertEqual(ontology.matches('foal', 'animal'), None)

    def test_matching_individual_term(self):
        entry = ontology.matches('animal', 'mimi momo')
        self.assertEqual(entry.depth, 2)
        self.assertEqual(entry.is_individual, True)
        self.assertEqual(ontology.matches('mimi momo', 'animal'), None)

    def test_hononym_behaviour(self):
        self.assertEqual(ontology.get_words_matching('horse'), {'vaulting horse', 'foal'})
        self.assertEqual(ontology.get_words_matching('gymnastics equipment'),
                {'horse', 'vaulting horse'})
        self.assertEqual(ontology.get_words_matching('animal'),
                {'dog', 'cat', 'horse', 'german shepherd dog', 'puppy', 'hound', 'kitten', 'pussy',
                'foal', 'fido', 'mimi momo', 'cat creature'})
        self.assertEqual(ontology.matches('animal', 'vaulting horse'), None)

    def test_multiwords_symmetric(self):
        self.assertTrue(symmetric_ontology.contains_multiword('gymnastics equipment'))
        self.assertTrue(symmetric_ontology.contains_multiword('German Shepherd dog'))
        self.assertTrue(symmetric_ontology.contains_multiword('MIMI MOMO'))
        self.assertFalse(symmetric_ontology.contains_multiword('horse'))
        self.assertFalse(symmetric_ontology.contains_multiword('economic development'))
        self.assertFalse(symmetric_ontology.contains_multiword('Fido'))

    def test_word_does_not_match_itself_symmetric(self):
        self.assertEqual(len(symmetric_ontology.get_words_matching('football')), 0)

    def test_word_matches_subclasses_synonyms_and_superclasses_symmetric(self):
        self.assertEqual(symmetric_ontology.get_words_matching('dog'),
                {'german shepherd dog', 'puppy', 'hound', 'fido', 'animal'})
        self.assertEqual(symmetric_ontology.get_words_matching('cat'),
                {'kitten', 'pussy', 'mimi momo', 'cat creature', 'animal'})
        self.assertEqual(symmetric_ontology.get_words_matching('hound'),
                {'german shepherd dog', 'puppy', 'dog', 'fido', 'animal'})
        self.assertEqual(symmetric_ontology.get_words_matching('pussy'),
                {'kitten', 'cat', 'mimi momo', 'cat creature', 'animal'})
        self.assertEqual(symmetric_ontology.get_words_matching('cat creature'),
                {'kitten', 'cat', 'mimi momo', 'pussy', 'animal'})
        self.assertEqual(symmetric_ontology.get_words_matching('mimi momo'),
                {'cat', 'cat creature', 'pussy', 'animal'})

    def test_matching_normal_term_symmetric(self):
        entry = symmetric_ontology.matches('animal', 'foal')
        self.assertEqual(entry.depth, 2)
        self.assertEqual(entry.is_individual, False)
        entry = symmetric_ontology.matches('foal', 'animal')
        self.assertEqual(entry.depth, -2)
        self.assertEqual(entry.is_individual, False)

    def test_matching_individual_term_symmetric(self):
        entry = symmetric_ontology.matches('animal', 'mimi momo')
        self.assertEqual(entry.depth, 2)
        self.assertEqual(entry.is_individual, True)
        entry = symmetric_ontology.matches('mimi momo', 'animal')
        self.assertEqual(entry.depth, -2)
        self.assertEqual(entry.is_individual, False)

    def test_homonym_behaviour_symmetric(self):
        self.assertEqual(symmetric_ontology.get_words_matching('horse'), {
                'vaulting horse', 'foal', 'animal', 'school gear', 'gymnastics equipment'})
        self.assertEqual(symmetric_ontology.get_words_matching('gymnastics equipment'),
                {'horse', 'vaulting horse'})
        self.assertEqual(symmetric_ontology.get_words_matching('animal'),
                {'dog', 'cat', 'horse', 'german shepherd dog', 'puppy', 'hound', 'kitten', 'pussy',
                'foal', 'fido', 'mimi momo', 'cat creature'})
        self.assertEqual(symmetric_ontology.matches('animal', 'vaulting horse'), None)
        self.assertEqual(symmetric_ontology.matches('vaulting horse', 'animal'), None)

    def test_most_general_hypernym_ancestor_good_case_class(self):
        self.assertEqual(ontology.get_most_general_hypernym_ancestor('cat'), 'animal')

    def test_most_general_hypernym_ancestor_good_case_multiword_class(self):
        self.assertEqual(ontology.get_most_general_hypernym_ancestor('cat creature'), 'animal')

    def test_most_general_hypernym_ancestor_good_case_homonym_class(self):
        result_set = set()
        for counter in range(1,20):
            working_ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
            result_set.add(working_ontology.get_most_general_hypernym_ancestor('horse'))
        self.assertEqual(result_set, {'animal'})

    def test_most_general_hypernym_ancestor_good_case_individual(self):
        self.assertEqual(ontology.get_most_general_hypernym_ancestor('Fido'), 'animal')

    def test_most_general_hypernym_ancestor_good_case_multiword_individual(self):
        self.assertEqual(ontology.get_most_general_hypernym_ancestor('Mimi Momo'), 'animal')

    def test_most_general_hypernym_ancestor_no_ancestor(self):
        self.assertEqual(ontology.get_most_general_hypernym_ancestor('animal'), 'animal')

    def test_most_general_hypernym_ancestor_not_in_ontology(self):
        self.assertEqual(ontology.get_most_general_hypernym_ancestor('toolbox'), 'toolbox')

    def test_most_general_hypernym_ancestor_good_case_class_symmetric(self):
        self.assertEqual(symmetric_ontology.get_most_general_hypernym_ancestor('cat'), 'animal')

    def test_most_general_hypernym_ancestor_good_case_multiword_class_symmetric(self):
        self.assertEqual(symmetric_ontology.get_most_general_hypernym_ancestor('cat creature'),
                'animal')

    def test_most_general_hypernym_ancestor_good_case_homonym_class_symmetric(self):
        result_set = set()
        for counter in range(1,20):
            working_ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')),
                    symmetric_matching=True)
            result_set.add(working_ontology.get_most_general_hypernym_ancestor('horse'))
        self.assertEqual(result_set, {'animal'})

    def test_most_general_hypernym_ancestor_good_case_individual_symmetric(self):
        self.assertEqual(symmetric_ontology.get_most_general_hypernym_ancestor('Fido'), 'animal')

    def test_most_general_hypernym_ancestor_good_case_multiword_individual_symmetric(self):
        self.assertEqual(symmetric_ontology.get_most_general_hypernym_ancestor('Mimi Momo'),
                'animal')

    def test_most_general_hypernym_ancestor_no_ancestor_symmetric(self):
        self.assertEqual(symmetric_ontology.get_most_general_hypernym_ancestor('animal'), 'animal')

    def test_most_general_hypernym_ancestor_not_in_ontology_symmetric(self):
        self.assertEqual(symmetric_ontology.get_most_general_hypernym_ancestor('toolbox'),
                'toolbox')
