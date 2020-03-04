import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
symmetric_ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')),
        symmetric_matching=True)
combined_ontology_1 = holmes.Ontology([os.sep.join((script_directory,'test_ontology.owl')),
        os.sep.join((script_directory,'test_ontology2.owl'))])
combined_ontology_2 = holmes.Ontology([os.sep.join((script_directory,'test_ontology2.owl')),
        os.sep.join((script_directory,'test_ontology.owl'))])
combined_ontology_symmetric = holmes.Ontology([os.sep.join((script_directory,'test_ontology.owl')),
        os.sep.join((script_directory,'test_ontology2.owl'))], symmetric_matching=True)
for term in ['horse', 'football', 'gymnastics equipment', 'dog', 'cat', 'hound', 'pussy', 'animal',
        'foal', 'fido', 'mimi momo', 'absent', 'cat creature']:
    ontology.add_to_dictionary(term)
    symmetric_ontology.add_to_dictionary(term)
    combined_ontology_1.add_to_dictionary(term)
    combined_ontology_2.add_to_dictionary(term)
    combined_ontology_symmetric.add_to_dictionary(term)
symmetric_ontology.add_to_dictionary('vaulting horse')
combined_ontology_1.add_to_dictionary('poodle')
combined_ontology_2.add_to_dictionary('poodle')
combined_ontology_symmetric.add_to_dictionary('poodle')
combined_ontology_1.add_to_dictionary('Schneeglöckchen')
combined_ontology_2.add_to_dictionary('Schneeglöckchen')
combined_ontology_symmetric.add_to_dictionary('Schneeglöckchen')

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

    def _test_combined_ontologies_nonsymmetric_class(self, ontology):
        self.assertEqual(ontology.get_words_matching('dog'),
                {'german shepherd dog', 'puppy', 'hound', 'fido', 'poodle'})
        self.assertEqual(len(ontology.get_words_matching('poodle')), 0)
        self.assertEqual(ontology.get_most_general_hypernym_ancestor('poodle'), 'animal')
        entry = ontology.matches('animal', 'poodle')
        self.assertEqual(entry.depth, 2)
        self.assertFalse(entry.is_individual)
        entry = ontology.matches('poodle', 'animal')
        self.assertEqual(entry, None)

    def test_combined_ontologies_nonsymmetric_class_1(self):
        self._test_combined_ontologies_nonsymmetric_class(combined_ontology_1)

    def test_combined_ontologies_nonsymmetric_class_2(self):
        self._test_combined_ontologies_nonsymmetric_class(combined_ontology_2)

    def _test_combined_ontologies_nonsymmetric_individual(self, ontology):
        self.assertEqual(ontology.get_words_matching('cat'),
                {'kitten', 'pussy', 'mimi momo', 'cat creature', 'schneeglöckchen'})
        self.assertEqual(len(ontology.get_words_matching('schneeglöckchen')), 0)
        self.assertEqual(ontology.get_most_general_hypernym_ancestor('schneeglöckchen'), 'animal')
        entry = ontology.matches('animal', 'schneeglöckchen')
        self.assertEqual(entry.depth, 2)
        self.assertTrue(entry.is_individual)
        entry = ontology.matches('schneeglöckchen', 'animal')
        self.assertEqual(entry, None)

    def test_combined_ontologies_nonsymmetric_individual_1(self):
        self._test_combined_ontologies_nonsymmetric_individual(combined_ontology_1)

    def test_combined_ontologies_nonsymmetric_individual_2(self):
        self._test_combined_ontologies_nonsymmetric_individual(combined_ontology_2)

    def test_combined_ontologies_symmetric_class(self):
        self.assertEqual(combined_ontology_symmetric.get_words_matching('dog'),
                {'german shepherd dog', 'puppy', 'hound', 'fido', 'poodle', 'animal'})
        self.assertEqual(combined_ontology_symmetric.get_words_matching('poodle'),
                {'dog', 'hound', 'animal'})
        self.assertEqual(combined_ontology_symmetric.get_most_general_hypernym_ancestor('poodle'), 'animal')
        entry = combined_ontology_symmetric.matches('animal', 'poodle')
        self.assertEqual(entry.depth, 2)
        self.assertFalse(entry.is_individual)
        entry = combined_ontology_symmetric.matches('poodle', 'animal')
        self.assertEqual(entry.depth, -2)
        self.assertFalse(entry.is_individual)

    def test_combined_ontologies_symmetric_individual(self):
        self.assertEqual(combined_ontology_symmetric.get_words_matching('cat'),
                {'kitten', 'mimi momo', 'cat creature', 'schneeglöckchen', 'animal', 'pussy'})
        self.assertEqual(combined_ontology_symmetric.get_words_matching('schneeglöckchen'),
                {'cat', 'cat creature', 'pussy', 'animal'})
        self.assertEqual(combined_ontology_symmetric.get_most_general_hypernym_ancestor(
                'schneeglöckchen'), 'animal')
        entry = combined_ontology_symmetric.matches('animal', 'schneeglöckchen')
        self.assertEqual(entry.depth, 2)
        self.assertTrue(entry.is_individual)
        entry = combined_ontology_symmetric.matches('schneeglöckchen', 'animal')
        self.assertEqual(entry.depth, -2)
        self.assertFalse(entry.is_individual)
