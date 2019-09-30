import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
ontology_holmes_manager = holmes.Manager(model='en_core_web_lg',
        perform_coreference_resolution=False, ontology=ontology)
symmetric_ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')),
        symmetric_matching = True)
symmetric_ontology_nocoref_holmes_manager = holmes.Manager(model='en_core_web_lg',
        ontology=symmetric_ontology, perform_coreference_resolution=False)
no_ontology_coref_holmes_manager = holmes.Manager(model='en_core_web_lg',
        perform_coreference_resolution=True)

class EnglishPhraseletProductionTest(unittest.TestCase):

    def _check_equals(self, manager, text_to_match, phraselet_labels,
            replace_with_hypernym_ancestors = True, match_all_words = False,
            include_reverse_only = False):
        manager.remove_all_search_phrases()
        doc = manager.semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_search_phrases = {}
        manager.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                replace_with_hypernym_ancestors=replace_with_hypernym_ancestors,
                match_all_words=match_all_words,
                returning_serialized_phraselets=False,
                ignore_relation_phraselets=False,
                include_reverse_only=include_reverse_only)
        self.assertEqual(
                set(phraselet_labels_to_search_phrases.keys()),
                set(phraselet_labels))
        self.assertEqual(len(phraselet_labels_to_search_phrases.keys()),
                len(phraselet_labels))

    def test_verb_subject_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A plant grows",
                ['predicate-actor: grow-plant', 'word: plant'])

    def test_phrasal_verb_subject_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A plant grows up quickly",
                ['governor-adjective: grow up-quickly', 'predicate-actor: grow up-plant',
                        'word: plant'])

    def test_verb_direct_object_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A plant is grown",
                ['predicate-patient: grow-plant', 'word: plant'])

    def test_verb_indirect_object_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "Somebody gives something to a plant",
                ['predicate-recipient: give-plant', 'word: plant'])

    def test_noun_adjective_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A healthy plant",
                ['governor-adjective: plant-healthy', 'word: plant'])

    def test_verb_adverb_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "They sailed rapidly",
                ['governor-adjective: sail-rapidly'])

    def test_noun_noun_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A hobby plant",
                ['noun-noun: plant-hobby', 'word: plant', 'word: hobby'])

    def test_possessor_possessed_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A gardener's plant",
                ['possessor-possessed: plant-gardener', 'word: plant', 'word: gardener'])

    def test_combined_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager,
                "A gardener's healthy hobby plant grows in the sun",
                ['predicate-actor: grow-plant', 'governor-adjective: plant-healthy',
                'noun-noun: plant-hobby', 'possessor-possessed: plant-gardener',
                'prepgovernor-noun: grow-sun', 'word: plant', 'word: hobby', 'word: gardener',
                'word: sun'])

    def test_class_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A dog progresses",
                ['predicate-actor: progress-animal', 'word: animal'])

    def test_multiword_class_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A big cat creature",
                ['governor-adjective: animal-big', 'word: animal'])

    def test_individual_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "Fido progresses",
                ['predicate-actor: progress-animal', 'word: animal'])

    def test_multiword_individual_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "Mimi Momo progresses",
                ['predicate-actor: progress-animal', 'word: animal'])

    def test_class_entry_in_ontology_no_hypernym_replacement(self):
        self._check_equals(ontology_holmes_manager, "A dog progresses",
                ['predicate-actor: progress-dog', 'word: dog'], False)

    def test_multiword_class_entry_in_ontology_no_hypernym_replacement(self):
        self._check_equals(ontology_holmes_manager, "A big cat creature",
                ['governor-adjective: cat creature-big', 'word: cat creature'], False)

    def test_individual_entry_in_ontology_no_hypernym_replacement(self):
        self._check_equals(ontology_holmes_manager, "Fido progresses",
                ['predicate-actor: progress-fido', 'word: fido'], False)

    def test_multiword_individual_entry_in_ontology_no_hypernym_replacement(self):
        self._check_equals(ontology_holmes_manager, "Mimi Momo progresses",
                ['predicate-actor: progress-mimi momo', 'word: mimi momo'], False)

    def test_multiword_in_ontology_no_hypernym(self):
        self._check_equals(ontology_holmes_manager, "School gear progresses",
                ['predicate-actor: progress-school gear', 'word: school gear'])

    def test_multiword_not_in_ontology(self):
        self._check_equals(ontology_holmes_manager,
                "Information extraction progresses with information",
                ['predicate-actor: progress-extraction', 'noun-noun: extraction-information',
                'prepgovernor-noun: progress-information', 'word: information', 'word: extraction'])

    def test_text_in_ontology_lemma_not_in_ontology(self):
        self._check_equals(ontology_holmes_manager,
                "He saw rainbows",
                ['predicate-patient: see-arc', 'word: arc'])

    def test_text_in_ontology_lemma_not_in_ontology_no_hypernym_replacement(self):
        self._check_equals(ontology_holmes_manager,
                "He saw rainbows",
                ['predicate-patient: see-rainbows', 'word: rainbows'], False)

    def test_class_entry_in_ontology_symmetric_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager, "A dog progresses",
                ['predicate-actor: progress-animal', 'word: animal'])

    def test_multiword_class_entry_in_ontology_symmetric_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager, "A big cat creature",
                ['governor-adjective: animal-big', 'word: animal'])

    def test_individual_entry_in_ontology_symmetric_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager, "Fido progresses",
                ['predicate-actor: progress-animal', 'word: animal'])

    def test_multiword_individual_entry_in_ontology_symmetric_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager, "Mimi Momo progresses",
                ['predicate-actor: progress-animal', 'word: animal'])

    def test_class_entry_in_ontology_no_hypernym_replacement_symmetric_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager, "A dog progresses",
                ['predicate-actor: progress-dog', 'word: dog'], False)

    def test_multiword_class_entry_in_ontology_no_hypernym_replacement_symmetric_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager, "A big cat creature",
                ['governor-adjective: cat creature-big', 'word: cat creature'], False)

    def test_individual_entry_in_ontology_no_hypernym_replacement_symmetric_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager, "Fido progresses",
                ['predicate-actor: progress-fido', 'word: fido'], False)

    def test_multiword_individual_entry_in_ontology_no_hypernym_replacement_symm_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager, "Mimi Momo progresses",
                ['predicate-actor: progress-mimi momo', 'word: mimi momo'], False)

    def test_multiword_not_in_ontology_symmetric_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager, "Information extraction progresses",
                ['predicate-actor: progress-extraction', 'noun-noun: extraction-information',
                        'word: information', 'word: extraction'])

    def test_text_in_ontology_lemma_not_in_ontology_symmetric_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager,
                "He saw rainbows",
                ['predicate-patient: see-arc', 'word: arc'])

    def test_text_in_ontology_lemma_not_in_ontology_no_hypernym_replacement_symm_ontology(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager,
                "He saw rainbows",
                ['predicate-patient: see-rainbows', 'word: rainbows'], False)

    def test_prepposs(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager,
                "He needs insurance for five years",
                ['predicate-patient: need-insurance', 'number-noun: year-five',
                'prepgovernor-noun: need-year', 'prepgovernor-noun: insurance-year',
                'word: insurance', 'word: year'], False)

    def test_reverse_only(self):
        self._check_equals(symmetric_ontology_nocoref_holmes_manager,
                "He needs insurance for five years",
                ['predicate-patient: need-insurance', 'number-noun: year-five',
                'prepgovernor-noun: need-year', 'prepgovernor-noun: insurance-year',
                'word: insurance', 'word: year', 'prep-noun: for-year'], False,
                include_reverse_only=True)

    def test_coref(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                "I saw a dog. He was chasing a cat and a cat",
                ['predicate-patient: see-dog', 'predicate-actor: chase-dog',
                'predicate-patient: chase-cat', 'word: dog', 'word: cat'])

    def test_phraselet_stop_words_governor(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                "Always he had it", ['word: always'])

    def test_phraselet_stop_words_governed(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                "So he did it at home", ['word: home', 'prepgovernor-noun: do-home'])

    def test_coref_and_serialization(self):
        no_ontology_coref_holmes_manager.remove_all_search_phrases()
        doc = no_ontology_coref_holmes_manager.semantic_analyzer.parse(
                "I saw a dog. He was chasing a cat and a cat")
        phraselet_labels_to_search_phrases = {}
        serialized_phraselets = \
                no_ontology_coref_holmes_manager.structural_matcher.add_phraselets_to_dict(
                doc,
                phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                replace_with_hypernym_ancestors=False,
                match_all_words=False,
                include_reverse_only=False,
                ignore_relation_phraselets=False,
                returning_serialized_phraselets=True)
        deserialized_phraselet_labels_to_search_phrases = \
                no_ontology_coref_holmes_manager.structural_matcher.deserialize_phraselets(
                serialized_phraselets)
        self.assertEqual(set(
                deserialized_phraselet_labels_to_search_phrases.keys()),
                set(['predicate-patient: see-dog', 'predicate-actor: chase-dog',
                'predicate-patient: chase-cat', 'word: dog', 'word: cat']))

    def test_only_verb(self):
        self._check_equals(ontology_holmes_manager, "jump",
                ['word: jump'])

    def test_only_preposition(self):
        self._check_equals(ontology_holmes_manager, "in",
                ['word: in'])

    def test_match_all_words(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                "I saw a dog. He was chasing a cat and a cat",
                ['predicate-actor: chase-dog', 'predicate-patient: chase-cat',
                'predicate-patient: see-dog', 'word: dog', 'word: cat',
                'word: see', 'word: chase', 'word: -pron-'], False, True)

    def test_entity_defined_multiword_not_match_all_words(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                "Richard Paul Hudson came",
                ['predicate-actor: come-richard paul hudson',
                'word: richard paul hudson'], False, False)

    def test_entity_defined_multiword_match_all_words(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                "Richard Paul Hudson came",
                ['predicate-actor: come-richard paul hudson',
                'word: richard', 'word: paul', 'word: hudson', 'word: come'], False, True)
