import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join(
    (script_directory, 'test_ontology.owl')))
ontology_holmes_manager = holmes.Manager(model='en_core_web_trf',
                                         perform_coreference_resolution=False, ontology=ontology)
ontology_holmes_manager_adm_false = holmes.Manager(model='en_core_web_trf',
                                                   perform_coreference_resolution=False, ontology=ontology,
                                                   analyze_derivational_morphology=False)
symmetric_ontology = holmes.Ontology(os.sep.join((script_directory, 'test_ontology.owl')),
                                     symmetric_matching=True)
symmetric_ontology_nocoref_holmes_manager = holmes.Manager(model='en_core_web_trf',
                                                           ontology=symmetric_ontology, perform_coreference_resolution=False)
no_ontology_coref_holmes_manager = holmes.Manager(model='en_core_web_trf',
                                                  perform_coreference_resolution=True)


class EnglishPhraseletProductionTest(unittest.TestCase):

    def _check_equals(self, manager, text_to_match, phraselet_labels,
                      replace_with_hypernym_ancestors=True, match_all_words=False,
                      include_reverse_only=False):
        manager.remove_all_search_phrases()
        doc = manager.semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_phraselet_infos = {}
        manager.structural_matcher.add_phraselets_to_dict(doc,
                                                          phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                                                          replace_with_hypernym_ancestors=replace_with_hypernym_ancestors,
                                                          match_all_words=match_all_words,
                                                          ignore_relation_phraselets=False,
                                                          include_reverse_only=include_reverse_only,
                                                          stop_lemmas=manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                                                          stop_tags=manager.semantic_analyzer.topic_matching_phraselet_stop_tags,
                                                          reverse_only_parent_lemmas=manager.semantic_analyzer.
                                                          topic_matching_reverse_only_parent_lemmas,
                                                          words_to_corpus_frequencies=None,
                                                          maximum_corpus_frequency=None)
        self.assertEqual(
            set(phraselet_labels_to_phraselet_infos.keys()),
            set(phraselet_labels))
        self.assertEqual(len(phraselet_labels_to_phraselet_infos.keys()),
                         len(phraselet_labels))

    def _get_phraselet_dict(self, manager, text_to_match, words_to_corpus_frequencies=None,
        maximum_corpus_frequency=None):
        manager.remove_all_search_phrases()
        doc = manager.semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_phraselet_infos = {}
        manager.structural_matcher.add_phraselets_to_dict(doc,
                                                          phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                                                          replace_with_hypernym_ancestors=False,
                                                          match_all_words=True,
                                                          ignore_relation_phraselets=False,
                                                          include_reverse_only=True,
                                                          stop_lemmas=manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                                                          stop_tags=manager.semantic_analyzer.topic_matching_phraselet_stop_tags,
                                                          reverse_only_parent_lemmas=manager.semantic_analyzer.
                                                          topic_matching_reverse_only_parent_lemmas,
                                                          words_to_corpus_frequencies=words_to_corpus_frequencies,
                                                          maximum_corpus_frequency=maximum_corpus_frequency)
        return phraselet_labels_to_phraselet_infos

    def test_verb_subject_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A plant grows",
                           ['predicate-actor: grow-plant', 'word: plant'])

    def test_phrasal_verb_subject_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A plant grows up quickly",
                           ['governor-adjective: grow up-quick', 'predicate-actor: grow up-plant',
                            'word: plant'])

    def test_phrasal_verb_subject_no_entry_in_ontology_adm_false(self):
        self._check_equals(ontology_holmes_manager_adm_false, "A plant grows up quickly",
                           ['governor-adjective: grow up-quickly', 'predicate-actor: grow up-plant',
                            'word: plant'])

    def test_verb_direct_object_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A plant is grown",
                           ['predicate-passivesubject: grow-plant', 'word: plant'])

    def test_verb_indirect_object_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "Somebody gives something to a plant",
                           ['predicate-recipient: give-plant', 'word: plant'])

    def test_noun_adjective_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A healthy plant",
                           ['governor-adjective: plant-healthy', 'word: plant'])

    def test_verb_adverb_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "They sailed rapidly",
                           ['governor-adjective: sail-rapid'])

    def test_verb_adverb_no_entry_in_ontology_adm_false(self):
        self._check_equals(ontology_holmes_manager_adm_false, "They sailed rapidly",
                           ['governor-adjective: sail-rapidly'])

    def test_noun_noun_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A hobby plant",
                           ['noun-noun: plant-hobby', 'word: plant', 'word: hobby'])

    def test_possessor_possessed_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager, "A gardener's plant",
                           ['word-ofword: plant-gardener', 'word: plant', 'word: gardener'])

    def test_combined_no_entry_in_ontology(self):
        self._check_equals(ontology_holmes_manager,
                           "A gardener's healthy hobby plant grows in the sun",
                           ['predicate-actor: grow-plant', 'governor-adjective: plant-healthy',
                            'noun-noun: plant-hobby', 'word-ofword: plant-gardener',
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
                           ['predicate-actor: progress-extract', 'noun-noun: extract-inform',
                            'prepgovernor-noun: progress-inform', 'word: inform', 'word: extract'])

    def test_multiword_not_in_ontology_analyze_derivational_morphology_false(self):
        self._check_equals(ontology_holmes_manager_adm_false,
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
                           ['predicate-actor: progress-extract', 'noun-noun: extract-inform',
                            'word: inform', 'word: extract'])

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

    def test_reverse_only_parent_lemma(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                           "Always he had it", ['governor-adjective: have-always'], include_reverse_only=True)

    def test_reverse_only_parent_lemma_suppressed(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                           "Always he had it", ['word: have', 'word: always'], include_reverse_only=False)

    def test_phraselet_stop_words_governed(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                           "So he did it at home", ['word: home', 'prepgovernor-noun: do-home',
                                                    'prep-noun: at-home'],
                           include_reverse_only=True)

    def test_phraselet_stop_words_governed_suppressed(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                           "So he did it at home", ['word: home'],
                           include_reverse_only=False)

    def test_coref_and_phraselet_labels(self):
        no_ontology_coref_holmes_manager.remove_all_search_phrases()
        doc = no_ontology_coref_holmes_manager.semantic_analyzer.parse(
            "I saw a dog. He was chasing a cat and a cat")
        phraselet_labels_to_phraselet_infos = {}
        no_ontology_coref_holmes_manager.structural_matcher.add_phraselets_to_dict(
            doc,
            phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors=False,
            match_all_words=False,
            include_reverse_only=False,
            ignore_relation_phraselets=False,
            stop_lemmas=no_ontology_coref_holmes_manager.
            semantic_analyzer.topic_matching_phraselet_stop_lemmas,
            stop_tags=no_ontology_coref_holmes_manager.
            semantic_analyzer.topic_matching_phraselet_stop_tags,
            reverse_only_parent_lemmas=no_ontology_coref_holmes_manager.semantic_analyzer.
            topic_matching_reverse_only_parent_lemmas,
            words_to_corpus_frequencies=None,
            maximum_corpus_frequency=None)
        self.assertEqual(set(
            phraselet_labels_to_phraselet_infos.keys()),
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
                            'word: see', 'word: chase'], False, True)

    def test_entity_defined_multiword_not_match_all_words(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                           "Richard Paul Hudson came",
                           ['predicate-actor: come-richard paul hudson',
                            'word: richard paul hudson'], False, False)

    def test_entity_defined_multiword_not_match_all_words_with_adjective(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                           "The big Richard Paul Hudson",
                           ['governor-adjective: richard paul hudson-big',
                            'word: richard paul hudson'], False, False)

    def test_ontology_defined_multiword_not_match_all_words_with_adjective(self):
        self._check_equals(ontology_holmes_manager,
                           "The big Mimi Momo",
                           ['governor-adjective: mimi momo-big',
                            'word: mimi momo'], False, False)

    def test_ontology_and_entity_defined_multiword_not_match_all_words_with_adjective(self):
        self._check_equals(ontology_holmes_manager,
                           "The big Richard Mimi Momo",
                           ['governor-adjective: richard mimi momo-big',
                            'word: richard mimi momo'], False, False)

    def test_entity_defined_multiword_match_all_words(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                           "Richard Paul Hudson came",
                           ['predicate-actor: come-richard paul hudson',
                            'word: richard', 'word: paul', 'word: hudson', 'word: come'], False, True)

    def test_entity_defined_multiword_match_all_words_with_adjective(self):
        self._check_equals(no_ontology_coref_holmes_manager,
                           "The big Richard Paul Hudson",
                           ['governor-adjective: richard paul hudson-big',
                            'word: richard', 'word: paul', 'word: hudson', 'word: big'], False, True)

    def test_ontology_defined_multiword_match_all_words_with_adjective(self):
        self._check_equals(ontology_holmes_manager,
                           "The big Mimi Momo",
                           ['governor-adjective: mimi momo-big',
                            'word: mimi momo', 'word: big'], False, True)

    def test_ontology_and_entity_defined_multiword_not_match_all_words_with_adjective(self):
        self._check_equals(ontology_holmes_manager,
                           "The big Richard Mimi Momo",
                           ['governor-adjective: mimi momo-big', 'noun-noun: mimi momo-richard',
                            'word: mimi momo', 'word: richard', 'word: big'], False, True)

    def test_noun_lemmas_preferred_noun_lemma_first(self):
        dict = self._get_phraselet_dict(no_ontology_coref_holmes_manager,
                                        "They discussed anonymity. They wanted to use anonymous.")
        self.assertFalse('word: anonymous' in dict)
        self.assertFalse('governor-adjective: use-anonymous' in dict)
        word_phraselet = dict['word: anonymity']
        self.assertEqual(word_phraselet.parent_lemma, 'anonymity')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'anonymity')
        relation_phraselet = dict['governor-adjective: use-anonymity']
        self.assertEqual(relation_phraselet.child_lemma, 'anonymous')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'anonymity')

    def test_noun_lemmas_preferred_noun_lemma_second(self):
        dict = self._get_phraselet_dict(no_ontology_coref_holmes_manager,
                                        "They wanted to use anonymous. They discussed anonymity.")
        self.assertFalse('word: anonymous' in dict)
        self.assertFalse('governor-adjective: use-anonymous' in dict)
        word_phraselet = dict['word: anonymity']
        self.assertEqual(word_phraselet.parent_lemma, 'anonymity')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'anonymity')
        relation_phraselet = dict['governor-adjective: use-anonymity']
        self.assertEqual(relation_phraselet.child_lemma, 'anonymous')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'anonymity')

    def test_noun_lemmas_preferred_control(self):
        dict = self._get_phraselet_dict(no_ontology_coref_holmes_manager,
                                        "They wanted to use anonymous.")
        self.assertFalse('word: anonymous' in dict)
        self.assertFalse('governor-adjective: use-anonymous' in dict)
        word_phraselet = dict['word: anonymity']
        self.assertEqual(word_phraselet.parent_lemma, 'anonymous')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'anonymity')
        relation_phraselet = dict['governor-adjective: use-anonymity']
        self.assertEqual(relation_phraselet.child_lemma, 'anonymous')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'anonymity')

    def test_shorter_lemmas_preferred_shorter_lemma_first(self):
        dict = self._get_phraselet_dict(no_ontology_coref_holmes_manager,
                                        "They discussed behavior. They discussed behaviour.")
        self.assertFalse('word: behaviour' in dict)
        self.assertFalse('word: behavior' in dict)
        self.assertFalse('predicate-patient: discuss-behaviour' in dict)
        self.assertFalse('predicate-patient: discuss-behavior' in dict)
        word_phraselet = dict['word: behave']
        self.assertEqual(word_phraselet.parent_lemma, 'behavior')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'behave')
        relation_phraselet = dict['predicate-patient: discuss-behave']
        self.assertEqual(relation_phraselet.child_lemma, 'behavior')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'behave')

    def test_shorter_lemmas_preferred_adm_false_control(self):
        dict = self._get_phraselet_dict(ontology_holmes_manager_adm_false,
                                        "They discussed behavior. They discussed behaviour.")
        self.assertTrue('word: behaviour' in dict)
        self.assertTrue('word: behavior' in dict)
        self.assertFalse('word: behave' in dict)
        self.assertTrue('predicate-patient: discuss-behaviour' in dict)
        self.assertTrue('predicate-patient: discuss-behavior' in dict)
        self.assertFalse('predicate-patient: discuss-behave' in dict)
        word_phraselet = dict['word: behavior']
        self.assertEqual(word_phraselet.parent_lemma, 'behavior')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'behavior')
        relation_phraselet = dict['predicate-patient: discuss-behavior']
        self.assertEqual(relation_phraselet.child_lemma, 'behavior')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'behavior')

    def test_shorter_lemmas_preferred_shorter_lemma_second(self):
        dict = self._get_phraselet_dict(no_ontology_coref_holmes_manager,
                                        "They discussed behaviour. They discussed behavior.")
        self.assertFalse('word: behaviour' in dict)
        self.assertFalse('word: behavior' in dict)
        self.assertFalse('predicate-patient: discuss-behaviour' in dict)
        self.assertFalse('predicate-patient: discuss-behavior' in dict)
        word_phraselet = dict['word: behave']
        self.assertEqual(word_phraselet.parent_lemma, 'behavior')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'behave')
        relation_phraselet = dict['predicate-patient: discuss-behave']
        self.assertEqual(relation_phraselet.child_lemma, 'behavior')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'behave')

    def test_shorter_lemmas_preferred_control(self):
        dict = self._get_phraselet_dict(no_ontology_coref_holmes_manager,
                                        "They discussed behaviour. They behaved")
        self.assertFalse('word: behaviour' in dict)
        self.assertFalse('word: behavior' in dict)
        self.assertFalse('predicate-patient: discuss-behaviour' in dict)
        self.assertFalse('predicate-patient: discuss-behavior' in dict)
        word_phraselet = dict['word: behave']
        self.assertEqual(word_phraselet.parent_lemma, 'behaviour')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'behave')
        relation_phraselet = dict['predicate-patient: discuss-behave']
        self.assertEqual(relation_phraselet.child_lemma, 'behaviour')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'behave')

    def test_reverse_derived_lemmas_in_ontology_one_lemma_1(self):
        dict = self._get_phraselet_dict(ontology_holmes_manager,
                                        "He ate moodily")
        self.assertFalse('word: moody' in dict)
        self.assertFalse('governor-adjective: eat-moody' in dict)
        word_phraselet = dict['word: moodiness']
        self.assertEqual(word_phraselet.parent_lemma, 'moodily')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'moodiness')
        relation_phraselet = dict['governor-adjective: eat-moodiness']
        self.assertEqual(relation_phraselet.child_lemma, 'moodily')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'moodiness')

    def test_reverse_derived_lemmas_in_ontology_one_lemma_2(self):
        dict = self._get_phraselet_dict(ontology_holmes_manager,
                                        "He offended the cat")
        self.assertFalse('word: offend' in dict)
        self.assertFalse('predicate-patient: offend-cat' in dict)
        word_phraselet = dict['word: offence']
        self.assertEqual(word_phraselet.parent_lemma, 'offend')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'offence')
        relation_phraselet = dict['predicate-patient: offence-cat']
        self.assertEqual(relation_phraselet.parent_lemma, 'offend')
        self.assertEqual(relation_phraselet.parent_derived_lemma, 'offence')
        doc = ontology_holmes_manager.semantic_analyzer.parse(
            'He took offense')
        ontology_holmes_manager.structural_matcher.add_phraselets_to_dict(doc,
                                                                          phraselet_labels_to_phraselet_infos=dict,
                                                                          replace_with_hypernym_ancestors=False,
                                                                          match_all_words=True,
                                                                          ignore_relation_phraselets=False,
                                                                          include_reverse_only=True,
                                                                          stop_lemmas=ontology_holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                                                                          stop_tags=ontology_holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_tags,
                                                                          reverse_only_parent_lemmas=ontology_holmes_manager.semantic_analyzer.
                                                                          topic_matching_reverse_only_parent_lemmas,
                                                                          words_to_corpus_frequencies=None,
                                                                          maximum_corpus_frequency=None)
        word_phraselet = dict['word: offence']
        self.assertEqual(word_phraselet.parent_lemma, 'offense')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'offence')
        doc = ontology_holmes_manager.semantic_analyzer.parse(
            'He took offence')
        ontology_holmes_manager.structural_matcher.add_phraselets_to_dict(doc,
                                                                          phraselet_labels_to_phraselet_infos=dict,
                                                                          replace_with_hypernym_ancestors=False,
                                                                          match_all_words=True,
                                                                          ignore_relation_phraselets=False,
                                                                          include_reverse_only=True,
                                                                          stop_lemmas=ontology_holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                                                                          stop_tags=ontology_holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_tags,
                                                                          reverse_only_parent_lemmas=ontology_holmes_manager.semantic_analyzer.
                                                                          topic_matching_reverse_only_parent_lemmas,
                                                                          words_to_corpus_frequencies=None,
                                                                          maximum_corpus_frequency=None)
        word_phraselet = dict['word: offence']
        self.assertEqual(word_phraselet.parent_lemma, 'offense')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'offence')

    def test_reverse_derived_lemmas_in_ontology_multiword(self):
        dict = self._get_phraselet_dict(ontology_holmes_manager,
                                        "He used a waste horse")
        self.assertFalse('word: waste horse' in dict)
        self.assertFalse('predicate-patient: use-waste horse' in dict)
        word_phraselet = dict['word: wastage horse']
        self.assertEqual(word_phraselet.parent_lemma, 'wastage horse')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'wastage horse')
        relation_phraselet = dict['predicate-patient: use-wastage horse']
        self.assertEqual(relation_phraselet.child_lemma, 'wastage horse')
        self.assertEqual(
            relation_phraselet.child_derived_lemma, 'wastage horse')

    def test_frequency_factors_small(self):
        dict = self._get_phraselet_dict(ontology_holmes_manager,
                                        "The dog chased the cat",
                                        words_to_corpus_frequencies={'dog': 1, 'chasing': 1, 'cat': 2}, maximum_corpus_frequency=5)
        dog_phraselet = dict['word: dog']
        self.assertEqual(str(dog_phraselet.frequency_factor), '1.0')
        cat_phraselet = dict['word: cat']
        self.assertEqual(str(cat_phraselet.frequency_factor), '1.0')
        chase_phraselet = dict['word: chasing']
        self.assertEqual(str(chase_phraselet.frequency_factor), '1.0')
        chase_dog_phraselet = dict['predicate-actor: chasing-dog']
        self.assertEqual(str(chase_dog_phraselet.frequency_factor), '1.0')
        chase_cat_phraselet = dict['predicate-patient: chasing-cat']
        self.assertEqual(str(chase_cat_phraselet.frequency_factor), '1.0')

    def test_frequency_factors_small_with_small_mcf(self):
        dict = self._get_phraselet_dict(ontology_holmes_manager,
                                        "The dog chased the cat",
                                        words_to_corpus_frequencies={'dog': 1, 'chasing': 1, 'cat': 2}, maximum_corpus_frequency=2)
        dog_phraselet = dict['word: dog']
        self.assertEqual(str(dog_phraselet.frequency_factor), '1.0')
        cat_phraselet = dict['word: cat']
        self.assertEqual(str(cat_phraselet.frequency_factor), '1.0')
        chase_phraselet = dict['word: chasing']
        self.assertEqual(str(chase_phraselet.frequency_factor), '1.0')
        chase_dog_phraselet = dict['predicate-actor: chasing-dog']
        self.assertEqual(str(chase_dog_phraselet.frequency_factor), '1.0')
        chase_cat_phraselet = dict['predicate-patient: chasing-cat']
        self.assertEqual(str(chase_cat_phraselet.frequency_factor), '1.0')

    def test_frequency_factors_large(self):
        dict = self._get_phraselet_dict(ontology_holmes_manager,
                                        "The dog chased the cat",
                                        words_to_corpus_frequencies={'dog': 3, 'chasing': 4, 'cat': 5}, maximum_corpus_frequency=5)
        dog_phraselet = dict['word: dog']
        self.assertEqual(str(dog_phraselet.frequency_factor), '0.5693234419266069')
        cat_phraselet = dict['word: cat']
        self.assertEqual(str(cat_phraselet.frequency_factor), '0.1386468838532139')
        chase_phraselet = dict['word: chasing']
        self.assertEqual(str(chase_phraselet.frequency_factor), '0.31739380551401464')
        chase_dog_phraselet = dict['predicate-actor: chasing-dog']
        self.assertEqual(str(chase_dog_phraselet.frequency_factor), '0.18069973380142287')
        chase_cat_phraselet = dict['predicate-patient: chasing-cat']
        self.assertEqual(str(chase_cat_phraselet.frequency_factor), '0.044005662088831145')

    def test_frequency_factors_large_with_ontology_match(self):
        dict = self._get_phraselet_dict(ontology_holmes_manager,
                                        "The dog chased the cat",
                                        words_to_corpus_frequencies={'dog': 2, 'puppy': 4, 'chasing': 4, 'cat': 5}, maximum_corpus_frequency=5)
        dog_phraselet = dict['word: dog']
        self.assertEqual(str(dog_phraselet.frequency_factor), '0.5693234419266069')
        cat_phraselet = dict['word: cat']
        self.assertEqual(str(cat_phraselet.frequency_factor), '0.1386468838532139')
        chase_phraselet = dict['word: chasing']
        self.assertEqual(str(chase_phraselet.frequency_factor), '0.31739380551401464')
        chase_dog_phraselet = dict['predicate-actor: chasing-dog']
        self.assertEqual(str(chase_dog_phraselet.frequency_factor), '0.18069973380142287')
        chase_cat_phraselet = dict['predicate-patient: chasing-cat']
        self.assertEqual(str(chase_cat_phraselet.frequency_factor), '0.044005662088831145')

    def test_frequency_factors_very_large(self):
        dict = self._get_phraselet_dict(ontology_holmes_manager,
                                        "The dog chased the cat",
                                        words_to_corpus_frequencies={'dog': 97, 'chasing': 98, 'cat': 99}, maximum_corpus_frequency=100)
        dog_phraselet = dict['word: dog']
        self.assertEqual(str(dog_phraselet.frequency_factor), '0.008864383480215898')
        cat_phraselet = dict['word: cat']
        self.assertEqual(str(cat_phraselet.frequency_factor), '0.0043869621537525605')
        chase_phraselet = dict['word: chasing']
        self.assertEqual(str(chase_phraselet.frequency_factor), '0.00661413286687762')
        chase_dog_phraselet = dict['predicate-actor: chasing-dog']
        self.assertEqual(str(chase_dog_phraselet.frequency_factor), '5.863021012110299e-05')
        chase_cat_phraselet = dict['predicate-patient: chasing-cat']
        self.assertEqual(str(chase_cat_phraselet.frequency_factor), '2.9015950566883042e-05')
