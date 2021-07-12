import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join(
    (script_directory, 'test_ontology.owl')))
holmes_manager = holmes.Manager('de_core_news_lg', ontology=ontology, number_of_workers=1)


class GermanPhraseletProductionTest(unittest.TestCase):

    def _check_equals(self, text_to_match, phraselet_labels, match_all_words=False,
                      include_reverse_only=False, replace_with_hypernym_ancestors=False):
        doc = holmes_manager.semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_phraselet_infos = {}
        holmes_manager.linguistic_object_factory.add_phraselets_to_dict(doc,
                                                                 phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                                                                 replace_with_hypernym_ancestors=replace_with_hypernym_ancestors,
                                                                 match_all_words=match_all_words,
                                                                 ignore_relation_phraselets=False,
                                                                 include_reverse_only=include_reverse_only,
                                                                 stop_lemmas=holmes_manager.semantic_matching_helper.topic_matching_phraselet_stop_lemmas,
                                                                 stop_tags=holmes_manager.semantic_matching_helper.topic_matching_phraselet_stop_tags,
                                                                 reverse_only_parent_lemmas=holmes_manager.semantic_matching_helper.
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
        manager.linguistic_object_factory.add_phraselets_to_dict(doc,
                                                          phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                                                          replace_with_hypernym_ancestors=False,
                                                          match_all_words=True,
                                                          ignore_relation_phraselets=False,
                                                          include_reverse_only=True,
                                                          stop_lemmas=manager.semantic_matching_helper.topic_matching_phraselet_stop_lemmas,
                                                          stop_tags=manager.semantic_matching_helper.topic_matching_phraselet_stop_tags,
                                                          reverse_only_parent_lemmas=manager.semantic_matching_helper.
                                                          topic_matching_reverse_only_parent_lemmas,words_to_corpus_frequencies=words_to_corpus_frequencies,                            maximum_corpus_frequency=maximum_corpus_frequency)
        return phraselet_labels_to_phraselet_infos

    def test_verb_nom(self):
        self._check_equals("Eine Pflanze wächst", [
                           'verb-nom: wachsen-pflanz', 'word: pflanz'])

    def test_separable_verb_nom(self):
        self._check_equals("Eine Pflanze wächst auf",
                           ['verb-nom: aufwachsen-pflanz', 'word: pflanz'])

    def test_verb_acc(self):
        self._check_equals("Eine Pflanze wird gepflanzt", ['verb-acc: pflanzen-pflanz',
                                                           'word: pflanz'])

    def test_verb_dat(self):
        self._check_equals("Jemand gibt einer Pflanze etwas", ['verb-dat: gabe-pflanz',
                                                               'word: pflanz'])

    def test_noun_dependent_adjective(self):
        self._check_equals("Eine gesunde Pflanze", ['noun-dependent: pflanz-gesund',
                                                    'word: pflanz'])

    def test_noun_dependent_noun(self):
        self._check_equals("Die Pflanze eines Gärtners", ['verb-acc: pflanz-gärtner',
                                                          'word: gärtner', 'word: pflanz'])

    def test_verb_adverb(self):
        self._check_equals("lange schauen", ['verb-adverb: schau-lang'])

    def test_combination(self):
        self._check_equals("Der Gärtner gibt der netten Frau ihr Mittagessen",
                           ['verb-nom: gabe-gärtner', 'verb-acc: gabe-mittagessen',
                            'verb-dat: gabe-frau', 'noun-dependent: frau-nett',
                            'noun-dependent: mittagessen-frau', 'word: gärtner', 'word: frau',
                            'word: mittagessen'])

    def test_phraselet_labels(self):
        doc = holmes_manager.semantic_analyzer.parse(
            "Der Gärtner gibt der netten Frau ihr Mittagessen")
        phraselet_labels_to_phraselet_infos = {}
        holmes_manager.linguistic_object_factory.add_phraselets_to_dict(doc,
                                                                 phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                                                                 replace_with_hypernym_ancestors=False,
                                                                 match_all_words=False,
                                                                 include_reverse_only=True,
                                                                 ignore_relation_phraselets=False,
                                                                 stop_lemmas=holmes_manager.semantic_matching_helper.topic_matching_phraselet_stop_lemmas,
                                                                 stop_tags=holmes_manager.semantic_matching_helper.topic_matching_phraselet_stop_tags,
                                                                 reverse_only_parent_lemmas=holmes_manager.semantic_matching_helper.
                                                                 topic_matching_reverse_only_parent_lemmas,
                                                                 words_to_corpus_frequencies=None,
                                                                 maximum_corpus_frequency=None)
        self.assertEqual(set(phraselet_labels_to_phraselet_infos.keys()),
                         set(['verb-nom: gabe-gärtner', 'verb-acc: gabe-mittagessen',
                              'verb-dat: gabe-frau', 'noun-dependent: frau-nett',
                              'noun-dependent: mittagessen-frau', 'word: gärtner', 'word: frau',
                              'word: mittagessen']))

    def test_phraselet_labels_with_intcompound(self):
        doc = holmes_manager.semantic_analyzer.parse(
            "Der Landschaftsgärtner gibt der netten Frau ihr Mittagessen")
        phraselet_labels_to_phraselet_infos = {}
        holmes_manager.linguistic_object_factory.add_phraselets_to_dict(doc,
                                                                 phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                                                                 replace_with_hypernym_ancestors=False,
                                                                 match_all_words=False,
                                                                 include_reverse_only=True,
                                                                 ignore_relation_phraselets=False,
                                                                 stop_lemmas=holmes_manager.semantic_matching_helper.topic_matching_phraselet_stop_lemmas,
                                                                 stop_tags=holmes_manager.semantic_matching_helper.topic_matching_phraselet_stop_tags,
                                                                 reverse_only_parent_lemmas=holmes_manager.semantic_matching_helper.
                                                                 topic_matching_reverse_only_parent_lemmas,
                                                                words_to_corpus_frequencies=None,
                                                                maximum_corpus_frequency=None)

        self.assertEqual(set(phraselet_labels_to_phraselet_infos.keys()),
                         set(['verb-nom: gabe-landschaftsgärtner', 'verb-acc: gabe-mittagessen',
                              'verb-dat: gabe-frau', 'noun-dependent: frau-nett',
                              'noun-dependent: mittagessen-frau', 'word: landschaftsgärtner',
                              'word: frau', 'word: mittagessen',
                              'intcompound: gärtner-landschaft', 'verb-nom: gabe-gärtner']))
        intcompound_phraselet_info = phraselet_labels_to_phraselet_infos[
            'intcompound: gärtner-landschaft']
        self.assertEqual(intcompound_phraselet_info.parent_lemma, 'gärtner')
        self.assertEqual(
            intcompound_phraselet_info.parent_derived_lemma, 'gärtner')
        self.assertEqual(intcompound_phraselet_info.child_lemma, 'landschaft')
        self.assertEqual(
            intcompound_phraselet_info.child_derived_lemma, 'landschaft')

    def test_reverse_only_parent_lemma(self):
        self._check_equals("Immer hat er es",
                           ['verb-adverb: haben-immer'], include_reverse_only=True)

    def test_reverse_only_parent_lemma_auxiliary(self):
        self._check_equals("Er hat es immer gehabt",
                           ['verb-adverb: haben-immer'], include_reverse_only=True)

    def test_reverse_only_parent_lemma_modal(self):
        self._check_equals("Immer soll er es haben",
                           ['verb-adverb: haben-immer'], include_reverse_only=True)

    def test_reverse_only_parent_lemma_suppressed(self):
        self._check_equals("Immer hat er es",
                           ['word: haben', 'word: immer'], include_reverse_only=False)

    def test_reverse_only_parent_lemma_suppressed_auxiliary(self):
        self._check_equals("Immer hat er es gehabt",
                           ['word: haben', 'word: immer'], include_reverse_only=False)

    def test_reverse_only_parent_lemma_suppressed_modal(self):
        self._check_equals("Immer soll er es haben",
                           ['word: haben', 'word: immer'], include_reverse_only=False)

    def test_phraselet_stop_words_governed(self):
        self._check_equals("Dann tat er es zu Hause",
                           ['word: haus', 'prepgovernor-noun: tat-haus',
                            'prep-noun: zu-haus'], include_reverse_only=True)

    def test_phraselet_stop_words_governed_suppressed(self):
        self._check_equals("Dann tat er es zu Hause",
                           ['word: haus'], include_reverse_only=False)

    def test_only_verb(self):
        self._check_equals("springen", ['word: sprung'])

    def test_only_preposition(self):
        self._check_equals("unter", ['word: unter'])

    def test_match_all_words(self):
        self._check_equals("Der Gärtner gibt der netten Frau ihr Mittagessen",
                           ['word: gärtner', 'word: frau', 'word: mittagessen',
                            'word: gabe', 'word: nett',  'verb-nom: gabe-gärtner',
                            'verb-dat: gabe-frau', 'verb-acc: gabe-mittagessen',
                            'noun-dependent: frau-nett', 'noun-dependent: mittagessen-frau'], True)

    def test_moposs(self):
        self._check_equals("Er braucht eine Versicherung für fünf Jahre",
                           ['verb-acc: brauchen-versichern', 'noun-dependent: jahr-fünf',
                            'prepgovernor-noun: brauchen-jahr', 'prepgovernor-noun: versichern-jahr',
                            'word: jahr', 'word: versichern'], False)

    def test_reverse_only(self):
        self._check_equals("Er braucht eine Versicherung für fünf Jahre",
                           ['verb-acc: brauchen-versichern', 'noun-dependent: jahr-fünf',
                            'prepgovernor-noun: brauchen-jahr', 'prepgovernor-noun: versichern-jahr',
                            'word: jahr', 'word: versichern', 'prep-noun: für-jahr'], False,
                           include_reverse_only=True)

    def test_entity_defined_multiword_not_match_all_words(self):
        self._check_equals("Richard Paul Hudson kam",
                           ['verb-nom: kommen-richard paul hudson',
                            'word: richard paul hudson'], False)

    def test_entity_defined_multiword_match_all_words(self):
        self._check_equals("Richard Paul Hudson kam",
                           ['verb-nom: kommen-richard paul hudson',
                            'word: richard', 'word: paul', 'word: hudson', 'word: kommen'], True)

    def test_simple_subwords_match_all_words(self):
        self._check_equals("Informationsextraktion aus den Daten wurde durchgeführt",
                           ['verb-acc: durchführen-informationsextraktion', 'word: extraktion',
                            'word: aus', 'word: informationsextraktion',
                            'prepgovernor-noun: informationsextraktion-datum', 'word: information',
                            'prepgovernor-noun: durchführen-datum', 'word: durchführen',
                            'intcompound: extraktion-information', 'word: datum',
                            'prepgovernor-noun: extraktion-datum', 'verb-acc: durchführen-extraktion'], True)

    def test_simple_subwords_not_match_all_words(self):
        self._check_equals("Informationsextraktion aus den Daten wurde durchgeführt",
                           ['verb-acc: durchführen-informationsextraktion',
                            'word: informationsextraktion',
                            'prepgovernor-noun: informationsextraktion-datum',
                            'prepgovernor-noun: durchführen-datum',
                            'intcompound: extraktion-information', 'word: datum',
                            'prepgovernor-noun: extraktion-datum', 'verb-acc: durchführen-extraktion'], False)

    def test_simple_subwords_match_all_words_include_reverse_only(self):
        self._check_equals("Informationsextraktion aus den Daten wurde durchgeführt",
                           ['verb-acc: durchführen-informationsextraktion', 'word: extraktion',
                            'word: aus', 'word: informationsextraktion',
                            'prepgovernor-noun: informationsextraktion-datum', 'word: information',
                            'prepgovernor-noun: durchführen-datum', 'word: durchführen',
                            'intcompound: extraktion-information', 'word: datum',
                            'prepgovernor-noun: extraktion-datum', 'verb-acc: durchführen-extraktion',
                            'prep-noun: aus-datum'], True,
                           True)

    def test_simple_subwords_not_match_all_words_include_reverse_only(self):
        self._check_equals("Informationsextraktion aus den Daten wurde durchgeführt",
                           ['verb-acc: durchführen-informationsextraktion',
                            'word: informationsextraktion',
                            'prepgovernor-noun: informationsextraktion-datum',
                            'prepgovernor-noun: durchführen-datum',
                            'intcompound: extraktion-information', 'word: datum',
                            'prepgovernor-noun: extraktion-datum', 'verb-acc: durchführen-extraktion',
                            'prep-noun: aus-datum'], False,
                           True)

    def test_subwords_replace_with_hypernym_ancestors_is_false(self):
        self._check_equals("Der Informationsmonitor war groß",
                           ['noun-dependent: informationsmonitor-groß',
                            'word: informationsmonitor',
                            'noun-dependent: monitor-groß',
                            'intcompound: monitor-information'], replace_with_hypernym_ancestors=False)

    def test_subwords_replace_with_hypernym_ancestors_is_true_not_match_all_words(self):
        self._check_equals("Der Informationsmonitor war groß",
                           ['noun-dependent: informationsmonitor-groß',
                            'word: informationsmonitor',
                            'noun-dependent: hardware-groß',
                            'intcompound: hardware-information'], replace_with_hypernym_ancestors=True)

    def test_subwords_replace_with_hypernym_ancestors_is_true_match_all_words(self):
        self._check_equals("Der Informationsmonitor war groß",
                           ['noun-dependent: informationsmonitor-groß',
                            'word: informationsmonitor',
                            'noun-dependent: hardware-groß',
                            'intcompound: hardware-information',
                            'word: groß', 'word: information', 'word: hardware', 'word: sein'],
                           replace_with_hypernym_ancestors=True,
                           match_all_words=True)

    def test_subwords_with_conjunction_match_all_words(self):
        self._check_equals("Der König von den Informationsinteressen-, -beschaffungs- und -problemmaßnahmen der Wettersituation",
                           ['word: wettersituation',
                            'intcompound: beschaffen-information',
                            'word: könig',
                            'verb-acc: könig-maßnahm',
                            'intcompound: problem-information',
                            'verb-acc: maßnahm-wettersituation',
                            'intcompound: situation-wetter',
                            'verb-acc: maßnahm-situation',
                            'intcompound: maßnahm-problem',
                            'intcompound: maßnahm-beschaffen',
                            'intcompound: maßnahm-interesse',
                            'intcompound: interesse-information',
                            'word: problem',
                            'word: information',
                            'word: interesse',
                            'word: beschaffen',
                            'word: wetter',
                            'word: situation',
                            'word: maßnahm'
                            ],
                           match_all_words=True)

    def test_subwords_with_conjunction_not_match_all_words(self):
        self._check_equals("Der König von den Informationsinteressen-, -beschaffungs- und -problemmaßnahmen der Wettersituation",
                           ['word: wettersituation',
                            'intcompound: beschaffen-information',
                            'word: könig',
                            'verb-acc: könig-maßnahm',
                            'intcompound: problem-information',
                            'verb-acc: maßnahm-wettersituation',
                            'intcompound: situation-wetter',
                            'verb-acc: maßnahm-situation',
                            'intcompound: maßnahm-problem',
                            'intcompound: maßnahm-beschaffen',
                            'intcompound: maßnahm-interesse',
                            'intcompound: interesse-information'
                            ],
                           match_all_words=False)

    def test_subwords_with_conjunction_one_not_hyphenated_not_match_all_words(self):
        self._check_equals("Der König der Informations- und Beschaffungsmaßnahmen der Wettersituation",
                           ['word: wettersituation',
                            'word: könig',
                            'verb-acc: könig-maßnahm',
                            'verb-acc: maßnahm-wettersituation',
                            'intcompound: situation-wetter',
                            'verb-acc: maßnahm-situation',
                            'intcompound: maßnahm-beschaffen',
                            'word: beschaffungsmaßnahmen',
                            'intcompound: maßnahm-information',
                            'verb-acc: beschaffungsmaßnahmen-wettersituation',
                            'verb-acc: beschaffungsmaßnahmen-situation',
                            'verb-acc: könig-beschaffungsmaßnahmen'
                            ],
                           match_all_words=False)

    def test_subwords_with_conjunction_one_not_hyphenated_match_all_words(self):
        self._check_equals("Der König der Informations- und Beschaffungsmaßnahmen der Wettersituation",
                           ['word: wettersituation',
                            'word: könig',
                            'verb-acc: könig-maßnahm',
                            'verb-acc: maßnahm-wettersituation',
                            'intcompound: situation-wetter',
                            'verb-acc: maßnahm-situation',
                            'intcompound: maßnahm-beschaffen',
                            'word: beschaffungsmaßnahmen',
                            'intcompound: maßnahm-information',
                            'verb-acc: beschaffungsmaßnahmen-wettersituation',
                            'verb-acc: beschaffungsmaßnahmen-situation',
                            'verb-acc: könig-beschaffungsmaßnahmen',
                            'word: information',
                            'word: beschaffen',
                            'word: wetter',
                            'word: situation',
                            'word: maßnahm'
                            ],
                           match_all_words=True)

    def test_noun_lemmas_preferred_noun_lemma_first(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Sie besprachen die Amputation. Sie hatten ein Amputieren vor")
        self.assertFalse('word: amputieren' in dict)
        self.assertFalse('verb-acc: vorhaben-amputieren' in dict)
        word_phraselet = dict['word: amputation']
        self.assertEqual(word_phraselet.parent_lemma, 'amputation')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'amputation')
        relation_phraselet = dict['verb-acc: vorhaben-amputation']
        self.assertEqual(relation_phraselet.child_lemma, 'amputieren')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'amputation')

    def test_noun_lemmas_preferred_noun_lemma_second(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Sie hatten ein Amputieren vor. Sie besprachen die Amputation.")
        self.assertFalse('word: amputieren' in dict)
        self.assertFalse('verb-acc: vorhaben-amputieren' in dict)
        word_phraselet = dict['word: amputation']
        self.assertEqual(word_phraselet.parent_lemma, 'amputieren')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'amputation')
        relation_phraselet = dict['verb-acc: vorhaben-amputation']
        self.assertEqual(relation_phraselet.child_lemma, 'amputieren')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'amputation')

    def test_noun_lemmas_preferred_control(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Sie hatten ein Amputieren vor.")
        self.assertFalse('word: amputieren' in dict)
        self.assertFalse('verb-acc: vorhaben-amputieren' in dict)
        word_phraselet = dict['word: amputation']
        self.assertEqual(word_phraselet.parent_lemma, 'amputieren')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'amputation')
        relation_phraselet = dict['verb-acc: vorhaben-amputation']
        self.assertEqual(relation_phraselet.child_lemma, 'amputieren')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'amputation')

    def test_shorter_lemmas_preferred_shorter_lemma_first(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Sie besprachen Information. Sie besprachen Informierung.")
        self.assertFalse('word: informierung' in dict)
        self.assertFalse('verb-acc: besprechen-informierung' in dict)
        word_phraselet = dict['word: information']
        self.assertEqual(word_phraselet.parent_lemma, 'information')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'information')
        relation_phraselet = dict['verb-acc: besprechen-information']
        self.assertEqual(relation_phraselet.child_lemma, 'information')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'information')

    def test_shorter_lemmas_preferred_shorter_lemma_second(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Sie besprachen Informierung. Sie besprachen Information.")
        self.assertFalse('word: informierung' in dict)
        self.assertFalse('verb-acc: besprechen-informierung' in dict)
        word_phraselet = dict['word: information']
        self.assertEqual(word_phraselet.parent_lemma, 'information')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'information')
        relation_phraselet = dict['verb-acc: besprechen-information']
        self.assertEqual(relation_phraselet.child_lemma, 'information')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'information')

    def test_shorter_lemmas_preferred_shorter_lemma_control(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Sie besprachen Informierung.")
        self.assertFalse('word: informierung' in dict)
        self.assertFalse('verb-acc: besprechen-informierung' in dict)
        word_phraselet = dict['word: information']
        self.assertEqual(word_phraselet.parent_lemma, 'informierung')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'information')
        relation_phraselet = dict['verb-acc: besprechen-information']
        self.assertEqual(relation_phraselet.child_lemma, 'informierung')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'information')

    def test_shorter_lemmas_preferred_subwords_shorter_lemma_first(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Eine Informationskomitee und eine Informierungskomitee.")
        self.assertFalse('word: informierung' in dict)
        self.assertFalse('intcompound: komitee-informierung' in dict)
        word_phraselet = dict['word: information']
        self.assertEqual(word_phraselet.parent_lemma, 'information')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'information')
        relation_phraselet = dict['intcompound: komitee-information']
        self.assertEqual(relation_phraselet.child_lemma, 'information')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'information')

    def test_shorter_lemmas_preferred_subwords_shorter_lemma_second(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Eine Informierungskomitee und eine Informationskomitee.")
        self.assertFalse('word: informierung' in dict)
        self.assertFalse('intcompound: komitee-informierung' in dict)
        word_phraselet = dict['word: information']
        self.assertEqual(word_phraselet.parent_lemma, 'information')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'information')
        relation_phraselet = dict['intcompound: komitee-information']
        self.assertEqual(relation_phraselet.child_lemma, 'information')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'information')

    def test_shorter_lemmas_preferred_subwords_control(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Eine Informierungskomitee.")
        self.assertFalse('word: informierung' in dict)
        self.assertFalse('intcompound: komitee-informierung' in dict)
        word_phraselet = dict['word: information']
        self.assertEqual(word_phraselet.parent_lemma, 'informierung')
        self.assertEqual(word_phraselet.parent_derived_lemma, 'information')
        relation_phraselet = dict['intcompound: komitee-information']
        self.assertEqual(relation_phraselet.child_lemma, 'informierung')
        self.assertEqual(relation_phraselet.child_derived_lemma, 'information')

    def test_intcompound_when_word_in_ontology(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Sprachwissenschaft.")
        self.assertEqual(set(dict.keys()), {'word: sprachwissenschaft', 'word: sprach',
                                            'word: wissenschaft', 'intcompound: wissenschaft-sprach'})

    def test_intcompound_when_reverse_derived_lemma_in_ontology(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Sammelabflug.")
        self.assertEqual(set(dict.keys()), {'word: sammelabflug', 'word: sammel', 'word: abfliegen',
                                            'intcompound: abfliegen-sammel'})

    def test_frequency_factors_with_subwords(self):
        dict = self._get_phraselet_dict(holmes_manager,
                                        "Sprachwissenschaft",
                                        words_to_corpus_frequencies={'sprach': 3,
                                        'sprachwissenschaft': 5, 'wissenschaft': 1},
                                        maximum_corpus_frequency=5)
        sprach_phraselet = dict['word: sprach']
        self.assertEqual(str(sprach_phraselet.frequency_factor), '0.5693234419266069')
        wissenschaft_phraselet = dict['word: wissenschaft']
        self.assertEqual(str(wissenschaft_phraselet.frequency_factor), '1.0')
        sprachwissenschaft_phraselet = dict['word: sprachwissenschaft']
        self.assertEqual(str(sprachwissenschaft_phraselet.frequency_factor), '0.1386468838532139')
