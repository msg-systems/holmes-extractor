import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
holmes_manager = holmes.Manager('de_core_news_md', ontology=ontology)

class GermanPhraseletProductionTest(unittest.TestCase):

    def _check_equals(self, text_to_match, phraselet_labels, match_all_words = False,
            include_reverse_only=False, replace_with_hypernym_ancestors=False):
        doc = holmes_manager.semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_phraselet_infos = {}
        holmes_manager.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                replace_with_hypernym_ancestors=replace_with_hypernym_ancestors,
                match_all_words=match_all_words,
                returning_serialized_phraselets=False,
                ignore_relation_phraselets=False,
                include_reverse_only=include_reverse_only,
                stop_lemmas = holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                reverse_only_parent_lemmas = holmes_manager.semantic_analyzer.\
                topic_matching_reverse_only_parent_lemmas)
        self.assertEqual(
                set(phraselet_labels_to_phraselet_infos.keys()),
                set(phraselet_labels))
        self.assertEqual(len(phraselet_labels_to_phraselet_infos.keys()),
                len(phraselet_labels))

    def test_verb_nom(self):
        self._check_equals("Eine Pflanze wächst", ['verb-nom: wachsen-pflanzen', 'word: pflanzen'])

    def test_separable_verb_nom(self):
        self._check_equals("Eine Pflanze wächst auf",
                ['verb-nom: aufwachsen-pflanzen', 'word: pflanzen'])

    def test_verb_acc(self):
        self._check_equals("Eine Pflanze wird gepflanzt", ['verb-acc: pflanzen-pflanzen',
                'word: pflanzen'])

    def test_verb_dat(self):
        self._check_equals("Jemand gibt einer Pflanze etwas", ['verb-dat: gabe-pflanzen',
                'word: pflanzen'])

    def test_noun_dependent_adjective(self):
        self._check_equals("Eine gesunde Pflanze", ['noun-dependent: pflanzen-gesund',
                'word: pflanzen'])

    def test_noun_dependent_noun(self):
        self._check_equals("Die Pflanze eines Gärtners", ['verb-acc: pflanzen-gärtner',
                'word: gärtner', 'word: pflanzen'])

    def test_verb_adverb(self):
        self._check_equals("lange schauen", ['verb-adverb: schau-lang'])

    def test_combination(self):
        self._check_equals("Der Gärtner gibt der netten Frau ihr Mittagessen",
                ['verb-nom: gabe-gärtnern', 'verb-acc: gabe-mittagessen',
                'verb-dat: gabe-frau', 'noun-dependent: frau-nett',
                'word: gärtnern', 'word: frau', 'word: mittagessen'])

    def test_phraselet_labels(self):
        doc = holmes_manager.semantic_analyzer.parse(
                "Der Gärtner gibt der netten Frau ihr Mittagessen")
        phraselet_labels_to_phraselet_infos = {}
        holmes_manager.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                replace_with_hypernym_ancestors=False,
                match_all_words=False,
                include_reverse_only=True,
                ignore_relation_phraselets=False,
                returning_serialized_phraselets=True,
                stop_lemmas = holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                reverse_only_parent_lemmas = holmes_manager.semantic_analyzer.\
                topic_matching_reverse_only_parent_lemmas)
        self.assertEqual(set(phraselet_labels_to_phraselet_infos.keys()),
                set(['verb-nom: gabe-gärtnern', 'verb-acc: gabe-mittagessen',
                'verb-dat: gabe-frau', 'noun-dependent: frau-nett',
                'word: gärtnern', 'word: frau', 'word: mittagessen']))

    def test_phraselet_labels_with_intcompound(self):
        doc = holmes_manager.semantic_analyzer.parse(
                "Der Landschaftsgärtner gibt der netten Frau ihr Mittagessen")
        phraselet_labels_to_phraselet_infos = {}
        holmes_manager.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_phraselet_infos=phraselet_labels_to_phraselet_infos,
                replace_with_hypernym_ancestors=False,
                match_all_words=False,
                include_reverse_only=True,
                ignore_relation_phraselets=False,
                returning_serialized_phraselets=True,
                stop_lemmas = holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                reverse_only_parent_lemmas = holmes_manager.semantic_analyzer.\
                topic_matching_reverse_only_parent_lemmas)
        self.assertEqual(set(phraselet_labels_to_phraselet_infos.keys()),
                set(['verb-nom: gabe-landschaftsgärtner', 'verb-acc: gabe-mittagessen',
                'verb-dat: gabe-frau', 'noun-dependent: frau-nett',
                'word: landschaftsgärtner', 'word: frau', 'word: mittagessen',
                'intcompound: gärtnern-landschaft', 'verb-nom: gabe-gärtnern']))
        intcompound_phraselet_info = phraselet_labels_to_phraselet_infos[
                'intcompound: gärtnern-landschaft']
        self.assertEqual(intcompound_phraselet_info.parent_lemma, 'gärtnern')
        self.assertEqual(intcompound_phraselet_info.parent_derived_lemma, 'gärtnern')
        self.assertEqual(intcompound_phraselet_info.child_lemma, 'landschaft')
        self.assertEqual(intcompound_phraselet_info.child_derived_lemma, 'landschaft')

    def test_reverse_only_parent_lemma(self):
        self._check_equals("Immer hat er es",
                ['verb-adverb: haben-immer'], include_reverse_only=True)

    def test_reverse_only_parent_lemma_auxiliary(self):
        self._check_equals("Immer hat er es gehabt",
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
                ['word: hausen', 'prepgovernor-noun: tat-hausen',
                        'prep-noun: zu-hausen'], include_reverse_only=True)

    def test_phraselet_stop_words_governed_suppressed(self):
        self._check_equals("Dann tat er es zu Hause",
                ['word: hausen'], include_reverse_only=False)

    def test_only_verb(self):
        self._check_equals("springen", ['word: sprung'])

    def test_only_preposition(self):
        self._check_equals("unter", ['word: unter'])

    def test_match_all_words(self):
        self._check_equals("Der Gärtner gibt der netten Frau ihr Mittagessen",
                ['word: gärtnern', 'word: frau', 'word: mittagessen',
                'word: gabe', 'word: nett',  'verb-nom: gabe-gärtnern',
                'verb-dat: gabe-frau', 'verb-acc: gabe-mittagessen',
                'noun-dependent: frau-nett'], True)

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
                'word: aus','word: informationsextraktion',
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
                'word: aus','word: informationsextraktion',
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
                match_all_words = True)

    def test_subwords_with_conjunction_match_all_words(self):
        self._check_equals("Der König der Informationsinteressen-, -beschaffungs- und -problemmaßnahmen der Wettersituation",
                ['word: wettersituation',
                'intcompound: beschaffen-information',
                'word: könig',
                'verb-acc: könig-maßnahm',
                'intcompound: problem-information',
                'verb-acc: maßnahm-wettersituation',
                'intcompound: situation-wettern',
                'verb-acc: maßnahm-situation',
                'intcompound: maßnahm-problem',
                'intcompound: maßnahm-beschaffen',
                'intcompound: maßnahm-interesse',
                'intcompound: interesse-information',
                'word: problem',
                'word: information',
                'word: interesse',
                'word: beschaffen',
                'word: wettern',
                'word: situation',
                'word: maßnahm'
                ],
                match_all_words = True)

    def test_subwords_with_conjunction_not_match_all_words(self):
        self._check_equals("Der König der Informationsinteressen-, -beschaffungs- und -problemmaßnahmen der Wettersituation",
                ['word: wettersituation',
                'intcompound: beschaffen-information',
                'word: könig',
                'verb-acc: könig-maßnahm',
                'intcompound: problem-information',
                'verb-acc: maßnahm-wettersituation',
                'intcompound: situation-wettern',
                'verb-acc: maßnahm-situation',
                'intcompound: maßnahm-problem',
                'intcompound: maßnahm-beschaffen',
                'intcompound: maßnahm-interesse',
                'intcompound: interesse-information'
                ],
                match_all_words = False)

    def test_subwords_with_conjunction_one_not_hyphenated_not_match_all_words(self):
        self._check_equals("Der König der Informations- und Beschaffungsmaßnahmen der Wettersituation",
                ['word: wettersituation',
                'word: könig',
                'verb-acc: könig-maßnahm',
                'verb-acc: maßnahm-wettersituation',
                'intcompound: situation-wettern',
                'verb-acc: maßnahm-situation',
                'intcompound: maßnahm-beschaffen',
                'word: beschaffungsmaßnahmen',
                'intcompound: maßnahm-information',
                'verb-acc: beschaffungsmaßnahmen-wettersituation',
                'verb-acc: beschaffungsmaßnahmen-situation',
                'verb-acc: könig-beschaffungsmaßnahmen'
                ],
                match_all_words = False)

    def test_subwords_with_conjunction_one_not_hyphenated_match_all_words(self):
        self._check_equals("Der König der Informations- und Beschaffungsmaßnahmen der Wettersituation",
                ['word: wettersituation',
                'word: könig',
                'verb-acc: könig-maßnahm',
                'verb-acc: maßnahm-wettersituation',
                'intcompound: situation-wettern',
                'verb-acc: maßnahm-situation',
                'intcompound: maßnahm-beschaffen',
                'word: beschaffungsmaßnahmen',
                'intcompound: maßnahm-information',
                'verb-acc: beschaffungsmaßnahmen-wettersituation',
                'verb-acc: beschaffungsmaßnahmen-situation',
                'verb-acc: könig-beschaffungsmaßnahmen',
                'word: information',
                'word: beschaffen',
                'word: wettern',
                'word: situation',
                'word: maßnahm'
                ],
                match_all_words = True)
