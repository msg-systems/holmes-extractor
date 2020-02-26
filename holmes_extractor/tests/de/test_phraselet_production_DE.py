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
        phraselet_labels_to_search_phrases = {}
        holmes_manager.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                replace_with_hypernym_ancestors=replace_with_hypernym_ancestors,
                match_all_words=match_all_words,
                returning_serialized_phraselets=False,
                ignore_relation_phraselets=False,
                include_reverse_only=include_reverse_only,
                stop_lemmas = holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                reverse_only_parent_lemmas = holmes_manager.semantic_analyzer.\
                topic_matching_reverse_only_parent_lemmas)
        self.assertEqual(
                set(phraselet_labels_to_search_phrases.keys()),
                set(phraselet_labels))
        self.assertEqual(len(phraselet_labels_to_search_phrases.keys()),
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
        self._check_equals("Jemand gibt einer Pflanze etwas", ['verb-dat: geben-pflanzen',
                'word: pflanzen'])

    def test_noun_dependent_adjective(self):
        self._check_equals("Eine gesunde Pflanze", ['noun-dependent: pflanzen-gesund',
                'word: pflanzen'])

    def test_noun_dependent_noun(self):
        self._check_equals("Die Pflanze eines Gärtners", ['noun-dependent: pflanzen-gärtner',
                'word: gärtner', 'word: pflanzen'])

    def test_verb_adverb(self):
        self._check_equals("lange schauen", ['verb-adverb: schauen-lang'])

    def test_combination(self):
        self._check_equals("Der Gärtner gibt der netten Frau ihr Mittagessen",
                ['verb-nom: geben-gärtnern', 'verb-acc: geben-mittagessen',
                'verb-dat: geben-frau', 'noun-dependent: frau-nett',
                'word: gärtnern', 'word: frau', 'word: mittagessen'])

    def test_serialization(self):
        doc = holmes_manager.semantic_analyzer.parse(
                "Der Gärtner gibt der netten Frau ihr Mittagessen")
        phraselet_labels_to_search_phrases = {}
        serialized_phraselets = holmes_manager.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                replace_with_hypernym_ancestors=False,
                match_all_words=False,
                include_reverse_only=True,
                ignore_relation_phraselets=False,
                returning_serialized_phraselets=True,
                stop_lemmas = holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                reverse_only_parent_lemmas = holmes_manager.semantic_analyzer.\
                topic_matching_reverse_only_parent_lemmas)
        deserialized_phraselet_labels_to_search_phrases = holmes_manager.structural_matcher.\
                deserialize_phraselets(serialized_phraselets)
        self.assertEqual(set(deserialized_phraselet_labels_to_search_phrases.keys()),
                set(['verb-nom: geben-gärtnern', 'verb-acc: geben-mittagessen',
                'verb-dat: geben-frau', 'noun-dependent: frau-nett',
                'word: gärtnern', 'word: frau', 'word: mittagessen']))

    def test_serialization_with_intcompound(self):
        doc = holmes_manager.semantic_analyzer.parse(
                "Der Landschaftsgärtner gibt der netten Frau ihr Mittagessen")
        phraselet_labels_to_search_phrases = {}
        serialized_phraselets = holmes_manager.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                replace_with_hypernym_ancestors=False,
                match_all_words=False,
                include_reverse_only=True,
                ignore_relation_phraselets=False,
                returning_serialized_phraselets=True,
                stop_lemmas = holmes_manager.semantic_analyzer.topic_matching_phraselet_stop_lemmas,
                reverse_only_parent_lemmas = holmes_manager.semantic_analyzer.\
                topic_matching_reverse_only_parent_lemmas)
        deserialized_phraselet_labels_to_search_phrases = holmes_manager.structural_matcher.\
                deserialize_phraselets(serialized_phraselets)
        self.assertEqual(set(deserialized_phraselet_labels_to_search_phrases.keys()),
                set(['verb-nom: geben-landschaftsgärtner', 'verb-acc: geben-mittagessen',
                'verb-dat: geben-frau', 'noun-dependent: frau-nett',
                'word: landschaftsgärtner', 'word: frau', 'word: mittagessen',
                'intcompound: gärtnern-landschaft', 'verb-nom: geben-gärtnern']))
        intcompound_search_phrase = deserialized_phraselet_labels_to_search_phrases[
                'intcompound: gärtnern-landschaft']
        self.assertEqual(intcompound_search_phrase.doc[1]._.holmes.lemma, 'gärtnern')
        self.assertEqual(intcompound_search_phrase.doc[4]._.holmes.lemma, 'landschaft')
        self.assertEqual(intcompound_search_phrase.doc[1]._.holmes.\
                string_representation_of_children(), '2:mnr; 4:intcompound')

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
                ['word: hausen', 'prepgovernor-noun: tun-hausen',
                        'prep-noun: zu-hausen'], include_reverse_only=True)

    def test_phraselet_stop_words_governed_suppressed(self):
        self._check_equals("Dann tat er es zu Hause",
                ['word: hausen'], include_reverse_only=False)

    def test_only_verb(self):
        self._check_equals("springen", ['word: springen'])

    def test_only_preposition(self):
        self._check_equals("unter", ['word: unter'])

    def test_match_all_words(self):
        self._check_equals("Der Gärtner gibt der netten Frau ihr Mittagessen",
                ['word: gärtnern', 'word: frau', 'word: mittagessen',
                'word: geben', 'word: nett',  'verb-nom: geben-gärtnern',
                'verb-dat: geben-frau', 'verb-acc: geben-mittagessen',
                'noun-dependent: frau-nett'], True)

    def test_moposs(self):
        self._check_equals("Er braucht eine Versicherung für fünf Jahre",
                ['verb-acc: brauchen-versicherung', 'noun-dependent: jahr-fünf',
                'prepgovernor-noun: brauchen-jahr', 'prepgovernor-noun: versicherung-jahr',
                'word: jahr', 'word: versicherung'], False)

    def test_reverse_only(self):
        self._check_equals("Er braucht eine Versicherung für fünf Jahre",
                ['verb-acc: brauchen-versicherung', 'noun-dependent: jahr-fünf',
                'prepgovernor-noun: brauchen-jahr', 'prepgovernor-noun: versicherung-jahr',
                'word: jahr', 'word: versicherung', 'prep-noun: für-jahr'], False,
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
                'intcompound: beschaffung-information',
                'word: könig',
                'noun-dependent: könig-maßnahme',
                'intcompound: problem-information',
                'noun-dependent: maßnahme-wettersituation',
                'intcompound: situation-wettern',
                'noun-dependent: maßnahme-situation',
                'intcompound: maßnahme-problem',
                'intcompound: maßnahme-beschaffung',
                'intcompound: maßnahme-interesse',
                'intcompound: interesse-information',
                'word: problem',
                'word: information',
                'word: interesse',
                'word: beschaffung',
                'word: wettern',
                'word: situation',
                'word: maßnahme'
                ],
                match_all_words = True)

    def test_subwords_with_conjunction_not_match_all_words(self):
        self._check_equals("Der König der Informationsinteressen-, -beschaffungs- und -problemmaßnahmen der Wettersituation",
                ['word: wettersituation',
                'intcompound: beschaffung-information',
                'word: könig',
                'noun-dependent: könig-maßnahme',
                'intcompound: problem-information',
                'noun-dependent: maßnahme-wettersituation',
                'intcompound: situation-wettern',
                'noun-dependent: maßnahme-situation',
                'intcompound: maßnahme-problem',
                'intcompound: maßnahme-beschaffung',
                'intcompound: maßnahme-interesse',
                'intcompound: interesse-information'
                ],
                match_all_words = False)

    def test_subwords_with_conjunction_one_not_hyphenated_not_match_all_words(self):
        self._check_equals("Der König der Informations- und Beschaffungsmaßnahmen der Wettersituation",
                ['word: wettersituation',
                'word: könig',
                'noun-dependent: könig-maßnahme',
                'noun-dependent: maßnahme-wettersituation',
                'intcompound: situation-wettern',
                'noun-dependent: maßnahme-situation',
                'intcompound: maßnahme-beschaffung',
                'word: beschaffungsmaßnahmen',
                'intcompound: maßnahme-information',
                'noun-dependent: beschaffungsmaßnahmen-wettersituation',
                'noun-dependent: beschaffungsmaßnahmen-situation',
                'noun-dependent: könig-beschaffungsmaßnahmen'
                ],
                match_all_words = False)

    def test_subwords_with_conjunction_one_not_hyphenated_match_all_words(self):
        self._check_equals("Der König der Informations- und Beschaffungsmaßnahmen der Wettersituation",
                ['word: wettersituation',
                'word: könig',
                'noun-dependent: könig-maßnahme',
                'noun-dependent: maßnahme-wettersituation',
                'intcompound: situation-wettern',
                'noun-dependent: maßnahme-situation',
                'intcompound: maßnahme-beschaffung',
                'word: beschaffungsmaßnahmen',
                'intcompound: maßnahme-information',
                'noun-dependent: beschaffungsmaßnahmen-wettersituation',
                'noun-dependent: beschaffungsmaßnahmen-situation',
                'noun-dependent: könig-beschaffungsmaßnahmen',
                'word: information',
                'word: beschaffung',
                'word: wettern',
                'word: situation',
                'word: maßnahme'
                ],
                match_all_words = True)
