import unittest
import holmes_extractor as holmes

holmes_manager = holmes.Manager('de_core_news_md')

class GermanPhraseletProductionTest(unittest.TestCase):

    def _check_equals(self, text_to_match, phraselet_labels, match_all_words = False):
        doc = holmes_manager.semantic_analyzer.parse(text_to_match)
        phraselet_labels_to_search_phrases = {}
        holmes_manager.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                replace_with_hypernym_ancestors=False,
                match_all_words=match_all_words,
                returning_serialized_phraselets=False)
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
                returning_serialized_phraselets=True)
        deserialized_phraselet_labels_to_search_phrases = holmes_manager.structural_matcher.\
                deserialize_phraselets(serialized_phraselets)
        self.assertEqual(set(deserialized_phraselet_labels_to_search_phrases.keys()),
                set(['verb-nom: geben-gärtnern', 'verb-acc: geben-mittagessen',
                'verb-dat: geben-frau', 'noun-dependent: frau-nett',
                'word: gärtnern', 'word: frau', 'word: mittagessen']))

    def test_phraselet_stop_words_governor(self):
        self._check_equals("Immer hatte er es", ['word: immer'])

    def test_phraselet_stop_words_governed(self):
        self._check_equals("Dann machte er es zu Hause", ['word: hausen',
                'prepgovernor-noun: machen-hausen'])

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
