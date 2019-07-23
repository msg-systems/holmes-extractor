import unittest
import holmes_extractor as holmes

holmes_manager = holmes.Manager('de_core_news_md')

class GermanPhraseletProductionTest(unittest.TestCase):

    def _check_equals(self, text_to_match, phraselet_labels, match_all_words = False):
        holmes_manager.remove_all_search_phrases()
        doc = holmes_manager.semantic_analyzer.parse(text_to_match)
        holmes_manager.structural_matcher.register_phraselets(doc,
                replace_with_hypernym_ancestors=False,
                match_all_words=match_all_words,
                returning_serialized_phraselets=False)
        self.assertEqual(
                set(holmes_manager.structural_matcher.list_search_phrase_labels()),
                set(phraselet_labels))
        self.assertEqual(len(holmes_manager.structural_matcher.list_search_phrase_labels()),
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
        holmes_manager.remove_all_search_phrases()
        doc = holmes_manager.semantic_analyzer.parse(
                "Der Gärtner gibt der netten Frau ihr Mittagessen")
        serialized_phraselets = holmes_manager.structural_matcher.register_phraselets(doc,
                replace_with_hypernym_ancestors=False,
                match_all_words=False,
                returning_serialized_phraselets=True)
        holmes_manager.remove_all_search_phrases()
        holmes_manager.structural_matcher.register_serialized_phraselets(serialized_phraselets)
        self.assertEqual(set(holmes_manager.structural_matcher.list_search_phrase_labels()),
                set(['verb-nom: geben-gärtnern', 'verb-acc: geben-mittagessen',
                'verb-dat: geben-frau', 'noun-dependent: frau-nett',
                'word: gärtnern', 'word: frau', 'word: mittagessen']))

    def test_only_verb(self):
        self._check_equals("springen", ['word: springen'])

    def test_only_preposition(self):
        self._check_equals("unter", ['word: unter'])

    def test_match_all_words(self):
        self._check_equals("Der Gärtner gibt der netten Frau ihr Mittagessen",
                ['word: gärtnern', 'word: frau', 'word: mittagessen',
                'word: geben', 'word: nett'], True)
