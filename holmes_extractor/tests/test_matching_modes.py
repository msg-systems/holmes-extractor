import unittest
import os
import holmes_extractor as holmes
from holmes_extractor.tests.testing_utils import HolmesInstanceManager

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
holmes_manager = HolmesInstanceManager(ontology).en_core_web_lg

class MatchingModesTest(unittest.TestCase):

    def _register_multiple_documents_and_search_phrases(self):
        holmes_manager.remove_all_search_phrases()
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(document_text=
                "All the time I am testing here, dogs keep on chasing cats.", label='pets')
        holmes_manager.parse_and_register_document(document_text=
                "Everything I know suggests that lions enjoy eating wildebeest", label='safari')
        holmes_manager.register_search_phrase("A dog chases a cat", label="test")
        holmes_manager.register_search_phrase("A lion eats a wildebeest", label="test")
        return

    def test_multiple(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match_returning_dictionaries()), 2)

    def test_remove_all_search_phrases(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.remove_all_search_phrases()
        holmes_manager.register_search_phrase("A dog chases a cat")
        self.assertEqual(len(holmes_manager.match_returning_dictionaries()), 1)

    def test_remove_all_documents(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(document_text=
                "All the time I am testing here, dogs keep on chasing cats.", label='pets')
        self.assertEqual(len(holmes_manager.match_returning_dictionaries()), 1)

    def test_remove_document(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.remove_document(label='pets')
        holmes_manager.remove_document(label='safari')
        holmes_manager.parse_and_register_document(document_text=
                "All the time I am testing here, dogs keep on chasing cats.", label='pets')
        self.assertEqual(len(holmes_manager.match_returning_dictionaries()), 1)

    def test_match_search_phrases_against(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match_search_phrases_against(
                "All the time I am testing here, dogs keep on chasing cats.")), 1)

    def test_match_documents_against(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match_documents_against(
                "A lion eats a wildebeest.")), 1)

    def test_get_labels(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(holmes_manager.structural_matcher.list_search_phrase_labels(),
                ['test'])

    def test_remove_all_search_phrases_with_label(self):
        holmes_manager.remove_all_search_phrases()
        holmes_manager.register_search_phrase("testa", label="test1")
        holmes_manager.register_search_phrase("testb", label="test1")
        holmes_manager.register_search_phrase("testc", label="test2")
        holmes_manager.register_search_phrase("testd", label="test2")
        holmes_manager.remove_all_search_phrases_with_label("test2")
        holmes_manager.remove_all_search_phrases_with_label("testb")
        self.assertEqual(holmes_manager.structural_matcher.list_search_phrase_labels(),
                ['test1'])
        self.assertEqual(len(holmes_manager.match_search_phrases_against(
                "testa")), 1)
        self.assertEqual(len(holmes_manager.match_search_phrases_against(
                "testb")), 1)
        self.assertEqual(len(holmes_manager.match_search_phrases_against(
                "testc")), 0)
        self.assertEqual(len(holmes_manager.match_search_phrases_against(
                "testd")), 0)
