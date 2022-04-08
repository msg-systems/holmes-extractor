import unittest
import holmes_extractor as holmes

holmes_manager = holmes.Manager(
    'en_core_web_trf', perform_coreference_resolution=False, number_of_workers=2)

class ManagerTest(unittest.TestCase):

    def _register_multiple_documents_and_search_phrases(self):
        holmes_manager.remove_all_search_phrases()
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets')
        holmes_manager.parse_and_register_document(
            document_text="Everything I know suggests that lions enjoy eating gnu", label='safari')
        holmes_manager.register_search_phrase(
            "A dog chases a cat", label="test")
        holmes_manager.register_search_phrase(
            "A lion eats a gnu", label="test")
        holmes_manager.register_search_phrase(
            "irrelevancy", label="alpha")
        return

    def test_multiple(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match()), 2)

    def test_remove_all_search_phrases(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.remove_all_search_phrases()
        holmes_manager.register_search_phrase("A dog chases a cat")
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_remove_all_documents(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets')
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_remove_document(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets2')
        self.assertEqual(len(holmes_manager.match()), 3)
        holmes_manager.remove_document(label='pets')
        holmes_manager.remove_document(label='safari')
        matches = holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['document'], 'pets2')

    def test_match_search_phrases_against(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match(document_text=
            "All the time I am testing here, dogs keep on chasing cats.")), 1)

    def test_match_documents_against(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match(search_phrase_text=
            "A lion eats a gnu.")), 1)

    def test_match_documents_and_search_phrases_against(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match(search_phrase_text= "burn",
            document_text="Burn. Everything I know suggests that lions enjoy eating gnu")), 1)
        holmes_manager.remove_all_documents()
        holmes_manager.remove_all_search_phrases()
        self.assertEqual(len(holmes_manager.match(search_phrase_text= "burn",
            document_text="Burn. Everything I know suggests that lions enjoy eating gnu")), 1)

    def test_get_labels(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(holmes_manager.list_search_phrase_labels(),
                         ['alpha', 'test'])

    def test_get_document(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(holmes_manager.get_document('safari')[5]._.holmes.lemma,
                         'lion')

    def test_remove_all_search_phrases_with_label(self):
        holmes_manager.remove_all_search_phrases()
        holmes_manager.register_search_phrase("testa", label="test1")
        holmes_manager.register_search_phrase("testb", label="test1")
        holmes_manager.register_search_phrase("testc", label="test2")
        holmes_manager.register_search_phrase("testd", label="test2")
        holmes_manager.remove_all_search_phrases_with_label("test2")
        holmes_manager.remove_all_search_phrases_with_label("testb")
        self.assertEqual(holmes_manager.list_search_phrase_labels(),
                         ['test1'])
        self.assertEqual(len(holmes_manager.match(document_text=
            "testa")), 1)
        self.assertEqual(len(holmes_manager.match(document_text=
            "testb")), 1)
        self.assertEqual(len(holmes_manager.match(document_text=
            "testc")), 0)
        self.assertEqual(len(holmes_manager.match(document_text=
            "testd")), 0)
