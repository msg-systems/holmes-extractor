import unittest
import os
import holmes_extractor as holmes

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join(
    (script_directory, 'test_ontology.owl')))
nocoref_holmes_manager = holmes.Manager('en_core_web_lg',
                                        perform_coreference_resolution=False)
nocoref_holmes_manager.register_search_phrase("A dog chases a cat")
german_holmes_manager = holmes.Manager('de_core_news_md')


class SerializationTest(unittest.TestCase):

    def test_matching_with_nocoref_holmes_manager_document_after_serialization(self):
        nocoref_holmes_manager.remove_all_documents()
        nocoref_holmes_manager.parse_and_register_document(
            "The cat was chased by the dog", 'pets')
        serialized_doc = nocoref_holmes_manager.serialize_document('pets')
        self.assertEqual(len(nocoref_holmes_manager.match()), 1)

    def test_matching_with_reserialized_nocoref_holmes_manager_document(self):
        nocoref_holmes_manager.remove_all_documents()
        nocoref_holmes_manager.parse_and_register_document(
            "The cat was chased by the dog", 'pets')
        serialized_doc = nocoref_holmes_manager.serialize_document('pets')
        nocoref_holmes_manager.remove_all_documents()
        nocoref_holmes_manager.deserialize_and_register_document(
            serialized_doc, 'pets')
        self.assertEqual(len(nocoref_holmes_manager.match()), 1)

    def test_matching_with_both_documents(self):
        nocoref_holmes_manager.remove_all_documents()
        nocoref_holmes_manager.parse_and_register_document(
            "The cat was chased by the dog", 'pets')
        serialized_doc = nocoref_holmes_manager.serialize_document('pets')
        nocoref_holmes_manager.deserialize_and_register_document(
            serialized_doc, 'pets2')
        self.assertEqual(len(nocoref_holmes_manager.match()), 2)

    def test_document_to_serialize_does_not_exist(self):
        nocoref_holmes_manager.remove_all_documents()
        serialized_doc = nocoref_holmes_manager.serialize_document('pets')
        self.assertEqual(serialized_doc, None)

    def test_matching_with_both_documents(self):
        nocoref_holmes_manager.remove_all_documents()
        nocoref_holmes_manager.parse_and_register_document(
            "The cat was chased by the dog", 'pets')
        serialized_doc = nocoref_holmes_manager.serialize_document('pets')
        nocoref_holmes_manager.deserialize_and_register_document(
            serialized_doc, 'pets2')
        self.assertEqual(len(nocoref_holmes_manager.match()), 2)

    def test_parent_token_indexes(self):
        nocoref_holmes_manager.remove_all_documents()
        nocoref_holmes_manager.parse_and_register_document(
            "Houses in the village.", 'village')
        serialized_doc = nocoref_holmes_manager.serialize_document('village')
        nocoref_holmes_manager.deserialize_and_register_document(
            serialized_doc, 'village2')
        old_doc = nocoref_holmes_manager.threadsafe_container.get_document(
            'village')
        new_doc = nocoref_holmes_manager.threadsafe_container.get_document(
            'village2')
        self.assertEqual(old_doc[0]._.holmes.string_representation_of_children(),
                         '1:prep; 3:pobjp')
        self.assertEqual(old_doc[3]._.holmes.parent_dependencies, [
                         [0, 'pobjp'], [1, 'pobj']])
        self.assertEqual(new_doc[0]._.holmes.string_representation_of_children(),
                         '1:prep; 3:pobjp')
        self.assertEqual(new_doc[3]._.holmes.parent_dependencies, [
                         [0, 'pobjp'], [1, 'pobj']])

    def test_subwords(self):
        german_holmes_manager.remove_all_documents()
        german_holmes_manager.parse_and_register_document(
            "Bundesoberbehörde.", 'bo')
        serialized_doc = german_holmes_manager.serialize_document('bo')
        german_holmes_manager.deserialize_and_register_document(
            serialized_doc, 'bo2')
        old_doc = german_holmes_manager.threadsafe_container.get_document('bo')
        new_doc = german_holmes_manager.threadsafe_container.get_document(
            'bo2')
        self.assertEqual(old_doc[0]._.holmes.subwords[0].text, 'Bundes')
        self.assertEqual(old_doc[0]._.holmes.subwords[0].lemma, 'bund')
        self.assertEqual(old_doc[0]._.holmes.subwords[1].text, 'oberbehörde')
        self.assertEqual(old_doc[0]._.holmes.subwords[1].lemma, 'oberbehörde')
        self.assertEqual(new_doc[0]._.holmes.subwords[0].text, 'Bundes')
        self.assertEqual(new_doc[0]._.holmes.subwords[0].lemma, 'bund')
        self.assertEqual(new_doc[0]._.holmes.subwords[1].text, 'oberbehörde')
        self.assertEqual(new_doc[0]._.holmes.subwords[1].lemma, 'oberbehörde')

    def test_derived_lemma(self):
        nocoref_holmes_manager.remove_all_documents()
        nocoref_holmes_manager.parse_and_register_document(
            "A lot of information.", 'information')
        serialized_doc = nocoref_holmes_manager.serialize_document(
            'information')
        nocoref_holmes_manager.deserialize_and_register_document(
            serialized_doc, 'information2')
        old_doc = nocoref_holmes_manager.threadsafe_container.get_document(
            'information')
        new_doc = nocoref_holmes_manager.threadsafe_container.get_document(
            'information2')
        self.assertEqual(old_doc[3]._.holmes.derived_lemma, 'inform')
        self.assertEqual(new_doc[3]._.holmes.derived_lemma, 'inform')
