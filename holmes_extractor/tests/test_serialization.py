import unittest
import os
import holmes_extractor as holmes
from holmes_extractor.tests.testing_utils import HolmesInstanceManager

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
holmes_manager = HolmesInstanceManager(ontology).en_core_web_lg
holmes_manager.register_search_phrase("A dog chases a cat")

class SerializationTest(unittest.TestCase):

    def test_matching_with_holmes_manager_document_after_serialization(self):
        holmes_manager.remove_all_documents()
        doc = holmes_manager.parse_and_register_document("The cat was chased by the dog", 'pets')
        serialized_doc = holmes_manager.serialize_document('pets')
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_matching_with_reserialized_holmes_manager_document(self):
        holmes_manager.remove_all_documents()
        doc = holmes_manager.parse_and_register_document("The cat was chased by the dog", 'pets')
        serialized_doc = holmes_manager.serialize_document('pets')
        holmes_manager.remove_all_documents()
        new_doc = holmes_manager.deserialize_and_register_document(serialized_doc, 'pets')
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_matching_with_both_documents(self):
        holmes_manager.remove_all_documents()
        doc = holmes_manager.parse_and_register_document("The cat was chased by the dog", 'pets')
        serialized_doc = holmes_manager.serialize_document('pets')
        new_doc = holmes_manager.deserialize_and_register_document(serialized_doc, 'pets2')
        self.assertEqual(len(holmes_manager.match()), 2)

    def test_document_to_serialize_does_not_exist(self):
        holmes_manager.remove_all_documents()
        serialized_doc = holmes_manager.serialize_document('pets')
        self.assertEqual(serialized_doc, None)
