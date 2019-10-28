import unittest
import holmes_extractor as holmes
import os
import time
from threading import Thread
from queue import Queue
from time import sleep

NUMBER_OF_THREADS = 50

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))

class MultiprocessingTest(unittest.TestCase):
    # We use en_core_web_sm to prevent memory exhaustion during the tests.

    def test_workers_specified(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology, number_of_workers=2,
                verbose=False)
        m.parse_and_register_documents({'specific' : "I saw a dog. It was chasing a cat",
                'exact': "The dog chased the animal",
                'specific-reversed': "The cat chased the dog",
                'exact-reversed': "The animal chased the dog"})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
                "A dog chases an animal"),
                [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 99.34666666666668, 'word_infos': [[4, 7, 'overlapping_relation', False], [8, 14, 'overlapping_relation', False], [19, 25, 'overlapping_relation', True]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 99.14666666666669, 'word_infos': [[8, 11, 'overlapping_relation', False], [20, 27, 'overlapping_relation', False], [30, 33, 'overlapping_relation', True]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 40.946666666666665, 'word_infos': [[4, 10, 'single', False], [11, 17, 'relation', False], [22, 25, 'relation', True]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 40.946666666666665, 'word_infos': [[4, 7, 'single', False], [8, 14, 'relation', False], [19, 22, 'relation', True]]}])
        m.close()

    def test_workers_not_specified(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology)
        m.parse_and_register_documents({'specific' : "I saw a dog. It was chasing a cat",
                'exact': "The dog chased the animal",
                'specific-reversed': "The cat chased the dog",
                'exact-reversed': "The animal chased the dog"})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
                "A dog chases an animal"),
                [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 99.34666666666668, 'word_infos': [[4, 7, 'overlapping_relation', False], [8, 14, 'overlapping_relation', False], [19, 25, 'overlapping_relation', True]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 99.14666666666669, 'word_infos': [[8, 11, 'overlapping_relation', False], [20, 27, 'overlapping_relation', False], [30, 33, 'overlapping_relation', True]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 40.946666666666665, 'word_infos': [[4, 10, 'single', False], [11, 17, 'relation', False], [22, 25, 'relation', True]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 40.946666666666665, 'word_infos': [[4, 7, 'single', False], [8, 14, 'relation', False], [19, 22, 'relation', True]]}])
        m.close()

    def test_deserialized_documents(self):
        normal_manager = holmes.Manager('en_core_web_sm', perform_coreference_resolution=False)
        normal_manager.parse_and_register_document("I saw a dog. It was chasing a cat", 'specific')
        normal_manager.parse_and_register_document("The dog chased the animal", 'exact')
        normal_manager.parse_and_register_document("The cat chased the dog", 'specific-reversed')
        normal_manager.parse_and_register_document("The animal chased the dog", 'exact-reversed')
        specific = normal_manager.serialize_document('specific')
        exact = normal_manager.serialize_document('exact')
        specific_reversed = normal_manager.serialize_document('specific-reversed')
        exact_reversed = normal_manager.serialize_document('exact-reversed')
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology, number_of_workers=2,
                verbose=False, perform_coreference_resolution=False)
        m.deserialize_and_register_documents({'specific' : specific,
                'exact': exact,
                'specific-reversed': specific_reversed,
                'exact-reversed': exact_reversed})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
                "A dog chases an animal"),
                [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 99.34666666666668, 'word_infos': [[4, 7, 'overlapping_relation', False], [8, 14, 'overlapping_relation', False], [19, 25, 'overlapping_relation', True]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '2=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 40.946666666666665, 'word_infos': [[4, 10, 'single', False], [11, 17, 'relation', False], [22, 25, 'relation', True]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '2=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 40.946666666666665, 'word_infos': [[4, 7, 'single', False], [8, 14, 'relation', False], [19, 22, 'relation', True]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '2=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 40.74666666666667, 'word_infos': [[8, 11, 'single', False], [20, 27, 'relation', False], [30, 33, 'relation', True]]}])
        m.close()

    def test_number_of_results(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology, number_of_workers=2,
                verbose=False)
        m.parse_and_register_documents({'specific' : "I saw a dog. It was chasing a cat",
                'exact': "The dog chased the animal",
                'specific-reversed': "The cat chased the dog",
                'exact-reversed': "The animal chased the dog"})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
                "A dog chases an animal", number_of_results=3),
                [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 99.34666666666668, 'word_infos': [[4, 7, 'overlapping_relation', False], [8, 14, 'overlapping_relation', False], [19, 25, 'overlapping_relation', True]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 99.14666666666669, 'word_infos': [[8, 11, 'overlapping_relation', False], [20, 27, 'overlapping_relation', False], [30, 33, 'overlapping_relation', True]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 40.946666666666665, 'word_infos': [[4, 10, 'single', False], [11, 17, 'relation', False], [22, 25, 'relation', True]]}])
        m.close()

    def test_parsed_document_registration_multithreaded(self):

        def add_document(counter):
            m.parse_and_register_documents({' '.join(('Irrelevant', str(counter))):
                    "People discuss irrelevancies"})

        m = holmes.MultiprocessingManager('en_core_web_sm', number_of_workers=4)

        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=add_document, args=(i,))
            t.start()

        last_number_of_matches = 0
        for counter in range(50):
            document_labels = m.document_labels()
            for label in document_labels:
                self.assertTrue(label.startswith("Irrelevant"))
            if len(document_labels) == NUMBER_OF_THREADS:
                break
            self.assertFalse(counter == 49)
            sleep(0.5)

    def test_deserialized_document_registration_multithreaded(self):

        def add_document(counter):
            m.deserialize_and_register_documents({' '.join(('Irrelevant', str(counter))):
                    irrelevant_doc})

        normal_m = holmes.Manager('en_core_web_sm', perform_coreference_resolution=False)
        normal_m.parse_and_register_document("People discuss irrelevancies", 'irrelevant')
        irrelevant_doc = normal_m.serialize_document('irrelevant')
        m = holmes.MultiprocessingManager('en_core_web_sm', number_of_workers=4,
                perform_coreference_resolution=False)

        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=add_document, args=(i,))
            t.start()

        last_number_of_matches = 0
        for counter in range(50):
            document_labels = m.document_labels()
            for label in document_labels:
                self.assertTrue(label.startswith("Irrelevant"))
            if len(document_labels) == NUMBER_OF_THREADS:
                break
            self.assertFalse(counter == 49)
            sleep(0.5)

    def _internal_test_multithreading_topic_matching(self, number_of_workers):

        def topic_match_within_thread():
            normal_dict = m.topic_match_documents_returning_dictionaries_against(
                    "A dog chases an animal")
            reversed_dict = m.topic_match_documents_returning_dictionaries_against(
                    "The animal chased the dog")
            queue.put((normal_dict, reversed_dict))

        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology,
                number_of_workers=number_of_workers, verbose=False)
        m.parse_and_register_documents({'specific' : "I saw a dog. It was chasing a cat",
                'exact': "The dog chased the animal",
                'specific-reversed': "The cat chased the dog",
                'exact-reversed': "The animal chased the dog"})
        queue = Queue()
        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=topic_match_within_thread)
            t.start()
        for i in range(NUMBER_OF_THREADS):
            normal_dict, reversed_dict = queue.get(True,30)
            self.assertEqual(normal_dict, [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 99.34666666666668, 'word_infos': [[4, 7, 'overlapping_relation', False], [8, 14, 'overlapping_relation', False], [19, 25, 'overlapping_relation', True]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 99.14666666666669, 'word_infos': [[8, 11, 'overlapping_relation', False], [20, 27, 'overlapping_relation', False], [30, 33, 'overlapping_relation', True]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 40.946666666666665, 'word_infos': [[4, 10, 'single', False], [11, 17, 'relation', False], [22, 25, 'relation', True]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 40.946666666666665, 'word_infos': [[4, 7, 'single', False], [8, 14, 'relation', False], [19, 22, 'relation', True]]}])
            self.assertEqual(reversed_dict, [{'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'The animal chased the dog', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 96.93333333333334, 'word_infos': [[4, 10, 'overlapping_relation', False], [11, 17, 'overlapping_relation', True], [22, 25, 'overlapping_relation', False]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'The animal chased the dog', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 96.93333333333334, 'word_infos': [[4, 7, 'overlapping_relation', False], [8, 14, 'overlapping_relation', True], [19, 22, 'overlapping_relation', False]]}, {'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'The animal chased the dog', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 36.93333333333334, 'word_infos': [[4, 7, 'relation', False], [8, 14, 'relation', True], [19, 25, 'single', False]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'The animal chased the dog', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 36.733333333333334, 'word_infos': [[8, 11, 'relation', False], [20, 27, 'relation', True], [30, 33, 'single', False]]}])

    def test_multithreading_topic_matching_with_2_workers(self):
        self._internal_test_multithreading_topic_matching(2)

    def test_multithreading_topic_matching_with_4_workers(self):
        self._internal_test_multithreading_topic_matching(4)

    def test_multithreading_topic_matching_with_8_workers(self):
        self._internal_test_multithreading_topic_matching(8)
