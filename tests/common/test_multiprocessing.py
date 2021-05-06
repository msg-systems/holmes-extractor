import unittest
import holmes_extractor as holmes
import os
import time
from threading import Thread
from queue import Queue
from time import sleep

NUMBER_OF_THREADS = 50

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join(
    (script_directory, 'test_ontology.owl')))


class MultiprocessingTest(unittest.TestCase):
    # We use en_core_web_sm to prevent memory exhaustion during the tests.

    def test_workers_specified(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology, number_of_workers=2,
                                          verbose=False)
        m.parse_and_register_documents({'specific': "I saw a dog. It was chasing a cat",
                                        'exact': "The dog chased the animal",
                                        'specific-reversed': "The cat chased the dog",
                                        'exact-reversed': "The animal chased the dog"})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                                               'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
            "A dog chases an animal",
            relation_score=30,
            reverse_only_relation_score=20,
            single_word_score=5,
            single_word_any_tag_score=2,
            different_match_cutoff_score=5),
            [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 44.27418511677267, 'word_infos': [[4, 7, 'overlapping_relation', False, "Matches DOG directly."], [8, 14, 'overlapping_relation', False, "Matches CHASE directly."], [19, 25, 'overlapping_relation', True, "Matches ANIMAL directly."]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '2', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 35.641203421657025, 'word_infos': [[8, 11, 'overlapping_relation', False, "Matches DOG directly."], [20, 27, 'overlapping_relation', False, "Is a synonym of CHASE in the ontology."], [30, 33, 'overlapping_relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 19.954440383093182, 'word_infos': [[4, 10, 'single', False, "Matches ANIMAL directly."], [11, 17, 'relation', False, "Matches CHASE directly."], [22, 25, 'relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 19.302449037645065, 'word_infos': [[4, 7, 'single', False, "Is a child of ANIMAL in the ontology."], [8, 14, 'relation', False, "Matches CHASE directly."], [19, 22, 'relation', True, "Is a child of ANIMAL in the ontology."]]}])
        m.close()

    def test_workers_specified_trf(self):
        m = holmes.MultiprocessingManager('en_core_web_trf', ontology=ontology, number_of_workers=2,
                                          verbose=False)
        m.parse_and_register_documents({'specific': "I saw a dog. It was chasing a cat",
                                        'exact': "The dog chased the animal",
                                        'specific-reversed': "The cat chased the dog",
                                        'exact-reversed': "The animal chased the dog"})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                                               'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
            "A dog chases an animal",
            relation_score=30,
            reverse_only_relation_score=20,
            single_word_score=5,
            single_word_any_tag_score=2,
            different_match_cutoff_score=5),
            [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 44.27418511677267, 'word_infos': [[4, 7, 'overlapping_relation', False, "Matches DOG directly."], [8, 14, 'overlapping_relation', False, "Matches CHASE directly."], [19, 25, 'overlapping_relation', True, "Matches ANIMAL directly."]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '2', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 35.641203421657025, 'word_infos': [[8, 11, 'overlapping_relation', False, "Matches DOG directly."], [20, 27, 'overlapping_relation', False, "Is a synonym of CHASE in the ontology."], [30, 33, 'overlapping_relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 19.954440383093182, 'word_infos': [[4, 10, 'single', False, "Matches ANIMAL directly."], [11, 17, 'relation', False, "Matches CHASE directly."], [22, 25, 'relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 19.302449037645065, 'word_infos': [[4, 7, 'single', False, "Is a child of ANIMAL in the ontology."], [8, 14, 'relation', False, "Matches CHASE directly."], [19, 22, 'relation', True, "Is a child of ANIMAL in the ontology."]]}])
        m.close()

    def test_workers_specified_one_worker_frequency_factor(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology, number_of_workers=1,
                                          verbose=False)
        m.parse_and_register_documents({'specific': "I saw a dog. It was chasing a cat",
                                        'exact': "The dog chased the animal",
                                        'specific-reversed': "The cat chased the dog",
                                        'exact-reversed': "The animal chased the dog"})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                                               'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
            "A dog chases an animal",
            relation_score=30,
            reverse_only_relation_score=20,
            single_word_score=5,
            single_word_any_tag_score=2,
            different_match_cutoff_score=5),
            [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 44.27418511677267, 'word_infos': [[4, 7, 'overlapping_relation', False, "Matches DOG directly."], [8, 14, 'overlapping_relation', False, "Matches CHASE directly."], [19, 25, 'overlapping_relation', True, "Matches ANIMAL directly."]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '2', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 35.641203421657025, 'word_infos': [[8, 11, 'overlapping_relation', False, "Matches DOG directly."], [20, 27, 'overlapping_relation', False, "Is a synonym of CHASE in the ontology."], [30, 33, 'overlapping_relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 19.954440383093182, 'word_infos': [[4, 10, 'single', False, "Matches ANIMAL directly."], [11, 17, 'relation', False, "Matches CHASE directly."], [22, 25, 'relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 19.302449037645065, 'word_infos': [[4, 7, 'single', False, "Is a child of ANIMAL in the ontology."], [8, 14, 'relation', False, "Matches CHASE directly."], [19, 22, 'relation', True, "Is a child of ANIMAL in the ontology."]]}])
        m.close()

    def test_workers_specified_two_workers_frequency_factor_control(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology, number_of_workers=2,
                                          verbose=False)
        m.parse_and_register_documents({'specific': "I saw a dog. It was chasing a cat",
                                        'exact': "The dog chased the animal",
                                        'specific-reversed': "The cat chased the dog",
                                        'exact-reversed': "The animal chased the dog"})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                                               'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
            "A dog chases an animal", use_frequency_factor=False,
            relation_score=30,
            reverse_only_relation_score=20,
            single_word_score=5,
            single_word_any_tag_score=2,
            different_match_cutoff_score=5),
            [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 99.34666666666668, 'word_infos': [[4, 7, 'overlapping_relation', False, "Matches DOG directly."], [8, 14, 'overlapping_relation', False, "Matches CHASE directly."], [19, 25, 'overlapping_relation', True, "Matches ANIMAL directly."]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '2', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 81.94686666666668, 'word_infos': [[8, 11, 'overlapping_relation', False, "Matches DOG directly."], [20, 27, 'overlapping_relation', False, "Is a synonym of CHASE in the ontology."], [30, 33, 'overlapping_relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 35.39866666666667, 'word_infos': [[4, 10, 'single', False, "Matches ANIMAL directly."], [11, 17, 'relation', False, "Matches CHASE directly."], [22, 25, 'relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 34.486666666666665, 'word_infos': [[4, 7, 'single', False, "Is a child of ANIMAL in the ontology."], [8, 14, 'relation', False, "Matches CHASE directly."], [19, 22, 'relation', True, "Is a child of ANIMAL in the ontology."]]}])
        m.close()

    def test_workers_not_specified(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology)
        m.parse_and_register_documents({'specific': "I saw a dog. It was chasing a cat",
                                        'exact': "The dog chased the animal",
                                        'specific-reversed': "The cat chased the dog",
                                        'exact-reversed': "The animal chased the dog"})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                                               'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
            "A dog chases an animal",
            relation_score=30,
            reverse_only_relation_score=20,
            single_word_score=5,
            single_word_any_tag_score=2,
            different_match_cutoff_score=5),
            [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 44.27418511677267, 'word_infos': [[4, 7, 'overlapping_relation', False, "Matches DOG directly."], [8, 14, 'overlapping_relation', False, "Matches CHASE directly."], [19, 25, 'overlapping_relation', True, "Matches ANIMAL directly."]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '2', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 35.641203421657025, 'word_infos': [[8, 11, 'overlapping_relation', False, "Matches DOG directly."], [20, 27, 'overlapping_relation', False, "Is a synonym of CHASE in the ontology."], [30, 33, 'overlapping_relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 19.954440383093182, 'word_infos': [[4, 10, 'single', False, "Matches ANIMAL directly."], [11, 17, 'relation', False, "Matches CHASE directly."], [22, 25, 'relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 19.302449037645065, 'word_infos': [[4, 7, 'single', False, "Is a child of ANIMAL in the ontology."], [8, 14, 'relation', False, "Matches CHASE directly."], [19, 22, 'relation', True, "Is a child of ANIMAL in the ontology."]]}])
        m.close()

    def test_deserialized_documents(self):
        normal_manager = holmes.Manager(
            'en_core_web_sm', perform_coreference_resolution=False)
        normal_manager.parse_and_register_document(
            "I saw a dog. It was chasing a cat", 'specific')
        normal_manager.parse_and_register_document(
            "The dog chased the animal", 'exact')
        normal_manager.parse_and_register_document(
            "The cat chased the dog", 'specific-reversed')
        normal_manager.parse_and_register_document(
            "The animal chased the dog", 'exact-reversed')
        specific = normal_manager.serialize_document('specific')
        exact = normal_manager.serialize_document('exact')
        specific_reversed = normal_manager.serialize_document(
            'specific-reversed')
        exact_reversed = normal_manager.serialize_document('exact-reversed')
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology, number_of_workers=2,
                                          verbose=False, perform_coreference_resolution=False)
        m.deserialize_and_register_documents({'specific': specific,
                                              'exact': exact,
                                              'specific-reversed': specific_reversed,
                                              'exact-reversed': exact_reversed})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                                               'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
            "A dog chases an animal",
            relation_score=30,
            reverse_only_relation_score=20,
            single_word_score=5,
            single_word_any_tag_score=2,
            different_match_cutoff_score=5),
            [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 44.27418511677267, 'word_infos': [[4, 7, 'overlapping_relation', False, "Matches DOG directly."], [8, 14, 'overlapping_relation', False, "Matches CHASE directly."], [19, 25, 'overlapping_relation', True, "Matches ANIMAL directly."]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '2=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 19.954440383093182, 'word_infos': [[4, 10, 'single', False, "Matches ANIMAL directly."], [11, 17, 'relation', False, "Matches CHASE directly."], [22, 25, 'relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '2=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 19.302449037645065, 'word_infos': [[4, 7, 'single', False, "Is a child of ANIMAL in the ontology."], [8, 14, 'relation', False, "Matches CHASE directly."], [19, 22, 'relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '4', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 17.955272564870686, 'word_infos': [[8, 11, 'single', False, "Matches DOG directly."], [20, 27, 'relation', False, "Is a synonym of CHASE in the ontology."], [30, 33, 'relation', True, "Is a child of ANIMAL in the ontology."]]}])
        m.close()

    def test_number_of_results(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology, number_of_workers=2,
                                          verbose=False)
        m.parse_and_register_documents({'specific': "I saw a dog. It was chasing a cat",
                                        'exact': "The dog chased the animal",
                                        'specific-reversed': "The cat chased the dog",
                                        'exact-reversed': "The animal chased the dog"})
        self.assertEqual(m.document_labels(), ['exact', 'exact-reversed', 'specific',
                                               'specific-reversed'])
        self.assertEqual(m.topic_match_documents_returning_dictionaries_against(
            "A dog chases an animal", number_of_results=3, use_frequency_factor=False,
            relation_score=30,
            reverse_only_relation_score=20,
            single_word_score=5,
            single_word_any_tag_score=2,
            different_match_cutoff_score=5),
            [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 99.34666666666668, 'word_infos': [[4, 7, 'overlapping_relation', False, "Matches DOG directly."], [8, 14, 'overlapping_relation', False, "Matches CHASE directly."], [19, 25, 'overlapping_relation', True, "Matches ANIMAL directly."]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '2', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 81.94686666666668, 'word_infos': [[8, 11, 'overlapping_relation', False, "Matches DOG directly."], [20, 27, 'overlapping_relation', False, "Is a synonym of CHASE in the ontology."], [30, 33, 'overlapping_relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 35.39866666666667, 'word_infos': [[4, 10, 'single', False, "Matches ANIMAL directly."], [11, 17, 'relation', False, "Matches CHASE directly."], [22, 25, 'relation', True, "Is a child of ANIMAL in the ontology."]]}])
        m.close()

    def test_parsed_document_registration_multithreaded(self):

        def add_document(counter):
            m.parse_and_register_documents({' '.join(('Irrelevant', str(counter))):
                                            "People discuss irrelevancies"})

        m = holmes.MultiprocessingManager(
            'en_core_web_sm', number_of_workers=4)

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

        normal_m = holmes.Manager(
            'en_core_web_sm', perform_coreference_resolution=False)
        normal_m.parse_and_register_document(
            "People discuss irrelevancies", 'irrelevant')
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
        m.close()

    def _internal_test_multithreading_topic_matching(self, number_of_workers):

        def topic_match_within_thread():
            normal_dict = m.topic_match_documents_returning_dictionaries_against(
                "A dog chases an animal", use_frequency_factor=False,
                relation_score=30,
                reverse_only_relation_score=20,
                single_word_score=5,
                single_word_any_tag_score=2,
                different_match_cutoff_score=5)
            reversed_dict = m.topic_match_documents_returning_dictionaries_against(
                "The animal chased the dog", use_frequency_factor=False,
                relation_score=30,
                reverse_only_relation_score=20,
                single_word_score=5,
                single_word_any_tag_score=2,
                different_match_cutoff_score=5)
            queue.put((normal_dict, reversed_dict))

        m = holmes.MultiprocessingManager('en_core_web_sm', ontology=ontology,
                                          number_of_workers=number_of_workers, verbose=False)
        m.parse_and_register_documents({'specific': "I saw a dog. It was chasing a cat",
                                        'exact': "The dog chased the animal",
                                        'specific-reversed': "The cat chased the dog",
                                        'exact-reversed': "The animal chased the dog"})
        queue = Queue()
        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=topic_match_within_thread)
            t.start()
        for i in range(NUMBER_OF_THREADS):
            normal_dict, reversed_dict = queue.get(True, 60)
            self.assertEqual(normal_dict, [{'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'A dog chases an animal', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 99.34666666666668, 'word_infos': [[4, 7, 'overlapping_relation', False, "Matches DOG directly."], [8, 14, 'overlapping_relation', False, "Matches CHASE directly."], [19, 25, 'overlapping_relation', True, "Matches ANIMAL directly."]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'A dog chases an animal', 'rank': '2', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 81.94686666666668, 'word_infos': [[8, 11, 'overlapping_relation', False, "Matches DOG directly."], [20, 27, 'overlapping_relation', False, "Is a synonym of CHASE in the ontology."], [30, 33, 'overlapping_relation', True, "Is a child of ANIMAL in the ontology."]]}, {
                             'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 35.39866666666667, 'word_infos': [[4, 10, 'single', False, "Matches ANIMAL directly."], [11, 17, 'relation', False, "Matches CHASE directly."], [22, 25, 'relation', True, "Is a child of ANIMAL in the ontology."]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'A dog chases an animal', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 34.486666666666665, 'word_infos': [[4, 7, 'single', False, "Is a child of ANIMAL in the ontology."], [8, 14, 'relation', False, "Matches CHASE directly."], [19, 22, 'relation', True, "Is a child of ANIMAL in the ontology."]]}])
            self.assertEqual(reversed_dict, [{'document_label': 'exact-reversed', 'text': 'The animal chased the dog', 'text_to_match': 'The animal chased the dog', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 96.93333333333334, 'word_infos': [[4, 10, 'overlapping_relation', False, "Matches ANIMAL directly."], [11, 17, 'overlapping_relation', True, "Matches CHASE directly."], [22, 25, 'overlapping_relation', False, "Matches DOG directly."]]}, {'document_label': 'specific-reversed', 'text': 'The cat chased the dog', 'text_to_match': 'The animal chased the dog', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 22, 'score': 87.446, 'word_infos': [[4, 7, 'overlapping_relation', False, "Is a child of ANIMAL in the ontology."], [8, 14, 'overlapping_relation', True, "Matches CHASE directly."], [19, 22, 'overlapping_relation', False, "Matches DOG directly."]]}, {'document_label': 'exact', 'text': 'The dog chased the animal', 'text_to_match': 'The animal chased the dog', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 25, 'score': 30.598666666666666, 'word_infos': [[4, 7, 'relation', False, "Is a child of ANIMAL in the ontology."], [8, 14, 'relation', False, "Matches CHASE directly."], [19, 25, 'single', True, "Matches ANIMAL directly."]]}, {'document_label': 'specific', 'text': 'I saw a dog. It was chasing a cat', 'text_to_match': 'The animal chased the dog', 'rank': '3=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 33, 'score': 27.704, 'word_infos': [[8, 11, 'relation', False, "Is a child of ANIMAL in the ontology."], [20, 27, 'relation', True, "Is a synonym of CHASE in the ontology."], [30, 33, 'single', False, "Is a child of ANIMAL in the ontology."]]}])
        m.close()

    def test_multithreading_topic_matching_with_2_workers(self):
        self._internal_test_multithreading_topic_matching(2)

    def test_multithreading_topic_matching_with_4_workers(self):
        self._internal_test_multithreading_topic_matching(4)

    def test_multithreading_topic_matching_with_8_workers(self):
        self._internal_test_multithreading_topic_matching(8)

    def test_multithreading_filtering_with_topic_match_dictionaries(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', number_of_workers=2,
                                          ontology=ontology, verbose=False)

        m.parse_and_register_documents({'T11': "The dog chased the cat",
                                        'T12': "The dog chased the cat",
                                        'T21': "The dog chased the cat",
                                        'T22': "The dog chased the cat"})
        topic_match_dictionaries = \
            m.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat")
        self.assertEqual(len(topic_match_dictionaries), 4)
        topic_match_dictionaries = \
            m.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", document_label_filter="T")
        self.assertEqual(len(topic_match_dictionaries), 4)
        topic_match_dictionaries = \
            m.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", document_label_filter="T1")
        self.assertEqual(len(topic_match_dictionaries), 2)
        topic_match_dictionaries = \
            m.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", document_label_filter="T22")
        self.assertEqual(len(topic_match_dictionaries), 1)
        topic_match_dictionaries = \
            m.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", document_label_filter="X")
        self.assertEqual(len(topic_match_dictionaries), 0)
        m.close()

    def test_different_match_cutoff_score_high(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', number_of_workers=2,
                                          ontology=ontology, verbose=False)
        m.parse_and_register_documents({'': "A dog then and then and then and then and then a dog"})
        topic_match_dictionaries = \
            m.topic_match_documents_returning_dictionaries_against(
                "A dog", different_match_cutoff_score=10000)
        self.assertEqual(len(topic_match_dictionaries), 2)
        m.close()

    def test_different_match_cutoff_score_control(self):
        m = holmes.MultiprocessingManager('en_core_web_sm', number_of_workers=2,
                                          ontology=ontology, verbose=False)
        m.parse_and_register_documents({'': "A dog then and then and then and then and then a dog"})
        topic_match_dictionaries = \
            m.topic_match_documents_returning_dictionaries_against(
                "A dog")
        self.assertEqual(len(topic_match_dictionaries), 1)
        m.close()
