import unittest
import holmes_extractor as holmes
import os
import json
from threading import Thread
from queue import Queue

NUMBER_OF_THREADS = 50

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
manager = holmes.Manager('en_core_web_lg', ontology=ontology, overall_similarity_threshold=0.90)
manager.parse_and_register_document(
        "The hungry lion chased the angry gnu.", 'lion')
manager.parse_and_register_document(
        "The hungry tiger chased the angry gnu.", 'tiger')
manager.parse_and_register_document(
        "The hungry panther chased the angry gnu.", 'panther')
manager.parse_and_register_document(
        "I saw a donkey. It was chasing the angry gnu.", 'donkey')
manager.parse_and_register_document("A foal", 'foal')
manager.register_search_phrase('A gnu is chased')
manager.register_search_phrase('An angry gnu')
manager.register_search_phrase('A tiger chases')
manager.register_search_phrase('I discussed various things with ENTITYPERSON')
manager.register_search_phrase("A horse")
sttb = manager.get_supervised_topic_training_basis(classification_ontology=ontology,
        oneshot=False, verbose=False)
sttb.parse_and_register_training_document("A puppy", 'puppy', 'd0')
sttb.parse_and_register_training_document("A pussy", 'cat', 'd1')
sttb.parse_and_register_training_document("A dog on a lead", 'dog', 'd2')
sttb.parse_and_register_training_document("Mimi Momo", 'Mimi Momo', 'd3')
sttb.parse_and_register_training_document("An animal", 'animal', 'd4')
sttb.parse_and_register_training_document("A computer", 'computers', 'd5')
sttb.parse_and_register_training_document("A robot", 'computers', 'd6')
sttb.register_additional_classification_label('parrot')
sttb.register_additional_classification_label('hound')
sttb.prepare()
trainer = sttb.train(minimum_occurrences=0, cv_threshold=0, mlp_max_iter=10000)
stc = trainer.classifier()

class MultithreadingTest(unittest.TestCase):

    def _process_threads(self, method, first_argument, expected_output):
        queue = Queue()
        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=method,
                    args=(first_argument, queue))
            t.start()
        for i in range(NUMBER_OF_THREADS):
            output = queue.get(True,5)
            self.assertEqual(output, expected_output)

    def _match_against_documents_within_thread(self, search_phrase, queue):
        queue.put(manager.match_documents_against(search_phrase))

    def _inner_match_against_documents(self, search_phrase, expected_output):
        self._process_threads(self._match_against_documents_within_thread,
                search_phrase, expected_output)

    def _match_against_search_phrases_within_thread(self, document, queue):
        queue.put(manager.match_search_phrases_against(document))

    def _inner_match_against_search_phrases(self, document, expected_output):
        self._process_threads(self._match_against_search_phrases_within_thread,
                document, expected_output)

    def _inner_classify(self, document, expected_output):
        self._process_threads(self._classify_within_thread,
                document, expected_output)

    def _classify_within_thread(self, document, queue):
        queue.put(stc.parse_and_classify(document))

    def test_multithreading_matching_against_documents_general(self):
        self._inner_match_against_documents("A gnu is chased",
        [{'search_phrase': 'A gnu is chased', 'document': 'lion', 'index_within_document': 3, 'sentences_within_document': 'The hungry lion chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chased', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}]}, {'search_phrase': 'A gnu is chased', 'document': 'tiger', 'index_within_document': 3, 'sentences_within_document': 'The hungry tiger chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chased', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}]}, {'search_phrase': 'A gnu is chased', 'document': 'panther', 'index_within_document': 3, 'sentences_within_document': 'The hungry panther chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chased', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}]}, {'search_phrase': 'A gnu is chased', 'document': 'donkey', 'index_within_document': 7, 'sentences_within_document': 'It was chasing the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chasing', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}]}])
        self._inner_match_against_documents("An angry gnu",
        [{'search_phrase': 'An angry gnu', 'document': 'lion', 'index_within_document': 6, 'sentences_within_document': 'The hungry lion chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'angry', 'document_word': 'angry', 'document_phrase': 'angry', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'angry'}, {'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}]}, {'search_phrase': 'An angry gnu', 'document': 'tiger', 'index_within_document': 6, 'sentences_within_document': 'The hungry tiger chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'angry', 'document_word': 'angry', 'document_phrase': 'angry', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'angry'}, {'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}]}, {'search_phrase': 'An angry gnu', 'document': 'panther', 'index_within_document': 6, 'sentences_within_document': 'The hungry panther chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'angry', 'document_word': 'angry', 'document_phrase': 'angry', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'angry'}, {'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}]}, {'search_phrase': 'An angry gnu', 'document': 'donkey', 'index_within_document': 10, 'sentences_within_document': 'It was chasing the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'angry', 'document_word': 'angry', 'document_phrase': 'angry', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'angry'}, {'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}]}])

    def test_multithreading_matching_against_documents_coreference(self):
        self._inner_match_against_documents("A donkey chases",
        [{'search_phrase': 'A donkey chases', 'document': 'donkey', 'index_within_document': 7, 'sentences_within_document': 'I saw a donkey. It was chasing the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': True, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'donkey', 'document_word': 'donkey', 'document_phrase': 'a donkey', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': True, 'extracted_word': 'donkey'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chasing', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}]}])

    def test_multithreading_matching_against_documents_embedding_matching(self):
        self._inner_match_against_documents("A tiger chases a gnu",
        [{'search_phrase': 'A tiger chases a gnu', 'document': 'tiger', 'index_within_document': 3, 'sentences_within_document': 'The hungry tiger chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'tiger', 'document_word': 'tiger', 'document_phrase': 'The hungry tiger', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'tiger'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chased', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}, {'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}]}, {'search_phrase': 'A tiger chases a gnu', 'document': 'lion', 'index_within_document': 3, 'sentences_within_document': 'The hungry lion chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '0.90286449', 'word_matches': [{'search_phrase_word': 'tiger', 'document_word': 'lion', 'document_phrase': 'The hungry lion', 'match_type': 'embedding', 'similarity_measure': '0.7359829', 'involves_coreference': False, 'extracted_word': 'lion'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chased', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}, {'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}]}])

    def test_multithreading_matching_against_documents_ontology_matching(self):
        self._inner_match_against_documents("A horse",
        [{'search_phrase': 'A horse', 'document': 'foal', 'index_within_document': 1, 'sentences_within_document': 'A foal', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'horse', 'document_word': 'foal', 'document_phrase': 'A foal', 'match_type': 'ontology', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'foal'}]}])

    def test_multithreading_matching_against_search_phrases_general(self):
        self._inner_match_against_search_phrases("The hungry lion chased the angry gnu.",
        [{'search_phrase': 'A gnu is chased', 'document': '', 'index_within_document': 3, 'sentences_within_document': 'The hungry lion chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chased', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}]}, {'search_phrase': 'An angry gnu', 'document': '', 'index_within_document': 6, 'sentences_within_document': 'The hungry lion chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'angry', 'document_word': 'angry', 'document_phrase': 'angry', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'angry'}, {'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}]}])
        self._inner_match_against_search_phrases("The hungry tiger chased the angry gnu.",
        [{'search_phrase': 'A gnu is chased', 'document': '', 'index_within_document': 3, 'sentences_within_document': 'The hungry tiger chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chased', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}]}, {'search_phrase': 'An angry gnu', 'document': '', 'index_within_document': 6, 'sentences_within_document': 'The hungry tiger chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'angry', 'document_word': 'angry', 'document_phrase': 'angry', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'angry'}, {'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'the angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}]}, {'search_phrase': 'A tiger chases', 'document': '', 'index_within_document': 3, 'sentences_within_document': 'The hungry tiger chased the angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'tiger', 'document_word': 'tiger', 'document_phrase': 'The hungry tiger', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'tiger'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chased', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}]}])
        self._inner_match_against_search_phrases(
                "I saw a hungry panther. It was chasing an angry gnu.",
            [{'search_phrase': 'A gnu is chased', 'document': '', 'index_within_document': 8, 'sentences_within_document': 'It was chasing an angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'an angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}, {'search_phrase_word': 'chase', 'document_word': 'chase', 'document_phrase': 'chasing', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'chase'}]}, {'search_phrase': 'An angry gnu', 'document': '', 'index_within_document': 11, 'sentences_within_document': 'It was chasing an angry gnu.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'angry', 'document_word': 'angry', 'document_phrase': 'angry', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'angry'}, {'search_phrase_word': 'gnu', 'document_word': 'gnu', 'document_phrase': 'an angry gnu', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'gnu'}]}])

    def test_multithreading_matching_against_search_phrases_entity_matching(self):
        self._inner_match_against_search_phrases(
            "I discussed various things with Richard Hudson.",
        [{'search_phrase': 'I discussed various things with ENTITYPERSON', 'document': '', 'index_within_document': 1, 'sentences_within_document': 'I discussed various things with Richard Hudson.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'discuss', 'document_word': 'discuss', 'document_phrase': 'discussed', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'discuss'}, {'search_phrase_word': 'various', 'document_word': 'various', 'document_phrase': 'various', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'various'}, {'search_phrase_word': 'thing', 'document_word': 'thing', 'document_phrase': 'various things', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'thing'}, {'search_phrase_word': 'with', 'document_word': 'with', 'document_phrase': 'with', 'match_type': 'direct', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'with'}, {'search_phrase_word': 'ENTITYPERSON', 'document_word': 'Richard Hudson', 'document_phrase': 'Richard Hudson', 'match_type': 'entity', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'Richard Hudson'}]}])

    def test_multithreading_matching_against_search_phrases_ontology_matching(self):
        self._inner_match_against_search_phrases(
            "I saw a foal.",
        [{'search_phrase': 'A horse', 'document': '', 'index_within_document': 3, 'sentences_within_document': 'I saw a foal.', 'negated': False, 'uncertain': False, 'involves_coreference': False, 'overall_similarity_measure': '1.0', 'word_matches': [{'search_phrase_word': 'horse', 'document_word': 'foal', 'document_phrase': 'a foal', 'match_type': 'ontology', 'similarity_measure': '1.0', 'involves_coreference': False, 'extracted_word': 'foal'}]}])

    def test_multithreading_supervised_document_classification(self):

        self._inner_classify("You are a robot.", ['computers'])
        self._inner_classify("You are a cat.", ['animal'])
        self._inner_classify("My name is Charles and I like sewing.", [])
        self._inner_classify("Your dog appears to be on a lead.", ['animal', 'dog', 'hound'])

    def test_multithreading_topic_matching(self):

        def topic_match_within_thread():
            topic_matches = manager.topic_match_documents_against(
                    "Once upon a time a foal chased a hungry panther")
            output = [topic_matches[0].document_label, topic_matches[0].text,
                    topic_matches[1].document_label, topic_matches[1].text]
            queue.put(output)

        queue = Queue()
        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=topic_match_within_thread)
            t.start()
        for i in range(NUMBER_OF_THREADS):
            output = queue.get(True,5)
            self.assertEqual(output, ['panther', 'The hungry panther chased the angry gnu.',
                    'foal', 'A foal'])

    def test_document_and_search_phrase_registration(self):

        def add_document_and_search_phrase(counter):
            manager.parse_and_register_document("People discuss irrelevancies",
                    ' '.join(('Irrelevant', str(counter))))
            manager.register_search_phrase("People discuss irrelevancies")

        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=add_document_and_search_phrase, args=(i,))
            t.start()

        last_number_of_matches = 0
        for counter in range(50):
            matches = [match for match in manager.match() if
                    match.search_phrase_label == "People discuss irrelevancies"]
            for match in matches:
                self.assertTrue(match.document_label.startswith('Irrelevant'))
                self.assertFalse(match.is_negated)
                self.assertFalse(match.is_uncertain)
                self.assertFalse(match.involves_coreference)
                self.assertFalse(match.from_single_word_phraselet)
                self.assertEqual(match.overall_similarity_measure, '1.0')
                self.assertEqual(match.index_within_document, 1)
                self.assertEqual(match.word_matches[0].document_word, 'people')
                self.assertEqual(match.word_matches[0].search_phrase_word, 'people')
                self.assertEqual(match.word_matches[0].type, 'direct')
                self.assertEqual(match.word_matches[0].document_token.i, 0)
                self.assertEqual(match.word_matches[0].search_phrase_token.i, 0)
                self.assertEqual(match.word_matches[1].document_word, 'discuss')
                self.assertEqual(match.word_matches[1].search_phrase_word, 'discuss')
                self.assertEqual(match.word_matches[1].type, 'direct')
                self.assertEqual(match.word_matches[1].document_token.i, 1)
                self.assertEqual(match.word_matches[1].search_phrase_token.i, 1)
                self.assertEqual(match.word_matches[2].document_word, 'irrelevancy')
                self.assertEqual(match.word_matches[2].search_phrase_word, 'irrelevancy')
                self.assertEqual(match.word_matches[2].type, 'direct')
                self.assertEqual(match.word_matches[2].document_token.i, 2)
                self.assertEqual(match.word_matches[2].search_phrase_token.i, 2)

            this_number_of_matches = len(matches)
            self.assertFalse(this_number_of_matches < last_number_of_matches)
            last_number_of_matches = this_number_of_matches
            if this_number_of_matches == NUMBER_OF_THREADS * NUMBER_OF_THREADS:
                break
            self.assertFalse(counter == 49)
