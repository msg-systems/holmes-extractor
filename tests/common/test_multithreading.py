import unittest
import holmes_extractor as holmes
import os
from threading import Thread
from queue import Queue
from collections import OrderedDict

NUMBER_OF_THREADS = 50

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory, "test_ontology.owl")))
manager = holmes.Manager(
    "en_core_web_trf",
    ontology=ontology,
    overall_similarity_threshold=0.90,
    number_of_workers=2,
)
manager.parse_and_register_document("The hungry lion chased the angry gnu.", "lion")
manager.parse_and_register_document("The hungry tiger chased the angry gnu.", "tiger")
manager.parse_and_register_document(
    "The hungry panther chased the angry gnu.", "panther"
)
manager.parse_and_register_document(
    "I saw a donkey. It was chasing the angry gnu.", "donkey"
)
manager.parse_and_register_document("A foal", "foal")
manager.register_search_phrase("A gnu is chased")
manager.register_search_phrase("An angry gnu")
manager.register_search_phrase("A tiger chases")
manager.register_search_phrase("I discussed various things with ENTITYPERSON")
manager.register_search_phrase("A horse")
sttb = manager.get_supervised_topic_training_basis(
    classification_ontology=ontology, one_hot=False, verbose=False
)
sttb.parse_and_register_training_document("A puppy", "puppy", "d0")
sttb.parse_and_register_training_document("A pussy", "cat", "d1")
sttb.parse_and_register_training_document("A dog on a lead", "dog", "d2")
sttb.parse_and_register_training_document("Mimi Momo", "Mimi Momo", "d3")
sttb.parse_and_register_training_document("An animal", "animal", "d4")
sttb.parse_and_register_training_document("A computer", "computers", "d5")
sttb.parse_and_register_training_document("A robot", "computers", "d6")
sttb.register_additional_classification_label("parrot")
sttb.register_additional_classification_label("hound")
sttb.prepare()


def get_first_key_in_dict(dictionary: OrderedDict) -> str:
    if dictionary is None or len(dictionary) == 0:
        return None
    return list(dictionary.keys())[0]


for i in range(20):
    trainer = sttb.train(
        minimum_occurrences=0,
        cv_threshold=0,
        max_epochs=1000,
        learning_rate=0.0001,
        convergence_threshold=0,
    )
    stc = trainer.classifier()
    if (
        get_first_key_in_dict(stc.parse_and_classify("You are a robot.")) == "computers"
        and get_first_key_in_dict(stc.parse_and_classify("You are a cat.")) == "animal"
    ):
        break
    if i == 19:
        print("Test setup failed.")
        exit(1)


class MultithreadingTest(unittest.TestCase):
    def _process_threads(self, method, first_argument, expected_output):
        queue = Queue()
        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=method, args=(first_argument, queue))
            t.start()
        for i in range(NUMBER_OF_THREADS):
            output = queue.get(True, 20)
            if (
                first_argument == "I saw a foal."
                and output[0]["sentences_within_document"]
                == "I saw a foal"  # The Transformer model does this occasionally
            ):
                output[0]["sentences_within_document"] = "I saw a foal."
            if first_argument == "A tiger chases a gnu":
                self.assertAlmostEqual(
                    float(output[1]["overall_similarity_measure"]), 0.90286449, places=3
                )
                self.assertAlmostEqual(
                    float(output[1]["word_matches"][0]["similarity_measure"]),
                    0.7359829,
                    places=3,
                )
                output[1]["overall_similarity_measure"] = "0"
                output[1]["word_matches"][0]["similarity_measure"] = "0"
            self.assertEqual(output, expected_output)

    def _match_against_documents_within_thread(self, search_phrase, queue):
        queue.put(manager.match(search_phrase_text=search_phrase))

    def _inner_match_against_documents(self, search_phrase, expected_output):
        self._process_threads(
            self._match_against_documents_within_thread, search_phrase, expected_output
        )

    def _match_against_search_phrases_within_thread(self, document_text, queue):
        queue.put(manager.match(document_text=document_text))

    def _inner_match_against_search_phrases(self, document_text, expected_output):
        self._process_threads(
            self._match_against_search_phrases_within_thread,
            document_text,
            expected_output,
        )

    def _inner_classify(self, documents, expected_output):
        self._process_threads(self._classify_within_thread, documents, expected_output)

    def _classify_within_thread(self, documents, queue):
        output = []
        for document in documents:
            output.append(get_first_key_in_dict(stc.parse_and_classify(document)))
        queue.put(output)

    def test_multithreading_matching_against_documents_general(self):
        self._inner_match_against_documents(
            "A gnu is chased",
            [
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "A gnu is chased",
                    "document": "donkey",
                    "index_within_document": 7,
                    "sentences_within_document": "It was chasing the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "gnu",
                            "document_token_index": 10,
                            "first_document_token_index": 10,
                            "last_document_token_index": 10,
                            "structurally_matched_document_token_index": 10,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                        {
                            "search_phrase_token_index": 3,
                            "search_phrase_word": "chase",
                            "document_token_index": 7,
                            "first_document_token_index": 7,
                            "last_document_token_index": 7,
                            "structurally_matched_document_token_index": 7,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chasing",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "A gnu is chased",
                    "document": "lion",
                    "index_within_document": 3,
                    "sentences_within_document": "The hungry lion chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                        {
                            "search_phrase_token_index": 3,
                            "search_phrase_word": "chase",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chased",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "A gnu is chased",
                    "document": "panther",
                    "index_within_document": 3,
                    "sentences_within_document": "The hungry panther chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                        {
                            "search_phrase_token_index": 3,
                            "search_phrase_word": "chase",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chased",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "A gnu is chased",
                    "document": "tiger",
                    "index_within_document": 3,
                    "sentences_within_document": "The hungry tiger chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                        {
                            "search_phrase_token_index": 3,
                            "search_phrase_word": "chase",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chased",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                    ],
                },
            ],
        )
        self._inner_match_against_documents(
            "An angry gnu",
            [
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "An angry gnu",
                    "document": "donkey",
                    "index_within_document": 10,
                    "sentences_within_document": "It was chasing the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "angry",
                            "document_token_index": 9,
                            "first_document_token_index": 9,
                            "last_document_token_index": 9,
                            "structurally_matched_document_token_index": 9,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "angry",
                            "document_phrase": "angry",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "angry",
                            "depth": 0,
                            "explanation": "Matches ANGRY directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "gnu",
                            "document_token_index": 10,
                            "first_document_token_index": 10,
                            "last_document_token_index": 10,
                            "structurally_matched_document_token_index": 10,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "An angry gnu",
                    "document": "lion",
                    "index_within_document": 6,
                    "sentences_within_document": "The hungry lion chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "angry",
                            "document_token_index": 5,
                            "first_document_token_index": 5,
                            "last_document_token_index": 5,
                            "structurally_matched_document_token_index": 5,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "angry",
                            "document_phrase": "angry",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "angry",
                            "depth": 0,
                            "explanation": "Matches ANGRY directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "An angry gnu",
                    "document": "panther",
                    "index_within_document": 6,
                    "sentences_within_document": "The hungry panther chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "angry",
                            "document_token_index": 5,
                            "first_document_token_index": 5,
                            "last_document_token_index": 5,
                            "structurally_matched_document_token_index": 5,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "angry",
                            "document_phrase": "angry",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "angry",
                            "depth": 0,
                            "explanation": "Matches ANGRY directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "An angry gnu",
                    "document": "tiger",
                    "index_within_document": 6,
                    "sentences_within_document": "The hungry tiger chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "angry",
                            "document_token_index": 5,
                            "first_document_token_index": 5,
                            "last_document_token_index": 5,
                            "structurally_matched_document_token_index": 5,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "angry",
                            "document_phrase": "angry",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "angry",
                            "depth": 0,
                            "explanation": "Matches ANGRY directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                    ],
                },
            ],
        )

    def test_multithreading_matching_against_documents_coreference(self):
        self._inner_match_against_documents(
            "A donkey chases",
            [
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "A donkey chases",
                    "document": "donkey",
                    "index_within_document": 7,
                    "sentences_within_document": "I saw a donkey. It was chasing the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": True,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "donkey",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 5,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "donkey",
                            "document_phrase": "a donkey",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": True,
                            "extracted_word": "donkey",
                            "depth": 0,
                            "explanation": "Matches DONKEY directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "chase",
                            "document_token_index": 7,
                            "first_document_token_index": 7,
                            "last_document_token_index": 7,
                            "structurally_matched_document_token_index": 7,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chasing",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                    ],
                }
            ],
        )

    def test_multithreading_matching_against_documents_embedding_matching(self):
        self._inner_match_against_documents(
            "A tiger chases a gnu",
            [
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "A tiger chases a gnu",
                    "document": "tiger",
                    "index_within_document": 3,
                    "sentences_within_document": "The hungry tiger chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "tiger",
                            "document_token_index": 2,
                            "first_document_token_index": 2,
                            "last_document_token_index": 2,
                            "structurally_matched_document_token_index": 2,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "tiger",
                            "document_phrase": "The hungry tiger",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "tiger",
                            "depth": 0,
                            "explanation": "Matches TIGER directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "chase",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chased",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                        {
                            "search_phrase_token_index": 4,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "A tiger chases a gnu",
                    "document": "lion",
                    "index_within_document": 3,
                    "sentences_within_document": "The hungry lion chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": "0",
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "tiger",
                            "document_token_index": 2,
                            "first_document_token_index": 2,
                            "last_document_token_index": 2,
                            "structurally_matched_document_token_index": 2,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "lion",
                            "document_phrase": "The hungry lion",
                            "match_type": "embedding",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": "0",
                            "involves_coreference": False,
                            "extracted_word": "lion",
                            "depth": 0,
                            "explanation": "Has a word embedding that is 73% similar to TIGER.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "chase",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chased",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                        {
                            "search_phrase_token_index": 4,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                    ],
                },
            ],
        )

    def test_multithreading_matching_against_documents_ontology_matching(self):
        self._inner_match_against_documents(
            "A horse",
            [
                {
                    "search_phrase_label": "",
                    "search_phrase_text": "A horse",
                    "document": "foal",
                    "index_within_document": 1,
                    "sentences_within_document": "A foal",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "horse",
                            "document_token_index": 1,
                            "first_document_token_index": 1,
                            "last_document_token_index": 1,
                            "structurally_matched_document_token_index": 1,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "foal",
                            "document_phrase": "A foal",
                            "match_type": "ontology",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "foal",
                            "depth": 1,
                            "explanation": "Is a child of HORSE in the ontology.",
                        }
                    ],
                }
            ],
        )

    def test_multithreading_matching_against_search_phrases_general(self):
        self._inner_match_against_search_phrases(
            "The hungry lion chased the angry gnu.",
            [
                {
                    "search_phrase_label": "A gnu is chased",
                    "search_phrase_text": "A gnu is chased",
                    "document": "",
                    "index_within_document": 3,
                    "sentences_within_document": "The hungry lion chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                        {
                            "search_phrase_token_index": 3,
                            "search_phrase_word": "chase",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chased",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "An angry gnu",
                    "search_phrase_text": "An angry gnu",
                    "document": "",
                    "index_within_document": 6,
                    "sentences_within_document": "The hungry lion chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "angry",
                            "document_token_index": 5,
                            "first_document_token_index": 5,
                            "last_document_token_index": 5,
                            "structurally_matched_document_token_index": 5,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "angry",
                            "document_phrase": "angry",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "angry",
                            "depth": 0,
                            "explanation": "Matches ANGRY directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                    ],
                },
            ],
        )
        self._inner_match_against_search_phrases(
            "The hungry tiger chased the angry gnu.",
            [
                {
                    "search_phrase_label": "A gnu is chased",
                    "search_phrase_text": "A gnu is chased",
                    "document": "",
                    "index_within_document": 3,
                    "sentences_within_document": "The hungry tiger chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                        {
                            "search_phrase_token_index": 3,
                            "search_phrase_word": "chase",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chased",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "A tiger chases",
                    "search_phrase_text": "A tiger chases",
                    "document": "",
                    "index_within_document": 3,
                    "sentences_within_document": "The hungry tiger chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "tiger",
                            "document_token_index": 2,
                            "first_document_token_index": 2,
                            "last_document_token_index": 2,
                            "structurally_matched_document_token_index": 2,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "tiger",
                            "document_phrase": "The hungry tiger",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "tiger",
                            "depth": 0,
                            "explanation": "Matches TIGER directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "chase",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chased",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "An angry gnu",
                    "search_phrase_text": "An angry gnu",
                    "document": "",
                    "index_within_document": 6,
                    "sentences_within_document": "The hungry tiger chased the angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "angry",
                            "document_token_index": 5,
                            "first_document_token_index": 5,
                            "last_document_token_index": 5,
                            "structurally_matched_document_token_index": 5,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "angry",
                            "document_phrase": "angry",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "angry",
                            "depth": 0,
                            "explanation": "Matches ANGRY directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "gnu",
                            "document_token_index": 6,
                            "first_document_token_index": 6,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "the angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                    ],
                },
            ],
        )
        self._inner_match_against_search_phrases(
            "I saw a hungry panther. It was chasing an angry gnu.",
            [
                {
                    "search_phrase_label": "A gnu is chased",
                    "search_phrase_text": "A gnu is chased",
                    "document": "",
                    "index_within_document": 8,
                    "sentences_within_document": "It was chasing an angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "gnu",
                            "document_token_index": 11,
                            "first_document_token_index": 11,
                            "last_document_token_index": 11,
                            "structurally_matched_document_token_index": 11,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "an angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                        {
                            "search_phrase_token_index": 3,
                            "search_phrase_word": "chase",
                            "document_token_index": 8,
                            "first_document_token_index": 8,
                            "last_document_token_index": 8,
                            "structurally_matched_document_token_index": 8,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "chase",
                            "document_phrase": "chasing",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "chase",
                            "depth": 0,
                            "explanation": "Matches CHASE directly.",
                        },
                    ],
                },
                {
                    "search_phrase_label": "An angry gnu",
                    "search_phrase_text": "An angry gnu",
                    "document": "",
                    "index_within_document": 11,
                    "sentences_within_document": "It was chasing an angry gnu.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "angry",
                            "document_token_index": 10,
                            "first_document_token_index": 10,
                            "last_document_token_index": 10,
                            "structurally_matched_document_token_index": 10,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "angry",
                            "document_phrase": "angry",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "angry",
                            "depth": 0,
                            "explanation": "Matches ANGRY directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "gnu",
                            "document_token_index": 11,
                            "first_document_token_index": 11,
                            "last_document_token_index": 11,
                            "structurally_matched_document_token_index": 11,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "gnu",
                            "document_phrase": "an angry gnu",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "gnu",
                            "depth": 0,
                            "explanation": "Matches GNU directly.",
                        },
                    ],
                },
            ],
        )

    def test_multithreading_matching_against_search_phrases_entity_matching(self):
        self._inner_match_against_search_phrases(
            "I discussed various things with Richard Hudson.",
            [
                {
                    "search_phrase_label": "I discussed various things with ENTITYPERSON",
                    "search_phrase_text": "I discussed various things with ENTITYPERSON",
                    "document": "",
                    "index_within_document": 1,
                    "sentences_within_document": "I discussed various things with Richard Hudson.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "discuss",
                            "document_token_index": 1,
                            "first_document_token_index": 1,
                            "last_document_token_index": 1,
                            "structurally_matched_document_token_index": 1,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "discuss",
                            "document_phrase": "discussed",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "discuss",
                            "depth": 0,
                            "explanation": "Matches DISCUSS directly.",
                        },
                        {
                            "search_phrase_token_index": 2,
                            "search_phrase_word": "various",
                            "document_token_index": 2,
                            "first_document_token_index": 2,
                            "last_document_token_index": 2,
                            "structurally_matched_document_token_index": 2,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "various",
                            "document_phrase": "various",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "various",
                            "depth": 0,
                            "explanation": "Matches VARIOUS directly.",
                        },
                        {
                            "search_phrase_token_index": 3,
                            "search_phrase_word": "thing",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "thing",
                            "document_phrase": "various things",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "thing",
                            "depth": 0,
                            "explanation": "Matches THING directly.",
                        },
                        {
                            "search_phrase_token_index": 4,
                            "search_phrase_word": "with",
                            "document_token_index": 4,
                            "first_document_token_index": 4,
                            "last_document_token_index": 4,
                            "structurally_matched_document_token_index": 4,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "with",
                            "document_phrase": "with",
                            "match_type": "direct",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "with",
                            "depth": 0,
                            "explanation": "Matches WITH directly.",
                        },
                        {
                            "search_phrase_token_index": 5,
                            "search_phrase_word": "ENTITYPERSON",
                            "document_token_index": 6,
                            "first_document_token_index": 5,
                            "last_document_token_index": 6,
                            "structurally_matched_document_token_index": 6,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "richard hudson",
                            "document_phrase": "Richard Hudson",
                            "match_type": "entity",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "richard hudson",
                            "depth": 0,
                            "explanation": "Has an entity label matching ENTITYPERSON.",
                        },
                    ],
                }
            ],
        )

    def test_multithreading_matching_against_search_phrases_ontology_matching(self):
        self._inner_match_against_search_phrases(
            "I saw a foal.",
            [
                {
                    "search_phrase_label": "A horse",
                    "search_phrase_text": "A horse",
                    "document": "",
                    "index_within_document": 3,
                    "sentences_within_document": "I saw a foal.",
                    "negated": False,
                    "uncertain": False,
                    "involves_coreference": False,
                    "overall_similarity_measure": 1.0,
                    "word_matches": [
                        {
                            "search_phrase_token_index": 1,
                            "search_phrase_word": "horse",
                            "document_token_index": 3,
                            "first_document_token_index": 3,
                            "last_document_token_index": 3,
                            "structurally_matched_document_token_index": 3,
                            "document_subword_index": None,
                            "document_subword_containing_token_index": None,
                            "document_word": "foal",
                            "document_phrase": "a foal",
                            "match_type": "ontology",
                            "negated": False,
                            "uncertain": False,
                            "similarity_measure": 1.0,
                            "involves_coreference": False,
                            "extracted_word": "foal",
                            "depth": 1,
                            "explanation": "Is a child of HORSE in the ontology.",
                        }
                    ],
                }
            ],
        )

    def test_multithreading_supervised_document_classification(self):

        self._inner_classify(
            [
                "You are a robot.",
                "You are a cat",
            ],
            ["computers", "animal"],
        )

    def test_multithreading_topic_matching(self):
        def topic_match_within_thread():
            topic_matches = manager.topic_match_documents_against(
                "Once upon a time a foal chased a hungry panther"
            )
            output = [
                topic_matches[0]["document_label"],
                topic_matches[0]["text"],
                topic_matches[1]["document_label"],
                topic_matches[1]["text"],
            ]
            queue.put(output)

        queue = Queue()
        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=topic_match_within_thread)
            t.start()
        for i in range(NUMBER_OF_THREADS):
            output = queue.get(True, 180)
            self.assertEqual(
                output,
                [
                    "panther",
                    "The hungry panther chased the angry gnu.",
                    "foal",
                    "A foal",
                ],
            )

    def test_parsed_document_and_search_phrase_registration(self):
        def add_document_and_search_phrase(counter):
            manager.parse_and_register_document(
                "People discuss relevancies", " ".join(("Relevant", str(counter)))
            )
            manager.register_search_phrase("People discuss relevancies")

        manager.remove_all_documents()
        manager.parse_and_register_document("something")

        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=add_document_and_search_phrase, args=(i,))
            t.start()

        last_number_of_matches = 0
        for counter in range(500):
            matches = [
                match
                for match in manager.match()
                if match["search_phrase_label"] == "People discuss relevancies"
            ]
            for match in matches:
                self.assertTrue(match["document"].startswith("Relevant"))
                self.assertFalse(match["negated"])
                self.assertFalse(match["uncertain"])
                self.assertFalse(match["involves_coreference"])
                self.assertEqual(match["overall_similarity_measure"], 1.0)
                self.assertEqual(match["index_within_document"], 1)
                self.assertEqual(match["word_matches"][0]["document_word"], "people")
                self.assertEqual(
                    match["word_matches"][0]["search_phrase_word"], "people"
                )
                self.assertEqual(match["word_matches"][0]["match_type"], "direct")
                self.assertEqual(match["word_matches"][0]["document_token_index"], 0)
                self.assertEqual(
                    match["word_matches"][0]["search_phrase_token_index"], 0
                )
                self.assertEqual(match["word_matches"][1]["document_word"], "discuss")
                self.assertEqual(
                    match["word_matches"][1]["search_phrase_word"], "discuss"
                )
                self.assertEqual(match["word_matches"][1]["match_type"], "direct")
                self.assertEqual(match["word_matches"][1]["document_token_index"], 1)
                self.assertEqual(
                    match["word_matches"][1]["search_phrase_token_index"], 1
                )
                self.assertEqual(match["word_matches"][2]["document_word"], "relevancy")
                self.assertEqual(
                    match["word_matches"][2]["search_phrase_word"], "relevancy"
                )
                self.assertEqual(match["word_matches"][2]["match_type"], "direct")
                self.assertEqual(match["word_matches"][2]["document_token_index"], 2)
                self.assertEqual(
                    match["word_matches"][2]["search_phrase_token_index"], 2
                )

            this_number_of_matches = len(matches)
            self.assertFalse(this_number_of_matches < last_number_of_matches)
            last_number_of_matches = this_number_of_matches
            if this_number_of_matches == NUMBER_OF_THREADS * NUMBER_OF_THREADS:
                break
            self.assertFalse(counter == 499)
        dictionary, maximum = manager.get_corpus_frequency_information()
        self.assertEqual(dictionary["people"], NUMBER_OF_THREADS)
        self.assertEqual(dictionary["discuss"], NUMBER_OF_THREADS)
        self.assertEqual(dictionary["relevancy"], NUMBER_OF_THREADS)
        self.assertEqual(maximum, NUMBER_OF_THREADS)

    def test_serialized_document_and_search_phrase_registration(self):
        def add_document_and_search_phrase(counter):
            manager.register_serialized_document(
                serialized_document, " ".join(("Irrelevant", str(counter)))
            )
            manager.register_search_phrase("People discuss irrelevancies")

        serialized_document = manager.nlp("People discuss irrelevancies").to_bytes()

        manager.remove_all_documents()
        manager.parse_and_register_document("something")
        for i in range(NUMBER_OF_THREADS):
            t = Thread(target=add_document_and_search_phrase, args=(i,))
            t.start()

        last_number_of_matches = 0
        for counter in range(500):
            matches = [
                match
                for match in manager.match()
                if match["search_phrase_label"] == "People discuss irrelevancies"
            ]
            for match in matches:
                self.assertTrue(match["document"].startswith("Irrelevant"))
                self.assertFalse(match["negated"])
                self.assertFalse(match["uncertain"])
                self.assertFalse(match["involves_coreference"])
                self.assertEqual(match["overall_similarity_measure"], 1.0)
                self.assertEqual(match["index_within_document"], 1)
                self.assertEqual(match["word_matches"][0]["document_word"], "people")
                self.assertEqual(
                    match["word_matches"][0]["search_phrase_word"], "people"
                )
                self.assertEqual(match["word_matches"][0]["match_type"], "direct")
                self.assertEqual(match["word_matches"][0]["document_token_index"], 0)
                self.assertEqual(
                    match["word_matches"][0]["search_phrase_token_index"], 0
                )
                self.assertEqual(match["word_matches"][1]["document_word"], "discuss")
                self.assertEqual(
                    match["word_matches"][1]["search_phrase_word"], "discuss"
                )
                self.assertEqual(match["word_matches"][1]["match_type"], "direct")
                self.assertEqual(match["word_matches"][1]["document_token_index"], 1)
                self.assertEqual(
                    match["word_matches"][1]["search_phrase_token_index"], 1
                )
                self.assertTrue(
                    match["word_matches"][2]["document_word"]
                    in ("irrelevancy", "irrelevancies")
                    # occasionally, depending on the initialisation of the transformer,
                    # the wrong lemma is returned
                )
                self.assertTrue(
                    match["word_matches"][2]["search_phrase_word"]
                    in ("irrelevancy", "irrelevancies")
                    # occasionally, depending on the initialisation of the transformer,
                    # the wrong lemma is returned
                )
                self.assertEqual(match["word_matches"][2]["match_type"], "direct")
                self.assertEqual(match["word_matches"][2]["document_token_index"], 2)
                self.assertEqual(
                    match["word_matches"][2]["search_phrase_token_index"], 2
                )

            this_number_of_matches = len(matches)
            self.assertFalse(this_number_of_matches < last_number_of_matches)
            last_number_of_matches = this_number_of_matches
            if this_number_of_matches == NUMBER_OF_THREADS * NUMBER_OF_THREADS:
                break
            self.assertFalse(counter == 499)
        dictionary, maximum = manager.get_corpus_frequency_information()
        self.assertEqual(dictionary["irrelevancy"], NUMBER_OF_THREADS)
        self.assertEqual(maximum, NUMBER_OF_THREADS)
