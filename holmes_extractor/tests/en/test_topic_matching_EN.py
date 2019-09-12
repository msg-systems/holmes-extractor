import unittest
import holmes_extractor as holmes
from holmes_extractor.extensive_matching import TopicMatcher
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
holmes_manager_coref = holmes.Manager(model='en_core_web_lg', ontology=ontology,
        overall_similarity_threshold=0.85, perform_coreference_resolution=True)
holmes_manager_coref_embedding_on_root = holmes.Manager(model='en_core_web_lg', ontology=ontology,
        overall_similarity_threshold=0.72, embedding_based_matching_on_root_words=True)

class EnglishTopicMatchingTest(unittest.TestCase):

    def _check_equals(self, text_to_match, document_text, highest_score, embedding_on_root = False):
        if embedding_on_root:
            manager = holmes_manager_coref_embedding_on_root
        else:
            manager = holmes_manager_coref
        manager.remove_all_documents()
        manager.parse_and_register_document(document_text)
        topic_matches = manager.topic_match_documents_against(text_to_match, relation_score=20,
                single_word_score=10)
        self.assertEqual(int(topic_matches[0].score), highest_score)

    def test_direct_matching(self):
        self._check_equals("A plant grows", "A plant grows", 29)

    def test_direct_matching_nonsense_word(self):
        self._check_equals("My friend visited gegwghg", "Peter visited gegwghg", 29)

    def test_coref_matching(self):
        self._check_equals("A plant grows", "I saw a plant. It was growing", 29)

    def test_entity_matching(self):
        self._check_equals("My friend visited ENTITYGPE", "Peter visited Paris", 29)

    def test_entitynoun_matching(self):
        self._check_equals("My friend visited ENTITYNOUN", "Peter visited a city", 20)

    def test_ontology_matching(self):
        self._check_equals("I saw an animal", "Somebody saw a cat", 29)

    def test_ontology_matching_word_only(self):
        self._check_equals("I saw an animal", "Somebody chased a cat", 10)

    def test_embedding_matching_not_root(self):
        self._check_equals("I saw a king", "Somebody saw a queen", 17)

    def test_embedding_matching_root(self):
        self._check_equals("I saw a king", "Somebody saw a queen", 23, True)

    def test_embedding_matching_root_word_only(self):
        self._check_equals("king", "queen", 7, True)

    def test_matching_only_adjective(self):
        self._check_equals("nice", "nice", 10, False)

    def test_matching_only_adjective_where_noun(self):
        self._check_equals("nice place", "nice", 10, False)

    def test_stopwords(self):
        self._check_equals("The donkey has a roof", "The donkey has a roof", 19, False)

    def test_stopwords_control(self):
        self._check_equals("The donkey paints a roof", "The donkey paints a roof", 82, False)

    def test_indexes(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "This is an irrelevant sentence. I think a plant grows.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A plant grows")
        self.assertEqual(topic_matches[0].sentences_start_index, 6)
        self.assertEqual(topic_matches[0].sentences_end_index, 11)
        self.assertEqual(topic_matches[0].start_index, 9)
        self.assertEqual(topic_matches[0].end_index, 10)
        self.assertEqual(topic_matches[0].relative_start_index, 3)
        self.assertEqual(topic_matches[0].relative_end_index, 4)

    def test_indexes_with_preceding_non_matched_dependent(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "I saw a big dog.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A big dog")
        self.assertEqual(topic_matches[0].sentences_start_index, 0)
        self.assertEqual(topic_matches[0].sentences_end_index, 5)
        self.assertEqual(topic_matches[0].start_index, 3)
        self.assertEqual(topic_matches[0].end_index, 4)
        self.assertEqual(topic_matches[0].relative_start_index, 3)
        self.assertEqual(topic_matches[0].relative_end_index, 4)

    def test_indexes_with_subsequent_non_matched_dependent(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "The dog I saw was big.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A big dog")
        self.assertEqual(topic_matches[0].sentences_start_index, 0)
        self.assertEqual(topic_matches[0].sentences_end_index, 6)
        self.assertEqual(topic_matches[0].start_index, 1)
        self.assertEqual(topic_matches[0].end_index, 5)
        self.assertEqual(topic_matches[0].relative_start_index, 1)
        self.assertEqual(topic_matches[0].relative_end_index, 5)

    def test_additional_search_phrases(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document(
                "Peter visited Paris and a dog chased a cat. Beef and lamb and pork.")
        doc = holmes_manager_coref.semantic_analyzer.parse("My friend visited ENTITYGPE")
        phraselet_labels_to_search_phrases = {}
        holmes_manager_coref.structural_matcher.add_phraselets_to_dict(doc,
                phraselet_labels_to_search_phrases=phraselet_labels_to_search_phrases,
                replace_with_hypernym_ancestors=False,
                match_all_words=False,
                returning_serialized_phraselets=False)
        holmes_manager_coref.structural_matcher.register_search_phrase("A dog chases a cat", None, True)
        holmes_manager_coref.structural_matcher.register_search_phrase("beef", None, True)
        holmes_manager_coref.structural_matcher.register_search_phrase("lamb", None, True)
        position_sorted_structural_matches = sorted(holmes_manager_coref.structural_matcher.
                        match_documents_against_search_phrase_list(
                        phraselet_labels_to_search_phrases.values(),False),
                        key=lambda match: (match.document_label, match.index_within_document))
        topic_matcher = TopicMatcher(holmes_manager_coref,
                maximum_activation_distance=75,
                relation_score=20,
                single_word_score=5,
                overlapping_relation_multiplier=1.5,
                overlap_memory_size=10,
                maximum_activation_value=1000,
                sideways_match_extent=100,
                only_one_result_per_document=False,
                number_of_results=1)
        score_sorted_structural_matches = topic_matcher.perform_activation_scoring(
                position_sorted_structural_matches)
        topic_matches = topic_matcher.get_topic_matches(score_sorted_structural_matches,
                position_sorted_structural_matches)
        self.assertEqual(topic_matches[0].start_index, 1)
        self.assertEqual(topic_matches[0].end_index, 2)

    def test_only_one_result_per_document(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document(
                """
                Peter came home. A great deal of irrelevant text. A great deal of irrelevant text.
                A great deal of irrelevant text. A great deal of irrelevant text. A great deal of
                irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text.
                A great deal of irrelevant text. A great deal of irrelevant text. A great deal of
                irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text.
                A great deal of irrelevant text. A great deal of irrelevant text. A great deal of
                irrelevant text. Peter came home.
                """)
        self.assertEqual(len(holmes_manager_coref.topic_match_documents_against("Peter")), 2)
        self.assertEqual(len(holmes_manager_coref.topic_match_documents_against("Peter",
                only_one_result_per_document=True)), 1)

    def test_match_cutoff(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document(
                """
                A cat.
                A great deal of irrelevant text. A great deal of irrelevant text. A great deal of
                irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text.
                A great deal of irrelevant text. A great deal of irrelevant text. A great deal of
                irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text.
                A great deal of irrelevant text. A great deal of irrelevant text. A great deal of
                irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text.
                The dog chased the cat.
                """)
        topic_matches = holmes_manager_coref.topic_match_documents_against(
                "The dog chased the cat")
        self.assertEqual(topic_matches[0].start_index, 117)
        self.assertEqual(topic_matches[0].end_index, 120)

    def test_result_ordering_by_match_length_different_documents(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document("""
        A dog chased a cat.
        A great deal of irrelevant text. A great deal of irrelevant text. A great deal of
        irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text.
        A great deal of irrelevant text. A great deal of irrelevant text. A great deal of
        irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text.
        A great deal of irrelevant text. A great deal of irrelevant text. A great deal of
        irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text.
        A dog chased a cat. A cat
        """)
        topic_matches = holmes_manager_coref.topic_match_documents_against(
                "The dog chased the cat")
        self.assertEqual(topic_matches[0].end_index - topic_matches[0].start_index, 7)
        self.assertEqual(topic_matches[1].end_index - topic_matches[1].start_index, 4)

    def test_dictionaries(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document("A dog chased a cat. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A great deal of irrelevant text. A dog chased a cat. A cat. Another irrelevant sentence.")
        holmes_manager_coref.parse_and_register_document("Dogs and cats.",
                "animals")
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat")
        self.assertEqual(topic_match_dictionaries,
        [{'document_label': '', 'text': 'A dog chased a cat. A cat.', 'rank': '1=', 'sentences_character_start_index_in_document': 515, 'sentences_character_end_index_in_document': 541, 'finding_character_start_index_in_sentences': 2, 'finding_character_end_index_in_sentences': 25, 'score': 99.80266666666668}, {'document_label': '', 'text': 'A dog chased a cat.', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 19, 'finding_character_start_index_in_sentences': 2, 'finding_character_end_index_in_sentences': 18, 'score': 99.80266666666668}, {'document_label': 'animals', 'text': 'Dogs and cats.', 'rank': '3', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 14, 'finding_character_start_index_in_sentences': 0, 'finding_character_end_index_in_sentences': 13, 'score': 9.866666666666667}])
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", tied_result_quotient=0.01)
        self.assertEqual(topic_match_dictionaries,
        [{'document_label': '', 'text': 'A dog chased a cat. A cat.', 'rank': '1=', 'sentences_character_start_index_in_document': 515, 'sentences_character_end_index_in_document': 541, 'finding_character_start_index_in_sentences': 2, 'finding_character_end_index_in_sentences': 25, 'score': 99.80266666666668}, {'document_label': '', 'text': 'A dog chased a cat.', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 19, 'finding_character_start_index_in_sentences': 2, 'finding_character_end_index_in_sentences': 18, 'score': 99.80266666666668}, {'document_label': 'animals', 'text': 'Dogs and cats.', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 14, 'finding_character_start_index_in_sentences': 0, 'finding_character_end_index_in_sentences': 13, 'score': 9.866666666666667}])

    def test_result_ordering_by_match_length_different_documents(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document("A dog chased a cat.",'1')
        holmes_manager_coref.parse_and_register_document("A dog chased a cat. A cat.",'2')
        topic_matches = holmes_manager_coref.topic_match_documents_against(
                "The dog chased the cat")
        self.assertEqual(topic_matches[0].end_index, 7)
        self.assertEqual(topic_matches[1].end_index, 4)
