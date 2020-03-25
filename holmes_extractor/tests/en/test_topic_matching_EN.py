import unittest
import holmes_extractor as holmes
from holmes_extractor.extensive_matching import TopicMatcher
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')),
        symmetric_matching=True)
holmes_manager_coref = holmes.Manager(model='en_core_web_lg', ontology=ontology,
        overall_similarity_threshold=0.65, perform_coreference_resolution=True)
holmes_manager_coref_embedding_on_root = holmes.Manager(model='en_core_web_lg', ontology=ontology,
        overall_similarity_threshold=0.65, embedding_based_matching_on_root_words=True)
holmes_manager_coref_no_embeddings = holmes.Manager(model='en_core_web_lg', ontology=ontology,
        overall_similarity_threshold=1, perform_coreference_resolution=True)

class EnglishTopicMatchingTest(unittest.TestCase):

    def _check_equals(self, text_to_match, document_text, highest_score, manager):
        manager.remove_all_documents()
        manager.parse_and_register_document(document_text)
        topic_matches = manager.topic_match_documents_against(text_to_match, relation_score=20,
                reverse_only_relation_score=15, single_word_score=10, single_word_any_tag_score=5)
        self.assertEqual(int(topic_matches[0].score), highest_score)

    def test_no_match(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document("A plant grows")
        topic_matches = holmes_manager_coref.topic_match_documents_against("fewfew",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5)
        self.assertEqual(topic_matches, [])

    def test_no_match_stopwords(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document("then")
        topic_matches = holmes_manager_coref.topic_match_documents_against("then",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5)
        self.assertEqual(topic_matches, [])

    def test_direct_matching(self):
        self._check_equals("A plant grows", "A plant grows", 34, holmes_manager_coref)

    def test_direct_matching_nonsense_word(self):
        self._check_equals("My friend visited gegwghg", "Peter visited gegwghg", 34,
                holmes_manager_coref)

    def test_dative_matching(self):
        self._check_equals("I gave Peter a dog", "I gave Peter a present", 34, holmes_manager_coref)

    def test_coref_matching(self):
        self._check_equals("A plant grows", "I saw a plant. It was growing", 34,
                holmes_manager_coref)

    def test_entity_matching(self):
        self._check_equals("My friend visited ENTITYGPE", "Peter visited Paris", 34,
                holmes_manager_coref)

    def test_entitynoun_matching(self):
        self._check_equals("My friend visited ENTITYNOUN", "Peter visited a city", 25,
                holmes_manager_coref)

    def test_ontology_matching_synonym(self):
        self._check_equals("I saw an pussy", "Somebody saw a cat", 31,
                holmes_manager_coref)

    def test_ontology_matching_hyponym_depth_1(self):
        self._check_equals("I saw an animal", "Somebody saw a cat", 28,
                holmes_manager_coref)

    def test_ontology_matching_hyponym_depth_2(self):
        self._check_equals("I saw an animal", "Somebody saw a kitten", 26,
                holmes_manager_coref)

    def test_ontology_matching_hypernym_depth_1(self):
        self._check_equals("I saw an cat", "Somebody saw an animal", 28,
                holmes_manager_coref)

    def test_ontology_matching_hypernym_depth_2(self):
        self._check_equals("I saw a kitten", "Somebody saw an animal", 26,
                holmes_manager_coref)

    def test_ontology_matching_both_poles(self):
        self._check_equals("A cat opens something", "An animal takes something out", 27,
                holmes_manager_coref)

    def test_ontology_matching_multiword_in_document(self):
        self._check_equals("I saw an animal", "Somebody saw Mimi Momo", 26,
                holmes_manager_coref)

    def test_ontology_matching_multiword_in_search_text(self):
        self._check_equals("I saw Mimi Momo", "Somebody saw an animal", 26,
                holmes_manager_coref)

    def test_ontology_matching_word_only(self):
        self._check_equals("I saw an animal", "Somebody chased a cat", 8,
                holmes_manager_coref)

    def test_ontology_matching_word_only_multiword_in_document(self):
        self._check_equals("I saw an animal", "Somebody chased Mimi Momo", 7,
                holmes_manager_coref)

    def test_ontology_matching_word_only_multiword_in_search_text(self):
        self._check_equals("I saw Mimi Momo", "Somebody chased an animal", 7,
                holmes_manager_coref)

    def test_embedding_matching_not_root(self):
        self._check_equals("I saw a king", "Somebody saw a queen", 15,
                holmes_manager_coref)

    def test_embedding_matching_root(self):
        self._check_equals("I saw a king", "Somebody saw a queen", 19,
                holmes_manager_coref_embedding_on_root)

    def test_embedding_matching_root_overall_similarity_too_low(self):
        self._check_equals("I saw a king", "Somebody viewed a queen", 4,
                holmes_manager_coref_embedding_on_root)

    def test_embedding_matching_root_word_only(self):
        self._check_equals("king", "queen", 4,
                holmes_manager_coref_embedding_on_root)

    def test_matching_only_adjective(self):
        self._check_equals("nice", "nice", 5, holmes_manager_coref)

    def test_matching_only_adjective_where_noun(self):
        self._check_equals("nice place", "nice", 5, holmes_manager_coref)

    def test_reverse_only_parent_lemma_threeway(self):
        self._check_equals("The donkey has a roof", "The donkey has a roof", 68,
                holmes_manager_coref)

    def test_reverse_only_parent_lemma_threeway_coreference(self):
        self._check_equals("A friend has a roof", "I saw a friend and I saw a roof. He had it.", 68,
                holmes_manager_coref)

    def test_reverse_only_parent_lemma_twoway(self):
        self._check_equals("The donkey has a roof", "The donkey has a house", 29,
                holmes_manager_coref)

    def test_reverse_only_parent_lemma_threeway_control(self):
        self._check_equals("The donkey paints a roof", "The donkey paints a roof", 82,
                holmes_manager_coref)

    def test_reverse_only_parent_lemma_twoway_control(self):
        self._check_equals("The donkey paints a roof", "The donkey paints a house", 58,
                holmes_manager_coref)

    def test_reverse_only_parent_lemma_twoway_control_no_embedding_based_match(self):
        self._check_equals("The donkey paints a roof", "The donkey paints a mouse", 34,
                holmes_manager_coref)

    def test_reverse_only_parent_lemma_be(self):
        self._check_equals("A president is a politician", "A president is a politician", 68,
                holmes_manager_coref)

    def test_reverse_only_parent_lemma_be_reversed(self):
        self._check_equals("A president is a politician", "A politician is a president", 24,
                holmes_manager_coref)

    def test_reverse_only_parent_lemma_aux_in_document(self):
        self._check_equals("A donkey has a roof", "A donkey has painted a roof", 24,
                holmes_manager_coref)

    def test_reverse_matching_noun_no_coreference(self):
        self._check_equals("A car with an engine", "An automobile with an engine", 51,
                holmes_manager_coref)

    def test_reverse_matching_noun_no_coreference_control_no_embeddings(self):
        self._check_equals("A car with an engine", "An automobile with an engine", 29,
                holmes_manager_coref_no_embeddings)

    def test_reverse_matching_noun_no_coreference_control_same_word(self):
        self._check_equals("A car with an engine", "A car with an engine", 75,
                holmes_manager_coref_no_embeddings)

    def test_forward_matching_noun_entity_governor_match(self):
        self._check_equals("An ENTITYPERSON with a car", "Richard Hudson with a vehicle", 23,
                holmes_manager_coref)

    def test_forward_matching_noun_entity_governor_no_match(self):
        self._check_equals("An ENTITYPERSON with a car", "Richard Hudson with a lion", 14,
                holmes_manager_coref)

    def test_reverse_matching_noun_entity_governed(self):
        self._check_equals("A car with an ENTITYPERSON", "A vehicle with Richard Hudson", 50,
                holmes_manager_coref)

    def test_forward_matching_noun_entitynoun_governor_match(self):
        self._check_equals("An ENTITYNOUN with a car", "Richard Hudson with a vehicle", 5,
                holmes_manager_coref)

    def test_forward_matching_noun_entitynoun_governor_no_match(self):
        self._check_equals("An ENTITYNOUN with a car", "Richard Hudson with a lion", 5,
                holmes_manager_coref)

    def test_reverse_matching_noun_entitynoun_governed(self):
        self._check_equals("A car with an ENTITYNOUN", "A vehicle with Richard Hudson", 5,
                holmes_manager_coref)

    def test_relation_matching_suppressed(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document("A dog chases a cat")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A dog chases a cat",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 0,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 24)

    def test_suppressed_relation_matching_picked_up_during_reverse_matching(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "A dog chases a cat. A lion chases a tiger.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A dog chases a cat",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 1,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 82)

    def test_suppressed_relation_matching_picked_up_during_reverse_matching_with_coreference(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "There was a man and there was a woman. He saw her. A lion sees a tiger.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A man sees a woman",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 1,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 83)

    def test_relation_matching_suppressed_control_embedding_based_matching_on_root_words(self):
        holmes_manager_coref_embedding_on_root.remove_all_documents()
        holmes_manager_coref_embedding_on_root.parse_and_register_document("A dog chases a cat")
        topic_matches = holmes_manager_coref_embedding_on_root.topic_match_documents_against(
                "A dog chases a cat",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 0,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 82)

    def test_reverse_matching_suppressed_with_relation_matching(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document("I was in Germany. I know Germany.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("in Germany",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 1,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 14)

    def test_reverse_matching_suppressed_with_relation_matching_embedding_value_also_1(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document("I was in Germany. I know Germany.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("in Germany",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 1,
                maximum_number_of_single_word_matches_for_embedding_matching = 1)
        self.assertEqual(int(topic_matches[0].score), 14)

    def test_reverse_matching_suppressed_with_embedding_reverse_matching_parent(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document("An automobile with an engine")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A car with an engine",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 29)

    def test_reverse_matching_suppressed_with_embedding_reverse_matching_parent_control(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document("An automobile with an engine")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A car with an engine",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_embedding_matching = 1)
        self.assertEqual(int(topic_matches[0].score), 51)

    def test_reverse_matching_suppressed_with_embedding_reverse_matching_child(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document("An engine with an automobile")
        topic_matches = holmes_manager_coref.topic_match_documents_against("An engine with a car",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 14)

    def test_reverse_matching_suppressed_with_embedding_reverse_matching_child_control(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document("An engine with an automobile")
        topic_matches = holmes_manager_coref.topic_match_documents_against("An engine with a car",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_embedding_matching = 1)
        self.assertEqual(int(topic_matches[0].score), 25)

    def test_entity_matching_suppressed_with_relation_matching_for_governor(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "I was tired Richard Paul Hudson. I was a tired Richard Paul Hudson.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("tired ENTITYPERSON",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 1,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 14)

    def test_entity_matching_suppressed_with_relation_matching_for_governor_control(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "I was Richard Paul Hudson. I was a tired Richard Paul Hudson.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("tired ENTITYPERSON",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 1,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 34)

    def test_entity_matching_suppressed_with_relation_matching_for_governed(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "I knew Richard Paul Hudson. I knew Richard Paul Hudson.")
        topic_matches = holmes_manager_coref.topic_match_documents_against(
                "someone knows an ENTITYPERSON",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 1,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 14)

    def test_entity_matching_suppressed_with_relation_matching_for_governed_control(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "I met Richard Paul Hudson. I knew Richard Paul Hudson.")
        topic_matches = holmes_manager_coref.topic_match_documents_against(
                "someone knows an ENTITYPERSON",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 1,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 34)

    def test_reverse_matching_noun_coreference_on_governor(self):
        self._check_equals("A car with an engine", "I saw an automobile. I saw it with an engine",
                50,
                holmes_manager_coref)

    def test_reverse_matching_noun_coreference_on_governor_control_no_embeddings(self):
        self._check_equals("A car with an engine", "I saw an automobile. I saw it with an engine",
                29,
                holmes_manager_coref_no_embeddings)

    def test_reverse_matching_noun_coreference_on_governor_control_same_word(self):
        self._check_equals("A car with an engine", "I saw a car. I saw it with an engine",
                73,
                holmes_manager_coref_no_embeddings)

    def test_reverse_matching_noun_coreference_on_governed(self):
        self._check_equals(
                "An engine with a car", "I saw an automobile. There was an engine with it", 25,
                holmes_manager_coref)

    def test_reverse_matching_noun_coreference_on_governed_control_no_embeddings(self):
        self._check_equals(
                "An engine with a car", "I saw an automobile. There was an engine with it", 14,
                holmes_manager_coref_no_embeddings)

    def test_reverse_matching_noun_coreference_on_governed_control_same_word(self):
        self._check_equals(
                "An engine with a car", "I saw a car. There was an engine with it", 76,
                holmes_manager_coref_no_embeddings)

    def test_reverse_matching_verb(self):
        self._check_equals("A company is bought", "A company is purchased", 20,
                holmes_manager_coref)

    def test_reverse_matching_verb_control_no_embeddings(self):
        self._check_equals("A company is bought", "A company is purchased", 10,
                holmes_manager_coref_no_embeddings)

    def test_reverse_matching_verb_control_same_word(self):
        self._check_equals("A company is bought", "A company is bought", 34,
                holmes_manager_coref_no_embeddings)

    def test_reverse_matching_verb_with_coreference_and_conjunction(self):
        self._check_equals("A company is bought", "A company is bought and purchased", 34,
                holmes_manager_coref)

    def test_two_matches_on_same_document_tokens_because_of_embeddings(self):
        self._check_equals("Somebody buys a vehicle",
                "Somebody buys a vehicle and a car", 34,
                holmes_manager_coref)

    def test_reverse_matching_only(self):
        self._check_equals("with an idea",
                "with an idea", 29,
                holmes_manager_coref)

    def test_repeated_single_word_label_tags_matched(self):
        self._check_equals("dog",
                "a dog and a dog", 10,
                holmes_manager_coref)

    def test_repeated_single_word_label_tags_not_matched(self):
        self._check_equals("in",
                "in and in", 5,
                holmes_manager_coref)

    def test_repeated_relation_label_not_reverse_only_no_common_term(self):
        self._check_equals("a big dog",
                "a big dog and a big dog", 34,
                holmes_manager_coref)

    def test_repeated_relation_label_not_reverse_only_common_term(self):
        self._check_equals("a big dog",
                "a big and big dog", 34,
                holmes_manager_coref)

    def test_repeated_relation_label_reverse_only_no_common_term(self):
        self._check_equals("in Germany",
                "in Germany and in Germany", 29,
                holmes_manager_coref)

    def test_repeated_relation_label_reverse_only_common_term(self):
        self._check_equals("in Germany",
                "in Germany and Germany", 29,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_and_in_document_not_root(self):
        self._check_equals("Richard Paul Hudson came",
                "I saw Richard Paul Hudson", 19,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_single_word_in_document_not_root(self):
        self._check_equals("Hudson came",
                "I saw Richard Paul Hudson", 10,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_dependent_words_in_document_not_root(self):
        self._check_equals("Richard Paul came",
                "I saw Richard Paul Hudson", 9,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_multiword_in_document_with_coref_not_root(self):
        self._check_equals("Richard Paul Hudson came",
                "I saw Richard Paul Hudson. He came", 44,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_multiword_in_document_with_noun_coref_not_root(self):
        self._check_equals("Richard Paul Hudson came",
                "I saw Richard Paul Hudson. Hudson came", 48,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_single_in_document_with_coref_not_root(self):
        self._check_equals("Hudson came",
                "I saw Richard Paul Hudson. He came", 34,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_and_in_document_root(self):
        self._check_equals("the tired Richard Paul Hudson",
                "I saw Richard Paul Hudson", 19,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_single_word_in_document_root(self):
        self._check_equals("the tired Hudson",
                "I saw Richard Paul Hudson", 10,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_dependent_words_in_document_root(self):
        self._check_equals("the tired Richard Paul",
                "I saw Richard Paul Hudson", 9,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_multiword_in_document_with_coref_root(self):
        self._check_equals("the tired Richard Paul Hudson",
                "I saw Richard Paul Hudson. He came", 19,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_single_in_document_with_coref_root(self):
        self._check_equals("the tired Hudson came",
                "I saw Richard Paul Hudson. He came", 34,
                holmes_manager_coref)

    def test_multiword_in_text_to_search_and_in_document_not_root_match_on_embeddings(self):
        self._check_equals("Richard Paul Hudson came",
                "I saw Richard Paul Hudson", 19,
                holmes_manager_coref_embedding_on_root)

    def test_multiword_in_text_to_search_and_in_document_root_match_on_embeddings(self):
        self._check_equals("the tired Richard Paul Hudson",
                "I saw Richard Paul Hudson", 19,
                holmes_manager_coref_embedding_on_root)

    def test_multiword_in_text_to_search_and_in_document_not_root_no_embeddings(self):
        self._check_equals("Richard Paul Hudson came",
                "I saw Richard Paul Hudson", 19,
                holmes_manager_coref_embedding_on_root)

    def test_multiword_in_text_to_search_and_in_document_root_no_embeddings(self):
        self._check_equals("the tired Richard Paul Hudson",
                "I saw Richard Paul Hudson", 19,
                holmes_manager_coref_embedding_on_root)

    def test_matches_in_opposite_directions(self):
        self._check_equals("Mirror of Erised",
                "Mirror of Erised", 39,
                holmes_manager_coref)

    def test_derived_form_text_to_match_single_word(self):
        self._check_equals("information",
                "inform", 10,
                holmes_manager_coref)

    def test_derived_form_document_text_single_word(self):
        self._check_equals("inform",
                "information", 5,
                holmes_manager_coref)

    def test_derived_form_text_to_match_single_word(self):
        self._check_equals("information",
                "inform", 10,
                holmes_manager_coref)

    def test_derived_form_single_word_control(self):
        self._check_equals("information",
                "information", 10,
                holmes_manager_coref)

    def test_derived_form_document_text_parent(self):
        self._check_equals("inform quickly",
                "quick information", 29,
                holmes_manager_coref)

    def test_derived_form_text_to_match_parent(self):
        self._check_equals("quick information",
                "inform quickly", 34,
                holmes_manager_coref)

    def test_derived_form_parent_control(self):
        self._check_equals("quick information",
                "quick information", 34,
                holmes_manager_coref)

    def test_derived_form_document_text_child(self):
        self._check_equals("He decided to inform",
                "He decided information", 29,
                holmes_manager_coref)

    def test_derived_form_text_to_match_child(self):
        self._check_equals("He decided information",
                "He decided to inform", 34,
                holmes_manager_coref)

    def test_derived_form_child_control(self):
        self._check_equals("He decided information",
                "He decided information", 34,
                holmes_manager_coref)

    def test_derived_forms_matched_by_ontology_1(self):
        self._check_equals("An invitation to a politician",
                "He explained to a politician", 35,
                holmes_manager_coref)

    def test_derived_forms_matched_by_ontology_2(self):
        self._check_equals("He explained to a politician",
                "An invitation to a politician", 31,
                holmes_manager_coref)

    def test_derived_multiword_child_matched_by_ontology_1(self):
        self._check_equals("He used a vault horse",
                "He used a vaulting horse", 30,
                holmes_manager_coref)

    def test_derived_multiword_child_matched_by_ontology_2(self):
        self._check_equals("He used a vaulting horse",
                "He used a vault horse", 30,
                holmes_manager_coref)

    def test_derived_multiword_parent_matched_by_ontology_1(self):
        self._check_equals("A big vault horse",
                "A big vaulting horse", 31,
                holmes_manager_coref)

    def test_derived_multiword_parent_matched_by_ontology_2(self):
        self._check_equals("A big vaulting horse",
                "A big vault horse", 31,
                holmes_manager_coref)

    def test_coreference_double_match_on_governed(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "I saw a man. The man walked")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A man walks",
                relation_score=20, single_word_score=10, single_word_any_tag_score=5)
        self.assertEqual(int(topic_matches[0].score), 34)
        self.assertEqual(topic_matches[0].sentences_start_index, 5)
        self.assertEqual(topic_matches[0].sentences_end_index, 7)
        self.assertEqual(topic_matches[0].start_index, 6)
        self.assertEqual(topic_matches[0].end_index, 7)
        self.assertEqual(topic_matches[0].relative_start_index, 1)
        self.assertEqual(topic_matches[0].relative_end_index, 2)

    def test_coreference_double_match_on_governor(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "I saw a big man. The man walked")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A big man", relation_score=20, single_word_score=10, single_word_any_tag_score=5)
        self.assertEqual(int(topic_matches[0].score), 34)
        self.assertEqual(topic_matches[0].sentences_start_index, 0)
        self.assertEqual(topic_matches[0].sentences_end_index, 8)
        self.assertEqual(topic_matches[0].start_index, 3)
        self.assertEqual(topic_matches[0].end_index, 7)
        self.assertEqual(topic_matches[0].relative_start_index, 3)
        self.assertEqual(topic_matches[0].relative_end_index, 7)

    def test_coreference_double_match_same_distance(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.parse_and_register_document(
                "The man was big. Man walked.")
        topic_matches = holmes_manager_coref.topic_match_documents_against("A big man",
                relation_score=20, single_word_score=10, single_word_any_tag_score=5)
        self.assertEqual(int(topic_matches[0].score), 34)
        self.assertEqual(topic_matches[0].sentences_start_index, 0)
        self.assertEqual(topic_matches[0].sentences_end_index, 7)
        self.assertEqual(topic_matches[0].start_index, 1)
        self.assertEqual(topic_matches[0].end_index, 5)
        self.assertEqual(topic_matches[0].relative_start_index, 1)
        self.assertEqual(topic_matches[0].relative_end_index, 5)

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
        [{'document_label': '', 'text': 'A dog chased a cat. A cat.', 'text_to_match': 'The dog chased the cat', 'rank': '1=', 'sentences_character_start_index_in_document': 515, 'sentences_character_end_index_in_document': 541, 'score': 99.34666666666668, 'word_infos': [[2, 5, 'overlapping_relation', False], [6, 12, 'overlapping_relation', False], [15, 18, 'overlapping_relation', True], [22, 25, 'single', False]]}, {'document_label': '', 'text': 'A dog chased a cat.', 'text_to_match': 'The dog chased the cat', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 19, 'score': 99.34666666666668, 'word_infos': [[2, 5, 'overlapping_relation', False], [6, 12, 'overlapping_relation', False], [15, 18, 'overlapping_relation', True]]}, {'document_label': 'animals', 'text': 'Dogs and cats.', 'text_to_match': 'The dog chased the cat', 'rank': '3', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 14, 'score': 9.866666666666667, 'word_infos': [[0, 4, 'single', False], [9, 13, 'single', True]]}])
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", tied_result_quotient=0.01)
        self.assertEqual(topic_match_dictionaries,
        [{'document_label': '', 'text': 'A dog chased a cat. A cat.', 'text_to_match': 'The dog chased the cat', 'rank': '1=', 'sentences_character_start_index_in_document': 515, 'sentences_character_end_index_in_document': 541, 'score': 99.34666666666668, 'word_infos': [[2, 5, 'overlapping_relation', False], [6, 12, 'overlapping_relation', False], [15, 18, 'overlapping_relation', True], [22, 25, 'single', False]]}, {'document_label': '', 'text': 'A dog chased a cat.', 'text_to_match': 'The dog chased the cat', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 19, 'score': 99.34666666666668, 'word_infos': [[2, 5, 'overlapping_relation', False], [6, 12, 'overlapping_relation', False], [15, 18, 'overlapping_relation', True]]}, {'document_label': 'animals', 'text': 'Dogs and cats.', 'text_to_match': 'The dog chased the cat', 'rank': '1=', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 14, 'score': 9.866666666666667, 'word_infos': [[0, 4, 'single', False], [9, 13, 'single', True]]}])

    def test_dictionaries_with_multiword_in_relation_not_final(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document("Richard Paul Hudson came home")
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "Richard Paul Hudson was coming")
        self.assertEqual(topic_match_dictionaries,
        [{'document_label': '', 'text': 'Richard Paul Hudson came home', 'text_to_match': 'Richard Paul Hudson was coming', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 29, 'score': 40.8, 'word_infos': [[0, 19, 'relation', False], [20, 24, 'relation', True]]}])

    def test_dictionaries_with_multiword_alone(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document("Richard Paul Hudson")
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "Richard Paul Hudson")
        self.assertEqual(topic_match_dictionaries,
        [{'document_label': '', 'text': 'Richard Paul Hudson', 'text_to_match': 'Richard Paul Hudson', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 19, 'score': 8.92, 'word_infos': [[0, 19, 'single', True]]}])

    def test_dictionaries_with_multiword_alone_and_entity_token_in_text_to_match(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document("Richard Paul Hudson")
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "ENTITYPERSON")
        self.assertEqual(topic_match_dictionaries,
        [{'document_label': '', 'text': 'Richard Paul Hudson', 'text_to_match': 'ENTITYPERSON', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 19, 'score': 5.0, 'word_infos': [[0, 19, 'single', True]]}])

    def test_result_ordering_by_match_length_different_documents(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document("A dog chased a cat.",'1')
        holmes_manager_coref.parse_and_register_document("A dog chased a cat. A cat.",'2')
        topic_matches = holmes_manager_coref.topic_match_documents_against(
                "The dog chased the cat")
        self.assertEqual(topic_matches[0].end_index, 7)
        self.assertEqual(topic_matches[1].end_index, 4)

    def test_filtering_with_topic_matches(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document("The dog chased the cat", "T11")
        holmes_manager_coref.parse_and_register_document("The dog chased the cat", "T12")
        holmes_manager_coref.parse_and_register_document("The dog chased the cat", "T21")
        holmes_manager_coref.parse_and_register_document("The dog chased the cat", "T22")
        topic_matches = \
                holmes_manager_coref.topic_match_documents_against(
                "The dog chased the cat")
        self.assertEqual(len(topic_matches), 4)
        topic_matches = \
                holmes_manager_coref.topic_match_documents_against(
                "The dog chased the cat", document_label_filter="T")
        self.assertEqual(len(topic_matches), 4)
        topic_matches = \
                holmes_manager_coref.topic_match_documents_against(
                "The dog chased the cat", document_label_filter="T1")
        self.assertEqual(len(topic_matches), 2)
        topic_matches = \
                holmes_manager_coref.topic_match_documents_against(
                "The dog chased the cat", document_label_filter="T22")
        self.assertEqual(len(topic_matches), 1)
        topic_matches = \
                holmes_manager_coref.topic_match_documents_against(
                "The dog chased the cat", document_label_filter="X")
        self.assertEqual(len(topic_matches), 0)

    def test_filtering_with_topic_match_dictionaries(self):
        holmes_manager_coref.remove_all_documents()
        holmes_manager_coref.remove_all_search_phrases()
        holmes_manager_coref.parse_and_register_document("The dog chased the cat", "T11")
        holmes_manager_coref.parse_and_register_document("The dog chased the cat", "T12")
        holmes_manager_coref.parse_and_register_document("The dog chased the cat", "T21")
        holmes_manager_coref.parse_and_register_document("The dog chased the cat", "T22")
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat")
        self.assertEqual(len(topic_match_dictionaries), 4)
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", document_label_filter="T")
        self.assertEqual(len(topic_match_dictionaries), 4)
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", document_label_filter="T1")
        self.assertEqual(len(topic_match_dictionaries), 2)
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", document_label_filter="T22")
        self.assertEqual(len(topic_match_dictionaries), 1)
        topic_match_dictionaries = \
                holmes_manager_coref.topic_match_documents_returning_dictionaries_against(
                "The dog chased the cat", document_label_filter="X")
        self.assertEqual(len(topic_match_dictionaries), 0)
