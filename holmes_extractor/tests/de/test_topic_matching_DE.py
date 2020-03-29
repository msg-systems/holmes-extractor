import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
holmes_manager = holmes.Manager('de_core_news_md', ontology=ontology)
holmes_manager_with_embeddings = holmes.Manager('de_core_news_md',
        overall_similarity_threshold=0.65)

class GermanTopicMatchingTest(unittest.TestCase):

    def _check_equals(self, text_to_match, document_text, highest_score, manager=holmes_manager):
        manager.remove_all_documents()
        manager.parse_and_register_document(document_text)
        topic_matches = manager.topic_match_documents_against(text_to_match,
                relation_score=20, reverse_only_relation_score=15,
                single_word_score=10, single_word_any_tag_score=5)
        self.assertEqual(int(topic_matches[0].score), highest_score)

    def test_direct_matching(self):
        self._check_equals("Eine Pflanze wächst", "Eine Pflanze wächst", 34)

    def test_direct_matching_nonsense_word(self):
        self._check_equals("Ein Gegwghg wächst", "Ein Gegwghg wächst", 34)

    def test_entity_matching(self):
        self._check_equals("Ein ENTITYPER singt", "Peter singt", 34)

    def test_entitynoun_matching(self):
        self._check_equals("Ein ENTITYNOUN singt", "Ein Vogel singt", 25)

    def test_matching_only_adjective(self):
        self._check_equals("nett", "nett", 5)

    def test_matching_only_adjective_where_noun(self):
        self._check_equals("netter Ort", "nett", 5)

    def test_matching_no_change_from_template_words(self):
        self._check_equals("Eine beschriebene Sache", "Eine beschriebene Sache", 34)

    def test_reverse_only_parent_lemma_aux_threeway(self):
        self._check_equals("Der Esel hat ein Dach", "Der Esel hat ein Dach", 68,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_aux_twoway(self):
        self._check_equals("Der Esel hat ein Dach", "Der Esel hat ein Haus", 29,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_aux_auxiliary_threeway(self):
        self._check_equals("Der Esel hat ein Dach", "Der Esel wird ein Dach haben", 69,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_aux_auxiliary_twoway(self):
        self._check_equals("Der Esel hat ein Dach", "Der Esel wird ein Haus haben", 29,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_aux_modal_threeway(self):
        self._check_equals("Der Esel hat ein Dach", "Der Esel soll ein Dach haben", 69,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_aux_modal_twoway(self):
        self._check_equals("Der Esel hat ein Dach", "Der Esel soll ein Haus haben", 29,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_verb_threeway(self):
        self._check_equals("Der Esel macht ein Dach", "Der Esel macht ein Dach", 68,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_verb_twoway(self):
        self._check_equals("Der Esel macht ein Dach", "Der Esel macht ein Haus", 29,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_threeway_control(self):
        self._check_equals("Der Esel malt ein Dach an", "Der Esel malt ein Dach an", 82,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_twoway_control_no_embedding_based_match(self):
        self._check_equals("Der Esel malt ein Dach an", "Der Esel malt eine Maus an", 34,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_be(self):
        self._check_equals("Ein Präsident ist ein Politiker", "Ein Präsident ist ein Politiker", 68,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_be_reversed(self):
        self._check_equals("Ein Präsident ist ein Politiker", "Ein Politiker ist ein Präsident", 24,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_become(self):
        self._check_equals("Ein Präsident wird ein Politiker", "Ein Präsident wird ein Politiker", 68,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_become_reversed(self):
        self._check_equals("Ein Präsident wird ein Politiker", "Ein Politiker wird ein Präsident", 24,
                holmes_manager_with_embeddings)

    def test_reverse_only_parent_lemma_aux_in_document(self):
        self._check_equals("Ein Esel hat ein Dach", "Ein Esel hat ein Dach gesehen", 24,
                holmes_manager_with_embeddings)

    def test_reverse_matching_noun(self):
        self._check_equals("Ein König mit einem Land", "Ein Präsident mit einem Land", 49,
                holmes_manager_with_embeddings)

    def test_reverse_matching_noun_control_no_embeddings(self):
        self._check_equals("Ein König mit einem Land", "Ein Präsident mit einem Land", 29,
                holmes_manager)

    def test_reverse_matching_noun_control_same_word(self):
        self._check_equals("Ein König mit einem Land", "Ein König mit einem Land", 75,
                holmes_manager)

    def test_reverse_matching_only(self):
        self._check_equals("mit einer Idee",
                "mit einer Idee", 29,
                holmes_manager)

    def test_multiword_in_text_to_search_and_in_document_not_root(self):
        self._check_equals("Richard Paul Hudson kam",
                "Ich sah Richard Paul Hudson", 19,
                holmes_manager)

    def test_multiword_in_text_to_search_single_word_in_document_not_root(self):
        self._check_equals("Hudson kam",
                "Ich sah Richard Paul Hudson", 10,
                holmes_manager)

    def test_multiword_in_text_to_search_dependent_words_in_document_not_root(self):
        self._check_equals("Richard Paul kam",
                "Ich sah Richard Paul Hudson", 9,
                holmes_manager)

    def test_multiword_in_text_to_search_and_in_document_root(self):
        self._check_equals("der müde Richard Paul Hudson",
                "Ich sah Richard Paul Hudson", 19,
                holmes_manager)

    def test_multiword_in_text_to_search_single_word_in_document_root(self):
        self._check_equals("der müde Hudson",
                "Ich sah Richard Paul Hudson", 10,
                holmes_manager)

    def test_multiword_in_text_to_search_dependent_words_in_document_root(self):
        self._check_equals("Richard Paul kam",
                "Ich sah Richard Paul Hudson", 9,
                holmes_manager)

    def test_double_match(self):
        self._check_equals("vier Ochsen und sechs Ochsen",
                "vier Ochsen", 34,
                holmes_manager_with_embeddings)

    def test_separate_words_in_text_to_match_subwords_in_document_text_with_fugen_s(self):
        self._check_equals("Die Extraktion der Information",
                "Informationsextraktion", 40,
                holmes_manager)

    def test_separate_words_in_text_to_match_subwords_in_document_text_without_fugen_s(self):
        self._check_equals("Eine Symphonie des Mozarts",
                "Mozartsymphonien", 40,
                holmes_manager)

    def test_subwords_in_text_to_match_separate_words_in_document_text_with_fugen_s(self):
        self._check_equals("Informationsextraktion",
                "Die Extraktion der Information", 29,
                holmes_manager)

    def test_subwords_in_text_to_match_separate_words_in_document_text_without_fugen_s(self):
        self._check_equals("Mozartsymphonien",
                "Eine Symphonie von Mozart", 29,
                holmes_manager)

    def test_subwords_in_text_to_match_subwords_in_document_text_with_fugen_s(self):
        self._check_equals("Informationsextraktion",
                "Informationsextraktion", 10,
                holmes_manager)

    def test_subwords_in_text_to_match_subwords_in_document_text_without_fugen_s(self):
        self._check_equals("Mozartsymphonie",
                "Mozartsymphonie", 10,
                holmes_manager)

    def test_subwords_in_text_to_match_subwords_in_document_text_lemmatization_failed(self):
        self._check_equals("Mozartsymphonien",
                "Mozartsymphonie", 30,
                holmes_manager)

    def test_subwords_conjunction_in_text_to_match(self):
        self._check_equals("Mozart- und Beethovensymphonie",
                "Mozartsymphonie", 30,
                holmes_manager)

    def test_subwords_conjunction_in_document_text(self):
        self._check_equals("Mozartsymphonie",
                "Mozart- und Beethovensymphonie", 30,
                holmes_manager)

    def test_subwords_conjunction_in_text_to_match_and_document_text(self):
        self._check_equals("Mozart- und Mahlersymphonie",
                "Mozart- und Beethovensymphonie", 30,
                holmes_manager)

    def test_subword_matches_verbal_expression(self):
        self._check_equals("Katzenjagen",
                "Ein Hund jagt eine Katze", 29,
                holmes_manager)

    def test_disjunct_relation_mapping_within_subword(self):
        self._check_equals("Extraktion von Information und Entführung von Löwen",
                "Informationsextraktionsentführung von Löwen", 78)

    def test_overlapping_relation_mapping_within_subword(self):
        self._check_equals("Extraktion von Information und Löwen",
                "Informationsextraktion von Löwen", 87)

    def test_word_with_subwords_matches_single_word_linked_via_ontology(self):
        self._check_equals("Komputerlinguistik",
                "Linguistik", 9,
                holmes_manager)

    def test_word_with_subwords_matches_single_word_linked_via_ontology_control(self):
        self._check_equals("Theorielinguistik",
                "Linguistik", 5,
                holmes_manager)

    def test_single_word_matches_word_with_subwords_linked_via_ontology(self):
        self._check_equals("Linguistik",
                "Komputerlinguistik", 9,
                holmes_manager)

    def test_single_word_matches_word_with_subwords_linked_via_ontology_control(self):
        self._check_equals("Linguistik",
                "Theorielinguistik", 10,
                holmes_manager)

    def test_embedding_matching_with_subwords(self):
        self._check_equals("Eine Königsabdanken",
                "Der Prinz dankte ab", 14,
                holmes_manager_with_embeddings)

    def test_embedding_matching_with_subwords_control(self):
        self._check_equals("Eine Königsabdanken",
                "Der Prinz dankte ab", 5,
                holmes_manager)

    def test_sibling_match_with_higher_similarity_and_subwords_1(self):
        self._check_equals("Das Abdanken eines Königs",
                "Ein Königs- und Prinzenabdanken", 40,
                holmes_manager_with_embeddings)

    def test_sibling_match_with_higher_similarity_no_embeddings_control(self):
        self._check_equals("Das Abdanken eines Königs",
                "Ein Königs- und Prinzenabdanken", 40,
                holmes_manager)

    def test_sibling_match_with_higher_similarity_and_subwords_2(self):
        self._check_equals("Das Abdanken eines Königs",
                "Ein Prinzen- und Königsabdanken", 40,
                holmes_manager_with_embeddings)

    def test_sibling_match_with_higher_similarity_and_subwords_3(self):
        self._check_equals("Ein Königs- und Prinzenabdanken",
                "Das Abdanken eines Königs", 29,
                holmes_manager_with_embeddings)

    def test_sibling_match_with_higher_similarity_and_subwords_4(self):
        self._check_equals("Ein Prinzen- und Königsabdanken",
                "Das Abdanken eines Königs", 29,
                holmes_manager_with_embeddings)

    def test_sibling_match_with_higher_similarity_and_subwords_control_1(self):
        self._check_equals("Das Abdanken eines Königs",
                "Ein Prinzen- und Informationsabdanken", 19,
                holmes_manager_with_embeddings)

    def test_sibling_match_with_higher_similarity_and_subwords_control_2(self):
        self._check_equals("Ein Prinzen- und Informationsabdanken",
                "Das Abdanken eines Königs", 14,
                holmes_manager_with_embeddings)

    def test_entity_matching_with_single_word_subword_match(self):
        self._check_equals("Ein ENTITYLOC singt", "Informationsextraktion hat sich durchgesetzt", 10)

    def test_entity_matching_with_relation_subword_match(self):
        self._check_equals("Ein ENTITYLOC setzt sich durch", "Informationsextraktion hat sich durchgesetzt", 34)

    def test_entitynoun_matching_with_relation_subword_match(self):
        self._check_equals("Ein ENTITYNOUN setzt sich durch", "Informationsextraktion hat sich durchgesetzt", 25)

    def test_derivation_in_subwords_1(self):
        self._check_equals("Informationextraktion", "Informierung wird extrahiert", 29)

    def test_derivation_in_subwords_2(self):
        self._check_equals("Informierung wird extrahiert", "Informationsextraktion", 35)

    def test_indexes(self):
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(
                "Dies ist ein irrelevanter Satz. Ich glaube, dass eine Pflanze wächst.")
        topic_matches = holmes_manager.topic_match_documents_against("Eine Pflanze wächst")
        self.assertEqual(topic_matches[0].sentences_start_index, 6)
        self.assertEqual(topic_matches[0].sentences_end_index, 13)
        self.assertEqual(topic_matches[0].start_index, 11)
        self.assertEqual(topic_matches[0].end_index, 12)
        self.assertEqual(topic_matches[0].relative_start_index, 5)
        self.assertEqual(topic_matches[0].relative_end_index, 6)

    def test_same_index_different_documents(self):
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(
                "Eine Pflanze wächst.", '1')
        holmes_manager.parse_and_register_document(
                "Eine Pflanze wächst.", '2')
        topic_matches = holmes_manager.topic_match_documents_against("Eine Pflanze wächst")
        self.assertEqual(len(topic_matches), 2)
        self.assertEqual(topic_matches[0].document_label, '1')
        self.assertEqual(topic_matches[1].document_label, '2')
        self.assertEqual(topic_matches[0].start_index, 1)
        self.assertEqual(topic_matches[0].end_index, 2)
        self.assertEqual(topic_matches[1].start_index, 1)
        self.assertEqual(topic_matches[1].end_index, 2)

    def test_suppressed_relation_matching_picked_up_during_reverse_matching_subwords(self):
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(
                "Der König dankte ab. Die Königin dankte ab.")
        topic_matches = holmes_manager.topic_match_documents_against("Das Königabdanken",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_relation_matching = 1,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 29)

    def test_reverse_matching_suppressed_with_embedding_reverse_matching_parent(self):
        holmes_manager_with_embeddings.remove_all_documents()
        holmes_manager_with_embeddings.parse_and_register_document("Der Prinz dankte ab")
        topic_matches = holmes_manager_with_embeddings.topic_match_documents_against(
                "Das Königsabdanken",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 5)

    def test_reverse_matching_suppressed_with_embedding_reverse_matching_parent_control(self):
        holmes_manager_with_embeddings.remove_all_documents()
        holmes_manager_with_embeddings.parse_and_register_document("Der Prinz dankte ab")
        topic_matches = holmes_manager_with_embeddings.topic_match_documents_against(
                "Das Königsabdanken",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_embedding_matching = 1)
        self.assertEqual(int(topic_matches[0].score), 14)

    def test_reverse_matching_suppressed_with_embedding_reverse_matching_child(self):
        holmes_manager_with_embeddings.remove_all_documents()
        holmes_manager_with_embeddings.parse_and_register_document("Der König vom Abdanken")
        topic_matches = holmes_manager_with_embeddings.topic_match_documents_against(
                "Die Abdankenprinzen",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_embedding_matching = 0)
        self.assertEqual(int(topic_matches[0].score), 5)

    def test_reverse_matching_suppressed_with_embedding_reverse_matching_child_control(self):
        holmes_manager_with_embeddings.remove_all_documents()
        holmes_manager_with_embeddings.parse_and_register_document("Der König vom Abdanken")
        topic_matches = holmes_manager_with_embeddings.topic_match_documents_against(
                "Die Abdankenprinzen",
                relation_score=20, reverse_only_relation_score=15, single_word_score=10,
                single_word_any_tag_score=5,
                maximum_number_of_single_word_matches_for_embedding_matching = 1)
        self.assertEqual(int(topic_matches[0].score), 14)

    def test_disjunct_relation_mapping_within_subword_dictionaries(self):
        holmes_manager.remove_all_documents()
        holmes_manager.remove_all_search_phrases()
        holmes_manager.parse_and_register_document("Informationssymphonieentführung von Löwen")
        topic_match_dictionaries = \
                holmes_manager.topic_match_documents_returning_dictionaries_against(
                "Symphonie von Information und Entführung von Löwen")
        self.assertEqual(topic_match_dictionaries,
        [{'document_label': '', 'text': 'Informationssymphonieentführung von Löwen', 'text_to_match': 'Symphonie von Information und Entführung von Löwen', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 41, 'score': 78.0, 'word_infos': [[0, 11, 'relation', False, "Matches INFORMATION directly."], [12, 21, 'relation', False, "Matches SYMPHONIE directly."], [21, 31, 'relation', False, "Matches ENTFÜHRUNG directly."], [36, 41, 'relation', True, "Matches LÖWE directly."]]}])

    def test_overlapping_relation_mapping_within_subword_dictionaries(self):
        holmes_manager.remove_all_documents()
        holmes_manager.remove_all_search_phrases()
        holmes_manager.parse_and_register_document("Informationsextraktion von Löwen")
        topic_match_dictionaries = \
                holmes_manager.topic_match_documents_returning_dictionaries_against(
                "Extraktion von Information und Löwen")
        print(topic_match_dictionaries)

        self.assertEqual(topic_match_dictionaries,
        [{'document_label': '', 'text': 'Informationsextraktion von Löwen', 'text_to_match': 'Extraktion von Information und Löwen', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 32, 'score': 102.33333333333334, 'word_infos': [[0, 11, 'overlapping_relation', False, "Matches INFORMATION directly."], [12, 22, 'overlapping_relation', False, "Matches EXTRAKTION directly."], [27, 32, 'overlapping_relation', True, "Matches LÖWE directly."]]}])

    def test_subword_dictionaries_subword_is_not_peak(self):
        holmes_manager.remove_all_documents()
        holmes_manager.remove_all_search_phrases()
        holmes_manager.parse_and_register_document("Information und Löwen wurden genommen")
        topic_match_dictionaries = \
                holmes_manager.topic_match_documents_returning_dictionaries_against(
                "Informationsnehmen der Löwen")
        print(topic_match_dictionaries)
        self.assertEqual(topic_match_dictionaries,
        [{'document_label': '', 'text': 'Information und Löwen wurden genommen', 'text_to_match': 'Informationsnehmen der Löwen', 'rank': '1', 'sentences_character_start_index_in_document': 0, 'sentences_character_end_index_in_document': 37, 'score': 98.75999999999999, 'word_infos': [[0, 11, 'overlapping_relation', False, "Matches INFORMATION directly."], [16, 21, 'overlapping_relation', False, "Matches LÖWE directly."], [29, 37, 'overlapping_relation', True, "Matches NEHMEN directly."]]}])
