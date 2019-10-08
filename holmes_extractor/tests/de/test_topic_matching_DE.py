import unittest
import holmes_extractor as holmes

holmes_manager = holmes.Manager('de_core_news_md')
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

    def test_reverse_matching_noun(self):
        self._check_equals("Ein König mit einem Land", "Ein Präsident mit einem Land", 49,
                holmes_manager_with_embeddings)

    def test_reverse_matching_noun_control_no_embeddings(self):
        self._check_equals("Ein König mit einem Land", "Ein Präsident mit einem Land", 29,
                holmes_manager)

    def test_reverse_matching_noun_control_same_word(self):
        self._check_equals("Ein König mit einem Land", "Ein König mit einem Land", 75,
                holmes_manager)

    def test_reverse_matching_verb(self):
        self._check_equals("Ein Kind schrie", "Das Kind weinte", 20,
                holmes_manager_with_embeddings)

    def test_reverse_matching_verb_control_no_embeddings(self):
        self._check_equals("Ein Kind schrie", "Das Kind weinte", 10,
                holmes_manager)

    def test_reverse_matching_verb_control_same_word(self):
        self._check_equals("Ein Kind schrie", "Das Kind schrie", 34,
                holmes_manager)

    def test_reverse_matching_only(self):
        self._check_equals("mit einer Idee",
                "mit einer Idee", 29,
                holmes_manager)

    def test_multiword_in_text_to_search_and_in_document_not_root(self):
        self._check_equals("Richard Paul Hudson kam",
                "Ich sah Richard Paul Hudson", 24,
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
                "Ich sah Richard Paul Hudson", 24,
                holmes_manager)

    def test_multiword_in_text_to_search_single_word_in_document_root(self):
        self._check_equals("der müde Hudson",
                "Ich sah Richard Paul Hudson", 10,
                holmes_manager)

    def test_multiword_in_text_to_search_dependent_words_in_document_root(self):
        self._check_equals("Richard Paul kam",
                "Ich sah Richard Paul Hudson", 9,
                holmes_manager)

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
