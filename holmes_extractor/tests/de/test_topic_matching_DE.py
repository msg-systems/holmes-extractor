import unittest
import holmes_extractor as holmes

holmes_manager = holmes.Manager('de_core_news_md')

class GermanTopicMatchingTest(unittest.TestCase):

    def _check_equals(self, text_to_match, document_text, highest_score):
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(document_text)
        topic_matches = holmes_manager.topic_match_documents_against(text_to_match,
                relation_score=20, single_word_score=10)
        self.assertEqual(int(topic_matches[0].score), highest_score)

    def test_direct_matching(self):
        self._check_equals("Eine Pflanze wächst", "Eine Pflanze wächst", 29)

    def test_direct_matching_nonsense_word(self):
        self._check_equals("Ein Gegwghg wächst", "Ein Gegwghg wächst", 29)

    def test_entity_matching(self):
        self._check_equals("Ein ENTITYPER singt", "Peter singt", 29)

    def test_entitynoun_matching(self):
        self._check_equals("Ein ENTITYNOUN singt", "Ein Vogel singt", 20)

    def test_matching_only_adjective(self):
        self._check_equals("nett", "nett", 10)

    def test_matching_only_adjective_where_noun(self):
        self._check_equals("netter Ort", "nett", 10)

    def test_matching_no_change_from_template_words(self):
        self._check_equals("Eine beschriebene Sache", "Eine beschriebene Sache", 30)

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
