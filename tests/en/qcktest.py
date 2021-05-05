import unittest
import holmes_extractor as holmes
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join(
    (script_directory, 'test_ontology.owl')))
coref_holmes_manager = holmes.Manager(model='en_core_web_lg', ontology=ontology,
                                      perform_coreference_resolution=True)
coref_holmes_manager.register_search_phrase("A dog chases a cat")
coref_holmes_manager.register_search_phrase("A big horse chases a cat")
coref_holmes_manager.register_search_phrase("A tiger chases a little cat")
coref_holmes_manager.register_search_phrase("A big lion chases a cat")
coref_holmes_manager.register_search_phrase("An ENTITYPERSON needs insurance")
coref_holmes_manager.register_search_phrase("University for four years")
coref_holmes_manager.register_search_phrase("A big company makes a loss")
coref_holmes_manager.register_search_phrase(
    "A dog who chases rats chases mice")
coref_holmes_manager.register_search_phrase("A tired dog")
coref_holmes_manager.register_search_phrase("A panther chases a panther")
coref_holmes_manager.register_search_phrase("A leopard chases a leopard")
coref_holmes_manager.register_search_phrase("A holiday is hard to find")
coref_holmes_manager.register_search_phrase("A man sings")
coref_holmes_manager.register_search_phrase("Somebody finds a policy")
coref_holmes_manager.register_search_phrase(
    "Somebody writes a book about an animal")
coref_holmes_manager.register_search_phrase("Hermione breaks")
coref_holmes_manager.register_search_phrase("Somebody attempts to explain")
coref_holmes_manager.register_search_phrase("An adopted boy")
coref_holmes_manager.register_search_phrase("A running boy")
no_coref_holmes_manager = holmes.Manager(model='en_core_web_lg', ontology=ontology,
                                         perform_coreference_resolution=False)
no_coref_holmes_manager.register_search_phrase("A dog chases a cat")
embeddings_coref_holmes_manager = holmes.Manager(model='en_core_web_lg',
                                                 overall_similarity_threshold=0.85)
embeddings_coref_holmes_manager.register_search_phrase('A man loves a woman')


class CoreferenceEnglishMatchingTest(unittest.TestCase):

    def _check_word_match(self, match, word_match_index, document_token_index, extracted_word):
        word_match = match.word_matches[word_match_index]
        self.assertEqual(word_match.document_token.i, document_token_index)
        self.assertEqual(word_match.extracted_word, extracted_word)

    def test_simple_pronoun_coreference_diff_sentence_conjunction_righthand_is_pronoun(self):
        coref_holmes_manager.remove_all_documents()
        coref_holmes_manager.parse_and_register_document(
            "I talked to Jane Jones. Both Peter Jones and she needed insurance.")
        matches = coref_holmes_manager.match()
        print(matches)
        self.assertEqual(len(matches), 2)
        self._check_word_match(matches[0], 0, 8, 'Peter Jones')
        self._check_word_match(matches[1], 0, 4, 'Jane Jones')
