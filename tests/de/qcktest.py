import unittest
import spacy
import coreferee
import holmes_extractor

nlp = spacy.load('de_core_news_lg')
nlp.add_pipe('coreferee')
nlp.add_pipe('holmes')

class GermanSemanticAnalyzerTest(unittest.TestCase):

    def test_question_word_control_1(self):
        doc = nlp(". Wem hast Du geholfen?")
        for token in doc:
            self.assertFalse(token._.holmes.is_initial_question_word)

    def test_question_word_control_2(self):
        doc = nlp("Du bist gekommen wegen wem?")
        for token in doc:
            self.assertFalse(token._.holmes.is_initial_question_word)
