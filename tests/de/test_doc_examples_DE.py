import unittest
import holmes_extractor as holmes

holmes_manager = holmes.Manager(model="de_core_news_lg", number_of_workers=1)
holmes_manager.register_search_phrase("Ein großer Hund jagt eine Katze")


class EnglishDocumentationExamplesTest(unittest.TestCase):

    positive_examples = (
        "Der große Hund hat die Katze ständig gejagt",
        "Der große Hund, der müde war, jagte die Katze",
        "Die Katze wurde vom großen Hund gejagt",
        "Die Katze wurde immer wieder durch den großen Hund gejagt",
        "Der große Hund wollte die Katze jagen",
        "Der große Hund entschied sich, die Katze zu jagen",
        "Die Katze, die der große Hund gejagt hatte, hatte Angst",
        "Dass der große Hund die Katze jagte, war ein Problem",
        "Es gab einen großen Hund, der eine Katze jagte",
        "Die Katzenjagd durch den großen Hund",
        "Es gab einen großen Hund und er jagte eine Katze",
        "Es gab einen großen Hund. Er hieß Fido. Er jagte meine Katze",
        "Es erschien ein Hund. Er jagte eine Katze. Er war sehr groß.",
        "Die Katze schlich sich in unser Wohnzimmer zurück, weil ein großer Hund sie draußen gejagt hatte",
        "Unser großer Hund war aufgeregt, weil er eine Katze gejagt hatte",
    )

    def test_positive_examples(self):
        for positive_example in self.positive_examples:
            with self.subTest():
                assert len(holmes_manager.match(document_text=positive_example)) == 1

    negative_examples = (
        "Der Hund jagte eine große Katze",
        "Die Katze jagte den großen Hund",
        "Der große Hund und die Katze jagten",
        "Der große Hund jagte eine Maus aber die Katze war müde",
        "Der große Hund wurde ständig von der Katze gejagt",
        "Der große Hund entschloss sich, von der Katze gejagt zu werden",
        "Die Hundejagd durch den große Katze",
    )

    def test_negative_examples(self):
        for negative_example in self.negative_examples:
            with self.subTest():
                assert len(holmes_manager.match(document_text=negative_example)) == 0
