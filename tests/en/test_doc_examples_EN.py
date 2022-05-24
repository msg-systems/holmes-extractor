import unittest
import holmes_extractor as holmes

holmes_manager = holmes.Manager(model="en_core_web_lg", number_of_workers=1)
holmes_manager.register_search_phrase("A big dog chases a cat")


class EnglishDocumentationExamplesTest(unittest.TestCase):

    positive_examples = (
        "A big dog chased a cat",
        "The big dog would not stop chasing the cat",
        "The big dog who was tired chased the cat",
        "The cat was chased by the big dog",
        "The cat always used to be chased by the big dog",
        "The big dog was going to chase the cat",
        "The big dog decided to chase the cat",
        "The cat was afraid of being chased by the big dog",
        "I saw a cat-chasing big dog",
        "The cat the big dog chased was scared",
        "The big dog chasing the cat was a problem",
        "There was a big dog that was chasing a cat",
        "The cat chase by the big dog",
        "There was a big dog and it was chasing a cat.",
        "I saw a big dog. My cat was afraid of being chased by the dog.",
        "There was a big dog. His name was Fido. He was chasing my cat.",
        "A dog appeared. It was chasing a cat. It was very big.",
        "The cat sneaked back into our lounge because a big dog had been chasing her outside.",
        "Our big dog was excited because he had been chasing a cat.",
    )

    def test_positive_examples(self):
        for positive_example in self.positive_examples:
            with self.subTest():
                assert len(holmes_manager.match(document_text=positive_example)) == 1

    negative_examples = (
        "The dog chased a big cat",
        "The big dog and the cat chased about",
        "The big dog chased a mouse but the cat was tired",
        "The big dog always used to be chased by the cat",
        "The big dog the cat chased was scared",
        "Our big dog was upset because he had been chased by a cat.",
        "The dog chase of the big cat",
    )

    def test_negative_examples(self):
        for negative_example in self.negative_examples:
            with self.subTest():
                assert len(holmes_manager.match(document_text=negative_example)) == 0
