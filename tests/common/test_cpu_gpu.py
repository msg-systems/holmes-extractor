import unittest
from thinc.api import prefer_gpu, require_cpu
import holmes_extractor as holmes

class CpuGpuTest(unittest.TestCase):

    def test_document_based_structural_matching_cpu_gpu(self):
        require_cpu()
        holmes_manager = holmes.Manager('en_core_web_sm', number_of_workers=2)
        holmes_manager.parse_and_register_document(
            document_text="The dog chased the cat.", label='pets')
        prefer_gpu()
        holmes_manager.register_search_phrase("A dog chases a cat")
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_document_based_structural_matching_gpu_cpu(self):
        prefer_gpu()
        holmes_manager = holmes.Manager('en_core_web_sm', number_of_workers=2)
        holmes_manager.parse_and_register_document(
            document_text="The dog chased the cat.", label='pets')
        require_cpu()
        holmes_manager.register_search_phrase("A dog chases a cat")
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_search_phrase_based_structural_matching_cpu_gpu(self):
        require_cpu()
        holmes_manager = holmes.Manager('en_core_web_sm', number_of_workers=2)
        holmes_manager.register_search_phrase("A dog chases a cat")
        prefer_gpu()
        holmes_manager.parse_and_register_document(
            document_text="The dog chased the cat.", label='pets')
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_search_phrase_based_structural_matching_gpu_cpu(self):
        prefer_gpu()
        holmes_manager = holmes.Manager('en_core_web_sm', number_of_workers=2)
        holmes_manager.register_search_phrase("A dog chases a cat")
        require_cpu()
        holmes_manager.parse_and_register_document(
            document_text="The dog chased the cat.", label='pets')
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_topic_matching_cpu_gpu(self):
        require_cpu()
        holmes_manager = holmes.Manager('en_core_web_sm', number_of_workers=2)
        holmes_manager.parse_and_register_document(
            document_text="The dog chased the cat.", label='pets')
        prefer_gpu()
        topic_matches = holmes_manager.topic_match_documents_against("A dog chases a cat")
        self.assertEqual(len(topic_matches), 1)

    def test_topic_matching_gpu_cpu(self):
        prefer_gpu()
        holmes_manager = holmes.Manager('en_core_web_sm', number_of_workers=2)
        holmes_manager.parse_and_register_document(
            document_text="The dog chased the cat.", label='pets')
        require_cpu()
        topic_matches = holmes_manager.topic_match_documents_against("A dog chases a cat")
        self.assertEqual(len(topic_matches), 1)

    def test_supervised_document_classification_cpu_gpu(self):
        require_cpu()
        holmes_manager = holmes.Manager('en_core_web_sm', number_of_workers=2)
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False
        )
        sttb.parse_and_register_training_document("An animal", "animal", "d4")
        sttb.parse_and_register_training_document("A computer", "computers", "d5")
        sttb.prepare()
        # With so little training data, the NN does not consistently learn correctly
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
                list(stc.parse_and_classify("You are an animal.").keys())[0] == "animal"
            ):
                break
            if i == 20:
                self.assertTrue(
                list(stc.parse_and_classify("You are an animal.").keys())[0] == "animal"
                )

        prefer_gpu()
        self.assertTrue(
            list(stc.parse_and_classify("You are an animal.").keys())[0] == "animal")
        self.assertIsNone(
            stc.parse_and_classify("My name is Charles and I like sewing.")
        )

    def test_supervised_document_classification_gpu_cpu(self):
        prefer_gpu()
        holmes_manager = holmes.Manager('en_core_web_sm', number_of_workers=2)
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False
        )
        sttb.parse_and_register_training_document("An animal", "animal", "d4")
        sttb.parse_and_register_training_document("A computer", "computers", "d5")
        sttb.prepare()
        # With so little training data, the NN does not consistently learn correctly
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
                list(stc.parse_and_classify("You are an animal.").keys())[0] == "animal"
            ):
                break
            if i == 20:
                self.assertTrue(
                list(stc.parse_and_classify("You are an animal.").keys())[0] == "animal"
                )

        require_cpu()
        self.assertTrue(
            list(stc.parse_and_classify("You are an animal.").keys())[0] == "animal")
        self.assertIsNone(
            stc.parse_and_classify("My name is Charles and I like sewing.")
        )
