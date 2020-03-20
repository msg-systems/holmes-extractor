import unittest
import holmes_extractor as holmes
from holmes_extractor.extensive_matching import SupervisedTopicClassifier
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
ontology = holmes.Ontology(os.sep.join((script_directory,'test_ontology.owl')))
holmes_manager = holmes.Manager('en_core_web_lg',
        perform_coreference_resolution=True, ontology=ontology)
no_ontology_holmes_manager = holmes.Manager('en_core_web_lg',
        perform_coreference_resolution=True)
no_coref_holmes_manager = holmes.Manager('en_core_web_lg',
        perform_coreference_resolution=False, ontology=ontology)

class EnglishSupervisedTopicClassificationTest(unittest.TestCase):

    def test_get_labels_to_classification_frequencies_direct_matching(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("A lion chases a tiger", 'animals')
        sttb.parse_and_register_training_document("fast", 'dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['predicate-actor: chase-lion'], {'animals': 1})
        self.assertEqual(freq['predicate-patient: chase-tiger'], {'animals': 1})
        self.assertEqual(freq['predicate-actor: chase-lion/predicate-patient: chase-tiger'],
                {'animals': 1})
        self.assertEqual(freq['word: lion'], {'animals': 1})
        self.assertEqual(freq['word: tiger'], {'animals': 1})

    def test_get_labels_to_classification_frequencies_ontology_matching(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("A dog chases a cat", 'animals')
        sttb.parse_and_register_training_document("fast", 'dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['predicate-actor: chase-animal'], {'animals': 1})
        self.assertEqual(freq['predicate-patient: chase-animal'], {'animals': 1})
        self.assertEqual(freq['predicate-actor: chase-animal/predicate-patient: chase-animal'],
                {'animals': 1})
        self.assertEqual(freq['word: animal'], {'animals': 2})

    def test_get_labels_to_classification_frequencies_ontology_multiword_matching(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("A gymnast jumps over a vaulting horse", 'gym')
        sttb.parse_and_register_training_document("fast", 'dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['predicate-actor: jump-gymnast'], {'gym': 1})
        self.assertEqual(freq['word: gymnast'], {'gym': 1})
        self.assertEqual(freq['word: gymnastics equipment'], {'gym': 1})

    def test_linked_matching_common_dependent(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("A lion eats and consumes a tiger", 'animals')
        sttb.parse_and_register_training_document("fast", 'dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['predicate-actor: consume-lion'], {'animals': 1})
        self.assertEqual(freq['predicate-actor: eat-lion'], {'animals': 1})
        self.assertEqual(freq['predicate-patient: consume-tiger'], {'animals': 1})
        self.assertEqual(freq['predicate-actor: consume-lion/predicate-patient: consume-tiger'],
                {'animals': 1})
        self.assertEqual(freq['predicate-actor: consume-lion/predicate-actor: eat-lion'],
                {'animals': 1})
        self.assertEqual(freq['word: lion'], {'animals': 1})
        self.assertEqual(freq['word: tiger'], {'animals': 1})

    def test_linked_matching_common_dependent_control(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("A lion eats and a lion consumes", 'animals')
        sttb.parse_and_register_training_document("fast", 'dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['predicate-actor: consume-lion'], {'animals': 1})
        self.assertEqual(freq['predicate-actor: eat-lion'], {'animals': 1})
        self.assertTrue('predicate-actor: consume-lion/predicate-actor: eat-lion' not in
                freq.keys())
        self.assertEqual(freq['word: lion'], {'animals': 2})

    def test_linked_matching_stepped_lower_first(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("A big lion eats", 'animals')
        sttb.parse_and_register_training_document("fast", 'dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['governor-adjective: lion-big'], {'animals': 1})
        self.assertEqual(freq['predicate-actor: eat-lion'], {'animals': 1})
        self.assertEqual(freq['governor-adjective: lion-big/predicate-actor: eat-lion'],
                {'animals': 1})
        self.assertEqual(freq['word: lion'], {'animals': 1})

    def test_linked_matching_stepped_lower_second(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("Something eats a big lion", 'animals')
        sttb.parse_and_register_training_document("fast", 'dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['governor-adjective: lion-big'], {'animals': 1})
        self.assertEqual(freq['predicate-patient: eat-lion'], {'animals': 1})
        self.assertEqual(freq['governor-adjective: lion-big/predicate-patient: eat-lion'],
                {'animals': 1})
        self.assertEqual(freq['word: lion'], {'animals': 1})

    def test_linked_matching_stepped_control(self):
        sttb = no_coref_holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document(
                "There is a big lion and the lion eats", 'animals')
        sttb.parse_and_register_training_document("fast", 'dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['governor-adjective: lion-big'], {'animals': 1})
        self.assertEqual(freq['predicate-actor: eat-lion'], {'animals': 1})
        self.assertTrue('governor-adjective: lion-big/predicate-actor: eat-lion' not in freq.keys())
        self.assertEqual(freq['word: lion'], {'animals': 2})

    def test_repeating_relation_through_coreference(self):
        sttb = no_ontology_holmes_manager.get_supervised_topic_training_basis()
        sttb.parse_and_register_training_document("The building was used last year. It is used this year", 'test')
        sttb.parse_and_register_training_document("fast", 'dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertFalse(
                'predicate-patient: use-building/predicate-patient: use-building' in freq)

    def test_oneshot(self):
        sttb1 = no_coref_holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb1.parse_and_register_training_document("A dog chases a cat. A dog chases a cat",
                'animals')
        sttb1.parse_and_register_training_document("A dog chases a cat. A dog chases a cat",
                'animals')
        sttb1.parse_and_register_training_document("A dog chases a cat. A dog chases a cat",
                'animals2')
        sttb1.parse_and_register_training_document("A dog chases a cat. A dog chases a cat",
                'animals2')
        sttb1.prepare()
        freq1 = sttb1.labels_to_classification_frequencies
        sttb2 = no_coref_holmes_manager.get_supervised_topic_training_basis(oneshot=True)
        sttb2.parse_and_register_training_document("A dog chases a cat. A dog chases a cat",
                'animals')
        sttb2.parse_and_register_training_document("A dog chases a cat. A dog chases a cat",
                'animals')
        sttb2.parse_and_register_training_document("A dog chases a cat. A dog chases a cat",
                'animals2')
        sttb2.parse_and_register_training_document("A dog chases a cat. A dog chases a cat",
                'animals2')
        sttb2.prepare()
        freq2 = sttb2.labels_to_classification_frequencies
        self.assertEqual(freq1['predicate-actor: chase-animal/predicate-patient: chase-animal'],
                {'animals': 4, 'animals2': 4})
        self.assertEqual(freq1['predicate-actor: chase-animal'],
                {'animals': 4, 'animals2': 4})
        self.assertEqual(freq1['predicate-patient: chase-animal'],
                {'animals': 4, 'animals2': 4})
        self.assertEqual(freq1['word: animal'],
                {'animals': 8, 'animals2': 8})
        self.assertEqual(freq2['predicate-actor: chase-animal/predicate-patient: chase-animal'],
                {'animals': 2, 'animals2': 2})
        self.assertEqual(freq2['predicate-actor: chase-animal'],
                {'animals': 2, 'animals2': 2})
        self.assertEqual(freq2['predicate-patient: chase-animal'],
                {'animals': 2, 'animals2': 2})
        self.assertEqual(freq2['word: animal'],
                {'animals': 2, 'animals2': 2})

    def test_multiple_document_classes(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("A dog chases a cat", 'animals')
        sttb.parse_and_register_training_document("A cat chases a dog", 'animals')
        sttb.parse_and_register_training_document("A cat chases a horse", 'animals')
        sttb.parse_and_register_training_document("A cat chases a horse", 'animals')
        sttb.parse_and_register_training_document("A gymnast jumps over a horse", 'gym')
        sttb.parse_and_register_training_document("A gymnast jumps over a vaulting horse", 'gym')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['predicate-actor: chase-animal'], {'animals': 4})
        self.assertEqual(freq['predicate-actor: jump-gymnast'], {'gym': 2})
        self.assertEqual(freq['predicate-patient: chase-animal'], {'animals': 4})
        self.assertEqual(freq['predicate-actor: chase-animal/predicate-patient: chase-animal'],
                {'animals': 4})
        self.assertEqual(freq['word: animal'], {'animals': 8, 'gym': 1})
        self.assertEqual(freq['word: gymnast'], {'gym': 2})
        self.assertEqual(freq['word: gymnastics equipment'], {'animals':2, 'gym': 2})

    def test_whole_scenario_with_classification_ontology(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(classification_ontology=ontology,
                oneshot=False)
        sttb.parse_and_register_training_document("A puppy", 'puppy', 'd0')
        sttb.parse_and_register_training_document("A pussy", 'cat', 'd1')
        sttb.parse_and_register_training_document("A dog on a lead", 'dog', 'd2')
        sttb.parse_and_register_training_document("Mimi Momo", 'Mimi Momo', 'd3')
        sttb.parse_and_register_training_document("An animal", 'animal', 'd4')
        sttb.parse_and_register_training_document("A computer", 'computers', 'd5')
        sttb.parse_and_register_training_document("A robot", 'computers', 'd6')
        sttb.register_additional_classification_label('parrot')
        sttb.register_additional_classification_label('hound')
        sttb.prepare()
        self.assertEqual({'Mimi Momo': ['animal', 'cat'], 'dog': ['animal', 'hound'],
                'puppy': ['animal', 'dog', 'hound'], 'cat': ['animal'], 'hound':
                ['animal', 'dog']},
                sttb.classification_implication_dict)
        self.assertEqual(['Mimi Momo', 'animal', 'cat', 'computers', 'dog', 'hound', 'puppy'],
                sttb.classifications)
        trainer = sttb.train(minimum_occurrences=0, cv_threshold=0, mlp_max_iter=10000)
        self.assertEqual(['prepgovernor-noun: animal-lead', 'word: animal', 'word: computer',
                'word: lead', 'word: robot'],
                list(trainer._sorted_label_dict.keys()))
        self.assertEqual([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]], trainer._input_matrix.toarray().tolist())
        self.assertEqual([[0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0]], trainer._output_matrix.toarray().tolist())
        self.assertEqual((5,5,6), trainer._hidden_layer_sizes)
        stc = trainer.classifier()
        self.assertEqual(stc.parse_and_classify("You are a robot."), ['computers'])
        self.assertEqual(stc.parse_and_classify("You are a cat."), ['animal'])
        self.assertEqual(stc.parse_and_classify("My name is Charles and I like sewing."), [])
        self.assertEqual(stc.parse_and_classify("Your dog appears to be on a lead."),
                ['animal', 'dog', 'hound'])
        serialized_supervised_topic_classifier_model = stc.serialize_model()
        stc2 = no_ontology_holmes_manager.deserialize_supervised_topic_classifier(
                serialized_supervised_topic_classifier_model, verbose=True)
        self.assertEqual(['prepgovernor-noun: animal-lead', 'word: animal', 'word: computer',
                'word: lead', 'word: robot'], list(stc2._model.sorted_label_dict.keys()))
        self.assertEqual(stc2.parse_and_classify("You are a robot."), ['computers'])
        self.assertEqual(stc2.parse_and_classify("You are a cat."), ['animal'])
        self.assertEqual(stc2.parse_and_classify("My name is Charles and I like sewing."), [])
        self.assertEqual(stc2.parse_and_classify("Your dog appears to be on a lead."),
                ['animal', 'dog', 'hound'])

    def test_whole_scenario_with_classification_ontology_and_match_all_words(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(classification_ontology=ontology,
                match_all_words=True, oneshot=False)
        sttb.parse_and_register_training_document("A puppy", 'puppy', 'd0')
        sttb.parse_and_register_training_document("A pussy", 'cat', 'd1')
        sttb.parse_and_register_training_document("A dog on a lead", 'dog', 'd2')
        sttb.parse_and_register_training_document("Mimi Momo", 'Mimi Momo', 'd3')
        sttb.parse_and_register_training_document("An animal", 'animal', 'd4')
        sttb.parse_and_register_training_document("A computer", 'computers', 'd5')
        sttb.parse_and_register_training_document("A robot", 'computers', 'd6')
        sttb.register_additional_classification_label('parrot')
        sttb.register_additional_classification_label('hound')
        sttb.prepare()
        self.assertEqual({'Mimi Momo': ['animal', 'cat'], 'dog': ['animal', 'hound'],
                'puppy': ['animal', 'dog', 'hound'], 'cat': ['animal'], 'hound':
                ['animal', 'dog']},
                sttb.classification_implication_dict)
        self.assertEqual(['Mimi Momo', 'animal', 'cat', 'computers', 'dog', 'hound', 'puppy'],
                sttb.classifications)
        trainer = sttb.train(minimum_occurrences=0, cv_threshold=0, mlp_max_iter=10000)
        self.assertEqual(['prepgovernor-noun: animal-lead', 'word: animal', 'word: computer',
                'word: lead', 'word: on', 'word: robot'],
                list(trainer._sorted_label_dict.keys()))
        self.assertEqual([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], trainer._input_matrix.toarray().tolist())
        self.assertEqual([[0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0]], trainer._output_matrix.toarray().tolist())
        self.assertEqual((6,6,6), trainer._hidden_layer_sizes)
        stc = trainer.classifier()
        self.assertEqual(stc.parse_and_classify("You are a robot."), ['computers'])
        self.assertEqual(stc.parse_and_classify("You are a cat."), ['animal'])
        self.assertEqual(stc.parse_and_classify("My name is Charles and I like sewing."), [])
        self.assertEqual(stc.parse_and_classify("Your dog appears to be on a lead."),
                ['animal', 'hound', 'dog'])
        serialized_supervised_topic_classifier_model = stc.serialize_model()
        stc2 = no_ontology_holmes_manager.deserialize_supervised_topic_classifier(
                serialized_supervised_topic_classifier_model)
        self.assertEqual(['prepgovernor-noun: animal-lead', 'word: animal', 'word: computer',
                'word: lead', 'word: on', 'word: robot'],
                list(stc2._model.sorted_label_dict.keys()))
        self.assertEqual(stc2.parse_and_classify("You are a robot."), ['computers'])
        self.assertEqual(stc2.parse_and_classify("You are a cat."), ['animal'])
        self.assertEqual(stc2.parse_and_classify("My name is Charles and I like sewing."), [])
        self.assertEqual(stc2.parse_and_classify("Your dog appears to be on a lead."),
                ['animal', 'hound', 'dog'])

    def test_filtering(self):
        sttb = holmes_manager.get_supervised_topic_training_basis()
        sttb.parse_and_register_training_document("A dog chases a cat", 'animals')
        sttb.parse_and_register_training_document("A cat chases a dog", 'animals')
        sttb.parse_and_register_training_document("A cat chases a horse", 'animals')
        sttb.parse_and_register_training_document("A cat chases a horse", 'animals')
        sttb.parse_and_register_training_document("A gymnast jumps over a horse", 'gym')
        sttb.parse_and_register_training_document("A gymnast jumps over a vaulting horse",
                'gym')
        sttb.prepare()
        trainer = sttb.train(minimum_occurrences=4, cv_threshold=0.0)
        self.assertEqual(list(trainer._sorted_label_dict.keys()),
                ['predicate-actor: chase-animal',
                'predicate-actor: chase-animal/predicate-patient: chase-animal',
                'predicate-patient: chase-animal', 'word: animal'])
        self.assertEqual(set(map(lambda phr: phr.label, trainer._phraselet_infos)),
                {'predicate-actor: chase-animal',
                'predicate-patient: chase-animal', 'word: animal'})
        trainer2 = sttb.train(minimum_occurrences=4, cv_threshold=1)
        self.assertEqual(list(trainer2._sorted_label_dict.keys()),
                ['predicate-actor: chase-animal',
                'predicate-actor: chase-animal/predicate-patient: chase-animal',
                'predicate-patient: chase-animal'])
        self.assertEqual(set(map(lambda phr: phr.label, trainer2._phraselet_infos)),
                {'predicate-actor: chase-animal',
                'predicate-patient: chase-animal'})
