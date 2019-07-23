import unittest
import holmes_extractor as holmes
from holmes_extractor.extensive_matching import SupervisedTopicClassifier

holmes_manager = holmes.Manager('de_core_news_md')

class GermanSupervisedTopicClassificationTest(unittest.TestCase):

    def test_get_labels_to_classification_frequencies_direct_matching(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("Ein Löwe jagt einen Tiger", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: jagen-löwe'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagen-tigern'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagen-tigern/verb-nom: jagen-löwe'], {'Tiere': 1})
        self.assertEqual(freq['word: löwe'], {'Tiere': 1})
        self.assertEqual(freq['word: tigern'], {'Tiere': 1})

    def test_linked_matching_common_dependent(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("Ein Löwe isst und frisst einen Tiger", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        # for some reason spaCy does not resolve 'isst' and 'frisst' to the infinitive forms
        self.assertEqual(freq['verb-nom: isst-löwe'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: frisst-löwe'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: frisst-tigern'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: frisst-tigern/verb-nom: frisst-löwe'],
                {'Tiere': 1})
        self.assertEqual(freq['verb-nom: frisst-löwe/verb-nom: isst-löwe'],
                {'Tiere': 1})
        self.assertEqual(freq['word: löwe'], {'Tiere': 1})
        self.assertEqual(freq['word: tigern'], {'Tiere': 1})

    def test_linked_matching_common_dependent_control(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("Ein Löwe isst und dann frisst ein Löwe", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        # for some reason spaCy does not resolve 'isst' and 'frisst' to the infinitive forms
        self.assertEqual(freq['verb-nom: isst-löwe'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: frisst-löwe'], {'Tiere': 1})
        self.assertTrue('verb-nom: frisst-löwe/verb-nom: isst-löwe' not in
                freq.keys())
        self.assertEqual(freq['word: löwe'], {'Tiere': 2})

    def test_linked_matching_stepped_lower_first(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("Ein großer Löwe isst", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: isst-löwe'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löwe-groß'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löwe-groß/verb-nom: isst-löwe'],
                {'Tiere': 1})
        self.assertEqual(freq['word: löwe'], {'Tiere': 1})

    def test_linked_matching_stepped_lower_second(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("Etwas isst einen großen Löwen", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-acc: isst-löwe'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löwe-groß'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löwe-groß/verb-acc: isst-löwe'],
                {'Tiere': 1})
        self.assertEqual(freq['word: löwe'], {'Tiere': 1})

    def test_linked_matching_stepped_control(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document(
                "Man sieht einen großen Löwen und dann isst ein Löwe", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: isst-löwe'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löwe-groß'], {'Tiere': 1})
        self.assertTrue('noun-dependent: löwe-groß/verb-nom: isst-löwe' not in
                freq.keys())
        self.assertEqual(freq['word: löwe'], {'Tiere': 2})

    def _test_whole_scenario(self, oneshot):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=oneshot)
        sttb.parse_and_register_training_document("Eine Katze jagt einen Hund. Eine Katze.", 'Tiere',
            't2')
        sttb.parse_and_register_training_document("Ein Hund jagt eine Katze", 'Tiere', 't1')
        sttb.parse_and_register_training_document("Eine Katze jagt eine Maus", 'Tiere', 't3')
        sttb.parse_and_register_training_document("Ein Programmierer benutzt eine Maus", 'IT', 'i1')
        sttb.parse_and_register_training_document("Ein Programmierer schreibt Python", 'IT', 'i2')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: jagen-hund'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: jagen-katze'], {'Tiere': 2})
        self.assertEqual(freq['verb-acc: jagen-katze/verb-nom: jagen-hund'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagen-hund/verb-nom: jagen-katze'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagen-maus/verb-nom: jagen-katze'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: benutzen-programmierer'], {'IT': 1})
        self.assertEqual(freq['verb-nom: schreiben-programmierer'], {'IT': 1})
        self.assertEqual(freq['verb-acc: benutzen-maus/verb-nom: benutzen-programmierer'],
                {'IT': 1})
        self.assertEqual(freq['verb-acc: schreiben-python/verb-nom: schreiben-programmierer'],
                {'IT': 1})
        self.assertEqual(freq['verb-acc: jagen-katze'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagen-hund'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagen-maus'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: benutzen-maus'], {'IT': 1})
        self.assertEqual(freq['verb-acc: schreiben-python'], {'IT': 1})
        self.assertEqual(freq['word: hund'], {'Tiere': 2})
        if (oneshot):
            self.assertEqual(freq['word: katze'], {'Tiere': 3})
        else:
            self.assertEqual(freq['word: katze'], {'Tiere': 4})
        self.assertEqual(freq['word: maus'], {'Tiere': 1, 'IT' : 1})
        self.assertEqual(freq['word: programmierer'], {'IT' : 2})
        self.assertEqual(freq['word: python'], {'IT' : 1})
        self.assertEqual(sttb.classifications, ['IT', 'Tiere'])
        trainer = sttb.train(minimum_occurrences=0, cv_threshold=0.0)
        holmes_manager.remove_all_search_phrases_with_label('verb-acc: benutzen-maus')
        # should not have any effect because the supervised topic objects have their own
        # StructuralMatcher instance
        self.assertEqual(list(trainer._sorted_label_dict.keys()),
                ['verb-acc: benutzen-maus',
                'verb-acc: benutzen-maus/verb-nom: benutzen-programmierer', 'verb-acc: jagen-hund',
                'verb-acc: jagen-hund/verb-nom: jagen-katze', 'verb-acc: jagen-katze',
                'verb-acc: jagen-katze/verb-nom: jagen-hund', 'verb-acc: jagen-maus',
                'verb-acc: jagen-maus/verb-nom: jagen-katze', 'verb-acc: schreiben-python',
                'verb-acc: schreiben-python/verb-nom: schreiben-programmierer',
                'verb-nom: benutzen-programmierer', 'verb-nom: jagen-hund',
                'verb-nom: jagen-katze', 'verb-nom: schreiben-programmierer', 'word: hund',
                'word: katze', 'word: maus', 'word: programmierer', 'word: python'])
        if oneshot:
            self.assertEqual([
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                    0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 1.0, 1.0, 0.0, 0.0]], trainer._input_matrix.toarray().tolist())
        else:
            self.assertEqual([
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0,
                    0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 1.0, 1.0, 0.0, 0.0]], trainer._input_matrix.toarray().tolist())

        self.assertEqual([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                trainer._output_matrix.toarray().tolist())
        self.assertEqual([19,13,7],trainer._hidden_layer_sizes)
        stc = trainer.classifier()
        self.assertEqual(stc.parse_and_classify(
                "Der Programmierer hat schon wieder Python geschrieben."),
                ['IT'])
        self.assertEqual(stc.parse_and_classify("Der Hund jagt"), ['Tiere'])
        self.assertEqual(stc.parse_and_classify("Das Parlament debattiert das Gesetz"), [])
        serialized_supervised_topic_classifier_model = stc.serialize_model()
        stc2 = holmes_manager.deserialize_supervised_topic_classifier(
                serialized_supervised_topic_classifier_model)
        self.assertEqual(list(stc2._model.sorted_label_dict.keys()),
                ['verb-acc: benutzen-maus',
                'verb-acc: benutzen-maus/verb-nom: benutzen-programmierer', 'verb-acc: jagen-hund',
                'verb-acc: jagen-hund/verb-nom: jagen-katze', 'verb-acc: jagen-katze',
                'verb-acc: jagen-katze/verb-nom: jagen-hund', 'verb-acc: jagen-maus',
                'verb-acc: jagen-maus/verb-nom: jagen-katze', 'verb-acc: schreiben-python',
                'verb-acc: schreiben-python/verb-nom: schreiben-programmierer',
                'verb-nom: benutzen-programmierer', 'verb-nom: jagen-hund',
                'verb-nom: jagen-katze', 'verb-nom: schreiben-programmierer', 'word: hund',
                'word: katze', 'word: maus', 'word: programmierer', 'word: python'])
        self.assertEqual(stc2.parse_and_classify(
                "Der Programmierer hat schon wieder Python geschrieben."),
                ['IT'])
        self.assertEqual(stc2.parse_and_classify("Der Hund jagt"), ['Tiere'])
        self.assertEqual(stc2.parse_and_classify("Das Parlament debattiert das Gesetz"), [])

    def test_whole_scenario(self):
        self._test_whole_scenario(False)

    def test_whole_scenario_oneshot(self):
        self._test_whole_scenario(True)

    def test_fed_in_hidden_layer_sizes(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(oneshot=False)
        sttb.parse_and_register_training_document("Ein Hund jagt eine Katze", 'Tiere1', 't1')
        sttb.parse_and_register_training_document("Ein Hund jagt eine Katze", 'Tiere2', 't2')
        sttb.prepare()
        trainer = sttb.train(minimum_occurrences=0, cv_threshold=0, hidden_layer_sizes=[15])
        self.assertEqual(trainer._hidden_layer_sizes, [15])

    def test_filtering(self):
        sttb = holmes_manager.get_supervised_topic_training_basis()
        sttb.parse_and_register_training_document("Eine Katze jagt einen Hund. Eine Katze.",
                'Tiere', 't2')
        sttb.parse_and_register_training_document("Ein Hund jagt eine Katze", 'Tiere', 't1')
        sttb.parse_and_register_training_document("Eine Katze jagt eine Maus", 'Tiere', 't3')
        sttb.parse_and_register_training_document("Ein Programmierer benutzt eine Maus", 'IT', 'i1')
        sttb.parse_and_register_training_document("Ein Programmierer benutzt eine Maus", 'IT', 'i3')
        sttb.parse_and_register_training_document("Ein Programmierer schreibt Python", 'IT', 'i2')
        sttb.prepare()
        trainer = sttb.train(minimum_occurrences=2, cv_threshold=0.0)
        self.assertEqual(list(trainer._sorted_label_dict.keys()),
                ['verb-acc: benutzen-maus',
                'verb-acc: benutzen-maus/verb-nom: benutzen-programmierer',
                'verb-nom: benutzen-programmierer', 'verb-nom: jagen-katze',
                'word: hund', 'word: katze', 'word: maus', 'word: programmierer'])
        self.assertEqual(set(map(lambda phr: phr.label, trainer._serialized_phraselets)),
                {'verb-acc: benutzen-maus',
                'verb-nom: benutzen-programmierer', 'verb-nom: jagen-katze',
                'word: hund', 'word: katze', 'word: maus', 'word: programmierer'})
        trainer2 = sttb.train(minimum_occurrences=2, cv_threshold=1)
        self.assertEqual(list(trainer2._sorted_label_dict.keys()),
                ['verb-acc: benutzen-maus',
                'verb-acc: benutzen-maus/verb-nom: benutzen-programmierer',
                'verb-nom: benutzen-programmierer', 'verb-nom: jagen-katze',
                'word: hund', 'word: katze', 'word: programmierer'])
        self.assertEqual(set(map(lambda phr: phr.label, trainer2._serialized_phraselets)),
                {'verb-acc: benutzen-maus',
                'verb-nom: benutzen-programmierer', 'verb-nom: jagen-katze',
                'word: hund', 'word: katze', 'word: programmierer'})
