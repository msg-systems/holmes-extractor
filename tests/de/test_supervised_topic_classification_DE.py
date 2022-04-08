import unittest
from collections import OrderedDict
import holmes_extractor as holmes
from holmes_extractor.classification import SupervisedTopicClassifier

holmes_manager = holmes.Manager('de_core_news_lg', number_of_workers=1)

def get_first_key_in_dict(dictionary: OrderedDict) -> str:
    return list(dictionary.keys())[0]

class GermanSupervisedTopicClassificationTest(unittest.TestCase):

    def test_get_labels_to_classification_frequencies_direct_matching(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Löwe jagt einen Tiger", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: jagd-löw'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagd-tiger'], {'Tiere': 1})
        self.assertEqual(
            freq['verb-acc: jagd-tiger/verb-nom: jagd-löw'], {'Tiere': 1})
        self.assertEqual(freq['word: löw'], {'Tiere': 1})
        self.assertEqual(freq['word: tiger'], {'Tiere': 1})

    def test_linked_matching_common_dependent(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Löwe isst und frisst einen Tiger", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        # for some reason spaCy does not resolve 'isst' and 'frisst' to the infinitive forms
        self.assertEqual(freq['verb-nom: isst-löw'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: frisst-löw'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: frisst-tiger'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: frisst-tiger/verb-nom: frisst-löw'],
                         {'Tiere': 1})
        self.assertEqual(freq['verb-nom: frisst-löw/verb-nom: isst-löw'],
                         {'Tiere': 1})
        self.assertEqual(freq['word: löw'], {'Tiere': 1})
        self.assertEqual(freq['word: tiger'], {'Tiere': 1})

    def test_linked_matching_common_dependent_control(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Löwe isst und dann frisst ein Löwe", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        # for some reason spaCy does not resolve 'isst' and 'frisst' to the infinitive forms
        self.assertEqual(freq['verb-nom: isst-löw'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: frisst-löw'], {'Tiere': 1})
        self.assertTrue('verb-nom: frisst-löw/verb-nom: isst-löw' not in
                        freq.keys())
        self.assertEqual(freq['word: löw'], {'Tiere': 2})

    def test_linked_matching_stepped_lower_first(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein großer Löwe isst", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: isst-löw'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löw-groß'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löw-groß/verb-nom: isst-löw'],
                         {'Tiere': 1})
        self.assertEqual(freq['word: löw'], {'Tiere': 1})

    def test_linked_matching_stepped_lower_second(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Etwas isst einen großen Löwen", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-acc: isst-löw'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löw-groß'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löw-groß/verb-acc: isst-löw'],
                         {'Tiere': 1})
        self.assertEqual(freq['word: löw'], {'Tiere': 1})

    def test_linked_matching_stepped_control(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Man sieht einen großen Löwen und dann isst ein Löwe", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: isst-löw'], {'Tiere': 1})
        self.assertEqual(freq['noun-dependent: löw-groß'], {'Tiere': 1})
        self.assertTrue('noun-dependent: löw-groß/verb-nom: isst-löw' not in
                        freq.keys())
        self.assertEqual(freq['word: löw'], {'Tiere': 2})

    def test_get_labels_to_classification_frequencies_direct_matching_with_subwords(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Informationslöwe jagt einen Informationstiger", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: jagd-löw'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagd-tiger'], {'Tiere': 1})
        self.assertEqual(
            freq['verb-acc: jagd-tiger/verb-nom: jagd-löw'], {'Tiere': 1})
        self.assertEqual(freq['word: informationslöwe'], {'Tiere': 1})
        self.assertEqual(freq['word: informationstiger'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: löw-information'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: tiger-information'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-information/verb-nom: jagd-löw'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: tiger-information/verb-acc: jagd-tiger'], {'Tiere': 1})

    def test_get_labels_to_classification_frequencies_direct_matching_with_subwords_and_conjunction_of_verb(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Informationslöwe jagt und trägt einen Informationstiger", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: jagd-löw'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: tragen-löw'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: tragen-tiger'], {'Tiere': 1})
        self.assertEqual(
            freq['verb-acc: tragen-tiger/verb-nom: tragen-löw'], {'Tiere': 1})
        self.assertEqual(freq['word: informationslöw'], {'Tiere': 1})
        self.assertEqual(freq['word: informationstiger'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: löw-information'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: tiger-information'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-information/verb-nom: jagd-löw'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-information/verb-nom: tragen-löw'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: tiger-information/verb-acc: tragen-tiger'], {'Tiere': 1})

    def test_get_labels_to_classification_frequencies_with_front_subword_conjunction(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Informationsextraktionsmaßnahmen- und Raketenlöwe fressen", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(
            freq['intcompound: extraktion-information'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: maßnahm-extraktion'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: löw-maßnahm'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: löw-raket'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: fressen-löw'], {'Tiere': 1})
        self.assertEqual(freq['word: raketenlöwe'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: extraktion-information/intcompound: maßnahm-extraktion'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-maßnahm/intcompound: maßnahm-extraktion'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-raket/verb-nom: fressen-löw'], {'Tiere': 1})
        # 'intcompound: löw-maßnahm/verb-nom: fressen-löw' should logically be added as well,
        # but would require considerable changes for very little additional functionality.

    def test_get_labels_to_classification_frequencies_with_back_subword_conjunction(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Informationsextraktionsmaßnahmen und -raketenlöwe fressen", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        print(freq)
        self.assertEqual(
            freq['intcompound: extraktion-information'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: maßnahm-extraktion'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: raket-extraktion'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: löw-raket'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-raket/intcompound: raket-extraktion'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: fressen-löw'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: fressen-maßnahm'], {'Tiere': 1})
        self.assertEqual(
            freq['word: informationsextraktionsmaßnahmen'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: extraktion-information/intcompound: maßnahm-extraktion'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-raket/verb-nom: fressen-löw'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: maßnahm-extraktion/verb-nom: fressen-maßnahm'], {'Tiere': 1})
        self.assertEqual(
            freq['verb-nom: fressen-löw/verb-nom: fressen-maßnahm'], {'Tiere': 1})

    def test_get_labels_to_classification_frequencies_with_front_and_back_subword_conjunction(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Informationsextraktionsmaßnahmen- und -raketenlöwe fressen", 'Tiere')
        sttb.parse_and_register_training_document("schnell", 'Dummy')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(
            freq['intcompound: extraktion-information'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: maßnahm-extraktion'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: raket-extraktion'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: löw-maßnahm'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: löw-raket'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-raket/intcompound: raket-extraktion'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-maßnahm/intcompound: maßnahm-extraktion'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: fressen-löw'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: extraktion-information/intcompound: maßnahm-extraktion'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: löw-raket/verb-nom: fressen-löw'], {'Tiere': 1})

    def _test_whole_scenario(self, one_hot):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=one_hot)
        sttb.parse_and_register_training_document("Eine Katze jagt einen Hund. Eine Katze.", 'Tiere',
                                                  't2')
        sttb.parse_and_register_training_document(
            "Ein Plüschhund jagt eine Katze", 'Tiere', 't1')
        sttb.parse_and_register_training_document(
            "Eine Katze jagt eine Maus", 'Tiere', 't3')
        sttb.parse_and_register_training_document(
            "Ein Programmierer benutzt eine Maus", 'IT', 'i1')
        sttb.parse_and_register_training_document(
            "Ein Programmierer schreibt Python", 'IT', 'i2')
        sttb.prepare()
        freq = sttb.labels_to_classification_frequencies
        self.assertEqual(freq['verb-nom: jagd-hund'], {'Tiere': 1})
        self.assertEqual(freq['intcompound: hund-plüsch'], {'Tiere': 1})
        self.assertEqual(
            freq['intcompound: hund-plüsch/verb-nom: jagd-hund'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: jagd-katz'], {'Tiere': 2})
        self.assertEqual(
            freq['verb-acc: jagd-katz/verb-nom: jagd-hund'], {'Tiere': 1})
        self.assertEqual(
            freq['verb-acc: jagd-hund/verb-nom: jagd-katz'], {'Tiere': 1})
        self.assertEqual(
            freq['verb-acc: jagd-maus/verb-nom: jagd-katz'], {'Tiere': 1})
        self.assertEqual(freq['verb-nom: benutzen-programmierer'], {'IT': 1})
        self.assertEqual(freq['verb-nom: schrift-programmierer'], {'IT': 1})
        self.assertEqual(freq['verb-acc: benutzen-maus/verb-nom: benutzen-programmierer'],
                         {'IT': 1})
        self.assertEqual(freq['verb-acc: schrift-python/verb-nom: schrift-programmierer'],
                         {'IT': 1})
        self.assertEqual(freq['verb-acc: jagd-katz'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagd-hund'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: jagd-maus'], {'Tiere': 1})
        self.assertEqual(freq['verb-acc: benutzen-maus'], {'IT': 1})
        self.assertEqual(freq['verb-acc: schrift-python'], {'IT': 1})
        self.assertEqual(freq['word: hund'], {'Tiere': 2})
        if (one_hot):
            self.assertEqual(freq['word: katz'], {'Tiere': 3})
        else:
            self.assertEqual(freq['word: katz'], {'Tiere': 4})
        self.assertEqual(freq['word: maus'], {'Tiere': 1, 'IT': 1})
        self.assertEqual(freq['word: programmierer'], {'IT': 2})
        self.assertEqual(freq['word: python'], {'IT': 1})
        self.assertEqual(sttb.classifications, ['IT', 'Tiere'])
        trainer = sttb.train(minimum_occurrences=0, cv_threshold=0.0)
        holmes_manager.remove_all_search_phrases_with_label(
            'verb-acc: benutzen-maus')
        # should not have any effect because the supervised topic objects have their own
        # StructuralMatcher instance
                # With so little training data, the NN does not consistently learn correctly
        for i in range(10):
            trainer = sttb.train(
                minimum_occurrences=0,
                cv_threshold=0,
                max_epochs=1000,
                learning_rate=0.0001,
                convergence_threshold=0,
            )
            stc = trainer.classifier()
            if (
                get_first_key_in_dict(stc.parse_and_classify("Der Programmierer hat schon wieder Python geschrieben."))
                == "IT"
                and get_first_key_in_dict(stc.parse_and_classify("Der Plüschhund debattiert"))
                == "Tiere"
            ):
                break
            if i == 10:
                self.assertTrue(
                    get_first_key_in_dict(stc.parse_and_classify("Der Programmierer hat schon wieder Python geschrieben."))
                    == "IT"
                    and get_first_key_in_dict(
                        stc.parse_and_classify("Der Plüschhund debattiert")
                    )
                    == "Tiere"
                )

        self.assertEqual(list(trainer.sorted_label_dict.keys()),
                         ['intcompound: hund-plüsch', 'intcompound: hund-plüsch/verb-nom: jagd-hund',
                          'verb-acc: benutzen-maus',
                          'verb-acc: benutzen-maus/verb-nom: benutzen-programmierer', 'verb-acc: jagd-hund',
                          'verb-acc: jagd-hund/verb-nom: jagd-katz', 'verb-acc: jagd-katz',
                          'verb-acc: jagd-katz/verb-nom: jagd-hund', 'verb-acc: jagd-maus',
                          'verb-acc: jagd-maus/verb-nom: jagd-katz', 'verb-acc: schrift-python',
                          'verb-acc: schrift-python/verb-nom: schrift-programmierer',
                          'verb-nom: benutzen-programmierer', 'verb-nom: jagd-hund',
                          'verb-nom: jagd-katz', 'verb-nom: schrift-programmierer', 'word: hund',
                          'word: katz', 'word: maus', 'word: plüschhund', 'word: programmierer',
                          'word: python'])
        if one_hot:
            self.assertEqual(
                [{20: 1, 12: 1, 2: 1, 3: 1, 18: 1}, {20: 1, 15: 1, 10: 1, 11: 1, 21: 1}, {19: 1, 16: 1, 0: 1, 13: 1, 1: 1, 6: 1, 7: 1, 17: 1}, {17: 1, 14: 1, 4: 1, 5: 1, 16: 1}, {17: 1, 14: 1, 8: 1, 9: 1, 18: 1}],
                trainer.occurrence_dicts)
        else:
            print(trainer.occurrence_dicts)
            self.assertEqual(
                [{20: 1, 12: 1, 2: 1, 3: 1, 18: 1}, {20: 1, 15: 1, 10: 1, 11: 1, 21: 1}, {19: 1, 16: 1, 0: 1, 13: 1, 1: 1, 6: 1, 7: 1, 17: 1}, {17: 2, 14: 1, 4: 1, 5: 1, 16: 1}, {17: 1, 14: 1, 8: 1, 9: 1, 18: 1}],
                trainer.occurrence_dicts)

        self.assertEqual([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                         trainer.output_matrix.tolist())
        self.assertEqual([22, 15, 8], trainer._hidden_layer_sizes)
        self.assertIsNone(stc.parse_and_classify(
            "Das Parlament debattiert das Gesetz"))
        serialized_supervised_topic_classifier_model = stc.serialize_model()
        stc2 = holmes_manager.deserialize_supervised_topic_classifier(
            serialized_supervised_topic_classifier_model)
        self.assertEqual(list(stc2.model.sorted_label_dict.keys()),
                         ['intcompound: hund-plüsch', 'intcompound: hund-plüsch/verb-nom: jagd-hund',
                          'verb-acc: benutzen-maus',
                          'verb-acc: benutzen-maus/verb-nom: benutzen-programmierer', 'verb-acc: jagd-hund',
                          'verb-acc: jagd-hund/verb-nom: jagd-katz', 'verb-acc: jagd-katz',
                          'verb-acc: jagd-katz/verb-nom: jagd-hund', 'verb-acc: jagd-maus',
                          'verb-acc: jagd-maus/verb-nom: jagd-katz', 'verb-acc: schrift-python',
                          'verb-acc: schrift-python/verb-nom: schrift-programmierer',
                          'verb-nom: benutzen-programmierer', 'verb-nom: jagd-hund',
                          'verb-nom: jagd-katz', 'verb-nom: schrift-programmierer', 'word: hund',
                          'word: katz', 'word: maus', 'word: plüschhund', 'word: programmierer',
                          'word: python'])
        self.assertEqual(get_first_key_in_dict(stc2.parse_and_classify(
            "Der Programmierer hat schon wieder Python geschrieben.")),
            'IT')
        self.assertEqual(get_first_key_in_dict(stc2.parse_and_classify(
            "Der Plüschhund debattiert")), 'Tiere')
        self.assertIsNone(stc2.parse_and_classify(
            "Das Parlament debattiert das Gesetz"))

    def test_whole_scenario(self):
        self._test_whole_scenario(False)

    def test_whole_scenario_one_hot(self):
        self._test_whole_scenario(True)

    def test_fed_in_hidden_layer_sizes(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Hund jagt eine Katze", 'Tiere1', 't1')
        sttb.parse_and_register_training_document(
            "Ein Hund jagt eine Katze", 'Tiere2', 't2')
        sttb.prepare()
        trainer = sttb.train(minimum_occurrences=0,
                             cv_threshold=0, hidden_layer_sizes=[15])
        self.assertEqual(trainer._hidden_layer_sizes, [15])

    def test_empty_fed_in_hidden_layer_sizes(self):
        sttb = holmes_manager.get_supervised_topic_training_basis(
            one_hot=False)
        sttb.parse_and_register_training_document(
            "Ein Hund jagt eine Katze", 'Tiere1', 't1')
        sttb.parse_and_register_training_document(
            "Ein Hund jagt eine Katze", 'Tiere2', 't2')
        sttb.prepare()
        trainer = sttb.train(minimum_occurrences=0,
                             cv_threshold=0, hidden_layer_sizes=[])
        print(trainer._hidden_layer_sizes)
        self.assertEqual(trainer._hidden_layer_sizes, [5, 4, 3])

    def test_filtering(self):
        sttb = holmes_manager.get_supervised_topic_training_basis()
        sttb.parse_and_register_training_document("Eine Katze jagt einen Hund. Eine Katze.",
                                                  'Tiere', 't2')
        sttb.parse_and_register_training_document(
            "Ein Hund jagt eine Katze", 'Tiere', 't1')
        sttb.parse_and_register_training_document(
            "Eine Katze jagt eine Maus", 'Tiere', 't3')
        sttb.parse_and_register_training_document(
            "Ein Programmierer benutzt eine Maus", 'IT', 'i1')
        sttb.parse_and_register_training_document(
            "Ein Programmierer benutzt eine Maus", 'IT', 'i3')
        sttb.parse_and_register_training_document(
            "Ein Programmierer schreibt Python", 'IT', 'i2')
        sttb.prepare()
        trainer = sttb.train(minimum_occurrences=2, cv_threshold=0.0)
        self.assertEqual(list(trainer.sorted_label_dict.keys()),
                         ['verb-acc: benutzen-maus',
                          'verb-acc: benutzen-maus/verb-nom: benutzen-programmierer',
                          'verb-nom: benutzen-programmierer', 'verb-nom: jagd-katz',
                          'word: hund', 'word: katz', 'word: maus', 'word: programmierer'])
        self.assertEqual(set(map(lambda phr: phr.label, trainer.phraselet_infos)),
                         {'verb-acc: benutzen-maus',
                          'verb-nom: benutzen-programmierer', 'verb-nom: jagd-katz',
                          'word: hund', 'word: katz', 'word: maus', 'word: programmierer'})
        trainer2 = sttb.train(minimum_occurrences=2, cv_threshold=1)
        self.assertEqual(list(trainer2.sorted_label_dict.keys()),
                         ['verb-acc: benutzen-maus',
                          'verb-acc: benutzen-maus/verb-nom: benutzen-programmierer',
                          'verb-nom: benutzen-programmierer', 'verb-nom: jagd-katz',
                          'word: hund', 'word: katz', 'word: programmierer'])
        self.assertEqual(set(map(lambda phr: phr.label, trainer2.phraselet_infos)),
                         {'verb-acc: benutzen-maus',
                          'verb-nom: benutzen-programmierer', 'verb-nom: jagd-katz',
                          'word: hund', 'word: katz', 'word: programmierer'})
