import unittest
import spacy
import coreferee
import holmes_extractor
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')
nlp.add_pipe('holmes')

class EnglishSemanticAnalyzerTest(unittest.TestCase):

    def test_initialize_semantic_dependencies(self):
        doc = nlp("The dog chased the cat.")
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '1:nsubj; 4:dobj')
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '')
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '')

    def test_one_righthand_sibling_with_and_conjunction(self):
        doc = nlp("The dog and the hound chased the cat")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4])
        self.assertFalse(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertFalse(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])

    def test_many_righthand_siblings_with_and_conjunction(self):
        doc = nlp("The dog, the wolf and the hound chased the cat")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4, 7])
        self.assertFalse(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertFalse(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertFalse(doc[7]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])
        self.assertEqual(doc[7]._.holmes.righthand_siblings, [])

    def test_one_righthand_sibling_with_or_conjunction(self):
        doc = nlp("The dog or the hound chased the cat")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4])
        self.assertTrue(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertTrue(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])

    def test_many_righthand_siblings_with_or_conjunction(self):
        doc = nlp("The dog, the wolf or the hound chased the cat")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4, 7])
        self.assertTrue(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertTrue(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertTrue(doc[7]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])
        self.assertEqual(doc[7]._.holmes.righthand_siblings, [])

    def test_righthand_siblings_of_semantic_children_two(self):
        doc = nlp("The large and strong dog came home")
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '1:amod; 3:amod')
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [3])

    def test_righthand_siblings_of_semantic_children_many(self):
        doc = nlp("The large or strong and fierce dog came home")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:amod; 3:amod; 5:amod')
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [3,5])
        self.assertEqual(doc[3]._.holmes.righthand_siblings, [])

    def test_semantic_children_of_righthand_siblings_two(self):
        doc = nlp("The large dog and cat")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                         '1:amod; 3:cc; 4:conj')
        self.assertEqual(doc[2]._.holmes.righthand_siblings, [4])
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '1:amod(U)')

    def test_semantic_children_of_righthand_siblings_many(self):
        doc = nlp("The large dog, cat and mouse")
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '1:amod; 4:conj')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '1:amod(U); 5:cc; 6:conj')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:amod(U)')

    def test_predicative_adjective(self):
        doc = nlp("The dog was big")
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '3:amod')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_predicative_adjective_with_conjunction(self):
        doc = nlp("The dog and the cat were big and strong")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                         '2:cc; 4:conj; 6:amod; 8:amod')
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '6:amod; 8:amod')

    def test_predicative_adjective_with_non_coreferring_pronoun(self):
        doc = nlp("It was big")
        self.assertEqual(
            doc[0]._.holmes.string_representation_of_children(), '2:amod')
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '-1:None')

    def test_predicative_adjective_with_coreferring_pronoun(self):
        doc = nlp("I saw a dog. It was big")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '7:amod')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '-6:None')

    def test_negator_negation_within_clause(self):
        doc = nlp("The dog did not chase the cat")
        self.assertEqual(doc[4]._.holmes.is_negated, True)

    def test_operator_negation_within_clause(self):
        doc = nlp("No dog chased any cat")
        self.assertEqual(doc[1]._.holmes.is_negated, True)
        self.assertEqual(doc[2]._.holmes.is_negated, False)

    def test_negator_negation_within_parent_clause(self):
        doc = nlp(
            "It had not been claimed that the dog had chased the cat")
        self.assertEqual(doc[4]._.holmes.is_negated, True)

    def test_operator_negation_within_parent_clause(self):
        doc = nlp("Nobody said the dog had chased the cat")
        self.assertEqual(doc[5]._.holmes.is_negated, True)

    def test_negator_negation_within_child_clause(self):
        doc = nlp("The dog chased the cat who was not happy")
        self.assertEqual(doc[2]._.holmes.is_negated, False)

    def test_operator_negation_within_child_clause(self):
        doc = nlp("The dog chased the cat who told nobody")
        self.assertEqual(doc[2]._.holmes.is_negated, False)

    def test_passive(self):
        doc = nlp("The dog was chased")
        self.assertEqual(
            doc[3]._.holmes.string_representation_of_children(), '1:nsubjpass; 2:auxpass')

    def test_used_to_positive(self):
        doc = nlp("The dog always used to chase the cat")
        self.assertEqual(
            doc[3]._.holmes.string_representation_of_children(), '-6:None')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:nsubj; 2:advmod; 4:aux; 7:dobj')

    def test_used_to_negative_1(self):
        doc = nlp("The dog was used to chase the cat")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:nsubjpass; 2:auxpass; 5:xcomp')

    def test_used_to_negative_2(self):
        doc = nlp("The dog used the mouse to chase the cat")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                         '1:nsubj; 4:dobj; 6:xcomp')

    def test_going_to(self):
        doc = nlp("The dog is going to chase the cat")
        self.assertEqual(
            doc[3]._.holmes.string_representation_of_children(), '-6:None')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:nsubj; 2:aux; 4:aux; 7:dobj')

    def test_was_going_to(self):
        doc = nlp("The dog was going to chase the cat")
        self.assertEqual(
            doc[3]._.holmes.string_representation_of_children(), '-6:None')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 2:aux(U); 4:aux(U); 7:dobj(U)')

    def test_complementizing_clause_active_child_clause_active(self):
        doc = nlp("The dog decided to chase the cat")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 3:aux; 6:dobj')

    def test_complementizing_clause_passive_child_clause_active(self):
        doc = nlp("The dog was ordered to chase the cat")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 4:aux; 7:dobj')

    def test_complementizing_clause_object_child_clause_active(self):
        doc = nlp("The mouse ordered the dog to chase the cat")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '4:nsubj(U); 5:aux; 8:dobj')

    def test_complementizing_clause_active_child_clause_passive(self):
        doc = nlp("The dog decided to be chased")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:nsubjpass(U); 3:aux; 4:auxpass')

    def test_complementizing_clause_passive_child_clause_passive(self):
        doc = nlp("The dog was ordered to be chased")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:nsubjpass(U); 4:aux; 5:auxpass')

    def test_complementizing_clause_object_child_clause_passive(self):
        doc = nlp("The mouse ordered the dog to be chased")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '4:nsubjpass(U); 5:aux; 6:auxpass')

    def test_complementization_with_conjunction_and_agent(self):
        doc = nlp(
            "The mouse ordered the dog and the cat to be chased by the cat and the tiger")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                         '4:nsubjpass(U); 7:nsubjpass(U); 8:aux; 9:auxpass; 11:agent; 13:pobjb; 16:pobjb')

    def test_complementizing_clause_atypical_conjunction(self):
        doc = nlp(
            "I had spent last week ruminating and that I knew")
        self.assertIn(doc[5]._.holmes.string_representation_of_children(),
                         ('0:nsubj(U)', '0:nsubj(U); 6:cc', '0:nsubj(U); 6:cc; 9:conj'))

    def test_who_one_antecedent(self):
        doc = nlp("The dog who chased the cat was tired")
        self.assertEqual(
            doc[3]._.holmes.string_representation_of_children(), '1:nsubj; 5:dobj')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_who_predicate_conjunction(self):
        doc = nlp("The dog who chased and caught the cat was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:nsubj; 4:cc; 5:conj')
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:nsubj; 7:dobj')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_who_many_antecedents(self):
        doc = nlp(
            "The lion, the tiger and the dog who chased the cat were tired")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 4:nsubj(U); 7:nsubj; 11:dobj')

    def test_which_one_antecedent(self):
        doc = nlp("The dog which chased the cat was tired")
        self.assertEqual(
            doc[3]._.holmes.string_representation_of_children(), '1:nsubj; 5:dobj')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_which_many_antecedents(self):
        doc = nlp(
            "The lion, the tiger and the dog which chased the cat were tired")
        self.assertIn(doc[9]._.holmes.string_representation_of_children(),
                         ('1:nsubj(U); 4:nsubj(U); 7:nsubj; 11:dobj', 
                         '1:nsubj; 4:nsubj(U); 7:nsubj(U); 11:dobj'))
        self.assertIn(
            doc[8]._.holmes.string_representation_of_children(), ('-8:None', '-2:None'))

    def test_that_subj_one_antecedent(self):
        doc = nlp("The dog that chased the cat was tired")
        self.assertEqual(
            doc[3]._.holmes.string_representation_of_children(), '1:nsubj; 5:dobj')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_that_predicate_conjunction(self):
        doc = nlp(
            "The dog that chased and caught the cat was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:nsubj; 4:cc; 5:conj')
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:nsubj; 7:dobj')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_that_subj_many_antecedents(self):
        doc = nlp(
            "The dog and the tiger that chased the cat were tired")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 4:nsubj; 8:dobj')

    def test_that_obj_one_antecedent(self):
        doc = nlp("The cat that the dog chased was tired")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:dobj; 4:nsubj')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_that_obj_many_antecedents(self):
        doc = nlp(
            "The cat and the mouse that the dog chased were tired")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '1:dobj; 4:dobj(U); 7:nsubj')

    def test_relant_one_antecedent(self):
        doc = nlp("The cat the dog chased was tired")
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '1:relant; 3:nsubj')

    def test_relant_predicate_conjunction(self):
        doc = nlp(
            "The cat the dog chased and pursued were tired")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '1:relant; 3:nsubj; 5:cc; 6:conj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:relant; 3:nsubj(U)')

    def test_displaced_preposition_phrasal_verb(self):
        doc = nlp("The office you ate your roll in was new")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:pobj')

    def test_displaced_preposition_no_complementizer(self):
        doc = nlp("The office you ate your roll at was new")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:pobj')
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '4:poss')
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:pobjp; 2:nsubj; 5:dobj; 6:prep')

    def test_displaced_preposition_no_complementizer_with_conjunction(self):
        doc = nlp(
            "The building and the office you ate and consumed your roll at were new")
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '1:pobj(U); 4:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:pobjp(U); 4:pobjp(U); 5:nsubj; 7:cc; 8:conj; 11:prep(U)')
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '1:pobjp(U); 4:pobjp; 5:nsubj(U); 10:dobj; 11:prep')

    def test_displaced_preposition_no_complementizer_with_second_preposition(self):
        doc = nlp(
            "The office you ate your roll with gusto at was new")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '1:pobj')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '4:poss; 6:prepposs(U); 7:pobjp(U)')
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:pobjp; 2:nsubj; 5:dobj; 6:prep; 7:pobjp; 8:prep')

    def test_displaced_preposition_no_complementizer_with_second_preposition_and_conjunction(self):
        doc = nlp(
            "The building and the office you ate and consumed your roll with gusto at were new")
        self.assertEqual(doc[13]._.holmes.string_representation_of_children(),
                         '1:pobj(U); 4:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:pobjp(U); 4:pobjp(U); 5:nsubj; 7:cc; 8:conj; 13:prep(U)')
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '1:pobjp(U); 4:pobjp; 5:nsubj(U); 10:dobj; 11:prep; 12:pobjp; 13:prep')

    def test_displaced_preposition_that(self):
        doc = nlp("The office that you ate your roll at was new")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '1:pobj')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '5:poss')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '1:pobjp; 3:nsubj; 6:dobj; 7:prep')

    def test_displaced_preposition_that_preposition_points_to_that(self):
        # For some reason gets a different spaCy representation that the previous one
        doc = nlp("The building that you ate your roll at was new")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '1:pobj')
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '5:poss')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '1:pobjp; 3:nsubj; 6:dobj; 7:prep')

    def test_displaced_preposition_that_with_conjunction(self):
        doc = nlp(
            "The building and the office that you ate and consumed your roll at were new")
        self.assertIn(doc[12]._.holmes.string_representation_of_children(),
                         ('1:pobj(U); 4:pobj', '1:pobj; 4:pobj(U)'))
        self.assertIn(doc[7]._.holmes.string_representation_of_children(),
                         ('1:pobjp(U); 4:pobjp(U); 6:nsubj; 8:cc; 9:conj; 12:prep(U)', 
                         '1:pobjp(U); 4:pobjp; 6:nsubj; 8:cc; 9:conj; 12:prep(U)'))
        self.assertIn(doc[9]._.holmes.string_representation_of_children(),
                         ('1:pobjp(U); 4:pobjp; 6:nsubj(U); 11:dobj; 12:prep', '1:pobjp; 4:pobjp(U); 6:nsubj(U); 11:dobj; 12:prep'))

    def test_displaced_preposition_that_with_second_preposition_preposition_points_to_that(self):
        doc = nlp(
            "The building that you ate your roll with gusto at was new")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '5:poss; 7:prepposs(U); 8:pobjp(U)')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '1:pobjp; 3:nsubj; 6:dobj; 7:prep; 8:pobjp; 9:prep')

    def test_displaced_preposition_that_with_second_preposition(self):
        doc = nlp(
            "The office that you ate your roll with gusto at was new")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '5:poss; 7:prepposs(U); 8:pobjp(U)')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '1:pobjp; 3:nsubj; 6:dobj; 7:prep; 8:pobjp; 9:prep')

    def test_displaced_preposition_that_with_second_preposition_and_conjunction(self):
        doc = nlp(
            "The building and the office that you ate and consumed your roll with gusto at were new")
        self.assertEqual(doc[14]._.holmes.string_representation_of_children(),
                         '1:pobj; 4:pobj(U)')
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '1:pobjp(U); 4:pobjp(U); 6:nsubj; 8:cc; 9:conj; 14:prep(U)')
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:pobjp; 4:pobjp(U); 6:nsubj(U); 11:dobj; 12:prep; 13:pobjp; 14:prep')

    def test_simple_whose_clause(self):
        doc = nlp("The dog whose owner I met was tired")
        self.assertEqual(
            doc[3]._.holmes.string_representation_of_children(), '1:poss')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_whose_clause_with_conjunction_of_possessor(self):
        doc = nlp("The dog whose owner and friend I met was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:poss; 4:cc; 5:conj')
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:poss')
        self.assertEqual(
            doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_whose_clause_with_conjunction_of_possessed(self):
        doc = nlp("The lion and dog whose owner I met were tired")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:poss(U); 3:poss')
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '-4:None')

    def test_phrasal_verb_1(self):
        doc = nlp("He took out insurance")
        self.assertEqual(doc[1]._.holmes.lemma, 'take out')
        self.assertEqual(
            doc[1]._.holmes.string_representation_of_children(), '0:nsubj; 3:dobj')

    def test_participle(self):
        doc = nlp("An adopted child")
        self.assertEqual(doc[1]._.holmes.lemma, 'adopt')

    def test_positive_modal_verb(self):
        doc = nlp("He should do it")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                         '0:nsubj(U); 1:aux; 3:dobj(U)')

    def test_negative_modal_verb(self):
        doc = nlp("He cannot do it")
        self.assertIn(doc[3]._.holmes.string_representation_of_children(),
                         ('0:nsubj(U); 1:aux; 2:neg(U); 4:dobj(U)', 
                         '0:nsubj(U); 1:aux; 2:aux; 4:dobj(U)'))
        self.assertTrue(doc[3]._.holmes.is_negated)

    def test_ought_to(self):
        doc = nlp("He ought to do it")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '0:nsubj(U); 2:aux; 4:dobj')

    def test_phrasal_verb_2(self):
        doc = nlp("He will have been doing it")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '0:nsubj; 1:aux; 2:aux; 3:aux; 5:dobj')

    def test_pobjb_1(self):
        doc = nlp("Eating by employees")
        self.assertEqual(
            doc[0]._.holmes.string_representation_of_children(), '1:prep; 2:pobjb')

    def test_pobjb_2(self):
        doc = nlp("Eating of icecream")
        self.assertEqual(
            doc[0]._.holmes.string_representation_of_children(), '1:prep; 2:pobjo')

    def test_pobjt(self):
        doc = nlp("Travelling to Munich")
        self.assertEqual(
            doc[0]._.holmes.string_representation_of_children(), '1:prep; 2:pobjt')

    def test_dative_prepositional_phrase(self):
        doc = nlp("He gave it to the employee")
        self.assertIn(doc[1]._.holmes.string_representation_of_children(),
                         ('0:nsubj; 2:dobj; 3:prep; 5:pobjt', '0:nsubj; 2:dobj; 3:dative; 5:dative'))
        self.assertFalse(doc[3]._.holmes.is_matchable)

    def test_dative_prepositional_phrase_with_conjunction(self):
        doc = nlp("He gave it to the employee and the boss")
        self.assertIn(doc[1]._.holmes.string_representation_of_children(),
                         ('0:nsubj; 2:dobj; 3:prep; 5:pobjt; 8:pobjt', '0:nsubj; 2:dobj; 3:dative; 5:dative; 8:dative'))
        self.assertFalse(doc[3]._.holmes.is_matchable)

    def test_simple_participle_phrase(self):
        doc = nlp("He talked about the cat chased by the dog")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '4:dobj; 6:agent; 8:pobjb')

    def test_participle_phrase_with_conjunction(self):
        doc = nlp(
            "He talked about the cat and the mouse chased by the dog and the tiger")
        self.assertIn(doc[8]._.holmes.string_representation_of_children(),
                         ('4:dobj; 7:dobj; 9:agent; 11:pobjb; 14:dobj', '4:dobj; 7:dobj; 9:agent; 11:pobjb; 14:pobjb'))

    def test_subjective_modifying_adverbial_phrase(self):
        doc = nlp("The lion-chased cat came home")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:advmodsubj; 4:advmodobj')

    def test_subjective_modifying_adverbial_phrase_with_conjunction(self):
        doc = nlp("The lion-chased cat and mouse came home")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:advmodsubj; 4:advmodobj; 6:advmodobj(U)')

    def test_objective_modifying_adverbial_phrase(self):
        doc = nlp("The cat-chasing lion came home")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:advmodobj; 4:advmodsubj')

    def test_objective_modifying_adverbial_phrase_with_conjunction(self):
        doc = nlp("The cat-chasing lion and dog came home")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '1:advmodobj; 4:advmodsubj; 6:advmodsubj(U)')

    def test_verb_prepositional_complement_simple_active(self):
        doc = nlp("The dog was thinking about chasing a cat")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:nsubj(U); 7:dobj')

    def test_verb_prepositional_complement_with_conjunction_active(self):
        doc = nlp(
            "The dog and the lion were thinking about chasing a cat and a mouse")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 4:nsubj(U); 10:dobj; 13:dobj')

    def test_verb_prepositional_complement_with_relative_clause_active(self):
        doc = nlp(
            "The dog who was thinking about chasing a cat came home")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:nsubj(U); 8:dobj')

    def test_verb_preposition_complement_with_coreferring_pronoun_active(self):
        doc = nlp(
            "He saw a dog. It was thinking about chasing a cat")
        self.assertEqual(
            doc[9]._.holmes.string_representation_of_children(), '5:nsubj(U); 11:dobj')

    def test_verb_preposition_complement_with_non_coreferring_pronoun_active(self):
        doc = nlp("It was thinking about chasing a cat")
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '6:dobj')

    def test_adjective_prepositional_complement_simple_active(self):
        doc = nlp("The dog was worried about chasing a cat")
        self.assertEqual(
            doc[5]._.holmes.string_representation_of_children(), '1:nsubj(U); 7:dobj')

    def test_adjective_prepositional_complement_with_conjunction_active(self):
        doc = nlp(
            "The dog and the lion were worried about chasing a cat and a mouse")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 4:nsubj(U); 10:dobj; 13:dobj')

    def test_adjective_prepositional_complement_with_relative_clause_active(self):
        doc = nlp(
            "The dog who was worried about chasing a cat came home")
        self.assertEqual(
            doc[6]._.holmes.string_representation_of_children(), '1:nsubj(U); 8:dobj')

    def test_adjective_preposition_complement_with_coreferring_pronoun_active(self):
        doc = nlp(
            "He saw a dog. He was worried about chasing a cat")
        self.assertEqual(
            doc[9]._.holmes.string_representation_of_children(), '5:nsubj(U); 11:dobj')

    def test_adjective_preposition_complement_with_non_coreferring_pronoun_active(self):
        doc = nlp("It was worried about chasing a cat")
        self.assertEqual(
            doc[4]._.holmes.string_representation_of_children(), '6:dobj')

    def test_verb_prepositional_complement_simple_passive(self):
        doc = nlp(
            "The cat was thinking about being chased by a dog")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:nsubjpass(U); 5:auxpass; 7:agent; 9:pobjb')

    def test_verb_prepositional_complement_with_conjunction_passive(self):
        doc = nlp(
            "The cat and the mouse were thinking about being chased by a dog and a lion")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:nsubjpass(U); 4:nsubjpass(U); 8:auxpass; 10:agent; 12:pobjb; 15:pobjb')

    def test_verb_prepositional_complement_with_relative_clause_passive(self):
        doc = nlp(
            "The cat who was thinking about being chased by a dog came home")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '1:nsubjpass(U); 6:auxpass; 8:agent; 10:pobjb')

    def test_verb_preposition_complement_with_coreferring_pronoun_passive(self):
        doc = nlp(
            "He saw a dog. It was thinking about being chased by a cat")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                         '5:nsubjpass(U); 9:auxpass; 11:agent; 13:pobjb')

    def test_verb_preposition_complement_with_non_coreferring_pronoun_passive(self):
        doc = nlp("It was thinking about being chased by a cat")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '4:auxpass; 6:agent; 8:pobjb')

    def test_adjective_prepositional_complement_simple_passive(self):
        doc = nlp("The cat was worried about being chased by a dog")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '1:nsubjpass(U); 5:auxpass; 7:agent; 9:pobjb')

    def test_adjective_prepositional_complement_with_conjunction_passive(self):
        doc = nlp(
            "The cat and the mouse were worried about being chased by a dog and a lion")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:nsubjpass(U); 4:nsubjpass(U); 8:auxpass; 10:agent; 12:pobjb; 15:pobjb')

    def test_adjective_prepositional_complement_with_relative_clause_passive(self):
        doc = nlp(
            "The cat who was worried about being chased by a dog came home")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '1:nsubjpass(U); 6:auxpass; 8:agent; 10:pobjb')

    def test_adjective_preposition_complement_with_coreferring_pronoun_passive(self):
        doc = nlp(
            "He saw a dog. It was worried about being chased by a cat")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                         '5:nsubjpass(U); 9:auxpass; 11:agent; 13:pobjb')

    def test_adjective_preposition_complement_with_non_coreferring_pronoun_passive(self):
        doc = nlp("It was worried about being chased by a cat")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '4:auxpass; 6:agent; 8:pobjb')

    def test_verb_prepositional_complement_with_conjunction_of_dependent_verb(self):
        doc = nlp(
            "The cat and the mouse kept on singing and shouting")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 4:nsubj(U); 8:cc; 9:conj')
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 4:nsubj(U)')

    def test_verb_p_c_with_conjunction_of_dependent_verb_and_coreferring_pronoun(self):
        doc = nlp("I saw a cat. It kept on singing and shouting")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '5:nsubj(U); 9:cc; 10:conj')
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                         '5:nsubj(U)')

    def test_verb_p_c_with_conjunction_of_dependent_verb_and_non_coreferring_pronoun_1(self):
        doc = nlp("It kept on singing and shouting")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '0:nsubj(U); 4:cc; 5:conj')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '0:nsubj(U)')

    def test_adjective_prepositional_complement_with_conjunction_of_dependent_verb(self):
        doc = nlp(
            "The cat and the mouse were worried about singing and shouting")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 4:nsubj(U); 9:cc; 10:conj')
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                         '1:nsubj(U); 4:nsubj(U)')

    def test_adjective_p_c_with_conjunction_of_dependent_verb_and_coreferring_pronoun(self):
        doc = nlp(
            "I saw a cat. It was worried about singing and shouting")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                         '5:nsubj(U); 10:cc; 11:conj')
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '5:nsubj(U)')

    def test_verb_p_c_with_conjunction_of_dependent_verb_and_non_coreferring_pronoun_2(self):
        doc = nlp("It was worried about singing and shouting")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '5:cc; 6:conj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '')

    def test_single_preposition_dependency_added_to_noun(self):
        doc = nlp(
            "The employee needs insurance for the next five years")
        self.assertIn(doc[3]._.holmes.string_representation_of_children(),
                         ('4:prepposs(U); 8:pobjp(U)', '4:prep; 8:pobjp'))

    def test_multiple_preposition_dependencies_added_to_noun(self):
        doc = nlp(
            "The employee needs insurance for the next five years and in Europe")
        self.assertIn(doc[3]._.holmes.string_representation_of_children(),
                         ('4:prepposs(U); 8:pobjp(U); 10:prepposs(U); 11:pobjp(U)',
                         '4:prep; 8:pobjp; 10:prep; 11:pobjp'))

    def test_single_preposition_dependency_added_to_coreferring_pronoun(self):
        doc = nlp(
            "We discussed the house. The employee needs it for the next five years")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '9:prepposs(U); 13:pobjp(U)')

    def test_dependencies_not_added_to_sibling_to_the_right(self):
        doc = nlp("He saw them and laughed")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '0:nsubj(U)')

    def test_coreference_within_sentence(self):
        doc = nlp("The employee got home and he was surprised")
        self.assertEqual(
            doc[1]._.holmes.token_and_coreference_chain_indexes, [1, 5])
        self.assertEqual(
            doc[5]._.holmes.token_and_coreference_chain_indexes, [5, 1])
        self.assertEqual(
            doc[3]._.holmes.token_and_coreference_chain_indexes, [3])

    def test_coreference_between_sentences(self):
        doc = nlp("The employee got home. He was surprised")
        self.assertEqual(
            doc[1]._.holmes.token_and_coreference_chain_indexes, [1, 5])
        self.assertEqual(
            doc[5]._.holmes.token_and_coreference_chain_indexes, [5, 1])
        self.assertEqual(
            doc[3]._.holmes.token_and_coreference_chain_indexes, [3])

    def test_coreference_three_items_in_chain(self):
        doc = nlp(
            "Richard was at work. He went home. He was surprised")
        self.assertEqual(
            doc[0]._.holmes.token_and_coreference_chain_indexes, [0, 5, 9])
        self.assertEqual(
            doc[5]._.holmes.token_and_coreference_chain_indexes, [5, 0, 9])
        self.assertEqual(
            doc[9]._.holmes.token_and_coreference_chain_indexes, [9, 0, 5])
        self.assertEqual(
            doc[3]._.holmes.token_and_coreference_chain_indexes, [3])

    def test_coreference_conjunction_in_antecedent(self):
        doc = nlp(
            "Richard and Carol came to work. They had a discussion")
        self.assertEqual(
            doc[0]._.holmes.token_and_coreference_chain_indexes, [0, 7])
        self.assertEqual(
            doc[2]._.holmes.token_and_coreference_chain_indexes, [2, 7])
        self.assertEqual(
            doc[7]._.holmes.token_and_coreference_chain_indexes, [7, 0, 2])
        self.assertEqual(
            doc[3]._.holmes.token_and_coreference_chain_indexes, [3])

    def test_coreference_within_relative_clause(self):
        doc = nlp("The man who knows himself has an advantage")
        self.assertEqual(
            doc[1]._.holmes.token_and_coreference_chain_indexes, [1, 4])
        self.assertEqual(
            doc[4]._.holmes.token_and_coreference_chain_indexes, [4, 1])

    def test_maximum_mentions_difference(self):
        doc = nlp("""Richard came to work. He was happy. He was happy. He was happy.
        He was happy. He was happy. He was happy. He was happy. He was happy.""")
        self.assertEqual(
            doc[0]._.holmes.token_and_coreference_chain_indexes, [0, 5, 9, 13])
        self.assertEqual(doc[5]._.holmes.token_and_coreference_chain_indexes, [
                         5, 0, 9, 13, 18])
        self.assertEqual(doc[9]._.holmes.token_and_coreference_chain_indexes, [
                         9, 0, 5, 13, 18, 22])
        self.assertEqual(doc[13]._.holmes.token_and_coreference_chain_indexes, [
                         13, 0, 5, 9, 18, 22, 26])
        self.assertEqual(doc[18]._.holmes.token_and_coreference_chain_indexes,
                         [18, 5, 9, 13, 22, 26, 30])
        self.assertEqual(doc[22]._.holmes.token_and_coreference_chain_indexes,
                         [22, 9, 13, 18, 26, 30, 34])
        self.assertEqual(doc[26]._.holmes.token_and_coreference_chain_indexes, [
                         26, 13, 18, 22, 30, 34])
        self.assertEqual(doc[30]._.holmes.token_and_coreference_chain_indexes, [
                         30, 18, 22, 26, 34])
        self.assertEqual(
            doc[34]._.holmes.token_and_coreference_chain_indexes, [34, 22, 26, 30])

    def test_most_specific_coreferring_term_index_with_pronoun(self):
        doc = nlp("I saw Richard. The person came home. He was surprised.")
        self.assertEqual(
            doc[2]._.holmes.most_specific_coreferring_term_index, 2)
        self.assertEqual(
            doc[5]._.holmes.most_specific_coreferring_term_index, 2)
        self.assertEqual(
            doc[9]._.holmes.most_specific_coreferring_term_index, 2)
        self.assertEqual(
            doc[3]._.holmes.most_specific_coreferring_term_index, None)

    def test_most_specific_coreferring_term_index_without_pronoun(self):
        doc = nlp("I saw Richard. The person came home.")
        self.assertEqual(
            doc[2]._.holmes.most_specific_coreferring_term_index, 2)
        self.assertEqual(
            doc[5]._.holmes.most_specific_coreferring_term_index, 2)
        self.assertEqual(
            doc[3]._.holmes.most_specific_coreferring_term_index, None)

    def test_most_specific_coreferring_term_index_with_coordination(self):
        doc = nlp("I saw Richard. The person and Maria were talking. They came home.")
        self.assertEqual(
            doc[2]._.holmes.most_specific_coreferring_term_index, 2)
        self.assertEqual(
            doc[5]._.holmes.most_specific_coreferring_term_index, 2)
        self.assertEqual(
            doc[7]._.holmes.most_specific_coreferring_term_index, None)
        self.assertEqual(
            doc[9]._.holmes.most_specific_coreferring_term_index, None)

    def test_adjective_verb_clause_subjective_simple(self):
        doc = nlp("Richard was glad to understand.")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '0:arg(U); 3:aux')

    def test_adjective_verb_clause_subjective_compound(self):
        doc = nlp(
            "Richard and Thomas were glad and happy to understand and to comprehend.")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 7:aux; 9:cc; 11:conj')
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 10:aux')

    def test_adjective_verb_clause_objective_simple(self):
        doc = nlp("Richard was hard to reach.")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                         '0:arg(U); 3:aux')

    def test_adjective_verb_clause_objective_compound(self):
        doc = nlp(
            "Richard and Thomas were hard and difficult to reach and to call.")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 7:aux; 9:cc; 11:conj')
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                         '0:arg(U); 2:arg(U); 10:aux')

    def test_prepositional_phrase_dependent_on_noun_no_conjunction(self):
        doc = nlp("Houses in the village.")
        self.assertEqual(doc[0]._.holmes.string_representation_of_children(),
                         '1:prep; 3:pobjp')

    def test_prepositional_phrase_dependent_on_noun_with_conjunction(self):
        doc = nlp("Houses in the village and the town.")
        self.assertEqual(doc[0]._.holmes.string_representation_of_children(),
                         '1:prep; 3:pobjp; 6:pobjp')

    def test_simple_relative_prepositional_phrase(self):
        doc = nlp("The table from which we ate.")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                         '-2:None')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                         '1:pobjp; 2:prep; 4:nsubj')

    def test_conjunction_relative_prepositional_phrase(self):
        doc = nlp(
            "The table and the chair from which you and I ate and drank.")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                         '-2:None')
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                         '1:pobjp(U); 4:pobjp(U); 5:prep(U); 7:nsubj; 9:nsubj; 11:cc; 12:conj')
        self.assertEqual(doc[12]._.holmes.string_representation_of_children(),
                         '1:pobjp; 4:pobjp(U); 5:prep; 7:nsubj(U); 9:nsubj(U)')

    def test_parent_token_indexes(self):
        doc = nlp("Houses in the village.")
        self.assertEqual(doc[0]._.holmes.string_representation_of_children(),
                         '1:prep; 3:pobjp')
        self.assertEqual(doc[3]._.holmes.coreference_linked_parent_dependencies, [
                         [0, 'pobjp'], [1, 'pobj']])
        self.assertEqual(doc[3]._.holmes.string_representation_of_parents(),
                         '0:pobjp; 1:pobj')

    def test_direct_matching_reprs_only_lemma(self):
        doc = nlp("dog")
        self.assertEqual(doc[0]._.holmes.direct_matching_reprs, ['dog'])

    def test_direct_matching_reprs_text_and_lemma(self):
        doc = nlp("dogs")
        self.assertEqual(doc[0]._.holmes.direct_matching_reprs, ['dog', 'dogs'])

    def test_derived_lemma_from_dictionary(self):
        doc = nlp("A long imprisonment.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'imprison')

    def test_derived_lemma_root_word_from_dictionary(self):
        doc = nlp("He was imprisoned.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'imprison')

    def test_derived_lemma_ization(self):
        doc = nlp("Linearization problems.")
        self.assertEqual(doc[0]._.holmes.derived_lemma, 'linearize')

    @unittest.skipIf(nlp.meta['version'] == '3.2.0', 'Version fluke')
    def test_derived_lemma_isation(self):
        doc = nlp("Linearisation problems.")
        self.assertEqual(doc[0]._.holmes.derived_lemma, 'linearise')

    def test_derived_lemma_ically(self):
        doc = nlp("They used it very economically.")
        self.assertEqual(doc[4]._.holmes.derived_lemma, 'economic')

    def test_derived_lemma_ibly(self):
        doc = nlp("It stank horribly.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'horrible')

    def test_derived_lemma_ably(self):
        doc = nlp("Regrettably it was a problem.")
        self.assertEqual(doc[0]._.holmes.derived_lemma, 'regrettable')

    def test_derived_lemma_ily(self):
        doc = nlp("He used the software happily.")
        self.assertEqual(doc[4]._.holmes.derived_lemma, 'happy')

    def test_derived_lemma_ly(self):
        doc = nlp("It went swingingly.")
        self.assertEqual(doc[2]._.holmes.derived_lemma, 'swinging')

    def test_derived_lemma_ness(self):
        doc = nlp("There was a certain laxness.")
        self.assertEqual(doc[4]._.holmes.derived_lemma, 'lax')

    def test_derived_lemma_ness_with_y(self):
        doc = nlp("There was a certain bawdiness.")
        self.assertEqual(doc[4]._.holmes.derived_lemma, 'bawdy')

    def test_derived_lemma_ing(self):
        doc = nlp("The playing was very loud.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'play')

    def test_derived_lemma_ing_with_doubling(self):
        doc = nlp("The ramming of the vehicle was very loud.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'ram')

    def test_derived_lemma_ication(self):
        doc = nlp("The verification of the results.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'verify')

    def test_derived_lemma_ation_3(self):
        doc = nlp("The manipulation of the results.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'manipulate')

    def test_derived_lemma_ication_in_icate(self):
        doc = nlp("The domestication of the dog.")
        self.assertEqual(doc[1]._.holmes.derived_lemma, 'domesticate')

    def test_derived_lemma_and_lemma_identical(self):
        doc = nlp("vehicle.")
        self.assertEqual(doc[0]._.holmes.derived_lemma, 'vehicle')

    def test_derivation_matching_reprs_only_lemma(self):
        doc = nlp("dog")
        self.assertEqual(doc[0]._.holmes.derivation_matching_reprs, None)

    def test_derivation_matching_reprs_text_and_lemma(self):
        doc = nlp("happiness")
        self.assertEqual(doc[0]._.holmes.derivation_matching_reprs, ['happy'])

    def test_formerly_problematic_sentence_no_exception_thrown(self):
        nlp(
            "Mothers with vouchers for themselves and their young childrenwere finding that many eligible products were gone.")

    def test_pipe(self):
        docs = list(nlp.pipe(['some dogs', 'some cats']))
        self.assertEqual(docs[0][1]._.holmes.lemma, 'dog')
        self.assertEqual(docs[1][1]._.holmes.lemma, 'cat')

    def test_predicative_adjective_in_relative_clause(self):
        doc = nlp("He saw his son, who was sad.")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(), '2:poss; 6:relcl; 7:amod')

    def test_question_word_initial(self):
        doc = nlp("Whom did you talk to?")
        self.assertTrue(doc[0]._.holmes.is_initial_question_word)

    def test_question_word_after_preposition(self):
        doc = nlp("To whom did you talk?")
        self.assertTrue(doc[1]._.holmes.is_initial_question_word)

    def test_question_word_after_double_preposition(self):
        doc = nlp("Because of whom did you come?")
        self.assertTrue(doc[2]._.holmes.is_initial_question_word)

    def test_question_word_in_complex_phrase(self):
        doc = nlp("On the basis of what information did you come?")
        self.assertTrue(doc[4]._.holmes.is_initial_question_word)

    def test_question_word_control_1(self):
        doc = nlp(". Whom did you talk to?")
        for token in doc:
            self.assertFalse(token._.holmes.is_initial_question_word)

    def test_question_word_control_2(self):
        doc = nlp("You came because of whom?")
        for token in doc:
            self.assertFalse(token._.holmes.is_initial_question_word)
