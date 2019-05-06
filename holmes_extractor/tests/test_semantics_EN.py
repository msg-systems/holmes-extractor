import unittest
from holmes_extractor.semantics import SemanticAnalyzerFactory

analyzer = SemanticAnalyzerFactory().semantic_analyzer(model='en_coref_lg', debug=False)

class EnglishSemanticAnalyzerTest(unittest.TestCase):

    def test_initialize_semantic_dependencies(self):
        doc = analyzer.parse("The dog chased the cat.")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '1:nsubj; 4:dobj')
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(), '')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '')

    def test_one_righthand_sibling_with_and_conjunction(self):
        doc = analyzer.parse("The dog and the hound chased the cat")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4])
        self.assertFalse(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertFalse(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])

    def test_many_righthand_siblings_with_and_conjunction(self):
        doc = analyzer.parse("The dog, the wolf and the hound chased the cat")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4, 7])
        self.assertFalse(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertFalse(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertFalse(doc[7]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])
        self.assertEqual(doc[7]._.holmes.righthand_siblings, [])

    def test_one_righthand_sibling_with_or_conjunction(self):
        doc = analyzer.parse("The dog or the hound chased the cat")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4])
        self.assertTrue(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertTrue(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])

    def test_many_righthand_siblings_with_or_conjunction(self):
        doc = analyzer.parse("The dog, the wolf or the hound chased the cat")
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [4, 7])
        self.assertTrue(doc[1]._.holmes.is_involved_in_or_conjunction)
        self.assertTrue(doc[4]._.holmes.is_involved_in_or_conjunction)
        self.assertTrue(doc[7]._.holmes.is_involved_in_or_conjunction)
        self.assertEqual(doc[4]._.holmes.righthand_siblings, [])
        self.assertEqual(doc[7]._.holmes.righthand_siblings, [])

    def test_righthand_siblings_of_semantic_children_two(self):
        doc = analyzer.parse("The large and strong dog came home")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(), '1:amod; 3:amod')
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [3])

    def test_righthand_siblings_of_semantic_children_many(self):
        doc = analyzer.parse("The large, strong and fierce dog came home")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(), '1:amod; 3:amod; 5:amod')
        self.assertEqual(doc[1]._.holmes.righthand_siblings, [])
        self.assertEqual(doc[3]._.holmes.righthand_siblings, [5])
        # Conjunction between 1 and 3 is already reflected in the underlying spaCy structure and does not need to be dealt with by Holmes

    def test_semantic_children_of_righthand_siblings_two(self):
        doc = analyzer.parse("The large dog and cat")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                '1:amod; 3:cc; 4:conj')
        self.assertEqual(doc[2]._.holmes.righthand_siblings, [4])
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(), '1:amod(U)')

    def test_semantic_children_of_righthand_siblings_many(self):
        doc = analyzer.parse("The large dog, cat and mouse")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '1:amod; 4:conj')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '1:amod(U); 5:cc; 6:conj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(), '1:amod(U)')

    def test_predicative_adjective(self):
        doc = analyzer.parse("The dog was big")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(), '3:amod')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_predicative_adjective_with_conjunction(self):
        doc = analyzer.parse("The dog and the cat were big and strong")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                '2:cc; 4:conj; 6:amod; 8:amod')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(), '6:amod; 8:amod')

    def test_predicative_adjective_with_non_coreferring_pronoun(self):
        doc = analyzer.parse("It was big")
        self.assertEqual(doc[0]._.holmes.string_representation_of_children(), '')
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(), '0:nsubj; 2:acomp')

    def test_predicative_adjective_with_coreferring_pronoun(self):
        doc = analyzer.parse("I saw a dog. It was big")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '7:amod')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(), '-6:None')

    def test_negator_negation_within_clause(self):
        doc = analyzer.parse("The dog did not chase the cat")
        self.assertEqual(doc[4]._.holmes.is_negated, True)

    def test_operator_negation_within_clause(self):
        doc = analyzer.parse("No dog chased any cat")
        self.assertEqual(doc[1]._.holmes.is_negated, True)
        self.assertEqual(doc[2]._.holmes.is_negated, False)

    def test_negator_negation_within_parent_clause(self):
        doc = analyzer.parse("It had not been claimed that the dog had chased the cat")
        self.assertEqual(doc[4]._.holmes.is_negated, True)

    def test_operator_negation_within_parent_clause(self):
        doc = analyzer.parse("Nobody said the dog had chased the cat")
        self.assertEqual(doc[5]._.holmes.is_negated, True)

    def test_negator_negation_within_child_clause(self):
        doc = analyzer.parse("The dog chased the cat who was not happy")
        self.assertEqual(doc[2]._.holmes.is_negated, False)

    def test_operator_negation_within_child_clause(self):
        doc = analyzer.parse("The dog chased the cat who told nobody")
        self.assertEqual(doc[2]._.holmes.is_negated, False)

    def test_passive(self):
        doc = analyzer.parse("The dog was chased")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(), '1:nsubjpass; 2:auxpass')

    def test_used_to_positive(self):
        doc = analyzer.parse("The dog used to chase the cat")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-5:None')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '1:nsubj; 3:aux; 6:dobj')

    def test_used_to_negative_1(self):
        doc = analyzer.parse("The dog was used to chase the cat")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '1:nsubjpass; 2:auxpass; 5:xcomp')

    def test_used_to_negative_2(self):
        doc = analyzer.parse("The dog used the mouse to chase the cat")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                '1:nsubj; 4:dobj; 6:xcomp')

    def test_going_to(self):
        doc = analyzer.parse("The dog is going to chase the cat")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(), '-6:None')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '1:nsubj; 2:aux; 4:aux; 7:dobj')

    def test_was_going_to(self):
        doc = analyzer.parse("The dog was going to chase the cat")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(), '-6:None')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 2:aux(U); 4:aux(U); 7:dobj(U)')

    def test_complementizing_clause_active_child_clause_active(self):
        doc = analyzer.parse("The dog decided to chase the cat")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 3:aux; 6:dobj')

    def test_complementizing_clause_passive_child_clause_active(self):
        doc = analyzer.parse("The dog was ordered to chase the cat")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:aux; 7:dobj')

    def test_complementizing_clause_object_child_clause_active(self):
        doc = analyzer.parse("The mouse ordered the dog to chase the cat")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '4:nsubj(U); 5:aux; 8:dobj')

    def test_complementizing_clause_active_child_clause_passive(self):
        doc = analyzer.parse("The dog decided to be chased")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '1:nsubjpass(U); 3:aux; 4:auxpass')

    def test_complementizing_clause_passive_child_clause_passive(self):
        doc = analyzer.parse("The dog was ordered to be chased")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '1:nsubjpass(U); 4:aux; 5:auxpass')

    def test_complementizing_clause_object_child_clause_passive(self):
        doc = analyzer.parse("The mouse ordered the dog to be chased")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                '4:nsubjpass(U); 5:aux; 6:auxpass')

    def test_complementization_with_conjunction_and_agent(self):
        doc = analyzer.parse(
                "The mouse ordered the dog and the cat to be chased by the cat and the tiger")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                '4:nsubjpass(U); 7:nsubjpass(U); 8:aux; 9:auxpass; 11:agent; 13:pobjb; 16:pobjb')

    def test_complementizing_clause_atypical_conjunction(self):
        doc = analyzer.parse("I had spent three years thinking and that I knew")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '4:nsubj(U)')

    def test_who_one_antecedent(self):
        doc = analyzer.parse("The dog who chased the cat was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(), '1:nsubj; 5:dobj')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_who_predicate_conjunction(self):
        doc = analyzer.parse("The dog who chased and caught the cat was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '1:nsubj; 4:cc; 5:conj')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '1:nsubj; 7:dobj')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_who_many_antecedents(self):
        doc = analyzer.parse("The lion, the tiger and the dog who chased the cat were tired")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:nsubj(U); 7:nsubj; 11:dobj')

    def test_which_one_antecedent(self):
        doc = analyzer.parse("The dog which chased the cat was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(), '1:nsubj; 5:dobj')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_who_predicate_conjunction(self):
        doc = analyzer.parse("The dog which chased and caught the cat was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '1:nsubj; 4:cc; 5:conj')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '1:nsubj; 7:dobj')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_which_many_antecedents(self):
        doc = analyzer.parse("The lion, the tiger and the dog which chased the cat were tired")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:nsubj(U); 7:nsubj; 11:dobj')
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(), '-8:None')

    def test_that_subj_one_antecedent(self):
        doc = analyzer.parse("The dog that chased the cat was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(), '1:nsubj; 5:dobj')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_that_predicate_conjunction(self):
        doc = analyzer.parse("The dog that chased and caught the cat was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '1:nsubj; 4:cc; 5:conj')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '1:nsubj; 7:dobj')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_that_subj_many_antecedents(self):
        doc = analyzer.parse("The dog and the tiger that chased the cat were tired")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:nsubj; 8:dobj')

    def test_that_obj_one_antecedent(self):
        doc = analyzer.parse("The cat that the dog chased was tired")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '1:dobj; 4:nsubj')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_that_obj_many_antecedents(self):
        doc = analyzer.parse("The cat and the mouse that the dog chased were tired")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '1:dobj(U); 4:dobj; 7:nsubj')

    def test_relant_one_antecedent(self):
        doc = analyzer.parse("The cat the dog chased was tired")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(), '1:relant; 3:nsubj')

    def test_relant_many_antecedents(self):
        doc = analyzer.parse("The cat and the mouse the dog and the tiger chased were tired")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                '1:relant(U); 4:relant; 6:nsubj; 9:nsubj')

    def test_relant_many_antecedents_and_predicate_conjunction(self):
        doc = analyzer.parse("The cat and the mouse the dog chased and pursued were tired")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                '1:relant(U); 4:relant; 6:nsubj; 8:cc; 9:conj')
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '1:relant(U); 4:relant; 6:nsubj(U)')

    def test_relant_multiple_predicate_conjunction(self):
        doc = analyzer.parse("The cat the dog pursued, chased and caught was dead")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '1:relant; 3:nsubj; 6:conj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '1:relant; 3:nsubj(U); 7:cc; 8:conj')
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '1:relant; 3:nsubj(U)')

    def test_displaced_preposition_phrasal_verb(self):
        doc = analyzer.parse("The office you ate your roll in was new")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(), '')

    def test_displaced_preposition_no_complementizer(self):
        doc = analyzer.parse("The office you ate your roll at was new")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '1:pobj')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),'4:poss')
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '2:nsubj; 5:dobj; 6:prep')

    def test_displaced_preposition_no_complementizer_with_conjunction(self):
        doc = analyzer.parse(
                "The building and the office you ate and consumed your roll at were new")
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                '1:pobj(U); 4:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '5:nsubj; 7:cc; 8:conj; 11:prep(U)')
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '5:nsubj(U); 10:dobj; 11:prep')

    def test_displaced_preposition_no_complementizer_with_second_preposition(self):
        doc = analyzer.parse("The office you ate your roll with gusto at was new")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '1:pobj')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '4:poss; 6:prepposs(U)')
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '2:nsubj; 5:dobj; 6:prep; 8:prep')

    def test_displaced_preposition_no_complementizer_with_second_preposition_and_conjunction(self):
        doc = analyzer.parse(
                "The building and the office you ate and consumed your roll with gusto at were new")
        self.assertEqual(doc[13]._.holmes.string_representation_of_children(),
                '1:pobj(U); 4:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '5:nsubj; 7:cc; 8:conj; 13:prep(U)')
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '5:nsubj(U); 10:dobj; 11:prep; 13:prep')

    def test_displaced_preposition_that(self):
        doc = analyzer.parse("The office that you ate your roll at was new")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                '1:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),'5:poss')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '3:nsubj; 6:dobj; 7:prep')

    def test_displaced_preposition_that_preposition_points_to_that(self):
        # For some reason gets a different spaCy representation that the previous one
        doc = analyzer.parse("The building that you ate your roll at was new")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                '1:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),'5:poss')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '3:nsubj; 6:dobj; 7:prep')

    def test_displaced_preposition_that_with_conjunction(self):
        doc = analyzer.parse(
                "The building and the office that you ate and consumed your roll at were new")
        self.assertEqual(doc[12]._.holmes.string_representation_of_children(),
                '1:pobj(U); 4:pobj')
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                '6:nsubj; 8:cc; 9:conj; 12:prep(U)')
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '6:nsubj(U); 11:dobj; 12:prep')

    def test_displaced_preposition_that_with_second_preposition_preposition_points_to_that(self):
        doc = analyzer.parse("The building that you ate your roll with gusto at was new")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '1:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '5:poss; 7:prepposs(U)')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '3:nsubj; 6:dobj; 7:prep; 9:prep')

    def test_displaced_preposition_that_with_second_preposition(self):
        doc = analyzer.parse("The office that you ate your roll with gusto at was new")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '1:pobj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '5:poss; 7:prepposs(U)')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '3:nsubj; 6:dobj; 7:prep; 9:prep')

    def test_displaced_preposition_that_with_second_preposition_and_conjunction(self):
        doc = analyzer.parse(
                "The building and the office that you ate and consumed your roll with gusto at were new")
        self.assertEqual(doc[14]._.holmes.string_representation_of_children(),
                '1:pobj(U); 4:pobj')
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                '6:nsubj; 8:cc; 9:conj; 14:prep(U)')
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '6:nsubj(U); 11:dobj; 12:prep; 14:prep')

    def test_simple_whose_clause(self):
        doc = analyzer.parse("The dog whose owner I met was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(), '1:poss')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_whose_clause_with_conjunction_of_possessor(self):
        doc = analyzer.parse("The dog whose owner and friend I met was tired")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '1:poss; 4:cc; 5:conj')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '1:poss')
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(), '-2:None')

    def test_whose_clause_with_conjunction_of_possessed(self):
        doc = analyzer.parse("The lion and dog whose owner I met were tired")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '1:poss(U); 3:poss')
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(), '-4:None')

    def test_phrasal_verb(self):
        doc = analyzer.parse("He took out insurance")
        self.assertEqual(doc[1]._.holmes.lemma, 'take out')
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(), '0:nsubj; 3:dobj')

    def test_positive_modal_verb(self):
        doc = analyzer.parse("He should do it")
        self.assertEqual(doc[2]._.holmes.string_representation_of_children(),
                '0:nsubj(U); 1:aux; 3:dobj(U)')

    def test_negative_modal_verb(self):
        doc = analyzer.parse("He cannot do it")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '0:nsubj(U); 1:aux; 2:neg(U); 4:dobj(U)')
        self.assertTrue(doc[3]._.holmes.is_negated)

    def test_ought_to(self):
        doc = analyzer.parse("He ought to do it")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '0:nsubj(U); 2:aux; 4:dobj')

    def test_phrasal_verb(self):
        doc = analyzer.parse("He will have been doing it")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '0:nsubj; 1:aux; 2:aux; 3:aux; 5:dobj')

    def test_pobjb(self):
        doc = analyzer.parse("Eating by employees")
        self.assertEqual(doc[0]._.holmes.string_representation_of_children(), '1:prep; 2:pobjb')

    def test_pobjb(self):
        doc = analyzer.parse("Eating of icecream")
        self.assertEqual(doc[0]._.holmes.string_representation_of_children(), '1:prep; 2:pobjo')

    def test_pobjt(self):
        doc = analyzer.parse("Travelling to Munich")
        self.assertEqual(doc[0]._.holmes.string_representation_of_children(), '1:prep; 2:pobjt')

    def test_dative_prepositional_phrase(self):
        doc = analyzer.parse("He gave it to the employee")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                '0:nsubj; 2:dobj; 3:dative; 5:dative')
        self.assertFalse(doc[3]._.holmes.is_matchable)

    def test_dative_prepositional_phrase_with_conjunction(self):
        doc = analyzer.parse("He gave it to the employee and the boss")
        self.assertEqual(doc[1]._.holmes.string_representation_of_children(),
                '0:nsubj; 2:dobj; 3:dative; 5:dative; 8:dative')
        self.assertFalse(doc[3]._.holmes.is_matchable)

    def test_simple_participle_phrase(self):
        doc = analyzer.parse("He talked about the cat chased by the dog")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '4:dobj; 6:agent; 8:pobjb')

    def test_participle_phrase_with_conjunction(self):
        doc = analyzer.parse(
                "He talked about the cat and the mouse chased by the dog and the tiger")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '4:dobj; 7:dobj; 9:agent; 11:pobjb; 14:pobjb')

    def test_subjective_modifying_adverbial_phrase(self):
        doc = analyzer.parse("The lion-chased cat came home")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '1:advmodsubj; 4:advmodobj')

    def test_subjective_modifying_adverbial_phrase_with_conjunction(self):
        doc = analyzer.parse("The lion-chased cat and mouse came home")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '1:advmodsubj; 4:advmodobj; 6:advmodobj(U)')

    def test_objective_modifying_adverbial_phrase(self):
        doc = analyzer.parse("The cat-chasing lion came home")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '1:advmodobj; 4:advmodsubj')

    def test_objective_modifying_adverbial_phrase_with_conjunction(self):
        doc = analyzer.parse("The cat-chasing lion and dog came home")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '1:advmodobj; 4:advmodsubj; 6:advmodsubj(U)')

    def test_verb_prepositional_complement_simple_active(self):
        doc = analyzer.parse("The dog was thinking about chasing a cat")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '1:nsubj(U); 7:dobj')

    def test_verb_prepositional_complement_with_conjunction_active(self):
        doc = analyzer.parse("The dog and the lion were thinking about chasing a cat and a mouse")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:nsubj(U); 10:dobj; 13:dobj')

    def test_verb_prepositional_complement_with_relative_clause_active(self):
        doc = analyzer.parse("The dog who was thinking about chasing a cat came home")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(), '1:nsubj(U); 8:dobj')

    def test_verb_preposition_complement_with_coreferring_pronoun_active(self):
        doc = analyzer.parse("He saw a dog. It was thinking about chasing a cat")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(), '5:nsubj(U); 11:dobj')

    def test_verb_preposition_complement_with_non_coreferring_pronoun_active(self):
        doc = analyzer.parse("It was thinking about chasing a cat")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(), '6:dobj')

    def test_adjective_prepositional_complement_simple_active(self):
        doc = analyzer.parse("The dog was worried about chasing a cat")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(), '1:nsubj(U); 7:dobj')

    def test_adjective_prepositional_complement_with_conjunction_active(self):
        doc = analyzer.parse("The dog and the lion were worried about chasing a cat and a mouse")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:nsubj(U); 10:dobj; 13:dobj')

    def test_adjective_prepositional_complement_with_relative_clause_active(self):
        doc = analyzer.parse("The dog who was worried about chasing a cat came home")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(), '1:nsubj(U); 8:dobj')

    def test_adjective_preposition_complement_with_coreferring_pronoun_active(self):
        doc = analyzer.parse("He saw a dog. He was worried about chasing a cat")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(), '5:nsubj(U); 11:dobj')

    def test_adjective_preposition_complement_with_non_coreferring_pronoun_active(self):
        doc = analyzer.parse("It was worried about chasing a cat")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(), '6:dobj')

    def test_verb_prepositional_complement_simple_passive(self):
        doc = analyzer.parse("The cat was thinking about being chased by a dog")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '1:nsubjpass(U); 5:auxpass; 7:agent; 9:pobjb')

    def test_verb_prepositional_complement_with_conjunction_passive(self):
        doc = analyzer.parse(
                "The cat and the mouse were thinking about being chased by a dog and a lion")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '1:nsubjpass(U); 4:nsubjpass(U); 8:auxpass; 10:agent; 12:pobjb; 15:pobjb')

    def test_verb_prepositional_complement_with_relative_clause_passive(self):
        doc = analyzer.parse("The cat who was thinking about being chased by a dog came home")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                '1:nsubjpass(U); 6:auxpass; 8:agent; 10:pobjb')

    def test_verb_preposition_complement_with_coreferring_pronoun_passive(self):
        doc = analyzer.parse("He saw a dog. It was thinking about being chased by a cat")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                '5:nsubjpass(U); 9:auxpass; 11:agent; 13:pobjb')

    def test_verb_preposition_complement_with_non_coreferring_pronoun_passive(self):
        doc = analyzer.parse("It was thinking about being chased by a cat")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '4:auxpass; 6:agent; 8:pobjb')

    def test_adjective_prepositional_complement_simple_passive(self):
        doc = analyzer.parse("The cat was worried about being chased by a dog")
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '1:nsubjpass(U); 5:auxpass; 7:agent; 9:pobjb')

    def test_adjective_prepositional_complement_with_conjunction_passive(self):
        doc = analyzer.parse(
                "The cat and the mouse were worried about being chased by a dog and a lion")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '1:nsubjpass(U); 4:nsubjpass(U); 8:auxpass; 10:agent; 12:pobjb; 15:pobjb')

    def test_adjective_prepositional_complement_with_relative_clause_passive(self):
        doc = analyzer.parse("The cat who was worried about being chased by a dog came home")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                '1:nsubjpass(U); 6:auxpass; 8:agent; 10:pobjb')

    def test_adjective_preposition_complement_with_coreferring_pronoun_passive(self):
        doc = analyzer.parse("He saw a dog. It was worried about being chased by a cat")
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                '5:nsubjpass(U); 9:auxpass; 11:agent; 13:pobjb')

    def test_adjective_preposition_complement_with_non_coreferring_pronoun_passive(self):
        doc = analyzer.parse("It was worried about being chased by a cat")
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '4:auxpass; 6:agent; 8:pobjb')

    def test_verb_prepositional_complement_with_conjunction_of_dependent_verb(self):
        doc = analyzer.parse("The cat and the mouse kept on singing and shouting")
        self.assertEqual(doc[7]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:nsubj(U); 8:cc; 9:conj')
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:nsubj(U)')

    def test_verb_p_c_with_conjunction_of_dependent_verb_and_coreferring_pronoun(self):
        doc = analyzer.parse("I saw a cat. It kept on singing and shouting")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '5:nsubj(U); 9:cc; 10:conj')
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                '5:nsubj(U)')

    def test_verb_p_c_with_conjunction_of_dependent_verb_and_non_coreferring_pronoun(self):
        doc = analyzer.parse("It kept on singing and shouting")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '4:cc; 5:conj')
        self.assertEqual(doc[5]._.holmes.string_representation_of_children(),
                '')

    def test_adjective_prepositional_complement_with_conjunction_of_dependent_verb(self):
        doc = analyzer.parse("The cat and the mouse were worried about singing and shouting")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:nsubj(U); 9:cc; 10:conj')
        self.assertEqual(doc[10]._.holmes.string_representation_of_children(),
                '1:nsubj(U); 4:nsubj(U)')

    def test_adjective_p_c_with_conjunction_of_dependent_verb_and_coreferring_pronoun(self):
        doc = analyzer.parse("I saw a cat. It was worried about singing and shouting")
        self.assertEqual(doc[9]._.holmes.string_representation_of_children(),
                '5:nsubj(U); 10:cc; 11:conj')
        self.assertEqual(doc[11]._.holmes.string_representation_of_children(),
                '5:nsubj(U)')

    def test_verb_p_c_with_conjunction_of_dependent_verb_and_non_coreferring_pronoun(self):
        doc = analyzer.parse("It was worried about singing and shouting")
        self.assertEqual(doc[4]._.holmes.string_representation_of_children(),
                '5:cc; 6:conj')
        self.assertEqual(doc[6]._.holmes.string_representation_of_children(),
                '')

    def test_single_preposition_dependency_added_to_noun(self):
        doc = analyzer.parse("The employee needs insurance for the next five years")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '4:prepposs(U)')

    def test_multiple_preposition_dependencies_added_to_noun(self):
        doc = analyzer.parse("The employee needs insurance for the next five years and in Europe")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '4:prepposs(U); 10:prepposs(U)')

    def test_single_preposition_dependency_added_to_coreferring_pronoun(self):
        doc = analyzer.parse(
                "We discussed the house. The employee needs it for the next five years")
        self.assertEqual(doc[8]._.holmes.string_representation_of_children(),
                '9:prepposs(U)')

    def test_single_preposition_dependency_not_added_to_non_coreferring_pronoun(self):
        doc = analyzer.parse("The employee needs it for the next five years")
        self.assertEqual(doc[3]._.holmes.string_representation_of_children(),
                '')

    def test_coreference_within_sentence(self):
        doc = analyzer.parse("The employee got home and he was surprised")
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[1]), [1,5])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[5]), [1,5])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[3]), [3])

    def test_coreference_between_sentences(self):
        doc = analyzer.parse("The employee got home. He was surprised")
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[1]), [1,5])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[5]), [1,5])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[3]), [3])

    def test_coreference_three_items_in_chain(self):
        doc = analyzer.parse("Richard was at work. He went home. He was surprised")
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[0]), [0,5,9])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[5]), [0,5,9])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[9]), [0,5,9])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[3]), [3])

    def test_coreference_conjunction_in_antecedent(self):
        doc = analyzer.parse("Richard and Carol came to work. They had a discussion")
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[0]), [0,7])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[2]), [2,7])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[7]), [0,2,7])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[3]), [3])

    def test_maximum_mentions_difference(self):
        doc = analyzer.parse("""Richard came to work. He was happy. He was happy. He was happy.
        He was happy. He was happy. He was happy. He was happy. He was happy.""")
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[0]), [0,5,9,13])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[5]), [0,5,9,13,18])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[9]), [0,5,9,13,18,22])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[13]), [0,5,9,13,18,22,26])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[18]),
                [5,9,13,18,22,26,30])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[22]),
                [9,13,18,22,26,30,34])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[26]), [13,18,22,26,30,34])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[30]), [18,22,26,30,34])
        self.assertEqual(analyzer.token_and_coreference_chain_indexes(doc[34]), [22,26,30,34])
