import copy
import sys
from spacy.tokens import Token
from .errors import DuplicateDocumentError, NoSearchPhraseError, NoDocumentError
from .parsing import Subword, Index
from word_matching.general import WordMatch



class Match:
    """A match between a search phrase and a document.

    Properties:

    word_matches -- a list of *WordMatch* objects.
    is_negated -- *True* if this match is negated.
    is_uncertain -- *True* if this match is uncertain.
    involves_coreference -- *True* if this match was found using coreference resolution.
    search_phrase_label -- the label of the search phrase that matched.
    search_phrase_text -- the text of the search phrase that matched.
    document_label -- the label of the document that matched.
    from_single_word_phraselet -- *True* if this is a match against a single-word
        phraselet.
    from_topic_match_phraselet_created_without_matching_tags -- **True** or **False**
    from_reverse_only_topic_match_phraselet -- **True** or **False**
    overall_similarity_measure -- the overall similarity of the match, or *1.0* if the embedding
        strategy was not involved in the match.
    index_within_document -- the index of the document token that matched the search phrase
        root token.
    """

    def __init__(
            self, search_phrase_label, search_phrase_text, document_label,
            from_single_word_phraselet, from_topic_match_phraselet_created_without_matching_tags,
            from_reverse_only_topic_match_phraselet):
        self.word_matches = []
        self.is_negated = False
        self.is_uncertain = False
        self.search_phrase_label = search_phrase_label
        self.search_phrase_text = search_phrase_text
        self.document_label = document_label
        self.from_single_word_phraselet = from_single_word_phraselet
        self.from_topic_match_phraselet_created_without_matching_tags = \
            from_topic_match_phraselet_created_without_matching_tags
        self.from_reverse_only_topic_match_phraselet = from_reverse_only_topic_match_phraselet
        self.index_within_document = None
        self.overall_similarity_measure = '1.0'

    @property
    def involves_coreference(self):
        for word_match in self.word_matches:
            if word_match.involves_coreference:
                return True
        return False

    def __copy__(self):
        match_to_return = Match(
            self.search_phrase_label, self.search_phrase_text,
            self.document_label, self.from_single_word_phraselet,
            self.from_topic_match_phraselet_created_without_matching_tags,
            self.from_reverse_only_topic_match_phraselet)
        match_to_return.word_matches = self.word_matches.copy()
        match_to_return.is_negated = self.is_negated
        match_to_return.is_uncertain = self.is_uncertain
        match_to_return.index_within_document = self.index_within_document
        match_to_return.overall_similarity_measure = self.overall_similarity_measure
        return match_to_return

    def get_subword_index(self):
        for word_match in self.word_matches:
            if word_match.search_phrase_token.dep_ == 'ROOT':
                if word_match.document_subword is None:
                    return None
                return word_match.document_subword.index
        raise RuntimeError('No word match with search phrase token with root dependency')

    def get_subword_index_for_sorting(self):
        # returns *-1* rather than *None* in the absence of a subword
        subword_index = self.get_subword_index()
        return subword_index if subword_index is not None else -1

class StructuralMatcher:
    """The class responsible for matching search phrases with documents."""

    def __init__(
            self, semantic_matching_helper,
            embedding_based_matching_on_root_words, analyze_derivational_morphology,
            perform_coreference_resolution, use_reverse_dependency_matching,
            entity_label_to_vector_dict):
        """Args:

        semantic_matching_helper -- the *SemanticMatchingHelper* object to use
        embedding_based_matching_on_root_words -- *True* if embedding-based matching should be
            attempted on search-phrase root tokens
        analyze_derivational_morphology -- *True* if matching should be attempted between different
            words from the same word family. Defaults to *True*.
        perform_coreference_resolution -- *True* if coreference resolution should be taken into
            account when matching.
        use_reverse_dependency_matching -- *True* if appropriate dependencies in documents can be
            matched to dependencies in search phrases where the two dependencies point in opposite
            directions.
        entity_label_to_vector_dict -- a dictionary from entity labels to vectors generated from
            words that mean roughly the same as the label. """
        self.semantic_matching_helper = semantic_matching_helper
        self.embedding_based_matching_on_root_words = embedding_based_matching_on_root_words
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.perform_coreference_resolution = perform_coreference_resolution
        self.use_reverse_dependency_matching = use_reverse_dependency_matching
        self.entity_label_to_vector_dict = entity_label_to_vector_dict

    def match_type(self, *match_types):
        """ Selects the most salient match type out of a list of relevant match types. """

        if 'derivation' in match_types:
            return 'derivation'
        else:
            return 'direct'

    def match_recursively(
            self, *, search_phrase, search_phrase_token, document, document_token,
            document_subword_index, search_phrase_tokens_to_word_matches,
            search_phrase_and_document_visited_table, is_uncertain,
            structurally_matched_document_token, compare_embeddings_on_non_root_words,
            process_initial_question_words, overall_similarity_threshold,
            initial_question_word_overall_similarity_threshold):
        """Called whenever matching is attempted between a search phrase token and a document
            token."""

        def handle_match(
                search_phrase_word, document_word, match_type, depth,
                *, similarity_measure=1.0, first_document_token=document_token,
                last_document_token=document_token, search_phrase_initial_question_word=False):
            """Most of the variables are set from the outer call.

            Args:

            search_phrase_word -- the textual representation of the search phrase word that matched.
            document_word -- the textual representation of the document word that matched.
            match_type -- *direct*, *derivation*, *entity*, *embedding*, *entity_embedding*,
            or *question*
            similarity_measure -- the similarity between the two tokens. Defaults to 1.0 if the
                match did not involve embeddings.
            search_phrase_initial_question_word -- *True* if *search_phrase_word* is an initial
                question word or governs an initial question word.
            """
            for dependency in (
                    dependency for dependency in search_phrase_token._.holmes.children
                    if dependency.child_token(search_phrase_token.doc)._.holmes.is_matchable or
                    (process_initial_question_words and
                    dependency.child_token(
                    search_phrase_token.doc)._.holmes.is_initial_question_word)):
                at_least_one_document_dependency_tried = False
                at_least_one_document_dependency_matched = False
                # Loop through this token and any tokens linked to it by coreference
                parents = [Index(document_token.i, document_subword_index)]
                if self.perform_coreference_resolution and (document_subword_index is None or
                        document_token._.holmes.subwords[document_subword_index].is_head):
                    parents.extend([
                        Index(token_index, None) for token_index in
                        document_token._.holmes.token_and_coreference_chain_indexes
                        if token_index != document_token.i])
                for working_document_parent_index in parents:
                    working_document_child_indexes = []
                    document_parent_token = document_token.doc[
                        working_document_parent_index.token_index]
                    if not working_document_parent_index.is_subword() or \
                            document_parent_token._.holmes.subwords[
                            working_document_parent_index.subword_index].is_head:
                            # is_head: e.g. 'Polizeiinformation über Kriminelle' should match
                            # 'Information über Kriminelle'

                        # inverse_polarity_boolean: *True* in the special case where the
                        # dependency has been matched backwards
                        document_dependencies_to_inverse_polarity_booleans = {
                            document_dependency: False for document_dependency in
                            document_parent_token._.holmes.children if
                            self.semantic_matching_helper.dependency_labels_match(
                            search_phrase_dependency_label=dependency.label,
                            document_dependency_label=document_dependency.label,
                            inverse_polarity=False)}
                        document_dependencies_to_inverse_polarity_booleans.update({
                            document_dependency: True for document_dependency in
                            document_parent_token._.holmes.parents if
                            self.use_reverse_dependency_matching and
                            self.semantic_matching_helper.dependency_labels_match(
                            search_phrase_dependency_label=dependency.label,
                            document_dependency_label=document_dependency.label,
                            inverse_polarity=True)})
                        for document_dependency, inverse_polarity in \
                                document_dependencies_to_inverse_polarity_booleans.items():
                            if not inverse_polarity:
                                document_child = document_dependency.child_token(document_token.doc)
                            else:
                                document_child = \
                                    document_dependency.parent_token(document_token.doc)
                            if self.perform_coreference_resolution:
                                # wherever a dependency is found, loop through any tokens linked
                                # to the child by coreference
                                working_document_child_indexes = [
                                    Index(token_index, None) for token_index in
                                    document_child._.holmes.token_and_coreference_chain_indexes
                                    if document_token.doc[token_index].pos_ != 'PRON' or not
                                    document_token.doc[token_index]._.holmes.\
                                    is_involved_in_coreference()]
                                        # otherwise where matching starts with a noun and there is
                                        # a dependency pointing back to the noun, matching will be
                                        # attempted against the pronoun only and will then fail.
                            elif not inverse_polarity:
                                working_document_child_indexes = \
                                    [Index(document_dependency.child_index, None)]
                            else:
                                working_document_child_indexes = \
                                    [Index(document_dependency.parent_index, None)]
                            # Where a dependency points to an entire word that has subwords, check
                            # the head subword as well as the entire word
                            for working_document_child_index in \
                                    working_document_child_indexes.copy():
                                working_document_child = \
                                    document_token.doc[working_document_child_index.token_index]
                                for subword in (
                                        subword for subword in
                                        working_document_child._.holmes.subwords
                                        if subword.is_head):
                                    working_document_child_indexes.append(Index(
                                        working_document_child.i, subword.index))
                            # Loop through the dependencies from each token
                            for working_document_child_index in (
                                    working_index for working_index
                                    in working_document_child_indexes):
                                at_least_one_document_dependency_tried = True
                                if search_phrase.question_phraselet and \
                                        document[
                                        working_document_parent_index.token_index] in \
                                        self.semantic_matching_helper.\
                                        get_subtree_list_for_question_answer(
                                        document[
                                        working_document_child_index.token_index]):
                                    continue
                                if working_document_child_index in \
                                        search_phrase_and_document_visited_table[
                                        dependency.child_index] or \
                                        self.match_recursively(
                                        search_phrase=search_phrase,
                                        search_phrase_token=dependency.child_token(
                                            search_phrase_token.doc),
                                        document=document,
                                        document_token=document[
                                            working_document_child_index.token_index],
                                        document_subword_index=
                                        working_document_child_index.subword_index,
                                        search_phrase_tokens_to_word_matches=
                                        search_phrase_tokens_to_word_matches,
                                        search_phrase_and_document_visited_table=
                                        search_phrase_and_document_visited_table,
                                        is_uncertain=(
                                            (document_dependency.is_uncertain and not
                                            dependency.is_uncertain) or inverse_polarity),
                                        structurally_matched_document_token=document_child,
                                        compare_embeddings_on_non_root_words=
                                        compare_embeddings_on_non_root_words,
                                        process_initial_question_words=
                                        process_initial_question_words,
                                        overall_similarity_threshold=
                                        overall_similarity_threshold,
                                        initial_question_word_overall_similarity_threshold=
                                        initial_question_word_overall_similarity_threshold):
                                    at_least_one_document_dependency_matched = True
                    if working_document_parent_index.is_subword():
                        # examine relationship to dependent subword in the same word
                        document_parent_subword = document_token.doc[
                            working_document_parent_index.token_index]._.holmes.\
                            subwords[working_document_parent_index.subword_index]
                        if document_parent_subword.dependent_index is not None and \
                                self.semantic_matching_helper.dependency_labels_match(
                                    search_phrase_dependency_label=dependency.label,
                                    document_dependency_label=
                                    document_parent_subword.dependency_label,
                                    inverse_polarity=False):
                            at_least_one_document_dependency_tried = True
                            if self.match_recursively(
                                    search_phrase=search_phrase,
                                    search_phrase_token=dependency.child_token(
                                        search_phrase_token.doc),
                                    document=document,
                                    document_token=document_token,
                                    document_subword_index=
                                    document_parent_subword.dependent_index,
                                    search_phrase_tokens_to_word_matches=
                                    search_phrase_tokens_to_word_matches,
                                    search_phrase_and_document_visited_table=
                                    search_phrase_and_document_visited_table,
                                    is_uncertain=False,
                                    structurally_matched_document_token=document_token,
                                    compare_embeddings_on_non_root_words=
                                    compare_embeddings_on_non_root_words,
                                    process_initial_question_words=
                                    process_initial_question_words,
                                    overall_similarity_threshold=
                                    overall_similarity_threshold,
                                    initial_question_word_overall_similarity_threshold=
                                    initial_question_word_overall_similarity_threshold):
                                at_least_one_document_dependency_matched = True
                        # examine relationship to governing subword in the same word
                        document_child_subword = document_token.doc[
                            working_document_parent_index.token_index]._.holmes.\
                            subwords[working_document_parent_index.subword_index]
                        if document_child_subword.governor_index is not None and \
                                self.use_reverse_dependency_matching and \
                                self.semantic_matching_helper.dependency_labels_match(
                                    search_phrase_dependency_label=dependency.label,
                                    document_dependency_label=
                                    document_parent_subword.governing_dependency_label,
                                    inverse_polarity=True):
                            at_least_one_document_dependency_tried = True
                            if self.match_recursively(
                                    search_phrase=search_phrase,
                                    search_phrase_token=dependency.child_token(
                                        search_phrase_token.doc),
                                    document=document,
                                    document_token=document_token,
                                    document_subword_index=
                                    document_parent_subword.governor_index,
                                    search_phrase_tokens_to_word_matches=
                                    search_phrase_tokens_to_word_matches,
                                    search_phrase_and_document_visited_table=
                                    search_phrase_and_document_visited_table,
                                    is_uncertain=False,
                                    structurally_matched_document_token=document_token,
                                    compare_embeddings_on_non_root_words=
                                    compare_embeddings_on_non_root_words,
                                    process_initial_question_words=
                                    process_initial_question_words,
                                    overall_similarity_threshold=
                                    overall_similarity_threshold,
                                    initial_question_word_overall_similarity_threshold=
                                    initial_question_word_overall_similarity_threshold):
                                at_least_one_document_dependency_matched = True
                if at_least_one_document_dependency_tried and not \
                        at_least_one_document_dependency_matched:
                        # it is already clear that the search phrase has not matched, so
                        # there is no point in pursuing things any further
                    return
            # store the word match
            if document_subword_index is None:
                document_subword = None
            else:
                document_subword = document_token._.holmes.subwords[document_subword_index]
            search_phrase_tokens_to_word_matches[search_phrase_token.i].append(WordMatch(
                search_phrase_token, search_phrase_word, document_token,
                first_document_token, last_document_token, document_subword,
                document_word, match_type, similarity_measure, is_negated, is_uncertain,
                structurally_matched_document_token, document_word, depth,
                search_phrase_initial_question_word))

        def loop_search_phrase_word_representations():
            yield search_phrase_token._.holmes.lemma, 'direct'
            hyphen_normalized_word = self.semantic_matching_helper.normalize_hyphens(
                search_phrase_token._.holmes.lemma)
            if hyphen_normalized_word != search_phrase_token._.holmes.lemma:
                yield hyphen_normalized_word, 'direct'
            if self.analyze_derivational_morphology and \
                    search_phrase_token._.holmes.derived_lemma is not None:
                yield search_phrase_token._.holmes.derived_lemma, 'derivation'
            if not search_phrase.topic_match_phraselet and \
                    search_phrase_token._.holmes.lemma == search_phrase_token.lemma_ and \
                    search_phrase_token._.holmes.lemma != search_phrase_token.text:
                # search phrase word is not multiword, phrasal or separable verb, so we can match
                # against its text as well as its lemma
                yield search_phrase_token.text, 'direct'

        def document_word_representations():
            list_to_return = []
            if document_subword_index is not None:
                working_document_subword = document_token._.holmes.subwords[document_subword_index]
                list_to_return.append((
                    working_document_subword.text, 'direct'))
                hyphen_normalized_word = self.semantic_matching_helper.normalize_hyphens(
                    working_document_subword.text)
                if hyphen_normalized_word != working_document_subword.text:
                    list_to_return.append((
                        hyphen_normalized_word, 'direct'))
                if working_document_subword.lemma != working_document_subword.text:
                    list_to_return.append((
                        working_document_subword.lemma, 'direct'))
                if self.analyze_derivational_morphology and \
                        working_document_subword.derived_lemma is not None:
                    list_to_return.append((
                        working_document_subword.derived_lemma,
                        'derivation'))
            else:
                list_to_return.append((
                    document_token.text, 'direct'))
                hyphen_normalized_word = self.semantic_matching_helper.normalize_hyphens(
                    document_token.text)
                if hyphen_normalized_word != document_token.text:
                    list_to_return.append((
                        hyphen_normalized_word, 'direct'))
                if document_token._.holmes.lemma != document_token.text:
                    list_to_return.append((
                        document_token._.holmes.lemma, 'direct'))
                if self.analyze_derivational_morphology:
                    if document_token._.holmes.derived_lemma is not None:
                        list_to_return.append((
                            document_token._.holmes.derived_lemma,
                            'derivation'))
            return list_to_return

        def loop_document_multiword_representations(multiword_span):
            yield multiword_span.text, 'direct', multiword_span.derived_lemma
            hyphen_normalized_word = \
                self.semantic_matching_helper.normalize_hyphens(multiword_span.text)
            if hyphen_normalized_word != multiword_span.text:
                yield hyphen_normalized_word, 'direct', multiword_span.derived_lemma
            if multiword_span.text != multiword_span.lemma:
                yield multiword_span.lemma, 'direct', multiword_span.derived_lemma
            if multiword_span.derived_lemma != multiword_span.lemma:
                yield multiword_span.derived_lemma, 'derivation', multiword_span.derived_lemma

        index = Index(document_token.i, document_subword_index)
        search_phrase_and_document_visited_table[search_phrase_token.i].add(index)
        is_negated = document_token._.holmes.is_negated
        if document_token._.holmes.is_uncertain:
            is_uncertain = True

        search_phrase_initial_question_word = process_initial_question_words and \
            search_phrase_token._.holmes.has_initial_question_word_in_phrase
        if self.semantic_matching_helper.is_entity_search_phrase_token(
                search_phrase_token, search_phrase.topic_match_phraselet) and \
                document_subword_index is None:
            if self.semantic_matching_helper.entity_search_phrase_token_matches(
                    search_phrase_token, search_phrase.topic_match_phraselet, document_token):
                for multiword_span in \
                        self.semantic_matching_helper.multiword_spans_with_head_token(
                        document_token):
                    for working_token in multiword_span.tokens:
                        if not self.semantic_matching_helper.entity_search_phrase_token_matches(
                                search_phrase_token, search_phrase.topic_match_phraselet,
                                document_token):
                            continue
                    for working_token in multiword_span.tokens:
                        search_phrase_and_document_visited_table[search_phrase_token.i].add(
                            working_token.i)
                    handle_match(
                        search_phrase_token.text, multiword_span.text, 'entity', 0,
                        first_document_token=multiword_span.tokens[0],
                        last_document_token=multiword_span.tokens[-1],
                        search_phrase_initial_question_word=search_phrase_initial_question_word)
                    return True
                search_phrase_and_document_visited_table[search_phrase_token.i].add(
                    document_token.i)
                handle_match(search_phrase_token.text, document_token.text, 'entity', 0,
                    search_phrase_initial_question_word=
                    search_phrase_initial_question_word)
                return True
            return False

        document_word_representations = document_word_representations()
        for search_phrase_word_representation, search_phrase_match_type in \
                loop_search_phrase_word_representations():
            # multiword matches
            if document_subword_index is None:
                for multiword_span in \
                        self.semantic_matching_helper.multiword_spans_with_head_token(
                        document_token):
                    for multiword_span_representation, document_match_type, \
                            multispan_derived_lemma in \
                            loop_document_multiword_representations(multiword_span):
                        if search_phrase_word_representation.lower() == \
                                multiword_span_representation.lower():
                            for working_token in multiword_span.tokens:
                                search_phrase_and_document_visited_table[search_phrase_token.i].add(
                                    working_token.i)
                            handle_match(
                                search_phrase_token._.holmes.lemma,
                                multiword_span_representation,
                                self.match_type(
                                    search_phrase_match_type, document_match_type),
                                0, first_document_token=multiword_span.tokens[0],
                                last_document_token=multiword_span.tokens[-1],
                                search_phrase_initial_question_word=
                                search_phrase_initial_question_word)
                            return True
            for document_word_representation, document_match_type in \
                    document_word_representations:
                if search_phrase_word_representation.lower() == \
                        document_word_representation.lower():
                    handle_match(
                        search_phrase_word_representation, document_word_representation,
                        self.match_type(
                            search_phrase_match_type, document_match_type)
                        , 0,
                        search_phrase_initial_question_word=search_phrase_initial_question_word)
                    return True

        if document_subword_index is not None:
            document_word_to_use = document_token._.holmes.subwords[document_subword_index].lemma
            document_vector = document_token._.holmes.subwords[document_subword_index].vector if \
                self.embedding_matching_permitted(
                document_token._.holmes.subwords[document_subword_index]) else None
        else:
            document_word_to_use = document_token.lemma_
            document_vector = document_vector = document_token._.holmes.vector if \
                self.embedding_matching_permitted(document_token) else None

        if (overall_similarity_threshold < 1.0 or (search_phrase_initial_question_word and
                initial_question_word_overall_similarity_threshold < 1.0)) and (
                compare_embeddings_on_non_root_words or search_phrase.root_token.i ==
                search_phrase_token.i) and search_phrase_token.i in \
                search_phrase.matchable_non_entity_tokens_to_vectors.keys() and \
                self.embedding_matching_permitted(search_phrase_token):
            search_phrase_vector = search_phrase.matchable_non_entity_tokens_to_vectors[
                search_phrase_token.i]
            if document_subword_index is not None:
                if not self.embedding_matching_permitted(
                        document_token._.holmes.subwords[document_subword_index]):
                    return False
            else:
                if not self.embedding_matching_permitted(document_token):
                    return False
            single_token_similarity_threshold = \
                (initial_question_word_overall_similarity_threshold if
                search_phrase_initial_question_word else overall_similarity_threshold) ** len(
                search_phrase.matchable_non_entity_tokens_to_vectors)
            if search_phrase_vector is not None and document_vector is not None:
                similarity_measure = \
                    self.semantic_matching_helper.cosine_similarity(search_phrase_vector,
                    document_vector)
                if similarity_measure > single_token_similarity_threshold:
                    if not search_phrase.topic_match_phraselet and \
                            len(search_phrase_token._.holmes.lemma.split()) > 1:
                        search_phrase_word_to_use = search_phrase_token.lemma_
                    else:
                        search_phrase_word_to_use = search_phrase_token._.holmes.lemma
                    handle_match(
                        search_phrase_word_to_use, document_word_to_use, 'embedding', 0,
                        similarity_measure=similarity_measure,
                        search_phrase_initial_question_word=search_phrase_initial_question_word)
                    return True
            if document_token.ent_type_ != '':
                cosine_similarity = self.semantic_matching_helper.token_matches_ent_type(
                    search_phrase_vector, self.entity_label_to_vector_dict,
                    (document_token.ent_type_,), single_token_similarity_threshold)
                if cosine_similarity > 0:
                    for multiword_span in \
                            self.semantic_matching_helper.multiword_spans_with_head_token(
                            document_token):
                        for working_token in multiword_span.tokens:
                            if not working_token.ent_type == document_token.ent_type:
                                continue
                        for working_token in multiword_span.tokens:
                            search_phrase_and_document_visited_table[search_phrase_token.i].add(
                                working_token.i)
                        handle_match(search_phrase_token.text, document_token.text,
                        'entity_embedding', 0, similarity_measure=cosine_similarity,
                        first_document_token=multiword_span.tokens[0],
                        last_document_token=multiword_span.tokens[-1],
                        search_phrase_initial_question_word=search_phrase_initial_question_word)
                        return True
                    handle_match(search_phrase_token.text, document_token.text, 'entity_embedding',
                        0, similarity_measure=cosine_similarity,
                        search_phrase_initial_question_word=search_phrase_initial_question_word)
                    return True

        if process_initial_question_words and search_phrase_token._.holmes.is_initial_question_word:
            if document_vector is not None:
                question_word_matches = self.semantic_matching_helper.question_word_matches(
                    search_phrase.label, search_phrase_token, document_token, document_vector,
                    self.entity_label_to_vector_dict,
                    initial_question_word_overall_similarity_threshold ** 2)
            else:
                question_word_matches = self.semantic_matching_helper.question_word_matches(
                    search_phrase.label, search_phrase_token, document_token, None, None, None)
            if question_word_matches:
                first_document_token_index = last_document_token_index = document_token.i
                if document_token.pos_ in self.semantic_matching_helper.noun_pos and \
                        len(document_token.ent_type_) > 0:
                    while first_document_token_index >= 1:
                        if document_token.doc[first_document_token_index - 1].pos_ in \
                                self.semantic_matching_helper.noun_pos:
                            first_document_token_index = first_document_token_index - 1
                        else:
                            break
                    while last_document_token_index + 1 < len(document_token.doc):
                        if document_token.doc[last_document_token_index + 1].pos_ in \
                                self.semantic_matching_helper.noun_pos:
                            last_document_token_index = last_document_token_index + 1
                        else:
                            break
                handle_match(search_phrase_token._.holmes.lemma, document_word_to_use, 'question',
                    0, first_document_token=document_token.doc[first_document_token_index],
                    last_document_token=document_token.doc[last_document_token_index],
                    search_phrase_initial_question_word=True)
                return True
        return False

    def embedding_matching_permitted(self, obj):
        """ Embedding matching is suppressed for some parts of speech as well as for very short
            words. """
        if isinstance(obj, Token):
            if len(obj._.holmes.lemma.split()) > 1:
                working_lemma = obj.lemma_
            else:
                working_lemma = obj._.holmes.lemma
            return obj.pos_ in self.semantic_matching_helper.permissible_embedding_pos and \
                len(working_lemma) >= \
                self.semantic_matching_helper.minimum_embedding_match_word_length
        elif isinstance(obj, Subword):
            return len(obj.lemma) >= \
                self.semantic_matching_helper.minimum_embedding_match_word_length
        else:
            raise RuntimeError("'obj' must be either a Token or a Subword")

    def build_matches(
            self, *, search_phrase, search_phrase_tokens_to_word_matches, document_label,
            overall_similarity_threshold, initial_question_word_overall_similarity_threshold):
        """Investigate possible matches when recursion is complete."""

        def mention_root_or_token_index(token):
            if len(token._.coref_chains) == 0:
                return token.i
            for mention in (m for m in token._.coref_chains[0].mentions if token.i in
                    m.token_indexes):
                return mention.root_index

        def filter_word_matches_based_on_coreference_resolution(word_matches):
            """ When coreference resolution is active, additional matches are sometimes
                returned that are filtered out again using this method.
            """
            structural_indexes_to_word_matches = {}
            # Find the structurally matching document tokens for this list of word matches
            for word_match in word_matches:
                structural_index = \
                    mention_root_or_token_index(word_match.structurally_matched_document_token)
                if structural_index in structural_indexes_to_word_matches.keys():
                    structural_indexes_to_word_matches[structural_index].append(word_match)
                else:
                    structural_indexes_to_word_matches[structural_index] = [word_match]
            new_word_matches = []
            for structural_index in structural_indexes_to_word_matches:
                # For each structural token, find the best matching coreference mention
                relevant_word_matches = structural_indexes_to_word_matches[structural_index]
                structurally_matched_document_token = \
                    relevant_word_matches[0].document_token.doc[structural_index]
                already_added_document_token_indexes = set()
                if structurally_matched_document_token._.holmes.is_involved_in_coreference():
                    working_index = -1
                    for relevant_word_match in relevant_word_matches:
                        this_index = mention_root_or_token_index(relevant_word_match.document_token)
                        # The best mention should be as close to the structural
                        # index as possible; if they are the same distance, the preceding mention
                        # wins.
                        if working_index == -1 or (
                                abs(structural_index - this_index) <
                                abs(structural_index - working_index)) or \
                                ((abs(structural_index - this_index) ==
                                abs(structural_index - working_index)) and
                                this_index < working_index):
                            working_index = this_index
                    # Filter out any matches from mentions other than the best mention
                    for relevant_word_match in relevant_word_matches:
                        if working_index == \
                                mention_root_or_token_index(relevant_word_match.document_token) \
                                and relevant_word_match.document_token.i not in \
                                already_added_document_token_indexes:
                            already_added_document_token_indexes.add(
                                relevant_word_match.document_token.i)
                            new_word_matches.append(relevant_word_match)
                else:
                    new_word_matches.extend(relevant_word_matches)
            return new_word_matches

        def revise_extracted_words_based_on_coreference_resolution(word_matches):
            """ When coreference resolution is active, there may be a more specific piece of
            information elsewhere in the coreference chain of a token that has been matched, in
            which case this piece of information should be recorded in *word_match.extracted_word*.
            """

            for word_match in (
                    word_match for word_match in word_matches
                    if word_match.word_match_type in ('direct', 'derivation')
                    and word_match.document_subword is None and
                    word_match.document_token._.holmes.most_specific_coreferring_term_index
                    is not None):
                most_specific_document_token = word_match.document_token.doc[
                    word_match.document_token._.holmes.most_specific_coreferring_term_index]
                if word_match.document_token._.holmes.lemma != \
                        most_specific_document_token._.holmes.lemma:
                    for multiword_span in \
                            self.semantic_matching_helper.multiword_spans_with_head_token(
                            word_match.document_token.doc[
                            word_match.document_token._.holmes.
                            most_specific_coreferring_term_index]):
                        word_match.extracted_word = multiword_span.text
                        break
                    else:
                        word_match.extracted_word = most_specific_document_token.text

            return word_matches

        def match_already_contains_structurally_matched_document_token(
                match, document_token, document_subword_index):
            """Ensure that the same document token or subword does not match multiple search phrase
                tokens.
            """
            for word_match in match.word_matches:
                if document_token.i == word_match.structurally_matched_document_token.i:
                    if word_match.document_subword is not None and document_subword_index == \
                            word_match.document_subword.index:
                        return True
                    if word_match.document_subword is None and document_subword_index is None:
                        return True
            return False

        def check_document_tokens_are_linked_by_dependency(
                parent_token, parent_subword, child_token, child_subword):
            """ The recursive nature of the main matching algorithm can mean that all the tokens
                in the search phrase have matched but that two of them are linked by a
                search-phrase dependency that is absent from the document, which invalidates the
                match.
            """
            if parent_subword is not None:
                if child_subword is not None and parent_subword.dependent_index == \
                        child_subword.index and parent_token.i == child_token.i:
                    return True
                elif parent_subword.is_head and (child_subword is None or (
                        child_subword.is_head and parent_subword.containing_token_index !=
                        child_subword.containing_token_index)):
                    return True
                else:
                    return False
            if child_subword is not None and not child_subword.is_head:
                return False
            if self.perform_coreference_resolution and (parent_subword is None
                    or parent_subword.is_head):
                parents = parent_token._.holmes.token_and_coreference_chain_indexes
                children = child_token._.holmes.token_and_coreference_chain_indexes
            else:
                parents = [parent_token.i]
                children = [child_token.i]
            for parent in parents:
                for child in children:
                    if parent_token.doc[parent]._.holmes.has_dependency_with_child_index(child):
                        return True
                    if child_token.doc[child]._.holmes.has_dependency_with_child_index(parent):
                        return True
            return False

        def match_with_subwords_involves_all_containing_document_tokens(word_matches):
            """ Where a match involves subwords and the subwords are involved in conjunction,
                we need to make sure there are no tokens involved in the match merely because they
                supply subwords to another token, as this would lead to double matching. An example
                is search phrase 'Extraktion der Information' and document
                'Informationsextraktionsüberlegungen und -probleme'.
            """
            token_indexes = []
            containing_subword_token_indexes = []
            for word_match in word_matches:
                if word_match.document_subword is not None:
                    token_indexes.append(word_match.document_token.i)
                    containing_subword_token_indexes.append(
                        word_match.document_subword.containing_token_index)
            return len([
                token_index for token_index in token_indexes if not token_index in
                containing_subword_token_indexes]) == 0

        matches = [Match(
            search_phrase.label, search_phrase.doc_text, document_label,
            search_phrase.topic_match_phraselet and search_phrase.has_single_matchable_word,
            search_phrase.topic_match_phraselet_created_without_matching_tags,
            search_phrase.reverse_only)]
        for search_phrase_token in search_phrase.matchable_tokens:
            word_matches = search_phrase_tokens_to_word_matches[search_phrase_token.i]
            if len(word_matches) == 0:
                # if there is any search phrase token without a matching document token,
                # we have no match and can return
                return []
            if self.perform_coreference_resolution:
                word_matches = filter_word_matches_based_on_coreference_resolution(word_matches)
            # handle any conjunction by distributing the matches amongst separate match objects
            working_matches = []
            for word_match in word_matches:
                for match in matches:
                    working_match = copy.copy(match)
                    if word_match.document_subword is None:
                        subword_index = None
                    else:
                        subword_index = word_match.document_subword.index
                    if not match_already_contains_structurally_matched_document_token(
                            working_match, word_match.structurally_matched_document_token,
                            subword_index):
                        working_match.word_matches.append(word_match)
                        if word_match.is_negated:
                            working_match.is_negated = True
                        if word_match.is_uncertain:
                            working_match.is_uncertain = True
                        if search_phrase_token.i == search_phrase.root_token.i:
                            working_match.index_within_document = word_match.document_token.i
                        working_matches.append(working_match)
            matches = working_matches

        matches_to_return = []
        for match in matches:
            failed = False
            not_normalized_overall_similarity_measure = 1.0
            # now carry out the coherence check, if there are two or fewer word matches (which
            # is the case during topic matching) no check is necessary
            if len(match.word_matches) > 2:
                for parent_word_match in match.word_matches:
                    for search_phrase_dependency in \
                            parent_word_match.search_phrase_token._.holmes.children:
                        for child_word_match in (
                                cwm for cwm in match.word_matches if cwm.search_phrase_token.i ==
                                search_phrase_dependency.child_index):
                            if not check_document_tokens_are_linked_by_dependency(
                                    parent_word_match.document_token,
                                    parent_word_match.document_subword,
                                    child_word_match.document_token,
                                    child_word_match.document_subword) and \
                                not check_document_tokens_are_linked_by_dependency(
                                    child_word_match.document_token,
                                    child_word_match.document_subword,
                                    parent_word_match.document_token,
                                    parent_word_match.document_subword):
                                failed = True
                        if failed:
                            break
                    if failed:
                        break
                if failed:
                    continue

            if not match_with_subwords_involves_all_containing_document_tokens(match.word_matches):
                continue

            for word_match in match.word_matches:
                not_normalized_overall_similarity_measure *= word_match.similarity_measure
            if not_normalized_overall_similarity_measure < 1.0:
                overall_similarity_measure = \
                    round(not_normalized_overall_similarity_measure ** \
                    (1 / len(search_phrase.matchable_non_entity_tokens_to_vectors)), 8)
            else:
                overall_similarity_measure = 1.0
            if overall_similarity_measure == 1.0 or \
                    overall_similarity_measure >= overall_similarity_threshold or \
                    overall_similarity_measure >= \
                    initial_question_word_overall_similarity_threshold:
                match.overall_similarity_measure = str(
                    overall_similarity_measure)
                matches_to_return.append(match)
        return matches_to_return

    def get_matches_starting_at_root_word_match(
            self, search_phrase, document, document_token, document_subword_index, document_label,
            compare_embeddings_on_non_root_words, process_initial_question_words,
            overall_similarity_threshold,
            initial_question_word_overall_similarity_threshold):
        """Begin recursive matching where a search phrase root token has matched a document
            token.
        """
        matches_to_return = []
        # array of arrays where each entry corresponds to a search_phrase token and is itself an
        # array of WordMatch instances
        search_phrase_tokens_to_word_matches = [[] for token in search_phrase.doc]
        # array of sets to guard against endless looping during recursion. Each set
        # corresponds to the search phrase token with its index and contains the Index objects
        # for the document words for which a match to that search phrase token has been attempted.
        search_phrase_and_document_visited_table = [set() for token in search_phrase.doc]
        self.match_recursively(
            search_phrase=search_phrase,
            search_phrase_token=search_phrase.root_token,
            document=document,
            document_token=document_token,
            document_subword_index=document_subword_index,
            search_phrase_tokens_to_word_matches=search_phrase_tokens_to_word_matches,
            search_phrase_and_document_visited_table=search_phrase_and_document_visited_table,
            is_uncertain=document_token._.holmes.is_uncertain,
            structurally_matched_document_token=document_token,
            compare_embeddings_on_non_root_words=compare_embeddings_on_non_root_words,
            process_initial_question_words=process_initial_question_words,
            overall_similarity_threshold=overall_similarity_threshold,
            initial_question_word_overall_similarity_threshold=
            initial_question_word_overall_similarity_threshold)
        working_matches = self.build_matches(
            search_phrase=search_phrase,
            search_phrase_tokens_to_word_matches=search_phrase_tokens_to_word_matches,
            document_label=document_label,
            overall_similarity_threshold=overall_similarity_threshold,
            initial_question_word_overall_similarity_threshold=
            initial_question_word_overall_similarity_threshold)
        matches_to_return.extend(working_matches)
        return matches_to_return

    def match(
            self, *, document_labels_to_documents,
            corpus_index_dict,
            search_phrases,
            match_depending_on_single_words,
            compare_embeddings_on_root_words,
            compare_embeddings_on_non_root_words,
            reverse_matching_corpus_word_positions,
            embedding_reverse_matching_corpus_word_positions,
            process_initial_question_words,
            overall_similarity_threshold,
            initial_question_word_overall_similarity_threshold,
            document_label_filter=None):
        """Finds and returns matches between search phrases and documents.
        match_depending_on_single_words -- 'True' to match only single word search phrases,
            'False' to match only non-single-word search phrases and 'None' to match both.
        compare_embeddings_on_root_words -- if 'True', embeddings on root words are compared.
        compare_embeddings_on_non_root_words -- if 'True', embeddings on non-root words are
            compared.
        reverse_matching_corpus_word_positions -- corpus word positions for non-embedding
            reverse matching only.
        embedding_reverse_matching_corpus_word_positions -- corpus word positions for embedding
            and non-embedding reverse matching.
        process_initial_question_words -- 'True' if interrogative pronouns in search phrases should
            be matched to answering phrases in documents. Only used with topic matching.
        overall_similarity_threshold -- the overall similarity threshold for embedding-based
            matching.
        initial_question_word_overall_similarity_threshold -- the overall similarity threshold for
            embedding-based matching where the search phrase word has a dependent initial question
            word.
        document_label_filter -- a string with which the label of a document must begin for that
            document to be considered for matching, or 'None' if no filter is in use.
        """

        def filter_out(document_label):
            return document_label_filter is not None and document_label is not None and \
                not document_label.startswith(str(document_label_filter))

        if overall_similarity_threshold == 1.0 and \
                initial_question_word_overall_similarity_threshold == 1.0:
            compare_embeddings_on_root_words = False
            compare_embeddings_on_non_root_words = False
        match_specific_indexes = reverse_matching_corpus_word_positions is not None \
            or embedding_reverse_matching_corpus_word_positions is not None
        if reverse_matching_corpus_word_positions is None:
            reverse_matching_corpus_word_positions = set()
        if embedding_reverse_matching_corpus_word_positions is None:
            embedding_reverse_matching_corpus_word_positions = set()

        matches = []
        # Dictionary used to improve performance when embedding-based matching for root tokens
        # is active and there are multiple search phrases with the same root token word: the
        # same corpus word positions will then match all the search phrase root tokens.
        root_lexeme_to_cwps_to_match_dict = {}

        for search_phrase in search_phrases:
            if not search_phrase.has_single_matchable_word and match_depending_on_single_words:
                continue
            if search_phrase.has_single_matchable_word and \
                    match_depending_on_single_words is False:
                continue
            if not match_specific_indexes and (search_phrase.reverse_only or \
                    search_phrase.treat_as_reverse_only_during_initial_relation_matching):
                continue
            if search_phrase.has_single_matchable_word and \
                    not compare_embeddings_on_root_words and \
                    not self.semantic_matching_helper.is_entity_search_phrase_token(
                        search_phrase.root_token, search_phrase.topic_match_phraselet):
                # We are only matching a single word without embedding, so to improve
                # performance we avoid entering the subgraph matching code.
                search_phrase_token = [
                    token for token in search_phrase.doc if token._.holmes.is_matchable][0]
                existing_minimal_match_cwps = []
                for word_matching_root_token in search_phrase.words_matching_root_token:
                    if word_matching_root_token in corpus_index_dict:
                        search_phrase_match_type, depth = \
                                search_phrase.root_word_to_match_info_dict[
                                    word_matching_root_token]
                        for corpus_word_position, document_word_representation, \
                                document_match_type_is_derivation in \
                                corpus_index_dict[word_matching_root_token]:
                            if filter_out(corpus_word_position.document_label):
                                continue
                            if corpus_word_position in existing_minimal_match_cwps:
                                continue
                            document_label = corpus_word_position.document_label
                            index = corpus_word_position.index
                            doc = document_labels_to_documents[document_label]
                            if document_match_type_is_derivation:
                                document_match_type = 'derivation'
                            else:
                                document_match_type = 'direct'
                            match_type = self.match_type(
                                search_phrase_match_type, document_match_type)
                            minimal_match = Match(
                                search_phrase.label, search_phrase.doc_text, document_label,
                                True, search_phrase.
                                topic_match_phraselet_created_without_matching_tags,
                                search_phrase.reverse_only)
                            minimal_match.index_within_document = index.token_index
                            matched = False
                            if len(word_matching_root_token.split()) > 1:
                                for multiword_span in \
                                        self.semantic_matching_helper.\
                                        multiword_spans_with_head_token(
                                        doc[index.token_index]):
                                    for textual_representation, _ in \
                                            self.semantic_matching_helper.\
                                            loop_textual_representations(multiword_span):
                                        if textual_representation == \
                                                word_matching_root_token:
                                            matched = True
                                            minimal_match.word_matches.append(WordMatch(
                                                search_phrase_token,
                                                search_phrase_token._.holmes.lemma,
                                                doc[index.token_index],
                                                multiword_span.tokens[0],
                                                multiword_span.tokens[-1],
                                                None,
                                                document_word_representation,
                                                match_type,
                                                1.0, False, False, doc[index.token_index],
                                                document_word_representation, depth, False))
                                            break
                                    if matched:
                                        break
                            if not matched:
                                token = doc[index.token_index]
                                if index.is_subword():
                                    subword = token._.holmes.subwords[index.subword_index]
                                else:
                                    subword = None
                                minimal_match.word_matches.append(WordMatch(
                                    search_phrase_token,
                                    search_phrase_token._.holmes.lemma,
                                    token,
                                    token,
                                    token,
                                    subword,
                                    document_word_representation,
                                    match_type,
                                    1.0, token._.holmes.is_negated, False, token,
                                    document_word_representation, depth, False))
                                if token._.holmes.is_negated:
                                    minimal_match.is_negated = True
                            existing_minimal_match_cwps.append(corpus_word_position)
                            matches.append(minimal_match)
                continue
            direct_matching_corpus_word_positions = []
            if self.semantic_matching_helper.is_entitynoun_search_phrase_token(
                    search_phrase.root_token): # phraselets are not generated for
                                               # ENTITYNOUN roots, so not relevant to topic matching
                for document_label, doc in document_labels_to_documents.items():
                    for token in doc:
                        if token.pos_ in self.semantic_matching_helper.noun_pos:
                            matches.extend(
                                self.get_matches_starting_at_root_word_match(
                                    search_phrase, doc, token, None, document_label,
                                    compare_embeddings_on_non_root_words,
                                    process_initial_question_words,
                                    overall_similarity_threshold,
                                    initial_question_word_overall_similarity_threshold))
                continue
            matched_corpus_word_positions = set()
            if self.semantic_matching_helper.is_entity_search_phrase_token(
                    search_phrase.root_token, search_phrase.topic_match_phraselet):
                if search_phrase.topic_match_phraselet:
                    entity_label = search_phrase.root_token._.holmes.lemma
                else:
                    entity_label = search_phrase.root_token.text
                if entity_label in corpus_index_dict.keys():
                    entity_matching_corpus_word_positions = [
                        cwp for cwp, _, _ in corpus_index_dict[entity_label]]
                    if match_specific_indexes:
                        entity_matching_corpus_word_positions = [
                            cwp for cwp in entity_matching_corpus_word_positions
                            if cwp in reverse_matching_corpus_word_positions
                            or cwp in embedding_reverse_matching_corpus_word_positions
                            and not cwp.index.is_subword()]
                    matched_corpus_word_positions.update(
                        entity_matching_corpus_word_positions)
            else:
                for word_matching_root_token in search_phrase.words_matching_root_token:
                    if word_matching_root_token in corpus_index_dict.keys():
                        direct_matching_corpus_word_positions = [
                            cwp for cwp, _, _ in corpus_index_dict[
                                word_matching_root_token]]
                        if match_specific_indexes:
                            direct_matching_corpus_word_positions = [
                                cwp for cwp in direct_matching_corpus_word_positions
                                if cwp in reverse_matching_corpus_word_positions
                                or cwp in embedding_reverse_matching_corpus_word_positions]
                        matched_corpus_word_positions.update(
                            direct_matching_corpus_word_positions)
            if compare_embeddings_on_root_words and not \
                    self.semantic_matching_helper.is_entity_search_phrase_token(
                        search_phrase.root_token, search_phrase.topic_match_phraselet) \
                    and not search_phrase.reverse_only and \
                    self.embedding_matching_permitted(search_phrase.root_token):
                if not search_phrase.topic_match_phraselet and \
                        len(search_phrase.root_token._.holmes.lemma.split()) > 1:
                    root_token_lemma_to_use = search_phrase.root_token.lemma_
                else:
                    root_token_lemma_to_use = search_phrase.root_token._.holmes.lemma
                if root_token_lemma_to_use in root_lexeme_to_cwps_to_match_dict:
                    matched_corpus_word_positions.update(
                        root_lexeme_to_cwps_to_match_dict[root_token_lemma_to_use])
                else:
                    working_cwps_to_match_for_cache = set()
                    for document_word in corpus_index_dict:
                        corpus_word_positions_to_match = [
                            cwp for cwp, _, _ in corpus_index_dict[document_word]]
                        if match_specific_indexes:
                            corpus_word_positions_to_match = [
                                cwp for cwp in corpus_word_positions_to_match
                                if cwp in embedding_reverse_matching_corpus_word_positions
                                and cwp not in direct_matching_corpus_word_positions]
                            if len(corpus_word_positions_to_match) == 0:
                                continue
                        search_phrase_vector = \
                                search_phrase.matchable_non_entity_tokens_to_vectors[
                                    search_phrase.root_token.i]
                        example_cwp = corpus_word_positions_to_match[0]
                        example_doc = document_labels_to_documents[example_cwp.document_label]
                        example_index = example_cwp.index
                        example_document_token = example_doc[example_index.token_index]
                        if example_index.is_subword():
                            if not self.embedding_matching_permitted(
                                    example_document_token._.holmes.subwords[
                                    example_index.subword_index]):
                                continue
                            document_vector = example_document_token._.holmes.subwords[
                                example_index.subword_index].vector
                        else:
                            if not self.embedding_matching_permitted(example_document_token):
                                continue
                            document_vector = example_document_token._.holmes.vector
                        if search_phrase_vector is not None and document_vector is not None:
                            similarity_measure = \
                                self.semantic_matching_helper.cosine_similarity(
                                search_phrase_vector,
                                document_vector)
                            search_phrase_initial_question_word = process_initial_question_words \
                                and search_phrase.root_token._.holmes.\
                                has_initial_question_word_in_phrase
                            single_token_similarity_threshold = \
                                (initial_question_word_overall_similarity_threshold if
                                search_phrase_initial_question_word else
                                overall_similarity_threshold) ** len(
                                search_phrase.matchable_non_entity_tokens_to_vectors)
                            if similarity_measure >= single_token_similarity_threshold:
                                matched_corpus_word_positions.update(
                                    corpus_word_positions_to_match)
                                working_cwps_to_match_for_cache.update(
                                    corpus_word_positions_to_match)
                    root_lexeme_to_cwps_to_match_dict[root_token_lemma_to_use] = \
                        working_cwps_to_match_for_cache
            for corpus_word_position in matched_corpus_word_positions:
                if filter_out(corpus_word_position.document_label):
                    continue
                doc = document_labels_to_documents[corpus_word_position.document_label]
                matches.extend(self.get_matches_starting_at_root_word_match(
                    search_phrase, doc, doc[corpus_word_position.index.token_index],
                    corpus_word_position.index.subword_index, corpus_word_position.document_label,
                    compare_embeddings_on_non_root_words, process_initial_question_words,
                    overall_similarity_threshold,
                    initial_question_word_overall_similarity_threshold))
        return sorted(matches, key=lambda match: (1 - float(match.overall_similarity_measure),
            match.document_label, match.index_within_document))

    def build_match_dictionaries(self, matches):
        """Builds and returns a sorted list of match dictionaries."""
        match_dicts = []
        for match in matches:
            earliest_sentence_index = sys.maxsize
            latest_sentence_index = -1
            for word_match in match.word_matches:
                sentence_index = word_match.document_token.sent.start
                if sentence_index < earliest_sentence_index:
                    earliest_sentence_index = sentence_index
                if sentence_index > latest_sentence_index:
                    latest_sentence_index = sentence_index
            sentences_string = ' '.join(
                sentence.text.strip() for sentence in
                match.word_matches[0].document_token.doc.sents if sentence.start >=
                earliest_sentence_index and sentence.start <= latest_sentence_index)

            match_dict = {
                'search_phrase_label': match.search_phrase_label,
                'search_phrase_text': match.search_phrase_text,
                'document': match.document_label,
                'index_within_document': match.index_within_document,
                'sentences_within_document': sentences_string,
                'negated': match.is_negated,
                'uncertain': match.is_uncertain,
                'involves_coreference': match.involves_coreference,
                'overall_similarity_measure': match.overall_similarity_measure}
            text_word_matches = []
            for word_match in match.word_matches:
                text_word_matches.append({
                    'search_phrase_token_index': word_match.search_phrase_token.i,
                    'search_phrase_word': word_match.search_phrase_word,
                    'document_token_index': word_match.document_token.i,
                    'first_document_token_index': word_match.first_document_token.i,
                    'last_document_token_index': word_match.last_document_token.i,
                    'structurally_matched_document_token_index':
                        word_match.structurally_matched_document_token.i,
                    'document_subword_index':
                        word_match.document_subword.index
                        if word_match.document_subword is not None else None,
                    'document_subword_containing_token_index':
                        word_match.document_subword.containing_token_index
                        if word_match.document_subword is not None else None,
                    'document_word': word_match.document_word,
                    'document_phrase': self.semantic_matching_helper.get_dependent_phrase(
                        word_match.document_token, word_match.document_subword),
                    'match_type': word_match.word_match_type,
                    'negated': word_match.is_negated,
                    'uncertain': word_match.is_uncertain,
                    'similarity_measure': str(word_match.similarity_measure),
                    'involves_coreference': word_match.involves_coreference,
                    'extracted_word': word_match.extracted_word,
                    'depth': word_match.depth,
                    'explanation': word_match.explain()})
            match_dict['word_matches'] = text_word_matches
            match_dicts.append(match_dict)
        return match_dicts

    def sort_match_dictionaries(self, match_dictionaries):
        return sorted(match_dictionaries,
            key=lambda match_dict: (1 - float(match_dict['overall_similarity_measure']),
                match_dict['document']))
