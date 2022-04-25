from typing import List, Dict, Set
import copy
import sys
from spacy.tokens import Token
from .parsing import Index
from holmes_extractor.word_matching.general import WordMatch, WordMatchingStrategy



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
            perform_coreference_resolution, use_reverse_dependency_matching):
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
        process_initial_question_words -- *True* if initial question words should be processed.
        """
        self.semantic_matching_helper = semantic_matching_helper
        self.embedding_based_matching_on_root_words = embedding_based_matching_on_root_words
        self.analyze_derivational_morphology = analyze_derivational_morphology
        self.perform_coreference_resolution = perform_coreference_resolution
        self.use_reverse_dependency_matching = use_reverse_dependency_matching

    def match_recursively(
            self, *, word_matching_strategies, search_phrase, search_phrase_token, document, document_token,
            document_subword_index, search_phrase_tokens_to_word_matches,
            search_phrase_and_document_visited_table, is_uncertain,
            structurally_matched_document_token, compare_embeddings_on_non_root_words, process_initial_question_words):
        """Called whenever matching is attempted between a search phrase token and a document
            token."""
        index = Index(document_token.i, document_subword_index)
        search_phrase_and_document_visited_table[search_phrase_token.i].add(index)
        if document_subword_index is None:
            for word_matching_strategy in word_matching_strategies:
                if document_token._.holmes.multiword_spans is not None:
                    potential_word_match = word_matching_strategy.match_multiwords(search_phrase, search_phrase_token, document_token, document_token._.holmes.multiword_spans)
                    if potential_word_match is not None:
                        break
                potential_word_match = word_matching_strategy.match_token(search_phrase, search_phrase_token, document_token)
                if potential_word_match is not None:
                    break
            else:
                return None
        else:
            for word_matching_strategy in word_matching_strategies:
                potential_word_match = word_matching_strategy.match_subword(search_phrase, search_phrase_token, document_token, document_token._.holmes.subwords[document_subword_index])
                if potential_word_match is not None:
                    break
            else:
                return None

        if not search_phrase.has_single_matchable_word:
            for dependency in (
                    dependency for dependency in search_phrase_token._.holmes.children
                    if dependency.child_token(search_phrase_token.doc)._.holmes.is_matchable or
                    (search_phrase.topic_match_phraselet and process_initial_question_words and
                    dependency.child_token(
                    search_phrase_token.doc)._.holmes.is_initial_question_word)):
                at_least_one_document_dependency_tried = False
                at_least_one_document_dependency_matched = False
                # Loop through this token and any tokens linked to it by coreference
                working_document_parent_indexes = [Index(document_token.i, document_subword_index)]
                if self.perform_coreference_resolution and (document_subword_index is None or
                        document_token._.holmes.subwords[document_subword_index].is_head):
                    working_document_parent_indexes.extend([
                        Index(token_index, None) for token_index in
                        document_token._.holmes.token_and_coreference_chain_indexes
                        if token_index != document_token.i])
                    working_document_parent_indexes.sort(key= lambda index: (abs(index.token_index - document_token.i), index.token_index > document_token.i))
                matched_document_indexes_for_parent = []
                for working_document_parent_index in working_document_parent_indexes:
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
                            working_document_child_mentions = []
                            if self.perform_coreference_resolution:
                                # wherever a dependency is found, loop through any tokens linked
                                # to the child by coreference
                                working_document_child_mentions = []
                                for chain in document_child._.coref_chains:
                                    document_child_mention = [m for m in chain if document_child.i in m][0]
                                    if len(document_child_mention) > 1:
                                        continue
                                    working_document_child_mentions.extend(mention.token_indexes for mention in chain if document_token.doc[mention.root_index].pos_ != 'PRON' or not
                                    document_token.doc[mention.root_index]._.holmes.\
                                    is_involved_in_coreference())
                                working_document_child_mentions.sort(key= lambda mention: (abs(mention[0] - document_child.i), mention[0] > document_child.i))
                            if len(working_document_child_mentions) == 0:
                                working_document_child_mentions = [[document_child.i]]
                            # Where a dependency points to an entire word that has subwords, check
                            # the head subword as well as the entire word
                            for working_document_child_mention in working_document_child_mentions:
                                working_document_child_indexes = []
                                for working_document_child_token_index in \
                                        working_document_child_mention.copy():
                                    working_document_child_indexes.append(Index(working_document_child_token_index, None))
                                    working_document_child = \
                                        document_token.doc[working_document_child_token_index]
                                    for subword in (
                                            subword for subword in
                                            working_document_child._.holmes.subwords
                                            if subword.is_head):
                                        working_document_child_indexes.append(Index(
                                            working_document_child.i, subword.index))
                                # Loop through the dependencies from each token
                                at_least_one_match_within_mention = False
                                for working_document_child_index in working_document_child_indexes:
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
                                            dependency.child_index]:
                                        at_least_one_document_dependency_matched = True
                                        continue
                                    if working_document_child_index in matched_document_indexes_for_parent:
                                        continue
                                    child_word_match = self.match_recursively(
                                            word_matching_strategies=word_matching_strategies,
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
                                            process_initial_question_words=process_initial_question_words)
                                    if child_word_match is not None:
                                        at_least_one_document_dependency_matched = True
                                        at_least_one_match_within_mention = True
                                        matched_document_indexes_for_parent.append(working_document_child_index)
                                if at_least_one_match_within_mention:
                                    break        
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
                            if self.match_recursively(
                                    word_matching_strategies=word_matching_strategies,
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
                                    process_initial_question_words=process_initial_question_words) is not None:
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
                                    word_matching_strategies=word_matching_strategies,
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
                                    process_initial_question_words=process_initial_question_words) is not None:
                                at_least_one_document_dependency_matched = True
                if at_least_one_document_dependency_tried and not \
                        at_least_one_document_dependency_matched:
                        # it is already clear that the search phrase has not matched, so
                        # there is no point in pursuing things any further
                    return None
                        
        # store the word match

        potential_word_match.structurally_matched_document_token = structurally_matched_document_token
        potential_word_match.is_negated = document_token._.holmes.is_negated
        potential_word_match.is_uncertain = is_uncertain or document_token._.holmes.is_uncertain
        if potential_word_match.first_document_token.i != potential_word_match.last_document_token.i: # multiword
            for working_token_index in range(potential_word_match.first_document_token.i, potential_word_match.last_document_token.i + 1):
                search_phrase_and_document_visited_table[search_phrase_token.i].add(Index(working_token_index, None))
        search_phrase_tokens_to_word_matches[search_phrase_token.i].append(potential_word_match)
        return potential_word_match

    def build_matches(
            self, *, search_phrase, search_phrase_tokens_to_word_matches, document_label,
            overall_similarity_threshold, initial_question_word_overall_similarity_threshold):
        """Investigate possible matches when recursion is complete."""

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
            self, word_matching_strategies, search_phrase, document, document_token, document_subword_index, document_label,
            compare_embeddings_on_non_root_words, overall_similarity_threshold,
            initial_question_word_overall_similarity_threshold, process_initial_question_words):
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
        if self.match_recursively(
            word_matching_strategies=word_matching_strategies,
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
        ) is not None:
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
            self, *, word_matching_strategies, document_labels_to_documents,
            reverse_dict,
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
            if self.semantic_matching_helper.get_entity_placeholder(
                    search_phrase.root_token) == "ENTITYNOUN": 
                for document_label, doc in document_labels_to_documents.items():
                    for token in doc:
                        if token.pos_ in self.semantic_matching_helper.noun_pos:
                            matches.extend(
                                self.get_matches_starting_at_root_word_match(
                                    word_matching_strategies, search_phrase, doc, token, None, document_label,
                                    compare_embeddings_on_non_root_words,
                                    overall_similarity_threshold,
                                    initial_question_word_overall_similarity_threshold,
                                    process_initial_question_words))
                continue
            direct_matching_corpus_word_positions = []
            matched_corpus_word_positions = set()
            entity_label = self.semantic_matching_helper.get_entity_placeholder(search_phrase.root_token)
            if entity_label is not None:
                if entity_label in reverse_dict.keys():
                    entity_matching_corpus_word_positions = [
                        riv.corpus_word_position for riv in reverse_dict[entity_label]]
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
                    if word_matching_root_token in reverse_dict.keys():
                        direct_matching_corpus_word_positions = [
                            riv.corpus_word_position for riv in reverse_dict[
                                word_matching_root_token]]
                        if match_specific_indexes:
                            direct_matching_corpus_word_positions = [
                                cwp for cwp in direct_matching_corpus_word_positions
                                if cwp in reverse_matching_corpus_word_positions
                                or cwp in embedding_reverse_matching_corpus_word_positions]
                        matched_corpus_word_positions.update(
                            direct_matching_corpus_word_positions)
            if compare_embeddings_on_root_words and self.semantic_matching_helper.get_entity_placeholder(search_phrase.root_token) is None \
                    and not search_phrase.reverse_only and \
                    self.semantic_matching_helper.embedding_matching_permitted(search_phrase.root_token):
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
                    for document_word in reverse_dict:
                        corpus_word_positions_to_match = [
                            riv.corpus_word_position for riv in reverse_dict[document_word]]
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
                            if not self.semantic_matching_helper.embedding_matching_permitted(
                                    example_document_token._.holmes.subwords[
                                    example_index.subword_index]):
                                continue
                            document_vector = example_document_token._.holmes.subwords[
                                example_index.subword_index].vector
                        else:
                            if not self.semantic_matching_helper.embedding_matching_permitted(example_document_token):
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
                    word_matching_strategies, search_phrase, doc, doc[corpus_word_position.index.token_index],
                    corpus_word_position.index.subword_index, corpus_word_position.document_label,
                    compare_embeddings_on_non_root_words,
                    overall_similarity_threshold,
                    initial_question_word_overall_similarity_threshold,
                    process_initial_question_words))
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
                    'explanation': word_match.explanation})
            match_dict['word_matches'] = text_word_matches
            match_dicts.append(match_dict)
        return match_dicts

    def sort_match_dictionaries(self, match_dictionaries):
        return sorted(match_dictionaries,
            key=lambda match_dict: (1 - float(match_dict['overall_similarity_measure']),
                match_dict['document']))
