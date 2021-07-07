import uuid
import statistics
import jsonpickle
from .parsing import SemanticMatchingHelperFactory, Index
from .errors import WrongModelDeserializationError, FewerThanTwoClassificationsError, \
        DuplicateDocumentError, NoPhraseletsAfterFilteringError, \
        EmbeddingThresholdGreaterThanRelationThresholdError, \
        IncompatibleAnalyzeDerivationalMorphologyDeserializationError

class TopicMatch:
    """A topic match between some text and part of a document. Note that the end indexes refer
        to the token in question rather than to the following token.

    Properties:

    document_label -- the document label.
    index_within_document -- the index of the token within the document where 'score' was achieved.
    subword_index -- the index of the subword within the token within the document where 'score'
            was achieved, or *None* if the match involved the whole word.
    start_index -- the start index of the topic match within the document.
    end_index -- the end index of the topic match within the document.
    sentences_start_index -- the start index within the document of the sentence that contains
        'start_index'
    sentences_end_index -- the end index within the document of the sentence that contains
        'end_index'
    relative_start_index -- the start index of the topic match relative to 'sentences_start_index'
    relative_end_index -- the end index of the topic match relative to 'sentences_start_index'
    score -- the similarity score of the topic match
    text -- the text between 'sentences_start_index' and 'sentences_end_index'
    structural_matches -- a list of `Match` objects that were used to derive this object.
    """

    def __init__(
            self, document_label, index_within_document, subword_index, start_index, end_index,
            sentences_start_index, sentences_end_index, score, text, structural_matches):
        self.document_label = document_label
        self.index_within_document = index_within_document
        self.subword_index = subword_index
        self.start_index = start_index
        self.end_index = end_index
        self.sentences_start_index = sentences_start_index
        self.sentences_end_index = sentences_end_index
        self.score = score
        self.text = text
        self.structural_matches = structural_matches

    @property
    def relative_start_index(self):
        return self.start_index - self.sentences_start_index

    @property
    def relative_end_index(self):
        return self.end_index - self.sentences_start_index

class PhraseletActivationTracker:
    """ Tracks the activation for a specific phraselet - the most recent score
        and the position within the document at which that score was calculated.
    """
    def __init__(self, position, score):
        self.position = position
        self.score = score

class TopicMatcher:
    """A topic matcher object. See manager.py for details of the properties."""

    def __init__(
            self, *, semantic_matching_helper, structural_matcher, indexed_documents,
            embedding_based_matching_on_root_words, maximum_activation_distance, relation_score,
            reverse_only_relation_score, single_word_score, single_word_any_tag_score,
            different_match_cutoff_score, overlapping_relation_multiplier, embedding_penalty,
            ontology_penalty, maximum_number_of_single_word_matches_for_relation_matching,
            maximum_number_of_single_word_matches_for_embedding_matching,
            sideways_match_extent, only_one_result_per_document, number_of_results,
            document_label_filter):
        if maximum_number_of_single_word_matches_for_embedding_matching > \
                maximum_number_of_single_word_matches_for_relation_matching:
            raise EmbeddingThresholdGreaterThanRelationThresholdError(' '.join((
                'embedding',
                str(maximum_number_of_single_word_matches_for_embedding_matching),
                'relation',
                str(maximum_number_of_single_word_matches_for_relation_matching))))
        self.semantic_matching_helper = semantic_matching_helper
        self.structural_matcher = structural_matcher
        self.indexed_documents = indexed_documents
        self.embedding_based_matching_on_root_words = embedding_based_matching_on_root_words
        self._ontology = structural_matcher.ontology
        self.maximum_activation_distance = maximum_activation_distance
        self.relation_score = relation_score
        self.reverse_only_relation_score = reverse_only_relation_score
        self.single_word_score = single_word_score
        self.single_word_any_tag_score = single_word_any_tag_score
        self.different_match_cutoff_score = different_match_cutoff_score
        self.overlapping_relation_multiplier = overlapping_relation_multiplier
        self.embedding_penalty = embedding_penalty
        self.ontology_penalty = ontology_penalty
        self.maximum_number_of_single_word_matches_for_relation_matching = \
                maximum_number_of_single_word_matches_for_relation_matching
        self.maximum_number_of_single_word_matches_for_embedding_matching = \
                maximum_number_of_single_word_matches_for_embedding_matching
        self.sideways_match_extent = sideways_match_extent
        self.only_one_result_per_document = only_one_result_per_document
        self.number_of_results = number_of_results
        self.document_label_filter = document_label_filter
        self._words_to_phraselet_word_match_infos = {}

    def _get_word_match_from_match(self, match, parent):
        ## child if parent==False
        for word_match in match.word_matches:
            if parent and word_match.search_phrase_token.dep_ == 'ROOT':
                return word_match
            if not parent and word_match.search_phrase_token.dep_ != 'ROOT':
                return word_match
        raise RuntimeError(''.join(('Word match not found with parent==', str(parent))))

    def _add_to_dict_list(self, dictionary, key, value):
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]

    def _add_to_dict_set(self, dictionary, key, value):
        if not key in dictionary:
            dictionary[key] = set()
        dictionary[key].add(value)

    def add_phraselets_to_dict(
            self, doc, *, phraselet_labels_to_phraselet_infos,
            replace_with_hypernym_ancestors, match_all_words,
            ignore_relation_phraselets, include_reverse_only, stop_lemmas, stop_tags,
            reverse_only_parent_lemmas, words_to_corpus_frequencies, maximum_corpus_frequency):
        """ Creates topic matching phraselets extracted from a matching text.

        Properties:

        doc -- the Holmes-parsed document
        phraselet_labels_to_phraselet_infos -- a dictionary from labels to phraselet info objects
            that are used to generate phraselet search phrases.
        replace_with_hypernym_ancestors -- if 'True', all words present in the ontology
            are replaced with their most general (highest) ancestors.
        match_all_words -- if 'True', word phraselets are generated for all matchable words
            rather than just for words whose tags match the phraselet template; multiwords
            are not taken into account when processing single-word phraselets; and single-word
            phraselets are generated for subwords.
        ignore_relation_phraselets -- if 'True', only single-word phraselets are processed.
        include_reverse_only -- whether to generate phraselets that are only reverse-matched.
            Reverse matching is used in topic matching but not in supervised document
            classification.
        stop_lemmas -- lemmas that should prevent all types of phraselet production.
        stop_tags -- tags that should prevent all types of phraselet production.
        reverse_only_parent_lemmas -- lemma / part-of-speech combinations that, when present at
            the parent pole of a relation phraselet, should cause that phraselet to be
            reverse-matched.
        words_to_corpus_frequencies -- a dictionary from words to the number of times each
            word occurs in the indexed documents, or *None* if corpus frequencies are not
            being taken into account.
        maximum_corpus_frequency -- the maximum value within *words_to_corpus_frequencies*,
            or *None* if corpus frequencies are not being taken into account.
        """

        index_to_lemmas_cache = {}
        def get_lemmas_from_index(index):
            """ Returns the lemma and the derived lemma. Phraselets form a special case where
                the derived lemma is set even if it is identical to the lemma. This is necessary
                because the lemma may be set to a different value during the lifecycle of the
                object. The property getter in the SemanticDictionary class ensures that
                derived_lemma is None is always returned where the two strings are identical.
            """
            if index in index_to_lemmas_cache:
                return index_to_lemmas_cache[index]
            token = doc[index.token_index]
            if self.semantic_matching_helper.is_entity_search_phrase_token(token, False):
                # False in order to get text rather than lemma
                index_to_lemmas_cache[index] = token.text, token.text
                return token.text, token.text
                # keep the text, because the lemma will be lowercase
            if index.is_subword():
                lemma = token._.holmes.subwords[index.subword_index].lemma
                if self.analyze_derivational_morphology:
                    derived_lemma = token._.holmes.subwords[index.subword_index].\
                        lemma_or_derived_lemma()
                else:
                    derived_lemma = lemma
                if self.ontology is not None and self.analyze_derivational_morphology:
                    for reverse_derived_word in self.semantic_matching_helper.\
                            reverse_derived_lemmas_in_ontology(
                            token._.holmes.subwords[index.subword_index]):
                        derived_lemma = reverse_derived_word.lower()
                        break
            else:
                lemma = token._.holmes.lemma
                if self.analyze_derivational_morphology:
                    derived_lemma = token._.holmes.lemma_or_derived_lemma()
                else:
                    derived_lemma = lemma
                if self.ontology is not None and not self.ontology.contains(lemma):
                    if self.ontology.contains(token.text.lower()):
                        lemma = derived_lemma = token.text.lower()
                    # ontology contains text but not lemma, so return text
                if self.ontology is not None and self.analyze_derivational_morphology:
                    for reverse_derived_word in self.semantic_matching_helper.\
                            reverse_derived_lemmas_in_ontology(token):
                        derived_lemma = reverse_derived_word.lower()
                        break
                        # ontology contains a word pointing to the same derived lemma,
                        # so return that. Note that if there are several such words the same
                        # one will always be returned.
            index_to_lemmas_cache[index] = lemma, derived_lemma
            return lemma, derived_lemma

        def replace_lemmas_with_most_general_ancestor(lemma, derived_lemma):
            new_derived_lemma = self.ontology.get_most_general_hypernym_ancestor(
                derived_lemma).lower()
            if derived_lemma != new_derived_lemma:
                lemma = derived_lemma = new_derived_lemma
            return lemma, derived_lemma

        def lemma_replacement_indicated(existing_lemma, existing_pos, new_lemma, new_pos):
            if existing_lemma is None:
                return False
            if not existing_pos in self.semantic_matching_helper.preferred_phraselet_pos and \
                    new_pos in self.semantic_matching_helper.preferred_phraselet_pos:
                return True
            if existing_pos in self.semantic_matching_helper.preferred_phraselet_pos and \
                     new_pos not in self.semantic_matching_helper.preferred_phraselet_pos:
                return False
            return len(new_lemma) < len(existing_lemma)

        def add_new_phraselet_info(
                phraselet_label, phraselet_template, created_without_matching_tags,
                is_reverse_only_parent_lemma, parent_lemma, parent_derived_lemma, parent_pos,
                child_lemma, child_derived_lemma, child_pos):

            def get_frequency_factor_for_pole(parent): # pole is 'True' -> parent, 'False' -> child
                original_word_set = {parent_lemma, parent_derived_lemma} if parent else \
                    {child_lemma, child_derived_lemma}
                word_set_including_any_ontology = original_word_set.copy()
                if self.ontology is not None:
                    for word in original_word_set:
                        for word_matching, _ in \
                                self.ontology.get_words_matching_and_depths(word):
                            word_set_including_any_ontology.add(word_matching)
                frequencies = []
                for word in word_set_including_any_ontology:
                    if word in words_to_corpus_frequencies:
                        frequencies.append(float(words_to_corpus_frequencies[word]))
                if len(frequencies) == 0:
                    return 1.0
                average_frequency = max(0, (sum(frequencies) / float(len(frequencies))) - 1)
                if average_frequency <= 1:
                    return 1.0
                else:
                    return 1 - (math.log(average_frequency) / math.log(maximum_corpus_frequency))

            if words_to_corpus_frequencies is not None:
                frequency_factor = get_frequency_factor_for_pole(True)
                if child_lemma is not None:
                    frequency_factor *= get_frequency_factor_for_pole(False)
            else:
                frequency_factor = 1.0
            if phraselet_label not in phraselet_labels_to_phraselet_infos:
                phraselet_labels_to_phraselet_infos[phraselet_label] = PhraseletInfo(
                    phraselet_label, phraselet_template.label, parent_lemma,
                    parent_derived_lemma, parent_pos, child_lemma, child_derived_lemma,
                    child_pos, created_without_matching_tags,
                    is_reverse_only_parent_lemma, frequency_factor)
            else:
                existing_phraselet = phraselet_labels_to_phraselet_infos[phraselet_label]
                if lemma_replacement_indicated(
                        existing_phraselet.parent_lemma, existing_phraselet.parent_pos,
                        parent_lemma, parent_pos):
                    existing_phraselet.parent_lemma = parent_lemma
                    existing_phraselet.parent_pos = parent_pos
                if lemma_replacement_indicated(
                        existing_phraselet.child_lemma, existing_phraselet.child_pos, child_lemma,
                        child_pos):
                    existing_phraselet.child_lemma = child_lemma
                    existing_phraselet.child_pos = child_pos

        def process_single_word_phraselet_templates(
                token, subword_index, checking_tags, token_indexes_to_multiword_lemmas):
            for phraselet_template in (
                    phraselet_template for phraselet_template in
                    self.semantic_matching_helper.phraselet_templates if
                    phraselet_template.single_word() and (
                        token._.holmes.is_matchable or subword_index is not None)):
                        # see note below for explanation
                if (not checking_tags or token.tag_ in phraselet_template.parent_tags) and \
                        token.tag_ not in stop_tags:
                    phraselet_doc = phraselet_template.template_doc.copy()
                    if token.i in token_indexes_to_multiword_lemmas and not match_all_words:
                        lemma = derived_lemma = token_indexes_to_multiword_lemmas[token.i]
                    else:
                        lemma, derived_lemma = get_lemmas_from_index(Index(token.i, subword_index))
                    if self.ontology is not None and replace_with_hypernym_ancestors:
                        lemma, derived_lemma = replace_lemmas_with_most_general_ancestor(
                            lemma, derived_lemma)
                    phraselet_doc[phraselet_template.parent_index]._.holmes.lemma = lemma
                    phraselet_doc[phraselet_template.parent_index]._.holmes.derived_lemma = \
                        derived_lemma
                    phraselet_label = ''.join((phraselet_template.label, ': ', derived_lemma))
                    if derived_lemma not in stop_lemmas and derived_lemma != 'ENTITYNOUN':
                        # ENTITYNOUN has to be excluded as single word although it is still
                        # permitted as the child of a relation phraselet template
                        add_new_phraselet_info(
                            phraselet_label, phraselet_template, not checking_tags,
                            None, lemma, derived_lemma, token.pos_, None, None, None)

        def add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(index_list):
            # for each token in the list, find out whether it has subwords and if so add the
            # head subword to the list
            for index in index_list.copy():
                token = doc[index.token_index]
                for subword in (
                        subword for subword in token._.holmes.subwords if
                        subword.is_head and subword.containing_token_index == token.i):
                    index_list.append(Index(token.i, subword.index))
            # if one or more subwords do not belong to this token, it is a hyphenated word
            # within conjunction and the whole word should not be used to build relation phraselets.
                if len([
                        subword for subword in token._.holmes.subwords if
                        subword.containing_token_index != token.i]) > 0:
                    index_list.remove(index)

        self._redefine_multiwords_on_head_tokens(doc)
        token_indexes_to_multiword_lemmas = {}
        token_indexes_within_multiwords_to_ignore = []
        for token in (token for token in doc if len(token._.holmes.lemma.split()) == 1):
            entity_defined_multiword, indexes = \
                self.semantic_matching_helper.get_entity_defined_multiword(token)
            if entity_defined_multiword is not None:
                for index in indexes:
                    if index == token.i:
                        token_indexes_to_multiword_lemmas[token.i] = entity_defined_multiword
                    else:
                        token_indexes_within_multiwords_to_ignore.append(index)
        for token in doc:
            if token.i in token_indexes_within_multiwords_to_ignore:
                if match_all_words:
                    process_single_word_phraselet_templates(
                        token, None, False, token_indexes_to_multiword_lemmas)
                continue
            if len([
                    subword for subword in token._.holmes.subwords if
                    subword.containing_token_index != token.i]) == 0:
                # whole single words involved in subword conjunction should not be included as
                # these are partial words including hyphens.
                process_single_word_phraselet_templates(
                    token, None, not match_all_words, token_indexes_to_multiword_lemmas)
            if match_all_words:
                for subword in (
                        subword for subword in token._.holmes.subwords if
                        token.i == subword.containing_token_index):
                    process_single_word_phraselet_templates(
                        token, subword.index, False, token_indexes_to_multiword_lemmas)
            if ignore_relation_phraselets:
                continue
            if self.perform_coreference_resolution:
                parents = [
                    Index(token_index, None) for token_index in
                    token._.holmes.token_and_coreference_chain_indexes]
            else:
                parents = [Index(token.i, None)]
            add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(parents)
            for parent in parents:
                for dependency in (
                        dependency for dependency in doc[parent.token_index]._.holmes.children
                        if dependency.child_index not in token_indexes_within_multiwords_to_ignore):
                    if self.perform_coreference_resolution:
                        children = [
                            Index(token_index, None) for token_index in
                            dependency.child_token(doc)._.holmes.
                            token_and_coreference_chain_indexes]
                    else:
                        children = [Index(dependency.child_token(doc).i, None)]
                    add_head_subwords_to_token_list_and_remove_words_with_subword_conjunction(
                        children)
                    for child in children:
                        for phraselet_template in (
                                phraselet_template for phraselet_template in
                                self.semantic_matching_helper.phraselet_templates if not
                                phraselet_template.single_word() and (
                                    not phraselet_template.reverse_only or include_reverse_only)):
                            if dependency.label in \
                                    phraselet_template.dependency_labels and \
                                    doc[parent.token_index].tag_ in phraselet_template.parent_tags\
                                    and doc[child.token_index].tag_ in \
                                    phraselet_template.child_tags and \
                                    doc[parent.token_index]._.holmes.is_matchable and \
                                    doc[child.token_index]._.holmes.is_matchable:
                                phraselet_doc = self.semantic_analyzer.parse(
                                    phraselet_template.template_sentence)
                                if parent.token_index in token_indexes_to_multiword_lemmas:
                                    parent_lemma = parent_derived_lemma = \
                                        token_indexes_to_multiword_lemmas[parent.token_index]
                                else:
                                    parent_lemma, parent_derived_lemma = \
                                        get_lemmas_from_index(parent)
                                if self.ontology is not None and replace_with_hypernym_ancestors:
                                    parent_lemma, parent_derived_lemma = \
                                        replace_lemmas_with_most_general_ancestor(
                                            parent_lemma, parent_derived_lemma)
                                if child.token_index in token_indexes_to_multiword_lemmas:
                                    child_lemma = child_derived_lemma = \
                                        token_indexes_to_multiword_lemmas[child.token_index]
                                else:
                                    child_lemma, child_derived_lemma = get_lemmas_from_index(child)
                                if self.ontology is not None and replace_with_hypernym_ancestors:
                                    child_lemma, child_derived_lemma = \
                                        replace_lemmas_with_most_general_ancestor(
                                            child_lemma, child_derived_lemma)
                                phraselet_doc[phraselet_template.parent_index]._.holmes.lemma = \
                                    parent_lemma
                                phraselet_doc[phraselet_template.parent_index]._.holmes.\
                                    derived_lemma = parent_derived_lemma
                                phraselet_doc[phraselet_template.child_index]._.holmes.lemma = \
                                    child_lemma
                                phraselet_doc[phraselet_template.child_index]._.holmes.\
                                    derived_lemma = child_derived_lemma
                                phraselet_label = ''.join((
                                    phraselet_template.label, ': ', parent_derived_lemma,
                                    '-', child_derived_lemma))
                                is_reverse_only_parent_lemma = False
                                if reverse_only_parent_lemmas is not None:
                                    for entry in reverse_only_parent_lemmas:
                                        if entry[0] == doc[parent.token_index]._.holmes.lemma \
                                                and entry[1] == doc[parent.token_index].pos_:
                                            is_reverse_only_parent_lemma = True
                                if parent_lemma not in stop_lemmas and child_lemma not in \
                                        stop_lemmas and not (
                                            is_reverse_only_parent_lemma
                                            and not include_reverse_only):
                                    add_new_phraselet_info(
                                        phraselet_label, phraselet_template, match_all_words,
                                        is_reverse_only_parent_lemma,
                                        parent_lemma, parent_derived_lemma,
                                        doc[parent.token_index].pos_,
                                        child_lemma, child_derived_lemma,
                                        doc[child.token_index].pos_)

            # We do not check for matchability in order to catch pos_='X', tag_='TRUNC'. This
            # is not a problem as only a limited range of parts of speech receive subwords in
            # the first place.
            for subword in (
                    subword for subword in token._.holmes.subwords if
                    subword.dependent_index is not None):
                parent_subword_index = subword.index
                child_subword_index = subword.dependent_index
                if token._.holmes.subwords[parent_subword_index].containing_token_index != \
                        token.i and \
                        token._.holmes.subwords[child_subword_index].containing_token_index != \
                        token.i:
                    continue
                for phraselet_template in (
                        phraselet_template for phraselet_template in
                        self.semantic_matching_helper.phraselet_templates if not
                        phraselet_template.single_word() and (
                            not phraselet_template.reverse_only or include_reverse_only)
                        and subword.dependency_label in phraselet_template.dependency_labels and
                        token.tag_ in phraselet_template.parent_tags):
                    phraselet_doc = self.semantic_analyzer.parse(
                        phraselet_template.template_sentence)
                    parent_lemma, parent_derived_lemma = get_lemmas_from_index(Index(
                        token.i, parent_subword_index))
                    if self.ontology is not None and replace_with_hypernym_ancestors:
                        parent_lemma, parent_derived_lemma = \
                            replace_lemmas_with_most_general_ancestor(
                                parent_lemma, parent_derived_lemma)
                    child_lemma, child_derived_lemma = get_lemmas_from_index(Index(
                        token.i, child_subword_index))
                    if self.ontology is not None and replace_with_hypernym_ancestors:
                        child_lemma, child_derived_lemma = \
                                replace_lemmas_with_most_general_ancestor(
                                    child_lemma, child_derived_lemma)
                    phraselet_doc[phraselet_template.parent_index]._.holmes.lemma = \
                        parent_lemma
                    phraselet_doc[phraselet_template.parent_index]._.holmes.derived_lemma = \
                        parent_derived_lemma
                    phraselet_doc[phraselet_template.child_index]._.holmes.lemma = \
                        child_lemma
                    phraselet_doc[phraselet_template.child_index]._.holmes.derived_lemma = \
                        child_derived_lemma
                    phraselet_label = ''.join((
                        phraselet_template.label, ': ', parent_derived_lemma, '-',
                        child_derived_lemma))
                    add_new_phraselet_info(
                        phraselet_label, phraselet_template, match_all_words,
                        False, parent_lemma, parent_derived_lemma, token.pos_, child_lemma,
                        child_derived_lemma, token.pos_)
        if len(phraselet_labels_to_phraselet_infos) == 0 and not match_all_words:
            for token in doc:
                process_single_word_phraselet_templates(
                    token, None, False, token_indexes_to_multiword_lemmas)

    def create_search_phrases_from_phraselet_infos(self, phraselet_infos):
        """ Creates search phrases from phraselet info objects, returning a dictionary from
            phraselet labels to the created search phrases.
        """

        def create_phraselet_label(phraselet_info):
            if phraselet_info.child_lemma is not None:
                return ''.join((
                    phraselet_info.template_label, ': ', phraselet_info.parent_derived_lemma, '-',
                    phraselet_info.child_derived_lemma))
            else:
                return ''.join((
                    phraselet_info.template_label, ': ', phraselet_info.parent_derived_lemma))

        def create_search_phrase_from_phraselet(phraselet_info):
            for phraselet_template in self.semantic_matching_helper.phraselet_templates:
                if phraselet_info.template_label == phraselet_template.label:
                    phraselet_doc = phraselet_info.template_doc
                    phraselet_doc[phraselet_template.parent_index]._.holmes.lemma = \
                        phraselet_info.parent_lemma
                    phraselet_doc[phraselet_template.parent_index]._.holmes.derived_lemma = \
                        phraselet_info.parent_derived_lemma
                    if phraselet_info.child_lemma is not None:
                        phraselet_doc[phraselet_template.child_index]._.holmes.lemma = \
                            phraselet_info.child_lemma
                        phraselet_doc[phraselet_template.child_index]._.holmes.derived_lemma = \
                            phraselet_info.child_derived_lemma
                    return self.create_search_phrase(
                        'topic match phraselet', phraselet_doc,
                        create_phraselet_label(phraselet_info), phraselet_template,
                        phraselet_info.created_without_matching_tags,
                        phraselet_info.reverse_only_parent_lemma)
            raise RuntimeError(' '.join((
                'Phraselet template', phraselet_info.template_label, 'not found.')))

        return {
            create_phraselet_label(phraselet_info) :
            create_search_phrase_from_phraselet(phraselet_info) for phraselet_info in
            phraselet_infos}

    def topic_match_documents_against(self, *,
            phraselet_labels_to_phraselet_infos, phraselet_labels_to_search_phrases):
        """ Performs a topic match against the loaded documents.

        """

        class CorpusWordPosition:
            def __init__(self, document_label, index):
                self.document_label = document_label
                self.index = index

            def __eq__(self, other):
                return isinstance(other, CorpusWordPosition) and self.index == other.index and \
                        self.document_label == other.document_label

            def __hash__(self):
                return hash((self.document_label, self.index))

            def __str__(self):
                return ':'.join((self.document_label, str(self.index)))

        class PhraseletWordMatchInfo:
            def __init__(self):
                self.single_word_match_corpus_words = set()
                # The indexes at which the single word phraselet for this word was matched.

                self.phraselet_labels_to_parent_match_corpus_words = {}
                # Dictionary from phraselets with this word as the parent to indexes where the
                # phraselet was matched.

                self.phraselet_labels_to_child_match_corpus_words = {}
                # Dictionary from phraselets with this word as the child to indexes where the
                # phraselet was matched.

                self.parent_match_corpus_words_to_matches = {}
                # Dictionary from indexes where phraselets with this word as the parent were matched
                # to the match objects.

                self.child_match_corpus_words_to_matches = {}
                # Dictionary from indexes where phraselets with this word as the child were matched
                # to the match objects.

        def get_phraselet_word_match_info(word):
            if word in self._words_to_phraselet_word_match_infos:
                return self._words_to_phraselet_word_match_infos[word]
            else:
                phraselet_word_match_info = PhraseletWordMatchInfo()
                self._words_to_phraselet_word_match_infos[word] = phraselet_word_match_info
                return phraselet_word_match_info

        def set_phraselet_to_reverse_only_where_too_many_single_word_matches(phraselet):
            """ Where the parent word of a phraselet matched too often in the corpus, the phraselet
                is set to reverse matching only to improve performance.
            """
            parent_token = phraselet.root_token
            parent_word = parent_token._.holmes.lemma_or_derived_lemma()
            if parent_word in self._words_to_phraselet_word_match_infos:
                parent_phraselet_word_match_info = self._words_to_phraselet_word_match_infos[
                    parent_word]
                parent_single_word_match_corpus_words = \
                    parent_phraselet_word_match_info.single_word_match_corpus_words
                if len(parent_single_word_match_corpus_words) > \
                    self.maximum_number_of_single_word_matches_for_relation_matching:
                    phraselet.treat_as_reverse_only_during_initial_relation_matching = True

        def get_indexes_for_reverse_matching(
                *, phraselet,
                parent_document_labels_to_indexes_for_direct_retry_sets,
                parent_document_labels_to_indexes_for_embedding_retry_sets,
                child_document_labels_to_indexes_for_embedding_retry_sets):
            """
            parent_document_labels_to_indexes_for_direct_retry_sets -- indexes where matching
                against a reverse matching phraselet should be attempted. These are ascertained
                by examining the child words.
            parent_document_labels_to_indexes_for_embedding_retry_sets -- indexes where matching
                against a phraselet should be attempted with embedding-based matching on the
                parent (root) word. These are ascertained by examining the child words.
            child_document_labels_to_indexes_for_embedding_retry_sets -- indexes where matching
                against a phraselet should be attempted with embedding-based matching on the
                child (non-root) word. These are ascertained by examining the parent words.
            """

            parent_token = phraselet.root_token
            parent_word = parent_token._.holmes.lemma_or_derived_lemma()
            if parent_word in self._words_to_phraselet_word_match_infos and not \
                    phraselet.reverse_only and not \
                    phraselet.treat_as_reverse_only_during_initial_relation_matching:
                parent_phraselet_word_match_info = self._words_to_phraselet_word_match_infos[
                    parent_word]
                parent_single_word_match_corpus_words = \
                    parent_phraselet_word_match_info.single_word_match_corpus_words
                if phraselet.label in parent_phraselet_word_match_info.\
                        phraselet_labels_to_parent_match_corpus_words:
                    parent_relation_match_corpus_words = \
                        parent_phraselet_word_match_info.\
                        phraselet_labels_to_parent_match_corpus_words[phraselet.label]
                else:
                    parent_relation_match_corpus_words = []
                if len(parent_single_word_match_corpus_words) <= \
                        self.maximum_number_of_single_word_matches_for_embedding_matching:
                    # we deliberately use the number of single matches rather than the difference
                    # because the deciding factor should be whether or not enough match information
                    # has been returned without checking the embeddings
                    for corpus_word_position in parent_single_word_match_corpus_words.difference(
                            parent_relation_match_corpus_words):
                        self._add_to_dict_set(
                            child_document_labels_to_indexes_for_embedding_retry_sets,
                            corpus_word_position.document_label, Index(
                                corpus_word_position.index.token_index,
                                corpus_word_position.index.subword_index))
            child_token = [token for token in phraselet.matchable_tokens if token.i !=
                           parent_token.i][0]
            child_word = child_token._.holmes.lemma_or_derived_lemma()
            if child_word in self._words_to_phraselet_word_match_infos:
                child_phraselet_word_match_info = \
                        self._words_to_phraselet_word_match_infos[child_word]
                child_single_word_match_corpus_words = \
                        child_phraselet_word_match_info.single_word_match_corpus_words
                if phraselet.label in child_phraselet_word_match_info.\
                        phraselet_labels_to_child_match_corpus_words:
                    child_relation_match_corpus_words = child_phraselet_word_match_info.\
                        phraselet_labels_to_child_match_corpus_words[phraselet.label]
                else:
                    child_relation_match_corpus_words = []
                if len(child_single_word_match_corpus_words) <= \
                        self.maximum_number_of_single_word_matches_for_embedding_matching:
                    set_to_add_to = parent_document_labels_to_indexes_for_embedding_retry_sets
                elif len(child_single_word_match_corpus_words) <= \
                        self.maximum_number_of_single_word_matches_for_relation_matching and (
                            phraselet.reverse_only or
                            phraselet.treat_as_reverse_only_during_initial_relation_matching):
                    set_to_add_to = parent_document_labels_to_indexes_for_direct_retry_sets
                else:
                    return
                linking_dependency = parent_token._.holmes.get_label_of_dependency_with_child_index(
                    child_token.i)
                for corpus_word_position in child_single_word_match_corpus_words.difference(
                        child_relation_match_corpus_words):
                    doc = self.indexed_documents[corpus_word_position.document_label].doc
                    working_index = corpus_word_position.index
                    working_token = doc[working_index.token_index]
                    if not working_index.is_subword() or \
                            working_token._.holmes.subwords[working_index.subword_index].is_head:
                        for parent_dependency in \
                                working_token._.holmes.coreference_linked_parent_dependencies:
                            if self.semantic_matching_helper.dependency_labels_match(
                                    search_phrase_dependency_label=linking_dependency,
                                    document_dependency_label=parent_dependency[1],
                                    inverse_polarity=False):
                                self._add_to_dict_set(
                                    set_to_add_to,
                                    corpus_word_position.document_label,
                                    Index(parent_dependency[0], None))
                        for child_dependency in \
                                working_token._.holmes.coreference_linked_child_dependencies:
                            if self.structural_matcher.use_reverse_dependency_matching and \
                                    self.semantic_matching_helper.dependency_labels_match(
                                    search_phrase_dependency_label=linking_dependency,
                                    document_dependency_label=child_dependency[1],
                                    inverse_polarity=True):
                                self._add_to_dict_set(
                                    set_to_add_to,
                                    corpus_word_position.document_label,
                                    Index(child_dependency[0], None))
                    else:
                        working_subword = \
                            working_token._.holmes.subwords[working_index.subword_index]
                        if self.semantic_matching_helper.dependency_labels_match(
                                search_phrase_dependency_label=linking_dependency,
                                document_dependency_label=
                                working_subword.governing_dependency_label,
                                inverse_polarity=False):
                            self._add_to_dict_set(
                                set_to_add_to,
                                corpus_word_position.document_label,
                                Index(working_index.token_index,
                                      working_subword.governor_index))
                        if self.structural_matcher.use_reverse_dependency_matching and \
                                self.semantic_matching_helper.dependency_labels_match(
                                search_phrase_dependency_label=linking_dependency,
                                document_dependency_label=
                                working_subword.dependency_label,
                                inverse_polarity=True):
                            self._add_to_dict_set(
                                set_to_add_to,
                                corpus_word_position.document_label,
                                Index(working_index.token_index,
                                      working_subword.dependent_index))

        def rebuild_document_info_dict(matches, phraselet_labels_to_phraselet_infos):

            def process_word_match(match, parent): # 'True' -> parent, 'False' -> child
                word_match = self._get_word_match_from_match(match, parent)
                word = word_match.search_phrase_token._.holmes.lemma_or_derived_lemma()
                phraselet_word_match_info = get_phraselet_word_match_info(word)
                corpus_word_position = CorpusWordPosition(
                    match.document_label, word_match.get_document_index())
                if parent:
                    self._add_to_dict_list(
                        phraselet_word_match_info.parent_match_corpus_words_to_matches,
                        corpus_word_position, match)
                    self._add_to_dict_list(
                        phraselet_word_match_info.phraselet_labels_to_parent_match_corpus_words,
                        match.search_phrase_label, corpus_word_position)
                else:
                    self._add_to_dict_list(
                        phraselet_word_match_info.child_match_corpus_words_to_matches,
                        corpus_word_position, match)
                    self._add_to_dict_list(
                        phraselet_word_match_info.phraselet_labels_to_child_match_corpus_words,
                        match.search_phrase_label, corpus_word_position)

            self._words_to_phraselet_word_match_infos = {}
            for match in matches:
                if match.from_single_word_phraselet:
                    phraselet_info = phraselet_labels_to_phraselet_infos[match.search_phrase_label]
                    word = phraselet_info.parent_derived_lemma
                    phraselet_word_match_info = get_phraselet_word_match_info(word)
                    word_match = match.word_matches[0]
                    phraselet_word_match_info.single_word_match_corpus_words.add(
                        CorpusWordPosition(match.document_label, word_match.get_document_index()))
                else:
                    process_word_match(match, True)
                    process_word_match(match, False)

        def filter_superfluous_matches(match):

            def get_other_matches_at_same_word(match, parent):  # 'True' -> parent, 'False' -> child
                word_match = self._get_word_match_from_match(match, parent)
                word = word_match.search_phrase_token._.holmes.lemma_or_derived_lemma()
                phraselet_word_match_info = get_phraselet_word_match_info(word)
                corpus_word_position = CorpusWordPosition(
                    match.document_label, word_match.get_document_index())
                if parent:
                    match_dict = phraselet_word_match_info.parent_match_corpus_words_to_matches
                else:
                    match_dict = phraselet_word_match_info.child_match_corpus_words_to_matches
                return match_dict[corpus_word_position]

            def check_for_sibling_match_with_higher_similarity(
                    match, other_match, word_match, other_word_match):
                    # We do not want the same phraselet to match multiple siblings, so choose
                    # the sibling that is most similar to the search phrase token.
                if self.structural_matcher.overall_similarity_threshold == 1.0:
                    return True
                if word_match.document_token.i == other_word_match.document_token.i:
                    return True
                working_sibling = word_match.document_token.doc[
                    word_match.document_token._.holmes.token_or_lefthand_sibling_index]
                for sibling in \
                        working_sibling._.holmes.loop_token_and_righthand_siblings(
                            word_match.document_token.doc):
                    if match.search_phrase_label == other_match.search_phrase_label and \
                            other_word_match.document_token.i == sibling.i and \
                            other_word_match.similarity_measure > word_match.similarity_measure:
                        return False
                return True

            def perform_checks_at_pole(match, parent): # pole is 'True' -> parent, 'False' -> child
                this_this_pole_word_match = self._get_word_match_from_match(match, parent)
                this_pole_index = this_this_pole_word_match.document_token.i
                this_other_pole_word_match = self._get_word_match_from_match(match, not parent)
                for other_this_pole_match in get_other_matches_at_same_word(match, parent):
                    other_other_pole_word_match = \
                        self._get_word_match_from_match(other_this_pole_match, not parent)
                    if this_other_pole_word_match.document_subword is not None:
                        this_other_pole_subword_index = this_other_pole_word_match.\
                            document_subword.index
                    else:
                        this_other_pole_subword_index = None
                    if other_other_pole_word_match.document_subword is not None:
                        other_other_pole_subword_index = other_other_pole_word_match.\
                            document_subword.index
                    else:
                        other_other_pole_subword_index = None
                    if this_other_pole_word_match.document_token.i == other_other_pole_word_match.\
                            document_token.i and this_other_pole_subword_index == \
                            other_other_pole_subword_index and \
                            other_other_pole_word_match.similarity_measure > \
                            this_other_pole_word_match.similarity_measure:
                        # The other match has a higher similarity measure at the other pole than
                        # this match. The matched tokens are the same. The matching phraselets
                        # must be different.
                        return False
                    if this_other_pole_word_match.document_token.i == other_other_pole_word_match.\
                            document_token.i and this_other_pole_subword_index is not None \
                            and other_other_pole_subword_index is None:
                        # This match is with a subword where the other match has matched the entire
                        # word, so this match should be removed.
                        return False
                        # Check unnecessary if parent==True as it has then already
                        # been carried out during structural matching.
                    if not parent and this_other_pole_word_match.document_token.i != \
                            other_other_pole_word_match.document_token.i and \
                            other_other_pole_word_match.document_token.i in \
                            this_other_pole_word_match.document_token._.\
                            holmes.token_and_coreference_chain_indexes and \
                            match.search_phrase_label == other_this_pole_match.search_phrase_label \
                            and (
                                    (
                                        abs(this_pole_index -
                                            this_other_pole_word_match.document_token.i) >
                                        abs(this_pole_index -
                                            other_other_pole_word_match.document_token.i)
                                    )
                                    or
                                    (
                                        abs(this_pole_index -
                                            this_other_pole_word_match.document_token.i) ==
                                        abs(this_pole_index -
                                            other_other_pole_word_match.document_token.i) and
                                        this_other_pole_word_match.document_token.i >
                                        other_other_pole_word_match.document_token.i
                                    )
                                ):
                        # The document tokens at the other poles corefer with each other and
                        # the other match's token is closer to the second document token (the
                        # one at this pole). Both matches are from the same phraselet.
                        # If the tokens from the two matches are the same distance from the document
                        # token at this pole but on opposite sides of it, the preceding one beats
                        # the succeeding one simply because we have to choose one or the other.
                        return False

                    if not check_for_sibling_match_with_higher_similarity(
                            match, other_this_pole_match, this_other_pole_word_match,
                            other_other_pole_word_match):
                        return False
                return True

            if match.from_single_word_phraselet:
                return True
            if not perform_checks_at_pole(match, True):
                return False
            if not perform_checks_at_pole(match, False):
                return False
            return True

        def remove_duplicates(matches):
            # Situations where the same document tokens have been matched by multiple phraselets
            matches_to_return = []
            if len(matches) == 0:
                return matches_to_return
            else:
                matches_to_return.append(matches[0])
            if len(matches) > 1:
                previous_whole_word_single_word_match = None
                for counter in range(1, len(matches)):
                    this_match = matches[counter]
                    previous_match = matches[counter-1]
                    if this_match.index_within_document == previous_match.index_within_document:
                        if previous_match.from_single_word_phraselet and \
                                previous_match.get_subword_index() is None:
                            previous_whole_word_single_word_match = previous_match
                        if this_match.get_subword_index() is not None and \
                                previous_whole_word_single_word_match is not None and \
                                this_match.index_within_document == \
                                previous_whole_word_single_word_match.index_within_document:
                            # This match is against a subword where the whole word has also been
                            # matched, so reject it
                            continue
                    if this_match.document_label != previous_match.document_label:
                        matches_to_return.append(this_match)
                    elif len(this_match.word_matches) != len(previous_match.word_matches):
                        matches_to_return.append(this_match)
                    else:
                        this_word_matches_indexes = [
                            word_match.get_document_index() for word_match in
                            this_match.word_matches]
                        previous_word_matches_indexes = [
                            word_match.get_document_index() for word_match in
                            previous_match.word_matches]
                        # In some circumstances the two phraselets may have matched the same
                        # tokens the opposite way round
                        if sorted(this_word_matches_indexes) != \
                                sorted(previous_word_matches_indexes):
                            matches_to_return.append(this_match)
            return matches_to_return

        # First get single-word matches
        structural_matches = self.structural_matcher.match(
            indexed_documents=self.indexed_documents,
            search_phrases=phraselet_labels_to_search_phrases.values(),
            output_document_matching_message_to_console=False,
            match_depending_on_single_words=True,
            compare_embeddings_on_root_words=False,
            compare_embeddings_on_non_root_words=False,
            document_labels_to_indexes_for_reverse_matching_sets=None,
            document_labels_to_indexes_for_embedding_reverse_matching_sets=None,
            document_label_filter=self.document_label_filter)
        if not self.embedding_based_matching_on_root_words:
            rebuild_document_info_dict(structural_matches, phraselet_labels_to_phraselet_infos)
            for phraselet in (
                    phraselet_labels_to_search_phrases[phraselet_info.label] for
                    phraselet_info in phraselet_labels_to_phraselet_infos.values() if
                    phraselet_info.child_lemma is not None):
                set_phraselet_to_reverse_only_where_too_many_single_word_matches(phraselet)

        # Now get normally matched relations
        structural_matches.extend(self.structural_matcher.match(
            indexed_documents=self.indexed_documents,
            search_phrases=phraselet_labels_to_search_phrases.values(),
            output_document_matching_message_to_console=False,
            match_depending_on_single_words=False,
            compare_embeddings_on_root_words=False,
            compare_embeddings_on_non_root_words=False,
            document_labels_to_indexes_for_reverse_matching_sets=None,
            document_labels_to_indexes_for_embedding_reverse_matching_sets=None,
            document_label_filter=self.document_label_filter))

        rebuild_document_info_dict(structural_matches, phraselet_labels_to_phraselet_infos)
        parent_document_labels_to_indexes_for_direct_retry_sets = {}
        parent_document_labels_to_indexes_for_embedding_retry_sets = {}
        child_document_labels_to_indexes_for_embedding_retry_sets = {}
        for phraselet in (
                phraselet_labels_to_search_phrases[phraselet_info.label] for
                phraselet_info in phraselet_labels_to_phraselet_infos.values() if
                phraselet_info.child_lemma is not None):
            get_indexes_for_reverse_matching(
                phraselet=phraselet,
                parent_document_labels_to_indexes_for_direct_retry_sets=
                parent_document_labels_to_indexes_for_direct_retry_sets,
                parent_document_labels_to_indexes_for_embedding_retry_sets=
                parent_document_labels_to_indexes_for_embedding_retry_sets,
                child_document_labels_to_indexes_for_embedding_retry_sets=
                child_document_labels_to_indexes_for_embedding_retry_sets)
        if len(parent_document_labels_to_indexes_for_embedding_retry_sets) > 0 or \
                len(parent_document_labels_to_indexes_for_direct_retry_sets) > 0:

            # Perform reverse matching at selected indexes
            structural_matches.extend(self.structural_matcher.match(
                indexed_documents=self.indexed_documents,
                search_phrases=phraselet_labels_to_search_phrases.values(),
                output_document_matching_message_to_console=False,
                match_depending_on_single_words=False,
                compare_embeddings_on_root_words=True,
                compare_embeddings_on_non_root_words=False,
                document_labels_to_indexes_for_reverse_matching_sets=
                parent_document_labels_to_indexes_for_direct_retry_sets,
                document_labels_to_indexes_for_embedding_reverse_matching_sets=
                parent_document_labels_to_indexes_for_embedding_retry_sets,
                document_label_filter=self.document_label_filter))

        if len(child_document_labels_to_indexes_for_embedding_retry_sets) > 0:

            # Retry normal matching at selected indexes with embedding-based matching on children
            structural_matches.extend(self.structural_matcher.match(
                indexed_documents=self.indexed_documents,
                search_phrases=phraselet_labels_to_search_phrases.values(),
                output_document_matching_message_to_console=False,
                match_depending_on_single_words=False,
                compare_embeddings_on_root_words=False,
                compare_embeddings_on_non_root_words=True,
                document_labels_to_indexes_for_reverse_matching_sets=None,
                document_labels_to_indexes_for_embedding_reverse_matching_sets=
                child_document_labels_to_indexes_for_embedding_retry_sets,
                document_label_filter=self.document_label_filter))
        if len(parent_document_labels_to_indexes_for_direct_retry_sets) > 0 or \
                len(parent_document_labels_to_indexes_for_embedding_retry_sets) > 0 or \
                len(child_document_labels_to_indexes_for_embedding_retry_sets) > 0:
            rebuild_document_info_dict(structural_matches, phraselet_labels_to_phraselet_infos)
        structural_matches = list(filter(filter_superfluous_matches, structural_matches))
        phraselet_labels_to_frequency_factors = {info.label: info.frequency_factor for info
            in phraselet_labels_to_phraselet_infos.values()}
        position_sorted_structural_matches = sorted(
            structural_matches, key=lambda match:
            (
                match.document_label, match.index_within_document,
                match.get_subword_index_for_sorting(), match.from_single_word_phraselet))
        position_sorted_structural_matches = remove_duplicates(position_sorted_structural_matches)
        # Read through the documents measuring the activation based on where
        # in the document structural matches were found
        score_sorted_structural_matches = self.perform_activation_scoring(
            position_sorted_structural_matches, phraselet_labels_to_frequency_factors)
        return self.get_topic_matches(
            score_sorted_structural_matches, position_sorted_structural_matches)

    def perform_activation_scoring(self, position_sorted_structural_matches,
        phraselet_labels_to_frequency_factors):
        """
        Read through the documents measuring the activation based on where
        in the document structural matches were found.
        """
        def get_set_from_dict(dictionary, key):
            if key in dictionary:
                return dictionary[key]
            else:
                return set()

        def get_current_activation_for_phraselet(phraselet_activation_tracker, current_index):
            distance_to_last_match = current_index - phraselet_activation_tracker.position
            tailoff_quotient = distance_to_last_match / self.maximum_activation_distance
            if tailoff_quotient > 1.0:
                tailoff_quotient = 1.0
            return (1-tailoff_quotient) * phraselet_activation_tracker.score

        document_labels_to_indexes_to_phraselet_labels = {}
        for match in (
                match for match in position_sorted_structural_matches if not
                match.from_single_word_phraselet and
                        match.word_matches[0].document_token.i !=
                        match.word_matches[1].document_token.i): # two subwords within word):
            if match.document_label in document_labels_to_indexes_to_phraselet_labels:
                inner_dict = document_labels_to_indexes_to_phraselet_labels[match.document_label]
            else:
                inner_dict = {}
                document_labels_to_indexes_to_phraselet_labels[match.document_label] = inner_dict
            parent_word_match = self._get_word_match_from_match(match, True)
            self._add_to_dict_set(
                inner_dict, parent_word_match.get_document_index(), match.search_phrase_label)
            child_word_match = self._get_word_match_from_match(match, False)
            self._add_to_dict_set(
                inner_dict, child_word_match.get_document_index(), match.search_phrase_label)
        current_document_label = None
        for pssm_index, match in enumerate(position_sorted_structural_matches):
            match.original_index_within_list = pssm_index # store for later use after resorting
            if match.document_label != current_document_label or pssm_index == 0:
                current_document_label = match.document_label
                phraselet_labels_to_phraselet_activation_trackers = {}
                indexes_to_phraselet_labels = document_labels_to_indexes_to_phraselet_labels.get(
                    current_document_label, {})
            match.is_overlapping_relation = False
            if match.from_single_word_phraselet or \
                    match.word_matches[0].document_token.i == \
                    match.word_matches[1].document_token.i: # two subwords within word
                if match.from_topic_match_phraselet_created_without_matching_tags:
                    this_match_score = self.single_word_any_tag_score
                else:
                    this_match_score = self.single_word_score
            else:
                if match.from_reverse_only_topic_match_phraselet:
                    this_match_score = self.reverse_only_relation_score
                else:
                    this_match_score = self.relation_score
                this_match_parent_word_match = self._get_word_match_from_match(match, True)
                this_match_parent_index = this_match_parent_word_match.get_document_index()
                this_match_child_word_match = self._get_word_match_from_match(match, False)
                this_match_child_index = this_match_child_word_match.get_document_index()
                other_relevant_phraselet_labels = get_set_from_dict(
                    indexes_to_phraselet_labels,
                    this_match_parent_index) | \
                    get_set_from_dict(indexes_to_phraselet_labels, this_match_child_index)
                other_relevant_phraselet_labels.remove(match.search_phrase_label)
                if len(other_relevant_phraselet_labels) > 0:
                    match.is_overlapping_relation = True
                    this_match_score *= self.overlapping_relation_multiplier

            # multiply the score by the frequency factor, which is 1.0 if frequency factors are
            # not being used
            this_match_score *= phraselet_labels_to_frequency_factors[match.search_phrase_label]

            overall_similarity_measure = float(match.overall_similarity_measure)
            if overall_similarity_measure < 1.0:
                this_match_score *= self.embedding_penalty * overall_similarity_measure
            for word_match in (word_match for word_match in match.word_matches \
                    if word_match.type == 'ontology'):
                this_match_score *= (self.ontology_penalty ** (abs(word_match.depth) + 1))
            if match.search_phrase_label in phraselet_labels_to_phraselet_activation_trackers:
                phraselet_activation_tracker = phraselet_labels_to_phraselet_activation_trackers[
                    match.search_phrase_label]
                current_score = get_current_activation_for_phraselet(
                    phraselet_activation_tracker, match.index_within_document)
                if this_match_score > current_score:
                    phraselet_activation_tracker.score = this_match_score
                else:
                    phraselet_activation_tracker.score = current_score
                phraselet_activation_tracker.position = match.index_within_document
            else:
                phraselet_labels_to_phraselet_activation_trackers[match.search_phrase_label] =\
                        PhraseletActivationTracker(match.index_within_document, this_match_score)
            match.topic_score = 0
            for phraselet_label in list(phraselet_labels_to_phraselet_activation_trackers):
                phraselet_activation_tracker = phraselet_labels_to_phraselet_activation_trackers[
                    phraselet_label]
                current_activation = get_current_activation_for_phraselet(
                    phraselet_activation_tracker, match.index_within_document)
                if current_activation <= 0:
                    del phraselet_labels_to_phraselet_activation_trackers[phraselet_label]
                else:
                    match.topic_score += current_activation
        return sorted(position_sorted_structural_matches, key=lambda match: 0-match.topic_score)

    def get_topic_matches(
            self, score_sorted_structural_matches, position_sorted_structural_matches):
        """Resort the matches starting with the highest (most active) and
            create topic match objects with information about the surrounding sentences.
        """

        def match_contained_within_existing_topic_match(topic_matches, match):
            for topic_match in topic_matches:
                if match.document_label == topic_match.document_label and \
                        match.index_within_document >= topic_match.start_index and \
                        match.index_within_document <= topic_match.end_index:
                    return True
            return False

        def alter_start_and_end_indexes_for_match(start_index, end_index, match):
            for word_match in match.word_matches:
                if word_match.first_document_token.i < start_index:
                    start_index = word_match.first_document_token.i
                if word_match.document_subword is not None and \
                        word_match.document_subword.containing_token_index < start_index:
                    start_index = word_match.document_subword.containing_token_index
                if word_match.last_document_token.i > end_index:
                    end_index = word_match.last_document_token.i
                if word_match.document_subword is not None and \
                        word_match.document_subword.containing_token_index > end_index:
                    end_index = word_match.document_subword.containing_token_index
            return start_index, end_index

        if self.only_one_result_per_document:
            existing_document_labels = []
        topic_matches = []
        counter = 0
        for score_sorted_match in score_sorted_structural_matches:
            if counter >= self.number_of_results:
                break
            if match_contained_within_existing_topic_match(topic_matches, score_sorted_match):
                continue
            if self.only_one_result_per_document and score_sorted_match.document_label \
                    in existing_document_labels:
                continue
            start_index, end_index = alter_start_and_end_indexes_for_match(
                score_sorted_match.index_within_document,
                score_sorted_match.index_within_document,
                score_sorted_match)
            previous_index_within_list = score_sorted_match.original_index_within_list
            while previous_index_within_list > 0 and position_sorted_structural_matches[
                    previous_index_within_list-1].document_label == \
                    score_sorted_match.document_label and position_sorted_structural_matches[
                        previous_index_within_list].topic_score > self.different_match_cutoff_score:
                    # previous_index_within_list rather than previous_index_within_list -1 :
                    # when a complex structure is matched, it will often begin with a single noun
                    # that should be included within the topic match indexes
                if match_contained_within_existing_topic_match(
                        topic_matches, position_sorted_structural_matches[
                            previous_index_within_list-1]):
                    break
                if score_sorted_match.index_within_document - position_sorted_structural_matches[
                        previous_index_within_list-1].index_within_document > \
                        self.sideways_match_extent:
                    break
                previous_index_within_list -= 1
                start_index, end_index = alter_start_and_end_indexes_for_match(
                    start_index, end_index,
                    position_sorted_structural_matches[previous_index_within_list])
            next_index_within_list = score_sorted_match.original_index_within_list
            while next_index_within_list + 1 < len(score_sorted_structural_matches) and \
                    position_sorted_structural_matches[next_index_within_list+1].document_label == \
                    score_sorted_match.document_label and \
                    position_sorted_structural_matches[next_index_within_list+1].topic_score >= \
                    self.different_match_cutoff_score:
                if match_contained_within_existing_topic_match(
                        topic_matches, position_sorted_structural_matches[
                            next_index_within_list+1]):
                    break
                if position_sorted_structural_matches[
                        next_index_within_list+1].index_within_document - \
                        score_sorted_match.index_within_document > self.sideways_match_extent:
                    break
                next_index_within_list += 1
                start_index, end_index = alter_start_and_end_indexes_for_match(
                    start_index, end_index,
                    position_sorted_structural_matches[next_index_within_list])
            working_document = self.indexed_documents[score_sorted_match.document_label].doc
            relevant_sentences = [
                sentence for sentence in working_document.sents
                if sentence.end > start_index and sentence.start <= end_index]
            sentences_start_index = relevant_sentences[0].start
            sentences_end_index = relevant_sentences[-1].end
            text = working_document[sentences_start_index: sentences_end_index].text
            topic_matches.append(
                TopicMatch(
                    score_sorted_match.document_label,
                    score_sorted_match.index_within_document,
                    score_sorted_match.get_subword_index(),
                    start_index, end_index, sentences_start_index, sentences_end_index - 1,
                    score_sorted_match.topic_score, text, position_sorted_structural_matches[
                        previous_index_within_list:next_index_within_list+1]))
            if self.only_one_result_per_document:
                existing_document_labels.append(score_sorted_match.document_label)
            counter += 1
        # If two matches have the same score, order them by length
        return sorted(
            topic_matches, key=lambda topic_match: (
                0-topic_match.score, topic_match.start_index - topic_match.end_index))

class TopicMatchDictionaryOrderer:
    # extracted into its own class to facilite use by MultiprocessingManager

    def order(self, topic_match_dicts, number_of_results, tied_result_quotient):

        topic_match_dicts = sorted(
            topic_match_dicts, key=lambda dict: (
                0-dict['score'], 0-len(dict['text'].split()), dict['document_label'],
                dict['word_infos'][0][0]))
        topic_match_dicts = topic_match_dicts[0:number_of_results]
        topic_match_counter = 0
        while topic_match_counter < len(topic_match_dicts):
            topic_match_dicts[topic_match_counter]['rank'] = str(topic_match_counter + 1)
            following_topic_match_counter = topic_match_counter + 1
            while following_topic_match_counter < len(topic_match_dicts) and \
                    topic_match_dicts[following_topic_match_counter]['score'] / topic_match_dicts[
                        topic_match_counter]['score'] > tied_result_quotient:
                working_rank = ''.join((str(topic_match_counter + 1), '='))
                topic_match_dicts[topic_match_counter]['rank'] = working_rank
                topic_match_dicts[following_topic_match_counter]['rank'] = working_rank
                following_topic_match_counter += 1
            topic_match_counter = following_topic_match_counter
        return topic_match_dicts
