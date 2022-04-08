class WordMatch:
    """A match between a searched phrase word and a document word.

    Properties:

    search_phrase_token -- the spaCy token from the search phrase.
    search_phrase_word -- the word that matched from the search phrase.
    document_token -- the spaCy token from the document.
    first_document_token -- the first token that matched from the document, which will equal
        *document_token* except with multiword matches.
    last_document_token -- the last token that matched from the document, which will equal
        *document_token* except with multiword matches.
    document_subword -- the subword from the token that matched, or *None* if the match was
        with the whole token.
    document_word -- the word or subword that matched structurally from the document.
    word_match_type -- *direct*, *entity*, *embedding*, or *derivation*.
    similarity_measure -- for type *embedding*, the similarity between the two tokens,
        otherwise 1.0.
    is_negated -- *True* if this word match leads to a match of which it
      is a part being negated.
    is_uncertain -- *True* if this word match leads to a match of which it
      is a part being uncertain.
    structurally_matched_document_token -- the spaCy token from the document that matched
        the dependency structure, which may be different from *document_token* if coreference
        resolution is active.
    involves_coreference -- *True* if *document_token* and *structurally_matched_document_token*
        are different.
    extracted_word -- the most specific term that corresponded to *document_word* within the
        coreference chain.
    depth -- currently unused (always 0).
    search_phrase_initial_question_word -- *True* if *search_phrase_token* is an initial question
        word or governs an initial question word.
    """

    def __init__(
            self, search_phrase_token, search_phrase_word, document_token,
            first_document_token, last_document_token, document_subword, document_word,
            word_match_type, similarity_measure, is_negated, is_uncertain,
            structurally_matched_document_token, extracted_word, depth,
            search_phrase_initial_question_word):

        self.search_phrase_token = search_phrase_token
        self.search_phrase_word = search_phrase_word
        self.document_token = document_token
        self.first_document_token = first_document_token
        self.last_document_token = last_document_token
        self.document_subword = document_subword
        self.document_word = document_word
        self.word_match_type = word_match_type
        self.similarity_measure = similarity_measure
        self.is_negated = is_negated
        self.is_uncertain = is_uncertain
        self.structurally_matched_document_token = structurally_matched_document_token
        self.extracted_word = extracted_word
        self.depth = depth
        self.search_phrase_initial_question_word = search_phrase_initial_question_word

    @property
    def involves_coreference(self):
        return self.document_token != self.structurally_matched_document_token

    def get_document_index(self):
        if self.document_subword is not None:
            subword_index = self.document_subword.index
        else:
            subword_index = None
        return Index(self.document_token.i, subword_index)

    def explain(self):
        """ Creates a human-readable explanation of the word match from the perspective of the
            document word (e.g. to be used as a tooltip over it)."""
        search_phrase_display_word = self.search_phrase_token._.holmes.lemma.upper()
        if self.word_match_type == 'direct':
            return ''.join(("Matches ", search_phrase_display_word, " directly."))
        elif self.word_match_type == 'derivation':
            return ''.join(("Has a common stem with ", search_phrase_display_word, "."))
        elif self.word_match_type == 'entity':
            return ''.join(("Has an entity label matching ", search_phrase_display_word, "."))
        elif self.word_match_type == 'question':
            return ''.join(("Matches the question word ", search_phrase_display_word, "."))
        elif self.word_match_type == 'embedding':
            printable_similarity = str(int(self.similarity_measure * 100))
            return ''.join((
                "Has a word embedding that is ", printable_similarity,
                "% similar to ", search_phrase_display_word, "."))
        elif self.word_match_type == 'entity_embedding':
            printable_similarity = str(int(self.similarity_measure * 100))
            return ''.join((
                "Has an entity label that is ", printable_similarity,
                "% similar to the word embedding corresponding to ", search_phrase_display_word,
                "."))