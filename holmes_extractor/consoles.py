from .about import __version__
from .errors import (
    SearchPhraseContainsNegationError,
    SearchPhraseContainsConjunctionError,
    SearchPhraseContainsCoreferringPronounError,
    SearchPhraseWithoutMatchableWordsError,
    NoSearchPhraseError,
    SearchPhraseContainsMultipleClausesError,
)


class HolmesConsoles:
    """Manages the consoles."""

    def __init__(self, holmes):
        self.holmes = holmes
        self.semantic_analyzer = holmes.semantic_analyzer
        self.structural_matcher = holmes.structural_matcher

    def match_description(self, match_dict):
        """Returns a user-readable representation of a match dictionary."""
        match_description_to_return = ""
        if match_dict["negated"]:
            match_description_to_return = "; negated"
        if match_dict["uncertain"]:
            match_description_to_return = "".join(
                (match_description_to_return, "; uncertain")
            )
        if match_dict["involves_coreference"]:
            match_description_to_return = "".join(
                (match_description_to_return, "; involves coreference")
            )
        overall_similarity_measure = float(match_dict["overall_similarity_measure"])
        if overall_similarity_measure < 1.0:
            match_description_to_return = "".join(
                (
                    match_description_to_return,
                    "; overall similarity measure=",
                    str(overall_similarity_measure),
                )
            )
        return match_description_to_return

    def string_representation_of_word_match(self, word_match):
        """Returns a user-readable representation of a word match."""
        if word_match["document_word"] != word_match["extracted_word"]:
            extracted_word = "".join(
                ("(refers to ", word_match["extracted_word"].upper(), ")")
            )
        else:
            extracted_word = ""
        string = "".join(
            (
                "'",
                word_match["document_phrase"],
                "'",
                extracted_word,
                "->'",
                word_match["search_phrase_word"],
                "' (",
                word_match["explanation"][:-1],
                ")",
            )
        )
        return string

    def common(self):
        """Contains functionality common to both consoles."""
        print("Holmes version", __version__, "written by richard.hudson@explosion.ai")
        print(
            "Note that the consoles do not display all information that is available when using Holmes programmatically."
        )
        print()
        print("Language is", self.semantic_analyzer.language_name)
        print("Model is", self.semantic_analyzer.model)
        if self.structural_matcher.perform_coreference_resolution:
            print("Coreference resolution is ON")
        else:
            print("Coreference resolution is OFF")
        if self.structural_matcher.analyze_derivational_morphology:
            print("Derivational morphology analysis is ON")
        else:
            print("Derivational morphology analysis is OFF")
        if self.structural_matcher.use_reverse_dependency_matching:
            print("Reverse dependency matching is ON")
        else:
            print("Reverse dependency matching is OFF")

    def print_document_info(self):
        document_labels = self.holmes.list_document_labels()
        if len(document_labels) == 0:
            raise RuntimeError("No documents registered.")
        document_labels_string = "; ".join(
            "".join(("'", l, "'")) for l in document_labels
        )
        print(": ".join(("Documents", document_labels_string)))

    def start_chatbot_mode(self):
        """Starts a chatbot mode console enabling the matching of pre-registered
        search phrases to short example documents entered ad-hoc by the user.
        """
        self.common()
        print(
            "Overall similarity threshold is",
            str(self.holmes.overall_similarity_threshold),
        )
        if self.holmes.overall_similarity_threshold < 1.0:
            if self.structural_matcher.embedding_based_matching_on_root_words:
                print("Embedding-based matching on root words is ON")
            else:
                print("Embedding-based matching on root words is OFF")
        print()
        print("Chatbot mode")
        print()
        if len(self.holmes.search_phrases) == 0:
            raise RuntimeError("No search_phrases registered.")
        # Display search phrases
        for search_phrase in self.holmes.search_phrases:
            print("".join(("Search phrase '", search_phrase.doc_text, "'")))

        print()
        print("Ready for input")
        while True:
            print()
            search_sentence = input()
            print()
            if search_sentence in ("exit", "exit()", "bye"):
                break
            match_dicts = self.holmes.match(document_text=search_sentence)
            for match_dict in match_dicts:
                print()
                print(
                    "".join(
                        (
                            "Matched search phrase with text '",
                            match_dict["search_phrase_text"],
                            "'",
                            self.match_description(match_dict),
                            ":",
                        )
                    )
                )
                word_matches_string = "; ".join(
                    map(
                        self.string_representation_of_word_match,
                        match_dict["word_matches"],
                    )
                )
                print(word_matches_string)

    def start_structural_extraction_mode(self):
        """Starts a structural extraction mode console enabling the matching of pre-registered
        documents to search phrases entered ad-hoc by the user.
        """
        self.common()
        print(
            "Overall similarity threshold is",
            str(self.holmes.overall_similarity_threshold),
        )
        if self.holmes.overall_similarity_threshold < 1.0:
            if self.structural_matcher.embedding_based_matching_on_root_words:
                print("Embedding-based matching on root words is ON")
            else:
                print("Embedding-based matching on root words is OFF")
        print()
        print("Structural extraction mode")
        print()
        self.print_document_info()
        print()
        while True:
            print("Ready for phrases")
            print()
            search_phrase = input()
            # removing question marks seems to lead to better results
            search_phrase = search_phrase.strip(" ").strip("?")
            if search_phrase == "":
                continue
            if search_phrase in ("exit", "exit()", "bye"):
                break
            print()
            match_dicts = []
            try:
                match_dicts = self.holmes.match(search_phrase_text=search_phrase)
                if len(match_dicts) == 0:
                    print("No structural matching results were returned.")
                else:
                    print("Structural matching results:")
            except SearchPhraseContainsNegationError:
                print(
                    "Structural matching was not attempted because the search phrase contained "
                    "negation (not, never)."
                )
                print()
            except SearchPhraseContainsConjunctionError:
                print(
                    "Structural matching was not attempted because the search phrase contained "
                    "conjunction (and, or)."
                )
                print()
            except SearchPhraseContainsCoreferringPronounError:
                print(
                    "Structural matching was not attempted because the search phrase contained a "
                    "pronoun that referred back to a noun."
                )
                print()
            except SearchPhraseWithoutMatchableWordsError:
                print(
                    "Structural matching was not attempted because the search phrase did not "
                    " contain any words that could be matched."
                )
                print()
            except SearchPhraseContainsMultipleClausesError:
                print(
                    "Structural matching was not attempted because the search phrase contained "
                    "multiple clauses."
                )
                print()
            print()
            for match_dict in match_dicts:
                print()
                print(
                    "".join(
                        (
                            "Matched document '",
                            match_dict["document"],
                            "' at index ",
                            str(match_dict["index_within_document"]),
                            self.match_description(match_dict),
                            ":",
                        )
                    )
                )
                print("".join(('"', match_dict["sentences_within_document"], '"')))
                word_matches_string = "; ".join(
                    map(
                        self.string_representation_of_word_match,
                        match_dict["word_matches"],
                    )
                )
                print(word_matches_string)

    def start_topic_matching_search_mode(
        self,
        only_one_result_per_document,
        word_embedding_match_threshold,
        initial_question_word_embedding_match_threshold,
    ):
        """Starts a topic matching search mode console enabling the matching of pre-registered
        documents to search texts entered ad-hoc by the user.
        """
        self.common()
        print(
            "The embedding similarity threshold for normal words is",
            str(word_embedding_match_threshold),
        )
        print(
            "The embedding similarity threshold for initial question words is",
            str(initial_question_word_embedding_match_threshold),
        )
        print("Topic matching search mode")
        print()
        self.print_document_info()
        print()
        while True:
            print("Ready for search texts")
            print()
            search_text = input()
            # removing question marks seems to lead to better results
            search_text = search_text.strip(" ").strip("?")
            if search_text == "":
                continue
            if search_text in ("exit", "exit()", "bye"):
                break
            print()
            print("Performing topic match searching ...")
            try:
                print()
                topic_match_dicts = self.holmes.topic_match_documents_against(
                    search_text,
                    number_of_results=5,
                    only_one_result_per_document=only_one_result_per_document,
                    word_embedding_match_threshold=word_embedding_match_threshold,
                    initial_question_word_embedding_match_threshold=initial_question_word_embedding_match_threshold,
                )
            except NoSearchPhraseError:
                pass
            if topic_match_dicts is None or len(topic_match_dicts) == 0:
                print("No topic match results were returned.")
                print()
                continue
            elif only_one_result_per_document:
                print("Topic matching results (maximum one per document):")
            else:
                print("Topic matching results:")
            print()
            for topic_match_dict in topic_match_dicts:
                textual_answers = []
                for dict_answer in topic_match_dict["answers"]:
                    textual_answers.append(
                        "".join(
                            (
                                "'",
                                topic_match_dict["text"][
                                    dict_answer[0] : dict_answer[1]
                                ],
                                "'",
                            )
                        )
                    )
                answers_string = (
                    "".join(("; question answer ", "; ".join(textual_answers)))
                    if len(textual_answers) > 0
                    else ""
                )
                output = "".join(
                    (
                        topic_match_dict["rank"],
                        ". Document ",
                        topic_match_dict["document_label"],
                        "; sentences at character indexes ",
                        str(topic_match_dict["sentences_character_start_index"]),
                        "-",
                        str(topic_match_dict["sentences_character_end_index"]),
                        "; score ",
                        str(topic_match_dict["score"]),
                        answers_string,
                        ":",
                    )
                )
                print(output)
                print()
                print(topic_match_dict["text"])
                print()
            print()
