import os
import re
import json
import urllib.request
from bs4 import BeautifulSoup
import holmes_extractor as holmes
import falcon

if __name__ in ('__main__', 'example_search_EN_literature'):

    working_directory = # REPLACE WITH PATH TO WORKING DIRECTORY IN SINGLE OR DOUBLE QUOTES
    HOLMES_EXTENSION = 'hdc'
    flag_filename = os.sep.join((working_directory, 'STORY_PARSING_COMPLETE'))
    print('Initializing Holmes (this may take some time) ...')

    script_directory = os.path.dirname(os.path.realpath(__file__))
    ontology = holmes.Ontology(os.sep.join((
        script_directory, 'example_search_EN_literature_ontology.owl')))

    # Start the Holmes manager with the English model
    holmes_manager = holmes.Manager(
        model='en_core_web_trf', ontology=ontology)

    def extract_chapters_from_book(book_uri, title):
        """ Download and save the chapters from a book."""

        print()
        print(title)
        print()
        book = urllib.request.urlopen(book_uri).read().decode()
        book = re.sub("\\nPage \|.+?Rowling \\n", "", book)
        book = re.sub("\\nP a g e \|.+?Rowling \\n", "", book)
        book = re.sub("\\nPage \|.+?\\n", "", book)
        book = book.replace("Harry Potter and the Half Blood Prince - J.K. Rowling", "")
        book = book.replace("Harry Potter and the Goblet of Fire - J.K. Rowling", "")
        book = book.replace("Harry Potter and the Deathly Hallows - J.K. Rowling", "")
        book = book[1:]
        chapter_headings = [heading for heading in re.finditer("(?<=((\\n\\n\\n\\n)|(\* \\n\\n)))((?!.*(WEASLEY WILL MAKE SURE)|(DO NOT OPEN THE PARCEL)|(HEADMISTRESS OF HOGWARTS))[A-Z][A-Z\-’., ]+)(\\n{1,2}((?!.*(WHO\-MUST))[A-Z\-’., ]+))?(?=(\\n\\n([^\\n]|(\\n\\n((“Harry!”)|(Harry’s)|(Ron’s)|(“Hagrid)|(Three o’clock))))))", book)]
        chapter_counter = 1
        labels = []
        chapter_texts = []
        chapter_dict = {}
        for chapter_heading in chapter_headings:
            label = ''.join((
                'Book ', title, ' Ch ', str(chapter_counter), " ‘",
                chapter_heading.group().replace('\n', '').strip(), "’"))
            labels.append(label)
            if chapter_counter == len(chapter_headings): # last chapter
                content = book[chapter_heading.end():]
            else:
                content = book[chapter_heading.end():chapter_headings[chapter_counter].start()]
            content = content.replace('\n', '')
            if content.endswith('& '):
                content = content[:-2]
            chapter_texts.append(content)
            print('Extracted', label)
            chapter_counter += 1
        parsed_chapters = holmes_manager.nlp.pipe(chapter_texts)
        for index, parsed_chapter in enumerate(parsed_chapters):
            label = labels[index]
            print('Saving', label)
            output_filename = os.sep.join((working_directory, label))
            output_filename = '.'.join((output_filename, HOLMES_EXTENSION))
            with open(output_filename, "wb") as file:
                file.write(parsed_chapter.to_bytes())

    def load_documents_from_working_directory():
        serialized_documents = {}
        for file in os.listdir(working_directory):
            if file.endswith(HOLMES_EXTENSION):
                print('Loading', file)
                label = file[:-4]
                long_filename = os.sep.join((working_directory, file))
                with open(long_filename, "rb") as file:
                    contents = file.read()
                serialized_documents[label] = contents
        print('Indexing documents (this may take some time) ...')
        holmes_manager.register_serialized_documents(serialized_documents)

    if os.path.exists(working_directory):
        if not os.path.isdir(working_directory):
            raise RuntimeError(' '.join((working_directory), 'must be a directory'))
    else:
        os.mkdir(working_directory)

    if os.path.isfile(flag_filename):
        load_documents_from_working_directory()
    else:
        extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20-%20The%20Philosopher's%20Stone.txt", "1 ‘The Philosopher\'s Stone’")
        extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%202%20-%20The%20Chamber%20of%20Secrets.txt", "2 ‘The Chamber of Secrets’")
        extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%203%20-%20The%20Prisoner%20of%20Azkaban.txt", "3 ‘The Prisoner of Azkaban’")
        extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%204%20-%20The%20Goblet%20of%20Fire.txt", "4 ‘The Goblet of Fire’")
        extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%205%20-%20The%20Order%20of%20the%20Phoenix.txt", "5 ‘The Order of the Phoenix’")
        extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%206%20-%20The%20Half%20Blood%20Prince.txt", "6 ‘The Half Blood Prince’")
        extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%207%20-%20The%20Deathly%20Hallows.txt", "7 ‘The Deathly Hallows’")
        # Generate flag file to indicate files can be reloaded on next run
        open(flag_filename, 'a').close()
        load_documents_from_working_directory()

    #Comment following line in to activate interactive console
    #holmes_manager.start_topic_matching_search_mode_console(only_one_result_per_document=True)

    # The following code starts a RESTful Http service to perform topic searches. It is deployed as
    # as WSGI application. An example of how to start it - issued from the directory that
    # contains the script - is

    # waitress-serve example_search_DE_literature:application

    class RestHandler():
        def on_get(self, req, resp):
            resp.text = \
                json.dumps(holmes_manager.topic_match_documents_against(
                    req.params['entry'][0:200]))
            resp.cache_control = ["s-maxage=31536000"]

    application = falcon.App()
    application.add_route('/english', RestHandler())
