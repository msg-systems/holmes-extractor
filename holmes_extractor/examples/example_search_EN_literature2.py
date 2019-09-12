import urllib.request
import holmes_extractor as holmes
import re

print('Initializing Holmes...')
#Start the Holmes manager with the English model
holmes_manager = holmes.Manager(model='en_core_web_lg',
        overall_similarity_threshold=0.85)

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
    chapter_headings = [heading for heading in re.finditer("(?<=((\\n\\n\\n\\n)|(\* \\n\\n)))((?!.*(WEASLEY WILL MAKE SURE)|(DO NOT OPEN THE PARCEL)|(HEADMISTRESS OF HOGWARTS))[A-Z\-’., ]+)(\\n{1,2}((?!.*(WHO\-MUST))[A-Z\-’., ]+))?(?=(\\n\\n([^\\n]|(\\n\\n((“Harry!”)|(Harry’s)|(Ron’s)|(“Hagrid)|(Three o’clock))))))", book)]
    chapter_counter = 1
    for chapter_heading in chapter_headings:
        label = ''.join(('Book ', title, '; Ch ', str(chapter_counter), ': ', chapter_heading.group().replace('\n', '')))
        if chapter_counter == len(chapter_headings): # last chapter
            content = book[chapter_heading.end():]
        else:
            content = book[chapter_heading.end():chapter_headings[chapter_counter].start()]
        content = content.replace('\n', '')
        if content.endswith('& '):
            content = content[:-2]
        print('Parsing and registering', label)
        holmes_manager.parse_and_register_document(content, label)
        chapter_counter += 1

extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20-%20The%20Philosopher's%20Stone.txt", '1: The Philosopher\'s Stone')
extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%202%20-%20The%20Chamber%20of%20Secrets.txt", '2: The Chamber of Secrets')
extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%203%20-%20The%20Prisoner%20of%20Azkaban.txt", '3: The Prisoner of Azkaban')
extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%204%20-%20The%20Goblet%20of%20Fire.txt", '4: The Goblet of Fire')
extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%205%20-%20The%20Order%20of%20the%20Phoenix.txt", '5: The Order of the Phoenix')
extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%206%20-%20The%20Half%20Blood%20Prince.txt", '6: The Half Blood Prince')
extract_chapters_from_book("https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%207%20-%20The%20Deathly%20Hallows.txt", '7: The Deathly Hallows')

#Comment following line in to activate interactive console
holmes_manager.start_search_mode_console()
#Only return one topic match per story

# The following code starts a RESTful Http service to perform topic searches. It is deployed as
# as WSGI application. Examples of how to start it are (both issued from the directory that
# contains the script):

# gunicorn --reload example_search_DE_literature (Linux)
# waitress-serve example_search_DE_literature:application (Windows)

#class RestHandler():
#    def on_get(self, req, resp):
#        resp.body = json.dumps(holmes_manager.topic_match_documents_returning_dictionaries_against(
#                req.params['entry'], only_one_result_per_document=True))
#
#application = falcon.API()
#application.add_route('/maerchenQuery', RestHandler())
