import urllib.request
from bs4 import BeautifulSoup
import holmes_extractor as holmes
import os

working_directory=REPLACE WITH PATH TO WORKING DIRECTORY IN SINGLE OR DOUBLE QUOTES
HOLMES_EXTENSION = 'hdc'
flag_filename = os.sep.join((working_directory,'STORY_PARSING_COMPLETE'))

print('Initializing Holmes...')
# Start the Holmes manager with the German model
holmes_manager = holmes.Manager(model='de_core_news_md', overall_similarity_threshold=0.9)

def process_documents_from_front_page(front_page_uri, front_page_label, labels_to_documents):
    """ Download and save all the stories from from a front page."""

    front_page = urllib.request.urlopen(front_page_uri)
    front_page_soup = BeautifulSoup(front_page, 'html.parser')
    # For each story ...
    for anchor in front_page_soup.find_all('a'):
        if not anchor['href'].startswith('/') and not anchor['href'].startswith('https'):
            this_document_url = '/'.join((front_page_uri, anchor['href']))
            print('Downloading story', anchor.contents[0], 'from front page', front_page_label)
            # Get the HTML document for the story
            this_document = urllib.request.urlopen(this_document_url)
            # Extract the raw text from the HTML document
            this_document_soup = BeautifulSoup(this_document, 'html.parser')
            this_document_text = this_document_soup.prettify()
            this_document_text = this_document_text.split('</h1>', 1)[1]
            this_document_text = this_document_text.split('<span class="autor"', 1)[0]
            this_document_text = this_document_text.replace('<br/>', ' ')
            # Remove any carriage returns and line feeds from the raw text
            this_document_text = this_document_text.replace('\n', ' ').replace('\r', ' ').replace('  ', ' ')
            # Replace multiple spaces with single spaces
            this_document_text = ' '.join(this_document_text.split())
            # Create a document label from the front page label and the story name
            this_document_label = ' - '.join((front_page_label, anchor.contents[0]))
            labels_to_documents[this_document_label] = this_document_text
            print('Parsing', this_document_label)
            # Parse the document
            document = holmes_manager.parse_and_register_document(this_document_text,
                    this_document_label)
            # Save the document
            print('Saving', this_document_label)
            output_filename = os.sep.join((working_directory, this_document_label))
            output_filename = '.'.join((output_filename, HOLMES_EXTENSION))
            with open(output_filename, "w") as f:
                f.write(holmes_manager.serialize_document(this_document_label))

def load_documents_from_working_directory(labels_to_documents):
    for file in os.listdir(working_directory):
        if file.endswith(HOLMES_EXTENSION):
            print('Loading', file)
            label = file[:-4]
            long_filename = os.sep.join((working_directory, file))
            with open(long_filename, "r") as f:
                contents = f.read()
            holmes_manager.deserialize_and_register_document(contents, label)

if os.path.exists(working_directory):
    if not os.path.isdir(working_directory):
        raise RuntimeError(' '.join((working_directory), 'must be a directory'))
else:
    os.mkdir(working_directory)
labels_to_documents={}

if os.path.isfile(flag_filename):
    load_documents_from_working_directory(labels_to_documents)
else:
    process_documents_from_front_page("https://maerchen.com/grimm/", 'Gebrüder Grimm',
            labels_to_documents)
    process_documents_from_front_page("https://maerchen.com/grimm2/", 'Gebrüder Grimm',
            labels_to_documents)
    process_documents_from_front_page("https://maerchen.com/andersen/", 'Hans Christian Andersen',
            labels_to_documents)
    process_documents_from_front_page("https://maerchen.com/bechstein/", 'Ludwig Bechstein',
            labels_to_documents)
    process_documents_from_front_page("https://maerchen.com/wolf/", 'Johann Wilhelm Wolf',
            labels_to_documents)
    # Generate flag file to indicate files can be reloaded on next run
    open(flag_filename, 'a').close()
# Only return one topic match per story
holmes_manager.start_search_mode_console(only_one_topic_match_per_document=True)
