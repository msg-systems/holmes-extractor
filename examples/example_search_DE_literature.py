import os
import json
import urllib.request
from bs4 import BeautifulSoup
import holmes_extractor as holmes
import falcon

if __name__ in ('__main__', 'example_search_DE_literature'):

    working_directory = # REPLACE WITH PATH TO WORKING DIRECTORY IN SINGLE OR DOUBLE QUOTES
    HOLMES_EXTENSION = 'hdc'
    flag_filename = os.sep.join((working_directory, 'STORY_PARSING_COMPLETE'))

    print('Initializing Holmes (this may take some time) ...')
    # Start the Holmes manager with the German model
    holmes_manager = holmes.Manager(
        model='de_core_news_lg')

    def process_documents_from_front_page(front_page_uri, front_page_label):
        """ Download and save all the stories from a front page."""

        front_page = urllib.request.urlopen(front_page_uri)
        front_page_soup = BeautifulSoup(front_page, 'html.parser')
        document_texts = []
        labels = []
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
                this_document_text = this_document_text.replace(
                    '\n', ' ').replace('\r', ' ').replace('  ', ' ')
                # Replace multiple spaces with single spaces
                this_document_text = ' '.join(this_document_text.split())
                # Create a document label from the front page label and the story name
                this_document_label = ' - '.join((front_page_label, anchor.contents[0]))
                document_texts.append(this_document_text)
                labels.append(this_document_label)
        parsed_documents = holmes_manager.nlp.pipe(document_texts)
        for index, parsed_document in enumerate(parsed_documents):
            label = labels[index]
            print('Saving', label)
            output_filename = os.sep.join((working_directory, label))
            output_filename = '.'.join((output_filename, HOLMES_EXTENSION))
            with open(output_filename, "wb") as file:
                file.write(parsed_document.to_bytes())

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
        process_documents_from_front_page(
            "https://maerchen.com/grimm/", 'Gebrüder Grimm')
        process_documents_from_front_page(
            "https://maerchen.com/grimm2/", 'Gebrüder Grimm')
        process_documents_from_front_page(
            "https://maerchen.com/andersen/", 'Hans Christian Andersen')
        process_documents_from_front_page(
            "https://maerchen.com/bechstein/", 'Ludwig Bechstein')
        process_documents_from_front_page(
            "https://maerchen.com/wolf/", 'Johann Wilhelm Wolf')
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
                    req.params['entry'][0:200], only_one_result_per_document=True))
            resp.cache_control = ["s-maxage=31536000"]

    application = falcon.App()
    application.add_route('/german', RestHandler())
