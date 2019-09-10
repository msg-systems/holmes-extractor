import urllib.request
from bs4 import BeautifulSoup
import holmes_extractor as holmes

print('Initializing Holmes...')
# Start the Holmes manager with the English model
# You can try setting overall_similarity_threshold to 0.9
# and/or perform_coreference_resolution to False
holmes_manager = holmes.Manager(model='en_core_web_lg', overall_similarity_threshold=1.0,
        perform_coreference_resolution=True)
# Get the HTML document with the list of stories by Hans Christian Andersen
front_page = urllib.request.urlopen("http://www.aesopfables.com/aesophca.html")
front_page_soup = BeautifulSoup(front_page, 'html.parser')
documents={}
# For each story ...
for anchor in front_page_soup.find_all('a'):
    if anchor['href'].startswith('../'):
        this_document_url = ''.join(("http://www.aesopfables.com/", anchor['href'][3:]))
        # Get the HTML document for the story
        print('Downloading', anchor.contents[0])
        this_document = urllib.request.urlopen(this_document_url)
        # Extract the raw text from the HTML document
        this_document_soup = BeautifulSoup(this_document, 'html.parser')
        # Remove any Javascript from the raw text
        for script in this_document_soup(["script", "style"]):
            script.decompose()    # rip it out
        # Remove any carriage returns and line feeds from the raw text
        this_document_text = this_document_soup.get_text().replace('\n', ' ').replace('\r', ' ').replace('  ', ' ')
        # Replace multiple spaces with single spaces
        this_document_text = ' '.join(this_document_text.split())
        # Remove 'Process took:', which for some reason remains at the end of each raw text document
        end_of_text_index = this_document_text.index('Process took:')
        if end_of_text_index > 0:
            this_document_text = this_document_text[0:end_of_text_index]
        # Retrieve the name of the story
        documents[anchor.contents[0]] = this_document_text

for label, text in documents.items():
    # Register the document with Holmes
    print('Parsing and registering', label)
    holmes_manager.parse_and_register_document(text, label)
# Start the search console
holmes_manager.start_search_mode_console()
