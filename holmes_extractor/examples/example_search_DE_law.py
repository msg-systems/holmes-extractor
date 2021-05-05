import urllib.request
from bs4 import BeautifulSoup
import holmes_extractor as holmes

def download_and_register(url, label):
    print('Downloading', label)
    # Download the content
    page = urllib.request.urlopen(url)
    # Extract the raw text from the HTML document
    soup = BeautifulSoup(page, 'html.parser')
    # Register the document with Holmes
    print('Parsing and registering', label)
    holmes_manager.parse_and_register_document(soup.get_text(), label)

# Start the Holmes Manager with the German model
holmes_manager = holmes.Manager(model='de_core_news_lg')
download_and_register('https://www.gesetze-im-internet.de/vvg_2008/BJNR263110007.html', 'VVG_2008')
download_and_register('https://www.gesetze-im-internet.de/vag_2016/BJNR043410015.html', 'VAG')
holmes_manager.start_topic_matching_search_mode_console()

# Example queries:
#
# Der Versicherer darf den Vertrag fristlos kündigen, wenn der Versicherungsnehmer beim Abschluss des Vertrags die vorvertragliche Anzeigepflicht verletzt hat.
# Der Versicherer darf Leistungen verweigern.
# Der Versicherer darf die Prämie anpassen.
# Eine Richtlinie einer ENTITYORG
