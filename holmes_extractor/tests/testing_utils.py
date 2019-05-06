import holmes_extractor as holmes

class HolmesInstanceManager:

    instance = None
    def __init__(self, ontology):
        if not HolmesInstanceManager.instance:
            HolmesInstanceManager.instance = HolmesInstanceManager.__HolmesInstanceManager(ontology)
            HolmesInstanceManager.instance.en_core_web_lg.remove_all_documents()
            HolmesInstanceManager.instance.en_core_web_lg.remove_all_search_phrases()
            HolmesInstanceManager.instance.en_core_web_lg_ontology.remove_all_documents()
            HolmesInstanceManager.instance.en_core_web_lg_ontology.remove_all_search_phrases()
            HolmesInstanceManager.instance.de_core_news_sm.remove_all_documents()
            HolmesInstanceManager.instance.de_core_news_sm.remove_all_search_phrases()
            HolmesInstanceManager.instance.en_coref_lg.remove_all_documents()
            HolmesInstanceManager.instance.en_coref_lg.remove_all_search_phrases()
            HolmesInstanceManager.instance.en_coref_lg_ontology.remove_all_documents()
            HolmesInstanceManager.instance.en_coref_lg_ontology.remove_all_search_phrases()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    class __HolmesInstanceManager:
        def __init__(self, ontology):
            self.en_core_web_lg = holmes.Manager('en_core_web_lg')
            self.en_core_web_lg_ontology = holmes.Manager(model='en_core_web_lg', ontology=ontology)
            self.de_core_news_sm = holmes.Manager('de_core_news_sm')
            self.en_coref_lg = holmes.Manager('en_coref_lg')
            self.en_coref_lg_ontology = holmes.Manager(model='en_coref_lg', ontology=ontology)
