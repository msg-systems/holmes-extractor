import os
import holmes_extractor as holmes

if __name__ in ('__main__', 'example_chatbot_EN_insurance'):
    script_directory = os.path.dirname(os.path.realpath(__file__))
    ontology = holmes.Ontology(os.sep.join((
        script_directory, 'example_chatbot_EN_insurance_ontology.owl')))
    holmes_manager = holmes.Manager(
        model='en_core_web_lg', number_of_workers=2)
    holmes_manager.register_search_phrase('Somebody requires insurance')
    holmes_manager.register_search_phrase('An ENTITYPERSON takes out insurance')
    holmes_manager.register_search_phrase('A company buys payment insurance')
    holmes_manager.register_search_phrase('An ENTITYPERSON needs insurance')
    holmes_manager.register_search_phrase('Insurance for a period')
    holmes_manager.register_search_phrase('An insurance begins')
    holmes_manager.register_search_phrase('Somebody prepays')
    holmes_manager.register_search_phrase('Somebody makes an insurance payment')

    holmes_manager.start_chatbot_mode_console()
    # e.g. 'Richard Hudson and John Doe require health insurance for the next five years'
