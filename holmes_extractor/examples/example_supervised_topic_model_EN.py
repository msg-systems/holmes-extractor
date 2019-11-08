import os
import shutil
import urllib.request
import zipfile
import holmes_extractor as holmes

working_directory=REPLACE WITH PATH TO WORKING DIRECTORY IN SINGLE OR DOUBLE QUOTES

def is_training_data(document_number):
    # We use any documents with numbers ending in 8,9,0 for test and all other documents for
    # training.
    return document_number[-1:] not in ('8','9','0')

def get_document_filename_info(filename):
    # e.g. 'bbc/business/001.txt'
    category = filename.split('/')[1]
    document_number = filename.split('/')[2].split('.')[0]
    return category, document_number

def evaluate_classifier(zip_filename, classifier):
    correct_classification_counter = wrong_classification_counter = \
    no_classification_counter = correct_as_additional_classification_counter = 0
    with zipfile.ZipFile(zip_filename) as bbc_zipfile:
        for filename in (filename for filename in bbc_zipfile.namelist() if
                filename.lower().endswith('.txt') and not filename.endswith('README.TXT')):
            category, document_number = get_document_filename_info(filename)
            if not is_training_data(document_number):
                with bbc_zipfile.open(filename, 'r') as test_doc:
                    test_contents = str(test_doc.read())
                    test_contents = test_contents.replace('\n', ' ').replace('\r', ' ')
                suggested_categories = classifier.parse_and_classify(test_contents)
                if len(suggested_categories) == 0:
                    no_classification_counter += 1
                elif suggested_categories[0] == category:
                    correct_classification_counter += 1
                elif category in suggested_categories:
                    correct_as_additional_classification_counter += 1
                else:
                    wrong_classification_counter += 1
                print(''.join((filename, ': actual category ', category,
                        '; suggested categories ', str(suggested_categories))))
    print()
    print('Totals:')
    print(correct_classification_counter, 'correct classifications;')
    print(no_classification_counter, 'unclassified documents;')
    print(wrong_classification_counter, 'incorrect classifications;')
    print(correct_as_additional_classification_counter, 'incorrect classifications where the correct classification was returned as an additional classification.')

def train_model(working_directory, zip_filename):
    training_basis = holmes_manager.get_supervised_topic_training_basis()
    with zipfile.ZipFile(zip_filename) as bbc_zipfile:
        for filename in (filename for filename in bbc_zipfile.namelist() if
                filename.lower().endswith('.txt') and not filename.endswith('README.TXT')):
            category, document_number = get_document_filename_info(filename)
            if is_training_data(document_number):
                with bbc_zipfile.open(filename, 'r') as training_doc:
                    training_contents = str(training_doc.read())
                    training_contents = training_contents.replace('\n', ' ').replace('\r', ' ')
                training_basis.parse_and_register_training_document(
                        training_contents, category, filename)
    training_basis.prepare()
    classifier = training_basis.train().classifier()
    output_filename = os.sep.join((working_directory, 'model.json'))
    with open(output_filename, "w") as f:
        f.write(classifier.serialize_model())
    evaluate_classifier(zip_filename, classifier)
holmes_manager = holmes.Manager('en_core_web_lg')

if os.path.exists(working_directory):
    if not os.path.isdir(working_directory):
        raise RuntimeError(' '.join((working_directory), 'must be a directory'))
else:
    os.mkdir(working_directory)
zip_filename = (os.sep.join((working_directory,'bbc-fulltext.zip')))
if not os.path.exists(zip_filename):
    url='http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'
    with urllib.request.urlopen(url) as response, open(zip_filename, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
model_filename = os.sep.join((working_directory, 'model.json'))
if not os.path.exists(model_filename):
    train_model(working_directory, zip_filename)
else:
    print('Reloading existing trained model. Delete model.json from working directory to repeat training.')
    with open(model_filename) as model_file:
        classifier = holmes_manager.deserialize_supervised_topic_classifier(model_file.read())
    evaluate_classifier(zip_filename, classifier)
