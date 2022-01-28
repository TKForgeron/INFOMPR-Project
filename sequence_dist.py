from preprocessing.preprocessing import _for_all_files, regex, BASE_DIR_PROCESSED
import pandas as pd
import pprint

pp = pprint.PrettyPrinter(indent=4)

total_type = {}
total_app = {}


def process_file(root, name, filename, **kwargs):
    global total_type, total_app
    match = regex.search(name)
    application_type = match.group(1)
    application_name = match.group(2)
    application_name = f"{application_type}_{application_name}"

    df = pd.read_csv(filename)
    num_sequences = len(df)

    if application_type not in total_type.keys():
        total_type[application_type] = 0

    if application_name not in total_app.keys():
        total_app[application_name] = 0

    total_type[application_type] += num_sequences
    total_app[application_name] += num_sequences


_for_all_files(process_file, BASE_DIR_PROCESSED)
print("Types: ")
pp.pprint(total_type)
print("Type_Applications: ")
pp.pprint(total_app)
