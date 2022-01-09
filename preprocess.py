import pandas as pd
import os
import re

BASE_DIR = "data/CSV/"
BASE_DIR_PROCESSED = "data/CSV-Labelled/"
CSV_FEATURES = ["Flow ID","Src IP","Src Port","Dst IP","Dst Port","Protocol"]

# File name format:
# applicationType_applicationName_index[a | b].pcap_Flow.csv

def process_files():
    """Goes through all files and folders in 'BASE_DIR' and adds labels for the application type and application name to them.
    Will also remove the features that are not required by the model.

    Make sure that all files conform to the format "applicationType_applicationName_index[a | b].pcap_Flow.csv", where [a | b] means an optional character that is 'a' or 'b'.
    """

    regex = re.compile("^([^_]*)_(.*)_\d*(.*)\.pcap_Flow.csv")

    for root, dirs, files in os.walk(BASE_DIR):
        for name in files:
            if name.endswith((".pcap_Flow.csv")):
                filename = os.path.join(root, name)
                match = regex.search(name)
                application_type = match.group(1)
                application_name = match.group(2)
                df = pd.read_csv(filename)[CSV_FEATURES]
                df["Label"] = application_type
                df["Application"] = application_name

                if not os.path.exists(BASE_DIR_PROCESSED + root):
                    os.makedirs(BASE_DIR_PROCESSED + root)
                df.to_csv(BASE_DIR_PROCESSED + filename, index=False)

process_files()