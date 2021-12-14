import pandas as pd
import os
import re

BASE_DIR = 'data/'

file_names = [file for file in os.listdir(BASE_DIR)]
frames = []

regex = re.compile("^(.*)_\d*.*\.pcap_Flow.*")

for name in file_names:
    match = regex.search(name)
    pcap_type = match.group(1)
    df = pd.read_csv(BASE_DIR + name)
    df['label'] = pcap_type
    frames.append(df)

csv = pd.concat(frames)
csv.to_csv("compleet.csv", index = False)
