import pandas as pd
import os
import re


def add_label(
    base_dir: str = "data/", out_dir: str = "data/cleaned/", out_name: str = "compleet"
) -> None:

    file_names = [file for file in os.listdir(base_dir)]
    frames = []
    # applicationType_applicationName_index[a | b].pcap_Flow.csv
    regex = re.compile("^(.*)_\d*.*\.pcap_Flow.*")

    for name in file_names:
        match = regex.search(name)
        pcap_type = match.group(1)
        df = pd.read_csv(base_dir + name)
        df["label"] = pcap_type
        frames.append(df)

    csv = pd.concat(frames)
    csv.to_csv(f"{out_dir + out_name}.csv", index=False)
