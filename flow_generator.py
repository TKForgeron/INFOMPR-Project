# The mini-flow generator

from ftplib import error_perm
import pandas as pd
import numpy as np
import datetime
from preprocess import _for_all_files
from os import path, makedirs

BASE_DIR_120s = "data/CSV"
BASE_DIR_3s = "data/CSV-3s"
BASE_DIR_sequences = "data/CSV-sequences/"
SEQUENCE_LENGTH = 5

# print(sec1, sec120)

seenIds = set()
fullTotal = 0


def d2T(timestamp, duration):
    date = datetime.datetime.strptime(
        timestamp.replace("p.m.", "PM"), "%d/%m/%Y %H:%M:%S %p"
    )
    timestamp120 = datetime.datetime.timestamp(date)
    endtime120 = timestamp120 + (int(duration) / 1e6)
    return (timestamp120, endtime120)


def sequenceToRows(sequence):
    """Given an array of miniflows, outputs an array of sequenced miniflows, with stride=1."""
    output = []
    for i in range(0, len(sequence) - SEQUENCE_LENGTH + 1):
        for j in range(i, i + SEQUENCE_LENGTH):
            output.append(sequence[j])
    return output


if __name__ == "__main__":
    error_count = []
    gTotal = 0
    files = 0

    def process_file(filename, root, name, **kwargs):
        try:
            global error_count, gTotal, files
            df120 = pd.read_csv(filename)
            df3 = pd.read_csv(filename.replace(BASE_DIR_120s, BASE_DIR_3s))
            files += len(df3)
            total = 0
            out_rows = []
            matched_js = set()
            df3["Start Time"] = 0
            df3["End Time"] = 0
            for j in range(0, len(df3)):
                starttime1, endtime1 = d2T(df3["Timestamp"][j], df3["Flow Duration"][j])
                df3["Start Time"][j] = starttime1
                df3["End Time"][j] = endtime1

            nf3 = df3.to_numpy()
            startIndex = df3.columns.get_loc("Start Time")
            endIndex = df3.columns.get_loc("End Time")
            flowIndex = df3.columns.get_loc("Flow ID")

            for i in range(0, len(df120)):
                rowTotal = 0
                flow120 = df120["Flow ID"][i]
                starttime120, endtime120 = d2T(
                    df120["Timestamp"][i], df120["Flow Duration"][i]
                )
                miniflow_row_ids = []

                for j in range(0, len(df3)):
                    starttime1, endtime1 = nf3[j, startIndex], nf3[j, endIndex]
                    if (
                        starttime1 >= starttime120
                        and endtime1 <= endtime120
                        and flow120 == nf3[j, flowIndex]
                        and not j in matched_js
                    ):
                        matched_js.add(j)
                        rowTotal += 1
                        miniflow_row_ids.append(j)
                if rowTotal >= SEQUENCE_LENGTH:
                    total += (
                        rowTotal - SEQUENCE_LENGTH + 1
                    )  # HET IS DUS WEL DIT, MAAR DAN HOEF JE JE EINDPERCENTAGE NIET MEER DOOR 5 TE DELEN
                    rows_subset = [nf3[row, :] for row in miniflow_row_ids]
                    for row in sequenceToRows(rows_subset):
                        out_rows.append(row)
            gTotal += total
            print(filename, ": ", total / len(df3))
            sequence_matrix = np.array([list(df3.columns), *(out_rows)])
            if not path.exists(BASE_DIR_sequences + root):
                makedirs(BASE_DIR_sequences + root)
            np.savetxt(
                path.join(BASE_DIR_sequences, filename),
                sequence_matrix,
                fmt="%s",
                delimiter=",",
            )
        except Exception as e:
            # print("Error on file: ", filename)
            error_count.append(e)
            print(e)

    _for_all_files(process_file, BASE_DIR_120s)

    if all(error_count):
        if len(error_count) == 0:
            print(f"No errors during processing.")
        else:
            print(
                f"{len(error_count)} files not processed due to '{str(error_count[0]).split(': ')[0]}...'"
            )
    else:
        print(f"{len(error_count)} files not processed due to error")

    print(f"Percentage of usable mini-flows: {gTotal / files *100}")
    # print(gTotalList)
