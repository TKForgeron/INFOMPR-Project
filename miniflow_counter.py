# The mini-flow generator

import csv
from ftplib import error_perm
import pandas as pd
import numpy as np
import datetime
from math import floor
from preprocess import CSV_FEATURES, _for_all_files

BASE_DIR_120s = "data/120s5s"
BASE_DIR_3s = "data/3s5s"

# print(sec1, sec120)

seenIds = set()
fullTotal = 0


def miniflow_counts_to_csv(miniflows_dict, csv_out_name, sep=",") -> None:

    df_percentages_per_file = pd.DataFrame(miniflows_dict).T
    df_percentages_per_file.to_csv(
        csv_out_name,
        sep=sep,
    )
    with open(csv_out_name) as csvfile:
        lines = csvfile.readlines()

    column_names = ["file", "usable mini-flows", "percentage of all mini-flows", "\n"]
    line0 = lines[0].split(",")

    if len(line0) > 1:
        lines[0] = ",".join(column_names)
    elif ";" in line0[0]:
        lines[0] = ";".join(column_names)
    else:
        print("your file (csv) is not separated with [',',';']")

    with open(csv_out_name, "w") as csvfile:
        csvfile.writelines(lines)


def d2T(timestamp, duration):
    date = datetime.datetime.strptime(
        timestamp.replace("p.m.", "PM"), "%d/%m/%Y %H:%M:%S %p"
    )
    timestamp120 = datetime.datetime.timestamp(date)
    endtime120 = timestamp120 + (int(duration) / 1e6)
    return (timestamp120, endtime120)


if __name__ == "__main__":

    def count_miniflows_per_file(filename, root, name, **kwargs):
        error_count = ["error_count"]
        gTotal = kwargs["gTotal"]
        files = kwargs["files"]

        try:
            df120 = pd.read_csv(filename)
            df3 = pd.read_csv(filename.replace(BASE_DIR_120s, BASE_DIR_3s))
            files += len(df3)
            total = 0
            matched_smallrows = {}
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

                for j in range(0, len(df3)):
                    # starttime1, endtime1 = df3["Start Time"][j], df3["End Time"][j]
                    # if starttime1 >= starttime120 and endtime1 <= endtime120:
                    #     if df120["Protocol"][i] == df3["Protocol"][j]:
                    #         if (
                    #             # df120["Src Port"][i] == df3["Src Port"][j]
                    #             # and df120["Dst Port"][i] == df3["Dst Port"][j]
                    #             # and df120["Src IP"][i] == df3["Src IP"][j]
                    #             # and df120["Dst IP"][i] == df3["Dst IP"][j]
                    #             flow120
                    #             == df3["Flow ID"][j]
                    #             # and j not in matched_js
                    #         ):
                    #             rowTotal += 1
                    #             # if j in matched_smallrows.keys():
                    #             #     print(
                    #             #         j,
                    #             #         " (in ",
                    #             #         i,
                    #             #         ") already present for ",
                    #             #         matched_smallrows[j],
                    #             #     )
                    #             # matched_smallrows[j] = i
                    #             # matched_js.add(j)
                    starttime1, endtime1 = nf3[j, startIndex], nf3[j, endIndex]

                    if (
                        starttime1 >= starttime120
                        and endtime1 <= endtime120
                        and flow120 == nf3[j, flowIndex]
                        and not j in matched_js
                    ):
                        matched_js.add(j)
                        rowTotal += 1

                if rowTotal >= kwargs["SEQUENCE_LENGTH"]:
                    total += floor(
                        (rowTotal - kwargs["SEQUENCE_LENGTH"]) / kwargs["STRIDE"] + 1
                    )  # HET IS DUS WEL DIT, MAAR DAN HOEF JE JE EINDPERCENTAGE NIET MEER DOOR 5 TE DELEN
            gTotal += total
            usable_for_mini_flow_perc = total / len(df3)
            # print(filename, ": ", usable_for_mini_flow_perc)

            return [total, usable_for_mini_flow_perc]

        except Exception as e:
            error_count.append(e)
            # print(e)

            return [0, str(e).split(": ")[0] + "..."]

    SEQUENCE_LENGTH = 20
    STRIDE = SEQUENCE_LENGTH
    # error_count = []
    # gTotal = 0
    # files = 0
    # for stride in range(1, STRIDE + 1):
    #     for sequence_length in range(3, SEQUENCE_LENGTH + 1):
    #         print(
    #             f"Now checking for sequence_length: {sequence_length} and stride: {stride}..."
    #         )
    #         output_dict, error_count, perc_usable = _for_all_files(
    #             count_miniflows_per_file,
    #             BASE_DIR_120s,
    #             STRIDE=STRIDE,
    #             SEQUENCE_LENGTH=SEQUENCE_LENGTH,
    #             error_count=[],
    #             gTotal=0,
    #             files=0,
    #         )
    #         miniflow_counts_to_csv(
    #             miniflows_dict=output_dict,
    #             csv_out_name=f"results/sequence{SEQUENCE_LENGTH}_stride{STRIDE}_{len(error_count)}errors_{perc_usable}%usable.csv",
    #             sep=";",
    #         )

    print(
        f"Now checking for sequence_length: {SEQUENCE_LENGTH} and stride: {STRIDE}..."
    )
    output_dict, error_count, perc_usable = _for_all_files(
        count_miniflows_per_file,
        BASE_DIR_120s,
        STRIDE=STRIDE,
        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
        error_count=[],
        gTotal=0,
        files=0,
    )
    miniflow_counts_to_csv(
        miniflows_dict=output_dict,
        csv_out_name=f"results/sequence{SEQUENCE_LENGTH}_stride{STRIDE}_{len(error_count)}errors_{perc_usable}%usable.csv",
        sep=";",
    )

    if all(error_count):
        print(
            f"{len(error_count)} files not processed due to '{str(error_count[0]).split(': ')[0]}...'"
        )
    else:
        print(f"{len(error_count)} files not processed due to error")

    print(f"Percentage of usable mini-flows: {gTotal / files *100}")
    # print(gTotalList)
