# The mini-flow generator

import pandas as pd
import datetime
from preprocess import _for_all_files

BASE_DIR_120s = "data/CSV"
BASE_DIR_3s = "data/CSV-3s"

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


if __name__ == "__main__":
    gTotal = 0
    files = 0

    def process_file(filename, root, name, **kwargs):
        try:
            global gTotal, files
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
                if rowTotal >= 5:
                    total += rowTotal
            gTotal += total
            print(filename, ": ", total / len(df3))
        except:
            print("Error on file ", filename)

    _for_all_files(process_file, BASE_DIR_120s)
    print(gTotal / files)
