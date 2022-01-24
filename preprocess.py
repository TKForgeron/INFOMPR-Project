from preprocessing.csv_rename import rename_all_in_folder
from preprocessing.flow_generator import generate_flow_sequences
import os

from preprocessing.preprocessing import (
    BASE_DIR_RAW,
    BASE_DIR_RENAMED,
    BASE_DIR_SEQUENCES,
    BASE_DIR_TAGGED,
    RANDOM_STATE,
    SEQUENCE_LENGTH,
    _process_files,
    _sequences_to_rows,
)
import preprocessing.preprocessing as pping


def total_features():
    return pping.total_features()


def labels():
    return pping.labels()


def get_train_validation_test_set():
    return pping.get_train_validation_test_set()


def do_all_preprocessing():
    if not os.path.exists(BASE_DIR_SEQUENCES):
        if os.path.exists(BASE_DIR_RAW):
            rename_all_in_folder()
        if os.path.exists(BASE_DIR_RENAMED):
            generate_flow_sequences(padding=False)

    if os.path.exists(BASE_DIR_SEQUENCES):
        _process_files()
    if os.path.exists(BASE_DIR_TAGGED):
        _sequences_to_rows()


if __name__ == "__main__":
    do_all_preprocessing()
    # generate_flow_sequences()
