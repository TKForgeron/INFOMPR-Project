from csv_rename import rename_all_in_folder
from flow_generator import generate_flow_sequences
import os

from preprocess import (
    BASE_DIR_RAW,
    BASE_DIR_RENAMED,
    BASE_DIR_SEQUENCES,
    BASE_DIR_TAGGED,
    _process_files,
    _sequences_to_rows,
)


def pipeline():
    if os.path.exists(BASE_DIR_RAW):
        rename_all_in_folder()
    if os.path.exists(BASE_DIR_RENAMED):
        generate_flow_sequences()
    if os.path.exists(BASE_DIR_SEQUENCES):
        _process_files()
    if os.path.exists(BASE_DIR_TAGGED):
        _sequences_to_rows()


if __name__ == "__main__":
    pipeline()
