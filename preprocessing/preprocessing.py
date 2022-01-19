import pandas as pd
import numpy as np
import os
import re

BASE_DIR_RAW = "data/raw/"
BASE_DIR_RENAMED = "data/renamed/"
BASE_DIR_SEQUENCES = "data/sequences/"
BASE_DIR_TAGGED = "data/tagged/"
BASE_DIR_PROCESSED = "data/processed/"
CSV_FEATURES = ["Src Port", "Dst Port", "Protocol"]
SEQUENCE_LENGTH = 5
RANDOM_STATE = 42

regex = re.compile("^([^_]*)_(.*)_\d*(.*)\.pcap_Flow.csv")

# File name format:
# applicationType_applicationName_[index][a | b].pcap_Flow.csv


def total_features():
    """Returns the total number of features that will be in every csv file"""
    return (
        len(CSV_FEATURES)
        + 1  # +1 for VPN
        # Note applications is not a feature, as this is what we try to find out!
    )


def labels():
    """Returns the names and total number of labels that will be in every csv file

    Returns: [(name, length)]
    """
    return [
        ("type", len(_get_distinct_types())),
        ("name", len(_get_distinct_applications())),
    ]


def get_train_validation_test_set(RANDOM_STATE=RANDOM_STATE):
    """Processes the data and splits it into a scaled train, validation and test set. The seed used for the split is not random, so this gives sets that can be used for reproducable results.

    Returns: (x_train, x_val, x_test, t_train, t_val, t_test), a tuple of scaled train and test data
    DO NOT MODIFY THIS CODE WITHOUT NOTIFYING THE REST OF THE GROUP
    """

    from sklearn.preprocessing import scale
    from sklearn.model_selection import train_test_split

    # Read in all files
    frames = []

    def process_file(filename, **kwargs):
        df = pd.read_csv(filename)
        frames.append(df)

    _for_all_files(process_file, BASE_DIR_PROCESSED)

    # Combine all files and split labels and data
    app_names = _get_distinct_applications(BASE_DIR_PROCESSED)
    app_types = _get_distinct_types(BASE_DIR_PROCESSED)
    complete = pd.concat(frames)
    labelsName = complete[app_names]
    labelsType = complete[app_types]
    data = complete.drop(app_names, axis=1).drop(app_types, axis=1)

    # Scale all columns to standard deviation of 1 and mean of 0
    for i in range(0, len(data.columns)):
        if data.columns[i] is not "VPN":
            data.iloc[:, i] = scale(data.iloc[:, i])

    # Transform data to numpy
    X = data.to_numpy()
    tName = labelsName.to_numpy()
    tType = labelsType.to_numpy()

    # Split test/validation and training set
    # X = X.reshape((-1, total_features()))
    x_train, x_test, tName_train, tName_test = train_test_split(
        X, tName, test_size=0.33, random_state=RANDOM_STATE
    )
    x_train, x_test, tType_train, tType_test = train_test_split(
        X, tType, test_size=0.33, random_state=RANDOM_STATE
    )

    # Split training and validation set, note test_size will be the (proportional) size of the validation set
    X = x_train
    t = tName_train
    x_train, x_val, tName_train, tName_val = train_test_split(
        X, t, test_size=0.25, random_state=RANDOM_STATE
    )
    t = tType_train
    x_train, x_val, tType_train, tType_val = train_test_split(
        X, t, test_size=0.25, random_state=RANDOM_STATE
    )

    return (
        x_train,
        x_val,
        x_test,
        [tType_train, tName_train],
        [tType_val, tName_val],
        [tType_test, tName_test],
    )


def _for_all_files(function, base_dir, kwargs={}, **passedkwargs):
    """
    !!! given '**passedkwargs' will be passed to given 'function'

    Goes through all files and folders in 'base_dir' and and calls the given function on those files

    Keyword arguments:
    function -- the function to run over all files
    base_dir -- directory in which to walk over files
    """
    function_outputs = {}

    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if name.endswith((".pcap_Flow.csv")):
                filename = os.path.join(root, name)
                func_output = function(
                    root=root,
                    dirs=dirs,
                    files=files,
                    name=name,
                    filename=filename,
                    **passedkwargs,
                )
                function_outputs[filename] = func_output

    return function_outputs  # , residuals


def _get_distinct_types(base_dir=BASE_DIR_PROCESSED):
    """Goes through all files and folders in 'base_dir' and returns a list of all distinct application types.

    Make sure that all files conform to the format "applicationType_applicationName_[index][a | b].pcap_Flow.csv", where [a | b] means an optional character that is 'a' or 'b'.

    Returns: List of distinct types
    """

    types = set()

    def process_file(name, **kwargs):
        match = regex.search(name)
        application_type = match.group(1)
        types.add(application_type)

    _for_all_files(process_file, base_dir)
    return sorted(list(types))


def _get_distinct_applications(base_dir=BASE_DIR_PROCESSED):
    """Goes through all files and folders in 'base_dir' and returns a list of all distinct application names.

    Make sure that all files conform to the format "applicationType_applicationName_[index][a | b].pcap_Flow.csv", where [a | b] means an optional character that is 'a' or 'b'.

    Returns: List of distinct names
    """
    names = set()

    def process_file(name, **kwargs):
        match = regex.search(name)
        application_name = match.group(2)
        names.add(application_name)

    _for_all_files(process_file, base_dir)
    return sorted(list(names))


def _create_one_hot_encoders(distinct_types, distinct_names):
    """Given a list of types and a list of names, returns a tuple of functions that can be used to one-hot encode the application types and names.

    Returns: (Function, Function), where both functions take two arguments argument: the type/name of the file and the dataframe
    """

    def typeEncoder(app_type, df):
        for t in distinct_types:
            df[t] = 1 if t == app_type else 0

    def nameEncoder(app_name, df):
        for t in distinct_names:
            df[t] = 1 if t == app_name else 0

    return (typeEncoder, nameEncoder)


def _process_files():
    """Goes through all files and folders in 'BASE_DIR_SEQUENCES' and adds labels for the application type and application name to them.
    Will also remove the features that are not required by the model.

    Make sure that all files conform to the format "applicationType_applicationName_[index][a | b].pcap_Flow.csv", where [a | b] means an optional character that is 'a' or 'b'.
    """

    encodeType, encodeName = _create_one_hot_encoders(
        _get_distinct_types(BASE_DIR_SEQUENCES),
        _get_distinct_applications(BASE_DIR_SEQUENCES),
    )

    def process_file(root, name, filename, **kwargs):
        match = regex.search(name)
        application_type = match.group(1)
        application_name = match.group(2)
        df = pd.read_csv(filename)[CSV_FEATURES]
        df["VPN"] = 0 if "NonVPN".lower() in filename.lower() else 1
        encodeType(application_type, df)
        encodeName(application_name, df)

        out_dir = root.replace(BASE_DIR_SEQUENCES, BASE_DIR_TAGGED)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df.to_csv(filename.replace(root, out_dir), index=False)

    _for_all_files(process_file, BASE_DIR_SEQUENCES)


def _sequences_to_rows():
    """Puts 5 rows that for one sequence into one row and writes it to csv in BASE_DIR_PROCESSED"""

    application_names = _get_distinct_applications(BASE_DIR_TAGGED)
    application_types = _get_distinct_types(BASE_DIR_TAGGED)

    def process_file(root, filename, **kwargs):
        df = pd.read_csv(filename)
        df_nolabels = df.drop(application_names, axis=1).drop(application_types, axis=1)
        new_columns = list(df_nolabels.columns) * SEQUENCE_LENGTH
        out_root = root.replace(BASE_DIR_TAGGED, BASE_DIR_PROCESSED)
        out_file = filename.replace(root, out_root)

        if not os.path.exists(out_root):
            os.makedirs(out_root)

        if len(df > 0):
            all_col_names = [*application_types, *application_names]
            all_cols = df[all_col_names].to_numpy()
            append_data = list(all_cols[0, :])

            for typeName in application_types:
                new_columns.append(typeName)
            for appName in application_names:
                new_columns.append(appName)

            df_matrix = df_nolabels.to_numpy()
            num_rows = int((len(df_matrix[:, 0]) / SEQUENCE_LENGTH))
            df_matrix = df_matrix.reshape(
                (
                    num_rows,
                    total_features() * SEQUENCE_LENGTH,
                ),
                order="C",
            )
            append_matrix = np.array((append_data * num_rows)).reshape(
                num_rows, len(append_data), order="C"
            )
            with open(out_file, "w") as fd:
                fd.write(",".join(new_columns) + "\n")
                np.savetxt(
                    fd,
                    np.concatenate((df_matrix, append_matrix), axis=1),
                    fmt="%s",
                    delimiter=",",
                )
        else:
            with open(out_file, "w") as fd:
                fd.write(",".join(new_columns) + "\n")

    _for_all_files(process_file, BASE_DIR_TAGGED)
