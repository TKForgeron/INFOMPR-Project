import pandas as pd
import os
import re

BASE_DIR = "data/CSV/"
BASE_DIR_PROCESSED = "data/CSV-Labelled/"
CSV_FEATURES = ["Src Port", "Dst Port", "Protocol"]

regex = re.compile("^([^_]*)_(.*)_\d*(.*)\.pcap_Flow.csv")

# File name format:
# applicationType_applicationName_index[a | b].pcap_Flow.csv


def total_features():
    """Returns the total number of features that will be in every csv file"""
    return (
        len(CSV_FEATURES)
        + len(_get_distinct_types())
        # Note applications is not a feature, as this is what we try to find out!
    )


def total_labels():
    """Returns the total number of labels that will be in every csv file"""
    return len(_get_distinct_applications())


# # Remove constant columns
# for i in reversed(range (0, 784)):
#     if min(digits_train[:,i]) == max(digits_train[:,i]):
#         digits_train = np.delete(digits_train, i, 1)
#         digits_test = np.delete(digits_test, i, 1)

# scaler = StandardScaler()
# digits_train_scaled = scaler.fit_transform(digits_train)
# digits_test_scaled  = scaler.transform(digits_test)


def get_train_test_set():
    """Processes the data and splits it into a scaled train and test set. The seed used for the split is not random, so this gives sets that can be used for reproducable results.

    Returns: (x_train, x_test, t_train, t_test), a tuple of scaled train and test data
    DO NOT MODIFY THIS CODE WITHOUT NOTIFYING THE REST OF THE GROUP
    """

    from sklearn.preprocessing import scale
    from sklearn.model_selection import train_test_split

    _process_files()

    # Read in all files
    frames = []

    def process_file(filename, **kwargs):
        df = pd.read_csv(BASE_DIR_PROCESSED + filename)
        frames.append(df)

    _for_all_files(process_file)

    # Combine all files and split labels and data
    complete = pd.concat(frames)
    labels = complete[_get_distinct_applications()]
    data = complete.drop(_get_distinct_applications(), axis=1)

    # Scale all columns
    types = _get_distinct_types()
    for col in data.columns:
        if col not in types:
            data[col] = scale(data[col])

    # Transform data to numpy
    X = data.to_numpy()
    t = labels.to_numpy()

    X = X.reshape(-1, total_features())
    x_train, x_test, t_train, t_test = train_test_split(
        X, t, test_size=0.33, random_state=42
    )

    return (x_train, x_test, t_train, t_test)


def _for_all_files(function):
    """Goes through all files and folders in 'BASE_DIR' and and calls the given function on those files

    Keyword arguments:
    function -- the function to run over all files
    """
    for root, dirs, files in os.walk(BASE_DIR):
        for name in files:
            if name.endswith((".pcap_Flow.csv")):
                filename = os.path.join(root, name)
                function(
                    root=root, dirs=dirs, files=files, name=name, filename=filename
                )


def _get_distinct_types():
    """Goes through all files and folders in 'BASE_DIR' and returns a list of all distinct application types.

    Make sure that all files conform to the format "applicationType_applicationName_index[a | b].pcap_Flow.csv", where [a | b] means an optional character that is 'a' or 'b'.

    Returns: List of distinct types
    """

    types = set()

    def process_file(name, **kwargs):
        match = regex.search(name)
        application_type = match.group(1)
        types.add(application_type)

    _for_all_files(process_file)
    return sorted(list(types))


def _get_distinct_applications():
    """Goes through all files and folders in 'BASE_DIR' and returns a list of all distinct application names.

    Make sure that all files conform to the format "applicationType_applicationName_index[a | b].pcap_Flow.csv", where [a | b] means an optional character that is 'a' or 'b'.

    Returns: List of distinct names
    """
    names = set()

    def process_file(name, **kwargs):
        match = regex.search(name)
        application_name = match.group(2)
        names.add(application_name)

    _for_all_files(process_file)
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
    """Goes through all files and folders in 'BASE_DIR' and adds labels for the application type and application name to them.
    Will also remove the features that are not required by the model.

    Make sure that all files conform to the format "applicationType_applicationName_index[a | b].pcap_Flow.csv", where [a | b] means an optional character that is 'a' or 'b'.
    """

    encodeType, encodeName = _create_one_hot_encoders(
        _get_distinct_types(), _get_distinct_applications()
    )

    def process_file(root, name, filename, **kwargs):
        match = regex.search(name)
        application_type = match.group(1)
        application_name = match.group(2)
        df = pd.read_csv(filename)[CSV_FEATURES]
        # df["Label"] = application_type
        # df["Application"] = application_name
        encodeType(application_type, df)
        encodeName(application_name, df)

        if not os.path.exists(BASE_DIR_PROCESSED + root):
            os.makedirs(BASE_DIR_PROCESSED + root)
        df.to_csv(BASE_DIR_PROCESSED + filename, index=False)

    _for_all_files(process_file)


if __name__ == "__main__":
    distinct_types = _get_distinct_types()
    distinct_applications = _get_distinct_applications()
    print(distinct_types, distinct_applications)
    _process_files()
