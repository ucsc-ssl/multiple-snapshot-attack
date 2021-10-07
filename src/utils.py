import os

def get_csv_files(dir):
    """
    Returns sorted, absolute filepaths to all csv files in the directory.

    dir: the directory to find csv's in
    """
    files = []
    for f in os.listdir(dir):
        if f.endswith(".csv"):
            files += [f]

    return sorted(files)