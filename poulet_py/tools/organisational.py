import datetime
import os


def check_or_create(path):
    """
    Function to check whether a folder exists.
    If it does NOT exist, it is created
    """
    folder_name = os.path.split(path)[1]
    if not os.path.isdir(path):
        print(
            f"\nFolder '{folder_name}' does NOT exist, creating it for you...\n"
        )
        os.mkdir(path)
    else:
        print(f"\nFolder '{folder_name}' exists, ready to continue...\n")
    return path


def define_folder_name(name):
    """
    Defines a folder name by ensuring it only contains letters, numbers, and underscores,
    replacing invalid characters with underscores, and prepending the current date in YYYYMMDD format.

    Parameters:
        name (str): The base name for the folder.

    Returns:
        str: The formatted folder name in the format YYYYMMDD_{name}.
    """
    name = str(name)
    # Replace invalid characters with underscores
    name = "".join(e if e.isalnum() or e == "_" else "_" for e in name)
    date = datetime.datetime.now().strftime("%Y%m%d")
    return f"{date}_{name}"
