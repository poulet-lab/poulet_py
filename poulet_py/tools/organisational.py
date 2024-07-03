import os

def check_or_create(path):
    """
    Function to check whether a folder exists.
    If it does NOT exist, it is created
    """
    folder_name = os.path.split(path)[1]
    if not os.path.isdir(path):
        print(f"\nFolder '{folder_name}' does NOT exist, creating it for you...\n")
        os.mkdir(path)
    else:
        print(f"\nFolder '{folder_name}' exists, ready to continue...\n")
    return path
