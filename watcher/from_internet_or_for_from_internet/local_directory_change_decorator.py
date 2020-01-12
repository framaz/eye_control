import os


def path_change_decorator(func):
    def kek(*args, **kwargs):
        old_path = os.getcwd()
        if old_path.find("from_internet") == -1:
            new_path = os.path.join(old_path, "from_internet_or_for_from_internet")
        else:
            new_path = old_path
        os.chdir(new_path)
        result = func(*args, **kwargs)
        os.chdir(old_path)
        return result
    return kek

