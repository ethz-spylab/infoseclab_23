from zipfile import ZipFile
from zipp import Path
import numpy as np
import os
import sys


def validate_zip1(zip_file):
    expected_files = ["x_adv_targeted.npy", "x_adv_detect.npy", "x_adv_detect_rf.npy", "x_adv_blur.npy", "x_adv_random.npy", "x_adv_discrete.npy"]

    if os.path.basename(zip_file) != "results1.zip":
        print("WARNING: Invalid zip file name:", zip_file)
        return False

    with ZipFile(zip_file, 'r') as archive:
        for name in archive.namelist():
            if Path(root=archive, at=name).is_dir():
                print("Zip file contains a directory:", name)
                return False
            else:
                if name not in expected_files:
                    print("Unexpected file in zip:", name)
                    return False

        for name in expected_files:
            if name not in archive.namelist():
                print("WARNING: Missing file in zip:", name)
            else:
                try:
                    x = np.load(archive.open(f"x_adv_targeted.npy"))
                    if x.shape != (200, 3, 224, 224):
                        print("Invalid shape for", name)
                        return False
                    if x.dtype != np.uint8:
                        print("Invalid dtype for", name)
                        return False
                except Exception as e:
                    print("Error loading", name, e)
                    return False
    print("Zip file is valid")
    return True

def validate_zip2(zip_file):
    expected_files = ["attack_scores.npy", "chatbot.npy"]

    if os.path.basename(zip_file) != "results2.zip":
        print("WARNING: Invalid zip file name:", zip_file)
        return False

    with ZipFile(zip_file, 'r') as archive:
        for name in archive.namelist():
            if Path(root=archive, at=name).is_dir():
                print("Zip file contains a directory:", name)
                return False
            else:
                if name not in expected_files:
                    print("Unexpected file in zip:", name)
                    return False

        for name in expected_files:
            if name not in archive.namelist():
                print("WARNING: Missing file in zip:", name)
                return False

        name = "attack_scores.npy"
        try:
            x = np.load(archive.open(name))
            if x.shape != (50000,):
                print("Invalid shape for", name)
                return False
            if x.dtype != np.float32:
                print("Invalid dtype for", name)
                return False
        except Exception as e:
            print("Error loading", name, e)
            return False

        # Validate chatbot part
        name = "chatbot.npy"
        try:
            chatbot_file = np.load(archive.open(name))
            if chatbot_file.shape != (10,):
                print("Invalid shape for", name)
                return False
            for string in chatbot_file:
                if len(string) != 6 or not string.isascii() or not string.isalnum():
                    print("Invalid string in", name)
                    return False
        except Exception as e:
            print("Error loading", name, e)
            return False

    print("Zip file is valid")
    return True


if __name__ == "__main__":
    zip_file = sys.argv[1]
    N = int(sys.argv[2])

    if N == 1:
        assert validate_zip1(zip_file)
    elif N == 2:
        assert validate_zip2(zip_file)
    else:
        print("Invalid submission number:", N)
        sys.exit(1)


