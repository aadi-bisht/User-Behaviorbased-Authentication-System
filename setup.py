import argparse
import json
import subprocess
import sys
import os


def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def create_launcher():
    startup_folder = os.path.expandvars(r"%AppData%\Microsoft\Windows\Start Menu\Programs\Startup")
    launcher_path = os.path.join(startup_folder, "launcher.pyw")
    script_path = f"{os.getcwd()}\\controller.py"
    launcher_content = f"""import subprocess
import os

main_script = r"{script_path}"
main_script_dir = os.path.dirname(main_script)
os.chdir(main_script_dir)
subprocess.Popen(["pythonw", main_script], shell=True)
"""
    with open(launcher_path, "w") as launcher_file:
        launcher_file.write(launcher_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='mail', type=str, required=True)

    dictionary = {'email': parser.parse_args().mail}
    json_object = json.dumps(dictionary, indent=4)
    with open('saved_files/config.json', 'w+') as file:
        file.write(json_object)

    required_packages = [
        "pandas",
        "keyboard",
        "numpy",
        "scikit-learn",
        "openpyxl",
        "tensorflow",
    ]

    for module_name in required_packages:
        install_and_import(module_name)
    create_launcher()
    subprocess.Popen(["pythonw", f"{os.getcwd()}\\controller.py"], shell=True)


if __name__ == "__main__":
    main()
