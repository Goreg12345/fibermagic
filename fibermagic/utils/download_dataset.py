import requests
import zipfile
from io import BytesIO


def download(project):
    if project == "arena-data":
        url = "https://fibermagic.org/datasets/arena-data.zip"
    elif project == "tutorial":
        url = "https://fibermagic.org/datasets/tutorial.zip"
    elif project == "fdrd2xadora_PR_Pilot":
        url = "https://fibermagic.org/datasets/fdrd2xadora_PR_Pilot.zip"
    elif project == "fdrd2xadora_PR_NAcc":
        url = "https://fibermagic.org/datasets/fdrd2xadora_PR_NAcc.zip"
    elif project == "tutorial-1":
        url = "https://fibermagic.org/datasets/tutorial-1.zip"
    elif project == "tutorial-2":
        url = "https://fibermagic.org/datasets/tutorial-2.zip"
    else:
        print("Requested dataset not found!")
        return

    print("Downloading started")
    req = requests.get(url)
    print("Downloading Completed, Extracting...")

    # extracting the zip file contents
    zip_file = zipfile.ZipFile(BytesIO(req.content))
    zip_file.extractall(".")
