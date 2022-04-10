import requests
import zipfile
from io import BytesIO


def download(project):
    if project == "arena-data":
        url = "http://georglange.com/fibermagic/arena-data.zip"
    elif project == "tutorial":
        url = "http://georglange.com/fibermagic/tutorial.zip"
    elif project == "fdrd2xadora_PR_Pilot":
        url = "http://georglange.com/fibermagic/fdrd2xadora_PR_Pilot.zip"
    elif project == "fdrd2xadora_PR_NAcc":
        url = "http://georglange.com/fibermagic/fdrd2xadora_PR_NAcc.zip"
    else:
        print("Requested dataset not found!")
        return

    print("Downloading started")
    req = requests.get(url)
    print("Downloading Completed, Extracting...")

    # extracting the zip file contents
    zip_file = zipfile.ZipFile(BytesIO(req.content))
    zip_file.extractall(".")
