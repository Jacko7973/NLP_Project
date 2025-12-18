# Python Code Identifier Prediction

**Notre Dame CSE 40657: Natural Language Processing**

**Final Project**

**Authors:**
- Jack O'Connor (joconn25@nd.edu)
- Brian Rabideau (brabidea@nd.edu)

## Usage

This project is intended to be run using Google Colab. To run this code directly, use the following steps...


1. Navigate to [colab.research.google.com](http://colab.research.google.com).

2. In the *Open notebook* dialoge, select **GitHub**, and enter the link to this GitHub repository (https://github.com/Jacko7973/NLP_Project). Select the `project.ipynb` notebook.

3. Create a copy of the notebook to make edits (`File > Save a Copy in Drive`)

4. You will likley not have access to the Shared Drive used to store project files. If this is the case, make a new folder in your Google Drive to store files and make the following edit to the first cell in your Jupyter notebook:

```python
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

SHARED_DRIVE = Path("/content/drive/MyDrive/<NAME OF YOUR NEW FOLDER>")
assert SHARED_DRIVE.exists(), "Drive folder not found"
```

5. Upload a copy of the Mostly Basic Python Problems dataset to your drive folder. You can do this manually or by adding the following cell to your noteboook below the previous one:

```
!wget -O /content/drive/MyDrive/<NAME OF YOUR NEW FOLDER>/mbpp.jsonl "https://github.com/google-research/google-research/raw/refs/heads/master/mbpp/mbpp.jsonl"
```
