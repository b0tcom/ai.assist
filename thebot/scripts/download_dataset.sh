#!/bin/bash
# Downloads and prepares the Roboflow dataset.
echo "Downloading dataset..."
curl -L "https://universe.roboflow.com/ds/ICmsOgXtkY?key=JojSX4jWuJ" > roboflow.zip
echo "Unzipping dataset..."
unzip roboflow.zip -d dataset
rm roboflow.zip
echo "Dataset ready in the 'dataset/' directory."