#/bin/bash
#cifar-url = "https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz"
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_url>"
    exit 1
fi
DATA_DIR="/share/landseer/Landseer/data" 

URL=$1

FILENAME=$(basename "$URL")
DATASET_PATH=$DATA_DIR/datasets

cd "$DATASET_PATH"
echo "Downloading dataset from $URL..."
wget -c "$URL" -O "$FILENAME"

if [[ "$FILENAME" == *.zip ]]; then
    echo "Extracting zip file..."
    unzip -o "$FILENAME" 
elif [[ "$FILENAME" == *.tar.gz ]] || [[ "$FILENAME" == *.tgz ]]; then
    echo "Extracting tar.gz file..."
    tar -xvzf "$FILENAME"
elif [[ "$FILENAME" == *.tar ]]; then
    echo "Extracting tar file..."
    tar -xvf "$FILENAME"
else
    echo "Downloaded file is not a recognized archive format."
fi

echo "Download and extraction complete!"
rm "$FILENAME"