export KAGGLE_USERNAME=charleshetterich
export KAGGLE_KEY=f5b91e2425ec647bb4ae7218087c47fc
export DATA_DIR=/root/data

# Install dependencies
apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y
pip install -r requirements.txt

# Download data
mkdir -p $DATA_DIR
kaggle competitions download -c blood-vessel-segmentation -p $DATA_DIR
unzip $DATA_DIR/*.zip -d $DATA_DIR
rm $DATA_DIR/*.zip
python -m src.util._data.setup