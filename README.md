# uNet_reliefmap
Train a u-net with Reliefmap

## DATA
In 'dbRelief' they are all reliefMap and labeled image for train u-net (suffixed by '_GT'). They are generated using this repository : https://github.com/FlorianDelconte/TLDDC.git
In 'dbRelief/test/' they are reliefMap for testing u-net after training.

## DEPENDENCIES
install tensorflow/openCV/tensorflow addon in virtual env via pip : 
```
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow
pip install tensorflow-addons
pip install opencv-python
```

## TRAIN a u-net with our DATA
The first step is to extract thumbnails of size 320*320 from the reliefMap. We made a bash/python script for that. Here is the command to extract the training pairs and distribute them in dbRelief/thumbnail/examples/ and dbRelief/thumbnail/labels/.
```
./train-valid_cut_All.sh dbRelief/ dbRelief/thumbnails/
```
we allow the user to do a cross validation with 5 fold. The second step is to distribute the data in the  5 folds. Here is the command to do that:
```
python3 kfold_split.py
```
Finaly train the 5 models with this command :
```
python3 train_kfold.py
```
## TEST a u-net with test DATA
You can test visualy the training using this command :
```
python3 predict.py pathToReliefMapTest PathToModel treshold
```