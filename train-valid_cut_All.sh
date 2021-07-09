#!/bin/bash
echo "Total Arguments:" $#
inputDirectory=$1
outputDirectoryName=$2

#create valid and training dirs.
echo "CREATE TREE DIRECTORY FOR DEEP..."
#mkdir -p ${outputDirectoryName}/train/input/
#mkdir -p ${outputDirectoryName}/train/output/
#mkdir -p ${outputDirectoryName}/valid/input/
#mkdir -p ${outputDirectoryName}/valid/output/

echo "LOOP ON INPUT DIRECTORY FILES AND CUT/FILL DEEP-DIRECTORY..."
#loop on the relief map to cut
for fileGT in $inputDirectory*_GT*
do
  fileimg=$(echo $fileGT | sed s/_[^\.]*//g)
  python3 train-valid_cut.py -i $fileimg -g $fileGT -o $outputDirectoryName
  echo "python3 train-valid_cut.py -i" $fileimg "-g" $fileGT "-o" $outputDirectoryName
done
