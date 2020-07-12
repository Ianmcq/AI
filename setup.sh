#!/usr/bin/env bash
trainingmade=true
testmade=true
if [ ! -d "Training" ]; then
 mkdir ./Training
 trainingmade=false
fi
if [ ! -d "Test" ]; then
 mkdir ./Test
 testmade=false
fi
if [ "$trainingmade" = false ]; then
 cp ./MNISTdata/train-images-idx3-ubyte.gz ./Training/train-images-idx3-ubyte.gz
 cp ./MNISTdata/train-labels-idx1-ubyte.gz ./Training/train-labels-idx1-ubyte.gz
 gunzip ./Training/train-images-idx3-ubyte.gz
 gunzip ./Training/train-labels-idx1-ubyte.gz
fi
if [ "$testmade" = false ]; then
 cp ./MNISTdata/t10k-images-idx3-ubyte.gz ./Test/t10k-images-idx3-ubyte.gz
 cp ./MNISTdata/t10k-labels-idx1-ubyte.gz ./Test/t10k-labels-idx1-ubyte.gz
 gunzip ./Test/t10k-images-idx3-ubyte.gz
 gunzip ./Test/t10k-labels-idx1-ubyte.gz
fi
