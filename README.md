# CS434_Final
Final Project for CS434
## Preprocessor
Changed command line argument, now:
```
python preproc.py <data_set_path> <word_vectors_path>
```
Expects a .txt, e.g. ```glove.6B.300.txt```.  Writes pickles.

## Training
Simple, if directory structure unchanged due to hardcoded paths.
```
python train.py
```
Saves model and weights.  Uses time of day to create name.

## Results
Seen in:
![alt text][results]

## Sources
I borrowed *alot* of how to handle nlp data from [this tutorial](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) on Keras.

[results]: https://github.com/samtjacobs/CS434_ML_proj/blob/master/val_acc80.png "Results of 10 epochs and ~300k samples"