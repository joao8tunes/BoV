## BoV
Bag of Vectors (BoV) is a text representation technique based in vector space model. More precisally, this BoV based script use a pre-trained word embedding model to generate an unique vector representation to each document, calculating the arithmetic mean of database words's vector representations found in model. The output is a matrix, where rows are the documents ids, and columns are the dimensions values to each document, i.e., the centroid generated from model term vectors.
> Generating a BoV based text representation matrix:
```
python3 BoV.py --n_gram 1 --model models/Google/GoogleVectors_300.txt --input in/db/ --output out/BoV/txt/
```
> Converting a Doc-Dimension matrix to Arff file (Weka):
```
python3 Bag2Arff.py --token - --input out/Bag/txt/ --output out/Bag/arff/
```


### Related scripts
* [BoV.py](https://github.com/joao8tunes/BoV/blob/master/BoV.py)
* [Bag2Arff.py](https://github.com/joao8tunes/Bag2Arff/blob/master/Bag2Arff.py)


### Assumptions
These scripts expect a database folder following an specific hierarchy like shown below:
```
in/db/                 (main directory)
---> class_1/          (class_1's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> class_2/          (class_2's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> ...
```


### Observations
All generated files use *TAB* character as a separator.


### Requirements installation (Linux)
> Python 3 + PIP installation as super user:
```
apt-get install python3 python3-pip
```
> NLTK installation as normal user:
```
pip3 install -U nltk
```


### See more
Project page on LABIC website: http://sites.labic.icmc.usp.br/Ms-Thesis_Antunes_2018