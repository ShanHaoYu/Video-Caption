# Video-Caption
The second assignment of ADLxMLDS course, NTU 2017 Fall.

### Package : 
Tensorflow,  Numpy,  Pandas and other python standard liberary

### Techniques:
`Seq2seq` &nbsp; ` s2vt`  &nbsp;` LSTM` &nbsp;  ` attention` &nbsp; 

## How to use this project
1. Use extract_feature.py to extract ".avi" to ".npy" on assigned file directory.
```
python extract_feature.py [input directory] [output directory] [model] [# of frames]
```
example:
python extract_features.py -i ./Data/peer_review/video -o ./Data/peer_review/feat -m vgg19 -b 80

2. for complete testing model 
```
python test.py ./Data "s2vt" prediction.txt peer_review
```

### Result:
Input a video:<br>

![Alt Text](https://github.com/ShanHaoYu/Video-Caption/blob/master/video.gif)
* Video from test dataset folder

Output text:
```
a man is pouring water from a plastic container. 
```
