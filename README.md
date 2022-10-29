# Real-world Video Anomaly Detection by Extracting Salient Features in Videos
## Requirements
* Python : 3.7.11
* Numpy : 1.19.5
* Pytorch : 1.9.1  
* scikit-learn : 0.24.2
* tqdm :　4.46.0
  
## Preparation
### 1. Download the extracted I3D features for UCF-crime.
We use the same features as in [this implementation](https://github.com/tianyu0207/RTFM). 

* Download **UCF-Crime train i3d Google drive** and put them in the `features/UCF-Train` directory.
  * You can skip downloading this training data, if you just want to reproduce the results of the paper using the pre-trained model.
* Download **UCF-Crime test i3d Google drive** and put them in the `features/UCF-Test` directory.
<pre>
.
|-- dataset.py
|-- features
|    |-- UCF-Test
|    |  |-- Abuse028_x264_i3d.npy
|    |  |-- Abuse030_x264_i3d.npy
|    |  |-- Arrest001_x264_i3d.npy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
|    |-- UCF-Train
|    |  |-- Abuse001_x264_i3d.npy
|    |  |-- Abuse002_x264_i3d.npy
|    |  |-- Abuse003_x264_i3d.npy
|    |  |-- Abuse004_x264_i3d.npy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
|-- test.py
|-- list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
</pre>

### 2. Reshape the features and make the list of them.
```
python preformer_feature.py
```
## Reproducing the results in the paper
Reproduce the results in Table 3, which shows the frame-level AUC performance on UCF-Crime dataset.

For  <img src="https://latex.codecogs.com/svg.image?d_a=64,r=3&space;" title="d_a=64,r=3 " />

```
python test.py --da 64 --r 3 --seed 9111 --test-split-size 28
```

For  <img src="https://latex.codecogs.com/svg.image?d_a=128,r=7&space;" title="d_a=128,r=7 " /> 

```
python test.py --da 128 --r 7 --seed 9111 --test-split-size 28
```

| model | AUC(%)|
|----|----|
|Sultani et al.|75.41|
|GCN-Anomaly|82.12|
|RTFM|84.30|
|Ours( <img src="https://latex.codecogs.com/svg.image?d_a=64,r=3&space;" title="d_a=64,r=3 " /> )|84.74|
|Ours( <img src="https://latex.codecogs.com/svg.image?d_a=128,r=7&space;" title="d_a=128,r=7 " /> )|84.91|


## Traininig from scratch
```
python train.py --da 64 --r 3 --seed 1111
```
## Bibtex
@INPROCEEDINGS{9897864,
  author={Watanabe, Yudai and Okabe, Makoto and Harada, Yasunori and Kashima, Naoji},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)}, 
  title={Real-World Video Anomaly Detection by Extracting Salient Features}, 
  year={2022},
  volume={},
  number={},
  pages={891-895},
  doi={10.1109/ICIP46576.2022.9897864}}
  
  
@article{ARXIV_WOHK22,
    author = {Yudai Watanabe and Makoto Okabe and Yasunori Harada and Naoji Kashima},
    journal = {arXiv preprint arXiv:2209.06435},
    title = {Real-world Video Anomaly Detection by Extracting Salient Features in Videos},
    year = {2022}}
