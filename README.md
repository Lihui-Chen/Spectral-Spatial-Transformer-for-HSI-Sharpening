# Spectral-Spatial-Transformer-for-HSI-Sharpening
The official codes of "Spectral-spatial Transformer for Hyperspectral Image Sharpening" published in IEEE Transactions on Neural Networks and Learning Systems, 2023.


A demo of the comand line for training.
```
python train.py -net_arch sst -opt options/train/train_HyperALi_CB.yml -log_dir vscode_debug/ -setmode LRHRRAM -batch_size 8 -patch_size 18 -convDim 120 -patchSize 1 -numLayers 14 -poolSize 4 -numHeads 8 -ksize 9 -lr_scheme warm_up -learning_rate 0.0002 -warmUpEpoch 0 
```
-opt is the path of option file for training or test.  
-setmode is the type of loading dataset, where LRHRRAM refers to loading the dataset to the RAM.  
-batch_size is the batch size for training.  
-patch_size is the patch size of LRHS image for training.  
-convDim 120 is the channel numbers, i.e., D in the paper, of convolution for SST.  
-numLayers is the number of SpeT/SpaT for SST.

For more descriptions of the args and setting please see the [train_HyperALi_CB.yml](options/train/train_HyperALi_CB.yml) under options/train/train_HyperALi_CB.yml

The folder tree for datasets is suggested as follows,
```
dataset
    --dataset1
        --train
            --LR
            --REF
            --GT
            --REF_FR
        --valid
            --LR
            --REF
            --GT
            --REF_FR
        --test
            --LR
            --REF
            --GT
            --REF_FR
    ...
    --datasetN
        --train
            --LR
            --REF
            --GT
            --REF_FR
        --valid
            --LR
            --REF
            --GT
            --REF_FR
        --test
            --LR
            --REF
            --GT
            --REF_FR
```

If you find the codes helpful in your research, please kindly cite the following paper,
```latex
@article{Chen2023SST,
	title={Band-Independent Encoder-Decoder Network for Pan-Sharpening of Remote Sensing Images},
	author={Chen, L. and Vivone, G. and Qin, J. and Chanussot, J. and Yang, X.},
	journal="IEEE Transactions on Neural Networks and Learning Systems",
	year={2023},
	publisher={IEEE}
}
```
