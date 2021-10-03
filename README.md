# Siamese-Network
Siamese Network for similarity and ranking
    
    This repo contains Basic Simese Network on MNIST Dataset. 
            .
        ├── README.md
        ├── models
        │   ├── __init__.py
        │   └── simese_network.py
        ├── output
        │   ├── plot_binary.png
        │   └── siamese_model
        │       ├── assets
        │       ├── imagePair.png
        │       ├── keras_metadata.pb
        │       ├── saved_model.pb
        │       └── variables
        │           ├── variables.data-00000-of-00001
        │           └── variables.index
        ├── train.py
        └── utils
            ├── __init__.py
            ├── config.py
            ├── createMontagePair.py
            ├── imagePairGenerator.py
            └── utility.py


## Training Positive and Negative Pairs
    ImagePairGenerator class generates postive and negative pairs. 
    see below sample pairs



![ImagePair](output/imagePair.png?raw=true "Title")

        
## Traing and Validation Loss Plot:

![binary_Siamese](output/plot_binary.png?raw=true "plot")


    Training : 
        python train.py

    Inference :
        python inference.py --inputDir Samples/


## Inference Result

![Result1](output/similarity/result/result0.png?raw=true "result1")


![Result2](output/similarity/result/result4.png?raw=true "result4")



![Result3](output/similarity/result/result9.png?raw=true "result9")

