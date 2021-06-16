# MCCF

Source code for AAAI2020 paper ["**Multi-Component Graph Convolutional Collaborative Filtering**"](http://www.shichuan.org/doc/77.pdf)



## Environment Settings

* Python == 3.6.9
* torchvision == 0.4.2
* numpy == 1.17.3
* scikit-learn == 0.21.3



## Parameter Settings

- epochs: the number of epochs to train
- lr: learning rate
- embed_dim: embedding dimension
- N: a parameter of L0, the default is the number of triples
- droprate: dropout rate
- batch_size: batch size for training



## Files in the folder

~~~~
MCCF/
├── run.py: training the model
├── utils/
│   ├── aggregator.py: aggregating the feature of neighbors
│   ├── l0dense.py: implementation of L0 regularization for a fully connected layer
│   ├── attention.py: implementation of the node-level attention
│   ├── encoder.py: together with aggregator to form the decomposer
│   └── combiner.py: implementation of the combiner
├── datasets/
│   ├── yelp/
│   │	├── business_user.txt
│   │   ├── preprocess.py: data preprocessing example
│   │   └── _allData.p
│   ├── amazon/ 
│   │   ├── user_item.dat
│   │   └── _allData.p
│   └── movielens/
│   	├── ub.base
│       ├── ub.test
│   	├── ua.base
│       ├── ua.test
│   	├── u5.base
│       ├── u5.test
│   	├── u4.base
│       ├── u4.test
│   	├── u3.base
│       ├── u3.test
│   	├── u2.base
│       ├── u2.test
│   	├── u1.base
│       ├── u1.test
│   	├── u.data
│       ├── u.user
│       ├── u.item
│       └── _allData.p
└── README.md
~~~~



## Data

### Input training data

* u_adj: user's purchased history (item set in training set)
* i_adj: user set (in training set) who have interacted with the item
* u_train, i_train, r_train: training set (user, item, rating)
* u_test, i_test, r_test: testing set (user, item, rating)



### Input pre-trained data

* u2e, i2e: for small data sets, the corresponding vectors in the rating matrix can be used as initial embeddings; for large data sets, we recommend using the embeddings of other models (e.g., GC-MC) as pre-training, which greatly reduces the complexity.




## Basic Usage

~~~
python run.py 
~~~



## Hyper-parameters Tuning

There are three key hyper-parameters: *number of components*, *lr* and *embed_dim*.

- number of components: [1, 2, 3, 4]
- lr: [0.0005, 0.001, 0.002, 0.0025]
- embed_dim: [8, 16, 32, 64, 128]

**HINT**: N and the sampling threshold in aggregator.py are calculated based on the dataset. Additionally, the number of epochs needs to be large enough to ensure that the model converges. According to our empirical results, generally 60+ is required, and the larger the dataset, the larger the number of epochs.

For the hyper-parameter settings of three benchmark datasets used in this paper, please refer to Section 4.4.





# Reference

```
@inproceedings{wang2020multi,
  title={Multi-component graph convolutional collaborative filtering},
  author={Wang, Xiao and Wang, Ruijia and Shi, Chuan and Song, Guojie and Li, Qingyong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={04},
  pages={6267--6274},
  year={2020}
}
```

