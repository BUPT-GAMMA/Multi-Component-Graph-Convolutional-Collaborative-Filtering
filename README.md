[toc]

# MCCF

Source code for AAAI2020 paper ["**Multi-Component Graph Convolutional Collaborative Filtering**"](https://arxiv.xilesou.top/abs/1911.10699)



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
│   │		├── business_user.txt
│   │   ├── preprocess.py: data preprocessing example
│   │   └── _allData.p
│   ├── amazon/ 
│   │   ├── user_item.dat
│   │   └── _allData.p
│   └── movielens/
│   		├── ub.base
│       ├── ub.test
│   		├── ua.base
│       ├── ua.test
│   		├── u5.base
│       ├── u5.test
│   		├── u4.base
│       ├── u4.test
│   		├── u3.base
│       ├── u3.test
│   		├── u2.base
│       ├── u2.test
│   		├── u1.base
│       ├── u1.test
│   		├── u.data
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

HINT: the sampling thresholds in aggregator.py change with dataset.




# Reference

```
@article{wang2019multi,
  title={Multi-Component Graph Convolutional Collaborative Filtering},
  author={Wang, Xiao and Wang, Ruijia and Shi, Chuan and Song, Guojie and Li, Qingyong},
  journal={arXiv preprint arXiv:1911.10699},
  year={2019}
}
```