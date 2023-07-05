# CREST: Coresets for Stochastic Gradient Descent
This is the official repository for the paper 
[Towards Sustainable Learning: Coresets for Data-efficient Deep Learning](https://arxiv.org/abs/2306.01244) (ICML 2023) by [Yu Yang](https://sites.google.com/g.ucla.edu/yuyang/home), [Hao Kang](https://github.com/HaoKang-Timmy), and [Baharan Mirzasoleiman](https://baharanm.github.io/). 


## Abstract
To improve the efficiency and sustainability of learning deep models, we propose CREST, the first scalable framework with rigorous theoretical guarantees to identify the most valuable examples for training non-convex models, particularly deep networks. To guarantee convergence to a stationary point of a non-convex function, CREST models the non-convex loss as a series of quadratic functions and extracts a coreset for each quadratic sub-region. In addition, to ensure faster convergence of stochastic gradient methods such as (mini-batch) SGD, CREST iteratively extracts multiple mini-batch coresets from larger random subsets of training data, to ensure nearly-unbiased gradients with small variances. Finally, to further improve scalability and efficiency, CREST identifies and excludes the examples that are learned from the coreset selection pipeline. Our extensive experiments on several deep networks trained on vision and NLP datasets, including CIFAR-10, CIFAR-100, TinyImageNet, and SNLI, confirm that CREST speeds up training deep networks on very large datasets, by 1.7x to 2.5x with minimum loss in the performance. By analyzing the learning difficulty of the subsets selected by CREST, we show that deep models benefit the most by learning from subsets of increasing difficulty levels.


## Installation
This code is tested with Python 3.8.8 and PyTorch 1.9.1 with CUDA 11.5.

To install the required packages, run
```
pip install -r requirements.txt
```

## Usage
```
python crest_train.py
```

`--dataset`: The dataset to use. (default: `cifar10`)
- `cifar10`: CIFAR-10 dataset
- `cifar100`: CIFAR-100 dataset
- `tinyimagenet`: TinyImageNet dataset

:warning: The TinyImageNet dataset is not included in this repository. Please download the dataset from [here](https://www.kaggle.com/c/tiny-imagenet).

`--data_dir`: The directory to store the dataset. (default: `./data`)

`--arch`: The model architecture to use. (default: `resnet20`)
- `resnet20`: ResNet-20 model for CIFAR-10
- `resnet18`: ResNet-18 model for CIFAR-100
- `resnet50`: ResNet-50 model for TinyImageNet

`--seed`: The random seed to use. (default: `0`)

`--selection_method`: The data selection method to use. (default: `random`)

`--train_frac`: The fractrion of training steps to use compared to full training. (default: `0.1`)

### Hyperparameter Tuning
To achieve the most cost-efficient training, the following hyperparameters need be tuned for different datasets and models.
- `--check_thresh_factor`: The fraction of the training loss to use as the threshold for coreset selection. (default: `0.1`)

## Adding New Datasets
To add a new dataset, you need to add the dataset loading code in `datasets/dataset.py`. 
Then, you need to add the dataset name to the choices of `--dataset` argument in `utils/arguments.py`.

## Adding New Models
To add a new model, you need to create a new file in `models/` folder, which contains the model class. 
For example, `models/resnet.py` contains the class `ResNet20`, which is the ResNet-20 model. 
Then, you need to add the model name to the choices of `--arch` argument in `utils/arguments.py`.

## Adding New Data Selection Methods
To add a new data selection method, you need to create a new file in `trainers/` folder, which contains a subclass of `SubsetTrainer` class defined in `trainers/subset_trainer.py`.
For example, `trainers/random_trainer.py` contains the class `RandomTrainer`, which is the trainer for random selection. 
Then, you need to add the method name to the choices of `--selection_method` argument in `utils/arguments.py`.

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{yang2023towards,
  title={Towards Sustainable Learning: Coresets for Data-efficient Deep Learning},
  author={Yang, Yu and Kang, Hao and Mirzasoleiman, Baharan},
  booktitle={In Proceedings of the 40th International Conference on Machine Learning},
  year={2023}
}
```

## Acknowledgement
The code is based on [Craig](https://github.com/baharanm/craig) and [AdaHessian](https://github.com/amirgholami/adahessian).

## Disclaimer and Contact
The current version of the code is refactored for better readability and extensibility,
but it is still under the process of testing and optimization.

If you have any questions or suggestions, please contact Yu Yang (yuyang@cs.ucla.edu).
