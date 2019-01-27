# [KDD 2009](https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Intro) Telecom Data Science Challenge

## My Results

In the end, for the training set we will have ROC-AUCs (**Gradient Boosting**):

- **Churn**: 0.73202
- **Upselling**: 0.86373
- **Appetency**: 0.82624
- **Average**: 0.80733

With these numbers, I would be around position 34 of the Fast Track Rankings. Come to think of it, I think I should have used Stacked classifiers, instead of only Gradient Boosting (GB). And GB can be deceiving sometimes, because it tends to overfit much more than Random Forests. Anyway, this performance seems acceptable as a first iteration.

## Main Files

- Notebooks:
    - `Churn Training.ipynb`
    - `Churn Testing.ipynb`
- Python Scripts:
    - `churn_hekima_v2.py` - training
    - `hekima_small_test.py` - testing

## Outline

1. **Preprocessing**
    1. Imports
    1. Opening the Data
    1. Verifying Consistency
    1. Feature Scaling
    1. Deleting Vars with too many NaNs
    1. Filling NaNs
    1. Deleting Vars with too many Categories
    1. Feature Selection with Decision Trees
1. **Modelling**
    1. Imports
    1. Train Test Split
    1. Evaluating Models' Performances (ROC-AUC)
1. **Best Model Optimization (Gradient Boosting)**
    1. Separate Optimization
    1. Global Optimization
    1. Final Model's ROC-AUC

### Short Notes

The targets are very sparse, which causes many problems, since putting everything to zeros should be good enough to get very high levels of accuracy.


To mitigate this effect, besides using ROC-AUC, FPR and TPR criteria, we could resample the positive values of the targets. However, this has not been proven very effective, neither **Subsampling** the negative values nor using **SMOTE** to oversample the positive values worked very well.


Another item to try in a later attempt is to calculate the **mutual info scores** (analogous to Pearson Correlations, but for classes), in order to get how much information even the variables with many NaNs carry. In this notebook, I've deliberately deleted variables with too many NaNs instead.


I've also tested SVMs, but, aside taking an unbearably long time to train, they do not perform very well, sitting between the Logistic Regression's and the Neural Network's performances. Also, the Neural Network has a lot of space to gain in improvements, since the number of hidden layers and their sizes still have to be optimized -- I've tried it with 4 hidden layers of 300 neurons (number of variables / 2), deeper networks would likely be better.

# Further Reading

I've also made a [post](http://fanaro.com.br/the-kdd-2009-data-science-challenge/) for my website, where I put some additional hopefully enlightening comments.
