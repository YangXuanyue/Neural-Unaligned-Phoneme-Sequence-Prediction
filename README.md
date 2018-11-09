# Neural-Unaligned-Phoneme-Sequence-Prediction
This is my code for Homework 3 of [(CMU 11785) Introduction to Deep Leaning](http://deeplearning.cs.cmu.edu/). Here is the link to the [Kaggle Competition](https://www.kaggle.com/c/Fall-11-785-homework-3-part-2), where I rank [__1__/216](https://www.kaggle.com/c/Fall-11-785-homework-3-part-2/leaderboard).

## Requirements
 - Data should be fetched from [Kaggle](https://www.kaggle.com/c/Fall-11-785-homework-3-part-2/data) and stored in the [`data/`](https://github.com/YangXuanyue/Neural-Unaligned-Phoneme-Sequence-Prediction/tree/master/data) directory
 - [Pytorch(>=0.4)](https://pytorch.org/), with Cuda(>=9.0) and Python(>=3.6)
 - [python-Levenshtein](https://pypi.org/project/python-Levenshtein/)
 - [PyTorch bindings for Warp-ctc](https://github.com/SeanNaren/warp-ctc)
 - [ctcdecode](https://github.com/parlance/ctcdecode)
 - I also used [Salesforce's implementation of `weight_drop`](https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py)

