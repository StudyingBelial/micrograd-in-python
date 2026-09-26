import numpy as np

def MSE(prediction, target):
    diff = prediction - target
    return (diff ** 2).mean()

def CrossEntropy(prediction, target):
    loss = -target * prediction.log()
    return loss.sum(axis=-1).mean()

def BinaryCrossEntropy(prediction, target):
    loss = (target * prediction.log()) + (1 - target) * (1 - prediction).log()
    return - loss.mean()