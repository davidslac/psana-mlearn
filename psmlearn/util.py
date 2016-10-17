from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import numpy as np

def logTrace(hdr, msg, flag=True):
    if not flag: return
    print("TRACE %s: %s" % (hdr,msg))
    sys.stdout.flush()

def logDebug(hdr, msg, flag=True):
    if not flag: return
    print("DBG %s: %s" % (hdr,msg))
    sys.stdout.flush()

def convert_to_one_hot(labels, numLabels):
    '''converts a 1D integer vector to one hot labeleling.
    '''
    if isinstance(labels, list):
        labels = np.array(labels)
    labelsOneHot = np.zeros((len(labels), numLabels), dtype=np.int32)
    for label in range(numLabels):
        rowsToSet = np.where(labels==label)[0]
        labelsOneHot[rowsToSet,label] = 1
    assert np.sum(labelsOneHot) == len(labels), "labels must have entries not in [0,%d)" % numLabels
    return labelsOneHot

def get_confusion_matrix_one_hot(model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results,axis=1)
    assert len(predictions)==truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:,actual_class]==1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class==predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix)==len(truth)
    assert np.sum(confusion_matrix)==np.sum(truth)
    return confusion_matrix

def cmat2str(confusion_matrix, fmtLen=None):
    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    if fmtLen is None:
        fmtLen = int(math.floor(math.log(np.max(confusion_matrix),10)))+1
    fmtstr = '%' + str(fmtLen) + 'd'
    cmat_rows = []
    for row in range(confusion_matrix.shape[0]):
        cmat_rows.append(' '.join(map(lambda x: fmtstr % x, confusion_matrix[row,:])))
    return accuracy, cmat_rows

def get_best_correct_one_hot(scores, truth, label):
    '''returns a case
    '''
    predict = np.argmax(scores, axis=1)
    truth = np.argmax(truth, axis=1)
    correct = predict == truth
    correct_and_label = np.logical_and(correct, truth == label)
    orig_rows = np.arange(scores.shape[0])
    orig_rows = orig_rows[correct_and_label]
    if len(orig_rows)==0:
        return None, None
    scores = scores[correct_and_label,label]
    best_score = np.max(scores)
    row = orig_rows[np.argmax(scores)]
    return row, best_score
