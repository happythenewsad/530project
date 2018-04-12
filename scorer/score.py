#!/usr/bin/env python3
import json
import math
import os.path
import sys
from sklearn.metrics import f1_score

[_, reference_file, submission_file] = sys.argv

truth_values = json.load(open(reference_file, 'r'))
submission = json.load(open(submission_file, 'r'))

observed = 0
correct = 0
total = len(truth_values.keys())
errors = []
#tp = 0
#fp = 0
#tn = 0
#fn = 0

yhats = []
truths = []
print(len(truth_values), 'entries in reference file')

for reference_id in truth_values.keys():
    if reference_id in submission.keys():
        print('matching entry:', reference_id)
        observed += 1
        try:
     
            yhat, confidence = submission[reference_id]
            yhats.append(yhat)
            truths.append(truth_values[reference_id])

        except ValueError:
            print('   Each entry should be a list of two values - [veracity, confidence]')
            print('   veracity is one of "true" or "false"; confidence is a float from 0..1.')
            print('   This entry was:', submission[reference_id], ',  for document key', reference_id)
            sys.exit('-- error: data format: stopping')

        if yhat == truth_values[reference_id]:
            correct += 1
            errors.append((1 - confidence) ** 2)
            """
            if yhat == 'true':
                tp += 1
            elif yhat == 'false':
                tn += 1
            """
        elif truth_values[reference_id] == 'unverified':
            errors.append((confidence) ** 2)
            """
            if yhat == 'true':
                fp += 1
            elif yhat == 'false':
                fn += 1
            """

        else:
            errors.append(1.0)
            """
            if yhat == 'true':
                fp += 1
            elif yhat == 'false':
                fn += 1
            """
    else:
        print('unmatched entry:', reference_id, '-- no reference value for this document')

score = correct / total
rmse = math.sqrt(sum(errors) / len(errors))


#precision = tp / float(tp + fp)
#recall = tp / float(tp + fn)
#fscore = (2 * precision * recall) / float(precision + recall)

print(observed, 'matched entries in submission')
print(total, 'entries in reference file')

print('veracity accuracy:', score)
print('Micro F1:', f1_score(truths, yhats, average='micro'))
print('confidence rmse:  ', rmse)
