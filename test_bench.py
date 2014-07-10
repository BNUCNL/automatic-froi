# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score

iris = load_iris()

# evaluate model with cross-validation
clf = RandomForestClassifier(n_estimators=20, criterion='gini')
scores = cross_val_score(clf, iris['data'], iris['target'])
print 'Mean accuracy with CV is ' + str(scores.mean())

# evaluate model with OOB error rate
clf = RandomForestClassifier(n_estimators=20, criterion='gini', oob_score=True)
clf.fit(iris['data'], iris['target'])
print 'OOB error rate is ' + str(clf.oob_score_)

