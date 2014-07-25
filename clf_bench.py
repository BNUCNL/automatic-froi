# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import time

# modules for data preparation
import multiprocessing as mps
import functools

# modules for modle training and testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

import autoroilib as arlib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

#-- 5-folds cross-validation
cv_num = 5

# split all subjects into 5 folds
subj_group = arlib.split_subject(sessid, cv_num)
arlib.save_subject_group(subj_group, data_dir)

# data preparation for cross-validation
for i in range(cv_num):
    cv_dir = os.path.join(data_dir, 'cv_' + str(i))
    os.system('mkdir ' + cv_dir)
    test_sessid = subj_group[i]
    train_sessid = []
    for j in range(cv_num):
        if not j == i:
            train_sessid += subj_group[j]

    # generate mask and probability map
    prob_data, mask_data = arlib.make_prior(train_sessid, cv_dir)
    mask_coords = arlib.get_mask_coord(mask_data, cv_dir)

    # extract features from each subject
    pool = mps.Pool(20)
    result = pool.map(functools.partial(arlib.ext_subj_feature,
                                        mask_coord=mask_coords,
                                        prob_data=prob_data,
                                        out_dir=cv_dir), sessid)
    print result
    pool.terminate()

## Cross-validation to evaluate the performance of the model
##for i in range(cv_num):
#for i in [0]:
#    print 'CV iter - ' + str(i)
#    
#    # split data into train and test group
#    test_sessid = subj_group[i]
#    train_sessid = []
#    for j in range(cv_num):
#        if not j == i:
#            train_sessid += subj_group[j]
#    cv_dir = os.path.join(data_dir, 'cv_' + str(i))
#    print 'Load data ... '
#    train_data = arlib.get_list_data(train_sessid, cv_dir)
#    test_data = arlib.get_list_data(test_sessid, cv_dir)
#    # sample stats
#    #print 'Samples stats for train-dataset:'
#    #arlib.samples_stat(train_data)
#    #print 'Samples stats for test-dataset:'
#    #arlib.samples_stat(test_data)
#
#    # remove samples whose z-value is less than 2.3
#    print 'Remove samples whose z-value is less than 2.3 ...'
#    train_data_mask = train_data[:, 3] >= 2.3
#    test_data_mask = test_data[:, 3] >= 2.3
#    train_data = train_data[train_data_mask]
#    test_data = test_data[test_data_mask]
#    print 'Samples stats for train-dataset:'
#    arlib.samples_stat(train_data)
#    print 'Samples stats for test-dataset:'
#    arlib.samples_stat(test_data)
#
#    # shuffle the training data
#    sample_idx = np.arange(train_data.shape[0])
#    np.random.shuffle(sample_idx)
#    train_data = train_data[sample_idx]
#
#    # split label and feature
#    train_x = train_data[..., :-1]
#    train_y = train_data[..., -1]
#    test_x = test_data[..., :-1]
#    test_y = test_data[..., -1]
#
#    # parameters of random forest
#    n_tree = [5, 10, 20, 30, 40, 50, 60]
#    depth = [5, 10, 20, 30, 40, 50, 60]
#
#    # output matrix
#    oob_score = np.zeros((len(n_tree), len(depth)))
#    cv_score = np.zeros((len(n_tree), len(depth)))
#    ofa_dice = np.zeros((len(n_tree), len(depth)))
#    ffa_dice = np.zeros((len(n_tree), len(depth)))
#
#    for t_idx in range(len(n_tree)):
#        for d_idx in range(len(depth)):
#            p = [n_tree[t_idx], depth[d_idx]]
#            print 'Parameter: n_tree - %s; depth - %s'%(p[0], p[1])
#            #-- compare OOB eror rate and the CV error rate
#            # Cross-Validation
#            clf = RandomForestClassifier(n_estimators=p[0], criterion='gini',
#                                         max_depth=p[1], n_jobs=20)
#            scores = cross_val_score(clf, train_x, train_y)
#            cv_score[t_idx, d_idx] = scores.mean()
#            print 'Mean score with CV is ' + str(scores.mean())
#            # OOB error rate
#            clf = RandomForestClassifier(n_estimators=p[0], criterion='gini',
#                                         max_depth=p[1], n_jobs=20,
#                                         oob_score=True)
#            #tt = time.time()
#            clf.fit(train_x, train_y)
#            #print 'Model training costs %s'%(time.time() - tt)
#            oob_score[t_idx, d_idx] = clf.oob_score_
#            print 'OOB score is ' + str(clf.oob_score_)
#            print 'Dice coefficient:'
#            oob_functions = clf.oob_decision_function_
#            oob_labels = []
#            for line in oob_functions:
#                if np.all(np.isnan(line)):
#                    oob_labels.append(np.nan)
#                else:
#                    oob_labels.append(line.argmax())
#            oob_labels = np.array(oob_labels)
#            nan_num = np.sum(np.isnan(oob_labels))
#            print 'NaN number: %d'%nan_num
#            label_idx = [1, 3]
#            oob_idx = [1, 2]
#            for j in range(2):
#                P = oob_labels == oob_idx[j]
#                T = train_y == label_idx[j]
#                dice_val = arlib.dice(T, P)
#                print 'Dice for label %s: %f'%(label_idx[j], dice_val)
#                print 'Sample number: %d'%P.sum()
#                if j:
#                    ffa_dice[t_idx, d_idx] = dice_val
#                else:
#                    ofa_dice[t_idx, d_idx] = dice_val
#            print '-----------------------'
#    np.savez('out_data.npz', cv_score, oob_score, ffa_dice, ofa_dice)
#    # model defination
#    clf = RandomForestClassifier(n_estimators=50, max_depth=20,
#                                 criterion='entropy', max_features='sqrt',
#                                 bootstrap=True, n_jobs=10)
#    # model training
#    tt = time.time()
#    clf.fit(train_x, train_y)
#    print 'Model training costs %s'%(time.time() - tt)
#    print 'Feature importance: ', clf.feature_importances_
#
#    # model testing
#    predicted_y = clf.predict(test_x)
#    # prob_predicted_y = clf.predict_proba(test_x)
#
#    # model evaluation
#    for l in range(1, 6, 2):
#        print 'Dice coefficient for label ' + str(l),
#        true_bool = test_y == l
#        predicted_bool = predicted_y == l
#        dice_coef = arlib.dice(true_bool, predicted_bool)
#        print dice_coef
    
