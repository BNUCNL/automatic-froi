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
from sklearn.metrics import accuracy_score

import autoroilib as arlib
from mypy import math as mymath

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data', 'cv')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

#-- 5-folds cross-validation
cv_num = 5

# split all subjects into 5 folds
#subj_group = arlib.split_subject(sessid, cv_num)
#arlib.save_subject_group(subj_group, data_dir)
#
## data preparation for cross-validation
#for i in range(cv_num):
#    cv_dir = os.path.join(data_dir, 'cv_' + str(i))
#    os.system('mkdir ' + cv_dir)
#    test_sessid = subj_group[i]
#    train_sessid = []
#    for j in range(cv_num):
#        if not j == i:
#            train_sessid += subj_group[j]
#
#    # generate mask and probability map
#    prob_data, mask_data = arlib.make_prior(train_sessid, cv_dir)
#    mask_coords = arlib.get_mask_coord(mask_data, cv_dir)
#
#    # extract features from each subject
#    pool = mps.Pool(20)
#    result = pool.map(functools.partial(arlib.ext_subj_feature,
#                                        mask_coord=mask_coords,
#                                        prob_data=prob_data,
#                                        out_dir=cv_dir), sessid)
#    pool.terminate()

##-- Cross-validation to select parameters
## split all subjects into 5 folds
#subj_group = arlib.split_subject(sessid, cv_num)
#for i in range(cv_num):
#    print 'CV iter - ' + str(i)
#    cv_dir = os.path.join(data_dir, 'cv_' + str(i))
#    
#    # split data into training and test group
#    test_sessid = subj_group[i]
#    # load test data
#    test_data = arlib.get_list_data(test_sessid, cv_dir)
#
#    train_sessid = []
#    for j in range(cv_num):
#        if not j == i:
#            train_sessid += subj_group[j]
#    # split training data into n folds for inner CV
#    inner_cv_group = arlib.split_subject(train_sessid, cv_num)
#
#    # parameters of random forest
#    n_tree = range(5, 70, 5)
#    depth = range(5, 70, 5)
#    
#    # output matrix
#    oob_score = np.zeros((len(n_tree), len(depth), cv_num))
#    cv_score = np.zeros((len(n_tree), len(depth), cv_num))
#    ofa_dice = np.zeros((len(n_tree), len(depth), cv_num))
#    ffa_dice = np.zeros((len(n_tree), len(depth), cv_num))
#
#    # inner CV for parameter selection
#    for inner_i in range(cv_num):
#        print 'Inner CV - %s'%(inner_i)
#        # get sessid
#        inner_test_sessid = inner_cv_group[inner_i]
#        inner_train_sessid = []
#        for inner_j in range(cv_num):
#            if not inner_j == inner_i:
#                inner_train_sessid += inner_cv_group[inner_j]
#        
#        print 'Load data ... '
#        inner_train_data = arlib.get_list_data(inner_train_sessid, cv_dir)
#        inner_test_data = arlib.get_list_data(inner_test_sessid, cv_dir)
#        ## sample stats
#        #print 'Samples stats for train-dataset:'
#        #arlib.samples_stat(train_data)
#        #print 'Samples stats for test-dataset:'
#        #arlib.samples_stat(test_data)
#
#        # split label and feature
#        train_x = inner_train_data[..., :-1]
#        train_y = inner_train_data[..., -1]
#        test_x = inner_test_data[..., :-1]
#        test_y = inner_test_data[..., -1]
#
#        for t_idx in range(len(n_tree)):
#            for d_idx in range(len(depth)):
#                p = [n_tree[t_idx], depth[d_idx]]
#                print 'Parameter: n_tree - %s; depth - %s'%(p[0], p[1])
#                #-- compare OOB eror rate and the CV error rate
#                # OOB error rate
#                clf = RandomForestClassifier(n_estimators=p[0],
#                                             criterion='gini',
#                                             max_depth=p[1],
#                                             n_jobs=20,
#                                             oob_score=True)
#                clf.fit(train_x, train_y)
#                oob_score[t_idx, d_idx, inner_i] = clf.oob_score_
#                print 'OOB score is %s'%(str(clf.oob_score_))
#                # Cross-Validation
#                clf = RandomForestClassifier(n_estimators=p[0],
#                                             criterion='gini',
#                                             max_depth=p[1],
#                                             n_jobs=20)
#                clf.fit(train_x, train_y)
#                cv_score[t_idx, d_idx, inner_i] = clf.score(test_x, test_y)
#                print 'Prediction score is %s'%(clf.score(test_x, test_y))
#                print 'Dice coefficient:'
#                pred_y = clf.predict(test_x)
#                for label_idx in [1, 3]:
#                    P = pred_y == label_idx
#                    T = test_y == label_idx
#                    dice_val = mymath.dice_coef(T, P)
#                    print 'Dice for label %s: %f'%(label_idx, dice_val)
#                    if label_idx == 3:
#                        ffa_dice[t_idx, d_idx, inner_i] = dice_val
#                    else:
#                        ofa_dice[t_idx, d_idx, inner_i] = dice_val
#                print '-----------------------'
#
#    out_data_file = os.path.join(cv_dir, 'parameter_cv_data.npz')
#    np.savez(out_data_file, cv_score=cv_score, oob_score=oob_score,
#             ffa_dice=ffa_dice, ofa_dice=ofa_dice)

#-- Cross-validation to evaluate performance of model
# parameters of random forest
n_tree = range(10, 70, 10)
depth = range(10, 70, 10)

# output matrix
oob_score = np.zeros((len(n_tree), len(depth), cv_num))
cv_score = np.zeros((len(n_tree), len(depth), cv_num))
ofa_dice = np.zeros((len(n_tree), len(depth), cv_num))
ffa_dice = np.zeros((len(n_tree), len(depth), cv_num))

# split all subjects into 5 folds
subj_group = arlib.split_subject(sessid, cv_num)
for i in range(cv_num):
    print 'CV iter - ' + str(i)
    cv_dir = os.path.join(data_dir, 'cv_' + str(i))
    
    # split data into training and test group
    test_sessid = subj_group[i]
    train_sessid = []
    for j in range(cv_num):
        if not j == i:
            train_sessid += subj_group[j]

    # load test and train data
    print 'Load data ...'
    test_data = arlib.get_list_data(test_sessid, cv_dir)
    train_data = arlib.get_list_data(train_sessid, cv_dir)
    
    ## sample stats
    #print 'Samples stats for train-dataset:'
    #arlib.samples_stat(train_data)
    #print 'Samples stats for test-dataset:'
    #arlib.samples_stat(test_data)

    # split label and feature
    train_x = inner_train_data[..., :-1]
    train_y = inner_train_data[..., -1]
    test_x = inner_test_data[..., :-1]
    test_y = inner_test_data[..., -1]

    for t_idx in range(len(n_tree)):
        for d_idx in range(len(depth)):
            p = [n_tree[t_idx], depth[d_idx]]
            print 'Parameter: n_tree - %s; depth - %s'%(p[0], p[1])
            #-- compare OOB eror rate and the CV error rate
            # OOB error rate
            clf = RandomForestClassifier(n_estimators=p[0],
                                         criterion='gini',
                                         max_depth=p[1],
                                         n_jobs=20,
                                         oob_score=True)
            clf.fit(train_x, train_y)
            oob_score[t_idx, d_idx, inner_i] = clf.oob_score_
            print 'OOB score is %s'%(str(clf.oob_score_))
            # cross-validation
            cv_score[t_idx, d_idx, inner_i] = clf.score(test_x, test_y)
            print 'Prediction score is %s'%(clf.score(test_x, test_y))
            print 'Dice coefficient:'
            pred_y = clf.predict(test_x)
            for label_idx in [1, 3]:
                P = pred_y == label_idx
                T = test_y == label_idx
                dice_val = mymath.dice_coef(T, P)
                print 'Dice for label %s: %f'%(label_idx, dice_val)
                if label_idx == 3:
                    ffa_dice[t_idx, d_idx, inner_i] = dice_val
                else:
                    ofa_dice[t_idx, d_idx, inner_i] = dice_val
            print '-----------------------'

out_data_file = os.path.join(data_dir, 'parameter_cv_data.npz')
np.savez(out_data_file, cv_score=cv_score, oob_score=oob_score,
         ffa_dice=ffa_dice, ofa_dice=ofa_dice)

##-- model validation
## model defination
#clf = RandomForestClassifier(n_estimators=50, max_depth=20,
#                             criterion='entropy', max_features='sqrt',
#                             bootstrap=True, n_jobs=10)
## model training
#tt = time.time()
#clf.fit(train_x, train_y)
#print 'Model training costs %s'%(time.time() - tt)
#print 'Feature importance: ', clf.feature_importances_

## model testing
#predicted_y = clf.predict(test_x)
## prob_predicted_y = clf.predict_proba(test_x)

## model evaluation
#for l in range(1, 6, 2):
#    print 'Dice coefficient for label ' + str(l),
#    true_bool = test_y == l
#    predicted_bool = predicted_y == l
#    dice_coef = arlib.dice(true_bool, predicted_bool)
#    print dice_coef
    
