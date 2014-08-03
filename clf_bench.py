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
import nibabel as nib
from mypy import base as mybase

import autoroilib as arlib
from mypy import math as mymath

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data', 'cv_mix')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

#-- 5-folds cross-validation
cv_num = 5

## split all subjects into 5 folds
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
#    # train dataset
#    pool = mps.Pool(20)
#    result = pool.map(functools.partial(arlib.ext_subj_feature,
#                                        mask_coord=mask_coords,
#                                        out_dir=cv_dir, mask_out=True),
#                      train_sessid)
#    pool.terminate()
#
#    # test dataset
#    pool = mps.Pool(20)
#    result = pool.map(functools.partial(arlib.ext_subj_feature,
#                                        mask_coord=mask_coords,
#                                        out_dir=cv_dir, mask_out=False),
#                      test_sessid)
#    pool.terminate()

##-- Cross-validation to select parameters with nested-CV
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

##-- Cross-validation to select parameters
## parameters of random forest
#n_tree = range(10, 70, 10)
#depth = range(10, 70, 10)
#
## output matrix
#oob_score = np.zeros((len(n_tree), len(depth), cv_num))
#cv_score = np.zeros((len(n_tree), len(depth), cv_num))
#ofa_dice = np.zeros((len(n_tree), len(depth), cv_num))
#ffa_dice = np.zeros((len(n_tree), len(depth), cv_num))
#
## feature type
#features = {}
#features['coord'] = [0, 1, 2]
#features['z_vxl'] = [3]
#features['mni_vxl'] = [4]
#features['fbeta_vxl'] = [5]
#features['obeta_vxl'] = [6]
#features['prob'] = [7, 8]
#features['z_type1'] = [9, 10, 11]
#features['mni_type1'] = [12, 13, 14]
#features['fbeta_type1'] = [15, 16, 17]
#features['obeta_type1'] = [18, 19, 20]
#features['z_type2'] = range(21, 114, 4)
#features['mni_type2'] = range(22, 115, 4)
#features['fbeta_type2'] = range(23, 116, 4)
#features['obeta_type2'] = range(24, 117, 4)
#features['z_type3'] = range(117, 894, 4)
#features['mni_type3'] = range(118, 895, 4)
#features['fbeta_type3'] = range(119, 896, 4)
#features['obeta_type3'] = range(120, 897, 4)
#
#feature_idx = features['coord'] + features['z_type3']
#
## split all subjects into 5 folds
#subj_group = arlib.split_subject(sessid, cv_num)
#for i in range(cv_num):
#    print 'CV iter - ' + str(i)
#    cv_dir = os.path.join(data_dir, 'cv_' + str(i))
#    
#    # split data into training and test group
#    test_sessid = subj_group[i]
#    train_sessid = []
#    for j in range(cv_num):
#        if not j == i:
#            train_sessid += subj_group[j]
#
#    # load test and train data
#    print 'Load data ...'
#    test_data = arlib.get_list_data(test_sessid, cv_dir)
#    train_data = arlib.get_list_data(train_sessid, cv_dir)
#    
#    ## sample stats
#    #print 'Samples stats for train-dataset:'
#    #arlib.samples_stat(train_data)
#    #print 'Samples stats for test-dataset:'
#    #arlib.samples_stat(test_data)
#
#    # split label and feature
#    train_x = train_data[..., feature_idx]
#    train_y = train_data[..., -1]
#    test_x = test_data[..., feature_idx]
#    test_y = test_data[..., -1]
#
#    for t_idx in range(len(n_tree)):
#        for d_idx in range(len(depth)):
#            p = [n_tree[t_idx], depth[d_idx]]
#            print 'Parameter: n_tree - %s; depth - %s'%(p[0], p[1])
#            #-- compare OOB eror rate and the CV error rate
#            # OOB error rate
#            clf = RandomForestClassifier(n_estimators=p[0],
#                                         criterion='gini',
#                                         max_depth=p[1],
#                                         n_jobs=20,
#                                         oob_score=True)
#            clf.fit(train_x, train_y)
#            oob_score[t_idx, d_idx, i] = clf.oob_score_
#            print 'OOB score is %s'%(str(clf.oob_score_))
#            # cross-validation
#            cv_score[t_idx, d_idx, i] = clf.score(test_x, test_y)
#            print 'Prediction score is %s'%(clf.score(test_x, test_y))
#            print 'Dice coefficient:'
#            pred_y = clf.predict(test_x)
#            for label_idx in [1, 3]:
#                P = pred_y == label_idx
#                T = test_y == label_idx
#                dice_val = mymath.dice_coef(T, P)
#                print 'Dice for label %s: %f'%(label_idx, dice_val)
#                if label_idx == 3:
#                    ffa_dice[t_idx, d_idx, i] = dice_val
#                else:
#                    ofa_dice[t_idx, d_idx, i] = dice_val
#            print '-----------------------'
#
#out_data_file = os.path.join(data_dir, 'parameter_cv_data.npz')
#np.savez(out_data_file, cv_score=cv_score, oob_score=oob_score,
#         ffa_dice=ffa_dice, ofa_dice=ofa_dice)

#-- Cross-validation to evaluate performance of model
cv_score = np.zeros((cv_num))
ofa_dice = np.zeros((cv_num))
ffa_dice = np.zeros((cv_num))
ffa_precision = np.zeros((cv_num))
ofa_precision = np.zeros((cv_num))
ffa_recall = np.zeros((cv_num))
ofa_recall = np.zeros((cv_num))

## get feature name
#feature_name_file = os.path.join(data_dir, 'feature_name.txt')
#feature_name = arlib.get_label(feature_name_file)
#feature_name = [feature_name[i] for i in feature_idx]

# predicted nifti directory
pred_dir = os.path.join(data_dir, 'predicted_files')
os.system('mkdir ' + pred_dir)

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
    #test_data_mask = test_data[:, 0] >= 1.96
    #test_data = test_data[test_data_mask, :]

    print 'Training dataset stats ...'
    arlib.samples_stat(train_data)
    print 'Testing dataset stats ...'
    arlib.samples_stat(test_data)

    # split label and feature
    train_x = train_data[..., 1:-1]
    train_y = train_data[..., -1]
    test_x = test_data[..., 1:-1]
    test_y = test_data[..., -1]

    # model defination
    clf = RandomForestClassifier(n_estimators=50, max_depth=30,
                                 criterion='gini', n_jobs=20)
    # model training
    tt = time.time()
    clf.fit(train_x, train_y)
    print 'Model training costs %s'%(time.time() - tt)
    #for f_idx in range(len(clf.feature_importances_)):
    #    print '%s %s'%(feature_name[f_idx], clf.feature_importances_[f_idx])

    # model testing
    cv_score[i] = clf.score(test_x, test_y)
    print 'Prediction score is %s'%(clf.score(test_x, test_y))
    print 'Dice coefficient:'
    pred_y = clf.predict(test_x)
    for label_idx in [1, 3]:
        P = pred_y == label_idx
        T = test_y == label_idx
        dice_val = mymath.dice_coef(T, P)
        print 'Dice for label %s: %f'%(label_idx, dice_val)
        precision_val = mymath.precision(T, P)
        print 'Precision for label %s: %f'%(label_idx, precision_val)
        recall_val = mymath.recall(T, P)
        print 'Recall for label %s: %f'%(label_idx, recall_val)
        if label_idx == 3:
            ffa_dice[i] = dice_val
            ffa_precision[i] = precision_val
            ffa_recall[i] = recall_val
        else:
            ofa_dice[i] = dice_val
            ofa_precision[i] = precision_val
            ofa_recall[i] = recall_val
    print '-----------------------'

    # get predict proba
    clf_classes = clf.classes_
    #print clf_classes
    pred_prob = clf.predict_proba(test_x)

    #-- save predicted label as nifti files
    fsl_dir = os.getenv('FSL_DIR')
    img = nib.load(os.path.join(fsl_dir, 'data', 'standard',
                                'MNI152_T1_2mm_brain.nii.gz'))
    header = img.get_header()
    
    # load sample number of each subject
    sample_num_file = os.path.join(cv_dir, 'sample_num.txt')
    subj_sample_num = arlib.get_subj_sample_num(sample_num_file)
    start_num = 0
    for subj_idx in range(len(test_sessid)):
        sample_num = subj_sample_num[subj_idx]
        end_num = start_num + sample_num
        coords = test_x[start_num:end_num, 0:3]

        ## save predicted label
        #voxel_val = pred_y[start_num:end_num]
        #pred_data = arlib.write2array(coords, voxel_val)
        #out_file = os.path.join(pred_dir,
        #                        test_sessid[subj_idx]+'_pred.nii.gz')
        #mybase.save2nifti(pred_data, header, out_file)
        #start_num += sample_num

        # probability map smoothing and save to the nifti files
        prob_data = np.zeros((91, 109, 91, len(clf_classes)))
        for k in range(len(clf_classes)):
            prob_val = pred_prob[start_num:end_num, k]
            prob_data[..., k] = arlib.write2array(coords, prob_val)
        mask = np.sum(prob_data, axis=3)
        sm_prob_data = arlib.smooth_data(prob_data, 1)
        sm_pred_data = np.argmax(sm_prob_data, axis=3)
        sm_pred_data[sm_pred_data==2] = 3
        sm_pred_data = sm_pred_data * mask
        out_file = os.path.join(pred_dir,
                                test_sessid[subj_idx]+'_pred.nii.gz')
        mybase.save2nifti(sm_pred_data, header, out_file)
        start_num += sample_num

print 'Mean CV score is %s'%(cv_score.mean())
print 'Mean FFA Dice is %s'%(ffa_dice.mean())
print 'Mean OFA Dice is %s'%(ofa_dice.mean())

##-- effect of sample numbers (subjects number)
#out_file = os.path.join(data_dir, 'subj_size_effect.txt')
#f = open(out_file, 'wb')
#f.write('subj_size,samples,label_1,label_3,score,ofa_dice,ffa_dice\n')
## subjects size range
#subj_num = range(20, 160, 10)
## repeat number
#repeat_num = 10
#
## feature type
#features = {}
#features['coord'] = [0, 1, 2]
#features['z_type3'] = range(117, 894, 4)
#feature_idx = features['coord'] + features['z_type3']
#
## split subjets into two groups: training and test dataset
#subj_group = arlib.split_subject(sessid, 10)
#for test_idx in range(10):
#    test_sessid = subj_group[test_idx]
#    train_sessid = []
#    for grp_idx in range(10):
#        if not grp_idx == test_idx:
#            train_sessid += subj_group[grp_idx]
#    # load test data
#    sample_dir = os.path.join(data_dir, 'samples')
#    test_data = arlib.get_list_data(test_sessid, sample_dir)
#    print 'Sample stats for testing data'
#    arlib.samples_stat(test_data)
#    test_x = test_data[..., :-1]
#    test_y = test_data[..., -1]
#    for subj_size in subj_num:
#        print 'Subjects number: %s'%(subj_size)
#        for p in range(repeat_num):
#            print 'repeat %s'%(p)
#            out_str = [str(subj_size)]
#            subj_idx = random.sample(range(len(train_sessid)), subj_size)
#            sel_sessid = [train_sessid[i] for i in subj_idx]
#            sample_data = arlib.get_list_data(sel_sessid, sample_dir)
#            # samples stats
#            print 'Sample stats of dataset:'
#            arlib.samples_stat(sample_data)
#            x = sample_data[..., :-1]
#            y = sample_data[..., -1]
#            label_all = y.shape[0]
#            label_1 = np.sum(y == 1)
#            label_3 = np.sum(y == 3)
#            out_str.append(str(label_all))
#            out_str.append(str(label_1))
#            out_str.append(str(label_3))
#            # model training
#            clf = RandomForestClassifier(n_estimators=60, max_depth=30,
#                                         criterion='gini', n_jobs=20)
#            clf.fit(x, y)
#            pred_y = clf.predict(test_x)
#            score = accuracy_score(test_y, pred_y)
#            print 'Predict score is %s'%(score)

