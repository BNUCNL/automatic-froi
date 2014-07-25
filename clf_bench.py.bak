# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import nibabel as nib

# modules for data preparation
import multiprocessing as mps
import functools

# modules for modle training and testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score

from mypy import base as mybase
import autoroilib as arlib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

##-- extract features from each subject
#sample_dir = os.path.join(data_dir, 'samples')
#os.system('mkdir ' + out_dir)

## generate mask and probability map
#prob_data, mask_data = arlib.make_prior(sessid, out_dir)
#mask_coords = arlib.get_mask_coord(mask_data, out_dir)

## extract features
#pool = mps.Pool(20)
#result = pool.map(functools.partial(arlib.ext_subj_feature,
#                                    mask_coord=mask_coords,
#                                    prob_data=prob_data,
#                                    out_dir=out_dir), sessid)

##-- parameter selection
#print 'Load data ... '
#sample_dir = os.path.join(data_dir, 'samples')
#sample_data = arlib.get_list_data(sessid, sample_dir)
#
## remove samples whose z-value is less than 2.3
#print 'Remove samples whose z-value is less than 2.3 ...'
#sample_data_mask = sample_data[:, 3] >= 2.3
#sample_data = sample_data[sample_data_mask]
#
## sample stats
#print 'Samples stats of dataset:'
#arlib.samples_stat(sample_data)
#
## shuffle the dataset
#sample_idx = np.arange(sample_data.shape[0])
#np.random.shuffle(sample_idx)
#sample_data = sample_data[sample_idx]
#
## split label and feature
#x = sample_data[..., :-1]
#y = sample_data[..., -1]
#
## parameters of random forest
#n_tree = range(10, 70, 10)
#depth = range(10, 70, 10)
#
## output matrix
#oob_score = np.zeros((len(n_tree), len(depth)))
#cv_score = np.zeros((len(n_tree), len(depth)))
#ofa_dice = np.zeros((len(n_tree), len(depth)))
#ffa_dice = np.zeros((len(n_tree), len(depth)))
#
#for t_idx in range(len(n_tree)):
#    for d_idx in range(len(depth)):
#        p = [n_tree[t_idx], depth[d_idx]]
#        print 'Parameter: n_tree - %s; depth - %s'%(p[0], p[1])
#        #-- compare OOB score and the CV score
#        # Cross-Validation score
#        clf = RandomForestClassifier(n_estimators=p[0], criterion='gini',
#                                     max_depth=p[1], n_jobs=20)
#        scores = cross_val_score(clf, x, y)
#        cv_score[t_idx, d_idx] = scores.mean()
#        print 'Mean score with CV is ' + str(scores.mean())
#        # OOB score
#        clf = RandomForestClassifier(n_estimators=p[0], criterion='gini',
#                                     max_depth=p[1], n_jobs=20,
#                                     oob_score=True)
#        clf.fit(x, y)
#        oob_score[t_idx, d_idx] = clf.oob_score_
#        print 'OOB score is ' + str(clf.oob_score_)
#        print 'Dice coefficient:'
#        oob_functions = clf.oob_decision_function_
#        oob_labels = []
#        for line in oob_functions:
#            if np.all(np.isnan(line)):
#                oob_labels.append(np.nan)
#            else:
#                oob_labels.append(line.argmax())
#        oob_labels = np.array(oob_labels)
#        nan_num = np.sum(np.isnan(oob_labels))
#        print 'NaN number: %d'%nan_num
#        label_idx = [1, 3]
#        oob_idx = [1, 2]
#        for j in range(2):
#            P = oob_labels == oob_idx[j]
#            T = y == label_idx[j]
#            dice_val = arlib.dice(T, P)
#            print 'Dice for label %s: %f'%(label_idx[j], dice_val)
#            print 'Sample number: %d'%P.sum()
#            if j:
#                ffa_dice[t_idx, d_idx] = dice_val
#            else:
#                ofa_dice[t_idx, d_idx] = dice_val
#        print '-----------------------'
#out_file = os.path.join(data_dir, 'parameter_score.npz')
#np.savez(out_file, cv_score=cv_score, oob_score=oob_score,
#         ffa_dice=ffa_dice, ofa_dice=ofa_dice)

#-- feature selection
print 'Load data ... '
sample_dir = os.path.join(data_dir, 'samples')
tt = time.time()
sample_data = arlib.get_list_data(sessid, sample_dir)
print 'Data loading costs %s'%(time.time() - tt)

# remove samples whose z-value is less than 2.3
print 'Remove samples whose z-value is less than 2.3 ...'
sample_data_mask = sample_data[:, 3] >= 2.3
sample_data = sample_data[sample_data_mask]

# sample stats
print 'Samples stats of dataset:'
arlib.samples_stat(sample_data)

# split label and feature
x = sample_data[..., :-1]
y = sample_data[..., -1]

clf = RandomForestClassifier(n_estimators=50, max_depth=40, criterion='gini',
                             oob_score=True, n_jobs=20)

# model training
tt = time.time()
clf.fit(x, y)
print 'Model training costs %s'%(time.time() - tt)
print 'OOB score is %s'%(clf.oob_score_)
feature_importance = clf.feature_importances_
print 'Sum of feature importances: %s'%(feature_importance.sum())
print 'Mean importance is %s'%(feature_importance.mean())

# Dice
print 'Dice coefficient:'
oob_functions = clf.oob_decision_function_
oob_labels = []
for line in oob_functions:
    if np.all(np.isnan(line)):
        oob_labels.append(np.nan)
    else:
        oob_labels.append(line.argmax())
oob_labels = np.array(oob_labels)
oob_labels[oob_labels==2] = 3
nan_num = np.sum(np.isnan(oob_labels))
print 'NaN number: %d'%nan_num
for j in [1, 3]:
    P = oob_labels == j
    T = y == j
    dice_val = arlib.dice(T, P)
    print 'Dice for label %s: %f'%(j, dice_val)
    print 'Sample number: %d'%P.sum()

# write predicted label to the nifti file
# load MNI template as data container
fsl_dir = os.getenv('FSL_DIR')
img = nib.load(os.path.join(fsl_dir, 'data', 'standard',
                            'MNI152_T1_2mm_brain.nii.gz'))
header = img.get_header()

# load samples number of each subject
sample_num_file = os.path.join(sample_dir, 'sample_num.txt')
subj_sample_num = arlib.get_subj_sample_num(sample_num_file)
start_num = 0
for i in range(len(sessid)):
    sample_num = subj_sample_num[i]
    end_num = start_num + sample_num
    coords = x[start_num:end_num, 0:3]
    voxel_val = oob_labels[start_num:end_num]
    predicted_data = arlib.write2array(coords, voxel_val)
    start_num += sample_num
    out_file = os.path.join(data_dir, 'predicted_files',
                            sessid[i] + '_predicted.nii.gz')
    mybase.save2nifti(predicted_data, header, out_file)

# load feature label
label_file = os.path.join(sample_dir, 'label.txt')
label = arlib.get_label(label_file)

# print feature importances
for i in range(len(label)):
    print '%d %s %s'%(i, label[i], feature_importance[i])

## sort label importance
#print 'Mean importance is %s'%(feature_importance.mean())
#sorted_idx = np.argsort(feature_importance)
#selected_feature_count = 0
#for i in sorted_idx:
#    if feature_importance[i] >= feature_importance.mean():
#        print '%d %s %s'%(i, label[i], feature_importance[i])
#        selected_feature_count += 1
#print 'Selected feature number: %d'%(selected_feature_count)

## get selected features
#selected_x = clf.transform(x)
#print selected_x.shape
#
## re-train the model
#clf = RandomForestClassifier(n_estimators=50, max_depth=40, criterion='gini',
#                             oob_score=True, n_jobs=20)
#tt = time.time()
#clf.fit(selected_x, y)
#print 'Model training costs %s'%(time.time() - tt)
#print 'OOB score is %s'%(clf.oob_score_)
#
## Dice
#print 'Dice coefficient:'
#oob_functions = clf.oob_decision_function_
#oob_labels = []
#for line in oob_functions:
#    if np.all(np.isnan(line)):
#        oob_labels.append(np.nan)
#    else:
#        oob_labels.append(line.argmax())
#oob_labels = np.array(oob_labels)
#nan_num = np.sum(np.isnan(oob_labels))
#print 'NaN number: %d'%nan_num
#label_idx = [1, 3]
#oob_idx = [1, 2]
#for j in range(2):
#    P = oob_labels == oob_idx[j]
#    T = y == label_idx[j]
#    dice_val = arlib.dice(T, P)
#    print 'Dice for label %s: %f'%(label_idx[j], dice_val)
#    print 'Sample number: %d'%P.sum()

