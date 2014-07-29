# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from mypy import math as mymath
import autoroilib as arlib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data', 'cv')

# read all subjects' SID
sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

##-- get sample number from each subject
#print 'Load data ... '
#count = 0
#sample_dir = os.path.join(data_dir, 'samples')
#out_file = os.path.join(sample_dir, 'sample_num.txt')
#f = open(out_file, 'wb')
#for subj in sessid:
#    data_file = os.path.join(sample_dir, subj + '_data.csv')
#    samples = arlib.get_subj_data(data_file)
#    # remove samples whose z-value is less than 2.3
#    sample_mask = samples[:, 3] >= 2.3
#    f.write(subj + ' ' + str(sample_mask.sum()) + '\n')
#    count += sample_mask.sum()
#print count

#-- merge ground truth and predicted nifti files
src_dir = r'/nfs/t2/atlas/database'
merged_true_file = os.path.join(data_dir, 'merged_true_label.nii.gz')
merged_zstat_file = os.path.join(data_dir, 'merged_zstat.nii.gz')
#merged_predicted_file = os.path.join(data_dir, 'predicted_files',
#                                     'merged_predicted_label.nii.gz')
cmd_str_1 = 'fslmerge -a ' + merged_true_file
cmd_str_2 = 'fslmerge -a ' + merged_zstat_file
#cmd_str_3 = 'fslmerge -a ' + merged_predicted_file
for subj in sessid:
    subj_dir = os.path.join(src_dir, subj, 'face-object')
    label_file = arlib.get_label_file(subj_dir)
    cmd_str_1 += ' ' + label_file
    zstat_file = os.path.join(subj_dir, 'zstat1.nii.gz')
    cmd_str_2 += ' ' + zstat_file
    #pre_file = os.path.join(data_dir, 'predicted_files',
    #                        subj + '_predicted.nii.gz')
    #cmd_str_3 += ' ' + pre_file
os.system(cmd_str_1)
os.system(cmd_str_2)

##-- compare predicted label and ground-truth for each subject
#src_dir = os.path.join(data_dir, 'predicted_files')
#merged_true_label = os.path.join(src_dir, 'merged_true_label.nii.gz')
#merged_predicted_label = os.path.join(src_dir, 'merged_predicted_label.nii.gz')
#true_data = nib.load(merged_true_label).get_data()
#predicted_data = nib.load(merged_predicted_label).get_data()
#
#print 'SID OFA FFA'
#for i in range(len(sessid)):
#    print sessid[i],
#    dice = []
#    for j in [1, 3]:
#        temp_1 = true_data[..., i].copy()
#        temp_2 = predicted_data[..., i].copy()
#        temp_1[temp_1!=j] = 0
#        temp_1[temp_1==j] = 1
#        temp_2[temp_2!=j] = 0
#        temp_2[temp_2==j] = 1
#        if not temp_1.sum():
#            if not temp_2.sum():
#                dice.append(1.0)
#            else:
#                dice.append(2 * np.sum(temp_1 * temp_2) / \
#                            (temp_1.sum() + temp_2.sum()))
#        else:
#            dice.append(2 * np.sum(temp_1 * temp_2) / \
#                        (temp_1.sum() + temp_2.sum()))
#    print '%s %s'%(dice[0], dice[1])

