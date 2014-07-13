# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np

import autoroilib as arlib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')

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
merged_true_file = os.path.join(data_dir, 'predicted_files',
                                'merged_true_label.nii.gz')
merged_predicted_file = os.path.join(data_dir, 'predicted_files',
                                     'merged_predicted_label.nii.gz')
cmd_str_1 = 'fslmerge -a ' + merged_true_file
cmd_str_2 = 'fslmerge -a ' + merged_predicted_file
for subj in sessid:
    subj_dir = os.path.join(src_dir, subj, 'face-object')
    label_file = arlib.get_label_file(subj_dir)
    cmd_str_1 += ' ' + label_file
    pre_file = os.path.join(data_dir, 'predicted_files',
                            subj + '_predicted.nii.gz')
    cmd_str_2 += ' ' + pre_file
os.system(cmd_str_1)
os.system(cmd_str_2)



