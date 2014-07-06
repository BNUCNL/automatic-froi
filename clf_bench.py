# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import multiprocessing as mps
import functools

import autoroilib as arlib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')

sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

#-- 5-folds cross validation
# split all subjects into 5 folds
cv_num = 5
subj_group = arlib.split_subject(sessid, cv_num)
arlib.save_subject_group(subj_group, data_dir)

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


