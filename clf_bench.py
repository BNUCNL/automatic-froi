# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import autoroilib as arlib

base_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
doc_dir = os.path.join(base_dir, 'doc')
data_dir = os.path.join(base_dir, 'data')

sessid_file = os.path.join(doc_dir, 'test_sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

# split all subjects into 4 folds
subj_group = arlib.split_subject(sessid, 3)
#arlib.save_subject_group(subj_group, data_dir)

# extract training subjects dataset
train_sessid = subj_group[0] + subj_group[1]
test_sessid = subj_group[2]

# generate mask and probability map
prob_data, mask_data = arlib.make_prior(train_sessid, data_dir)
mask_coords = arlib.get_mask_coord(mask_data, data_dir)

# extract features from each subject
for subj in test_sessid:
    print subj
    st = time.time()
    print 'Start time: ' + str(st)
    sample_label, subj_data = arlib.ext_feature(subj, mask_coords, prob_data)
    out_file = os.path.join(data_dir, subj + '_data.csv')
    arlib.save_sample(sample_label, subj_data, out_file)
    print 'Cost time: %s'%(time.time() - st)

