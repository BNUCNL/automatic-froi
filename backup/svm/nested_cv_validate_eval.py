# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import functools
from multiprocessing import Pool

from mypy import base as mybase
import autoroi

def perf_eval(cv_idx, workingdir):
    roi_dict = {'rOFA': 1, 'rpFFA': 3, 'raFFA':5}
    roi_list = ['rOFA', 'rpFFA', 'raFFA']
    # get CV directory
    cv_dir = os.path.join(workingdir, 'Outer_CV_' + str(cv_idx))
    test_file = os.path.join(cv_dir, 'auto_test_data.nii.gz')
    validate_file = os.path.join(cv_dir, 'validate_test_data.nii.gz')
    test_img = nib.load(test_file)
    test_data = test_img.get_data()
    validate_img = nib.load(validate_file)
    validate_data = validate_img.get_data()
    subj_num = validate_data.shape[3]
    for roi_name in roi_list:
        out_file = os.path.join(cv_dir, roi_name + '_stats.csv')
        f = open(out_file, 'w')
        f.write('index, auto_roi, intersect, Dice, precision, recall\n')
        for idx in range(subj_num):
            auto_roi = test_data[..., idx].copy()
            auto_roi[auto_roi!=roi_dict[roi_name]] = 0
            auto_roi[auto_roi==roi_dict[roi_name]] = 1
            auto_roi_sum = auto_roi.sum()
            manual_roi = validate_data[..., idx].copy()
            manual_roi[manual_roi!=roi_dict[roi_name]] = 0
            manual_roi[manual_roi==roi_dict[roi_name]] = 1
            manual_roi_sum = manual_roi.sum()
            intersect = auto_roi * manual_roi
            intersect_sum = intersect.sum()
            if manual_roi_sum == 0:
                if auto_roi_sum == 0:
                    dice = 1
                    precision = 1
                    recall = 1
                else:
                    dice = 0
                    precision = 0
                    recall = 0
            else:
                dice = 2 * intersect_sum / (manual_roi_sum + auto_roi_sum)
                recall = intersect_sum / manual_roi_sum
                if auto_roi_sum == 0:
                    precision = 0
                else:
                    precision = intersect_sum / auto_roi_sum
            f.write(','.join([str(idx), str(auto_roi_sum), str(manual_roi_sum),
                              str(intersect_sum), str(dice),
                              str(precision), str(recall)]) + '\n')


# path configuration
root_dir = r'/nfs/t3/workingshop/huanglijie/autoroi/current'
working_dir = os.path.join(root_dir, 'workingdir', 'cv_perf')

# 5-fold cross-validation
iter_num = 5

print 'Cross-Validation Outer Loop'
print '-----------------------------------'
pool = Pool(processes=iter_num)
pool.map(functools.partial(perf_eval, workingdir=working_dir), range(iter_num))


