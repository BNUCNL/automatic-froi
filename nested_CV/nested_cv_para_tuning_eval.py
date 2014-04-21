# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import functools
from multiprocessing import Pool

from mypy import base as mybase
import autoroi

def perf_eval(para_idx, working_dir):
    cv_num = 5
    roi_dict = {'rOFA': 1, 'rpFFA': 3, 'raFFA':5}
    roi_list = ['rOFA', 'rpFFA', 'raFFA']

    validation_dir = os.path.join(working_dir, 'validate_test_data')
    para_dir = os.path.join(working_dir, 'parameter_' + str(para_idx))
    
    for i in range(cv_num):
        test_file = os.path.join(para_dir, 
                                 'test_CV_iter_' + str(i) + '.nii.gz')
        validate_file = os.path.join(validation_dir, 
                                'validate_CV_iter' + str(i) + '.nii.gz')
        test_img = nib.load(test_file)
        test_data = test_img.get_data()
        validate_img = nib.load(validate_file)
        validate_data = validate_img.get_data()
        subj_num = validate_data.shape[3]
        for roi_name in roi_list:
            out_file = os.path.join(para_dir,
                                    roi_name + '_CV_' + str(i) + '.csv')
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
                union = auto_roi + manual_roi
                union[union!=0] = 1
                union_sum = union.sum()
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
                f.write(','.join([str(idx), str(auto_roi_sum),
                                  str(manual_roi_sum), str(intersect_sum),
                                  str(dice), str(precision),
                                  str(recall)]) + '\n')


# path configuration
root_dir = r'/nfs/t3/workingshop/huanglijie/autoroi/current'
working_dir = os.path.join(root_dir, 'workingdir', 'cv_perf')

# 5-fold nested cross-validation
# inner loop was used to get optimized parameters using grid search,
# outter loop was used to evaluate the performance.
iter_num = 5

for i in range(iter_num):
    print 'Cross-Validation Outer Loop - ' + str(i)
    print '-----------------------------------'
    CV_dir = os.path.join(working_dir, 'Outer_CV_' + str(i))
    pool = Pool(processes=10)
    pool.map(functools.partial(perf_eval, working_dir=CV_dir), range(100))
    pool.terminate()


