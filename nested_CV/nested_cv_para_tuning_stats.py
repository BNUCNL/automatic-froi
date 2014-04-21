# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import functools
from multiprocessing import Pool

def perf_stats(para_idx, working_dir):
    cv_num = 5
    roi_list = ['rOFA', 'rpFFA', 'raFFA']

    para_dir = os.path.join(working_dir, 'parameter_' + str(para_idx))

    for roi_name in roi_list:
        output_file = os.path.join(para_dir, roi_name + '_CV_stats.csv')
        f = open(output_file, 'w')
        f.write('Iter, dice, precision, recall\n')

        dice_mean = []
        precision_mean = []
        recall_mean = []
        for i in range(cv_num):
            sfile = os.path.join(para_dir, 
                                 roi_name + '_CV_' + str(i) + '.csv')
            info = open(sfile).readlines()
            info = [line.strip().split(',') for line in info]
            info.pop(0)
            d = []
            p = []
            r = []
            for line in info:
                d.append(float(line[4]))
                p.append(float(line[5]))
                r.append(float(line[6]))
            d_mean = np.mean(d)
            dice_mean.append(d_mean)
            p_mean = np.mean(p)
            precision_mean.append(p_mean)
            r_mean = np.mean(r)
            recall_mean.append(r_mean)
            f.write(','.join([str(i), str(d_mean), str(p_mean), str(r_mean)]) + '\n')
        f.write('mean,')
        f.write(str(np.mean(dice_mean)) + ',')
        f.write(str(np.mean(precision_mean)) + ',')
        f.write(str(np.mean(recall_mean)) + '\n')


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
    pool = Pool(processes=20)
    pool.map(functools.partial(perf_stats, working_dir=CV_dir), range(100))
    pool.terminate()


