# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np

# path configuration
root_dir = r'/nfs/t3/workingshop/huanglijie/autoroi/current'
working_dir = os.path.join(root_dir, 'workingdir', 'cv_perf')

# 5-fold cross-validation
iter_num = 5

roi_list = ['rOFA', 'rpFFA', 'raFFA']

for roi_name in roi_list:
    output_file = os.path.join(working_dir, roi_name + '_CV_stats.csv')
    f = open(output_file, 'w')
    f.write('Iter, dice, precision, recall\n')

    dice_mean = []
    precision_mean = []
    recall_mean = []
    for i in range(iter_num):
        cv_dir = os.path.join(working_dir, 'Outer_CV_' + str(i))
        sfile = os.path.join(cv_dir, roi_name + '_stats.csv')
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
    f.write('sd,')
    f.write(str(np.std(dice_mean)) + ',')
    f.write(str(np.std(precision_mean)) + ',')
    f.write(str(np.std(recall_mean)) + '\n')





