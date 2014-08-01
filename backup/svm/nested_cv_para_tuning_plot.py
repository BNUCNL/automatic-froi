# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pylab as pl

def grid_search_plot(cv_dir, roi_para='all'):
    roi_list = ['rOFA', 'raFFA', 'rpFFA']
    if roi_para == 'all':
        roi_name = roi_list
    else:
        roi_name = [roi_para]
    print roi_name
    print '-------------'

    dice_list = np.zeros((10, 10))
    precision_list = np.zeros((10, 10))
    recall_list = np.zeros((10, 10))

    for idx in range(100):
        c_idx = idx / 10
        gamma_idx = np.mod(idx, 10)
        para_dir = os.path.join(cv_dir, 'parameter_' + str(idx))
        d = []
        p = []
        r = []
        for roi in roi_name:
            sfile = os.path.join(para_dir, roi + '_CV_stats.csv')
            info = open(sfile).readlines()
            info = [line.strip().split(',') for line in info]
            iter_num = 5
            temp = info[iter_num + 1]
            d.append(float(temp[1]))
            p.append(float(temp[2]))
            r.append(float(temp[3]))
        dice_list[gamma_idx, c_idx] = np.mean(d)
        precision_list[gamma_idx, c_idx] = np.mean(p)
        recall_list[gamma_idx, c_idx] = np.mean(r)

    print 'Dice index'
    print np.unravel_index(dice_list.argmax(), dice_list.shape)
    print dice_list.max()
    print 'Precision'
    print np.unravel_index(precision_list.argmax(), precision_list.shape)
    print precision_list.max()
    print 'Recall'
    print np.unravel_index(recall_list.argmax(), recall_list.shape)
    print recall_list.max()

    # row: gamma, colume: c
    # dice
    fig = pl.figure(figsize=(15, 10))
    pl.subplot(2, 2, 1)
    c_idx = np.linspace(-3, 3, 10)
    gamma_idx = np.linspace(-3, 3, 10)
    pl.pcolor(gamma_idx, c_idx, dice_list)
    pl.colorbar()
    pl.xlabel('C')
    pl.ylabel('Gamma')
    pl.title('Dice index')
    # precision
    pl.subplot(2, 2, 2)
    pl.pcolor(gamma_idx, c_idx, precision_list)
    pl.colorbar()
    pl.xlabel('C')
    pl.ylabel('Gamma')
    pl.title('precision')
    # recall
    pl.subplot(2, 2, 3)
    pl.pcolor(gamma_idx, c_idx, recall_list)
    pl.colorbar()
    pl.xlabel('C')
    pl.ylabel('Gamma')
    pl.title('recall')
    #pl.show()
    fig_name = os.path.join(cv_dir, roi_para + '_fig.png')
    fig.savefig(fig_name)

# path configuration
root_dir = r'/nfs/t3/workingshop/huanglijie/autoroi/current/workingdir'
working_dir = os.path.join(root_dir, 'cv_perf')

iter_num = 5

for i in range(iter_num):
    print 'Cross-validation Outer Loop - ' + str(i)
    CV_dir = os.path.join(working_dir, 'Outer_CV_' + str(i))
    grid_search_plot(CV_dir, roi_para='all')

