# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import functools
from multiprocessing import Pool

from mypy import base as mybase
import autoroi

def get_tuned_parameter(cv_dir):
    roi_name = ['rOFA', 'raFFA', 'rpFFA']
    dice_list = np.zeros((10, 10))
    for idx in range(100):
        c_idx = idx / 10
        gamma_idx = np.mod(idx, 10)
        para_dir = os.path.join(cv_dir, 'parameter_' + str(idx))
        d = []
        for roi in roi_name:
            sfile = os.path.join(para_dir, roi + '_CV_stats.csv')
            info = open(sfile).readlines()
            info = [line.strip().split(',') for line in info]
            iter_num = 5
            temp = info[iter_num + 1]
            d.append(float(temp[1]))
        dice_list[gamma_idx, c_idx] = np.mean(d)
    print 'Dice index'
    peak_coord = np.unravel_index(dice_list.argmax(), dice_list.shape)
    print peak_coord
    print dice_list.max()
    para_idx = peak_coord[1] * 10 + peak_coord[0]
    print para_idx
    return para_idx

def clf_validate(cv_idx, workingdir, zstat_data, roi_data, roi_header):
    """
    Train a new clfer based on the tuned parameters.

    """
    # get train- and test-dataset
    subj_num = 202
    subj_list = np.arange(subj_num)
    cv_subj_list = np.array_split(subj_list, 5)
    train_subj = list(cv_subj_list)
    test_subj = train_subj.pop(cv_idx)
    train_subj = np.concatenate(train_subj)
    # get tuned SVC parameters
    cv_dir = os.path.join(workingdir, 'Outer_CV_' + str(cv_idx))
    tuned_para_idx = get_tuned_parameter(cv_dir)
    # parameter configuration
    c_value = np.logspace(-3, 3, 10)
    gamma_value = np.logspace(-3, 3, 10)
    # set c and gamma
    c = c_value[tuned_para_idx/10]
    gamma = gamma_value[np.mod(tuned_para_idx, 10)]
    print 'Tuned parameters: ',
    print 'C - ' + str(c),
    print ' Gamma - ' + str(gamma)
    # -- start validation
    # generate ROI mask and probabilioty map
    mask_data, prob_data = autoroi.make_priori(train_subj, zstat_data, roi_data)
    # generate training samples
    train_samples = []
    train_labels = []
    for subj_idx in train_subj:
        print 'Subject ' + str(subj_idx)
        marker, result = autoroi.segment(zstat_data[..., subj_idx])
        subj_samples, subj_labels, parcel_id = autoroi.sample_extract(
                                                    zstat_data[..., subj_idx],
                                                    roi_data[..., subj_idx],
                                                    mask_data,
                                                    prob_data,
                                                    marker,
                                                    result)
        train_samples += subj_samples
        train_labels += subj_labels
    # train classifier
    scaler, svc = autoroi.roi_svc(train_samples, train_labels, kernel='rbf', 
                                      c = c, gamma=gamma)
    # test test classifier performance on the test dataset
    test_subj_num = len(test_subj)
    auto_roi_data = np.zeros([91, 109, 91, test_subj_num])
    subj_count = 0
    for subj_idx in test_subj:
        marker, result = autoroi.segment(zstat_data[..., subj_idx])
        subj_samples, subj_labels, parcel_id = autoroi.sample_extract(
                                                    zstat_data[..., subj_idx],
                                                    roi_data[..., subj_idx],
                                                    mask_data,
                                                    prob_data,
                                                    marker,
                                                    result)
        # remove all parcels except samples
        all_parcel_id = np.unique(marker)
        all_parcel_id = all_parcel_id.tolist()
        all_parcel_id.pop(0)
        for idx in all_parcel_id:
            if idx not in parcel_id:
                result[result==idx] = 0
        subj_samples = scaler.transform(subj_samples)
        predict_labels = svc.predict(subj_samples)
        parcel_num = len(parcel_id)
        auto_res = auto_roi_data[..., subj_count]
        for idx in range(parcel_num):
            auto_res[result==parcel_id[idx]] = predict_labels[idx]
        subj_count += 1
    # save predicted ROI
    roi_header.set_data_shape(auto_roi_data.shape)
    out_file_name = os.path.join(cv_dir, 'auto_test_data.nii.gz')
    mybase.save2nifti(auto_roi_data, roi_header, out_file_name)


# path configuration
root_dir = r'/nfs/t3/workingshop/huanglijie/autoroi/current'
zstat_file = os.path.join(root_dir, 'data', 'face_obj_zstat.nii.gz')
roi_file = os.path.join(root_dir, 'data', 'manual_face_roi.nii.gz')
working_dir = os.path.join(root_dir, 'workingdir', 'cv_perf')

# load nifti data
zstat_img = nib.load(zstat_file)
zstat_data = zstat_img.get_data()
roi_img = nib.load(roi_file)
roi_header = roi_img.get_header()
roi_data = roi_img.get_data()
# remove other ROIs except 
# rpFFA, rOFA and raFFA
roi_data[roi_data==2] = 0
roi_data[roi_data==4] = 0
roi_data[roi_data>5] = 0

# 5-fold cross-validation
# Parameters of SVC was tuned by an inner cross-validation procedure.
# This is an outer loop of cross-validation used to evaluate the performance
# of the method.
iter_num = 5

print 'Cross-Validation of Outer Loop'
print '-----------------------------------'
pool = Pool(processes=iter_num)
pool.map(functools.partial(clf_validate, workingdir=working_dir, 
                           zstat_data=zstat_data, 
                           roi_data=roi_data, roi_header=roi_header), 
         range(iter_num))

