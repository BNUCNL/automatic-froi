# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import functools
from multiprocessing import Pool

from mypy import base as mybase
import autoroi

def clf_gen(para_idx, subj_list, zstat_data, roi_data, working_dir, roi_header):
    """
    Train a new clfer based on the subject list and specified parameters.

    """
    # configuration
    iter_num = 5
    c_value = np.logspace(-3, 3, 10)
    gamma_value = np.logspace(-3, 3, 10)
    # set c and gamma
    c = c_value[para_idx/10]
    gamma = gamma_value[np.mod(para_idx, 10)]
    cv_idx = np.array_split(subj_list, iter_num)
    for i in range(iter_num):
        # divide subjects into train-set and test-set
        subj_train = list(cv_idx)
        subj_test = subj_train.pop(i)
        subj_train = np.concatenate(subj_train)
        # generate ROI mask and probabilioty map
        mask_data, prob_data = autoroi.make_priori(subj_train, zstat_data, roi_data)
        # generate training samples
        train_samples = []
        train_labels = []
        for subj_idx in subj_train:
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
        test_subj_num = len(subj_test)
        auto_roi_data = np.zeros([91, 109, 91, test_subj_num])
        subj_count = 0
        for subj_idx in subj_test:
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
        out_file_dir = os.path.join(working_dir, 'parameter_' + str(para_idx))
        if not os.path.exists(out_file_dir):
            os.system('mkdir ' + out_file_dir)
        out_file_name = os.path.join(out_file_dir, 'test_CV_iter_' + str(i) + '.nii.gz')
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

# generate subject list
subj_num = 202
subj_list = np.arange(subj_num)

# 5-fold nested cross-validation
# inner loop was used to get optimized parameters using grid search,
# outter loop was used to evaluate the performance.
iter_num = 5

out_cv_idx = np.array_split(subj_list, iter_num)
for i in range(iter_num):
    print 'Cross-Validation Outer Loop - ' + str(i)
    print '-----------------------------------'
    CV_dir = os.path.join(working_dir, 'Outer_CV_' + str(i))
    os.system('mkdir ' + CV_dir)
    # divide subjects into outer-train-set and outer-test-set
    outer_train_subj = list(out_cv_idx)
    outer_test_subj = outer_train_subj.pop(i)
    outer_train_subj = np.concatenate(outer_train_subj)
    pool = Pool(processes=10)
    pool.map(functools.partial(clf_gen, subj_list=outer_train_subj, 
                               zstat_data=zstat_data, roi_data=roi_data,
                               working_dir=CV_dir, roi_header=roi_header), range(100))
    pool.terminate()


