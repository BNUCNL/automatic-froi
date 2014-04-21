# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import matplotlib.pylab as pl
from scipy.stats.mstats import mquantiles
import scipy.ndimage as ndimage
from sklearn import preprocessing, svm, metrics

from mypy import roi as myroi
from mypy import math as mymath
from mypy import base as mybase
from froi.algorithm import segment as bpsegment


def make_priori(subj_list, zstat_data, roi_data):
    print 'Create a whole-fROI mask, a mask for each fROI' + \
          ' and peak probability map...'
    roi_data = roi_data[..., subj_list]
    zstat_data = zstat_data[..., subj_list]
    # create a mask
    mask = roi_data.sum(axis=3)
    mask[mask > 0] = 1
    # create a peak probability map and a mask for each fROI
    label_list = np.unique(roi_data)
    label_list = label_list.tolist()
    label_list.pop(0)
    label_num = len(label_list)
    prob = np.zeros((91, 109, 91, label_num))
    #roi_mask = np.zeros((91, 109, 91, label_num))
    for i in range(label_num):
        for j in range(len(subj_list)):
            mask_temp = roi_data[..., j].copy()
            mask_temp[mask_temp != label_list[i]] = 0
            mask_temp[mask_temp == label_list[i]] = 1
            temp = zstat_data[..., j] * mask_temp
            temp[temp < temp.max()] = 0
            x, y, z = np.nonzero(temp)
            if x:
                #print 'peak number: ' + str(len(x))
                for t in range(len(x)):
                    prob[x[t], y[t], z[t], i] += 1
            #roi_mask[..., i] += mask_temp
        prob[..., i] = ndimage.gaussian_filter(prob[..., i], 2)
    #roi_mask[roi_mask>0] = 1
    #print 'Probability map shape: ',
    #print prob.shape
    #return mask, prob, roi_mask
    return mask, prob


def segment(zstat_data):
    """
    Segment a zstats img.

    """
    seg_marker, seg_input, seg_result = bpsegment.watershed(zstat_data, 0, 2.3)
    return seg_marker, seg_result


def sample_extract(zstat_data, roi_data, mask_data, prob_data,  # roi_mask,
                   seg_marker, seg_result):
    """
    Extract samples from one subject, and return samples and labels.

    """
    samples = []
    labels = []
    parcel_id = []
    #-- sample labels
    # parcel size (number of voxels), overlap ratio with mask
    # peak x, y, z
    # roi probability for each fROI of peak
    # geometry center x, y, z, mass center x, y, z
    # coordinate: x_max, x_min, y_max, y_min, z_max, z_min
    # coordinate: eigen value 1-3, eigen vector 1-3 (x, y, z)
    # activation info: mean, std, min, max, median, first and third quartile
    # gradient info: mean, std, min, max, median, first and third quartile
    # ---------------
    # New Version
    # ---------------
    # parcel size (number of voxels), overlap ratio with mask
    # peak x, y, z
    # probability for each fROI of peak
    # overlap with each fROI mask
    # activation mean
    # Euclidean distance from peak of one parcel to the peak with largest
    # probaibility on each fROI
    #--
    seg_marker = seg_marker * mask_data
    marker_list = np.unique(seg_marker).tolist()
    marker_list.pop(0)
    print 'Parcel number: ' + str(len(marker_list))
    # if parcel voxel number is less than 3, drop it.
    new_marker_list = []
    for idx in marker_list:
        parcel_mask = seg_result.copy()
        parcel_mask[parcel_mask != idx] = 0
        parcel_mask[parcel_mask == idx] = 1
        parcel_voxels = parcel_mask.sum()
        if parcel_voxels < 3:
            print 'Voxel number in parcel ' + str(idx) + \
                  ' is less than 3, drop it!'
            seg_marker[seg_marker == idx] = 0
            continue
        else:
            intersection = mask_data * parcel_mask
            intersection_voxels = intersection.sum()
            overlap_r = float(intersection_voxels) / parcel_voxels
            if overlap_r < 0.5:
                continue
            else:
                new_marker_list.append(idx)
    # find voxel with largest probability on each fROI
    if new_marker_list:
        prob_num = prob_data.shape[3]
        roi_peak = np.zeros((prob_num, 3))
        peak_mask = seg_marker.copy()
        peak_mask[peak_mask > 0] = 1
        for idx in range(prob_num):
            prob_temp = prob_data[..., idx]
            prob_temp = prob_temp * peak_mask
            prob_temp[prob_temp < prob_temp.max()] = 0
            i, j, k = np.nonzero(prob_temp)
            if len(i):
                roi_peak[idx] = [i[0], j[0], k[0]]
            else:
                print 'No peak found on ROI ' + str(idx)
    # extract samples
    for idx in new_marker_list:
        parcel_mask = seg_result.copy()
        parcel_mask[parcel_mask != idx] = 0
        parcel_mask[parcel_mask == idx] = 1
        parcel_voxels = parcel_mask.sum()
        # a variable store features
        sample_item = []
        sample_item.append(parcel_voxels)
        # compute the intersection ratio between parcel and the mask
        intersection = mask_data * parcel_mask
        intersection_voxels = intersection.sum()
        overlap_r = float(intersection_voxels) / parcel_voxels
        sample_item.append(overlap_r)
        ## compute gradient map
        #gx = ndimage.sobel(zstat_data, 0)
        #gy = ndimage.sobel(zstat_data, 1)
        #gz = ndimage.sobel(zstat_data, 2)
        #grad_data = np.sqrt(gx**2 + gy**2 + gz**2)
        # get peak information
        parcel_peak = seg_marker.copy()
        parcel_peak[parcel_peak != idx] = 0
        peak_x, peak_y, peak_z = np.nonzero(parcel_peak)
        peak_x = peak_x[0]
        peak_y = peak_y[0]
        peak_z = peak_z[0]
        sample_item.append(peak_x)
        sample_item.append(peak_y)
        sample_item.append(peak_z)
        peak_roi_prob = prob_data[peak_x, peak_y, peak_z]
        for item in peak_roi_prob:
            sample_item.append(item)
        for i in range(prob_num):
            prob_dist = np.square([peak_x, peak_y, peak_z] - roi_peak[i]).sum()
            sample_item.append(prob_dist)
        #for i in range(prob_num):
        #    intersect = roi_mask[..., i] * parcel_mask
        #    intersect = intersect.sum()
        #    overlap_roi = float(intersect) / parcel_voxels
        #    sample_item.append(overlap_roi)
        ## get parcel morphometric information
        #geometric_center = myroi.geometric_center(parcel_mask)
        #gcenter_x = geometric_center[0]
        #gcenter_y = geometric_center[1]
        #gcenter_z = geometric_center[2]
        #sample_item.append(gcenter_x)
        #sample_item.append(gcenter_y)
        #sample_item.append(gcenter_z)
        #mass_center = myroi.mass_center(parcel_mask*zstat_data)
        #mcenter_x = mass_center[0]
        #mcenter_y = mass_center[1]
        #mcenter_z = mass_center[2]
        #sample_item.append(mcenter_x)
        #sample_item.append(mcenter_y)
        #sample_item.append(mcenter_z)
        #parcel_x, parcel_y, parcel_z = np.nonzero(parcel_mask)
        #x_max = parcel_x.max()
        #x_min = parcel_x.min()
        #sample_item.append(x_max)
        #sample_item.append(x_min)
        #y_max = parcel_y.max()
        #y_min = parcel_y.min()
        #sample_item.append(y_max)
        #sample_item.append(y_min)
        #z_max = parcel_z.max()
        #z_min = parcel_z.min()
        #sample_item.append(z_max)
        #sample_item.append(z_min)
        #x_coord, y_coord, z_coord = np.nonzero(parcel_mask)
        #coord_mtx = np.zeros((x_coord.shape[0], 3))
        #coord_mtx[:, 0] = x_coord
        #coord_mtx[:, 1] = y_coord
        #coord_mtx[:, 2] = z_coord
        #eigval, eigvtr = mymath.eig(coord_mtx)
        #eigval_1 = eigval[0]
        #eigval_2 = eigval[1]
        #eigval_3 = eigval[2]
        #eigvtr_1_x = eigvtr[0, 0]
        #eigvtr_1_y = eigvtr[1, 0]
        #eigvtr_1_z = eigvtr[2, 0]
        #eigvtr_2_x = eigvtr[0, 1]
        #eigvtr_2_y = eigvtr[1, 1]
        #eigvtr_2_z = eigvtr[2, 1]
        #eigvtr_3_x = eigvtr[0, 2]
        #eigvtr_3_y = eigvtr[1, 2]
        #eigvtr_3_z = eigvtr[2, 2]
        #sample_item.append(eigval_1)
        #sample_item.append(eigval_2)
        #sample_item.append(eigval_3)
        #sample_item.append(eigvtr_1_x)
        #sample_item.append(eigvtr_1_y)
        #sample_item.append(eigvtr_1_z)
        #sample_item.append(eigvtr_2_x)
        #sample_item.append(eigvtr_2_y)
        #sample_item.append(eigvtr_2_z)
        #sample_item.append(eigvtr_3_x)
        #sample_item.append(eigvtr_3_y)
        #sample_item.append(eigvtr_3_z)
        # get parcel label information
        parcel_label_data = parcel_mask * roi_data
        parcel_label_data = np.ma.array(parcel_label_data,
                                        mask=1 - parcel_mask)
        unique_label = np.ma.unique(parcel_label_data)
        label_list = unique_label.data.tolist()
        label_list = [label_list[j] for j in range(len(label_list))
                      if not unique_label.mask[j]]
        if 0 in label_list:
            label_list.pop(label_list.index(0))
            label_list.append(0)
        # if the parcel has more than one label, get the larger one
        if len(label_list) > 1:
            print 'The parcel has more than one label:',
            for lbl in label_list:
                print str(lbl) + ',',
            label_ratio = []
            for lbl in label_list:
                if lbl != 0:
                    temp = parcel_label_data.copy()
                    temp.data[temp.data != lbl] = 0
                    temp.data[temp.data == lbl] = 1
                    temp = temp.sum()
                    ratio = temp / parcel_voxels
                    label_ratio.append(ratio)
                else:
                    label_ratio.append(1.0 - sum(label_ratio))
            print str(sum(label_ratio))
            parcel_label = label_list[label_ratio.index(max(label_ratio))]
            print 'choose ' + str(parcel_label) + ' as label.'
        else:
            parcel_label = label_list[0]
        # get activation information
        parcel_activation = parcel_mask * zstat_data
        temp_data = np.ma.array(parcel_activation, mask=1 - parcel_mask)
        parcel_act_mean = temp_data.mean()
        #parcel_act_std = temp_data.std()
        #parcel_act_min = temp_data.min()
        #parcel_act_max = temp_data.max()
        #temp_data = temp_data.reshape((91*109*91, 1))
        #quartiles = mquantiles(temp_data)
        #parcel_act_first_quartile = quartiles[0]
        #parcel_act_median = quartiles[1]
        #parcel_act_third_quartile = quartiles[2]
        sample_item.append(parcel_act_mean)
        #sample_item.append(parcel_act_std)
        #sample_item.append(parcel_act_min)
        #sample_item.append(parcel_act_max)
        #sample_item.append(parcel_act_first_quartile)
        #sample_item.append(parcel_act_median)
        #sample_item.append(parcel_act_third_quartile)
        ## get gradient information
        #parcel_grad = parcel_mask * grad_data
        #temp_data = np.ma.array(parcel_grad, mask=1-parcel_mask)
        #parcel_grad_mean = temp_data.mean()
        #parcel_grad_std = temp_data.std()
        #parcel_grad_min = temp_data.min()
        #parcel_grad_max = temp_data.max()
        #temp_data = temp_data.reshape((91*109*91, 1))
        #quartiles = mquantiles(temp_data)
        #parcel_grad_first_quartile = quartiles[0]
        #parcel_grad_median = quartiles[1]
        #parcel_grad_third_quartile = quartiles[2]
        #sample_item.append(parcel_grad_mean)
        #sample_item.append(parcel_grad_std)
        #sample_item.append(parcel_grad_min)
        #sample_item.append(parcel_grad_max)
        #sample_item.append(parcel_grad_first_quartile)
        #sample_item.append(parcel_grad_median)
        #sample_item.append(parcel_grad_third_quartile)
        samples.append(sample_item)
        labels.append(parcel_label)
        parcel_id.append(idx)
    return samples, labels, parcel_id


def parcel_info(roi_data, mask_data, seg_marker, seg_result):
    """
    Extract parcel info from one subject, return voxel size,
    overlap ratio and labels.

    """
    samples = []
    labels = []
    #-- sample labels
    # parcel size (number of voxels), overlap ratio with mask
    #--
    seg_marker = seg_marker * mask_data
    marker_list = np.unique(seg_marker).tolist()
    marker_list.pop(0)
    print 'Parcel number: ' + str(len(marker_list))
    # extract parcel information
    for idx in marker_list:
        parcel_mask = seg_result.copy()
        parcel_mask[parcel_mask != idx] = 0
        parcel_mask[parcel_mask == idx] = 1
        parcel_voxels = parcel_mask.sum()
        # a variable store features
        sample_item = []
        sample_item.append(parcel_voxels)
        # compute the intersection ratio between parcel and the mask
        intersection = mask_data * parcel_mask
        intersection_voxels = intersection.sum()
        overlap_r = float(intersection_voxels) / parcel_voxels
        sample_item.append(overlap_r)
        # get parcel label information
        parcel_label_data = parcel_mask * roi_data
        parcel_label_data = np.ma.array(parcel_label_data,
                                        mask=1 - parcel_mask)
        unique_label = np.ma.unique(parcel_label_data)
        label_list = unique_label.data.tolist()
        label_list = [label_list[j] for j in range(len(label_list))
                      if not unique_label.mask[j]]
        if 0 in label_list:
            label_list.pop(label_list.index(0))
            label_list.append(0)
        # if the parcel has more than one label, get the larger one
        if len(label_list) > 1:
            print 'The parcel has more than one label:',
            for lbl in label_list:
                print str(lbl) + ',',
            label_ratio = []
            for lbl in label_list:
                if lbl != 0:
                    temp = parcel_label_data.copy()
                    temp.data[temp.data != lbl] = 0
                    temp.data[temp.data == lbl] = 1
                    temp = temp.sum()
                    ratio = temp / parcel_voxels
                    label_ratio.append(ratio)
                else:
                    label_ratio.append(1.0 - sum(label_ratio))
            print str(sum(label_ratio))
            parcel_label = label_list[label_ratio.index(max(label_ratio))]
            print 'choose ' + str(parcel_label) + ' as label.'
        else:
            parcel_label = label_list[0]
        samples.append(sample_item)
        labels.append(parcel_label)
    return samples, labels


def scatter_plot(sample_data, sample_label, feature_names, dest_dir):
    """
    Plot scatter of each pair of features.

    """
    # color configuration
    color_list = [(0, 0, 1),
                  (0, 1, 0),
                  (1, 0, 0),
                  (0, 0.75, 0.75),
                  (0, 0.5, 0),
                  (0.75, 0, 0.75),
                  (0, 0, 0),
                  (0.5, 0.5, 0.5),
                  (1, 1, 0),
                  (0.8, 0.8, 0.8),
                  (0, 1, 1),
                  (1, 0, 1),
                  (0.75, 0.75, 0)]

    sample_shape = sample_data.shape
    feature_num = sample_shape[1]
    print 'The dataset has %d features.' % feature_num
    class_list = np.unique(sample_label)
    # draw scatter plot
    for i in range(0, feature_num):
        for j in range(i + 1, feature_num):
            x_data = sample_data[:, i]
            y_data = sample_data[:, j]
            x_title = feature_names[i]
            y_title = feature_names[j]
            fig = pl.figure()
            pl.title(x_title + ' -- ' + y_title)
            for label in class_list:
                l = [index for index in range(x_data.shape[0])
                     if sample_label[index] == label]
                if label:
                    pl.scatter(x_data[l], y_data[l], marker='.',
                               edgecolors='none',
                               facecolors=color_list[int(label)])
                else:
                    pl.scatter(x_data[l], y_data[l], marker='o',
                               facecolors='none',
                               edgecolors=color_list[int(label)])

            file_name = os.path.join(dest_dir,
                                     x_title + '_' + y_title + '.png')
            fig.savefig(file_name)


def sample2file(file_path, header, data):
    f = open(file_path, 'w')
    f.write(','.join(header) + '\n')
    for line in data:
        temp = [str(item) for item in line]
        f.write(','.join(temp) + '\n')
    f.close()


def roi_svc_model_0(X_train, y_train, X_test, y_test):
    """
    An instance of multi-classes classifier -- model-0.
    Return the predict accuracy.

    """
    # data preprocessing
    y_train_bin = y_train.copy()
    y_train_bin[y_train_bin != 0] = 1
    scaler = preprocessing.Scaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # train the binary-classes classifier
    bin_svc = svm.SVC(C=1, kernel='rbf', cache_size=1000, class_weight='auto')
    bin_svc.fit(X_train, y_train_bin)

    # train the multi-classes classifier
    labeled_sample_idx = [idx for idx in range(y_train.shape[0])
                          if y_train[idx] != 0]
    X_train_mul = X_train[labeled_sample_idx, :]
    y_train_mul = y_train[labeled_sample_idx]
    mul_label_list = np.unique(y_train_mul)
    mul_label_list = mul_label_list.tolist()
    print mul_label_list
    mul_svc = svm.SVC(C=1, kernel='rbf', cache_size=1000, class_weight='auto')
    mul_svc.fit(X_train_mul, y_train_mul)

    # test the classifier using an independent dataset
    y_predict_bin = bin_svc.predict(X_test)
    selected_sample_idx = [idx for idx in range(y_predict_bin.shape[0])
                           if y_predict_bin[idx] != 0]
    y_predict_mul = mul_svc.predict(X_test[selected_sample_idx, :])

    # calculate the predict score
    y_predict = np.zeros((y_test.shape[0]))
    y_predict[selected_sample_idx] = y_predict_mul
    score = metrics.zero_one_score(y_test, y_predict)
    precision = metrics.precision_score(y_test, y_predict,
                                        labels=mul_label_list,
                                        pos_label=None)
    recall = metrics.recall_score(y_test, y_predict,
                                  labels=mul_label_list,
                                  pos_label=None)
    return score, precision, recall


def roi_svc(X_train, y_train, kernel='rbf', c=1, gamma=0):
    """
    An instance of 2-classes classifier.
    Return an instance of SVC and the corresponding data preprocessing.

    """
    # data preprocessing
    scaler = preprocessing.Scaler().fit(X_train)
    X_train = scaler.transform(X_train)

    # train the binary-classes classifier
    svc = svm.SVC(C=c, kernel=kernel, cache_size=1000,
                  gamma=gamma, class_weight='auto')
    svc.fit(X_train, y_train)

    return scaler, svc
