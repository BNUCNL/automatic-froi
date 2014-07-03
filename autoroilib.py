# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import nibabel as nib
import numpy as np
import re

def ext_feature(sid, mask_coord, prob_data):
    """
    Feature extraction.

    Input: Subject ID, a mask data (voxel coordinates in the mask),
           and a probabilistic activation volume.
    Source data: subject-specific zstat map,
                 subject-specific beta map (face and object condition),
                 subject-specific label,
                 probabilistic activation map,
                 and MNI standard brain.
    Output: sample table per subject.

    """
    #-- data preparation
    # zstat nad label file
    db1_dir = r'/nfs/t2/atlas/database'
    subject_dir = os.path.join(db_dir, sid, 'face-object')
    if not os.path.exists(subject_dir):
        print 'Subject ' + sid + 'does not exist in database.'
        return
    zstat_file = os.path.join(subject_dir, 'zstat1.nii.gz')
    label_file = get_label_file(subject_dir)
    # beta map
    # face - fix: cope4
    # object - fix: cope7
    db2_dir = r'/nfs/t2/fmricenter/volume'
    face_beta_file = os.path.join(db2_dir, sid, 'obj.gfeat', 'cope4.feat',
                                  'stats', 'cope1.nii.gz')
    object_beta_file = os.path.join(db2_dir, sid, 'obj.gfeat', 'cope7.feat',
                                    'stats', 'cope1.nii.gz')
    # mni brain template from fsl
    fsl_dir = os.getenv('FSL_DIR')
    mni_file = os.path.join(fsl_dir, 'data', 'standard',
                            'MNI152_T1_2mm_brain.nii.gz')

    # load data
    label_data = nib.load(label_file).get_data()
    zstat_data = nib.load(zstat_file).get_data()
    face_beta_data = nib.load(face_beta_file).get_data()
    object_beta_data = nib.load(object_beta_file).get_data()
    mni_data = nib.load(mni_file).get_data()

    # extract features for each voxel in the mask
    sample_data = []
    sample_num = len(mask_coord)
    
    # get neighbor offset
    neighbor_offset_1 = get_neighbor_offset(1)
    neighbor_offset_2 = get_neighbor_offset(2)
    neighbor_offset_3 = get_neighbor_offset(3)
    
    # 

    for idx in range()



def get_label_file(subject_dir):
    """
    Return subject-specific label file.

    """
    f_list = os.listdir(subject_dir)
    for f in f_list:
        if re.search('_ff.nii.gz', f):
            return os.path.join(subject_dir, f)

def get_neighbor_offset(radius):
    """
    Get neighbor offset for generating cubiods.

    """
    offsets = []
    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            for z in range(-radius, randius+1):
                offsets.append([x, y, z])
    return np.array(offsets)



