#!/usr/bin/env python
#Filename: feature_ext.py
"""
======================================================================
Feature extraction module.

Input:     subject zstat map and prior brain image
           zstate.nii.gz and MNI_brain_standard.nii.gz

Output:    sample dataset file per Subject
           train_sid.txt
           test_sid.txt

           Feature types:
           1.r         (spherical coordinates)
           ...

Author: dangxiaobin@gmail.com
Date:   2013.12.25
=======================================================================
"""

print (__doc__)

import os
import sys
import re
import time
import nibabel as nib
import numpy as np
import multiprocessing as mul
import functools 

def main():
    #data reposition and mask constraint
    database = '/nfs/t2/atlas/database'
    contrast = 'face-object'
    sess_list = read_sess_list('./sess')
    roi_mask = nib.load('./pre_mask.nii.gz')
    mask_data = roi_mask.get_data()
    
    #get cubiods mask by max/min the spatical range.
    non = mask_data.nonzero()
    coor_r = np.zeros(6)
    coor_r[0] = np.min(non[0])
    coor_r[1] = np.max(non[0])
    coor_r[2] = np.min(non[1])
    coor_r[3] = np.max(non[1])
    coor_r[4] = np.min(non[2])
    coor_r[5] = np.max(non[2])
    mask_index,mask_img = get_mask(coor_r,mask_data)
    cx = np.median(non[0])
    cy = np.median(non[1])
    cz = np.median(non[2])
    
    cn = [cx,cy,cz]
    print mask_index.shape,cn

    '''
    brain_m = nib.load("brain_mask.nii.gz")
    data = brain_m.get_data()
    mask_f = data*mask_img
    mask_index = np.transpose(mask_f.nonzero())
    print mask_index.shape
    '''

    final_mask = roi_mask
    final_mask._data = mask_img
    nib.save(final_mask,'final_mask.nii.gz')

    #output the mask coordinate indexs
    writeout(mask_index)

    st = time.time()
    subject_num = len(sess_list)
    sample_num = len(mask_index) # per subject
    
    sub_stat = {}

    #get neighbor offset 1,2,3 radiud cubois.
    of_1 = get_neighbor_offset(1)
    of_2 = get_neighbor_offset(2)
    of_3 = get_neighbor_offset(3)
    #print offset_1.shape, offset_2.shape, offset_3.shape
    vecs = np.array([[1,0,0],
                     [-1,0,0],
                     [0,1,0],
                     [0,-1,0],
                     [0,0,1],
                     [0,0,-1],
                     [2,0,0],
                     [-2,0,0],
                     [0,2,0],
                     [0,-2,0],
                     [0,0,2],
                     [0,0,-2],
                     [3,0,0],
                     [-3,0,0],
                     [0,3,0],
                     [0,-3,0],
                     [0,0,3],
                     [0,0,-3]])
    print vecs,vecs.shape
    
    r_os = np.zeros((54,4))
    for i in range(0,54):
        r_os[i][0] = np.random.randint(0,18)
        tmp= np.random.randint(0,18)
        while tmp==r_os[i][0]:
            tmp= np.random.randint(0,18)
        r_os[i][1] = tmp
        r_os[i][2] = np.random.randint(0,2)
        r_os[i][3] = np.random.randint(0,2)
    print r_os
    ##functions test
    #img = nib.load("MNI_brain.nii.gz")
    #data = img.get_data()
    #m = [[44,22,33],[44,23,33],[44,23,22]]
    #u = get_mean(data,m)
    #print m,u
    #offset = get_neighbor_offset(1)
    #print offset,offset.shape
    #offset = get_neighbor_offset(3)
    #print offset,offset.shape
    #########################################################################
    
    #feature extraction on train set.
    #save the every subject sample per file named sample_sid.txt
    pool = mul.Pool(30)
    result = pool.map(functools.partial(ext,database=database,mask_index=mask_index,
        of_1=of_1,of_2=of_2,of_3=of_3,of_vec=vecs,r_os=r_os), sess_list)
    print result
    sf = open('sub_filt','wb')
    for s in result:
        if s!= 'F':
            sf.write(s+'\n')
    sf.close()

    print "Feature extraction total time:%s"%(time.time()-st)
    return

def ext(sub=None,database=None,mask_index=None,of_1=None,of_2=None,of_3=None,of_vec=None,r_os=None):
    """
    Wrapper function.
    Tested
    """
    print "Starting extraction: %s"%sub
    contrast = 'face-object'
    sub_dir =  os.path.join(database,sub)
    sub_ct_dir = os.path.join(sub_dir,contrast)
    f_list = os.listdir(sub_ct_dir)
    for file in f_list:
        if re.search('zstat1.nii.gz',file):
            x_img =  os.path.join(sub_ct_dir,file)
           # print x_img
        if re.search('_ff.nii.gz',file): 
            y_img =  os.path.join(sub_ct_dir,file)
           # print y_img
    #initial the feature array.
    #3+3+3+3+30+30+20+1=93
    feat_buff = np.zeros(229)
    samples = feature_ext(x_img,y_img,mask_index,
                          of_1,of_2,of_3,of_vec,r_os,feat_buff)
    #mask = samples[:,92]!=0
    #print samples[mask,92]
    #output samples as a fiile
    length = len(samples)
    if length >=27:
        mask1 = samples[:,-4]==1
        mask2 = samples[:,-4]==3
        if np.sum(mask1)>=27 and np.sum(mask2)>=27:
            print np.sum(mask1),np.sum(mask2),length
            print samples
            np.savetxt('samples_dir/sample_%s'%sub,samples,fmt='%10.5f',delimiter='    ',newline='\n')
            return sub
    return 'F'

def writeout(mask_index):
    """
    Write mask coordinates out.
    Tested
    """
    coorf = open('coordinates','w')
    for c in mask_index:
        coorf.write("%d %d %d\n"%(c[0],c[1],c[2]))
    coorf.close()
    return 0

def get_neighbor_offset(radius):
    """
    get neighbor offset for generating cubiods.
    Tested
    """
    offsets = [] 
    for x in np.arange(-radius,radius+1):
        for y in np.arange(-radius,radius+1):
            for z in np.arange(-radius,radius+1):
                offsets.append([x,y,z])
    offsets = np.array(offsets)
    return offsets

def feature_ext(sub_x,sub_y,mask_index,os1,os2,os3,of_vec,r_os,feat):
    """
    Feature extraction fucntion.
    """
    #offset vectors
    vecs = of_vec
    #load the zstat image, labeled image and standard brain image
    x = nib.load(sub_x)
    y = nib.load(sub_y)
    MNI = nib.load("./MNI_brain.nii.gz")
    #prior = nib.load("./prob_allsub_sthr0.nii.gz")
    
    # x data -> zstat
    x_data = x.get_data()
    # y data -> label
    y_data = y.get_data()
    # s data -> mni structure
    s_data = MNI.get_data()
    # p data -> probability
    #p_data = prior.get_data()

    #center of the region.
    c = [24.0, 28.0, 27.0]
    feat_all = []
    flag = 0
    #one coordinate one feature vec
    for i,coor in enumerate(np.array(mask_index)):
        #st = time.time()
        #1.local f (0-3:x,y,z,v)
        #print i,coor
        I = x_data[tuple(coor)]
        if I < 2.3:
            continue
        feat[226] = coor[0]
        feat[227] = coor[1]
        feat[228] = coor[2]
        
        feat[0] = I
        
        x = coor[0] - c[0]
        y = coor[1] - c[1]
        z = coor[2] - c[2]
        r,t,p = spherical_coordinates(x,y,z)
        
        feat[1] = r
        feat[2] = t
        feat[3] = p

        S = s_data[tuple(coor)]
        feat[114] = S
        
        #context feature:1.neighord smooth
        neigh1 = coor + os1
        neigh2 = coor + os2
        #neigh3 = coor + os3
        
        mean1 = get_mean(x_data,neigh1)
        mean2 = get_mean(x_data,neigh2)
        #mean3 = get_mean(x_data,neigh3)
        feat[4] = mean1
        feat[5] = mean2
        #feat[i][6] = mean3
        
        mean_s1 = get_mean(s_data,neigh1)
        mean_s2 = get_mean(s_data,neigh2)
        #mean_s3 = get_mean(s_data,neigh3)
        feat[115] = mean_s1
        feat[116] = mean_s2
        #feat[i][154] = mean_s3
        #print feat[i]
        #context feature:2.compare with offset region
        for j,v in enumerate(vecs):
            #print j,v
            p = v + coor
            pn1 = p + os1
            pn2 = p + os2
           # pn3 = p + os3
            feat[6+j] = I - get_mean(x_data,pn1)
            feat[24+j] =  mean1 - get_mean(x_data,pn1)
            feat[42+j] =  mean2 - get_mean(x_data,pn2)
          #  feat[i][61+j] =  mean3 - get_mean(x_data,pn3)
            
            feat[117+j]  =  S - get_mean(s_data,pn1)
            feat[135+j] =  mean_s1 - get_mean(s_data,pn1)
            feat[153+j] =  mean_s2 - get_mean(s_data,pn2)
            #feat[i][209+j] =  mean_s3 - get_mean(s_data,pn3)
            #break
        #print feat[i]
        #context feature:3.compare the two offset regions
        
        os = [os1,os2]
        for j,rand in enumerate(r_os):
            #print rand
            vs = vecs[rand[0]]
            vt = vecs[rand[1]]
            p1 = vs + coor
            p2 = vt + coor
            pn1 = p1 + os[int(rand[2])]
            pn2 = p2 + os[int(rand[3])]
            
            feat[60+j] = get_mean(x_data,pn1) - get_mean(x_data,pn2)
            feat[171+j]= get_mean(s_data,pn1) - get_mean(s_data,pn2)
        
        #Label 1/0
        label = y_data[tuple(coor)]
        if label == 1 or label == 3:
            feat[225] = label
        #    print feat[i],feat[i].shape
        else:
            feat[225] = 0
       # print 'time:',time.time()-st
        if flag == 0:
            feat_all = feat
            flag = 1
        else:
            feat_all = np.vstack((feat_all,feat))
    return  feat_all

def output_sess_list(sess,list):
    """
    Output session list for data split
    Tested
    """
    file = open(sess,'w')
    for l in list:
        file.write("%s\n"%l)
    file.close()
    return True

def is_inside(v, shape):
    """
    Is coordinate inside the image.
    Tested
    """
    return ((v[0] >= 0) & (v[0] < shape[0]) &
            (v[1] >= 0) & (v[1] < shape[1]) &
            (v[2] >= 0) & (v[2] < shape[2]))

def get_mean(img,indexs):
    """
    Get mean intensity with in mask. 
    Tested
    """
    intensity = 0
    for c in indexs:
        #print c,img[tuple(c)]
        intensity+= img[tuple(c)] 
    return intensity/len(indexs)

def get_mask(c_r,img):
    """
    Get cuboids region mask as spacial constraint.
    Tested
    """
    indexs = []
    mask = np.zeros_like(img)
    shape = mask.shape
    for x in np.arange(c_r[0],c_r[1]+1):
        for y in np.arange(c_r[2], c_r[3]+1):
            for z in np.arange(c_r[4], c_r[5]+1):    
                tmp = [x,y,z]
                inside = is_inside(tmp,shape)
                if inside :
                    indexs.append(tmp)
                    mask[x][y][z]=1
    return np.array(indexs),mask

def spherical_coordinates(x,y,z):
    """
    Rectangular coordinate to spherical coordinate.
    Tested
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if z==0:
        theta = np.arctan(np.sqrt(x**2+y**2)/(z+0.000001))
    else:
        theta = np.arctan(np.sqrt(x**2+y**2)/z)
    if x==0:
        phi = np.arctan(float(y)/(x+0.000001))
    else:
        phi = np.arctan(float(y)/x)
       
    return r,theta,phi

def distance(c,t):
    """
    Compute the spacial distance.
    Tested
    """
    return np.sqrt((c[0]-t[0])**2+(c[1]-t[1])**2+(c[2]-t[2])**2)

def read_sess_list(sess):
    """
    Load subject Id list.
    Tested
    """
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\r\n') for line in sess]
    sess = np.array(sess)
    print sess
    return sess

if __name__=="__main__":
    main()
