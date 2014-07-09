#!/usr/bin/env python
"""
=======================================================================
Random forest trainning and cv for paramter selection.

Input  : a sess list with train and cv.
Output : a pair of best parameter forest classifier.

TODO:
    1. Cross-splite the sess list for train and cv by combining the sample parts
       trainning part and cv part.
    2. feature scaling and setting.
    3. fit the training dataset to get clf, using DICE score for evaluation.
    4. average the scores with fold and different classes.
    5. save the best parameter of clf and write scores of different fold and classes.
    
    Note : manual change the parameter for grid search.
Paramter: the number of the forest N and the maximum depth of the tree.
=======================================================================
"""

import numpy as np
import nibabel as nib
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split,KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import cross_validation
from sklearn.metrics import average_precision_score,roc_auc_score,\
                            precision_score,recall_score,f1_score

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

def load_train(sess):
    dict = {}
    for sid in sess:
        dict[sid]=np.loadtxt('samples_dir/sample_%s'%sid)
    return dict

def load_test(sess):
    dict = {}
    for sid in sess:
        dict[sid]=np.loadtxt('samples_dir/sample_%s'%sid)
    return dict


def combin_sample(sess,dict):
    sam = np.array([])
    flag = 0
    for i in sess:
        se = dict[i]
        if flag==0:
            sam = se
            flag = 1
        else:
            sam= np.vstack((sam,se))
 #   print sam.shape
    return sam

def read_test_sample(id):
    se = np.loadtxt('samples_dir/sample_%s'%id)
    return se

def testset_predict(sess,clf,mean,std,masks,feat_s,fold,img):
    print "test subject IDs:",sess
    flag = 0
    metric_all = np.array([])
    for sub in sess:
        metric = np.ones(12)
        test_set = read_test_sample(sub)
        #1. mask z>2.3
        #mask_c = test_set[:,0]>=2.3
        #TE = test_set[mask_c]
        TE = test_set
        #2. split x and y
        X_cv = TE[:,masks]
        y_cv = TE[:,-4]
        coor = TE[:,-3:]
        #3. Standardize
        X_true = (X_cv - mean)/std
        y_true = y_cv
        y_pred = clf.predict(X_true)
        
        list = [0,1,3]
        for i,l in enumerate(list):
            #print y_p
            #print y_t
            P = y_pred== l
            T = y_true== l
            #print "A:",A
            #print "B:",B
            di = dice(P,T)
            #print di
            if np.isnan(di):
                metric[i] = 1
            else:
                metric[i] = di
        #4. Compute metrics
        precision = precision_score(y_true, y_pred,average=None)
        recall = recall_score(y_true, y_pred,average=None)
        f1 = f1_score(y_true,y_pred,average=None)
        
        #print 'precision,recall,f1',precision,recall,f1
        if len(precision)==len(list):
            metric[3:6] = precision
        if len(recall)==len(list):
            metric[6:9] = recall
        if len(f1)==len(list):
            metric[9:12] = f1
        
        #print metric

        if flag==0:
            metric_all = metric
            flag = 1
        else:
            metric_all = np.vstack((metric_all,metric))

        tmp = np.zeros_like(img.get_data())

        for j,c in enumerate(coor):
            tmp[tuple(c)] = y_pred[j]
        img._data = tmp
        nib.save(img,"pred_%s/predicted_%s.nii.gz"%(feat_s,sub))
        print 'saving predicted_%s done!'%sub
    
    print metric_all
    filename = "pred_%s/testset_metrics_fold%d.txt"%(feat_s,fold)
    np.savetxt(filename,metric_all,fmt='%1.4e',delimiter=',',newline='\n')
    print "Metric score for test set have been saved in the %s"%filename

def dice(r,p):
    return 2.0*np.sum(r*p)/(np.sum(r)+np.sum(p))

def main():
    #read train and cv part sess list
    sa = time.time()
    sess = read_sess_list('./sess')
    sample = load_train(sess)
    print 'load data use:',time.time()-sa

    coor = np.loadtxt('./coordinates')
   # print coor,coor.shape
    #rean img template
    img = nib.load('MNI_brain.nii.gz')
    data = img.get_data()

    print "train+cv number:",len(sess)
    sa = time.time()
    
    m1 = range(0,4)
    m2 = range(0,114)
    m3 = range(0,225)
    mask_all = [m1,m2,m3]
    print mask_all

    for fe,masks in enumerate(mask_all):
        print masks
        #initial the parameter
        T = [1,5,10,20,30,40,50,60]
        #T = [1,5,10,30,50]
        D = [1,5,10,15,20,25]
        #D = [1,5,10,15,20]
        #T = [1,50]
        #D = [2,25]
        
        param = [[t,d] for t in T for d in D]
        #print param
        
        #second level cv
        kf = KFold(len(sess), n_folds=3, indices=False)
        k = 0
        Test_Metrics = []
        fflag = 0
        for train, test in kf:
            print 'K-fold: ',k,fe
            #parameter selection for second level cross validation.
            #for a pair of parameter,do kFold cross validation and compute the DICE
            Metrics = np.zeros((48,10)) # average all fold and classes. mean and std.
            sess_train = sess[train]
            sess_test  = sess[test]
            #-------------------------------parameter selection part-----------------------------
        
            sk = time.time()
            for i,p in enumerate(param):
                print 'parameter:',p
                
                Metrics[i][0] = p[0]
                Metrics[i][1] = p[1]
                metric_all = np.array([]) #record the classes Dice detail of specific parameter.
                flag = 0

                #fisrt level cv.
                st = time.time()
                kfn = KFold(len(sess_train), n_folds=3, indices=False)
                for train_m, cv_m in kfn:
                    sp = time.time()
                    #print("%s %s" % (train_m, cv_m))
                    TR_sample = combin_sample(sess_train[train_m],sample)
                    CV_sample = combin_sample(sess_train[cv_m],sample)
                    #intensity
                    #mask_t = TR_sample[:,0]>=2.3
                    #mask_c = CV_sample[:,0]>=2.3
                    #TR = TR_sample[mask_t]
                    #CV = CV_sample[mask_c]
                    TR = TR_sample
                    CV = CV_sample
                    #feature select
                    X = TR[:,masks]
                    y = TR[:,-4]
                    X_c = CV[:,masks]
                    y_c = CV[:,-4]
                    print X.shape,y.shape,X_c.shape,y_c.shape
                    
                    # Shuffle
                    idx = np.arange(X.shape[0])
                    np.random.seed(10)
                    np.random.shuffle(idx)
                    X = X[idx]
                    y = y[idx]

                    # Standardize
                    mean = X.mean(axis=0)
                    std = X.std(axis=0)
                    X_train = (X - mean)/std
                    X_cv = (X_c - mean)/std
                    y_train = y
                    y_cv = y_c
                 
                    print "time prepare sample once used:%s"%(time.time()-sp)
                    
                    # training model and save model.
                    clf = RandomForestClassifier(bootstrap=True,criterion='entropy',max_depth=p[1],
                        max_features='sqrt',n_estimators=p[0],min_samples_split=1, n_jobs=100)
                
                    tt = time.time()
                    clf.fit(X_train,y_train)
                    #print "feature importances:",clf.feature_importances_
                    print "fit once used:%s"%(time.time()-tt)
                
                    tt = time.time()
                    y_pred = clf.predict(X_cv)
                    print "predict once used:%s"%(time.time()-tt)
                    #y_pp = clf.predict_proba(X_test)
                
                    y_true = y_cv
                    #print(classification_report(y_true, y_pred))
                
                    #compute dice
                    tt = time.time()
                    metric = []
                    labels = [0,1,3]
                        
                    precision = precision_score(y_true, y_pred,average=None)
                    recall = recall_score(y_true, y_pred,average=None)
                    f1 = f1_score(y_true,y_pred,average=None)
                    #print 'precision,recall,f1',precision,recall,f1

                    for j in labels:
                        P = y_pred==j
                        T = y_true==j
                        di = dice(P,T)
                        #print 'Dice for label:',j,di
                        metric.append(di)

                    metric = np.array(metric)
                    metric = np.hstack((metric,precision))
                    metric = np.hstack((metric,recall))
                    metric = np.hstack((metric,f1))
                    print 'Dice,precision,recall,f1',metric

                    if flag==0:
                        metric_all = metric
                        flag = 1
                    else:
                        metric_all = np.vstack((metric_all,metric))
                    print "time compute metrics once used:%s"%(time.time()-tt)

                print 'Metrics score for parameter pair:',p,metric_all
                #stat the metrics and save the score results.
                #average cross the classes
                Dice_mean = np.mean(metric_all[:,0:3],axis=1)
                Precision_mean = np.mean(metric_all[:,3:6],axis=1)
                Recall_mean = np.mean(metric_all[:,6:9],axis=1)
                F1_mean = np.mean(metric_all[:,9:12],axis=1)

                print Dice_mean
                print Precision_mean
                print Recall_mean
                print F1_mean
                
                #average and std cross the k-fold
                Dice_m = np.mean(Dice_mean)
                Dice_s = np.std(Dice_mean)
                Precision_m = np.mean(Precision_mean)
                Precision_s = np.std(Precision_mean)
                Recall_m = np.mean(Recall_mean)
                Recall_s = np.std(Recall_mean)
                F1_m = np.mean(F1_mean)
                F1_s = np.std(F1_mean)
                
                print 'Dice mean:', Dice_m
                print 'Dice std:', Dice_s
                print 'Precision mean:', Precision_m
                print 'Precision std:', Precision_s
                print 'Recall mean:', Recall_m
                print 'Recall std:', Recall_s
                print 'F1 score mean:', F1_m
                print 'F1 score std:', F1_s
                
                Metrics[i][2] = Dice_m
                Metrics[i][3] = Dice_s
                Metrics[i][4] = Precision_m
                Metrics[i][5] = Precision_s
                Metrics[i][6] = Recall_m
                Metrics[i][7] = Recall_s
                Metrics[i][8] = F1_m
                Metrics[i][9] = F1_s

                
                filename = "score/fold_%d_p_%d_%d.txt"%(k,p[0],p[1])
                np.savetxt(filename,metric_all,fmt='%1.5e',delimiter=',',newline='\n')
                print "metrics score for every class and flod have been saved in the %s"%filename
                print "first level cv once, time used:%s"%(time.time()-st)
            
            print "Select parameter for one fold, time used:%s"%(time.time()-sk)
            print Metrics
            
            filename = "Metrics_%d_fold.txt"%k
            np.savetxt(filename,Metrics,fmt='%1.5e',delimiter=',',newline='\n')
        
            #-----------------------------------------test part------------------------------------------
            #compute dice for every single test subject. compute the score and save the prediction as .nii
            #using the optimized paramter train and predict the testset.
            ss = time.time()
            #print("%s %s" % (train_m, cv_m))
            TR_sample = combin_sample(sess_train,sample)
            TE_sample = combin_sample(sess_test,sample)
            #intensity
            #mask_t = TR_sample[:,0]>=2.3
            #mask_e = TE_sample[:,0]>=2.3
            #TR = TR_sample[mask_t]
            #TE = TE_sample[mask_e]
            TR = TR_sample
            TE = TE_sample

            #feature select
            X = TR[:,masks]
            y = TR[:,-4]
            X_t = TE[:,masks]
            y_t = TE[:,-4]
            print X.shape,y.shape
            print y, np.unique(y_t)

            # Shuffle
            idx = np.arange(X.shape[0])
            np.random.seed(10)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # Standardize
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X_train = (X - mean)/std
            y_train = y
            X_test = (X_t - mean)/std
            y_test = y_t
            #depth = raw_input('input the best maximum depth  :>>>')
            #ntree = raw_input('input the best number of tree :>>>')
            depth = 20
            ntree = 50

            clf = RandomForestClassifier(bootstrap=True,criterion='entropy',max_depth=depth,
                    max_features='sqrt',n_estimators=ntree,min_samples_split=1, n_jobs=100)
            
            clf.fit(X_train,y_train)
            print "feature importances:",clf.feature_importances_
            filename = "feat_importance_%s_%d_fold.txt"%(fe,k)
            np.savetxt(filename,clf.feature_importances_,fmt='%1.5e',delimiter=',',newline='\n')
            
            tt = time.time()
            y_pred = clf.predict(X_test)
            print "predict once used:%s"%(time.time()-tt)
        
            y_true = y_test
            print(classification_report(y_true, y_pred))
                
            #compute dice
            metric = []
            labels = [0,1,3]
                        
            precision = precision_score(y_true, y_pred,average=None)
            recall = recall_score(y_true, y_pred,average=None)
            f1 = f1_score(y_true,y_pred,average=None)
            #print 'precision,recall,f1',precision,recall,f1

            for j in labels:
                P = y_pred==j
                T = y_true==j
                di = dice(P,T)
                #print 'Dice for label:',j,di
                metric.append(di)

            metric = np.array(metric)
            metric = np.hstack((metric,precision))
            metric = np.hstack((metric,recall))
            metric = np.hstack((metric,f1))
            print 'Dice,precision,recall,f1',metric
            
            if fflag==0:
                Test_Metrics = metric
                fflag = 1
            else:
                Test_Metrics = np.vstack((Test_Metrics,metric))

            testset_predict(sess_test,clf,mean,std,masks,fe,k,img)
            print "test time used:%s"%(time.time()-ss)
            k+=1
        

        print 'Test_Metrics',Test_Metrics
        filename = "test_metric_feat_%d.txt"%(fe)
        np.savetxt(filename,Test_Metrics,fmt='%1.5e',delimiter=',',newline='\n')
        print "metrics score for test have been saved in the %s"%filename
    print "all time used:%s"%(time.time()-sa)
    return

if __name__=="__main__":
    main()
