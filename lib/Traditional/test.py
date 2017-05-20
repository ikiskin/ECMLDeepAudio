from DataProcessing.SyntheticData import SyntheticData
import os
import numpy as np
cwd = os.getcwd()
from Traditional.Features import FeatureLabelGen

# Initialize the synthetic data object and generate the dataset
data_obj = SyntheticData(N_rec = 100, N_sample_per_rec=8000*10)
data = data_obj.generateData()

# Create training set
labelled_dataset = data['labelled_dataset']
signal_list_train = labelled_dataset['signal_list']
label_list_train = labelled_dataset['true_label_list']

# test.py data set
unlabelled_dataset = data['unlabelled_dataset']
signal_list_test = unlabelled_dataset['signal_list']
label_list_test = unlabelled_dataset['true_label_list']

# # Initialize and train detectors
detector_used = 'SVM'#'CNN'
if detector_used == "NB":
    from sklearn.naive_bayes import GaussianNB
    fs,winlen,winstep = 8000,0.089,0.0445
    feat_lst,proc_lst = ["mfcc"],[]
    clf = GaussianNB()
elif detector_used == "RF":
    from sklearn.ensemble import RandomForestClassifier
    fs,winlen,winstep = 8000,0.058,0.029
    feat_lst = ['mfcc', 'mel', 'energy', 'specentropy', 'specflux', 'specspreadcent', 'zcr', 'specrolloff', 'energyentropy', 'spec']
    proc_lst = ["del_mask"]
    clf = RandomForestClassifier(n_estimators=200,verbose=1,n_jobs=-1)
elif detector_used == "SVM":
    from sklearn import svm
    fs,winlen,winstep = 8000,0.089,0.0445
    feat_lst = ['mfcc', 'mel', 'energy', 'specentropy', 'specflux', 'specspreadcent', 'zcr', 'specrolloff', 'energyentropy', 'spec']
    proc_lst = ["del_mask","normalise"]
    clf = svm.SVC(verbose=1)
if detector_used in ["NB","RF","SVM"]:
    from sklearn.metrics import f1_score
    feats_trn, labels_trn, nrm_dat = FeatureLabelGen(signal_list_train,label_list_train,fs,winlen,winstep,feat_lst,proc_lst)
    feats_tst, labels_tst, _ = FeatureLabelGen(signal_list_test,label_list_test,fs,winlen,winstep,feat_lst,proc_lst,nrm_dat)
    clf.fit(feats_trn,labels_trn)
    predict_labels = clf.predict(feats_tst)
    print("F1 Score: %.3f"%f1_score(labels_tst,predict_labels))