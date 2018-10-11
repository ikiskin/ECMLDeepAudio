# Custom packages
import sys 
import os, os.path
sys.path.append('lib/kiswav')
import bumpwavelet_minimal as bw

# Library modules

from scipy.io.wavfile import read, write
import csv
import matplotlib.pyplot as plt
import numpy as np

# Keras-related imports
from keras.models import Sequential
from keras.regularizers import WeightRegularizer
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution1D, MaxPooling2D, Convolution2D
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint
from keras.callbacks import RemoteMonitor
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Data post-processing and analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from scipy.signal import medfilt



# File import 

def import_file(data_path, label_path):
    dirs = os.listdir(data_path)
    dirs_label = os.listdir(label_path)


    label_list = []
    label_names = []

    # Process .txt labels. Create list with structure:
    # list of [start_time, end_time, tag] entries
    signal_list = []

    for file in dirs:
        if file.endswith('.wav'):
            fs, signal = read(data_path + file)
            # print 'Processing', file
            signal_list.append(signal)
    print 'Processed %i files.'%len(signal_list) 
           
    for file in dirs_label:
        if file.endswith('.txt'):
            filename = label_path + file
            with open(filename) as f:
                reader = csv.reader(f, delimiter = ' ')
                label_list.append(list(reader)) # Class labels
                label_names.append(file) # Save class name in separate array
    print 'Processed %i labels.'%len(label_list) 
    return signal_list, label_list, label_names, fs


# Data label processing


# Select majority voting or other options
def proc_label(signal_list, label_list, label_names, select_label = 5, maj_vote = False):

    select_label = 5 # 0: ac, 4 cm, 5 dz, 6 ms
    count_method = 'dz'

    t = np.array(label_list[0])
    print label_names


    t_array = []
    t = []    

    print 'Using label', label_names[select_label]
    for index, item in enumerate(label_list[select_label]):
        t_entry = [float(i) for i in item]
    #     t_entry = np.repeat(t_entry,int(label_interval*fs))  # Upsample labels to match sample rate of signal
        t.append(t_entry[:len(signal_list[index-1])])    # -1 Corrects for extra label present

    t = np.delete(t, (0), axis = 0)
    # t_array.append(np.array(t))
    return t

# Training, test signal split for ECML paper

def data_split(signal_list, t, test_idx):
    train_idx = np.arange(len(t))
    train_idx = np.delete(train_idx, test_idx)

    test_x = []
    test_t = []
    train_x = []
    train_t = []

    for idx in test_idx:
        test_x.append(signal_list[idx])
        test_t.append(t[idx])

    for idx in train_idx:
        train_x.append(signal_list[idx])
        train_t.append(t[idx])
    return train_x, train_t, test_x, test_t



# Feature extraction function

def proc_data_humbug(signal_list, t, fs, img_height = 256, img_width = 10, nfft = 512, overlap = 256, label_interval = 0.1):
               
    """Returns the training data x,y and the parameter input_shape required for initialisation of neural networks. 
    
    Takes as input arguments a signal list, and its associated time coordinates t. Would probably be better to integrate into
    one function, however remains separate as a t input was not required for the birdsong application (the origin of this 
    function). 
    
    img_height and width are used to determine the dimensions of the spectrogram chunk required as input to the NN. nfft and 
    overlap (optional) parameterise the spectrogram.
    
    label_interval: time interval between labels (in seconds)
    """

    spec_list = []
    t_list = []
    spec_fs = (1.*fs/(nfft - overlap))
    location_to_remove = []
    
    for i, t_i in enumerate(t):

        SpecInfo = plt.specgram(signal_list[i], Fs = fs, NFFT = nfft, noverlap = overlap)
        spec_list.append(np.array(SpecInfo))
        
        t_initial = np.arange(0, len(t_i))
        t_final = np.arange(0, len(t_i),1./(label_interval*spec_fs)) # Match fs_labels to fs_spec
        t_corrected = np.interp(t_final, t_initial, t_i)        
        t_corrected = np.rint(t_corrected)
        
        location = np.where(t_corrected == 3)
        #print 'length of t_corrected', len(t_corrected)
        location_to_remove.append(location)
        t_list.append(t_corrected)
   
    # Removal of samples where there is no mutual agreement, labelled by a 3 in pre-processing
    
    for index, item in enumerate(spec_list):
        t_spec = np.delete(spec_list[index][2], location_to_remove[index])
        f_spec = np.delete(spec_list[index][0], location_to_remove[index], axis = 1)
        short_t = np.delete(t[index], location_to_remove[index])
        
        # print index, location_to_remove[index], 'Cut', np.shape(f_spec), 'Original', np.shape(spec_list[index][0])
        
        spec_list[index][0] = f_spec
        spec_list[index][2] = t_spec
###################################################################################################################    
    
    # return spectrogram frequencies for visualisations. Not necessary for minimal working example    
    spec_freq = SpecInfo[1][:img_height]
    
    # Dimensions for image fed into network
    nb_classes = 2
    x_train = []
    y_train = []


    for i in np.arange(len(spec_list)):
        n_max = np.floor(np.divide(np.shape(spec_list[i][0])[1],img_width)).astype('int')

        #print 'Processing signal number', i
        #print 'Length of spec', np.shape(spec_list[i][0])[1]
        #print 'Number of training inputs for this signal:', n_max    
        for n in np.arange(n_max):
            #print 'n', n, np.shape(spec_training_list[i][0])
            x_train.append(spec_list[i][0][:img_height,img_width*n:img_width*(n+1)])
            # Match sampling rate of labels to sampling rate of spectrogram output

            if not t_list[i][img_width*n:img_width*(n+1)].size:
                print spec_list[i][0][:img_height,img_width*n:img_width*(n+1)]
                print 'start and stop indices for t and spec', img_width*n, img_width*(n+1)
                plt.figure()
                plt.plot(t_list[i])
                plt.show()
                
                #raise ValueError('No location label found for signal number %i'%i) 
                
            y_train.append(t_list[i][img_width*n:img_width*(n+1)])
    
    
    y_train_mean = np.zeros([np.shape(y_train)[0], nb_classes])
    x_train = np.array(x_train).reshape((np.shape(x_train)[0],1,img_height,img_width))
        
    for ii, i in enumerate(y_train):
        if i.size:
            y_train_mean[ii,0] = int(np.round(np.mean(i)))
            y_train_mean[ii,1] = 1 - y_train_mean[ii,0]
        else:
            raise ValueError('Empty label at index %i'%ii)
    print '\nx dimensions', np.shape(x_train)
    print 'y dimensions', np.shape(y_train_mean)
    #print 'Fraction of positive samples', y_positive_frac

    input_shape = (1, img_height, img_width)

    
    return x_train, y_train_mean, input_shape, spec_freq




# Wavelet feature extraction

def proc_data_bumpwav(signal_list, t, fs, img_height, img_width, wav_params = np.array([3., 0.1]), scales = bw.create_scale(200.,6000, 5/(2*np.pi), 8000, 20),
 label_interval = 0.1, binning = 'mean', nfft = 512, overlap = 256, save_weights = False):
               
    """Returns the training data x,y and the parameter input_shape required for initialisation of neural networks. 
    
    Takes as input arguments a signal list, and its associated time coordinates t. Would probably be better to integrate into
    one function, however remains separate as a t input was not required for the birdsong application (the origin of this 
    function). 
    
    img_height and width are used to determine the dimensions of the spectrogram chunk required as input to the NN. nfft and 
    overlap (optional) parameterise the spectrogram.
    
    label_interval: time interval between labels (in seconds)
    """

    cwt_spec_list = []
    t_list = []
    location_to_remove = []
    spec_fs = (1.*fs/(nfft - overlap))
    
    
    for i, t_i in enumerate(t):
        print 'Working on index', i
        t_initial = np.arange(0, len(t_i))
        
        
        wavelet = bw.bumpcwt(scales, wav_params, signal_list[i], fs)
        print 'Finished wavelet transform for index', i        

        spec_res = plt.specgram(signal_list[i], Fs = fs, NFFT = nfft, noverlap = overlap)
        b = wavelet.cwtcfs
        
        if binning == 'mean':
            print 'mean-binning in time domain'# to match dimensions of spectrogram
            R = int(np.ceil(float(np.shape(b)[1])/len(spec_res[2]))) # Downsampling rate R

            pad_size = int(np.ceil(float(b.shape[1])/R)*R - b.shape[1])

            b_padded = np.hstack([b, np.zeros([b.shape[0], pad_size])*np.NaN])
            print 'b padded shape',b_padded.shape
            rotato = b_padded.T.reshape(-1,R,b.shape[0])
            cwtcfs_reshape = nanmean(np.abs(rotato), axis=1)  # Maybe take the mean of the spectrum/abs vals here instead?
            cwtcfs_reshape = cwtcfs_reshape.T
            print 'cwt reshaped size', cwtcfs_reshape.shape
            print 'Reshape factor', R
   
            t_final = np.arange(0, len(t_i),1./(label_interval*spec_fs)) # Match fs_labels to fs_spec

        elif binning == 'interp':
            print 'interp-binning in time domain'
            
            xt = spec_res[2]
            xp = np.arange(b.shape[1],dtype = 'float32')/fs
            
            new_coefs = np.zeros([b.shape[0], len(spec_res[2])])
            print 'Interpolated shape', np.shape(new_coefs)
            for i, coefs in enumerate(b):    
                new_coefs[i,:] = np.interp(xt,xp, np.abs(coefs))
                cwtcfs_reshape = new_coefs
            t_final = np.arange(0, len(t_i),1./(label_interval*spec_fs)) # Match fs_labels to fs_spec

            
        else: 
            t_final = np.arange(0, len(t_i),1./(label_interval*fs))
            cwtcfs_reshape = b        
        t_corrected = np.interp(t_final, t_initial, t_i)        
        t_corrected = np.rint(t_corrected)
        location = np.where(t_corrected == 3)
        #print 'length of t_corrected', len(t_corrected)
        location_to_remove.append(location)
        t_list.append(t_corrected)
        cwt_spec_list.append(cwtcfs_reshape)
   
#      Removal of samples where there is no mutual agreement, labelled by a 3 in pre-processing
    
#     for index, item in enumerate(spec_list):
#         t_spec = np.delete(spec_list[index][2], location_to_remove[index])
#         f_spec = np.delete(spec_list[index][0], location_to_remove[index], axis = 1)
#         short_t = np.delete(t[index], location_to_remove[index])
        
#         print index, location_to_remove[index], 'Cut', np.shape(f_spec), 'Original', np.shape(spec_list[index][0])
        
#         spec_list[index][0] = f_spec
#         spec_list[index][2] = t_spec
# ###################################################################################################################    
    
#     # return spectrogram frequencies for visualisations. Not necessary for minimal working example    
#     spec_freq = SpecInfo[1][:img_height]
    
#    Dimensions for image fed into network
    nb_classes = 2
    x_train = []
    y_train = []


    for i in np.arange(len(cwt_spec_list)):
        n_max = np.floor(np.divide(np.shape(cwt_spec_list[i])[1],img_width)).astype('int')

        print 'Processing signal number', i
        print 'Length of cwt', np.shape(cwt_spec_list[i])[1]
        print 'Number of training inputs for this signal:', n_max    
        for n in np.arange(n_max):
            #print 'n', n, np.shape(spec_training_list[i][0])
            x_train.append(cwt_spec_list[i][:img_height,img_width*n:img_width*(n+1)])

            y_train.append(t_list[i][img_width*n:img_width*(n+1)])
    
    print '\nx dimensions before reshape', np.shape(x_train)

    y_train_mean = np.zeros([np.shape(y_train)[0], nb_classes])
    x_train = np.array(x_train).reshape((np.shape(x_train)[0],1,img_height,img_width))
        
    for ii, i in enumerate(y_train):
        if i.size:
            y_train_mean[ii,0] = int(np.round(np.mean(i)))
            y_train_mean[ii,1] = 1 - y_train_mean[ii,0]
        else:
            raise ValueError('Empty label at index %i'%ii)
    print '\nx dimensions', np.shape(x_train)
    print 'y dimensions', np.shape(y_train_mean)
    #print 'Fraction of positive samples', y_positive_frac

    input_shape = (1, img_height, img_width)

    return x_train, y_train_mean, input_shape#, spec_list, t_list
    
# Neural network training

def train_nn(model, x_train, y_train, x_test, y_test, conv, batch_size = 256, nb_epoch = 20, save_weights = False):
    # if data_method == 'wav':
    #     x_train_caged = x_train_wav
    #     y_train_caged = y_train_wav
    #     x_train = x_train_wav
    #     y_train = y_train_wav
    #     x_test = x_test_wav
    #     y_test= y_test_wav

    # elif data_method == 'spec':
    #     y_test = y_test_spec
    #     x_test = x_test_spec
    #     x_train = x_train_spec
    #     y_train = y_train_spec
    #     x_train_caged = x_train
    #     y_train_caged = y_train_spec
    # #input_shape = (1, x_train.shape[2], 10)

    # elif data_method == 'spec_cut':
    #     # Load wavelet for parameter matching
    #     data = np.load('Outputs/humbug_conv_wavelet_247_1_1_dz_interp.npz')
    #     print data.files
    #     # x_train_wav = data["x_train"]
    #     # y_train_wav = data["y_train"]
    #     # x_test_wav = data["x_test"]
    #     # y_test_wav = data["y_test"]
    #     wav_freq = data["wav_freq"]
    #     cut_freq = np.where(spec_freq > wav_freq[-1])[0]
    #     # Reshape for CNN
    #     if conv:
    #         x_train = x_train_spec[:,:,cut_freq,:]
    #         x_test = x_test_spec[:,:,cut_freq,:]
    #     # Reshape for MLP
    #     else:
    #         x_train = x_train_spec[:,cut_freq*spec_window]
    #         x_test = x_test_spec[:,cut_freq*spec_window]
    #         x_train_caged = x_train
    #     y_test = y_test_spec
    #     y_train = y_train_spec
    #     y_train_caged = y_train_spec
    #     print input_shape
    #     spec_freq = spec_freq[cut_freq]
    # elif data_method == 'raw':
    #     x_train_caged = x_train_raw
    #     y_train_caged = y_train_raw
    #     x_train = x_train_raw
    #     y_train = y_train_raw
    #     x_test = x_test_raw
    #     y_test= y_test_raw


    ##################### MLP execution Code ##########################
    if not conv:
        print 'Using MLP:'
        if x_train.ndim > 2:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[-2]*x_train.shape[-1])
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[-2]*x_test.shape[-1])
       
        # Optional code to save weights for further visualisation

        if save_weights == False:
            print 'Training data shape:', np.shape(x_train)
            early_stopping = EarlyStopping(monitor='acc', patience = 5, verbose=0, mode='auto')

            hist = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
                      verbose=2, callbacks = [early_stopping])
            return model, hist    

        #y_train_caged = y_train_wav
        if save_weights == True:    
            n_hidden = model.layers[0].get_config()['output_dim']
            print 'Training data shape:', np.shape(x_train)
            Weights = np.zeros([nb_epoch,np.shape(x_train)[1],n_hidden])
            for i in range(nb_epoch):
                print 'Epoch number', i+1, 'of', nb_epoch
                hist = model.fit(x_train, y_train, batch_size=batch_size, validation_split = 0., nb_epoch=1, 
                          verbose=2)
                W = model.layers[0].W.get_value(borrow=True)
                Weights[i,:,:] = W
        return model, Weights
                
    ##################### CNN execution code ##########################
    if conv:
        print 'Using CNN:'
        print 'Training data shape:', np.shape(x_train)

        early_stopping = EarlyStopping(monitor='acc', patience = 5, verbose=0, mode='auto')

        hist = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
                  verbose=2, callbacks = [early_stopping])
        return model, hist


# Visualise output, plot performance metrics from predictions and ground truth

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_output(x_test, y_test, model, median_filtering = False, kernel_size = 31, x_lim = None):
    

    score = model.evaluate(x_test, y_test, verbose=2)
    predictions = model.predict(x_test)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])





    ########### 2 class predictions #####################################################
    positive_predictions = predictions[:,0][np.where(y_test[:,0])]
    negative_predictions = predictions[:,1][np.where(y_test[:,1])]

    if median_filtering:
        filt_predictions = np.zeros(np.shape(predictions))
        predictions[:,0] = medfilt(predictions[:,0], kernel_size = kernel_size)
        predictions[:,1] = 1 - filt_predictions[:,0]
        positive_predictions = predictions[:,0][np.where(y_test[:,0])]
        negative_predictions = predictions[:,1][np.where(y_test[:,1])]



    true_positive_rate = (sum(np.round(positive_predictions)))/sum(y_test[:,0])
    true_negative_rate = sum(np.round(negative_predictions))/sum(y_test[:,1])



    figs = []

    f = plt.figure(figsize = (12,6))
    plt.plot(predictions[:,0],'g.', markersize = 12, label = 'y_pred')
    plt.plot(y_test[:,0], '--b', linewidth = 1, markersize = 2, label = 'y_test')
        
    plt.legend(loc = 9, ncol = 2)

    plt.ylim([-0.1,1.4])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if x_lim:
        plt.xlim([x_lim[0], x_lim[1]])
    plt.ylabel('Classifier output')
    plt.xlabel('Signal window number')
    # if save_fig == 'publication_softmax':
    #     plt.tight_layout()
    #     plt.savefig('../../../TexFiles/Papers/ECML/Images/softmax_' + model_name + '.pdf')
        
    # if save_fig_individual:
    #     plt.savefig('Outputs/' + 'solo_softmax_' + model_name + '.pdf')

    # figs.append(f)
    print 'True positive rate', true_positive_rate, 'True negative rate', true_negative_rate

    #plt.savefig('Outputs/' + 'ClassOutput_' + model_name + '.pdf', transparent = True)
    #print 'saved as', 'ClassOutput_' + model_name + '.pdf' 
    #plt.show()


    cnf_matrix = confusion_matrix(y_test[:,1], np.round(predictions[:,1]).astype(int))

    f, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

    y_true = y_test[:,0]
    y_score = predictions[:,0]
    roc_score = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)


    #plt.subplot(1,2,2)
    #plt.figure(figsize=(4,4))
    ax1.plot(fpr, tpr, '.-')
    ax1.plot([0,1],[0,1],'k--')
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC, area = %.4f'%roc_score)




    y_test_pr = y_test[:,0]

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(1):
        precision[i], recall[i], _ = precision_recall_curve(y_test_pr,
                                                            y_score)
        average_precision[i] = average_precision_score(y_test_pr, y_score)

    # Plot Precision-Recall curve
    #plt.clf()
    ax2.plot(recall[0], precision[0], color='b',
             label='Precision-Recall curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_title('Precision-Recall, area = {0:0.3f}'.format(average_precision[0]))
    #plt.legend(loc="lower left")
    # if save_fig == 'publication':
    #     plt.tight_layout()
    #     plt.savefig('/Outputs/Papers/ECML/Images/metric_' + model_name + '.pdf')
    # plt.show()
        
    F1 = f1_score(np.round(predictions[:,0]), y_test[:,0], average='binary')  
    print 'F1 score', F1    
    perf_metrics ={"tpr": true_positive_rate, "tnr": true_negative_rate, "f1": F1, "roc":roc_score, "pr":average_precision[0], "conf_matrix":cnf_matrix}
    
    return predictions, perf_metrics


## Analyse statistics of test samples

############### Chooose whether or not to save figure#################
def anal_test(x_train, y_train, predictions, x_test, feat_freq, feat, frac_data = 0.1, save_fig = False, conv = True, latex_labels = False):
    suffix = ''


    ### Representative of mozz or no_mozz from raw data labels

    mozz = np.where(y_train[:,0])
    no_mozz = np.where(y_train[:,1])

    mozz_samples = x_train[mozz].reshape(x_train[mozz].shape[0] * x_train[mozz].shape[-1], x_train[mozz].shape[2])
    no_mozz_samples = x_train[no_mozz].reshape(x_train[no_mozz].shape[0] * x_train[no_mozz].shape[-1], x_train[no_mozz].shape[2])

    binned_fft = np.mean(mozz_samples, axis = 0)
    binned_fft_negative = np.mean(no_mozz_samples, axis = 0)


    #### Statistics of samples that trigger a high response
    frac_data = 0.1
    sort_index = np.argsort(predictions[:,0])
    sort_index_low = np.argsort(predictions[:,1])


    print 'Number of test samples', len(x_test)
    n_response_samples = int(len(x_test) * frac_data)
    print 'Number of high response samples,', n_response_samples, 'percentage of test data', frac_data * 100, '%'

    high_response_samples = x_test[sort_index[(len(sort_index) - n_response_samples):]]
    low_response_samples = x_test[sort_index_low[(len(sort_index_low) - n_response_samples):]]

    print 'Number of high response samples', n_response_samples, 'data shape', np.shape(high_response_samples)
    #plot amplitude vs frequency for 1 dimensional signal bin?
    #plt.plot(spec_freq or wav_freq, )
    if conv:
        high_response_samples = high_response_samples.reshape([high_response_samples.shape[0]*high_response_samples.shape[-1],
                                                             high_response_samples.shape[-2]])
        low_response_samples = low_response_samples.reshape([low_response_samples.shape[0]*low_response_samples.shape[-1],
                                                             low_response_samples.shape[-2]])

    print 'reshaped', np.shape(high_response_samples)
    high_response_samples = np.fliplr(high_response_samples)
    low_response_samples = np.fliplr(low_response_samples)


    # # With python's cwt peak finder
    # function_points = np.mean(high_response_samples[:50], axis = 0) # pick top 50 detections
    # ind = find_peaks_cwt(function_points, np.arange(1,3),min_length=1.5) # filter out fewer labels

    # function_points_low = np.mean(low_response_samples[:50], axis = 0) # pick top 50 detections
    # ind_low = find_peaks_cwt(function_points_low, np.arange(1,3),min_length=1.5) # filter out fewer labels

    # # High sample box plot
    # d = np.zeros(len(wav_freq)).astype(str)
    # d[ind] = np.round(wav_freq[ind]).astype(int)
    # d[np.where(d == '0.0')] = ''
    # print 'y_ticklabels', d[::-1]

    # plt.title('Spectrum of strongest response')
    # plt.plot(wav_freq, high_response_samples[0])
    # plt.xlabel('Wavelet centre frequency')
    # plt.ylabel('Amplitude of wavelet coefficient')
    # plt.show()
    if feat == 'wav':
        box_labels = feat_freq.astype(int)[::-1]
        feat_freq = feat_freq
        x_label = 'Wavelet centre frequency (Hz)'
        print 'Wavelet feature representation'
    elif feat == 'spec': 
        box_labels = feat_freq.astype(int)
        feat_freq = feat_freq[::-1]
        x_label = 'Spectrogram frequency (Hz)'
        print 'Spectogram feature representation'
    else:
        print 'Feature representation not recognised.'

    # f = plt.figure(figsize = (5,5))
    # ax = plt.subplot()
    # ax.boxplot(high_response_samples, vert = False, labels=box_labels, manage_xticks=True, showbox=True, showfliers=False,  showmeans=True,  showcaps=True)
    # ax.set_yticklabels(d[::-1])#, fontsize = 12)
    # ax.set_title('Strongest positive responses')
    # plt.ylabel('Wavelet centre frequency')
    # plt.xlabel('Amplitude of wavelet coefficient')
    # plt.grid()
    # if save_fig:
    #     plt.savefig('../../../TexFiles/Papers/ECML/Images/WavPositive' + suffix + '.pdf')
    # plt.show()


    # ### Low sample box plot
    # d = np.zeros(len(wav_freq)).astype(str)
    # d[ind_low] = np.round(wav_freq[ind_low]).astype(int)
    # d[np.where(d == '0.0')] = ''
    # print 'y_ticklabels', d[::-1]

    # box_labels = wav_freq.astype(int)[::-1]



    # f = plt.figure(figsize = (5,5))
    # ax = plt.subplot()
    # ax.boxplot(low_response_samples, vert = False, labels=box_labels, manage_xticks=True, showbox=True, showfliers=False,  showmeans=True,  showcaps=True)
    # ax.set_yticklabels(d[::-1])#, fontsize = 12)
    # ax.set_title('Strongest negative responses')
    # plt.ylabel('Wavelet centre frequency')
    # plt.xlabel('Amplitude of wavelet coefficient')
    # plt.grid()
    # if save_fig:
    #     plt.savefig('../../../TexFiles/Papers/ECML/Images/WavNegative' + suffix + '.pdf')
    # plt.show()



    # low_response_samples = np.fliplr(low_response_samples)

    if latex_labels:
        label_size = 12
        plt.rcParams['xtick.labelsize'] = label_size 
        plt.rcParams['ytick.labelsize'] = label_size 
        #Optional: LaTeX cm font (slow)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        test_1 = '$\\mathbf{x}_{1,\\textrm{test}}(f)$'
        test_0 = '$\\mathbf{x}_{0,\\textrm{test}}(f)$'
        train_1 = '$\\mathbf{x}_{1,\\textrm{train}}(f)$'
        train_0 = '$\\mathbf{x}_{0,\\textrm{train}}(f)$'
    else:
        test_1 = 'x_1_test(f)'
        test_0 = 'x_0_test(f)'
        train_1 = 'x_1_train(f)'
        train_0 = 'x_0_test(f)'
    plt.figure(figsize = (7,4))
    box_means = np.mean(high_response_samples, axis = 0)
    box_means_low = np.mean(low_response_samples, axis = 0)
    plt.plot(feat_freq, (box_means-np.mean(box_means))/np.std(box_means),'g', label = test_1)
    plt.plot(feat_freq, (box_means_low-np.mean(box_means_low))/np.std(box_means_low),'r', 
             label = test_0)
    plt.plot(feat_freq,(binned_fft -np.mean(binned_fft))/np.std(binned_fft), 'blue', label = train_1)
    plt.plot(feat_freq,(binned_fft_negative -np.mean(binned_fft_negative))/np.std(binned_fft_negative),
             'k', label = train_0)

    # plt.plot(wav_freq, (box_means-0),'b', label = 'top score means' )
    # plt.plot(wav_freq, (box_means_low-0),'.r', label = 'bottom score means' )
    # plt.plot(wav_freq,(binned_fft -0), 'g', label = 'mosquito mean')
    # plt.plot(wav_freq,(binned_fft_negative -0), '.k', label = 'noise mean')

    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(x_label, fontsize = 12)
    plt.ylabel('Standardised coefficient amplitude', fontsize = 12)
    plt.tight_layout()
    if save_fig:
        print 'saved figure'
        plt.savefig('Outputs/figures/MeanMosquito' + suffix + '.pdf')
    plt.show()

    return None





## Cross-validation data processing

def plot_acc_mean(acc_tuple,nb_filters,dense_sizes,kernel_sizes):
    plt.figure()
    idx = 0
    for i in range(len(nb_filters)):
        for j in range(len(dense_sizes)):
            for k in range(len(kernel_sizes)):
                plt.errorbar(idx, np.mean(acc_tuple[0][i,j,k]), np.std(acc_tuple[0][i,j,k]),fmt='g.', ecolor='g')    
                plt.errorbar(idx, np.mean(acc_tuple[1][i,j,k]), np.std(acc_tuple[0][i,j,k]),fmt='r.', ecolor='r')  
                idx += 1
    plt.title('opt\_acc: ' + str(hyp_opt_acc) + ' opt\_acc\_val: ' + str(hyp_opt_val_acc))
    plt.legend(['train', 'val'], loc = 4)
    plt.xlabel('Hyperparameter combination')
    plt.ylabel('mean accuracies by 10-fold CV')


def retrieve_hyp_opt(acc_xval_full_list, nb_filters, dense_sizes, kernel_sizes):
    acc = np.array(acc_xval_full_list)
    mean_val = np.zeros([acc.shape[0], acc.shape[1], acc.shape[2]])
    for i in range(acc.shape[0]):    
        for j in range(acc.shape[1]):
            for k in range(acc.shape[2]):
                mean_val[i,j,k] = np.mean(acc[i,j,k])
                print mean_val[i,j,k]

    print 'index of highest metric:', np.argmax(mean_val)
    print 'value', np.max(mean_val)

    hyp_opt_idx = np.argmax(mean_val)


    #hyper_params = np.zeros([len(kernel_sizes),len(dense_sizes)])
    hyper_params = []
    
    print nb_filters
    print dense_sizes
    print kernel_sizes
    for index, i in enumerate(nb_filters):
        for index, j in enumerate(dense_sizes):
            for jndex, k in enumerate(kernel_sizes):
                #hyper_params[index,jndex] = [i,j]
                hyper_params.append([i,j,k])
#     print 'debug:', hyper_params
    print 'Optimum hyper_params by cross-validation:', hyper_params[hyp_opt_idx]
    return hyper_params[hyp_opt_idx]
#     return hyper_params, hyp_opt_idx

## Cross-validation functions

def crossval(x, y, folds, filter_numbers, dense_numbers, kernel_sizes, max_epoch, x_test = None, y_test = None, early_stop_metric = 'acc'):
#     Create dictionary with test scores that are labelled by the keys: kernel_size, nb_dense etc.
# Choose early stopping metric (default training accuracy)
    input_shape = (1, x.shape[2], x.shape[-1])
    nb_classes = 2
    batch_size = 256
    
    xval_dict = {"kernel_sizes":kernel_sizes, "n_dense":dense_numbers,"filter_numbers":filter_numbers}
    full_dict = {}
    acc_xval_full_list = []
    acc_xval_test_full_list = []
    val_acc_xval_full_list = []

    for idx, n_filt in enumerate(filter_numbers):
        print 'Processing number of conv filters:', n_filt, '...', idx, 'out of', len(filter_numbers) - 1
        acc_xval_list_filt = [] 
        acc_xval_test_list_filt = []
        val_acc_xval_list_filt = []
        
        for idx, n_dense in enumerate(dense_numbers):
            print 'Processing dense number:', n_dense, '...', idx, 'out of', len(dense_numbers) - 1
            acc_xval_list = [] 
            acc_xval_test_list = []
            val_acc_xval_list = []

            for idx, kernel_size in enumerate(kernel_sizes):
                print 'Processing kernel size:', kernel_size, '...', idx, 'out of', len(kernel_sizes) - 1
                acc_xval = []   # Reset accuracy list every 10 folds
                val_acc_xval = []
                acc_xval_test = []
                for n in range(folds):

                    start_idx = n*(len(x)/folds)
                    end_idx = (n+1)*(len(x)/folds)
                    print 'n_start', 'n_end', n*(len(x)/folds), (n+1)*(len(x)/folds)
                    x_val = x[start_idx:end_idx]
                    y_val = y[start_idx:end_idx]
                    x_tr = np.vstack([x[:start_idx], x[end_idx:]])
                    y_tr = np.vstack([y[:start_idx], y[end_idx:]])

                    model = Sequential()

                    model.add(Convolution2D(n_filt, kernel_size[0], kernel_size[1],
                                    border_mode = 'valid',
                                    input_shape = input_shape))
                    convout1 = Activation('relu')
                    model.add(convout1)
                    model.add(Flatten())
                    model.add(Activation('relu'))
                    model.add(Dense(n_dense))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.5))
                    model.add(Dense(nb_classes))
                    model.add(Activation('softmax'))

                    model.compile(loss='categorical_crossentropy',
                                  optimizer='adadelta',
                                  metrics=['accuracy'])    
                    print 'train shape', np.shape(x_tr) 

                    early_stopping = EarlyStopping(monitor=early_stop_metric, patience = 5, verbose=0, mode='auto')

                    hist = model.fit(x_tr, y_tr, batch_size=batch_size, nb_epoch=max_epoch,
                              verbose=0, callbacks = [early_stopping], validation_data = (x_val, y_val))

                    if x_test is not None:
                        score = model.evaluate(x_test, y_test, verbose=0)
                    else:
                        score = [0,0]
                    acc = hist.history["acc"][-1]
                    val_acc = hist.history["val_acc"][-1]

                    acc_xval.append(acc)
                    val_acc_xval.append(val_acc)
                    acc_xval_test.append(score[1])
                    #predictions = model.predict(x_test)
    #                 print('xval run %i score:'%n, score[0])
                    print('xval run', n, 'val accuracy:', val_acc, 'train accuracy:', acc)
                    if x_test is not None:
                        print('xval run', n, 'test accuracy:', score[1])  
                acc_xval_list.append(acc_xval)
                acc_xval_test_list.append(acc_xval_test)
                val_acc_xval_list.append(val_acc_xval)
            acc_xval_list_filt.append(acc_xval_list)
            acc_xval_test_list_filt.append(acc_xval_test_list)
            val_acc_xval_list_filt.append(val_acc_xval_list)
        acc_xval_full_list.append(acc_xval_list_filt)
        acc_xval_test_full_list.append(acc_xval_test_list_filt)
        val_acc_xval_full_list.append(val_acc_xval_list_filt)

    xval_dict["acc_test"] = acc_xval_test_full_list
    xval_dict["acc"] = acc_xval_full_list
    xval_dict["val_acc"] = val_acc_xval_full_list

    return xval_dict



def crossval_MLP(x, y, folds, filter_numbers, dense_numbers, kernel_sizes, max_epoch, x_test = None, y_test = None, early_stop_metric = 'acc'):
#     Create dictionary with test scores that are labelled by the keys: kernel_size, nb_dense etc.
# Choose early stopping metric (default training accuracy)
#     input_shape = (1, x.shape[2], x.shape[-1])
    nb_classes = 2
    batch_size = 256
    
    xval_dict = {"filter_numbers":filter_numbers, "n_dense":dense_numbers,"kernel_sizes":kernel_sizes}
    full_dict = {}
    acc_xval_full_list = []
    acc_xval_test_full_list = []
    val_acc_xval_full_list = []

    for idx, n_hidden in enumerate(filter_numbers):
        print 'Processing number of hidden units', n_hidden, '...', idx, 'out of', len(filter_numbers) - 1
        acc_xval_list_filt = [] 
        acc_xval_test_list_filt = []
        val_acc_xval_list_filt = []
        
        for idx, n_dense in enumerate(dense_numbers):
            print 'Processing dense number:', n_dense, '...', idx, 'out of', len(dense_numbers) - 1
            acc_xval_list = [] 
            acc_xval_test_list = []
            val_acc_xval_list = []

            for idx, kernel_size in enumerate(kernel_sizes):
                print 'Processing kernel size:', kernel_size, '...', idx, 'out of', len(kernel_sizes) - 1
                acc_xval = []   # Reset accuracy list every 10 folds
                val_acc_xval = []
                acc_xval_test = []
        # All commands that can be executed without the for loop



                for n in range(folds):

                    start_idx = n*(len(x)/folds)
                    end_idx = (n+1)*(len(x)/folds)
                    print 'n_start', 'n_end', n*(len(x)/folds), (n+1)*(len(x)/folds)
            # Add uneven sample number to last dataset OR dispose? Currently dispose to keep an equal number of samples in each chunk
            #         if ((not discard) and n == folds): 
            #             val_data = dataset[n*(len(dataset)/folds):((n+1)*(len(dataset)/folds) + len(dataset)%folds)]

                    x_val = x[start_idx:end_idx]
                    y_val = y[start_idx:end_idx]
                    x_tr = np.vstack([x[:start_idx], x[end_idx:]])
                    y_tr = np.vstack([y[:start_idx], y[end_idx:]])

                    model = Sequential()

        

                    model.add(Dense(n_hidden, input_dim=np.shape(x_tr)[1]))
                    model.add(Activation('relu'))
                    #model.add(Dense(128))
                    #model.add(Activation('relu'))
                    #model.add(Dense(128))
                    #model.add(Activation('relu'))
                    model.add(Dense(n_dense))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.5))
                    model.add(Dense(nb_classes))
                    model.add(Activation('softmax'))

                    model.compile(loss='categorical_crossentropy',
                                  optimizer='adadelta',
                                  metrics=['accuracy'])

                    print 'train shape', np.shape(x_tr) 

                    early_stopping = EarlyStopping(monitor=early_stop_metric, patience = 5, verbose=0, mode='auto')

                    hist = model.fit(x_tr, y_tr, batch_size=batch_size, nb_epoch=max_epoch,
                              verbose=0, callbacks = [early_stopping], validation_data = (x_val, y_val))

                    if x_test is not None:
                        score = model.evaluate(x_test, y_test, verbose=0)
                    else:
                        score = [0,0]
                    acc = hist.history["acc"][-1]
                    val_acc = hist.history["val_acc"][-1]

                    acc_xval.append(acc)
                    val_acc_xval.append(val_acc)
                    acc_xval_test.append(score[1])
                    #predictions = model.predict(x_test)
    #                 print('xval run %i score:'%n, score[0])
                    print('xval run', n, 'val accuracy:', val_acc, 'train accuracy:', acc)
                    if x_test is not None:
                        print('xval run', n, 'test accuracy:', score[1])                    

                acc_xval_list.append(acc_xval)
                acc_xval_test_list.append(acc_xval_test)
                val_acc_xval_list.append(val_acc_xval)
            acc_xval_list_filt.append(acc_xval_list)
            acc_xval_test_list_filt.append(acc_xval_test_list)
            val_acc_xval_list_filt.append(val_acc_xval_list)
        acc_xval_full_list.append(acc_xval_list_filt)
        acc_xval_test_full_list.append(acc_xval_test_list_filt)
        val_acc_xval_full_list.append(val_acc_xval_list_filt)
        
        
    xval_dict["acc_test"] = acc_xval_test_full_list
    xval_dict["acc"] = acc_xval_full_list
    xval_dict["val_acc"] = val_acc_xval_full_list

    return xval_dict



