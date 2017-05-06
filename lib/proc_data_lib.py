import sys 
import os, os.path
sys.path.append('lib/kiswav')
import bumpwavelet_minimal as bw


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
            print 'Processing', file
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
    return signal_list, label_list, fs


# Data label processing


# Select majority voting or other options
def proc_label(signal_list = signal_list, label_list = label_list, select_label = 5, maj_vote = False):

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




# Feature extraction function

def proc_data_humbug(signal_list, t = t, fs = fs, img_height = 256, img_width = 10, nfft = 512, overlap = 256, label_interval = 0.1):
               
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

        print 'Processing signal number', i
        print 'Length of spec', np.shape(spec_list[i][0])[1]
        print 'Number of training inputs for this signal:', n_max    
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


def proc_data_bumpwav(signal_list, t, fs, img_height, img_width, label_interval = 0.1, binning = 'mean', nfft = 512, overlap = 256):
               
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
    





# Data post-processing functions

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

def crossval(x, y, folds, filter_numbers, dense_numbers, kernel_sizes, max_epoch, early_stop_metric = 'acc', discard = True):
#     Create dictionary with test scores that are labelled by the keys: kernel_size, nb_dense etc.
# Choose early stopping metric (default training accuracy)
    input_shape = (1, x.shape[2], x.shape[-1])
    nb_classes = 2
    batch_size = 256
    
    xval_dict = {"kernel_sizes":kernel_sizes, "n_dense":n_dense,"filter_numbers":filter_numbers}
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

                    score = model.evaluate(x_test, y_test, verbose=0)
                    acc = hist.history["acc"][-1]
                    val_acc = hist.history["val_acc"][-1]

                    acc_xval.append(acc)
                    val_acc_xval.append(val_acc)
                    acc_xval_test.append(score[1])
                    #predictions = model.predict(x_test)
    #                 print('xval run %i score:'%n, score[0])
                    print('xval run', n, 'val accuracy:', val_acc, 'train accuracy:', acc)
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



def crossval_MLP(x, y, folds, filter_numbers, dense_numbers, kernel_sizes, max_epoch, early_stop_metric = 'acc', discard = True):
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

                    score = model.evaluate(x_test, y_test, verbose=0)
                    acc = hist.history["acc"][-1]
                    val_acc = hist.history["val_acc"][-1]

                    acc_xval.append(acc)
                    val_acc_xval.append(val_acc)
                    acc_xval_test.append(score[1])
                    #predictions = model.predict(x_test)
    #                 print('xval run %i score:'%n, score[0])
                    print('xval run', n, 'val accuracy:', val_acc, 'train accuracy:', acc)
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



