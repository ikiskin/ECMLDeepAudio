import numpy
import matplotlib.pyplot
import scipy.integrate
import sklearn.linear_model
import peakutils

##########################################

# Helper function used to create scales, based on MATLAB's CWT time frequency helper functions

def create_scale(minfreq, maxfreq, f0, fs, NumVoices):
    """f0 is the wavelet centre frequency, fs sample frequency, NumVoices number of voices per octave"""
    a0 = 2**(1./NumVoices)
    minscale = 1.*f0/(maxfreq/(1.*fs))
    maxscale = 1.*f0/(minfreq/(1.*fs))
    minscale = numpy.floor(NumVoices*numpy.log2(minscale))
    maxscale = numpy.ceil(NumVoices*numpy.log2(maxscale))
    scales = a0**numpy.arange(minscale,maxscale+1)/(1.*fs)
    return scales

##########################################


# Helper functions used for detection

def meanvar(D):
    m = scipy.mean(D, axis = 0)
    s = scipy.shape(D)
    if len(s) == 1:
        c = D.var()
        if c == 0: c = 1.0
    elif (s[0] == 1) + (s[1] == 1):
        c = D.var()
        if c == 0: c = 1.0
    else:
        c = scipy.diag(scipy.cov(D, rowvar = False)).copy()
        c[c == 0] = 1
    return m, c

def normalis(X, D):
    m, c = meanvar(D)
    return (X - m) / scipy.sqrt(c)

def analyse_scale(wav_signal):
   energy = numpy.sum(numpy.abs(wav_signal.cwtcfs)**2, axis = 1) 
   percentage = 100.*energy/sum(energy)

   X_train = wav_signal.frequencies.reshape(-1,1) # since y = wT*X
   X_test = X_train
   y_train = percentage


   ## Store predicted signal, FFT, weights for each window
   model = sklearn.linear_model.LinearRegression(fit_intercept = False)
    
   y_pred = (model.fit(X_train, y_train).predict(X_test))
  
   percentage_detrend = percentage - wav_signal.frequencies * model.coef_

   maxpercent = numpy.max(percentage)
   maxscale_index = numpy.argmax(percentage)
   maxscale = wav_signal.scales[int(maxscale_index)]

   return percentage, percentage_detrend   

##########################################







# Alternate (improved) implementation of bump wavelet based on MATLAB's example directly

class bumpcwt:
    """Create bump wavelet in frequency domain to be used in a continuous wavelet transform in the frequency domain.
    wav_params[0]: mu, in range [3, 6]
    wav_params[1]: sigma, in range [0.1, 1.2]
    scales input must be logarithmically spaced for correct reconstruction, but can have gaps"""
    def __init__(self, scales, wav_params, signal, fs):
        mu = wav_params[0]
        sigma = wav_params[1]
        self.wav_params = wav_params
        self.scales = scales
        meanSIG = numpy.mean(signal)
        x0 = signal - meanSIG
        nbSamp = len(x0)
        #print 'Number of samples is:', nbSamp        
        # Padding signal: wextend.m
        np2 = int(1+numpy.floor(numpy.log2(nbSamp) + 0.4999))             
        x1 = numpy.hstack([x0, numpy.zeros([2**np2 - nbSamp,])])
        n = len(x1)    
        #print 'Length of padded signal is:', n
        # Construct wavenumber array used in transform
        omega = numpy.arange(1,int(numpy.floor(n/2.)+1)) #  omega = (1:fix(n/2));
        omega_scaled = omega*((2.*numpy.pi)/(n*(1./fs)));
        self.omega = numpy.hstack([0., omega_scaled, -omega_scaled[int(numpy.floor((n-1)/2.))-1::-1]]) 
        #omega = [0., omega, -omega(fix((n-1)/2):-1:1)];
        #print 'stacked omega length', len(self.omega)
        StpFrq = self.omega[1]
        NbFrq  = len(self.omega)
        #print NbFrq, 'NbFrq'
        SqrtNbFrq = numpy.sqrt(NbFrq)
        cfsNORM = numpy.sqrt(StpFrq)*SqrtNbFrq
        NbSc = len(scales)
        wft = numpy.zeros([NbSc,NbFrq])
        
        for jj, s in enumerate(scales):
            w = numpy.divide(s*self.omega-mu,sigma)
        
            #expnt = -1./(1-w**2)
            # Re-write line to avoid overflow and logical index use
            #daughter = numpy.exp(1)*numpy.exp(expnt)*(numpy.abs(w)<1-numpy.spacing(1))  
            #daughter(isnan(daughter)) = 0
            
            wSupportNot =  numpy.abs(w) > 1 # > (1 + numpy.spacing(1))
            w[numpy.where(wSupportNot)] = 0 # Avoid overflow in exp
            #print 'w', w, 'wSupportNot', wSupportNot
            #matplotlib.pyplot.plot(w)
            expnt = -1./(1-w**2)
            Psi = numpy.exp(1)*numpy.exp(expnt)
            Psi[numpy.where(wSupportNot)] = 0
            wft[jj,:] = Psi

 
        FourierFactor = mu/(2.*numpy.pi)
        self.frequencies = FourierFactor/scales;
        self.Psi = wft


        # Calculate Fourier transform of padded signal and scale coefficients via IFFT
        f = numpy.fft.fft(x1)
        cwtcfs = numpy.fft.ifft(self.Psi*f)
        self.cwtcfs = cwtcfs[:,0:nbSamp]
        self.omega = self.omega[0:nbSamp]
        self.Psi = self.Psi[:,0:nbSamp]
        

        # Return constant required for inverse CWT. N.B. LOGARITHMICALLY spaced initial scales required
        bwavfunc = lambda w: numpy.abs(numpy.exp(1)*numpy.exp(-1./(1-((w-mu)/sigma)**2)))/w
        Cpsi = scipy.integrate.quad(bwavfunc,mu-sigma,mu+sigma)
        a0 = scales[1]/scales[0]
        self.alpha = 2*numpy.log(a0)*(1./Cpsi[0])

        # Peform iCWT
        X = self.alpha*numpy.real(self.cwtcfs)
        self.reconstruction = numpy.sum(X, axis = 0) + meanSIG  # Mean re-added
        #print 'ran bumpcwtft'
        #return self.cwtfs, self.omega
        #cwtcfs = ifft(repmat(f,NbSc,1).*psift,[],2);
        #cwtcfs = cwtcfs(:,1:nbSamp);
        #omega  = omega(1:nbSamp); 

##########################################


# Calculation of wavelet weights per window

def CWT(data_list, fs, CWT_scales, CWT_mu = 6., CWT_sigma = 0.1, window_size = 1024):
    """Calculate wavelet weights to be used in detection of mosquitoes. Calculation performed on 
       windows of data. 
       
       Inputs:
       
       window_size: Number of samples to analyse per window. Use window to divide signal into chunks without remainder. 
       data_list: Supply a list of numpy arrays, each array is a full signal
       fs: Sampling frequency of data in data_list. 
       CWT_scales: Scales to be used in wavelet transform
       CWT_mu: [3, 6], CWT_sigma: [0.1, 1.2]: Parameters for wavelet. Defaults to highest frequency resolution

       
       Outputs:
       

       w_signal_aggregate: reconstruction weights (per scale). List of numpy.arrays, dimensions:
                           (NumSignals x (NumScales))"""
                              

    w_percentage_signal_aggregate = []

    for index, data in enumerate(data_list): 
        w_aggregate = []
        w_percentage_aggregate = []
        y_aggregate = []
        y_aggregate_FFT = []
        print 'Processing file #', index, 'containing', len(data), 'samples at sampling frequency', fs, 'Hz.'
        n_end = int(numpy.floor(len(data)/window_size))
        print 'Number of windows:', n_end

        t_data = (numpy.arange(len(data), dtype = 'float32'))/fs
        for n in range(0,n_end):
                data_windowed = data[n*window_size:(n+1)*window_size]
                t_windowed = t_data[n*window_size:(n+1)*window_size]
                freq = numpy.fft.fftfreq(len(data_windowed),1./fs) 
                #matplotlib.pyplot.figure(1)
                #matplotlib.pyplot.plot(t_windowed, data_windowed)
                
                ## Create wavelet object
                wavelet = bumpcwt(CWT_scales, [CWT_mu, CWT_sigma], data_windowed, fs)
                
                ## MATLAB energy method
                
                percentage, percentage_detrend = analyse_scale(wavelet)
               
                w_percentage_aggregate.append(percentage_detrend)
                #matplotlib.pyplot.figure(2)
                #matplotlib.pyplot.plot(wavelet.frequencies, lassomodel.coef_, '.-')
        w_percentage_signal_aggregate.append(w_percentage_aggregate)
        
    return  w_percentage_signal_aggregate, wavelet

##########################################



# Detection of mosquitoes based on wavelet output

# Helper functions

def peakcheck(locs, pks, sigma, target_freq, fraction = 0.15, div_tolerance = 0.2):
    locspks = zip(locs,pks)
    locspks.sort()
    #print numpy.shape(locspks)

    pks = numpy.array([peak for location, peak in locspks])
    locs = numpy.array([location for location, peak in locspks])

    
    # Sort in ascending order. 
    
    #print 'sorted locs', locs
    
    lower_freq = (locs > target_freq * (1-fraction))
    upper_freq = (locs < target_freq * (1+fraction))
    candidate_locs = locs[scipy.logical_and(lower_freq,upper_freq)]
    candidate_pks = pks[scipy.logical_and(lower_freq,upper_freq)]
    
    
    output, output_pks = divcheck(candidate_locs, locs, candidate_pks, div_tolerance)
    
    
    #return division, candidate_locs, candidate_pks
    return output, output, output_pks 

def divcheck(potential_locs, locs, candidate_pks, div_tolerance):
    div = []
    output = []
    output_pks = []
    for l in potential_locs:  # Divide every detected peak by fundamental harmonic candidate peak
        remainder = list(numpy.divide(locs,l) - numpy.round(numpy.divide(locs,l)))
#        print 'remainder', remainder
        for ii, i in enumerate(remainder):
            if (numpy.abs(i) < div_tolerance) and (numpy.abs(i) > 0.0000001):
#                print 'found less than 0.1'
                if len(remainder) > 1:
                    div.append(numpy.divide(locs[ii],l))
                    output.append(locs)
                    output_pks.append(candidate_pks)
#                   print 'Added peaks', 'length of remainder', len(remainder)
            else:
            	ii # Dummy line
#                print 'skip loop'
                    

      
    return output, output_pks  # dimensions: (number of candidate peaks, all peaks)

##########################################


# Detector core algorithm

def detect_mosquito(signal_list, fs, win_size, wavelet, w_percentage, SNR_index, upper_bound, div_tolerance = 0.2, fraction = 0.05, target_freq = 350):
    
    nwin = len(signal_list[0])/win_size
    
    mosquito_lik = numpy.zeros(nwin)

    for win_num in range(nwin):
#        print 'Start time', win_num * 400. / fs
        condition = w_percentage[SNR_index][win_num] > upper_bound
        loc_peaks = wavelet.frequencies[condition]
        sig_peaks = w_percentage[SNR_index][win_num][condition]


        if not len(sig_peaks):
            #print "No significant peaks found"
            #matplotlib.pyplot.show()
            target_freq  # Dummy command
        else:
            y_thresh = w_percentage[SNR_index][win_num]
            thresh_norm = (upper_bound - numpy.min(y_thresh)) / (numpy.max(y_thresh) - numpy.min(y_thresh))
            indexes = peakutils.indexes(y_thresh, thres = thresh_norm)
            #matplotlib.pyplot.plot(wavelet.frequencies[indexes], y_thresh[indexes], '.r', markersize = 10)
            #print 'freq', wavelet.frequencies[indexes], 'val', y_thresh[indexes]
            #matplotlib.pyplot.show()


            div, a, b = peakcheck(wavelet.frequencies[indexes], y_thresh[indexes], 0.3, target_freq, fraction, div_tolerance)

            #matplotlib.pyplot.plot(wavelet.frequencies[indexes], y_thresh[indexes], 'x', a, b, 'o')
            #matplotlib.pyplot.title('Candidate peaks')
            #matplotlib.pyplot.show()

            if len(a):

                #print a, b   # candidate locs, candidate peaks
                #print 'div',div
                #c = div[0]
                #print 'b',b
                mosquito_lik[win_num] =  b[0][0]/upper_bound
            else:
                mosquito_lik[win_num] = 0
#                print 'No candidate accepted'

#        print '==============================================================================================================='

    xcoords = win_size / numpy.float(fs) * (numpy.arange(nwin))
    matplotlib.pyplot.title('Mosquito likelihood at target frequency %.0f Hz'%target_freq)        
    matplotlib.pyplot.plot((xcoords + win_size * (numpy.arange(nwin) + 1) / numpy.float(fs))/2 , mosquito_lik, '.', markersize = 10)
    matplotlib.pyplot.xlabel('Time (s)')
    matplotlib.pyplot.ylabel('Mosquito likelihood')

    for xc in xcoords:
        matplotlib.pyplot.axvline(x=xc, color = 'k', ls = 'dashed')
    matplotlib.pyplot.show()        
    
    return mosquito_lik

##########################################


