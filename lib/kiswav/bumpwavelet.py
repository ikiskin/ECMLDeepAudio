import numpy
import matplotlib.pyplot

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

# Basic class implementation 

class bumpwav:
    """Create bump wavelet in frequency domain to be used in a continuous wavelet transform in the frequency domain.
    wav_params[0]: mu, in range [3, 6]
    wav_params[1]: sigma, in range [0.1, 1.2]
    freq: frequency range to match FFT of data"""
    def __init__(self, freq, scales, wav_params):
        mu = numpy.array(wav_params)[0]
        sigma = numpy.array(wav_params)[1]
        self.wav_params = wav_params
        self.scales = scales
        f = numpy.sort(freq)
        self.Psi = numpy.zeros([len(scales),len(f)])
        # Note, overflow occurs for sigma > ~1.00, however we are interested in the lowest possible values for accurate
        # frequency localisation. Fix later.
        matplotlib.pyplot.figure(figsize = (30,10))
        for s_i, s in enumerate(scales):
            PsiSupport = (2*numpy.pi*f <= numpy.float(mu + sigma)/s) & (2*numpy.pi*f >= numpy.float(mu - sigma)/s) 
            f_sup = f[numpy.where(PsiSupport)]
            self.Psi[s_i,numpy.where(PsiSupport)] = numpy.exp(1) * numpy.exp(-1./((1-(s*2*numpy.pi*f_sup - mu)**2)/(sigma**2)))
            matplotlib.pyplot.plot(f, self.Psi[s_i,:])
    

        matplotlib.pyplot.show()
        print 'Number of scales:', len(scales)
        matplotlib.pyplot.plot(scales,'.')
        matplotlib.pyplot.title('Scales of wavelet transform')
        matplotlib.pyplot.show()

##########################################

# Alternate (improved) implementation of bump wavelet based on MATLAB's example directly

class bumpcwt:
    """Create bump wavelet in frequency domain to be used in a continuous wavelet transform in the frequency domain.
    wav_params[0]: mu, in range [3, 6]
    wav_params[1]: sigma, in range [0.1, 1.2]"""
    def __init__(self, scales, wav_params, signal, fs):
        mu = wav_params[0]
        sigma = wav_params[1]
        self.wav_params = wav_params
        self.scales = scales
        meanSIG = numpy.mean(signal)
        x0 = signal - meanSIG
        nbSamp = len(x0)
        print 'Number of samples is', nbSamp        
        # Padding signal: wextend.m
        numpy2 = int(1+numpy.floor(numpy.log2(nbSamp) + 0.4999))             
        x1 = numpy.hstack([x0, numpy.zeros([2**numpy2 - nbSamp,])])
        n = len(x1)    
        print 'Length of padded signal is', n
        # Construct wavenumber array used in transform
        omega = numpy.arange(1,int(numpy.floor(n/2.)+1)) #  omega = (1:fix(n/2));
        omega_scaled = omega*((2.*numpy.pi)/(n*(1./fs)));
        self.omega = numpy.hstack([0., omega_scaled, -omega_scaled[int(numpy.floor((n-1)/2.))-1::-1]]) 
        #omega = [0., omega, -omega(fix((n-1)/2):-1:1)];
        print 'stacked omega length', len(self.omega)
        StpFrq = self.omega[1]
        NbFrq  = len(self.omega)
        print NbFrq, 'NbFrq'
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
            matplotlib.pyplot.plot(self.omega / (2.*numpy.pi), wft[jj,:])
 
        FourierFactor = mu/(2.*numpy.pi)
        self.frequencies = FourierFactor/scales;
        self.Psi = wft


        # Calculate Fourier transform of padded signal and scale coefficients via IFFT
        f = numpy.fft.fft(x1)
        cwtcfs = numpy.fft.ifft(self.Psi*f)
        self.cwtcfs = cwtcfs[:,0:nbSamp]
        self.omega = self.omega[0:nbSamp]
        self.Psi = self.Psi[:,0:nbSamp]
        
        #print 'ran bumpcwtft'
        #return self.cwtfs, self.omega
        #cwtcfs = ifft(repmat(f,NbSc,1).*psift,[],2);
        #cwtcfs = cwtcfs(:,1:nbSamp);
        #omega  = omega(1:nbSamp); 

##########################################

# Wavelet transform for wavlets created with bumpwav (deprecated method):

def cwtft(data, wavelet):
    
    # Wavelet parameter unpacking
    Psi = wavelet.Psi  # Included in this format to support scaling function Phi in future
    mu = wavelet.wav_params[0]
    sigma = wavelet.wav_params[1]
    levels = wavelet.scales
    
    # Data parameters
    t_data = (numpy.arange(len(data), dtype = 'float32'))/fs
    freq = numpy.fft.fftfreq(len(t_data),1./fs) 
    Y = numpy.fft.fft(data)
    # Sort Y in same order as freq, using just freq to perform the sort        
    freqY = zip(freq,Y)
    freqY.sort()

    Y = [coefficient for frequency, coefficient in freqY]
    freq = [frequency for frequency, coefficient in freqY]


    # Illustration for a single wavelet level.
    plt.figure(figsize=(11,3))
    plt.plot(t_data,data)
    plt.title('Signal(t)')
    plt.xlabel('Time (s)')
    plt.figure(figsize=(11,3))
    plt.plot(freq, numpy.abs(Y))
    plt.xlabel('Frequency (Hz)')
    plt.title('Signal(f)')



    plt.figure(figsize=(11,3))
    plot_legend = []
    plt.title('Wavelets (f)')
    plt.xlabel('Frequency (Hz)')
    cfs = numpy.zeros([len(levels), len(t_data)], dtype=complex)
    for i, level in enumerate(levels):  # Don't need for loop here, optimise
        cfs[i,:] = numpy.fft.ifft(Psi[i,:] * Y)  # Calculate coefficients
        plot_i = plt.plot(freq,Psi[i,:])  # Create wavelet helper plot

    S = numpy.abs(cfs)**2
    S = 100*S/numpy.sum(S[:])
    plt.figure(figsize=(15,3))
    plt.pcolormesh(t_data, mu/levels/(2*numpy.pi), numpy.abs(cfs))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Wavelet centre frequency, absolute value of coefficients')
    plt.colorbar()
    plt.xlim([t_data[0], t_data[-1]])
    plt.ylim([mu/levels[-1]/(2*numpy.pi), mu/levels[0]/(2*numpy.pi)])
    plt.show()

    max_scale_index = numpy.argmax(numpy.sum(S, axis = 1))
    max_scale = scales[max_scale_index]
    max_freq = mu/max_scale/(2*numpy.pi)
    print 'The maximum scale bin corresponds to a frequency of ' + str(max_freq) + ' Hz.'
    return cfs

##########################################

