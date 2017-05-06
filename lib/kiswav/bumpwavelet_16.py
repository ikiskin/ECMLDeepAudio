import numpy
import matplotlib.pyplot
import scipy.integrate
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
        print cfsNORM
        NbSc = len(scales)
        wft = numpy.zeros([NbSc,NbFrq])
        experimental_constant = []
        wavelet_area = []  
        wavelet_area_analytic = []

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
            
            bwavfunc_test = lambda w: numpy.abs(numpy.exp(1)*numpy.exp(-1./(1-((w-mu)/sigma)**2)))/w
            Cpsi_test = scipy.integrate.quad(bwavfunc_test,mu-sigma,mu+sigma)
            a0_test = scales[1]/scales[0]
            alpha_test = 2*numpy.log(a0_test)*(1./Cpsi_test[0])
                        
            wavelet_area.append(numpy.trapz(Psi))
            
            experimental_constant.append(alpha_test * s)
            #wft[jj,:] = Psi/(numpy.trapz(Psi))
            wft[jj,:] = Psi * s
            wavelet_area_analytic.append(numpy.trapz(wft[jj,:]))
            #matplotlib.pyplot.plot(self.omega / (2.*numpy.pi), wft[jj,:])
            
 
        FourierFactor = mu/(2.*numpy.pi)
        self.frequencies = FourierFactor/scales;
        self.Psi = wft
        self.wavelet_area = wavelet_area
        self.experimental_constant = experimental_constant
        self.wavelet_area_analytic = wavelet_area_analytic
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

# Wavelet transform for wavlets created with bumpwav (deprecated method):

def cwtft(data, fs, scales, wavelet):
    
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
    matplotlib.pyplot.figure(figsize=(11,3))
    matplotlib.pyplot.plot(t_data,data)
    matplotlib.pyplot.title('Signal(t)')
    matplotlib.pyplot.xlabel('Time (s)')
    matplotlib.pyplot.figure(figsize=(11,3))
    matplotlib.pyplot.plot(freq, numpy.abs(Y))
    matplotlib.pyplot.xlabel('Frequency (Hz)')
    matplotlib.pyplot.title('Signal(f)')



    matplotlib.pyplot.figure(figsize=(11,3))
    plot_legend = []
    matplotlib.pyplot.title('Wavelets (f)')
    matplotlib.pyplot.xlabel('Frequency (Hz)')
    cfs = numpy.zeros([len(levels), len(t_data)], dtype=complex)
    for i, level in enumerate(levels):  # Don't need for loop here, optimise
        cfs[i,:] = numpy.fft.ifft(Psi[i,:] * Y)  # Calculate coefficients
        plot_i = matplotlib.pyplot.plot(freq,Psi[i,:])  # Create wavelet helper plot

    S = numpy.abs(cfs)**2
    S = 100*S/numpy.sum(S[:])
    matplotlib.pyplot.figure(figsize=(15,3))
    matplotlib.pyplot.pcolormesh(t_data, mu/levels/(2*numpy.pi), numpy.abs(cfs))
    matplotlib.pyplot.xlabel('Time (s)')
    matplotlib.pyplot.ylabel('Frequency (Hz)')
    matplotlib.pyplot.title('Wavelet centre frequency, absolute value of coefficients')
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.xlim([t_data[0], t_data[-1]])
    matplotlib.pyplot.ylim([mu/levels[-1]/(2*numpy.pi), mu/levels[0]/(2*numpy.pi)])
    matplotlib.pyplot.show()

    max_scale_index = numpy.argmax(numpy.sum(S, axis = 1))
    max_scale = scales[max_scale_index]
    max_freq = mu/max_scale/(2*numpy.pi)
    print 'The maximum scale bin corresponds to a frequency of ' + str(max_freq) + ' Hz.'
    return cfs

##########################################

