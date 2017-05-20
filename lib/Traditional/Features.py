from scipy.fftpack import fft
import numpy as np
from python_speech_features import mfcc, logfbank
from python_speech_features.sigproc import framesig
from scipy.signal import resample

import logging

logger = logging.getLogger(__name__)

eps = 0.00000001

def FeatureLabelGen(sig_lst,lbl_lst,fs,winlen,winstep,feat_lst,proc_lst,dat={}):
    # FEATURES
    feats = []
    sig_all = np.concatenate(sig_lst)
    #chnkd_sig = framesig(sig_all,winlen*fs,winstep*fs)
    for feat_id in feat_lst:
        # need to put winlen and winstep in
        print("Generating: %s"%feat_id)
        feat=globals()["%s_extractor"%feat_id].extract(sig_all,fs,winlen,winstep)
        if len(feat.shape)==1:
            feat=np.reshape(feat,[feat.shape[0],1])
        feats.append(feat)
    feats=np.concatenate(feats,axis=1)
    if "del_mask" in proc_lst:
        print("Removing some features...")
        mask = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, False, False, True]
        feats = np.array([feat for feat,rm in zip(feats.T,mask) if not rm]).T
    if "normalise" in proc_lst:
        print("Normalising...")
        stds = dat["scalefactors"] if "scalefactors" in dat else np.std(feats, axis=0)
        feats = feats / stds
        dat["scalefactors"] = stds

        logger.debug("Shifting to zero mean.")
        means = dat["shifts"] if "shifts" in dat else np.mean(feats, axis=0)
        feats = feats - means
        dat["shifts"] = means
    # LABELS
    lbls_all = np.concatenate(lbl_lst)
    lbls_all = resample(lbls_all,len(sig_all)) # resample labels from orig to same number of samples as signals
    lbls_all = np.around(lbls_all)
    lbls_chnkd = framesig(lbls_all, winlen * fs, winstep * fs)
    lbls =  [float(mode(chunk)[0]) for chunk in lbls_chnkd]
    return feats,lbls,dat


# mfcc, mel, zcr, energy, energyentropy, specspreadcentspecentropy,specflux,specrolloff,,spec
class FeatXtractr(object):
    def __init__(self, name, gen_feats, default_args):
        """Initialise feature extractor with name, a function to generate features, and default arguments for
        the function.
        Args:
            name (str):             ame of feature
            gen_feats (func(chnkd_sig_lst,fs_lst,args)):  function to generate features
                        sig_lst (list): list of audio signals
                        fs_lst  (list): list of sampling rates
                        args    (dict): dictionary of arguments to override default, can be none
            default_args (dict):    default arguments for generating features
        """
        self.name = name
        self.gen_feats = gen_feats
        self.default_args = default_args  # must NOT be nested

    def extract(self, sig, fs,winlen,winstep):
        """Return features from input data using gen_feats function."""
        feats = self.gen_feats(sig, fs, self.default_args,winlen,winstep)
        return feats

##########################
######### MFCC ###########
##########################
def mfcc_xtr_func(sig, fs, args, winlen, winstep):
    mfcc_feat = np.clip(np.nan_to_num( mfcc(sig, samplerate=fs, winlen=winlen,winstep=winstep, numcep=args["nmfccs"])),a_min=args["a_min"], a_max=args["a_max"])
    return mfcc_feat
mfcc_args = {
    "nmfccs": 13,
    "a_min": -100,
    "a_max": 100,
    "winlen": 0.025,
    "winstep": 0.01
}
mfcc_extractor = FeatXtractr("mfcc", mfcc_xtr_func, mfcc_args)

##########################
########## MEL ###########
##########################
def mel_xtr_func(sig, fs, args, winlen, winstep):
    mfb = logfbank(sig, fs, lowfreq=args["low_frq"], nfilt=args["n_flt"], winlen=winlen, winstep=winstep)
    mel_feat = np.clip(np.nan_to_num(mfb), a_min=args["a_min"], a_max=args["a_max"])
    slices = slice_spectrum(mel_feat, args["delta"])
    return slices
mel_args = {
    "low_frq": 500,
    "n_flt": 26,
    "delta": 0,
    "a_min": -100,
    "a_max": 100,
}
mel_extractor = FeatXtractr("mel", mel_xtr_func, mel_args)

##########################
### ZERO-CROSSING RATE ###
##########################
def zcr_xtr_func(sig, fs, args, winlen, winstep):
    chnkd_sig = framesig(sig, winlen * fs, winstep * fs)
    
    """Sign-change rate of signal per frame."""

    #chnkd_sig = raw_chnkd_xtr_func(chnkd_sig_lst, fs, args)
    zcr_wins = []
    for chnk in chnkd_sig:
        zcr_win = np.sum(chnk[:-1] * chnk[1:] < 0)
        zcr_wins.append(zcr_win)
    return np.array(zcr_wins)

zcr_args = {}
zcr_extractor = FeatXtractr("zcr", zcr_xtr_func, zcr_args)

##########################
######### ENERGY #########
##########################
def energy_xtr_func(sig, fs, args, winlen, winstep):
    chnkd_sig = framesig(sig, winlen * fs, winstep * fs)
    
    """Sum of squares of signal values, normalised by window length."""

    #chnkd_sigs = raw_chnkd_xtr_func(chnkd_sig_lst, fs_lst, args)
    #for chnkd_sig, fs in zip(chnkd_sigs, fs_lst):
    energy_wins = []
    for chnk in chnkd_sig:
        nrm_energy_win = 1. / len(chnk) * np.sum(chnk ** 2)
        energy_wins.append(nrm_energy_win)
    return np.array(energy_wins)
energy_args = {}
energy_extractor = FeatXtractr("energy", energy_xtr_func, energy_args)

###########################
#### ENTROPY OF ENERGY ####
###########################
def energyentropy_xtr_func(sig, fs, args, winlen, winstep):
    chnkd_sig = framesig(sig, winlen * fs, winstep * fs)
    
    """Entropy of energy -  based on pyAudioAnalysis.audioFeatureExtraction library
    [github.com/tyiannak/pyAudioAnalysis]"""
    #chnkd_sigs = raw_chnkd_xtr_func(chnkd_sig_lst, fs_lst, args)
    entropies = []
    for chnk in chnkd_sig:
        tot_enrgy = np.sum(chnk ** 2)
        subwin_len = int(np.floor(len(chnk) / args["n_subwins"]))
        if len(chnk) != subwin_len * args["n_subwins"]:
            chnk = chnk[0:subwin_len * args["n_subwins"]]
        subwins = chnk.reshape(subwin_len, args["n_subwins"], order='F').copy()
        subwin_enrgy = np.sum(subwins ** 2, axis=0) / float(tot_enrgy + eps)
        entropy = -np.sum(subwin_enrgy * np.log2(subwin_enrgy + eps))
        entropies.append(entropy)
    return np.array(entropies)

energyentropy_args = {
    "n_subwins": 10
}
energyentropy_extractor = FeatXtractr("energyentropy", energyentropy_xtr_func, energyentropy_args)

##############################
# SPECTRAL SPREAD & CENTROID #
##############################
def specspreadcent_xtr_func(sig, fs, args, winlen, winstep):
    chnkd_sig = framesig(sig, winlen * fs, winstep * fs)
    
    """Spectral spread and centroid of windows -  based on pyAudioAnalysis.audioFeatureExtraction library
    [github.com/tyiannak/pyAudioAnalysis]"""
    #chnkd_sigs = raw_chnkd_xtr_func(chnkd_sig_lst, fs_lst, args)
    centroids, spreads = [], []
    for chnk in chnkd_sig:
        spec = get_win_fft(chnk, winlen, fs)
        ind = (np.arange(1, len(spec) + 1)) * (fs / (2.0 * len(spec)))
        Xt = spec.copy()
        Xt = Xt / Xt.max()
        NUM = np.sum(ind * Xt)
        DEN = np.sum(Xt) + eps

        # Centroid:
        C = (NUM / DEN)

        # Spread:
        S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

        # Normalize:
        C = C / (fs / 2.0)
        S = S / (fs / 2.0)

        centroids.append(C)
        spreads.append(S)
    res = [[cent, spread] for cent, spread in zip(centroids, spreads)]
    return np.array(res)

specspreadcent_args = {}
specspreadcent_extractor = FeatXtractr("specspreadcent", specspreadcent_xtr_func, specspreadcent_args)

##############################
###### SPECTRAL ENTROPY ######
##############################
def specentropy_xtr_func(sig, fs, args, winlen, winstep):
    chnkd_sig = framesig(sig, winlen * fs, winstep * fs)
    
    """Spetral entropy -  based on pyAudioAnalysis.audioFeatureExtraction library
    [github.com/tyiannak/pyAudioAnalysis]"""
    #chnkd_sigs = raw_chnkd_xtr_func(chnkd_sig_lst, fs_lst, args)
    entropies = []
    for chnk in chnkd_sig:
        spec = get_win_fft(chnk, winlen, fs)
        tot_enrgy = np.sum(spec ** 2)  # total spectral energy
        subwin_len = int(np.floor(len(spec) / args["n_subwins"]))  # length of sub-frame
        if len(spec) != subwin_len * args["n_subwins"]:
            spec = spec[0:subwin_len * args["n_subwins"]]
        subwins = spec.reshape(subwin_len, args["n_subwins"],
                               order='F').copy()  # define sub-frames (using matrix reshape)
        subwin_enrgy = np.sum(subwins ** 2, axis=0) / float(tot_enrgy + eps)  # compute spectral sub-energies
        entropy = -np.sum(subwin_enrgy * np.log2(subwin_enrgy + eps))  # compute spectral entropy
        entropies.append(entropy)
    return np.array(entropies)

specentropy_args = {
    "n_subwins": 10,
}
specentropy_extractor = FeatXtractr("specentropy", specentropy_xtr_func, specentropy_args)

###########################
###### SPECTRAL FLUX ######
###########################
def specflux_xtr_func(sig, fs, args, winlen, winstep):
    chnkd_sig = framesig(sig, winlen * fs, winstep * fs)
    
    """Spectral flux as sum of square differences -  based on pyAudioAnalysis.audioFeatureExtraction library
    [github.com/tyiannak/pyAudioAnalysis]"""
    #chnkd_sigs = raw_chnkd_xtr_func(chnkd_sig_lst, fs_lst, args)
    fluxs, prev_chnk, prev_chnk_sum = [], [], []
    for chnk in chnkd_sig:
        spec = get_win_fft(chnk, winlen, fs)
        specsum = np.sum(spec + eps)
        if prev_chnk != []:
            flux = np.sum((spec / specsum - prev_chnk / prev_chnk_sum) ** 2)
            fluxs.append(flux)
        else:
            fluxs.append(0.)
        prev_chnk = spec
        prev_chnk_sum = specsum
    return np.array(fluxs)

specflux_args = {}
specflux_extractor = FeatXtractr("specflux", specflux_xtr_func, specflux_args)

############################
##### SPECTRAL ROLL-OFF #####
############################
def specrolloff_xtr_func(sig, fs, args, winlen, winstep):
    chnkd_sig = framesig(sig, winlen * fs, winstep * fs)
    
    """Find spectral rollof where spectral power is args["ratio"]*100 % of total energy -
    based on pyAudioAnalysis.audioFeatureExtraction library [github.com/tyiannak/pyAudioAnalysis]"""
    #chnkd_sigs = raw_chnkd_xtr_func(chnkd_sig_lst, fs_lst, args)
    rolloffs = []
    for chnk in chnkd_sig:
        spec = get_win_fft(chnk, winlen, fs)
        tot_enrgy = np.sum(spec ** 2)
        thresh = args["ratio"] * tot_enrgy
        cumsum = np.cumsum(spec ** 2) + eps
        [a, ] = np.nonzero(cumsum > thresh)
        if len(a) > 0:
            rolloff = np.float64(a[0]) / (float(len(spec)))
        else:
            rolloff = 0.0
        rolloffs.append(rolloff)
    return np.array(rolloffs)

specrolloff_args = {
    "ratio": 0.9
}
specrolloff_extractor = FeatXtractr("specrolloff", specrolloff_xtr_func, specrolloff_args)


##########################
######### SPEC ###########
##########################
def spec_xtr_func(sig, fs, args, winlen, winstep):
    import python_speech_features as psf
    chnkd_sig = framesig(sig, winlen * fs, winstep * fs)
    maglist = psf.sigproc.magspec(np.array(chnkd_sig), args["nfft"])
    return np.array(maglist)
spec_args = {
    "nfft": 512
}
spec_extractor = FeatXtractr("spectrogram", spec_xtr_func, spec_args)


##########################
### HELPER FUNCTIONS #####
##########################

def slice_spectrum(feat, delta):
    '''
    Stack the spectrum in slices, such that `delta` consecutive slices 
    appear as one row. 

    If `delta` is 0, return the input spectrum.
    '''

    if delta > 0:

        slices = np.empty((np.floor(feat.shape[0] / delta), feat.shape[1] * delta))
        if np.floor(feat.shape[0] / delta) == 0:
            return None

        for i in range(0, slices.shape[0]):
            slices[i] = np.ravel(feat[i * delta:i * delta + delta, :])

    else:
        slices = feat
    return np.array(slices)

def get_win_fft(sig,winlen,fs):
    nfft = int(winlen*fs/2.)
    sigspec = abs(fft(sig))
    sigspec = sigspec[0:nfft]
    sigspec = sigspec/float(len(sigspec))
    return sigspec

def mode(ndarray,axis=0):
    if ndarray.size == 1:
        return (ndarray[0],1)
    elif ndarray.size == 0:
        raise Exception('Attempted to find mode on an empty array!')
    try:
        axis = [i for i in range(ndarray.ndim)][axis]
    except IndexError:
        raise Exception('Axis %i out of range for array with %i dimension(s)' % (axis,ndarray.ndim))
    srt = np.sort(ndarray,axis=axis)
    dif = np.diff(srt,axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1,-1) for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = np.diff(indices,axis=axis)
    location = np.argmax(bins,axis=axis)
    mesh = np.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel() for i in range(bins.ndim)]
    counts = bins[tuple(index)].reshape(location.shape)
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    return (modals, counts)