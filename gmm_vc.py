"""GMM-based voice conversion"""

from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.datasets.cmu_arctic import CMUArcticWavFileDataSource
from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.util import apply_each2d_trim
from nnmnkwii.preprocessing import remove_zeros_frames, delta_features
from nnmnkwii.metrics import melcd
# from nnmnkwii.baseline.gmm import MLPG

from os.path import join, expanduser, basename, splitext,exists
import sys, os
import time

import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
from pysptk.synthesis import MLSADF, Synthesizer


import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
import bandmat as bm
import bandmat.linalg as bla

DATA_ROOT = '/home/chander/Documents/study_material/speech/project/cmu_arctic/'

fs = 16000
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
order = 2 #define order here 
frame_period = 5
max_files = 10 #numer of files to ve used 

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

class CMUArcticSpectrumDataSource(CMUArcticWavFileDataSource):
    def __init__(self, *args, **kwargs):
        super(CMUArcticSpectrumDataSource, self).__init__(*args, **kwargs)
        self.test_paths = None

    def collect_files(self):
        paths = super(
            CMUArcticSpectrumDataSource, self).collect_files()
        paths_train, paths_test = train_test_split(
            paths, test_size=0.2, random_state=1234)

        # keep paths for later testing
        self.test_paths = paths_test

        return paths_train

    def collect_features(self, path):
        fs, x = wavfile.read(path)
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
        return mc

def build_win_mats(windows, frames):
    """Builds a window matrix of a given size for each window in a collection.
    """
    winMats = []
    for l, u, winCoeff in windows:
        assert l >= 0 and u >= 0
        assert len(winCoeff) == l + u + 1
        winCoeffs = np.tile(np.reshape(winCoeff, (l + u + 1, 1)), frames)
        winMat = bm.band_c_bm(u, l, winCoeffs).T
        winMats.append(winMat)
  
    return winMats
def build_poe(bFrames, tauFrames, winMats, sdw = None):
    if sdw is None:
        sdw = max([ winMat.l + winMat.u for winMat in winMats ])
    numWindows = len(winMats)
    frames = len(bFrames)
    assert np.shape(bFrames) == (frames, numWindows)
    assert np.shape(tauFrames) == (frames, numWindows)
    assert all([ winMat.l + winMat.u <= sdw for winMat in winMats ])
  
    b = np.zeros((frames,))
    prec = bm.zeros(sdw, sdw, frames)
  
    for winIndex, winMat in enumerate(winMats):
        bm.dot_mv_plus_equals(winMat.T, bFrames[:, winIndex], target = b)
        bm.dot_mm_plus_equals(winMat.T, winMat, target_bm = prec,
                              diag = tauFrames[:, winIndex])
  
    return b, prec

def mlpg(mean_frames, variance_frames, windows):
    """Maximum Parameter Likelihood Generation (MLPG)
    """
    dtype = mean_frames.dtype
    T, D = mean_frames.shape
    # expand variances over frames
    if variance_frames.ndim == 1 and variance_frames.shape[0] == D:
        variance_frames = np.tile(variance_frames, (T, 1))
    assert mean_frames.shape == variance_frames.shape
    static_dim = D // len(windows)

    num_windows = len(windows)
    win_mats = build_win_mats(windows, T)

    # workspaces; those will be updated in the following generation loop
    means = np.zeros((T, num_windows))
    precisions = np.zeros((T, num_windows))
    # Perform dimension-wise generation
    y = np.zeros((T, static_dim), dtype=dtype)
    for d in range(static_dim):

        for win_idx in range(num_windows):
            means[:, win_idx] = mean_frames[:, win_idx * static_dim + d]
            precisions[:, win_idx] = 1 / \
                variance_frames[:, win_idx * static_dim + d]

        bs = precisions * means
        b, P = build_poe(bs, precisions, win_mats)
        y[:, d] = bla.solveh(P, b)

    return y


class MLPGBase(object):
    def __init__(self, gmm, swap=False, diff=False):
        assert gmm.covariance_type == "full"
        # D: static + delta dim
        D = gmm.means_.shape[1] // 2
        self.num_mixtures = gmm.means_.shape[0]
        self.weights = gmm.weights_

        # Split source and target parameters from joint GMM
        self.src_means = gmm.means_[:, :D]
        self.tgt_means = gmm.means_[:, D:]
        self.covarXX = gmm.covariances_[:, :D, :D]
        self.covarXY = gmm.covariances_[:, :D, D:]
        self.covarYX = gmm.covariances_[:, D:, :D]
        self.covarYY = gmm.covariances_[:, D:, D:]

        if diff:
            self.tgt_means = self.tgt_means - self.src_means
            self.covarYY = self.covarXX + self.covarYY - self.covarXY - self.covarYX
            self.covarXY = self.covarXY - self.covarXX
            self.covarYX = self.covarXY.transpose(0, 2, 1)

        # swap src and target parameters
        if swap:
            self.tgt_means, self.src_means = self.src_means, self.tgt_means
            self.covarYY, self.covarXX = self.covarXX, self.covarYY
            self.covarYX, self.covarXY = self.covarXY, self.covarYX

        # p(x), which is used to compute posterior prob. for a given source
        # spectral feature in mapping stage.
        self.px = GaussianMixture(
            n_components=self.num_mixtures, covariance_type="full")
        self.px.means_ = self.src_means
        self.px.covariances_ = self.covarXX
        self.px.weights_ = self.weights
        self.px.precisions_cholesky_ = _compute_precision_cholesky(
            self.px.covariances_, "full")

    def transform(self, src):
        if src.ndim == 2:
            tgt = np.zeros_like(src)
            for idx, x in enumerate(src):
                y = self._transform_frame(x)
                tgt[idx][:len(y)] = y
            return tgt
        else:
            return self._transform_frame(src)

    def _transform_frame(self, src):
        """Mapping source spectral feature x to target spectral feature y
        so that minimize the mean least squared error.
        More specifically, it returns the value E(p(y|x)].
        Args:
            src (array): shape (`order of spectral feature`) source speaker's
                spectral feature that will be transformed
        Returns:
            array: converted spectral feature
        """
        D = len(src)

        # Eq.(11)
        E = np.zeros((self.num_mixtures, D))
        for m in range(self.num_mixtures):
            xx = np.linalg.solve(self.covarXX[m], src - self.src_means[m])
            E[m] = self.tgt_means[m] + self.covarYX[m].dot(xx)

        # Eq.(9) p(m|x)
        posterior = self.px.predict_proba(np.atleast_2d(src))

        # Eq.(13) conditinal mean E[p(y|x)]
        return posterior.dot(E).flatten()

class MLPG(MLPGBase):
    """Maximum likelihood Parameter Generation (MLPG) for GMM-basd voice
    conversion [1]_.

   
    """

    def __init__(self, gmm, windows=None, swap=False, diff=False):
        super(MLPG, self).__init__(gmm, swap, diff)
        if windows is None:
            windows = [
                (0, 0, np.array([1.0])),
                (1, 1, np.array([-0.5, 0.0, 0.5])),
            ]
        self.windows = windows
        self.static_dim = gmm.means_.shape[-1] // 2 // len(windows)

    def transform(self, src):
        """Mapping source feature x to target feature y so that maximize the
        likelihood of y given x.
        """
        T, feature_dim = src.shape[0], src.shape[1]

        if feature_dim == self.static_dim:
            return super(MLPG, self).transform(src)

        # A suboptimum mixture sequence  (eq.37)
        optimum_mix = self.px.predict(src)

        # Compute E eq.(40)
        E = np.empty((T, feature_dim))
        for t in range(T):
            m = optimum_mix[t]  # estimated mixture index at time t
            xx = np.linalg.solve(self.covarXX[m], src[t] - self.src_means[m])
            # Eq. (22)
            E[t] = self.tgt_means[m] + np.dot(self.covarYX[m], xx)

        # Compute D eq.(23)
        # Approximated variances with diagonals so that we can do MLPG
        # efficiently in dimention-wise manner
        D = np.empty((T, feature_dim))
        for t in range(T):
            m = optimum_mix[t]
            # Eq. (23), with approximating covariances as diagonals
            D[t] = np.diag(self.covarYY[m]) - np.diag(self.covarYX[m]) / \
                np.diag(self.covarXX[m]) * np.diag(self.covarXY[m])

        # Once we have mean and variance over frames, then we can do MLPG
        return mlpg(E, D, self.windows)

source = CMUArcticSpectrumDataSource(data_root=DATA_ROOT,
                                         speakers=["ksp"], max_files=max_files)
target = CMUArcticSpectrumDataSource(data_root=DATA_ROOT,
                                         speakers=["slt"], max_files=max_files)

# Build dataset as 3D tensor (NxTxD)
X = FileSourceDataset(source).asarray(padded_length=1200)
Y = FileSourceDataset(target).asarray(padded_length=1200)

# Alignment
X, Y = DTWAligner(verbose=0, dist=melcd).transform((X, Y))

# Drop 1st dimention
X, Y = X[:, :, 1:], Y[:, :, 1:]

static_dim = X.shape[-1]

X = apply_each2d_trim(delta_features, X, windows)
Y = apply_each2d_trim(delta_features, Y, windows)

# Joint features
XY = np.concatenate((X, Y), axis=-1).reshape(-1, X.shape[-1] * 2)
XY = remove_zeros_frames(XY)
print(XY.shape)
gmm = GaussianMixture(
    n_components=2, covariance_type="full", max_iter=100, verbose=1)

gmm.fit(XY)

# Parameter generation
paramgen = MLPG(gmm, windows=windows, diff=True)

# Waveform generation for test set
for idx, path in enumerate(source.test_paths):
    fs, x = wavfile.read(path)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    # aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

    mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    
    mc = delta_features(mc, windows)
    since = time.time()
    mc = paramgen.transform(mc)
    print("{}, Elapsed time in conversion: {}s".format(idx, time.time() - since))
    assert mc.shape[-1] == static_dim
    mc = np.hstack((c0[:, None], mc))

    mc[:, 0] = 0
    engine = Synthesizer(MLSADF(order=order, alpha=alpha), hopsize=80)
    b = pysptk.mc2b(mc.astype(np.float64), alpha=alpha)
    waveform = engine.synthesis(x, b)
    if not exists('resultsVC'):
        os.makedirs('resultsVC')
    wavfile.write("resultsVC/{}_{}.wav".format(splitext(basename(path))[0],'mlpg'),
                  fs, waveform.astype(np.int16))