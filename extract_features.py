# -*- coding: utf-8 -*-
"""
extract_features.py
Extracts MFCC features from wav files

Requires librosa and numpy
Created on Sat Feb 21 13:10:40 2015

@author: vurga
"""

import numpy as np, librosa, sys

def get_mfcc(sound_file, num_coeffs=13, deltas=False, ddeltas=False):
    """Computes MFCC features and deltas
    
    Parameters: sound_file -- file name of wav file to be processed
                num_coeffs -- no. of MFCC coefficients (default: 13)
                deltas -- if True, first-order derivatives are appended
                ddeltas -- if True, second-order derivatives are appended
    
    Returns MFCC feature vector (+ delta & ddelta values)"""
    
    #load file    
    try:
        wav, sample_rate = librosa.load(sound_file)
    except IOError:
        print('File not found: ' + sound_file)
        return
    
    #Compute MFCC features from signal
    mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=num_coeffs)
    
    #Append D and DD values if necessary
    if deltas:
        mfcc_deltas = librosa.feature.delta(mfcc)
        if ddeltas:
            mfcc_ddeltas = librosa.feature.delta(mfcc_deltas)
            mfcc = np.concatenate((mfcc, mfcc_deltas, mfcc_ddeltas), axis=0)
        else:
            mfcc = np.concatenate((mfcc, mfcc_deltas), axis=0)
    
    return np.transpose(mfcc)
    
def compute_avg(mfcc_array, num_coeffs=13, deltas=False, ddeltas=False):
    mfcc_avg = np.mean(mfcc_array[:,:num_coeffs], axis=0)
    if deltas:
        deltas_avg = np.mean(np.absolute(\
        mfcc_array[:,num_coeffs:num_coeffs*2]), axis=0)
        if ddeltas:
            ddeltas_avg = np.mean(np.abs(mfcc_array[:,num_coeffs*2:]), axis=0)
            return np.concatenate((mfcc_avg, deltas_avg, ddeltas_avg), axis=1)
            #return mfcc_avg, deltas_avg, ddeltas_avg
        else:
            return np.concatenate((mfcc_avg, deltas_avg), axis=0)
            #return mfcc_avg, deltas_avg
    else:
        return mfcc_avg

if __name__ == "__main__":
    print(np.shape(get_mfcc(sys.argv[1], num_coeffs=26, 
                   deltas=True, ddeltas=True)))
                   
