#!/usr/bin/env python

'''
    NAME    : LDC TIMIT Dataset
    URL     : https://catalog.ldc.upenn.edu/ldc93s1
    HOURS   : 5
    TYPE    : Read - English
    AUTHORS : Garofolo, John, et al.
    TYPE    : LDC Membership
    LICENCE : LDC User Agreement
'''

import errno
import os
from os import path
import sys
import tarfile
import fnmatch
import pandas as pd
import subprocess
import argparse
from mapping import phone_maps
import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy as np
timit_phone_map = phone_maps(mapping_file="kaldi_60_48_39.map")

def clean(word):
    # LC ALL & strip punctuation which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new

def compute_mfcc(wav, n_delta=0):
    mfcc_feat = psf.mfcc(wav)
    if(n_delta == 0):
        return(mfcc_feat)
    elif(n_delta == 1):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1))))
    elif(n_delta == 2):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1), psf.delta(mfcc_feat, 2))))
    else:
        return 0

def read_transcript(full_wav):
    trans_file = full_wav[:-8] + ".PHN"
    with open(trans_file, "r") as file:
        trans = file.readlines()
    durations = [ele.strip().split(" ")[:-1] for ele in trans]
    durations_int = []
    for duration in durations:
        durations_int.append([int(duration[0]), int(duration[1])])
    trans = [ele.strip().split(" ")[-1] for ele in trans]
    trans = [timit_phone_map.map_symbol_reduced(symbol=phoneme) for phoneme in trans]
    # trans = " ".join(trans)
    return trans, durations_int

def _preprocess_data(args):
    datapath = args.timit
    pre_processed = args.preprocessed
    print("Preprocessing data")
    print(pre_processed)

    if(pre_processed):
        print("Preprocessing is already done")
    full_wavs = []
    full_wavs_train = []
    full_wavs_test = []
    for root, dirnames, filenames in os.walk(datapath):
        for filename in fnmatch.filter(filenames, "*.WAV"):
            wav = os.path.join(root, filename)[:-4] + "_rif.wav"
            sph = os.path.join(root, filename)
            full_wavs.append(wav)
            if("TEST" in wav):
                full_wavs_test.append(wav)
            else:
                full_wavs_train.append(wav)
            print("converting {} to {}".format(sph, wav))
            if(~pre_processed):
                subprocess.check_call(["sox", sph, wav])

    print("Preprocessing Complete")

    mfcc_features = []
    mfcc_labels = []

    for full_wav in full_wavs:
        print("Computing features for file: ", full_wav)

        trans, durations = read_transcript(full_wav = full_wav)
        n_delta = int(args.n_delta)
        labels = []

        (sample_rate,wav) = wav.read(full_wav)
        mfcc_feats = compute_mfcc(wav[durations[0][0]:durations[0][1]], n_delta=n_delta)

        for i in range(len(mfcc_feats)):
                labels.append(trans[0])
        for index, chunk in enumerate(durations[1:]):
            mfcc_feat = compute_mfcc(wav[chunk[0]:chunk[1]], n_delta=n_delta)
            mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
            for i in range(len(mfcc_feat)):
                labels.append(trans[index])
        mfcc_features.extend(mfcc_feats)
        mfcc_labels.extend(labels)
    


    timit_df = pd.DataFrame()
    timit_df["features"] = mfcc_features
    timit_df["labels"] = mfcc_labels
    
    if(n_delta==0):
        timit_df.to_hdf("./features/mfcc/timit.hdf", "timit")
    elif(n_delta==1):
        timit_df.to_hdf("./features/mfcc_delta/timit.hdf", "timit")
    elif(n_delta==2):
        timit_df.to_hdf("./features/mfcc_delta_delta/timit.hdf", "timit")

    mfcc_features_train=[]
    mfcc_labels_train=[]
    for full_wav in full_wavs_train:
        print("Computing for file: ", full_wav)
        trans, durations = read_transcript(full_wav = full_wav)
        n_delta = int(args.n_delta)
        labels = []
        (sample_rate,wav) = wav.read(full_wav)
        mfcc_feats = compute_mfcc(wav[durations[0][0]:durations[0][1]], n_delta=n_delta)
        for i in range(len(mfcc_feats)):
            labels.append(trans[0])
        for index, chunk in enumerate(durations[1:]):
            mfcc_feat = compute_mfcc(wav[chunk[0]:chunk[1]], n_delta=n_delta)
            mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
            for i in range(len(mfcc_feat)):
                labels.append(trans[index])
        mfcc_features_train.extend(mfcc_feats)
        mfcc_labels_train.extend(labels)
    timit_df = pd.DataFrame()
    timit_df["features"] = mfcc_features_train
    timit_df["labels"] = mfcc_labels_train

    if(n_delta==0):
        timit_df.to_hdf("./features/mfcc/timit_train.hdf", "timit")
    elif(n_delta==1):
        timit_df.to_hdf("./features/mfcc_delta/timit_train.hdf", "timit")
    elif(n_delta==2):
        timit_df.to_hdf("./features/mfcc_delta_delta/timit_train.hdf", "timit")

    mfcc_sentence_test=[]
    mfcc_region_test=[]
    mfcc_labels_test=[]
    mfcc_speaker_test=[]
    mfcc_features_test=[]
    for full_wav in full_wavs_test:
        print("Computing features for file: ", full_wav)
        dir_list=full_wav.split("/")
        trans, durations = read_transcript(full_wav = full_wav)
        n_delta = int(args.n_delta)
        labels = []
        temp_region=[]
        temp_speaker=[]
        temp_sentence=[]
        (sample_rate,wav) = wav.read(full_wav)
        mfcc_feats = compute_mfcc(wav[durations[0][0]:durations[0][1]], n_delta=n_delta)
        for i in range(len(mfcc_feats)):
            labels.append(trans[0])
            temp_region.append(dir_list[-3])
            temp_speaker.append(dir_list[-2])
            temp_sentence.append((dir_list[-1].split("_"))[0])
        for index, chunk in enumerate(durations[1:]):
            mfcc_feat = compute_mfcc(wav[chunk[0]:chunk[1]], n_delta=n_delta)
            mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
            for i in range(len(mfcc_feat)):
                labels.append(trans[index])
                temp_region.append(dir_list[-3])
                temp_speaker.append(dir_list[-2])
                temp_sentence.append((dir_list[-1].split("_"))[0])
        mfcc_features_test.extend(mfcc_feats)
        mfcc_labels_test.extend(labels)
        mfcc_region_test.extend(temp_region)
        mfcc_speaker_test.extend(temp_speaker)
        mfcc_sentence_test.extend(temp_sentence)

    timit_df = pd.DataFrame()
    timit_df["features"] = mfcc_features_test
    timit_df["labels"] = mfcc_labels_test
    timit_df["region"]=mfcc_region_test
    timit_df["speaker"]=mfcc_speaker_test
    timit_df["sentence"]=mfcc_sentence_test

    if(n_delta==0):
        timit_df.to_hdf("./features/mfcc/timit_test.hdf", "timit")
    elif(n_delta==1):
        timit_df.to_hdf("./features/mfcc_delta/timit_test.hdf", "timit")
    elif(n_delta==2):
        timit_df.to_hdf("./features/mfcc_delta_delta/timit_test.hdf", "timit")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--timit', type=str, default="./",
                       help='TIMIT root directory')
    parser.add_argument('--n_delta', type=str, default="0",
                       help='Number of delta features to compute')
    parser.add_argument('--preprocessed', type=bool, default=False,
                       help='Set to True if the preprocessing already completed')

    args = parser.parse_args()
    print(args)
    print("TIMIT path is: ", args.timit)
    _preprocess_data(args)
    print("Completed")
