'''
This file is used as a module for EEGAnalysis in the Bachelor Research Project by Camille de Valk, spring 2019.
Some parts are inspired by the Master Thesis of Thomas Pool.
'''



#import all dependencies
import numpy as np
import matplotlib.pyplot as plt

import mne
import time
import tensorflow as tf
import keras
import pandas as pd
import gc

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import Callback

from scipy.signal import resample

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

#Initialize the random seed.
SEED = int(time.time())

global channels
channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

# Class that stores the performance of the networks after every trained epoch. Can be used as callback in keras.
class Histories(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.mcor = []
        self.val_mcor = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.mcor.append(logs.get('mcor'))
        self.val_mcor.append(logs.get('val_mcor'))
        return


def readEDF(filename, preload = True, markers = True, motion = False, CQ = False, verbose = 0, powerLineHumFilter = False):
    '''
    Import raw EDF-file from filename and returns raw-object(s).
    If preload = True, the mne.io.read_raw_edf function preloads the data, such that the data is stored in the RAM, thus it requires more memory, but is faster.
    If markers = True, the markers are also returned in a tuple, consisting of two arrays with value, time respectively
    If motionData = True, 2 objects are returned (rawData, rawMotionData).
    If CQ is True, also the contact quality measures of every channel are appended such that you have 14 EEG-channels and 14 CQ-channels (now you can always use the first 14 for analysis and if there is CQ returned, you can use channels 14-27).
    Verbose = 0 means no info during reading. Makes things cleaner when running.
    powerLineHumFilter corrects for the 50Hz hum.
    '''
    #Initialize CUDA for GPU-use for MNE-library, this makes filtering faster.
    mne.cuda.init_cuda(ignore_config = True, verbose = True)    
    
    #Channels to exclude (all the channels except for the EEG- and marker-channels).
    excludeChannels = ['COUNTER', 'INTERPOLATED', 'RAW_CQ', 'GYROX', 'GYROY', 'MARKER_HARDWARE',
                       'SYNC', 'TIME_STAMP_s', 'TIME_STAMP_ms', 'CQ_CMS', 'STI 014']
    if not CQ: #If the contact quality CQ is to be ignored, the CQ-channels are also excluded.
        excludeChannels = excludeChannels + ['CQ_AF3', 'CQ_F7', 'CQ_F3', 'CQ_FC5', 'CQ_T7', 
                                             'CQ_P7', 'CQ_O1', 'CQ_O2', 'CQ_P8', 'CQ_T8', 'CQ_FC6', 'CQ_F4', 'CQ_F8', 'CQ_AF4']
    
    #Read the raw data.
    rawData = mne.io.read_raw_edf(filename, preload = preload, verbose = verbose, exclude = excludeChannels)
    
    #All the channels you wish to read.
    channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    motionChannels = ['GYROX', 'GYROY', 'GYROZ', 'ACCX', 'ACCY', 'ACCZ', 'MAGX', 'MAGY', 'MAGZ']
    
    #Filter out the slow drifts (which are present a lot).
    rawData.filter(0.1, None, fir_design='firwin', n_jobs='cuda')
    
    #Apply a filter for the powerlinehum 50Hz.
    if powerLineHumFilter:
        rawData.notch_filter(50, notch_widths= 1, fir_design='firwin', n_jobs='cuda')
    
    #Take into account the contact quality of the EMOTIV.
    if CQ:
        channels = channels + ['CQ_AF3', 'CQ_F7', 'CQ_F3', 'CQ_FC5', 'CQ_T7', 
                               'CQ_P7', 'CQ_O1', 'CQ_O2', 'CQ_P8', 'CQ_T8', 'CQ_FC6', 'CQ_F4', 'CQ_F8', 'CQ_AF4']
    
    #Pick the channels with data.
    #Because the .pick_channels() command removes all the other data from memory, it is needed to copy the raw-file.
    if markers:
        EEGData = rawData.copy().pick_channels(channels)
    else:
        EEGData = rawData.pick_channels(channels)
    
    #Pick the markers.
    if markers:
        markerData = mne.io.read_raw_edf(filename, preload = False, verbose = verbose)[20] #20 is the marker-channel
        
        #Pick motionData
        if motion:
            rawMotionData = mne.io.read_raw_edf(filename[0:-4] + '.md.edf', preload = preload, verbose = verbose)
            motionData = rawMotionData.copy().pick_channels(motionChannels)
            print('returning EEGData, markerData, motionData')
            del rawData, rawMotionData #memorymanagement :)
            return EEGData, markerData, motionData
        
        del rawData #memorymanagement :)
        return EEGData, markerData
    
    #Pick motionData
    if motion:
        rawMotionData = mne.io.read_raw_edf(filename[0:-4] + '.md.edf', preload = preload, verbose = verbose)
        motionData = rawMotionData.copy().pick_channels(motionChannels)
        del rawData, rawMotionData #memorymanagement :)
        return EEGData, motionData
    
    del rawData #memorymanagement :)
    return EEGData




def plotEEGChannel(EEGChannel, xmin = 0, xmax = None):
    '''
    This function plots an EEG-channel, whose array has a quite odd shape.
    It is important that the input is really what the readEDF-function outputs.
    '''
    plt.figure()
    plt.plot(EEGChannel[1], EEGChannel[0][0])
    plt.xlabel('time (s)')
    plt.ylabel('data (units)') #usually this is uV, but the marker channel has arbitrary units.
    plt.xlim(xmin, xmax)
    plt.show()

    

<<<<<<< HEAD
def plotAverage(xTrain, yTrain, classes, tmin, tmax, experiment, domain = 'time', norm = False, cutOffFrequency = 64):
=======
def plotAverage(xTrain, yTrain, classes, tmin, tmax, experiment, domain = 'time', norm = False, cutOffFreq = 128):
>>>>>>> 16d51d12afb18d5c5615e88a2d7610917f7f39ce
    global channels
    ylims = []
    filenames = []
    for class_ in range(len(classes)):
        average = np.mean(xTrain[yTrain[:,class_] == 1], axis = 0)
        plt.figure(class_)
        if domain == 'time':
            for ch in range(average.shape[-1]):
                y = average[:,ch]
                if not norm:
                    y *= 1e6
                plt.plot(np.linspace(tmin, tmax, xTrain.shape[1]), y, label = channels[ch])
                plt.xlabel('time (s)', weight = 'semibold')
                plt.ylabel('electric potential ($\mu V$)', weight = 'semibold')
                if norm:
                    plt.ylabel('normalised signal (arbitrary units)')
            plt.title(f"Average signal for examples classified as: {classes[class_]}")
            filenames.append(f"./{experiment}/average-signal-exp-{experiment}-class-{classes[class_]}-tmin={tmin}-tmax={tmax}.png")
            ylims.append(plt.gca().get_ylim())
        elif domain == 'frequency':
<<<<<<< HEAD
            x = np.linspace(0, cutOffFrequency, xTrain.shape[1])
=======
            x = np.linspace(0, cutOffFreq, average.shape[-2])
>>>>>>> 16d51d12afb18d5c5615e88a2d7610917f7f39ce
            for ch in range(average.shape[-1]):
                plt.semilogy(x, average[:,ch], label = channels[ch])
                plt.xlabel('frequency (Hz)', weight = 'semibold')
                plt.ylabel('power', weight = 'semibold')
                plt.title(f"Average signal for examples classified as: {classes[class_]}")
            filenames.append(f"./{experiment}/average-spectrum-exp-{experiment}-class-{classes[class_]}-tmin={tmin}-tmax={tmax}.png")
            ylims.append(plt.gca().get_ylim())
    ylims = np.array(ylims)
    for class_ in range(len(classes)):
        plt.figure(class_)
        plt.ylim(min(ylims[:,0]), max(ylims[:,1]))
        plt.savefig(filenames[class_], dpi = 300)
        plt.legend()
    plt.show()
    

def createEvents(markers, eventIDs = np.array([4,5])):
    '''
    This function can make an event-object from a marker-object with marker-values specified in the eventIDs-array.
    '''
    eventsTime = []
    classList = []
    
    for i in np.arange(len(markers[0][0])):       
        marker = markers[0][0][i]
        if int(marker*1e6) in eventIDs: #The markers are always outputted as (almost an) interger * 1e-6, thus need to multiply by 1e6.
            eventsTime.append(i)
            classList.append(np.argwhere(eventIDs == int(marker*1e6))[0][0])
    
    previousMarkerValue = np.zeros(len(eventsTime)) #List with previous id, required to create Epoch object, no idea why :(
    
    #Stack everything in 1 array
    events = np.column_stack((eventsTime, previousMarkerValue, classList))
    events = events.astype(int)
        
    del eventsTime, previousMarkerValue, classList #memorymanagement :)
        
    gc.collect()
    return events
    

    
    
def createEventsSetIntervals(iStart, interval, numIntervals = 50, classes = ['left', 'right', 'blank'], rate = 256, experiment = 'arms',
                            lengthInterval = 10, lengthEpoch = 2, compensateImbalance = False):
    '''
    This function sets an 'event' at regular intervals for all the things in the classes-array.
    iStart is the index (in the EEG recording) of the start of the first event.
    interval is how many seconds one interval is. One interval is one entire loop.
    numIntervals is how many intervals you have.
    classes are the different classes that have to be classified.
    rate is the rate at which the data is aquired. This is especially important when using bandpower data, which is only recorded at 8 Hz.
    lengthInterval and lengthEpoch are used when the samples are so long that you can extract multiple epochs from them.
    '''
    #Make sure iStart is the right index for every rate. Assumption!! EEG-measurementrate = 256Hz.
    iStart = (iStart/256) * rate
    
    #Create a range of evenly spaced numbers, these represent the indices of the moments when an interval started.
    intervalStarts = np.arange(iStart, iStart + numIntervals*(rate*interval), rate*interval)
    
    #initialize lists
    eventsTime = []
    classList = []
    
    #switch between experiments
    if experiment == 'arms':
    
        #Initialize the left, right, blank variables
        left, right, blank = None, None, None

        #Checks againts the classes array which classes to use in this run.
        #Usually this is class1 = left arm, class2 = right arm, class3 = no movement
        for i in range(len(classes)):
            if classes[i] == 'move':
                left, right = 0, 0
            if classes[i] == 'left':
                left = i
            if classes[i] == 'right':
                right = i
            if classes[i] == 'blank':
                blank = i


        #Now we place a marker at regular intervals. Alternating between 0, 1 and 2.
        for i in range(len(intervalStarts)):
            j = iStart + i*rate*interval
            if (blank and not compensateImbalance) or (blank and np.random.random() > 0.5):
                eventsTime.append(j)
                classList.append(blank)
            eventsTime.append(j+((rate*interval)/4))
            classList.append(left)
            if (blank and not compensateImbalance) or (blank and np.random.random() > 0.5):
                eventsTime.append(j+((rate*interval)/2))
                classList.append(blank)
            eventsTime.append(j+(3*(rate*interval)/4))
            classList.append(right)
                
    
    elif experiment == 'custom':
        
        #Place markers at regular intervals. Alternating between n intergers.
        numbers = np.arange(len(classes))
        fractions = numbers/(len(classes))
        for i in range(len(intervalStarts)):
            j = iStart + i*rate*interval
            for k in numbers:
                eventsTime.append(j+((rate*interval)*fractions[k]))
                classList.append(k)        
    
    elif experiment == 'light' or 'music':
        #These experiments are special, because the intervals are so long that you can extract multiple epochs per sample
        for i in range(len(intervalStarts)):
            j = iStart + i*rate*interval
            for class_ in range(len(classes)):
                for x in range(int(lengthInterval/lengthEpoch)):
                    eventsTime.append(j + (x*lengthEpoch + class_*lengthInterval)*rate)
                    classList.append(class_)
    
    previousMarkerValue = np.zeros(len(eventsTime)) #List with previous id, required to create Epoch object, no idea why :(
    
    #Stack everything in 1 array
    events = np.column_stack((eventsTime, previousMarkerValue, classList))
    events = events.astype(int)
        
    del eventsTime, previousMarkerValue, classList #memorymanagement :)
    
    print('last event at time: ', iStart + numIntervals*(rate*interval)) #useful for checking if you did the numIntervals correct
    
    gc.collect()
    return events



def createEpochs(rawFile, events, tmin, tmax, verbose = True, CQ = False, makePSD = False, amplify = 1,
                 rate = 256, wavelet = False, freqs = np.array([0]), fmax = 64, fourier = None, newRate = None,
                 slices = 20, ICA = False, noiseValue = 1e8, normalize = True,
                 shuffle = True, fileFormat = 'timeseries', specificChannels = None,
                 badChannels = None, returnEpochs = False, cutOffFreq = 128, autoencoder = None, split = 0.9):
    '''This function creates the actual Epochs-object which consist of N entries with a time-series/timefreq-plot and for 14 channels, where N is the number of examples.
    The rawFile needed is and MNE-object of type raw.
    events is the array needed by mne.Epochs(), this can be made using the createEvents and createEventsSetIntervals functions
    tmin and tmax set the window around which the mne.Epochs() function should make Epochs using the events-array.
    verbose controls if the process should be silent or output the processing.
    There is done some preprocessing. If needed, the contact quality CQ of the EEG-channels is taken into account.
    The makePSD controls for the creation of a Power Density Spectrum made from range tmin, tmax, using the MNE-package.
    Sometimes (usaully) amplification is needed to get the signal in range of the NN-weights. A good amplification makes max(xTrain) ~ 1.
    rate is the samplingrate of the Raw-file.
    wavelet controls for the making of a wavelet-transformation, which would result in a time-freq array.
    freqs is an array of frequencies to transform the wavelets to.
    fmax controls a cut-off frequency in the PSD (using MNE).
    fourier controls the fourier-transform.
    newRate changes the resolution of this new array, because in general it is very big.
    For memory efficiency you can use slices to cut up the wavelet-transformation using slices.
    ICA controls the application of an Independent Component Analysis (ICA), for which a method exists in mne.preprocessing. The value of ICA is how many independent components are taken into account.
    If a signal is above noiseValue, it is discarded.
    normalize lets you control normalisation such that mean = 0 and std = 1.
    shuffle controls the seed of the shuffling. If false, there is nothing shuffled.
    Two fileformats can be used: timeseries and bandpower.
    specificChannels let you pick out channels.
    badChannels let you exclude channels.
    If returnEpochs = True, the MNE-epochs object is returned and not the numpy-array.
    cutOffFreq is a frequency above which the fourier transform deletes the data.
    autoencoder lets you use a pretrained autoencoder.
    '''
    gc.collect() #empty memory
    #Set verboseprint
    verboseprint = print if verbose else lambda *a, **k: None
    
    #Turn GPU-acceleration on for filters.
    mne.cuda.init_cuda(ignore_config = True, verbose = verbose)
    
    if fileFormat == 'timeseries':
        #Create EEGData (without destroying the original rawFile)
        channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
        if specificChannels:
            channels = specificChannels
        elif badChannels: #delete bad channels
            badChannelsIndices = []
            for i in badChannels:
                badChannelsIndices.append(channels.index(i))
                channels.remove(i)
                verboseprint('Dropping channel', i)
        EEGData = rawFile.copy().pick_channels(channels)
    elif fileFormat == 'bandpower':
        #The bandpower file-format has 70 channels: 5 bands for every EEG-channel.
        interesetingChannels = ['AF3_THETA','AF3_ALPHA','AF3_LOW_BETA','AF3_HIGH_BETA','AF3_GAMMA','F7_THETA','F7_ALPHA',
                'F7_LOW_BETA','F7_HIGH_BETA','F7_GAMMA','F3_THETA','F3_ALPHA','F3_LOW_BETA','F3_HIGH_BETA','F3_GAMMA',
                'FC5_THETA','FC5_ALPHA','FC5_LOW_BETA','FC5_HIGH_BETA','FC5_GAMMA','T7_THETA','T7_ALPHA','T7_LOW_BETA','T7_HIGH_BETA',
                'T7_GAMMA','P7_THETA','P7_ALPHA','P7_LOW_BETA','P7_HIGH_BETA','P7_GAMMA','O1_THETA','O1_ALPHA','O1_LOW_BETA','O1_HIGH_BETA',
                'O1_GAMMA','O2_THETA','O2_ALPHA','O2_LOW_BETA','O2_HIGH_BETA','O2_GAMMA','P8_THETA','P8_ALPHA','P8_LOW_BETA','P8_HIGH_BETA',
                'P8_GAMMA','T8_THETA','T8_ALPHA','T8_LOW_BETA','T8_HIGH_BETA','T8_GAMMA','FC6_THETA','FC6_ALPHA','FC6_LOW_BETA',
                'FC6_HIGH_BETA','FC6_GAMMA','F4_THETA','F4_ALPHA','F4_LOW_BETA','F4_HIGH_BETA','F4_GAMMA','F8_THETA','F8_ALPHA',
                'F8_LOW_BETA','F8_HIGH_BETA','F8_GAMMA','AF4_THETA','AF4_ALPHA','AF4_LOW_BETA','AF4_HIGH_BETA','AF4_GAMMA']
        if specificChannels:
            channels = specificChannels
        else:
            channels = interesetingChannels
    
    #Apply Independent Component Analysis (ICA):
    if ICA:
        if type(ICA) == bool: #meaning, if ICA == True, all the independent components are used.
            ICA = len(channels)
        ica = mne.preprocessing.ICA(n_components = ICA).fit(EEGData)
        ica.detect_artifacts(EEGData) #detect artifacts (hopefully blinks, head movement etc.)
        gc.collect() #memorymanagement
    
    #Create Epoch-object:
    xTrain = mne.Epochs(rawFile, events=events, tmin=tmin, baseline = None,
                                tmax=tmax, preload = True, verbose = verbose)
    xTrain.pick_channels(channels)
    xTrainNPArray = xTrain.copy().get_data() #make a numpy-array.
    
    #Apply the ICA:
    if ICA:
        verboseprint('Applying ICA')
        xTrain = ica.apply(xTrain)
    
    #rawFile does not have to be kept in memory if the contact quality is not used.
    if not CQ:
        del rawFile #memorymanagement
    
    #Read in target classes, which are in the second column.
    targetEvents = np.array(events[:,2])
    
    #Resample the Epochs
    if newRate:
        xTrain.resample(newRate, n_jobs='cuda')
        verboseprint('resampling')
    else:
        newRate = rate
    
    #Check for bad epochs:
    verboseprint('samples before clean:' , len(xTrainNPArray))
    index = []
    stds = np.std(xTrainNPArray, axis = (0, 2)) #the std of every channel (for every epoch)
    means = np.mean(xTrainNPArray, axis = (0, 2)) #the mean of every channel (for every epoch)
    for i in np.arange(len(xTrainNPArray)):
        #In the different fileFormats, noiseValue is used differently
        if (np.max(np.abs(xTrainNPArray[i])) > noiseValue):
            index.append(i)
        elif fileFormat == 'bandpower':
            temp = 0
            for j in range(xTrainNPArray.shape[1]):
                if np.max(xTrainNPArray[i][j]) > (noiseValue*stds[j]): #check if it is an outlier
                    temp = temp + 1
                if temp > xTrainNPArray.shape[1] / 10: #arbitraraly chosen number of channels to be bad
                    index.append(i)
                    break
    
    #Make a Power Spectrum Density for every example for every channel:
    if makePSD:
        xTrainPSD = []
        for i in range(int(((tmax - tmin)*newRate)/(2*fmax))):
            tempXTrain, freqsArray = mne.time_frequency.psd_multitaper(xTrain,
                                                                       tmin = (2*i*fmax/newRate) + tmin,
                                                                       tmax = (2*(i+1)*fmax/newRate) + tmin,
                                                                       fmin=0, fmax=fmax, verbose = verbose, n_jobs= 6)
            xTrainPSD.append(tempXTrain)
        xTrain = np.array(xTrainPSD)
        del xTrainPSD
        xTrain = np.swapaxes(xTrain, 0, 1)
        xTrain = np.swapaxes(xTrain, 1, 2)
        print(xTrain.shape) #To check the shape.
    
    #Perform the wavelet-transformation, a lot of care is taken in the memory-efficiency
    if wavelet:
        if not freqs.any():
            freqs = np.linspace(1., 128., 64.)
        print('wavelet transforming')
        tfrArrayMultitaper = np.array([]) #init
        tfrArrayMultitaperSlice = None #init
        splitArray = int((len(xTrain)) *(1/slices))
        n_cycles = freqs / 2.
        #Slice up the whole array in slices, for memory-effiency. It does make it a bit slower, but at least RAM doesn't overflow.
        while len(xTrain):
            del tfrArrayMultitaperSlice
            tfrArrayMultitaperSlice = mne.time_frequency.tfr_array_multitaper(xTrain[:splitArray], newRate, freqs, output = 'power',
                                                                     n_jobs = 6, verbose = verbose, zero_mean = True, n_cycles = n_cycles)
            gc.collect()
            if not tfrArrayMultitaper.any():
                tfrArrayMultitaper = np.float32(tfrArrayMultitaperSlice)
            else:
                tfrArrayMultitaper = np.vstack((tfrArrayMultitaper, np.float32(tfrArrayMultitaperSlice)))
            xTrain = xTrain[splitArray:]
            gc.collect()
        del tfrArrayMultitaperSlice
        del xTrain
        tfrArrayMultitaper = np.array(tfrArrayMultitaper, dtype = np.float32)
        verboseprint('done wavelet transforming to shape:', tfrArrayMultitaper.shape)
        gc.collect()
        xTrain = tfrArrayMultitaper
        del tfrArrayMultitaper
        xTrain = np.moveaxis(xTrain, 1, 3)
        print('Shape:', xTrain.shape)
        if newRate != rate:
            verboseprint('done with wavelet + resampling')
        else:
            verboseprint('done with wavelet')
    
    #convert to numpy array for more fun
    if type(xTrain) != np.ndarray and np.array(xTrain).shape == xTrainNPArray.shape:
        verboseprint('making nparray')
        epochs = xTrain.copy() #makes it possible to return epoch-object
        xTrain = xTrainNPArray
    elif type(xTrain) != np.ndarray:
        xTrain = np.array(xTrain)
        
    #Make sure that channels are always last:
    while xTrain.shape[-1] != len(channels):
        xTrain = np.moveaxis(xTrain, 1, -1)
    
    if CQ and not (wavelet or fourier): #in other words, if we still have data in the time-domain
        #Here we take into account the contact quality of the channel. Bad channels will automatically be filtered out this way.
        CQNormalized = np.array([])
        verboseprint('Applying CQ-weights to channels')
        #Make the names of the CQChannels (this is useful for when you filter out entire channels)
        CQChannels = []
        for i in range(len(channels)):
            CQChannels.append('CQ_'+channels[i])
        CQData = rawFile.copy().pick_channels(CQChannels)
        for i in range(xTrain.shape[-1]):
            CQNormalized = np.append(CQNormalized, np.mean(CQData[i][0][0])*CQ)
        #Here, weird CQ-values are manually filtered out
        for i in range(len(CQNormalized)):
            if (CQNormalized[i]/1e-22 > 1e2) or (CQNormalized[i]/1e-22 < 1e-2):
                CQNormalized[i] = 0.
        for ch in range(xTrain.shape[-1]):
            xTrain[:,:,ch] = xTrain[:,:,ch] * (CQNormalized[ch] * 1e22) #The CQ is a very small value ~1e-22
        xTrain = xTrain[:,:,0:int(len(xTrain[-1])/2)]
    elif CQ and wavelet:
        print('CQ can (for now) only be considered with 1D-data.')
    
    #Normalize the data:
    if (normalize and not (fourier or wavelet)) or (normalize and fourier and amplify == 'log'):
        verboseprint('normalizing data, mean & std before: ', np.mean(xTrain), np.std(xTrain))
        xTrain = (xTrain - np.mean(xTrain))/np.std(xTrain)
        verboseprint('mean & std after: ', np.mean(xTrain), np.std(xTrain))
    elif normalize and fourier:
        verboseprint('normalizing data, max before:', np.max(xTrain))
        xTrain = xTrain/np.std(xTrain)
        verboseprint('max after:', np.max(xTrain))
        
    '''if autoencoder and not wavelet: #in other words, if we still have 1D data in the time-domain
        for i in badChannelsIndices:
            xTrain = np.insert(xTrain, 0, i, axis = 1)
        autoencoder = load_model(autoencoder)
        print(xTrain.shape)
        #print('Subtracting prediction by the autoencoder')
        autoEncPrediction = autoencoder.predict(np.moveaxis(xTrain, 1, 2))'''
        
    if fourier:
        #Do a fourier transform of every epoch. Is a more simple way of doing makePSD
        data = xTrain
        xTrainFourier = np.abs(np.fft.fft(data, axis=1))
        xTrainFourier = xTrainFourier[:,0:int(xTrainFourier.shape[-2]/2),:]
        xTrain = xTrainFourier
        df = rate/xTrain.shape[-2]
        if cutOffFreq:
<<<<<<< HEAD
            xTrain = xTrain[:,0:int(cutOffFreq/df),:]
=======
            xTrain = xTrain[:,0:int(2*cutOffFreq/df),:]
>>>>>>> 16d51d12afb18d5c5615e88a2d7610917f7f39ce
        verboseprint('done fourier transform')
    
    yTrain = packClasses(targetEvents)
    
    #Amplify the data:
    if amplify == 'log':
        xTrain = np.log10(xTrain)
    elif amplify:
        xTrain = xTrain*amplify
    
    #Delete bad samples:
    xTrain = np.delete(xTrain, index, axis = 0)
    yTrain = np.delete(yTrain, index, axis = 0)
    verboseprint('samples after clean:' , len(xTrain))
    
    #shuffle things around (caution: you do need to set the seed again, otherwise things will get lost)
    if shuffle:
        verboseprint('Everyday I\'m shuffling')
        np.random.seed(SEED)
        np.random.shuffle(xTrain)
        np.random.seed(SEED)
        np.random.shuffle(yTrain)
        
    #split in test and trainset:
    if not split:
        split = (len(xTrain))
    else:
        split = int((len(xTrain)) *split)
    xTest = xTrain[split:]
    xTrain = xTrain[:split]
    yTest = yTrain[split:]
    yTrain = yTrain[:split]

    print('Done making training/test-sets, shapes: ', xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)
    
    if xTest.shape[0] != yTest.shape[0]:
        raise Exception('Something went wrong in picking out the Epochs, probably wrong marking.')
    
    gc.collect()
    if returnEpochs:
        return epochs
    return (xTrain, yTrain),(xTest, yTest)


#Function that builds a nice name for the neural networks
def makeNameModel(model):
    name = ''
    for i in  model.get_config()['layers']:
        config_ = i['config']
        name += str(i['class_name'])
        
        info = ['units', 'rate', 'filters', 'kernel_size', 'pool_size']
        for i in info:
            if i in config_:
                temp = config_[i]
                if type(config_[i]) == tuple:
                    temp = '.'
                    for j in range(len(config_[i])):
                        if j > 0:
                            temp += 'x'
                        temp += str(config_[i][j])
                name += str(temp)
                #name += '-'
                #break
        name += '-'
    name += type(model.optimizer).__name__ + '-' + 'LR='
    name += '{:.0e}'.format(model.optimizer.get_config()['lr'])
    name = name.replace('Flatten', 'Fl')
    name = name.replace('Dense', 'D')
    name = name.replace('Dropout', 'DO=')
    name = name.replace('Conv1DTranspose', '')
    name = name.replace('Conv1D', 'C')
    name = name.replace('Conv2D', 'C')
    name = name.replace('AveragePooling1D', 'A')
    name = name.replace('AveragePooling2D', 'A')
    name = name.replace('UpSampling1D', 'Up')
    name = name.replace('UpSampling2D', 'Up')
    name = name.replace('LocallyConnected1D', 'Loc')
    name = name.replace('MaxPooling1D', 'M')
    name = name.replace('MaxPooling2D', 'M')
    name = name.replace('Reshape', 'Rshp')
    return name
 
#Function that displays confusion matrix.
def dispMat(x, names = None):
    return True
    if names:
        index = names
    else:
        index = list(range(x.shape[0]))
    display(pd.DataFrame(x, index = index, columns = index))
    

#Pack the classes as a one-hot (binary) vector.
def packClasses(numericalClassesArray):
    nClasses = int(np.max(numericalClassesArray) + 1)
    categoricalTrain = np.array([])
    for i in numericalClassesArray:
        temp = np.zeros(nClasses)
        temp[i] = 1
        categoricalTrain = np.append(categoricalTrain, [temp])
    
    return np.array(categoricalTrain.reshape(len(numericalClassesArray), nClasses))    


#Unpack the classes again to numerical.
def unpackClasses(arrayData):
    numericalClassesArray = np.array([])
    for i in arrayData:
        numericalClassesArray = np.append(numericalClassesArray, np.argmax(i))
    return numericalClassesArray


#Calculates the f1-measure with Keras-backend. Source: https://github.com/keras-team/keras/issues/5400#issuecomment-314747992
def f1(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    TP = K.sum(y_pos * y_pred_pos)
    FP = K.sum(y_neg * y_pred_pos)
    FN = K.sum(y_pos * y_pred_neg)
    
    return 2*TP/(2*TP + FP + FN + K.epsilon())

def mcor(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    TP = K.sum(y_pos * y_pred_pos)
    TN = K.sum(y_neg * y_pred_neg)
    
    FP = K.sum(y_neg * y_pred_pos)
    FN = K.sum(y_pos * y_pred_neg)
    
    numerator = (TP * TN - FP * FN)
    denominator = K.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    return  numerator / (denominator + K.epsilon())