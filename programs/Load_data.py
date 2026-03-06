import mne
import scipy.io
import numpy as np
import warnings
import random
from sklearn.model_selection import train_test_split

mne.set_log_level('WARNING') # Just display messages warning level above
warnings.filterwarnings("ignore", message="Channel names are not unique")
warnings.filterwarnings("ignore", message="Highpass cutoff frequency 100.0 is greater than lowpass cutoff frequency 0.5, setting values to 0 and Nyquist.")


def Load_BCI2a_data(root_path,subject,type,samples = "1000"):
    data_path = root_path + 'Data/A{:02d}{}.gdf'.format(subject,type)
    # Đọc tệp GDF
    raw = mne.io.read_raw_gdf(data_path, preload=True, verbose= False)
    raw.filter(l_freq=None, h_freq=40, verbose=False)
    events, event_ids = mne.events_from_annotations(raw,verbose= False)

    label_path = root_path + 'True_labels/A{:02d}{}.mat'.format(subject,type)
    if(type == "T"):
        # lấy data từ file Train từ 1.5s đến 6s cho 22 kênh EEG
        # event_id: [769 0x0301 Cue onset left (class 1), 770 0x0302 Cue onset right (class 2),771 0x0303 Cue onset foot (class 3),772 0x0304 Cue onset tongue (class 4)]
        event_id = [event_ids[key] for key in ["769","770","771","772"]]
        if samples == '1000':
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=4-1/250, baseline=None, verbose= False)
        elif samples == '1125':
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=4-1/250, baseline=None, verbose= False)
        elif samples == '750':
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=1, tmax=4-1/250, baseline=None, verbose= False)
        data = epochs.get_data(verbose=False)[:,:22,:]
        label = scipy.io.loadmat(label_path)['classlabel']

    if(type == "E"):
        # lấy data từ file Train từ 1.5s đến 6s cho 22 kênh EEG
        # event_id: [783 0x030F Cue unknown]
        event_id = [event_ids[key] for key in ["783"]]
        if samples == '1000':
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=4-1/250, baseline=None,verbose= False)
        elif samples == '1125':
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=4-1/250, baseline=None, verbose= False)
        elif samples == '750':
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=1, tmax=4-1/250, baseline=None, verbose= False)
        data = epochs.get_data(verbose= False)[:,:22,:]
        label = scipy.io.loadmat(label_path)['classlabel']     
    return data,(label-1).flatten()

def Load_BCI2b_data(root_path,subject,type,samples = "1000"):
    data = []
    label = []
    if type == 'T':
        sessions = {1,2,3}
    elif type == 'E':
        sessions = {4,5}
    for session in sessions:
        data_path = root_path +  'B{:02d}{:02d}{}.gdf'.format(subject,session,type)
        # Đọc tệp GDF
        raw = mne.io.read_raw_gdf(data_path, preload=True,verbose=False)
        raw.filter(l_freq=None, h_freq=40, verbose=False)

        events, event_ids = mne.events_from_annotations(raw,verbose=False)
        label_path = root_path + 'True_labels/B{:02d}{:02d}{}.mat'.format(subject,session,type)

        if(type == "T"):
            # lấy data từ file Train từ 3s đến 7s cho 3 kênh EEG
            # event_id: [769 0x0301 Cue onset left (class 1), 770 0x0302 Cue onset right (class 2)]
            event_id = [event_ids[key] for key in ["769","770"]]
            if samples == "1000":
                epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=4-1/250, baseline=None,verbose=False)
            elif samples == "1125":
                epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=4-1/250, baseline=None,verbose=False)
            data_T = epochs.get_data(verbose=False)[:120,:3,:]
            label_T = scipy.io.loadmat(label_path)['classlabel'][:120]
            data.append(data_T)
            label.append(label_T)

        if(type == "E"):
            # lấy data cho file Evalution từ 3s đến 7s cho 3 kênh EEG
            # event_id: 781 0x030D BCI feedback (continuous)
            event_id = [event_ids[key] for key in ["781"]]
            if samples == "1000":
                epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=3.5-1/250, baseline=None,verbose=False)
            elif samples == "1125":
                epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-1.0, tmax=3.5-1/250, baseline=None,verbose=False)
            data_E = epochs.get_data(verbose=False)[:120,:3,:]
            label_E = scipy.io.loadmat(label_path)['classlabel'][:120]
            data.append(data_E)
            label.append(label_E)
    data = np.concatenate(data,axis=0)
    label = np.concatenate(label,axis=0)
    return data,(label-1).flatten()

def Load_Physionet_data(root_path, nsub):
    datas = []
    labels = []
    runs = {4,6,8,10,12,14} # MI EEG: left fist, right fist, both fist and both feet
    for run in runs:
        raw = mne.io.read_raw_edf(input_fname= root_path + "S{:03d}/".format(nsub) + "S{:03d}R{:02d}.edf".format(nsub,run) , preload=True, verbose= False)
        raw.notch_filter(freqs=60)
        raw.filter(l_freq=0.5,h_freq=40)

        events, event_ids = mne.events_from_annotations(raw,verbose= False)
        event_id = [event_ids[key] for key in ["T1","T2"]] # Task events
        # print(event_id)

        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0+1/160, tmax=4, baseline=None, verbose= False, preload=False)
        equal_trials,_ = epochs.equalize_event_counts(method="mintime")    # For equlize trials per task by using event count
        data = equal_trials.get_data(verbose=False)
        datas.append(data)
        if run in  {4, 8, 12}:
            label = equal_trials.events[:,2] - 2  # Label = 0: MI Left fist, label = 1: MI Right fist
        elif run in {6, 10, 14}:
            label = equal_trials.events[:,2]    # Label = 2: MI Both fist, label = 3: MI Both feet
        labels.append(label)
        annot = equal_trials.get_annotations_per_epoch()    # Print annotation and time period of each event

    datas = np.concatenate(datas,axis=0)
    labels = np.concatenate(labels,axis=0) 

    # Apply z score
    datas = standardize_data_cross_subject(datas)
    return datas,labels

def standardize_data(X_train, X_test): 
    Train_mean = np.mean(X_train)
    Train_std = np.std(X_train)
    X_train = (X_train - Train_mean)/Train_std
    X_test = (X_test - Train_mean)/Train_std
    return X_train, X_test

def standardize_data_cross_subject(x): 
    # print(np.shape(x))
    mean = np.mean(x)
    std = np.std(x)
    x_zscored = (x - mean)/std
    return x_zscored


def get_data(get_datapath, subject, dataset = 'BCI2a', isStandard = True, seed_n=0, Shuffle = True):
    # print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    
    if dataset == 'BCI2a':    
        X_train, y_train = Load_BCI2a_data(root_path= get_datapath, subject = subject+1,type="T",samples='1000')
        X_test, y_test = Load_BCI2a_data(root_path= get_datapath, subject = subject+1,type="E",samples='1000')
    
    if dataset == 'BCI2b':
        X_train, y_train = Load_BCI2b_data(root_path= get_datapath, subject = subject+1,type="T",samples='1000')
        X_test, y_test = Load_BCI2b_data(root_path= get_datapath, subject = subject+1,type="E",samples='1000')
    
    # Shuffle the data
    if Shuffle:
        shuffle_num = np.random.permutation(len(X_train))        
        X_train = X_train[shuffle_num,:, :]
        y_train = y_train[shuffle_num]
        
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test)
    
    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)
    return X_train,y_train,X_test,y_test

def cross_validation(root_path,n_splits:int, K:int):
    """K fold Cross validaion of Physionet datasets
    Args:
        n_splits (int): Number of folds. Must be at least 2.
        K (int): The K-th iteration
    """
    # Split Subjects of each K iteration
    print(f"\n-----Fold {K+1}-----")
    subject_indices = np.linspace(1,100,100, dtype=int) # Subjects 1 -> 100
    fold_size = len(subject_indices) // n_splits
    test_indices = subject_indices[K * fold_size:(K + 1) * fold_size]
    train_val_indices = np.setdiff1d(subject_indices, test_indices)
    print(f"Train Val set: Subject {train_val_indices[0]} -> Subject {train_val_indices[-1]}")
    print(f"Test set: Subject {test_indices[0]} -> Subject {test_indices[-1]}")

    # Load data
    Train_val_datas, Train_val_labels = (np.concatenate(x, axis=0) for x in zip(*[Load_Physionet_data(root_path,nsub) for nsub in train_val_indices]))
    Test_datas, Test_labels = (np.concatenate(x, axis=0) for x in zip(*[Load_Physionet_data(root_path,nsub) for nsub in test_indices]))
    
    # Split Train validation set into Train set and Val set
    # Train_datas,Val_datas,Train_labels,Val_labels = train_test_split(Train_val_datas,Train_val_labels,test_size=0.2,shuffle=False)
    Train_datas,Val_datas,Train_labels,Val_labels = train_test_split(Train_val_datas,Train_val_labels,test_size=0.2,shuffle=True, random_state=1)

    print(f"Train Shape: {np.shape(Train_datas)}, Val Shape: {np.shape(Val_datas)}, Test Shape: {np.shape(Test_datas)}")

    return Train_datas,Train_labels,Val_datas,Val_labels,Test_datas,Test_labels

def LOSO(dataset,root_path,K,Train = True):
# Split Subjects of each K iteration
    print(f"\n-----Fold {K+1}-----")
    subject_indices = np.linspace(1,9,9, dtype=int) # Subjects 1 -> 9
    n = len(subject_indices)

    test_nsub = subject_indices[K]
    val_indices = [(K+8) % n, (K+7) % n]
    val_nsub = [subject_indices[i] for i in val_indices]
    train_nsub = [x for x in subject_indices if x not in [test_nsub] +val_nsub]
    print(f"Test Subject: {test_nsub}, Val Subject: {val_nsub}, Train Subject: {train_nsub}")
    
    if Train:
        # Train data and label load as below
        load_data = lambda subject, type: (
            Load_BCI2a_data(root_path, subject, type=type)
            if dataset == "BCI2a"
            else Load_BCI2b_data(root_path, subject, type=type)
        )
        combine_data = lambda data_list: np.concatenate(data_list, axis=0)
        combine_labels = lambda label_list: np.concatenate(label_list, axis=0)
        normalize = lambda data: standardize_data_cross_subject(data)
        
        # Using List Comprehension for load, concate and normalize val, train data
        normalized_data_labels = [
            (
                normalize(combine_data([load_data(subject, type)[0] for type in ["T", "E"]])),
                combine_labels([load_data(subject, type)[1] for type in ["T", "E"]])
            )
            for subject in val_nsub
        ]
        normalized_data_list, normalized_label_list = zip(*normalized_data_labels)
        Val_datas = np.concatenate(normalized_data_list, axis=0)
        Val_labels = np.concatenate(normalized_label_list, axis=0)
        print(f"Val Set Shape: {np.shape(Val_labels),np.shape(Val_datas)}")

        normalized_data_labels = [
            (
                normalize(combine_data([load_data(subject, type)[0] for type in ["T", "E"]])),
                combine_labels([load_data(subject, type)[1] for type in ["T", "E"]])
            )
            for subject in train_nsub
        ]
        normalized_data_list, normalized_label_list = zip(*normalized_data_labels)
        Train_datas = np.concatenate(normalized_data_list, axis=0)
        Train_labels = np.concatenate(normalized_label_list, axis=0)
        print(f"Train Set Shape: {np.shape(Train_labels),np.shape(Train_datas)}")
        return Train_datas, Train_labels, Val_datas, Val_labels

    else:
        Test_datas, Test_labels = (np.concatenate(x, axis=0) 
                               for x in zip(*[
                                   Load_BCI2a_data(root_path,subject=test_nsub,type = type)
                                   if dataset == "BCI2a"
                                   else Load_BCI2b_data(root_path,subject=test_nsub,type = type)  
                                   for type in ["T","E"]]))
        Test_datas = standardize_data_cross_subject(Test_datas)
        print(f"Test Set Shape: {np.shape(Test_labels),np.shape(Test_datas)}")
        return Test_datas, Test_labels
