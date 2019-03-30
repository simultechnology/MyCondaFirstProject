
print('start')

import pandas as pd             # tabular data management, processing
import numpy as np              # computing module
import matplotlib.pyplot as plt # data visualization
import seaborn as sns           # data visualization
import os                       # os to load file
from IPython.display import Audio  # play the audio
import librosa                  # Audio management
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import StandardScaler
import pickle

# Those below are helper functions
def load_data_table():
    file_names = os.listdir("./train0")
    file_names_code = [x.split(".")[0] for x in file_names]
    label_all = pd.read_csv("train_class.txt", sep="\t", header = None)
    label_all.columns = ["code","classes"]
    label = label_all[label_all["code"].isin(file_names_code)].copy()
    classes = label["classes"]
    label["gender_classes"] = [x.split("_")[0] for x in classes]
    label.index = "./train0/" + label["code"] + ".wav"
    return label

def pick_random_file_name():
    file_names = os.listdir("./train0")
    return "./train0/" + str(np.random.choice(file_names))

def load_data_pickle():
    with open('data.pickle', 'rb') as f:
        data_full = pickle.load(f)
    with open('sr.pickle', 'rb') as f:
        sr = pickle.load(f)
    return data_full, sr

# table = load_data_table()
# print(table.head())
#
#random_files = pick_random_file_name()
random_files = './sample_data/2019-03-17T14_30_44.622Z.wav'

print(random_files)
data, sr = librosa.load(random_files)
Audio(data, rate = sr)

extractions = pd.DataFrame(columns=['spec_cent',
                           'zcr',
                           'chroma_stft',
                           'rolloff',
                           'spec_bw'
                                    ]
                                   + ["MFCC_"+str(x) for x in range (1,21)]
                           )

print(len(data))
middle = len(data) // 2
data = data[middle - sr // 2:middle + sr // 2]

# Label of the file FE or MA
# label = table.loc[file_path, "gender_classes"]

#### Spectral Centroid extraction ######
spec_cent = np.mean(librosa.feature.spectral_centroid(y=data, sr=sr))

##### Zero Crossing Rate extraction ######
zcr = np.mean(librosa.feature.zero_crossing_rate(y=data))

##### Chroma Frequencies extraction ######
chroma_stft = np.mean(librosa.feature.chroma_stft(y=data, sr=sr))

##### Spectral Roll-off extraction ######
rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr))

##### Spectral Bandwidth extraction ######
spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sr))

##### Mel-frequency cepstral coefficients (MFCC) extraction ######
mfcc = [np.mean(x) for x in librosa.feature.mfcc(y=data, sr=sr)]

extractions = [spec_cent, zcr, chroma_stft, rolloff, spec_bw] + mfcc

print(extractions)
# df.loc[file_path] = [label, spec_cent, zcr, chroma_stft, rolloff, spec_bw]

#scaler = StandardScaler() # initialise the scaler
scaler = pickle.load(open('knn_voice_detect_scaler.pkl', 'rb'))
#test_data = scaler.fit_transform([df])
test_data2 = scaler.transform([extractions])
print(test_data2)

filename = 'knn_voice_detect_model.pkl'

loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.predict(np.array([[spec_cent, zcr, chroma_stft, rolloff, spec_bw] + mfcc]))

result = loaded_model.predict(test_data2)

map_to_labels = {0: 'MA', 1: 'FE'}

print(result)
print('result :',map_to_labels[result[0]])