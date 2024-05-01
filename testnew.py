import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import heartpy as hp
# Load ECG and PPG data from files 
df = pd.read_csv("./parthiv.csv")
print(df.columns)
df.rename(columns={'Unnamed: 3': 'spo2'}, inplace=True)
print(df.columns)
ppg=pd.read_csv("./parthiv.csv", usecols=["ppg"])
ecg=(pd.read_csv("./parthiv.csv", usecols=["ecg"]))
ECG = ecg['ecg'].values
PPG = ppg['ppg'].values

ecg = ECG[20000:21000] 
ppg = PPG[20000:21000] 
# ecg = hp.filter_signal(eecg, cutoff = 5, sample_rate = 100.0, order = 3, filtertype='highpass')
print(ecg)
ln = len(ecg) 
l = len(ppg) 

# Plot the ECG and PPG waveforms 
plt.plot(ecg, label='ECG')
plt.plot(ppg, label='PPG') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 

# ECG PEAK
m = 0.6 * (max(ecg))
ECGX = np.zeros(ln) 
for n in range(1, ln-1): 
    if ecg[n] >= m and ecg[n] > ecg[n+1] and ecg[n] > ecg[n-1]: 
        ECGX[n] = n 

ECGX = [i for i in ECGX if i != 0] 
RX = ECGX[1:] 
length = len(RX)  # PPGPEAK 
t = 0.8 * (max(ppg)) 
PPGX = np.zeros(l) 
for k in range(0, l): 
    if ppg[k] >= t: 
        if ppg[k] > ppg[k+1] and ppg[k] >= ppg[k-1]: 
            PPGX[k] = k 

PPGX = [j for j in PPGX if j != 0] 
PX = PPGX[:-1] 
len = len(PX) 

# PAT 
PAT = np.zeros(len) 
for p in range(length):
    try:
        PAT[p] = RX[p] - PX[p]
    except IndexError as e:
        print("Index out of bounds at index:", p)


# pat = np.mean(PAT) 
DBP = 54.53 + (0.77 * PAT) 
SBP = 149.8 - (0.765 * PAT) 

print("DBP=", DBP) 
print("SBP=", SBP)
