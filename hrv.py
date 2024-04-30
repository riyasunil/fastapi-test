import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heartpy as hp
from datetime import datetime
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI

app = FastAPI()


# Write the DataFrame back to a CSV file
# =# --------------------------------------------------------------------------------------------------------------------
# Load ECG data from CSV
# show difference btw ecgsample r peaks and bpv4 r peaks

hrdata = hp.get_data('bpv4.csv', delim=',', column_name='ecg')
print(hrdata)

# sampling rate = 100
fs = 200.137457
working_data_hrv, measures_hrv = hp.process(
    hrdata, fs, report_time=True, calc_freq=True, high_precision=True, high_precision_fs=1000.0)
print(measures_hrv)  # returns RMSSD HRV measure

# --------------------------------------------------------------------------------------------------------------------


# def detect_peaks(ppg_signal, t):
#     # Calculate first derivative
#     first_derivative = np.diff(ppg_signal)

#     # Find turning points (where slope changes sign)
#     turning_points = np.where(np.diff(np.sign(first_derivative)))[0] + 1

#     systolic_peaks = []
#     diastolic_peaks = []

#     # Iterate through turning points
#     for i in turning_points:
#         # Check if turning point is a systolic or diastolic peak based on signal characteristics

#         if ppg_signal[i] > t:
#             systolic_peaks.append(i)
#         else:
#             diastolic_peaks.append(i)

#     return systolic_peaks, diastolic_peaks



# data = hp.get_data('bpv4.csv', delim=",", column_name='ppg')
# timer = hp.get_data('bpv4.csv', delim=",", column_name='TimeStamp')

# def detect_peaks(ppg_signal, threshold):
#     systolic_peaks = []
#     diastolic_peaks = []

#     # Iterate through the signal
#     for i in range(1, len(ppg_signal) - 1):
#         if ppg_signal[i] > threshold and ppg_signal[i] > ppg_signal[i-1] and ppg_signal[i] > ppg_signal[i+1]:
#             systolic_peaks.append(i)
#         elif ppg_signal[i] <= threshold and ppg_signal[i] < ppg_signal[i-1] and ppg_signal[i] < ppg_signal[i+1]:
#             diastolic_peaks.append(i)

#     return systolic_peaks, diastolic_peaks



# sample_rate = hp.get_samplerate_datetime(timer, timeformat='%H:%M:%S')

# print('sample rate is: %f Hz' % sample_rate)
# wd, m = hp.process(data, sample_rate, report_time=True)

# # Calculate threshold dynamically
# # max_amplitude = max(wd['filtered'])
# # for key in wd.keys():
# #     print(key)
# print("wdpeaklist", wd["peaklist"])
# threshold_value = 0.5   # Adjust fraction as needed

# # Call the peak detection function
# sys_peaks, dia_peaks = detect_peaks(data, threshold_value)


# print("Systolic peaks:", sys_peaks)
# print("Diastolic peaks:", dia_peaks)

# # display measures computed
# for x in m.keys():
#     print('%s: %f' % (x, m[x]))
