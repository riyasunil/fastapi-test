import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import heartpy as hp
df = pd.read_csv("./bpv4.csv")
# print("Column Names Before Renaming:")
# print(df.columns)


ppg=pd.read_csv("./bpv4.csv", usecols=["ppg"])
ecg=(pd.read_csv("./bpv4.csv", usecols=["ecg"]))
ecg_values_init = ecg['ecg'].values
ppg_values_init = ppg['ppg'].values

timer = hp.get_data('bpv4.csv', delim=",", column_name='TimeStamp')
sample_rate = hp.get_samplerate_datetime(timer, timeformat='%H:%M:%S')
print("sample_rate = ", sample_rate)

# reference for filtering : 
# https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/heartpy.filtering.html
ecg_values = hp.filtering.filter_signal(ecg_values_init, cutoff = 5, sample_rate = sample_rate, order = 3, filtertype='highpass')
ppg_values = hp.filtering.filter_signal(ppg_values_init, cutoff = 5, sample_rate = sample_rate, order = 3, filtertype='lowpass')
m = np.max(ecg_values)
th = 0.4 * m  # setting threshold
a = 1
r = []
ri = []

print(type(ecg_values))
print(len(ecg_values))

# Detect R-peaks and their time indices
for i in range(1, len(ecg_values) - 1):
    if ecg_values[i] > th and ecg_values[i] > ecg_values[i + 1] and ecg_values[i] > ecg_values[i - 1]:
        r.append(ecg_values[i])
        ri.append(i)

print("r = ", r)
print("ri = ", ri)
# Calculate RR intervals
rr = []
f = len(ri)
for j in range(f - 1):
    rr.append(ri[j + 1] - ri[j])
print("rr=", rr)

# Calculate heart rate
sum1 = 0
t = len(rr)
for k in range(t):
    sum1 += rr[k]
avg = sum1 / t
hr1 = 60 / avg
hr = hr1 * 125
print('heartrate =', hr)




t1 = max(ppg_values)
th1 = 0.5 * t1
sys_peak_i = []
sys_peak = []
j = 0

for j in range(len(ppg_values)):  # systolic peak detection
    if ppg_values[j] > th1 and ppg_values[j] > ppg_values[j + 1] and ppg_values[j] > ppg_values[j - 1]:
        sys_peak.append(ppg_values[j])
        sys_peak_i.append(j)

print('ppg peaks =', sys_peak)
print('Time index of systolic peak =', sys_peak_i)

# ppg diastolic peak detection
dias_peak_i = []
dias_peak = []

for k in range(len(sys_peak_i) - 1):
    for i3 in range(sys_peak_i[k] + 1, sys_peak_i[k + 1] - 1):
        if ppg_values[i3] > ppg_values[i3 + 1] and ppg_values[i3] > ppg_values[i3 - 1]:
            dias_peak.append(ppg_values[i3])

print('dbp =', dias_peak_i)

# ptt
s = []
v = len(sys_peak_i) - len(ri)
g = 0

for g in range(v):
    ri.append(0)

for l in range(len(sys_peak_i)):
    if ri[l] != 0:
        s.append(sys_peak_i[l] - ri[l])  # difference between the time index of R peak and systolic peak
    else:
        break

print('ptt =', s)

# matrix
sp = sys_peak #txt sp
dp = dias_peak #txt dp
g5 = len(s)
g5 = len(s)
print(g5)
hr2 = ecg_values # txt hr 
print('len of hr =', len(hr2))
w6 = g5 - len(sp)
e1 = 0

if isinstance(sp, list):
    sp = np.array(sp)
if isinstance(dp, list):
    dp = np.array(dp)

result_matrix = np.column_stack((sp, dp))
print(result_matrix)
print(sp.shape)

print(dp.shape)
if len(sp) != len(dp):
    max_length = max(len(sp), len(dp))
    sp += [0] * (max_length - len(sp))  # Append zeros to sp to match the length of dp
    dp += [0] * (max_length - len(dp))



if len(hr2) > len(s):
    d1 = len(hr2) - len(s)
    for e1 in range(w6):
        sp = np.append(sp, 0)
        dp = np.append(dp, 0)

    print('sbp list =', sp)
    print('dbp list =', dp)
    a1 = np.array([sp, dp])
    b1 = np.asmatrix(a1)
    c5 = np.transpose(b1)  # SBP and DBP matrix
    print('matrix Y =', c5)
    print(len(sp))

    for m6 in range(d1):
        s.append(0)

    i5 = np.ones(len(hr2)) * 1
    sp = np.append(sp, 0)
    dp = np.append(dp, 0)
    h5 = np.array([s, hr2, i5])
    m5 = np.asmatrix(h5)
    n5 = np.transpose(m5)
    print('matrix x =', n5)
    print('sbp list =', sp)
    print('dbp list =', dp)
    a1 = np.array([sp, dp])
    b1 = np.asmatrix(a1)
    c5 = np.transpose(b1)
else:
    d2 = len(s) - len(hr2)

    for e1 in range(w6):
        sp = np.append(sp, 0)
        dp = np.append(dp, 0)

    print('sbp list =', sp)
    print('dbp list =', dp)
    a1 = np.array([sp, dp])
    b1 = np.asmatrix(a1)
    c5 = np.transpose(b1)
    print(len(sp))

    for m6 in range(d2):
        s.append(0)

    i5 = np.ones(len(hr2)) * 1
    sp = np.append(sp, 0)
    dp = np.append(dp, 0)
    h5 = np.array([s, hr2, i5])
    m5 = np.asmatrix(h5)
    n5 = np.transpose(m5)
    print('matrix x =', n5)
    a1 = np.array([sp, dp])
    b1 = np.asmatrix(a1)
    c5 = np.transpose(b1)

# theta
q5 = np.conjugate(n5)
p5 = m5.dot(n5)
print('xT*x =', p5)
r5 = np.linalg.inv(p5)
print('inverse =', r5)
s5 = q5.dot(r5)
print('inverse * conj =', s5)
u5 = np.transpose(s5)
t5 = u5.dot(c5)
print('theta =', t5)

# sbp and dbp
x5 = np.mean(s)
v5 = t5[0, 0]
y5 = t5[0, 1]
w5 = t5[1, 0]
z5 = t5[1, 1]
a6 = t5[2, 0]
b6 = t5[2, 1]

print('a1 =', v5)
print('a2 =', y5)
print('b1 =', w5)
print('b2 =', z5)
print('c1 =', a6)
print('c2 =', b6)

c6 = v5 * x5
d6 = w5 * hr2
sbp = c6 + d6 + a6
print('sbp =', sbp)  # List of SBP values
plt.figure(2)
plt.xlabel('data number')
plt.ylabel('sbp(mmhg)')
title('sbp')
plt.plot(sbp)
plt.show()  # plot of SBP

mn = np.mean(sbp)  # mean of SBP
print('mn =', mn)

e6 = y5 * x5
f6 = z5 * hr2
dbp = e6 + f6 + b6
print('dbp =', dbp)  # List of DBP values
plt.figure(2)
plt.xlabel('data number')
plt.ylabel('dbp(mmhg)')
title('dbp')
plt.plot(dbp)
plt.plot(x, y)
plt.plot(y, x)
plt.show()  # plot of DBP

nm = np.mean(dbp)  # mean of DBP
print('nm =', nm)