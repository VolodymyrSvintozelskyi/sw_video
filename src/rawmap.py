# %%
import numpy as np
import os

# inputfolder = '..\\data\\06_02_2022-19_11_19_test_run_10minpixSTCE2_0v_2\\{}'
# inputfolder = '..\\data\\06_04_2022-21_51_26stce2_gradientcal5_AC0.5V_600sec\\{}'
inputfolder = '../data/06_14_2022-13_50_31stce3_pcb_15vRW_600SEC_3/{}'
# inputfolder = '../data/06_13_2022-12_02_16stce3_pcb_15vFWPULSE_600SEC/{}'

# time px py v i
# data = np.zeros((1,8,8,1,1))
inputfolder = inputfolder[3:]

data = {}
print(os.listdir("."))
for file in os.listdir(inputfolder.format('')):
    if file.endswith(".txt"):
        data[file[:-4]] = np.loadtxt(inputfolder.format(file)) # t V IOError
        # print(data[file[:-4]].shape)
        
sig_ind = 1

print("{} files loaded.".format(len(data)))

# %%
def mov_av(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

mov_av_w = 15

# %%
sm_data = {}
for k,v in data.items():
    newI = mov_av(v[:,2],mov_av_w)
    newV = mov_av(v[:,1],mov_av_w)
    newT = v[:-mov_av_w+1,0]
    sm_data[k] = np.vstack([newT,newV,newI]).T

# %%
mintime = np.min([np.max(val[:,0]) for val in sm_data.values()])
minpoints = np.min([val.shape[0] for val in sm_data.values()])

absmaxcurrent = ([np.max(val[:,sig_ind]) for val in sm_data.values()])
absmincurrent = ([np.min(val[:,sig_ind]) for val in sm_data.values()])
absmaxframe = np.argmax(absmaxcurrent)
absminframe = np.argmax(absmincurrent)
absmaxcurrent = np.max(absmaxcurrent)
absmincurrent = np.min(absmincurrent)
print("ABS MIN:",absmincurrent, "MAX:",absmaxcurrent)
print("Frame: ", absminframe, absmaxframe)
# absmaxcurrent = np.partition(absmaxcurrent, -2)[-2]
# absmincurrent = np.partition(absmincurrent, -10)[-10]

# print(*absmaxcurrent, "===",*absmincurrent, sep="\n")
print("Min pixel total time:",mintime)
print("Min pixel points:",minpoints)
fps = round(minpoints/mintime)
print("FPS:",fps)
print("Timestep:",1/fps)
# print("Max current:",maxcurrent)
# print("Min current:",mincurrent)

# %%
import cv2, tqdm
from PIL import Image
import matplotlib.pyplot as plt
size = 8*30, 8*30

fps = 120
duration = 10
fig,ax = plt.subplots(1)
frames = []
imframes = []
import time as timeTTT
for frame_ind in tqdm.tqdm(range(fps * duration)):
    time = mintime * frame_ind / fps / duration
    frame = np.zeros((8,8))

    for g_i in range(7,-1,-1):
        for r_i in range(0,8,1):
            frame_data = sm_data['R{}G{}'.format(r_i, g_i)]
            maxcurrent = np.average(frame_data[:20,sig_ind])
            mincurrent = 1#np.min(frame_data[:,2])
            time_ind = np.argmin( np.abs(frame_data[:,0] - time) )
            curr = frame_data[time_ind,sig_ind]
            frame[7-g_i, r_i] = (curr-mincurrent)/(maxcurrent-mincurrent)
    frames.append(frame)
from matplotlib import colors
for frame in tqdm.tqdm(frames):
    imframes.append([plt.imshow(frame, animated=True, vmin= np.min(frames),vmax=np.max(frames))])
    # imframes.append([plt.imshow(frame, animated=True, norm=colors.LogNorm(vmin= np.min(frames),vmax=np.max(frames)))])

plt.colorbar(imframes[-1][0])

import matplotlib.animation as anim
ani = anim.ArtistAnimation(fig, imframes, interval=1/fps*1000, blit=True,
                                repeat_delay=1000)
# ani.save('output/raw_{}.mp4'.format(inputfolder[5:-3]), dpi=500)
ani.save('output/raw_{}.mp4'.format(inputfolder[5:-3]))
# plt.show()


