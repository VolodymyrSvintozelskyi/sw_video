# %%
import numpy as np
import os

# inputfolder = '..\\data\\06_02_2022-19_11_19_test_run_10minpixSTCE2_0v_2\\{}'
# inputfolder = '..\\data\\06_04_2022-21_51_26stce2_gradientcal5_AC0.5V_600sec\\{}'
inputfolder = '../data/06_14_2022-13_50_31stce3_pcb_15vRW_600SEC_3/{}'
# inputfolder = '../data/06_13_2022-12_02_16stce3_pcb_15vFWPULSE_600SEC/{}'


inputfolder = inputfolder[3:]
data = {}

for file in os.listdir(inputfolder.format('')):
    if file.endswith(".txt"):
        data[file[:-4]] = np.loadtxt(inputfolder.format(file)) # t V IOError
        # print(data[file[:-4]].shape)

# mult_file = np.loadtxt(inputfolder.format("gradient.txt")) # Non numeric lines should start with # !
mult_file_matrix = np.zeros((8,8))
# for line in range(mult_file.shape[0]):
#     mult_file_matrix[int(mult_file[line,0]),int(mult_file[line,1])] = mult_file[line,2]
# print ( mult_file.shape)

from PIL import Image
arr = Image.open('data/dog.png').convert('RGB')    
arr = 255-np.array(arr)
# print(arr.shape)
mult_file_matrix = arr[:,:,0]
mult_file_matrix = mult_file_matrix / np.max(mult_file_matrix)
# print(arr[:,:,0])
# print(arr[:,:,1])
# print(arr[:,:,2])

sig_ind = 1

print("{} files loaded.".format(len(data)))

# %%
def mov_av(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

import matplotlib.pyplot as plt
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
# maxcurrent = [np.max(val[:,2]) for val in sm_data.values()]
# mincurrent = [np.min(val[:,2]) for val in sm_data.values()]
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
size = 8*30, 8*30

duration = 10
fps = 120

out = cv2.VideoWriter('output/video_{}.mp4'.format(inputfolder[8:-3]), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)

frames = []
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
            frame[7-g_i, r_i] = (curr-mincurrent)/(maxcurrent-mincurrent) * mult_file_matrix[7-g_i, r_i]
    frame = np.repeat(np.repeat(frame,30,0),30,1)
    frames.append(frame)

for frame in tqdm.tqdm(frames):
    nframe = np.round((frame - np.min(frames))/(np.max(frames) - np.min(frames)) * 255)
    assert np.all(nframe >= 0) and np.all(nframe <= 255), "Ooops"
    out.write(nframe.astype('uint8'))
out.release()


