import tifffile as tif
import  matplotlib.pyplot as plt
import numpy as np
import cv2 # install: pip3 install opencv-python

from tqdm.notebook import tqdm
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
csfont = {'fontname':'Times New Roman'}


    
def make_norm(ti):
    ti = (ti - np.mean(ti))/np.std(ti)
    ti[ti<2] = 0
    return ti

reference = np.asarray(tif.imread('First.tif'), dtype='float')
moving    =  np.asarray(tif.imread('Second.tif'), dtype='float')

shiftz_2 = []
errorz_2 = []
phasediffz_2 = []

## Here exists calculation of all the pairwise mean RMS error values
for reference_i in tqdm(reference):
    for moving_i in (moving):
        
        reference_i_small = make_norm(cv2.resize(reference_i, dsize=(500, 500)))
        moving_i_small = make_norm(cv2.resize(moving_i, dsize=(500, 500)))
        shifts, error, phasediff = phase_cross_correlation(reference_i_small,moving_i_small, upsample_factor=1)
        shiftz_2.append(shifts)
        errorz_2.append(error)
        phasediffz_2.append(phasediff)
        


# Below we see the code to normalize the mean RMS error
errors = np.asarray(errorz_2).reshape(reference.shape[0],moving.shape[0])
i_errors = (1/errors)
i_errors = i_errors - np.min(i_errors)
i_errors = i_errors /np.sum(i_errors)

#Check for correctness
plt.figure(figsize=(10,10))
plt.title('RMS Error in Image Pair \nConvolution in Fourier Space',fontsize=32, **csfont)
plt.imshow(np.asarray(errorz_2).reshape(reference.shape[0],moving.shape[0]),cmap='jet')


fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(10,5), dpi=300)

domain= np.arange(i_errors.shape[1])
for i in i_errors.T:
    axes.plot(domain, i,linewidth=1, c='#34ebba')
axes.set_facecolor('k')
axes.set_xlabel('Z-Slice in Moving Image',  fontsize=12,**csfont)
axes.set_ylabel('Inverse of the \nPhase Cross-Correlation RMS', fontsize=12, **csfont)
axes.set_title('Image Parity Likelihood Distribution', fontsize=24, weight='bold',**csfont)


fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(10,5), dpi=300)

domain= np.arange(i_errors.shape[0])
ranges = i_errors.T
empty_vals = np.zeros(141)
ones_list = np.ones(141)


mask = []
norm_mask = []
for i in np.arange(len(ranges)):
    auc = np.trapz(ranges[i], dx = 1)
    axes.plot(domain, ranges[i]/auc,linewidth=1, c='#34ebba')
    if np.argmax(ranges[i]) < 120:
        top_indicies = np.sort(np.argsort(ranges[i])[::-1][:7])
        axes.plot(domain[top_indicies], ranges[i][top_indicies]/(auc), c='red')
        mask_i = empty_vals.copy()
        mask_i[top_indicies] = 1
        mask_i = mask_i
        mask.append(mask_i)
    else:
        mask.append(empty_vals)
    norm_mask_i = ones_list/auc
    norm_mask.append(norm_mask_i)
mask = np.asarray(mask)
norm_mask = np.asarray(norm_mask)
axes.set_facecolor('k')
axes.set_xlabel('Z-Slice in Moving Image',  fontsize=12,**csfont)
axes.set_ylabel('Normalized Inverse of the \nPhase Cross-Correlation RMS', fontsize=12, **csfont)
axes.set_title('Image Parity Likelihood Distribution', fontsize=24, weight='bold',**csfont)


intermediate_field = i_errors*norm_mask.T*mask.T
intermediate_field = intermediate_field - np.min(intermediate_field)
intermediate_field = intermediate_field /np.sum(intermediate_field)


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
    
plt.figure(figsize=(10,10))
plt.title('Normalized Probability Distribution \nFunction of Optimal Overlap',fontsize=32, **csfont)
ax = plt.gca()
im = ax.imshow(intermediate_field,cmap='hot')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
   
plt.colorbar(im, cax=cax)


# Create a flat copy of the PDF
flat = intermediate_field.flatten()

points = []
for i in tqdm(np.arange(100000)):
    sample_index = np.random.choice(a=flat.size, p=flat)

    # Take this index and adjust it so it matches the original array
    adjusted_index = np.unravel_index(sample_index, i_errors.shape)
    points.append(list(adjusted_index))

points = np.asarray(points)

points = points.T[::-1].T

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(points.T[0].reshape(-1, 1), points.T[1])

print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
