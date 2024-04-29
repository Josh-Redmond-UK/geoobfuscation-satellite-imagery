import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from src.utils import *
from models.diffusion.utils import *
import statsmodels.api as sm
import pickle
import os

results = pickle.load(open("/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240227-224206/results.pkl", "rb"))


baseline = []
blurred_wd = []
pixelate_wd = []
clouds_wd = []
repaint_wd = []

for idx, r in enumerate(results):
    pixelate_wd.append(r['pixelate'][idx])
    blurred_wd.append(r['blurred'][idx])
    clouds_wd.append(r['clouds'][idx])
    repaint_wd.append(r['repaint'][idx])
    baseline.append(0)


baseline_match = []
blurred_match = []
pixelate_match = []
clouds_match = []
repaint_match = []


for idx, r in enumerate(results):
    pixelate_match.append(np.argsort(r['pixelate'])[idx])
    blurred_match.append(np.argsort(r['blurred'])[idx])
    clouds_match.append(np.argsort(r['clouds'])[idx])
    repaint_match.append(np.argsort(r['repaint'])[idx])
    baseline_match.append(1)


import pandas as pd

frame = pd.DataFrame({'Blurred': blurred_wd, 'Pixelated': pixelate_wd, 'Anisotropic Noise': clouds_wd, 'Repainted': repaint_wd})#.to_csv("/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240220-102502/results.csv")
match_frame = pd.DataFrame({'Blurred': blurred_match, 'Pixelated': pixelate_match, 'Anisotropic Noise': clouds_match, 'Repainted': repaint_match})#.to_csv("/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240220-102502/results.csv")
print(frame.head())
print(match_frame.head())
plot = frame.boxplot(showmeans=True, meanline=True, meanprops={'color': 'red'})
plt.title("WD of Anonymised Images")
plt.ylabel("WD")
plt.savefig("results_boxplot.png")
plt.close()

plot = match_frame.boxplot(showmeans=True, meanline=True, meanprops={'color': 'red'})
plt.title("Match of Anonymised Images")
plt.ylabel("Match N")
plt.savefig("results_match_boxplot.png")
plt.close()




plot = frame.plot.density()
plt.title("WD of Anonymised Images")
plt.savefig("results_density.png")
plt.close()



plot = match_frame.plot.density()
plt.title("Matching N of Anonymised Images")
plt.savefig("results_match_density.png")
plt.close()

plt.subplots(2,2, figsize=(10, 10), sharex=True, sharey=True)
plt.subplot(2,2,1)
plt.hist(blurred_wd)
plt.title("Blurred")
plt.subplot(2,2,2)
plt.hist(pixelate_wd)
plt.title("Pixelated")
plt.subplot(2,2,3)
plt.hist(clouds_wd)
plt.title("Anisotropic Noise")
plt.subplot(2,2,4)
plt.hist(repaint_wd)
plt.title("Repainted")
plt.savefig("results_histogram.png")


plt.subplots(2,2, figsize=(10, 10), sharex=True, sharey=True)
plt.subplot(2,2,1)
plt.hist(blurred_match)
plt.title("Blurred")
plt.subplot(2,2,2)
plt.hist(pixelate_match)
plt.title("Pixelated")
plt.subplot(2,2,3)
plt.hist(clouds_match)
plt.title("Anisotropic Noise")
plt.subplot(2,2,4)
plt.hist(repaint_match)
plt.title("Repainted")
plt.savefig("results_match_histogram.png")



image_indices = [225, 161, 47, 24]
num_images = len(image_indices)
fig, ax = plt.subplots(num_images, 6, gridspec_kw = {'wspace':0, 'hspace':0}, sharex=True, sharey=True, figsize=(10, 10))
#fig.subplots_adjust(wspace=0, hspace=0)
#fig.set_tight_layout(True)
for idx, i in enumerate(image_indices):
    orig_image = plt.imread(f"/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240227-224206/original_{i}.png")
    poi_image = plt.imread(f"/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240227-224206/poi_{i}.png")
    blurred_image = plt.imread(f"/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240227-224206/blurred_{i}.png")
    pixelated_image = plt.imread(f"/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240227-224206/pixelated_{i}.png")
    clouded_image = plt.imread(f"/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240227-224206/clouded_{i}.png")
    repainted_image = plt.imread(f"/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240227-224206/repainted_{i}.png")
    ax[idx, 0].imshow(orig_image)
    ax[idx, 1].imshow(poi_image)
    ax[idx, 2].imshow(blurred_image)
    ax[idx, 3].imshow(pixelated_image)
    ax[idx, 4].imshow(clouded_image)
    ax[idx, 5].imshow(repainted_image)
    for a in ax[idx]:
        a.axis('off')

ax[0, 0].set_title("Original")
ax[0, 1].set_title("POI")
ax[0, 2].set_title("Blurred")
ax[0, 3].set_title("Pixelated")
ax[0, 4].set_title("Anisotropic Noise")
ax[0, 5].set_title("Repainted")
fig.suptitle("Comparison of Delocated Images")
plt.savefig("results_images.png")

frame = pd.DataFrame({'Blurred': blurred_wd, 'Pixelated': pixelate_wd, 'Anisotropic Noise': clouds_wd, 'Repainted': repaint_wd, 'Baseline':baseline})#.to_csv("/Users/joshredmond/Documents/GitHub/Diffusion-for-GeoDislocation/experiments/generated_images_20240220-102502/results.csv")


melted = frame.melt()
melted.columns = ['Method', 'WD']
import statsmodels.formula.api as smf

print(melted.head())
log_reg = smf.ols("WD ~ C(Method, Treatment(reference='Baseline'))", data=melted).fit()#sm.Logit(melted['WD'], melted['Method']).fit()
print(log_reg.summary())

with open('treatment_estimation.csv', 'w') as fh:
    fh.write(log_reg.summary().as_csv())
