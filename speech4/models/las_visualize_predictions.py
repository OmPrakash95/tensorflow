import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser(description='Tool to plot the encoder predictions emitted by our LAS model.')
parser.add_argument('--logdir', type=str)

args = parser.parse_args()


features_fbank = np.transpose(np.load(os.path.join(args.logdir, "features_fbank.npy")))
encoder_predictions = np.transpose(np.load(os.path.join(args.logdir, "encoder_predictions.npy")))

frames_to_predict = features_fbank.shape[1] - encoder_predictions.shape[1]

print features_fbank.shape
print encoder_predictions.shape


fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.set_axis_off()

img = [features_fbank]
for offset in range(frames_to_predict):
  subimg = encoder_predictions[offset * 40:(offset + 1) * 40,:]
  lpad = [np.zeros([40, 1])] * (offset + 1)
  rpad = [np.zeros([40, 1])] * (frames_to_predict - offset - 1)
  subimg = np.hstack(lpad + [subimg] + rpad)
  img.append(subimg)

img = np.vstack(img)[:,:500]

ax.imshow(img, interpolation="none")

#gs = gridspec.GridSpec(frames_to_predict + 1, 1, wspace=0.0, hspace=0.0)
#
#ax1 = fig.add_subplot(gs[0])
#ax1.set_axis_off()
#plt.imshow(features_fbank)
#for f in range(frames_to_predict):
#  ax = fig.add_subplot(gs[f + 1])
#  ax.set_axis_off()
#  plt.imshow(encoder_predictions[f * 40:(f + 1)*40,:], aspect="equal")
#
#gs.tight_layout(fig)
fig.savefig("prediction_visualization.png")
