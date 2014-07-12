# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pylab as pl

data_dir = r'/nfs/h1/workingshop/huanglijie/autoroi/data'
data_file = os.path.join(data_dir, 'parameter_score.npz')
data = np.load(data_file)
cv_score = data['cv_score']
oob_score = data['oob_score']
ffa_dice = data['ffa_dice']
ofa_dice = data['ofa_dice']

# row: n_tree, colume: depth
fig = pl.figure(figsize=(10, 10))
ntree_idx = np.arange(0, 70, 10)
depth_idx = np.arange(0, 70, 10)
# cv_score
pl.subplot(2, 2, 1)
pl.pcolor(ntree_idx, depth_idx, cv_score)
pl.colorbar()
pl.xlabel('depth')
pl.ylabel('# of trees')
pl.title('CV score')
# oob_score
pl.subplot(2, 2, 2)
pl.pcolor(ntree_idx, depth_idx, oob_score)
pl.colorbar()
pl.xlabel('depth')
pl.ylabel('# of trees')
pl.title('OOB score')
# ffa_dice
pl.subplot(2, 2, 3)
pl.pcolor(ntree_idx, depth_idx, ffa_dice)
pl.colorbar()
pl.xlabel('depth')
pl.ylabel('# of trees')
pl.title('FFA Dice')
# ffa_dice
pl.subplot(2, 2, 4)
pl.pcolor(ntree_idx, depth_idx, ofa_dice)
pl.colorbar()
pl.xlabel('depth')
pl.ylabel('# of trees')
pl.title('OFA Dice')

pl.show()

