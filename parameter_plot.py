# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

src_dir = r'/nfs/h1/workingshop/huanglijie/autoroi'
data_dir = os.path.join(src_dir, 'data', 'cv')

data_file = os.path.join(data_dir, 'parameter_cv_data.npz')
data = np.load(data_file)
cv_score = data['cv_score']
oob_score = data['oob_score']
ffa_dice = data['ffa_dice']
ofa_dice = data['ofa_dice']

##-- 2d matrix
## row: n_tree, colume: depth
#fig = pl.figure(figsize=(10, 10))
#ntree_idx = np.arange(0, 70, 5)
#depth_idx = np.arange(0, 70, 5)
## cv_score
#pl.subplot(2, 2, 1)
#pl.pcolor(ntree_idx, depth_idx, np.mean(cv_score, axis=2))
#pl.colorbar()
#pl.xlabel('depth')
#pl.ylabel('# of trees')
#pl.title('CV score')
## oob_score
#pl.subplot(2, 2, 2)
#pl.pcolor(ntree_idx, depth_idx, np.mean(oob_score, axis=2))
#pl.colorbar()
#pl.xlabel('depth')
#pl.ylabel('# of trees')
#pl.title('OOB score')
## ffa_dice
#pl.subplot(2, 2, 3)
#pl.pcolor(ntree_idx, depth_idx, np.mean(ffa_dice, axis=2))
#pl.colorbar()
#pl.xlabel('depth')
#pl.ylabel('# of trees')
#pl.title('FFA Dice')
## ffa_dice
#pl.subplot(2, 2, 4)
#pl.pcolor(ntree_idx, depth_idx, np.mean(ofa_dice, axis=2))
#pl.colorbar()
#pl.xlabel('depth')
#pl.ylabel('# of trees')
#pl.title('OFA Dice')
#pl.show()

#-- 3d surface
# row: n_tree, colume: depth
fig = pl.figure()
ax = fig.gca(projection='3d')
ntree_idx, depth_idx = np.mgrid[10:70:10, 10:70:10]
# cv_score
surf = ax.plot_surface(ntree_idx, depth_idx, np.mean(cv_score, axis=2),
                       cmap=cm.coolwarm, rstride=1, cstride=1)
ax.set_xlabel('depth')
ax.set_ylabel('# of trees')
ax.set_zlabel('CV  score')
fig.colorbar(surf, shrink=0.5, aspect=5)
## oob_score
#surf = ax.plot_surface(ntree_idx, depth_idx, np.mean(oob_score, axis=2),
#                       cmap=cm.coolwarm, rstride=1, cstride=1)
#ax.set_xlabel('depth')
#ax.set_ylabel('# of trees')
#ax.set_zlabel('OOB  score')
#fig.colorbar(surf, shrink=0.5, aspect=5)
## ffa_dice
#surf = ax.plot_surface(ntree_idx, depth_idx, np.mean(ffa_dice, axis=2),
#                       cmap=cm.coolwarm, rstride=1, cstride=1)
#ax.set_xlabel('depth')
#ax.set_ylabel('# of trees')
#ax.set_zlabel('FFA Dice')
#fig.colorbar(surf, shrink=0.5, aspect=5)
## ofa_dice
#surf = ax.plot_surface(ntree_idx, depth_idx, np.mean(ofa_dice, axis=2),
#                       cmap=cm.coolwarm, rstride=1, cstride=1)
#ax.set_xlabel('depth')
#ax.set_ylabel('# of trees')
#ax.set_zlabel('OFA Dice')
#fig.colorbar(surf, shrink=0.5, aspect=5)
pl.show()

