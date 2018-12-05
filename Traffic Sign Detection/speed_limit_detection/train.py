##
## Simple Training Script
##

from model import Model

M = Model(mode = 'train')
#M.restore_from_checkpoint('tf_data\sample_model')
M.train()