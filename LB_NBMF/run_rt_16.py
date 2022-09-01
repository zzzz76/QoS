########################################################
# run_rt.py 
# Author: zzzz76
# Created: 2022/3/6
# Implemented approach: NBMF
# Evaluation metrics: MAE, NMAE, RMSE, MRE, NPRE
########################################################

import numpy as np
import os, sys, time
import multiprocessing
import core

sys.path.append('src')
# Build external model
if not os.path.isfile('src/core.pyd'):
    print 'Lack of core.so (built from the C++ module).'
    print 'Please first build the C++ code into core.so by using: '
    print '>> python setup.py build_ext --inplace'
    sys.exit()
from utilities import *
import evaluator
import dataloader

#########################################################
# config area
#
para = {'dataType': 'rt',  # set the dataType as 'rt' or 'tp'
        'dataPath': '../data/dataset1/',
        'outPath': 'result/',
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NPRE',
                    ('NDCG', [1, 5, 10, 20, 50, 100])],  # delete where appropriate
        'density': list(np.arange(0.1, 0.11, 0.05)),  # matrix density
        'rounds': 8,  # how many runs are performed at each matrix density
        'dimension': 10,  # dimenisionality of the latent factors
        'etaInit': 0.01,  # inital learning rate. We use line search
        # to find the best eta at each iteration
        'lambda': 18,  # regularization parameter
        'beta': 15, # the parameter of location regularization
        'theta': 200, # the distance threshold to control the neighborhood size
        'maxIter': 300,  # the max iterations
        'alpha': 0.2,
        'topU': 5,
        'topS': 10,
        'saveTimeInfo': False,  # whether to keep track of the running time
        'saveLog': False,  # whether to save log into file
        'debugMode': False,  # whether to record the debug info
        'parallelMode': False  # whether to leverage multiprocessing for speedup
        }

initConfig(para)
#########################################################

def main():
    startTime = time.clock()  # start timing
    logger.info('==============================================')
    logger.info('LB_NBMF: Network biased Matrix Factorization.')

    # load the dataset
    dataMatrix = dataloader.load(para)
    logger.info('Loading data done.')

    # load user information and service information
    userRegions = dataloader.loadUserList(para)
    serviceRegions = dataloader.loadServiceList(para)

    # run for each density
    if para['parallelMode']:  # run on multiple processes
        pool = multiprocessing.Pool()
        for density in para['density']:
            pool.apply_async(evaluator.execute, (dataMatrix, density, userRegions, serviceRegions, para))
        pool.close()
        pool.join()
    else:  # run on single processes
        for density in para['density']:
            for it in [45, 50, 55]:
                para['beta'] = it
                evaluator.execute(dataMatrix, density, userRegions, serviceRegions, para)

    logger.info(time.strftime('All done. Total running time: %d-th day - %Hhour - %Mmin - %Ssec.',
                              time.gmtime(time.clock() - startTime)))
    logger.info('==============================================')
    sys.path.remove('src')

if __name__ == '__main__':
    main()
