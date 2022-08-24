########################################################
# core.pyx
# Author: zzzz76
# Created: 2022/3/6
########################################################

import time
import numpy as np
from utilities import *
cimport numpy as np  # import C-API
from libcpp cimport bool

#########################################################
# Make declarations on functions from cpp file
#
cdef extern from "LB_NBMF.h":
    void getLocSim_core(double *geoData, double *locSimData,
        int numLine, double theta)
    void LB_NBMF(double *locSimData, double *removedData, double *predData, int numUser, int numService,
          int dim, double lmda, int maxIter, double etaInit, double alpha, double beta,
          double *bu, double *bs, double *Udata, double *Sdata,
          double *userRegion, double *serviceRegion, double *lossData, bool debugMode)
#########################################################


#########################################################
# Function to perform the prediction algorithm
# Wrap up the C++ implementation
#
def predict(removedMatrix, userRegion, serviceRegion, locSim, para):
    cdef int numService = removedMatrix.shape[1]
    cdef int numUser = removedMatrix.shape[0]
    cdef int dim = para['dimension']
    cdef double lmda = para['lambda']
    cdef int maxIter = para['maxIter']
    cdef double etaInit = para['etaInit']
    cdef double alpha = para['alpha']
    cdef double beta = para['beta']
    cdef bool debugMode = para['debugMode']

    # initialization
    cdef np.ndarray[double, ndim=2, mode='c'] predMatrix = np.zeros((numUser, numService))
    cdef np.ndarray[double, ndim=2, mode='c'] U = np.random.rand(numUser, dim)
    cdef np.ndarray[double, ndim=2, mode='c'] S = np.random.rand(numService, dim)
    cdef np.ndarray[double, ndim=1, mode='c'] bu = np.random.rand(numUser)
    cdef np.ndarray[double, ndim=1, mode='c'] bs = np.random.rand(numService)
    cdef np.ndarray[double, ndim=1, mode='c'] loss = np.zeros(maxIter)

    logger.info('Iterating...')

    # Wrap up PMF.cpp
    LB_NBMF(
        <double *> (<np.ndarray[double, ndim=2, mode='c']> locSim).data,
        <double *> (<np.ndarray[double, ndim=2, mode='c']> removedMatrix).data,
        <double *> predMatrix.data,
        numUser,
        numService,
        dim,
        lmda,
        maxIter,
        etaInit,
        alpha,
        beta,
        <double *> bu.data,
        <double *> bs.data,
        <double *> U.data,
        <double *> S.data,
        <double *> (<np.ndarray[double, ndim=1, mode='c']> userRegion).data,
        <double *> (<np.ndarray[double, ndim=1, mode='c']> serviceRegion).data,
        <double *> loss.data,
        debugMode
    )

    return predMatrix, loss
#########################################################


#########################################################
# Function to compute the location similarity
#
def getLocSim(para):
    logger.info('Computing location similarity...')
    userLocFile = para['dataPath'] + 'userlist.txt'
    data = np.genfromtxt(userLocFile, comments='$', delimiter='\t',
        skip_header=2)
    geoAxis = data[:, 5:7] # columns of geo axis
    geoAxis[np.isnan(geoAxis)] = 999999 #unavailable geo data
    geoAxis = geoAxis.copy() # keep memory consist into C++

    cdef double theta = para['theta']
    cdef int numLine = geoAxis.shape[0]
    cdef np.ndarray[double, ndim=2, mode='c'] locSim \
        = np.zeros((numLine, numLine), dtype=np.float64)

    getLocSim_core(
        <double *> (<np.ndarray[double, ndim=2, mode='c']> geoAxis).data,
        <double *> locSim.data,
        numLine,
        theta
        )

    return locSim
#########################################################