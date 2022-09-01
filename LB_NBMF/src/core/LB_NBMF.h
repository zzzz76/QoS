/********************************************************
 * LB_NBMF.h: header file of LB_NBMF.cpp
 * Author: zzzz76
 * Created: 2022/3/6
********************************************************/
#include "UIPCC.h"


/* Perform the core approach of LB_NBMF */
void LB_NBMF(double *removedData, double *removedDataT, double *predData, int numUser, int numService,
             int dim, double lmda, int maxIter, double etaInit, double alpha, double beta, double topU, double topS,
             double *bu, double *bs, double *Udata, double *Sdata,
             double *userRegion, double *serviceRegion, double *lossData, bool debugMode);

/* Compute the loss value of LB_NBMF */
double loss(double *bu, double *bs, double **U, double **S,
            double **removedMatrix, double **predMatrix,
            double lmda, int numUser, int numService, int dim, double beta,
            vector<vector<pair<int, double> > > &pccUserMatrix,
            vector<vector<pair<int, double> > > &pccServiceMatrix);

/* Compute the gradients of the loss function */
void gradLoss(double *bu, double *bs, double **U, double **S,
              double **removedMatrix, double **predMatrix,
              double *gradbu, double *gradbs, double **gradU, double **gradS,
              double lmda, int numUser, int numService, int dim, double alpha, double beta,
              vector<vector<pair<int, double> > > &pccUserMatrix,
              vector<vector<pair<int, double> > > &pccServiceMatrix);

/* Perform line search to find the best learning rate */
double linesearch(double **meanMatrix, double *bu, double *bs, double **U, double **S,
                  double **removedMatrix, double lastLossValue,
                  double *gradbu, double *gradbs, double **gradU, double **gradS,
                  double etaInit, double lmda, int numUser, int numService, int dim, double alpha, double beta,
                  vector<vector<pair<int, double> > > &pccUserMatrix,
                  vector<vector<pair<int, double> > > &pccServiceMatrix);

/* Compute predMatrix */
void predict(bool flag, double **meanMatrix, double *bu, double *bs, double **U, double **S,
             double **removedMatrix, double **predMatrix, int numUser, int numService, int dim, double alpha);


/* Compute the dot product of two vectors */
double dotProduct(double *vec1, double *vec2, int len);


/* Free memory for a 2D array */
void delete2DMatrix(double **ptr);

/* Compute the mean value between locals */
void localMean(double **removedMatrix, int numUser, int numService,
               double *userRegion, double *serviceRegion, double **meanMatrix);


