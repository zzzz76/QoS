/********************************************************
 * LB_NBMF.h: header file of LB_NBMF.cpp
 * Author: zzzz76
 * Created: 2022/3/6
********************************************************/

void getLocSim_core(double *geoData, double *locSimData, int numLine, double theta);

/* Perform the core approach of LB_NBMF */
void LB_NBMF(double *locSimData, double *removedData, double *predData, int numUser, int numService,
          int dim, double lmda, int maxIter, double etaInit, double alpha, double beta,
          double *bu, double *bs, double *Udata, double *Sdata,
          double *userRegion, double *serviceRegion, double *lossData, bool debugMode);

/* Compute the loss value of LB_NBMF */
double loss(double *bu, double *bs, double **U, double **S,
            double **locSim, double **removedMatrix, double **predMatrix,
            double lmda, int numUser, int numService, int dim, double beta);

/* Compute the gradients of the loss function */
void gradLoss(double *bu, double *bs, double **U, double **S,
              double **locSim, double **removedMatrix, double **predMatrix,
              double *gradbu, double *gradbs, double **gradU, double **gradS,
              double lmda, int numUser, int numService, int dim, double alpha, double beta);

/* Perform line search to find the best learning rate */
double linesearch(double **meanMatrix, double *bu, double *bs, double **U, double **S,
                  double **locSim, double **removedMatrix, double lastLossValue,
                  double *gradbu, double *gradbs, double **gradU, double **gradS,
                  double etaInit, double lmda, int numUser, int numService, int dim, double alpha, double beta);

/* Compute predMatrix */
void predict(bool flag, double **meanMatrix, double *bu, double *bs, double **U, double **S,
             double **removedMatrix, double **predMatrix, int numUser, int numService, int dim, double alpha);

/* Transform a vector into a matrix */
double **vector2Matrix(double *vector, int row, int col);

/* Compute the dot product of two vectors */
double dotProduct(double *vec1, double *vec2, int len);

/* Allocate memory for a 2D array */
double **createMatrix(int row, int col);

/* Free memory for a 2D array */
void delete2DMatrix(double **ptr);

/* Compute the mean value between locals */
void localMean(double **removedMatrix, int numUser, int numService,
               double *userRegion, double *serviceRegion, double **meanMatrix);


