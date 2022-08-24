/********************************************************
 * NBMF.h: header file of NBMF.cpp
 * Author: zzzz76
 * Created: 2022/3/6
********************************************************/


/* Perform the core approach of NBMF */
void NBMF(double *removedData, double *predData, int numUser, int numService, int dim,
          double lmda, int maxIter, double etaInit,
          double *bu, double *bs, double *Udata, double *Sdata,
          double *userRegion, double *serviceRegion, double *lossData, bool debugMode);

/* Compute the loss value of NBMF */
double loss(double *bu, double *bs, double **U, double **S,
            double **removedMatrix, double **predMatrix, double lmda, int numUser, int numService, int dim);

/* Compute the gradients of the loss function */
void gradLoss(double *bu, double *bs, double **U, double **S,
              double **removedMatrix, double **predMatrix,
              double *gradbu, double *gradbs, double **gradU, double **gradS,
              double lmda, int numUser, int numService, int dim);

/* Perform line search to find the best learning rate */
double linesearch(double miu, double *bu, double *bs, double **U, double **S,
                  double **removedMatrix, double lastLossValue,
                  double *gradbu, double *gradbs, double **gradU, double **gradS,
                  double etaInit, double lmda, int numUser, int numService, int dim);

/* Compute predMatrix */
void predict(bool flag, double miu, double *bu, double *bs, double **U, double **S,
             double **removedMatrix, double **predMatrix, int numUser, int numService, int dim);

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


