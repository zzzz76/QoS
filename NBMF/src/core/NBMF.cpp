/********************************************************
 * NBMF.cpp
 * C++ implements on NBMF
 * Author: zzzz76
 * Created: 2022/3/6
********************************************************/

#include <iostream>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include "NBMF.h"

using namespace std;

#define eps 1e-10

/********************************************************
* Udata and Sdata are the output values
 * 训练集
 * 预测矩阵
 * 用户数量
 * 服务数量
 * 维度
 * lambda
 * maxIter
 * etaInit
 * 初始化的用户向量
 * 初始化的服务向量
 * 初始化的用户矩阵
 * 初始化的服务矩阵
********************************************************/
void NBMF(double *removedData, double *predData, int numUser, int numService,
          int dim, double lmda, int maxIter, double etaInit, double alpha,
          double *bu, double *bs, double *Udata, double *Sdata,
          double *userRegion, double *serviceRegion, double *lossData, bool debugMode) {
    // --- transfer the 1D pointer to 2D array pointer
    double **removedMatrix = vector2Matrix(removedData, numUser, numService);
    double **predMatrix = vector2Matrix(predData, numUser, numService);
    double **U = vector2Matrix(Udata, numUser, dim);
    double **S = vector2Matrix(Sdata, numService, dim);

    // --- create a set of temporal matries
    double **gradU = createMatrix(numUser, dim);
    double **gradS = createMatrix(numService, dim);
    double *gradbu = new double[numUser];
    double *gradbs = new double[numService];

    // -- local average
    double **meanMatrix = createMatrix(numUser, numService);
    localMean(removedMatrix, numUser, numService, userRegion, serviceRegion, meanMatrix);

    // --- iterate by standard gradient descent algorithm
    double lossValue;
    int iter, i, j, k;
    for (iter = 0; iter < maxIter; iter++) {
        // update predict
        predict(false, meanMatrix, bu, bs, U, S, removedMatrix, predMatrix, numUser, numService, dim, alpha);
        // update loss value
        lossValue = loss(bu, bs, U, S, removedMatrix, predMatrix, lmda, numUser, numService, dim);

        // update gradients
        gradLoss(bu, bs, U, S, removedMatrix, predMatrix,
                 gradbu, gradbs, gradU, gradS, lmda, numUser, numService, dim, alpha);

        // line search to find the best learning rate eta
        double eta = linesearch(meanMatrix, bu, bs, U, S, removedMatrix, lossValue,
                                gradbu, gradbs, gradU, gradS,
                                etaInit, lmda, numUser, numService, dim, alpha);

        // gradient descent updates
        for (k = 0; k < dim; k++) {
            // update U
            for (i = 0; i < numUser; i++) {
                U[i][k] -= eta * gradU[i][k];
            }
            // update S
            for (j = 0; j < numService; j++) {
                S[j][k] -= eta * gradS[j][k];
            }
        }
        // update bu
        for (i = 0; i < numUser; i++) {
            bu[i] -= eta * gradbu[i];
        }
        // update bs
        for (j = 0; j < numService; j++) {
            bs[j] -= eta * gradbs[j];
        }
        //cout << lossValue << endl;
        if (debugMode) {
            lossData[iter] = lossValue;
        }
    }

    predict(true, meanMatrix, bu, bs, U, S, removedMatrix, predMatrix, numUser, numService, dim, alpha);

    delete2DMatrix(gradU);
    delete2DMatrix(gradS);
    delete2DMatrix(meanMatrix);
    delete gradbu;
    delete gradbs;
    delete ((char *) U);
    delete ((char *) S);
    delete ((char *) removedMatrix);
    delete ((char *) predMatrix);
}

/* Compute the loss value of NBMF */
double loss(double *bu, double *bs, double **U, double **S,
            double **removedMatrix, double **predMatrix, double lmda, int numUser, int numService, int dim) {
    int i, j, k, g;
    double loss = 0;

    // cost
    for (i = 0; i < numUser; i++) {
        for (j = 0; j < numService; j++) {
            if (removedMatrix[i][j] > eps) {
                loss += 0.5 * (removedMatrix[i][j] - predMatrix[i][j])
                        * (removedMatrix[i][j] - predMatrix[i][j]);
            }
        }
    }

    // L2 regularization
    for (k = 0; k < dim; k++) {
        for (i = 0; i < numUser; i++) {
            loss += 0.5 * lmda * U[i][k] * U[i][k];
        }
        for (j = 0; j < numService; j++) {
            loss += 0.5 * lmda * S[j][k] * S[j][k];
        }
    }
    for (i = 0; i < numUser; i++) {
        loss += 0.5 * lmda * bu[i] * bu[i];
    }
    for (j = 0; j < numService; j++) {
        loss += 0.5 * lmda * bs[j] * bs[j];
    }

    return loss;
}


void gradLoss(double *bu, double *bs, double **U, double **S,
              double **removedMatrix, double **predMatrix,
              double *gradbu, double *gradbs, double **gradU, double **gradS,
              double lmda, int numUser, int numService, int dim, double alpha) {
    int i, j, k, g;
    double grad = 0;

    // gradU
    for (i = 0; i < numUser; i++) {
        for (k = 0; k < dim; k++) {
            grad = 0;
            for (j = 0; j < numService; j++) {
                if (removedMatrix[i][j] > eps) {
                    grad += (removedMatrix[i][j] - predMatrix[i][j])
                            * (-S[j][k]) * (1 - alpha);
                }
            }
            grad += lmda * U[i][k];
            gradU[i][k] = grad;
        }
    }

    // gradS
    for (j = 0; j < numService; j++) {
        for (k = 0; k < dim; k++) {
            grad = 0;
            for (i = 0; i < numUser; i++) {
                if (removedMatrix[i][j] > eps) {
                    grad += (removedMatrix[i][j] - predMatrix[i][j])
                            * (-U[i][k]) * (1 - alpha);
                }
            }
            grad += lmda * S[j][k];
            gradS[j][k] = grad;
        }
    }

    // gradbu
    for (i = 0; i < numUser; i++) {
        grad = 0;
        for (j = 0; j < numService; j++) {
            if (removedMatrix[i][j] > eps) {
                grad += (removedMatrix[i][j] - predMatrix[i][j]) * alpha;
            }
        }
        grad = -grad + lmda * bu[i];
        gradbu[i] = grad;
    }

    // gradbs
    for (j = 0; j < numService; j++) {
        grad = 0;
        for (i = 0; i < numUser; i++) {
            if (removedMatrix[i][j] > eps) {
                grad += (removedMatrix[i][j] - predMatrix[i][j]) * alpha;
            }
        }
        grad = -grad + lmda * bs[j];
        gradbs[j] = grad;
    }

}


double linesearch(double **meanMatrix, double *bu, double *bs, double **U, double **S,
                  double **removedMatrix, double lastLossValue,
                  double *gradbu, double *gradbs, double **gradU, double **gradS,
                  double etaInit, double lmda, int numUser, int numService, int dim, double alpha) {
    double eta = etaInit;
    double lossValue;
    double *bu1 = new double[numUser];
    double *bs1 = new double[numService];
    double **U1 = createMatrix(numUser, dim);
    double **S1 = createMatrix(numService, dim);
    double **predMatrix1 = createMatrix(numUser, numService);

    int iter, i, j, k;
    for (iter = 0; iter < 20; iter++) {
        // fake gradient descent updates
        for (k = 0; k < dim; k++) {
            // update U1
            for (i = 0; i < numUser; i++) {
                U1[i][k] = U[i][k] - eta * gradU[i][k];
            }
            // update S1
            for (j = 0; j < numService; j++) {
                S1[j][k] = S[j][k] - eta * gradS[j][k];
            }
        }
        // update bu
        for (i = 0; i < numUser; i++) {
            bu1[i] = bu[i] - eta * gradbu[i];
        }
        // update bs
        for (j = 0; j < numService; j++) {
            bs1[j] = bs[j] - eta * gradbs[j];
        }

        predict(false, meanMatrix, bu1, bs1, U1, S1, removedMatrix, predMatrix1, numUser, numService, dim, alpha);
        lossValue = loss(bu1, bs1, U1, S1, removedMatrix, predMatrix1, lmda, numUser, numService, dim);

        if (lossValue <= lastLossValue)
            break;
        eta = eta / 2;
    }

    delete bu1;
    delete bs1;
    delete2DMatrix(U1);
    delete2DMatrix(S1);
    delete2DMatrix(predMatrix1);

    return eta;
}


void predict(bool flag, double **meanMatrix, double *bu, double *bs, double **U, double **S,
             double **removedMatrix, double **predMatrix, int numUser, int numService, int dim, double alpha) {
    int i, j;
    for (i = 0; i < numUser; i++) {
        for (j = 0; j < numService; j++) {
            predMatrix[i][j] = 0;
            if (flag || removedMatrix[i][j] > eps) {
                predMatrix[i][j] += (1 - alpha) * dotProduct(U[i], S[j], dim)
                        + alpha * (meanMatrix[i][j] + bu[i] + bs[j]);
            }
        }
    }
}


double **vector2Matrix(double *vector, int row, int col) {
    double **matrix = new double *[row];
    if (!matrix) {
        cout << "Memory allocation failed in vector2Matrix." << endl;
        return NULL;
    }

    int i;
    for (i = 0; i < row; i++) {
        matrix[i] = vector + i * col;
    }
    return matrix;
}


double dotProduct(double *vec1, double *vec2, int len) {
    double product = 0;
    int i;
    for (i = 0; i < len; i++) {
        product += vec1[i] * vec2[i];
    }
    return product;
}


double **createMatrix(int row, int col) {
    double **matrix = new double *[row];
    matrix[0] = new double[row * col];
    memset(matrix[0], 0, row * col * sizeof(double)); // Initialization
    int i;
    for (i = 1; i < row; i++) {
        matrix[i] = matrix[i - 1] + col;
    }
    return matrix;
}


void delete2DMatrix(double **ptr) {
    delete ptr[0];
    delete ptr;
}

// the mean value between locals
void localMean(double **removedMatrix, int numUser, int numService,
               double *userRegion, double *serviceRegion, double **meanMatrix) {
    unordered_map<int, double> regionSum;
    unordered_map<int, int> regionCount;
    for (int i = 0; i < numUser; i++) {
        for (int j = 0; j < numService; j++) {
            if (removedMatrix[i][j] > eps) {
                int ur = (int) userRegion[i];
                int sr = (int) serviceRegion[j];
                int key = ur * 100 + sr;
                regionSum[key] += removedMatrix[i][j];
                regionCount[key]++;
            }
        }
    }

    int cnt = 0;
    double miu = 0;
    for (int i = 0; i < numUser; i++) {
        for (int j = 0; j < numService; j++) {
            if (removedMatrix[i][j] > eps) {
                miu += removedMatrix[i][j];
                cnt++;
            }
        }
    }
    miu /= cnt;

    for (int i = 0; i < numUser; i++) {
        for (int j = 0; j < numService; j++) {
            int ur = (int) userRegion[i];
            int sr = (int) serviceRegion[j];
            int key = ur * 100 + sr;
            if (regionCount[key] > 0) {
                meanMatrix[i][j] = regionSum[key] / regionCount[key];
            } else {
                meanMatrix[i][j] = miu;
            }
        }
    }
}


