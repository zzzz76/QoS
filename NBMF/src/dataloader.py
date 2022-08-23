########################################################
# dataloader.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/10/12
########################################################

import numpy as np
from utilities import *
import xlrd


########################################################
# Function to load the dataset
#
def load(para):
    datafile = para['dataPath'] + para['dataType'] + 'Matrix.txt'
    logger.info('Load data: %s' % datafile)
    dataMatrix = np.loadtxt(datafile)
    dataMatrix = preprocess(dataMatrix, para)
    return dataMatrix

def loadUserList(para):
    datafile = para['dataPath'] + 'userlist.xlsx'
    logger.info('Load user list: %s' % datafile)

    book = xlrd.open_workbook(datafile)
    sheet = book.sheet_by_index(0)
    regions = sheet.col_values(2, 1)
    return np.array(regions)

def loadServiceList(para):
    datafile = para['dataPath'] + 'wslist.xlsx'
    logger.info('Load service list: %s' % datafile)

    book = xlrd.open_workbook(datafile)
    sheet = book.sheet_by_index(0)
    regions = sheet.col_values(2, 1)
    return np.array(regions)
########################################################


########################################################
# Function to preprocess the dataset
# delete the invalid values
# 
def preprocess(matrix, para):
    if para['dataType'] == 'rt':
        matrix = np.where(matrix == 0, -1, matrix)
        matrix = np.where(matrix >= 20, -1, matrix)
    elif para['dataType'] == 'tp':
        matrix = np.where(matrix == 0, -1, matrix)
    return matrix
########################################################
