from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('test2').getOrCreate()

#data_analysis
import pandas as pd
import numpy as np
from numpy import sqrt
import sys
from collections import Counter
import cloudpickle
import time
import pickle
import json
import re
import math
from math import*
import collections
from functools import reduce
from operator import add
from typing import Iterator,Tuple

#sql
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,when,lit
from delta.tables import *
from pyspark.sql import functions as f
from pyspark.sql.functions import *
from pyspark.sql import *
from pyspark.sql.types import IntegerType,FloatType,DoubleType
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import row_number, monotonically_increasing_id
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

#visualization
import matplotlib.pyplot as plt
import seaborn as sns

#preprocessing
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from scipy.stats import norm
#from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import RFE
import shap
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

#model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score 
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

#smote
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

#feature_store
from databricks.feature_store import FeatureLookup
from databricks import feature_store

#model_registry
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.xgboost
from mlflow.utils.environment import _mlflow_conda_env
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
import mlflow.pyfunc
import mlflow.sklearn

#metrics

from sklearn import metrics
from sklearn.metrics import DetCurveDisplay, det_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score, roc_curve,classification_report,precision_recall_curve,recall_score,f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.neighbors import KNeighborsClassifier

# Importing libraries
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
