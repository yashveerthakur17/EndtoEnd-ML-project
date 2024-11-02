import sys
from dataclasses import dataclass
from statistics import median

import pandas as pd
import numpy as np
from pandas.core.interchange.from_dataframe import categorical_column_to_series
#from pywin32_testutil import testmain
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.mlproject.exceptions import  CustomException
from src.mlproject.logger import logging
import os
from src.mlproject.utils import save_object

@dataclass
class DataTransformationConfig:
    '''
    this will return the trained model as pickle file on the given path , /artifact
    '''
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        :return:
        '''

        try:
            #hard coded, make general purpose if had to use different dataset
            numerical_columns=['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',  # Corrected column name
                'lunch',
                'test_preparation_course'
            ]

            #pipeline

            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            categorical_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f'Numerical_columns:{numerical_columns}')

            #column transformer takes a list of pipelines
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline', categorical_pipeline, categorical_columns)
                ]
            )
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info('reading test and train files')

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']

            #logging.info('printing reading test and train files')
            #print("Target column name:", target_column_name)


            #segregating independent and dependent data set
            input_feature_train_df=train_df.drop(columns=target_column_name)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=target_column_name)
            target_feature_test_df = test_df[target_column_name]

            logging.info('preprocessing applied on train and test data')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            # not fit_transform due to data leakage concept imp****
            #input_features_test_arr=preprocessing_obj.fit(S


            #recombining
            #c_ is concat in np
            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f'Saved preprocessing obj')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)