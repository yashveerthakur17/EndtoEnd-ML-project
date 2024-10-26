import os
import sys
from urllib.request import localhost

from src.mlproject.logger import logging
from src.mlproject.exceptions import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql

#to make use of env
load_dotenv()

#same var used in dot env and at try:
host=os.getenv("localhost")
user=os.getenv("root")
password=os.getenv("root")
db=os.getenv("college")

def read_sql_data():
    logging.info('reading sql database started')
    try:
        mydb = pymysql.connect(host=host, user=user, password=password, db=db)
        logging.info('mysql connection successful')

        df=pd.read_sql_query('SELECT * FROM students.student', mydb)
        print(df.head())

        return df


    except Exception as e:
        raise CustomException(e,sys)