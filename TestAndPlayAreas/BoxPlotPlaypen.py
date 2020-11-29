import matplotlib
import MyFunctions as myf
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd

conn = myf.get_db_connection()
cursor = conn.cursor(dictionary=True)

sqlEngine = create_engine('mysql+mysqlconnector://' + myf.db_username + ':' + myf.db_pwd + '@' + myf.db_host + '/tfpp',
                          pool_recycle=3600)

dbConnection    = sqlEngine.connect()

query1 = ("select  *, mid(description, instr(Description, 'NoiseTestExec_')+14, instr(description, 'Iteration') - 34 + 8) + 0E0 InputNoise FROM tfpp.executionlog "
"where filename like 'ModelV2B%' "
"and instr(Description, 'NoiseTestExec') > 0 "
"and rule like 'BuyV3%' "
"and unique_id = 26911 ")

query2 = ("select  *, mid(description, instr(Description, 'NoiseTestExec_')+14, instr(description, 'Iteration') - 34 + 8) + 0E0 InputNoise FROM tfpp.executionlog "
"where filename like 'ModelV2B%' "
"and instr(Description, 'NoiseTestExec') > 0 "
"and rule like 'BuyV3b' "
"and unique_id = 26911 ")

frame1           = pd.read_sql(query1, dbConnection);
frame2           = pd.read_sql(query2, dbConnection);

frame1.boxplot('model_best_val_loss', by=['InputNoise','Rule'])
matplotlib.pyplot.show()

frame2.boxplot('model_best_val_loss', by=['InputNoise'])
matplotlib.pyplot.show()

del dbConnection
