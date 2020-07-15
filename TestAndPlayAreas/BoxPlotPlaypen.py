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

query = ("select  *, mid(description, instr(Description, 'NoiseTestExec_')+14, instr(description, 'Iteration') - 34 + 8) + 0E0 InputNoise FROM tfpp.executionlog "
"where filename like 'ModelV2B%' "
"and instr(Description, 'NoiseTestExec') > 0 "
"and rule = 'BuyV3' "
"and unique_id = 26911 ")

frame           = pd.read_sql(query, dbConnection);


#results = pd.DataFrame()
#for InputNoise in frame.InputNoise.unique():
#    new_results = frame.query('InputNoise == ' + str(InputNoise))
#    results[InputNoise] = new_results.model_best_val_loss

#print (frame)

#results.boxplot
frame.boxplot('model_best_val_loss', by='InputNoise')

matplotlib.pyplot.show()
del dbConnection

cursor.execute(query)

rows = cursor.fetchall()

print (rows)