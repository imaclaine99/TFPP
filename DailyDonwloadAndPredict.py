import MyFunctions as myf
import ModelV2Config
import MyLSTMModelV2b       # Has dependencies that I need...

import mysql.connector

# The sequence here needs to be:
#
###
# 1. Download / get the latest data.  This needs to be the correct size and shape (i.e. Date, Open, High, Low, Close, with 220)
#   1. a - Where to best get this from?  Automated would obviously be preferable, just need to be aware of the date that is has data for, given google data seems to be a bit late
#    1. b - Make this flexible - don't want to be bound to anything too rigid' \
#
# 2. Have a loop to process through new data, and run predictions with different model / rule combos  (note - a saved model effectively has the rule built in)
# 3. This is somewhat separate, but a means to do more training on new data.  This doesn't need to be run daily, but needs to be able to be run.


# Comment the below out - AV seems to be having an issue - weekend??
#myf.download_and_parse('^GDAXI')

# Now what do I want?
# Function that:
# Takes a CODE (e.g. ^GDAXI)
# Takes a number of days (e.g. 1, 10, 100)
# Takes a MODEL_FILENAME
# Runs N predictions
# Returns this is an array

# 22nd June
# Given there are challenges in getting good quality data over extended timeframes, it makes sense to store data locally, and add to it daily.
# This splits the task of getting volume of data vs the need for current data.
# The model will be as follows:
# a. DB Table that stores OHLCV history with dates
# b. The above can be updated however makes sense
# c. Function to get up to date data, and save as a local .csv file
# d. Function to upload the csv into the DB - flag any duplicates that are not identical
# e. Then need a function that extracts from this - i.e. update the current predict_code_model to read from a DB rather than from a file

# Status:
# a - DONE - note - may want another table to match codes with code aliases?
# b - Needs a two step process, due to formatting challenges - e.g. having SYMBOL field, DATE formatting, etc.
#   - So, use a temp table, and then insert/update from there
# c - Need a new data source - 12 months is FINE!!

#predictions = myf.predict_code_model ('^GDAXI', '31204_5_Layers.h5', predictions=1)

# Daily process - say at 9am
# 1 - Select all UNIQUE codes from prediction_pairs
# 2 - For each of these, download and store data
#   - added capture of ATR20 and date - hacky but helpful
# 3 - For each prediction_pairs, run prediction

ModelV2Config.disableGPU=True           # Better to be slow and work, than risk failing, given lots of stuff could be running
num_predictions = 50

cnx = myf.get_db_connection()
cursor = cnx.cursor(dictionary=True)

qry = ("select distinct symbol from prediction_pairs ")
cursor.execute(qry);
rows = cursor.fetchall()

for symbol in rows:
    myf.download_to_db(symbol['symbol'])        # Step 1
    last_date, atr20 = myf.parse_from_db(symbol['symbol'], 500)    # Step 2
    # Need function to store ATR20
    qry = ("insert ignore into predictions values (%s, %s, %s, %s, %s)")
    cursor.execute(qry, (symbol['symbol'], last_date, 'ATR20_Actual', atr20, 1));
    # How to get the date???
    cnx.commit()


qry = ("select * from prediction_pairs ")
cursor.execute(qry);
rows = cursor.fetchall()

for row in rows:
    print (row)
    parsed_filename = row['Symbol'] + '_tempfile.csv'  # This comes from the above
    predictions = myf.predict_code_model (parsed_filename, row['Model'] + '.h5', predictions=num_predictions)
    qry = ("insert ignore into predictions values (%s, %s, %s, %s, %s)")
    for prediction in predictions:                  # It will normally be one, but this is useful
            cursor.execute(qry, (row['Symbol'], "%.0f" %(prediction[0]), row['Model'], float(prediction[3]), myf.processing_range));


cnx.commit()
cnx.close()
#parsed_filename = '^GDAXI_Now.csv'