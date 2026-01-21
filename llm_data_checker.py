#Libraries
import pandas as pd
import pathlib 

########################
#Data Read in function 
########################
def read_df(path):
    if path.suffix == '.csv':
        dataframe = pd.read_csv(path)
        return dataframe
    else:
        print("File type not accepted. Please use a CSV file type.")

#advanced path reading

#def read_df_adv(path):

    #reads csv from path
 #   if p

    #reads xlsx file from path
  #  elif 

    #reads csv direct
   # elif 

    #reads xlsx direct
    #elif 



########################
#df checker 
#(should check for empty data frames and do some basic data Anonymisation)
########################





########################
#Data Processing Prompt + Constraints
#(you can build registries in here that help guide the LLM)
########################




########################
#Prompt builder
#(this is where the registries and the prompt will be created)
########################


########################
#API Client call
########################