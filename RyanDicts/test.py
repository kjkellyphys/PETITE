import pickle

my_file=open('xSec_Dicts.pkl', 'rb')
pickled_data=pickle.load(my_file)

my_file.close()

print(pickled_data)
