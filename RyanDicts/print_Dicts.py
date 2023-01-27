import pickle

my_file=open('samp_Dicts.pkl', 'rb')
pickled_data=pickle.load(my_file)


print(pickled_data)
my_file.close()
