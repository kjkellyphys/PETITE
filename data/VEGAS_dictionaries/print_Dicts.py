import pickle

my_file=open('samp_Dicts.pkl', 'rb')


pickled_data=pickle.load(my_file)

print(pickled_data.keys())

print(pickled_data['PairProd'][0]) 
#for n in range(0,100):
#    print('OG', pickled_data['PairProd'][n][1]['max_F'])

my_file.close()

