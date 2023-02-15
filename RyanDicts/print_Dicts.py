import pickle

my_file=open('samp_Dicts.pkl', 'rb')
my_file_Feb7=open('Feb7_samp_Dicts.pkl', 'rb')


pickled_data=pickle.load(my_file)
pickled_data_Feb7=pickle.load(my_file_Feb7)

print(pickled_data.keys())
for n in range(0,100):
    print('Feb7', pickled_data_Feb7['PairProd'][n][1]['max_F'])
    print('OG', pickled_data['PairProd'][n][1]['max_F'])

my_file.close()
my_file_Feb7.close()
