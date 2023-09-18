import pickle

cross_section_file=open("sm_xsec.pkl", 'rb')
cross_section_dict=pickle.load(cross_section_file)

for key in cross_section_dict:
    print (cross_section_dict[key].keys())

print(cross_section_dict["Brem"][6.0])
