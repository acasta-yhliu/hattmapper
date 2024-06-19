import json

name = input()
f = open(name)
molecule = json.load(f)
represent = ""
for i in range(len(molecule["PC_Compounds"][0]['atoms'])):
    represent = represent + str(molecule["PC_Compounds"][0]['atoms']['element'][i]) + " " + \
    str(molecule["PC_Compounds"][0]['coords'][0]['conformers'][0]['x'][i]) + " " + \
    str(molecule["PC_Compounds"][0]['coords'][0]['conformers'][0]['y'][i]) + " " + \
    str(molecule["PC_Compounds"][0]['coords'][0]['conformers'][0]['z'][i]) + \
    "; "
print(represent)
