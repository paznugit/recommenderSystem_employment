# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 09:15:05 2015

@author: IGPL3460
"""

csv_input = '../input/dm_off.csv'
csv_cible = '../input/dm_off_ng.csv'

#==============================================================================
# Convert a file which represent:
# - For each line a job offer
# - For a job offer we have:
#   => kc_offre
#   => dn_frequencedeplacement
#   => dn_typedeplacement     
#   => dc_typexperienceprof_id
#   => dn_dureeminexperienceprof      
#   => dc_typdureeexperienceprof
#   => dc_rome_id       
#   => dc_appelationrome_id
#   => dc_naturecontrat_id 
#   => dc_typecontrat_id
#   => dc_unitedureecontrat
#   => dc_duree_contrat_id 
#   => dn_salaireannuelminimumeuros
#   => dc_naf2 
#   => dc_qualification_id 
#   => dc_modepresentation_emp_id  
#   => dc_langue_1_id   
#   => dc_niveaulangue_1_id
#   => dc_exigibilitelangue_1_id
#   => dc_permis_1_id   
#   => dc_exigibilitepermis_1_id
#   => dc_communelieutravail
#   => dc_departementlieutravail
#   => dc_typelieutravail  
#   => dc_lbllieutravail
#
# To a new file which represent:
# - For each line a job offer
# - The same information as above but we aggregate:
#   => The Duration of the contract
#   => The Duration of experience
#==============================================================================
  
def convertDureeJour(dc_unitedureecontrat,dc_duree_contrat_id ):
    duree = 0
    if dc_duree_contrat_id == '00':
        return str(duree)
    if dc_unitedureecontrat == 'MO':
        duree = int(dc_duree_contrat_id)*30
    else:
        duree = int(dc_duree_contrat_id)
    return str(duree)
    
def convertExperience(dn_dureeminexperienceprof,dc_typdureeexperienceprof):
    duree = 0
    if dn_dureeminexperienceprof == '0':
        return str(duree)
    if dc_typdureeexperienceprof == 'AN':
        duree = int(dn_dureeminexperienceprof)*12
    else:
        duree = dn_dureeminexperienceprof
    return str(duree)
    
with open(csv_cible, 'w') as outfile:
    with open(csv_input, 'r') as infile:

        i = 0
        for line in infile:
            joboffer = line.split("\t")
                          
            experienceMois = convertExperience(joboffer[4],joboffer[5])
            dureeContratJour = convertDureeJour(joboffer[10],joboffer[11])
            
            outfile.write(joboffer[0]+",")
            outfile.write(joboffer[1]+",")
            outfile.write(joboffer[2]+",")
            outfile.write(joboffer[3]+",")
            outfile.write(experienceMois+",")
            outfile.write(joboffer[6]+",")
            outfile.write(joboffer[7]+",")
            outfile.write(joboffer[8]+",")
            outfile.write(joboffer[9]+",")
            outfile.write(dureeContratJour+",")
            outfile.write(joboffer[12]+",")
            outfile.write(joboffer[13]+",")
            outfile.write(joboffer[14]+",")
            outfile.write(joboffer[15]+",")
            outfile.write(joboffer[16]+",")
            outfile.write(joboffer[17]+",")
            outfile.write(joboffer[18]+",")
            outfile.write(joboffer[19]+",")
            outfile.write(joboffer[20]+",")
            outfile.write(joboffer[21]+",")
            outfile.write(joboffer[22]+",")
            outfile.write(joboffer[23]+",")
            outfile.write(joboffer[24])
            
            i += 1
            #if i == 5:
                #break
            
            if i%10000 == 0:
                print i