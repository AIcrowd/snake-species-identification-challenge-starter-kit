#!/usr/bin/env python

import numpy as np
import os
import glob

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


LINES = []
LINES.append('filename,agkistrodon_contortrix,agkistrodon_piscivorus,boa_imperator,carphophis_amoenus,charina_bottae,coluber_constrictor,crotalus_adamanteus,crotalus_atrox,crotalus_horridus,crotalus_pyrrhus,crotalus_ruber,crotalus_scutulatus,crotalus_viridis,diadophis_punctatus,haldea_striatula,heterodon_platirhinos,hierophis_viridiflavus,lampropeltis_californiae,lampropeltis_triangulum,lichanura_trivirgata,masticophis_flagellum,natrix_natrix,nerodia_erythrogaster,nerodia_fasciata,nerodia_rhombifer,nerodia_sipedon,opheodrys_aestivus,opheodrys_vernalis,pantherophis_alleghaniensis,pantherophis_emoryi,pantherophis_guttatus,pantherophis_obsoletus,pantherophis_spiloides,pantherophis_vulpinus,pituophis_catenifer,regina_septemvittata,rhinocheilus_lecontei,storeria_dekayi,storeria_occipitomaculata,thamnophis_elegans,thamnophis_marcianus,thamnophis_ordinoides,thamnophis_proximus,thamnophis_radix,thamnophis_sirtalis')

for _file_path in glob.glob("round1/*.jpg"):
	probs = softmax(np.random.rand(45))
	probs = list(map(str, probs))
	LINES.append(",".join([os.path.basename(_file_path)] + probs))

fp = open("random_prediction.csv", "w")
fp.write("\n".join(LINES))
fp.close()
