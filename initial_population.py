#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:13:52 2020

@author: root
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pymoo.model.population import pop_from_array_or_individual
import compute_genome
# make initial population for genetic algorithm
def initialize_spatial(pop_size,default_directory):
    all_landusemaps = []
    landuse_map_in = np.load(default_directory + "/Landuse_maps/landuse_map_in.npy")
    
    rows = landuse_map_in.shape[0]
    cols = landuse_map_in.shape[1]
    
    # iterate to get multiple realisations for the initial population
    for i in range(1,pop_size+1):

        
        #use uniform distribution to select 30% of the cells    
        landuse_map_ini = np.zeros((rows,cols),dtype='uint8')
        random_map = np.random.uniform(0.0,1.0,(rows,cols))
        random_map_mw = np.zeros((rows,cols))
        
        for x in range(0,cols):
            for y in range(0,rows):
                if x == 0 or y == 0 or x == cols-1 or y == rows-1:
                    random_map_mw[x,y] = 1.0            
                else:
                    random_map_mw[y,x] = random_map[y-1:y+2,x-1:x+2].mean()
        
        
        # 70% of the map remains the current land use
        landuse_map_ini = np.where(random_map_mw>=0.3,landuse_map_in,landuse_map_ini)
            
        # 30% of the map will become new
        # urban, water and no data will remain the same
        landuse_map_ini = np.where(landuse_map_in >= 8,landuse_map_in,landuse_map_ini)
            
        # other land use types can change into 4, 5, 6 or 7        
        #choose which land cover type
        landuse_map_ini = np.where(landuse_map_ini == 0, np.random.uniform(0.0,1.0,(rows,cols)), landuse_map_ini)
        
        landuse_map_ini = np.where(landuse_map_ini < 0.25, 4, landuse_map_ini)
        landuse_map_ini = np.where(landuse_map_ini < 0.50, 5, landuse_map_ini)
        landuse_map_ini = np.where(landuse_map_ini < 0.75, 6, landuse_map_ini)
        landuse_map_ini = np.where(landuse_map_ini < 1.00, 7, landuse_map_ini)
        #patchmap, genome =compute_genome.create_patch_ID_map(landuse_map_ini,0,[8,9],'True')
        all_landusemaps.append(landuse_map_ini)
    return np.array(all_landusemaps)
