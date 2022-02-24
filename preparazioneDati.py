# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:04:53 2021

@author: Michele
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os 

"""salvo i tempi di produzione"""
tempi = pd.read_excel("Tempi.xlsx", sheet_name="Foglio1")

"""raggruppo per codice e calcolo la media dei tempi di ogni codice"""
tempi_by_codice = tempi.groupby("codice")
t_eff_medio = tempi_by_codice["t_eff"].mean()
t_eff_medio = t_eff_medio.reset_index()
t_eff_medio["codice"] = t_eff_medio["codice"].astype(dtype="string")

"""salvo i codici padre"""
padri = pd.DataFrame(pd.read_csv("Bom.csv"), columns =["Code","Revision","_FullName" ])
padri = padri.fillna(0)

"""salvo le anagrafiche"""
anagrafiche= pd.DataFrame(pd.read_csv("Item.csv"), columns = ["Code", "Revision", "Description"])

"""salvo le linee dei figli"""
distinte = pd.DataFrame(pd.read_csv("Bom.BomLines.csv"), columns = ["_FullName", "_Index1", "Code", "Revision", "Quantity"])
distinte = distinte.fillna(0)

"""faccio il merge per recuperare il codice padre"""
df = padri.merge(distinte, how ="inner", left_on = "_FullName", right_on="_FullName")


df = df.rename(columns = {"Code_x" : "Codice_Padre", "Code_y" : "Codice_Figlio"})


df = df.merge(anagrafiche,how = "left", left_on= "Codice_Figlio", right_on = "Code" )


df = df[["Codice_Padre", "Quantity", "Description"]]

df = df.rename(columns = {"Description" : "Descrizione_Figlio"})


df = df.set_index("Codice_Padre")

count_letter = pd.DataFrame( { k:df["Descrizione_Figlio"].str.count(k)*df["Quantity"] for k in "ABCDEFGHIKLMNOPQRSTUVWXYZ" })

df = count_letter.groupby(["Codice_Padre"]).sum()

df = df.reset_index()

df = df.merge(t_eff_medio, how="inner", left_on ="Codice_Padre", right_on ="codice")

df = df.drop(columns = ["Codice_Padre", "codice"])

if not os.path.exists(r"C:\TIROCINIO\dati\input.csv"):
    df.to_csv(r"C:\TIROCINIO\dati\input.csv",index = False)


