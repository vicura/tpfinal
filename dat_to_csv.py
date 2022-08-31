import sys
import os
import numpy as np
import pandas as pd

def dat_to_csv(archivo_dat,archivo_csv):

   with open(archivo_dat, 'r') as in_file:
      with open(archivo_csv, 'w') as out_file:
         in_file.readline()
         for i in range(10000000):
            f = in_file.readline()
            if f != '':
                  f = f.rstrip('        \n')
                  f = f.replace('   ',',')
                  f = f.split(',')
                  f = list(filter(('').__ne__, f))
                  f = ",".join(map(str, f))           
                  out_file.write(f)
                  out_file.write("\n")
            else:
                  break
   
   return archivo_csv
   
   
def main(parametros):
    
   if len(parametros) == 3:
      archivo_dat = parametros[1]
      archivo_csv = parametros[2]
      salida = dat_to_csv(archivo_dat,archivo_csv)

      return salida

if __name__ == '__main__':
    main(sys.argv)
   

