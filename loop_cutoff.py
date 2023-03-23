import sys, argparse


def ejecutar_pre_entrenamiento(cutoff):
    import pre_entrenamiento

    # Configurar argumentos del script
    args = argparse.Namespace(path='.', out_name=f'prueba_cutoff_{cutoff}_maxneigh_20', cutoff=cutoff, max_neigh=20, n_samples=80200)


def ejecutar_entrenamiento(cutoff):
    import entrenamiento
    args = argparse.Namespace(dataset=f'prueba_cutoff_{cutoff}_maxneigh_20_scaled_shuffled_equal_samples.npy', 
                              labels=f'prueba_cutoff_{cutoff}_maxneigh_20_scaled_shuffled_equal_labels.npy',
                              batch_size=16, nepochs=15)


def ejecutar_testeo(file,cutoff):
    import testeo
    args = argparse.Namespace(n_classes=2, file_trj=file, cutoff=cutoff, maxneigh=20, 
                              outname=f'salida_cutoff_{cutoff}_maxneigh_20'
    
   
   
def main():       

   args = get_args() 
   # Lista de cutoffs

   cutoffs = ['1.35','1.4','1.45','1.5','1.55','1.6','1.65','1.7','1.75','1.8','1.85','1.9','1.95','2.0']

   for cutoff in cutoffs:
    ejecutar_pre_entrenamiento(cutoff)
    ejecutar_entrenamiento(cutoff)
    ejecutar_testeo(args.file,cutoff)                
   

def get_args():
    #Parse Arguments
    parser = argparse.ArgumentParser(description='Evalua a cada cutoff especificado la fracción de cada clase')

    parser.add_argument('--file_trj', help='path to files', type=str, required=True)
   
    args = parser.parse_args()
    
    return args
   
if __name__ == "__main__":
   main()