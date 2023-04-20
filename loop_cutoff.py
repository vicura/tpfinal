import sys, argparse
import subprocess

#def ejecutar_pre_entrenamiento(cutoff):
   # !python3 pre_entrenamiento.py --path '.' --out_name f'prueba_cutoff_{cutoff}_maxneigh_20' --cutoff cutoff --max_neigh 20 --n_samples 80200
    

#def ejecutar_entrenamiento(cutoff):
 #   !python3 entrenamiento.py --dataset f'prueba_cutoff_{cutoff}_maxneigh_20_scaled_shuffled_equal_samples.npy' --labels f'prueba_cutoff_{cutoff}_maxneigh_20_scaled_shuffled_equal_labels.npy'  --batch_size 16 --nepochs 15


#def ejecutar_testeo(file,cutoff):
 #   !python3 testeo.py --n_classes 2 --file_trj file --cutoff cutoff --maxneigh 20 --outname f'lamelar_cutoff_{cutoff}_maxneigh_20' 

   
def main():       

   args = get_args() 
   # Lista de cutoffs
#'0.6','0.7','0.8','0.9','1.0','1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9',
   cutoffs = ['2.0']

   for cutoff in cutoffs:
      subprocess.run(['python3', 'pre_entrenamiento.py', '--path', '.', '--out_name', 
                      f'prueba_cutoff_{cutoff}_maxneigh_50', '--cutoff', str(cutoff),
                        '--max_neigh', '50', '--n_samples', '60150'])
      subprocess.run(['python3', 'entrenamiento.py', '--dataset', 
                      f'prueba_cutoff_{cutoff}_maxneigh_50_scaled_shuffled_equal_samples.npy', 
                      '--labels', f'prueba_cutoff_{cutoff}_maxneigh_50_scaled_shuffled_equal_labels.npy',
                      '--batch_size', '32', '--nepochs', '50'])
      subprocess.run(['python3','testeo_ultimos_frames.py','--n_classes','2','--file_trj', args.file_trj,'--cutoff',cutoff,
                       '--maxneigh','50','--outname', f'lam_cutoff_{cutoff}_maxneigh_50'])
      
      #ejecutar_pre_entrenamiento(cutoff)
      #ejecutar_entrenamiento(cutoff)
      #ejecutar_testeo(args.file_trj,cutoff)  
              
   return   


def get_args():
    #Parse Arguments
    parser = argparse.ArgumentParser(description='Evalua a cada cutoff especificado la fracción de cada clase')

    parser.add_argument('--file_trj', help='path to files', type=str, required=True)
   
    args = parser.parse_args()
    
    return args
   
if __name__ == "__main__":
   main()
