import numpy as np

def cohens_kappa(annotation1, annotation2):
  p_o = np.sum(annotation1==annotation2) / annotation1.shape[0]
  p_e_yes = np.sum(annotation1==1)/annotation1.shape[0]*np.sum(annotation2==1)/annotation2.shape[0]
  p_e_no = np.sum(annotation1==0)/annotation1.shape[0]*np.sum(annotation2==0)/annotation2.shape[0]
  p_e = p_e_yes + p_e_no

  return (p_o - p_e) / (1-p_e)

def preprocess(file_path):
  f = open(file_path, 'r')
  annotation1 = list()
  annotation2 = list()
  for line in f.readlines():
    a1, a2 = line.split()
    annotation1.append(int(a1))
    annotation2.append(int(a2))
  return np.array(annotation1), np.array(annotation2)

if __name__ == '__main__':
  annotation1, annotation2 = preprocess('./annotations.txt')
  print(cohens_kappa(annotation1, annotation2))
