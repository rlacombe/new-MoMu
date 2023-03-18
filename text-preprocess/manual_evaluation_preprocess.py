


def process_file(infile_path):
  outfile_path = "processed.txt"
  outfile_str = list()
  with open(infile_path, 'r') as f:
    for i, para in enumerate(f.readlines()):
      para = " ".join(para.split()[:330]) 
      outfile_str.append("-"*80)
      outfile_str.append("i: {} {}".format(i, para))
  outfile_str = "\n".join(outfile_str)
  f = open(outfile_path, 'w')
  f.write(outfile_str)

if __name__ == '__main__':
  process_file('/Users/agaut/Downloads/text_502.txt')




