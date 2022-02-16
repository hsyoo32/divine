'''
Written by lyc0324@agape.hanyang.ac.kr
'''
import argparse
'''
parser = argparse.ArgumentParser('side/embedding revise')
parser.add_argument('--file', nargs='?', default='epinions_prepro', help='input file name')
parser.add_argument('--output', nargs='?', default='../../_Emb/SIDE/epinions_prepro.emb', help='output file name')
parser.add_argument('--emb_dim', nargs='?', default=20, help='embedding size')

args = parser.parse_args()
filename = args.file
embed_filename = args.output
num_embed = args.emb_dim
'''
def revise_emb(deleted, epoch):

  vocab_file = deleted
  filename = deleted + "_ep{}".format(epoch)
  embed_filename = "../../_Emb/SIDE/" + deleted + "_ep{}".format(epoch)
  num_embed = 128
  # '''

  with open("./output/" + vocab_file + ".vocab", "r") as f1, \
       open("./output/" + filename + ".emb", "r") as f2, \
       open(embed_filename + ".emb", "w+") as f3:
       # open(embed_filename[:-4] + "_its_wiki" + "_ep{}".format(epoch) + ".emb", "w+") as f3:
    lines = f1.readlines()

    nodes = []
    for line in lines:
      splited_line = line.split(' ')
      idx = splited_line[0].replace("'","")
      idx = int(idx.replace("b",""))
      nodes.append(idx)

    lines = f2.readlines()
    f3.writelines(str(len(nodes)) + " " + str(num_embed) + "\n")

    index = 0
    for line in lines:
      f3.writelines(str(nodes[index]) + " " + str(line))
      index += 1
  # '''

  print("End .emb")


# deleted = "190719_SIDE_slashdot_filtered_nodeg-opt"
# deleted = "190807_n_SIDE_wiki_noEdit_withDeg1"
# for e in range(11, 37):
deleted = "bitcoin-alpha"
# deleted = "190808_n_SIDE_epiAmi_noEdit_withDeg1"
for e in range(0, 10):
  print(e)
  revise_emb(deleted, e)
# 

# ws = [10, 7, 5]
# deleted = "190802_n_SIDE_ws{}r{}_slashAmi_noEdit"
# for lr in [0.01, 0.1, 0.0001]:
#   for w in ws:
#     for e in range(0, 30):
#       print(e, w, lr)
#       revise_emb(deleted.format(w, lr), e)




'''
## for embedding 2
with open("./output/" + vocab_file + ".vocab", "r") as f1, \
     open("./output/" + filename + ".emb2", "r") as f2, \
     open(embed_filename + ".emb2", "w+") as f3:
  lines = f1.readlines()

  nodes = []
  for line in lines:
    splited_line = line.split(' ')
    idx = splited_line[0].replace("'","")
    idx = int(idx.replace("b",""))
    nodes.append(idx)

  lines = f2.readlines()
  f3.writelines(str(len(nodes)) + " " + str(num_embed) + "\n")

  index = 0
  for line in lines:
    f3.writelines(str(nodes[index]) + " " + str(line))
    index += 1
print("End .emb2")
'''


'''
## for bias
with open("./output/" + vocab_file + ".vocab", "r") as f1, \
     open("./output/" + filename + ".bias", "r") as f2, \
     open(embed_filename + ".bias", "w+") as f3:
  f1_lines = f1.readlines()

  nodes = []
  for line in f1_lines:
    splited_line = line.split(' ')
    idx = splited_line[0].replace("'","")
    idx = int(idx.replace("b",""))
    nodes.append(idx)

  # pos_in = []
  # neg_in = []
  # pos_out = []
  # neg_out = []
  index = 0
  f2_lines = f2.readlines()
  for line in f2_lines:
    f3.writelines(str(nodes[index]) + " " + str(line))
    index += 1
print("End .bias")
'''
