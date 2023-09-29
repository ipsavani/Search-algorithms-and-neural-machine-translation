import sacrebleu
from sacremoses import MosesDetokenizer
from matplotlib import pyplot as plt
md = MosesDetokenizer(lang='en')
import numpy as np

def get_tokens(path):
    # Read target file and append tokenized lines to list
    tokens = []
    with open(path) as file:
        for l in file: 
            l = l.strip().split() 
            l = md.detokenize(l) 
            tokens.append(l)
    return tokens

# Project path
p = '/content/drive/MyDrive/neural_machine_translation/'

refs = get_tokens(p+'data/test.eng')
refs = [refs]
preds = get_tokens(p+'test_beam_1.out')
bleu = sacrebleu.corpus_bleu(preds, refs)
print(f'BLEU score for beam search 1 on test.fra : {bleu.score}')
print()
print('saving BLEU score vs beam size for plot for beam_search run on valid.fra')
refs = get_tokens(p+'data/valid.eng')
refs = [refs]
bleu_scores = []
beam_widths = []
# loop for calculating scores
for i in range(20):
  preds = get_tokens(p+'valid_beam_'+str(i+1)+'.out')
  # Calculate the BLEU score and append to score list
  bleu = sacrebleu.corpus_bleu(preds, refs)
  bleu_scores.append(bleu.score)
  beam_widths.append(i+1)
# plot figure and save plot to plot.png
plt.figure(figsize=(8, 4))
plt.plot(beam_widths,bleu_scores)
plt.title('BLEU Scores vs Beam Sizes on valid.fra')
plt.xlabel('Beam Size')
plt.xticks(np.arange(1,21,1))
plt.ylabel('BLEU Score')
plt.savefig(p+'plot.png')
# plt.show()
# get the highest BLEU score and corrosponding beam size
m = max(bleu_scores)
print(f'Max BLEU score(valid.fra) is : {m} ,With Beam size : {bleu_scores.index(m)+1}')
# got beam_size = 8 for my run

refs = get_tokens(p+'data/test.eng')
refs = [refs]
preds = get_tokens(p+'test_beam_8.out')
bleu = sacrebleu.corpus_bleu(preds, refs)
print()
print(f'BLEU score for beam search 8 on test.fra : {bleu.score}')