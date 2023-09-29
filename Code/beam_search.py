# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from cmath import log
import string
import random
import sys
from torch._C import Size
from data_utils import *
from rnn import *
import torch
import codecs
from tqdm import tqdm
import string
import math
p = '/content/drive/MyDrive/neural_machine_translation/'
#Set GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load vocabulary files
input_lang = torch.load(p+'data-bin/fra.data')
output_lang = torch.load(p+'data-bin/eng.data')

#Create and empty RNN model
encoder = EncoderRNN(input_size=input_lang.n_words, device=device)
attn_decoder = AttnDecoderRNN(output_size=output_lang.n_words, device=device)

#Load the saved model weights into the RNN model
encoder.load_state_dict(torch.load(p+'model/encoder'))
attn_decoder.load_state_dict(torch.load(p+'model/decoder'))

#Return the decoder output given input sentence 
#Additionally, the previous predicted word and previous decoder state can also be given as input
def translate_single_word(encoder, decoder, sentence, decoder_input=None, decoder_hidden=None, max_length=MAX_LENGTH, device=device):
  with torch.no_grad():
      input_tensor = tensorFromSentence(input_lang, sentence, device)
      input_length = input_tensor.size()[0]
      
      encoder = encoder.to(device)
      decoder = decoder.to(device)
      
      encoder_hidden = encoder.initHidden()

      encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

      for ei in range(input_length):
          encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
          encoder_outputs[ei] += encoder_output[0, 0]

      if decoder_input==None:
          decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
      else:
          decoder_input = torch.tensor([[output_lang.word2index[decoder_input]]], device=device) 
      
      if decoder_hidden == None:        
          decoder_hidden = encoder_hidden
      
      decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
      return decoder_output.data, decoder_hidden


def beam_search(encoder,decoder,input_sentence,beam_size=1,max_length=MAX_LENGTH,device=device):
  decoded_output = []
  dh = []
  #Predicted the first words
  decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, decoder_input=None, decoder_hidden=None)
  for _ in range(beam_size):
    dh.append(decoder_hidden)
  #Get the probability of all output words
  decoder_output_probs = decoder_output.data
  #Select the ids of the words with top k maximum probability
  val,idx = torch.topk(decoder_output_probs.flatten(), beam_size)
  # if first words prediction has EOS in top k get next most likely prediction
  for i in range(beam_size):
    if idx[i].item()==EOS_token:
      d=1
      for x in range(len(decoder_output_probs.flatten())):
        diff = val[-1].item()-decoder_output_probs.flatten()[x].item()
        if diff<d and diff>0:
          d=diff
          new_val = decoder_output_probs.flatten()[x]
          new_idx = x
      new_idx = torch.tensor([new_idx])
      new_idx= new_idx.to(device)
      val = torch.cat([val[0:i], val[i+1:]])
      idx = torch.cat([idx[0:i], idx[i+1:]])
      val = torch.cat((val,new_val.unsqueeze(0)),0)
      idx = torch.cat((idx,new_idx),0)
  #get log probabilities for predicted probabilities
  pval = torch.log(val)
  #get words in list from indexes
  previous_decoded_output = [[(output_lang.index2word[idx[i].item()],pval[i].item())] for i in range(len(pval))]# if not idx[i].item()==EOS_token]
  flag='noteos'
  temp = []
  dhn=dh.copy()
  final_sent=[]
  #Loop until the maximum length
  for i in range(max_length):
    prob_tensor=[]
    for j in range(beam_size):
        if(j>=len(previous_decoded_output)):
          break
        if j<len(dh):
          decoder_hidden=dh[j]
        #Predict the next word given the previous prediction and the previous decoder hidden state
        decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, previous_decoded_output[j][-1][0], decoder_hidden)
        #Get the probability of all output words
        decoder_output_probs = decoder_output.data
        #Select the id of the words with maximum probability
        val,idx = torch.topk(decoder_output_probs.flatten(), beam_size)
        #calculate normalized log probs for each prediction
        nval = torch.add(torch.mul(torch.log(val),(1/(i+2))),((pval[j].item()*(i+1))/(i+2)))
        #store sentence in another list if EOS is predicted
        for _ in range(beam_size):
          for b in range(len(nval)):
            if idx[b].item() == EOS_token:
              if len(nval)==1:
                nval = []
                idx = []
                break
              nval = torch.cat([nval[0:b], nval[b+1:]])
              idx = torch.cat([idx[0:b], idx[b+1:]])
              if previous_decoded_output[j] not in final_sent:
                final_sent.append(previous_decoded_output[j])
              break
  
        if len(nval)>=1:
          #store current decoder hidden state
          dhn[j]=decoder_hidden
          #get words from word indexes
          selected_words = [[(output_lang.index2word[idx[x].item()],nval[x].item())] for x in range(len(nval))]
          #add new words to previous_decoded_output
          for word in selected_words:
              ts = previous_decoded_output[j].copy()
              ts.extend(word)
              temp.append(ts)
          #concantenate all probabilities
          if len(prob_tensor) ==j:
              prob_tensor = nval
          else:
              # print('yo')
              prob_tensor = torch.cat((prob_tensor,nval),0)
        else:
          if len(prob_tensor)==0:
            flag='end'
          continue
    # terminate if all sentences have EOS token
    if(flag=='end'):
      break
    # get top k=beam_size probs from concantenated probabilities
    if len(prob_tensor)<beam_size:
      pval,pidx = torch.topk(prob_tensor.flatten(),len(prob_tensor))
    else:
      pval,pidx = torch.topk(prob_tensor.flatten(),beam_size)
    # keep k new words with highest probabilities
    previous_decoded_output = [temp[x.item()] for x in pidx]
    temp = []
    # keep k decoder hidden states corrosponding to previous_decoded_output
    dh=[dhn[math.floor(x.item()/beam_size)] for x in pidx]
  
  #get max prob sentence as final output sentence
  max = -999999
  seni = 0
  final_sent.extend(previous_decoded_output)
  for index,v in enumerate(final_sent):
    if v[-1][1]>max:
      max = v[-1][1]
      seni = index
  
  #Convert list of predicted words to a sentence and detokenize 
  output_translation = " ".join(i[0] for i in final_sent[seni])
  return output_translation


# target_sentences = ["i can speak a bit of french .",
#         "i ve bought some cheese and milk .",
#         "boy where is your older brother ?",
#         "i ve just started reading this book .",
#         "she loves writing poems ."]
target_sentences = []
part1 = open(p+'data/valid.eng', 'r')
part2 = part1.readlines()
for line in part2:
   target_sentences.append(line)

# source_sentences = ["je parle un peu francais .",
#             "j ai achete du fromage et du lait .",
#             "garcon ou est ton grand frere ?",
#             "je viens justement de commencer ce livre .",
#             "elle adore ecrire des poemes ."]
source_sentences = []
part1 = open(p+'data/valid.fra', 'r')
part2 = part1.readlines()
for line in part2:
   source_sentences.append(line)
for k in range(20):
  target = codecs.open(p+'valid_beam_'+str(k+1)+'.out','w',encoding='utf-8')

  beam_size = k+1
  sencount = 0
  for i,source_sentence in enumerate(source_sentences):
    target_sentence = normalizeString(target_sentences[i])
    input_sentence = normalizeString(source_sentences[i])
    sencount+=1
    hypothesis = beam_search(encoder, attn_decoder, input_sentence, beam_size=beam_size)
    # print("S-"+str(i)+": "+input_sentence)
    # print("T-"+str(i)+": "+target_sentence)
    # print("H-"+str(i)+": "+hypothesis)
    # print()
    target.write(hypothesis+'\n')
    # break
  target.close() 
  # break
  