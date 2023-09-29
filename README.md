# Search-algorithms-and-neural-machine-translation
implemented search algorithms and applied them to two real-world datasets.  


#### Part 1. NYC Taxi Data  
performed Uniform Cost Search and A∗ search on a list of the taxi trips made in NYC in the January 2015.  
Dataset columns used:  
• pickup longitude • pickup latitude  
• dropoff longitude • dropoff latitude • trip distance  
  
1. Represented the data as a graph. For the sake of simplicity, we assume that edges are only between locations which had a valid taxi trip between them.  
   • We can either do it as an adjacency matrix or as an adjacency list. Each node in the graph will be represented by the lat/long values.  
2. Output the graph as two csv files.  
   • nodes.csv: Containing a list of lat/long and the corresponding node ID. The file should have the following columns: nodeid, lat, long.  
   • edges.csv: Containing tuple of node IDs which have an edges between them. The file should have the following columns: nodeid1, nodeid2  
  
3. Implemented Uniform Cost search where you can use the trip distances as the edge costs.  
   • The program inputs two node ids from the user and outputs the path as well as the cost between them.  
4. Implement A∗ search using a heuristic. One idea of a heuristic value is to use straight line distance between 2 points. This can be computed using the geopy package.  
   • The program inputs two node ids from the user and output the path as well as the cost between them.  
    
#### Part 2. Neural Machine Translation  
implemented beam search for Neural Machine Translation (NMT). This NMT model is already trained on the French to English translation task and a sample file is provided to run for few examples. The data files have already been pre-processed including tokenization and normalization.  
We have the following files:  
• models/encoder & models/decoder: trained model files  
• data-bin/fra.data & data-bin/eng.data: vocabulary files created using the training data  
• data/test.fra & data/test.eng: normalized and tokenized test files  
• data utils.py & rnn.py: supporting files for data processing and model initialization  
• beam search.py: Main file to translate input sentences  
  
  
1. Implemented beam search with beam size 1 (greedy prediction) in beam_search.py . Ran the Search for beam size 1 and saved the output in a new file named test_beam_1.out. Used sacrebleu pacakage to compute the BLEU Score[1] performance of the model to get an initial baseline.  
2. Implemented beam search algorithm to use any beam size.  
3. Applied beam search on the valid.fra file for beam size K = 1, 2, · · · , 20. Computed the BLEU scores for each of the output files and Plotted them.  
4. Applied the beam search on the test.fra file using the K which gives the highest BLEU score in part 3. Computed the BLEU score for the test translations.  

References
[1] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002.
4

