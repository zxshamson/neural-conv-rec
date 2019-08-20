# Introduction:
This is the implementation in PyTorch for my EMNLP2019 paper:

"Neural Conversation Recommendation with Online Interaction Modeling"


# Requirement:

* Python: 2.7+

* Pytorch: 0.4+

* Sklearn: 0.20.0

# Before running:
You need to download Glove pre-training embeddings from: 
https://nlp.stanford.edu/projects/glove/

"glove.twitter.27B.200d.txt" for twitter.data.

"glove.6B.200d.txt" for reddit.data.

# Usage:

`python conv_rec.py [filename] [modelname]`

```
[filename]: "twitter.data" or "reddit.data".
[modelname]: "NCF", "NCFBiLSTM", "NCFGCN", "NCFGRN".

optional arguments:
  --cuda_dev          choose to use which GPU (default: "0")
  --max_word_num      max word number in vocabulary, -1 means all words appear in dataset (default: -1)
  --pred_pc           percentage of turns in each conversation for training (default: 0.75)
  --factor_dim        embedding dimension for replying factor modeling part (default: 20)
  --text_factor_dim   embedding dimension for conversation interaction modeling part (default: 100)
  --neg_sample_num    sampling numbers for negative instances each positive instance when training (default: 5)
  --kernal_num        number of kernals for CNN encoder (default: 150)
  --embedding_dim     dimension for word embedding (default: 200)
  --hidden_dim        dimension for hidden states (default: 200)
  --batch_size        batch size during training (default: 256)
  --max_epoch         maximum iteration times (default: 199)
  --lr                learning rate during training (default: 0.01)
  --dropout           dropout rate (default: 0.2)
  --mlp_layers_num    number of layers for MLP part (default: 3)
  --gcn_layers_num    number of layers for GCN in conversation interaction modeling (default: 1)
  --grn_states_num    number of states for GLSTM in conversation interaction modeling (default: 6)
  --runtime           record the current running time (default: 0)
  --pos_weight        weights for positive instances in loss function during training (default: 100)
  --optim             optimizer for training (default:"adam", choices: "adam", "sgd")
  --bi_direction      whether change LSTM to BiLSTM (action="store_true")
  --use_gates         whether using gates in GCN modeling (action="store_true")
  --use_lstm          whether adding LSTM before GCN modeling (action="store_true")
  --ncf_pretrained    whether using NCF pretrained parameters to initialize the factors (action="store_true")
```

# About NCF pretraining

You need to first run a NCF model result first (then you can find the model parameters in "BestModels/NCF/"). 

Then put the model file into "BestModels/NCF/For\_pretrained/[Datafilename]", and change file name to be "[factor\_dim]\_[text\_factor\_dim]\_[pred\_pc].model". 

One possible example can be "BestModels/NCF/For\_pretrained/reddit/20\_100\_0.75.model".

# Datasets:

including "twitter.data" and "reddit.data"

format in each line:

[Conv ID] \t [Msg ID] \t [Parent ID] \t [Original sentence] \t [words after preprocessing] \t [User ID] \t [posting time]

(twitter dataset doesn't have time infos, but the conversations are ordered by posting time)
