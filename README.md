# Transformers 

In this repository, I implement the Transformer and apply it to various seq2seq language taks: **spelling and grammar error corrector**, **paraphrase generation** and **text summarization**. The goal is to explore how the following modifications influence performance on each task:

- **Static Positional Encoding** vs **Learned Positional Encoding** 
- **Label Smoothing**
- **Word-level Encoding** vs **Character-level Encoding** vs **Byte Part Encoding**
- **Greedy Decoding** vs **Beam Search**

## Spelling and Grammar Error Corrector
We begin with the spelling and grammar error corrector. The goal of such a system is to correct typos made in a given text if they are present. For example *I training a spell corecotr* needs to be corrected to *I am training a spell corrector*, making sure to correct both spelling errors *corecotr* -> *corrector* as well as grammatical errors *I training* -> *I am training*. 

Dataset used: [Github typo corpus](https://github.com/mhagiwara/github-typo-corpus) 

Positional encoding is designed to incorporate a noition of order of tokens in a sentence. Learned positional encoding is implemented by introducing an embedding layer of maximum token size and embedding size. Static positional encoding are implemented by introducing fixed sinusoidal functions of different frequencies. For more information about positional encodings, check [this paper](https://www.aclweb.org/anthology/2020.emnlp-main.555/). 

We train the corrector with character-level and and byte-part encoding (BPE) encoding, since word-level encoding is not applicable for this task. While training with character-level encoding, the validation loss did not decrease for a very large number of iterations. 
To create BPE encodings, we use [BertTokenizer](https://huggingface.co/transformers/model_doc/bert.html) from the [transofrmers library](https://huggingface.co/transformers/index.html). With BPE we observed a more rapid decrease in validation loss. 

# Data Augmentation 

To improve the corrector, more diverse examples of missplellings are required. One powerful method is to train a reverse "corruptor" model that will introduce errors similr to those made by humans to given clean examples. To do this, we reverse source and target during training. Now the trained "corrputor" model an be used to create artificial typo corpus of infinite size (in theory). For more information about this idea, refer to [this blog](http://www.realworldnlpbook.com/blog/unreasonable-effectiveness-of-transformer-spell-checker.html). 

## How To Use

Create a conda environment and install required packages
```
conda env create -f env.yml
conda activate transformer_env
```
Read github data and split into train, validation and test sets
```
python3 process_data.py
```

Start training of the typo corrector model, set encoding and embedding types accordingly
```
python3 train.py -encoding_type char -embedding_type learned 
```

