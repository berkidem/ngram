# ngram

In this module we build the n-gram Occupation Prediction Model. In the process, we learn a lot of the basics of machine learning (training, evaluation, data splits, hyperparameters, overfitting) and the basics of autoregressive language modeling (tokenization, next token prediction, perplexity, sampling). GPT is "just" a very large n-gram model, too. The only difference is that GPT uses a neural network to calculate the probability of the next token, while n-gram uses a simple count-based approach.

Our dataset is that of ~50m work experiences from the Burning Glass Institute profiles dataset, which were split into 10%/10%/80% for test/validation/training. Therefore, our n-gram model will essentially try to learn the statistics of the occupations in these careers, and then generate new careers by sampling from the model.

A great reference for this module is [Chapter 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf) of "Speech and Language Processing" by Jurafsky and Martin. (In fact, I read this chapter as a part of a reading group and remembered Karpathy's repo so I wanted to give it a go.)

Currently, the best "build this repo from scratch" reference is the ["The spelled-out intro to language modeling: building makemore"](https://www.youtube.com/watch?v=PaCmpygFfXo) YouTube video, though some of the details have changed around a bit. The major departure is that the video covers a bigram Language Model, which for us is just a special case when `n = 2` for the n-gram.

## Python version

To run the Python code, ensure you have `numpy` installed (e.g. `pip install numpy`), and then run the script:

```bash
python ngram.py
```

You'll see that the script first "trains" a small occupation-level Tokenizer (the vocab size is 1017 for all 1016 O*NET occupations and the special "end of career" token), then it conducts a small grid search of n-gram models with various hyperparameter settings for the n-gram order `n` and the smoothing factor, using the validation split. With default settings on our data, the values that turn out to be optimal are `n=3, smoothing=0.03`. It then takes this best model, samples 10 occupations from it, and finally reports the test loss and perplexity. Here is the full output:

```
python ngram.py
vocab_size: 1017
Reading data done
seq_len 3 | smoothing 0.03 | train_loss 3.5006 | val_loss 3.6978
seq_len 3 | smoothing 0.30 | train_loss 3.7044 | val_loss 3.8055
seq_len 3 | smoothing 3.00 | train_loss 4.0951 | val_loss 4.1294
seq_len 3 | smoothing 30.00 | train_loss 4.7108 | val_loss 4.7196
seq_len 2 | smoothing 0.03 | train_loss 3.7602 | val_loss 3.7761
seq_len 2 | smoothing 0.30 | train_loss 3.7625 | val_loss 3.7729
seq_len 2 | smoothing 3.00 | train_loss 3.7833 | val_loss 3.7888
seq_len 2 | smoothing 30.00 | train_loss 3.9217 | val_loss 3.9239
seq_len 1 | smoothing 0.03 | train_loss 4.7932 | val_loss 4.7931
seq_len 1 | smoothing 0.30 | train_loss 4.7932 | val_loss 4.7931
seq_len 1 | smoothing 3.00 | train_loss 4.7932 | val_loss 4.7931
seq_len 1 | smoothing 30.00 | train_loss 4.7932 | val_loss 4.7931
best hyperparameters: {'seq_len': 3, 'smoothing': 0.03}
Cutters and Trimmers, Hand
Costume Attendants
Electromechanical Equipment Assemblers
Structural Metal Fabricators and Fitters
Structural Metal Fabricators and Fitters
Structural Metal Fabricators and Fitters
Structural Metal Fabricators and Fitters
Welders, Cutters, Solderers, and Brazers
<END_OF_TEXT>
Pharmacists

test_loss 3.699246, test_perplexity 40.416836
wrote dev/ngram_probs_best.npy to disk (for visualization)
```


The Python code also writes out the n-gram probabilities to disk into the `dev/` folder, which you can then inspect with the attached Jupyter notebook [dev/visualize_probs.ipynb](dev/visualize_probs.ipynb).

## C version

The version code is legacy from Karpathy's work but I plan to update it to work with the labor market data as well in the future. (I'll probably prioritize a transformer model so it might take a while before I circle back to this.)

## TODOs

- Make better
- Make exercises
- Call for help: nice visualization / webapp that shows and animates the 4-gram language model and how it works.
