This repository contains code used for creating a model to predict which hero will die in the near future in a DOTA 2 match.
The input of the model is a large number of features (2800+): for example the position, health, mana of the heroes, the cooldowns on their abilities and items, the health of the towers, etc. 
The output of the model is a probability estimate of each player dying in the next 5 seconds.

The model is described in the paper:
Adam Katona, Ryan Spick, Victoria Hodge, Simon Demediuk, Florian Block, Anders Drachen and James Alfred Walker
Time to Die: Death Prediction in Dota 2 using Deep Learning, (accepted) IEEE Conference on Games (CoG) 2019
https://arxiv.org/abs/1906.03939

```
@article{katona2019time,
  title={Time to Die: Death Prediction in Dota 2 using Deep Learning},
  author={Katona, Adam and Spick, Ryan and Hodge, Victoria and Demediuk, Simon and Block, Florian and Drachen, Anders and Walker, James Alfred},
  journal={arXiv preprint arXiv:1906.03939},
  year={2019}
}
```


The repository contains:
 - The Parser (java program using the clarity parser)
 - Parse scripts and job files (SLURM job)
 - Preproces scripts (python scripts transfroming raw data to a dataset suitable for machine learning)
 - Model and train scripts (PyTorch model and training scripts)

Warning: this is research code, it is not written for reuse. There are hardcoded paths, some unused code...


The Parser:

The purpuse of this program is to take a replay file, and save a timeseries of a set of attributes.
input: 
- dem file,  Binary format used to store replays for DOTA 2
outputs:
- attribute csv: first row is the name of the attributes, the rest are datapoints
- life state csv: life state is recorded at full resolution, so it is in a different file.

Credit:
The parser is built upon clarity: https://github.com/skadistats/clarity
Some of the code is taken from open dota: https://github.com/odota/parser/blob/master/src/main/java/opendota/Parse.java

Usage:
mvn -P my_processor package     //(this should download dependencies like clarity)
java -jar target/my_processor.one-jar.jar /path/to/demfile /path/to/output_folder


Parse scripts:
cluster_scripts/   These files call the parser and the preprocessor on the whole dataset.
testing_parse_whole_match.py   Parse a whole match for testing.


The Preprocess scripts

- preprocess.py  
Contains read_and_preprocess_data(game_name,sample=True) function. Takes a csv file of game object attribute timeseries, and outputs a pandas table with a time series of features and labels.
Sampling: Since the data is not balanced (there are much more timesteps when noone is dying than someone is dying) we downsample negative examples.
To save memory some features are added at a post process step (for example the one hot encoded hero ID)
The post processing takes place after we loaded a batch of data (for training or predicting)

- data_loader.py
Contains functions:
To normalize the data 
To load the data (mini batch)
To select a subset of features


Model and train scripts:

model.py  Contains the pytorch model class. The model is a feed-forward neural network with weight sharing between the features of the 10 heroes.
train.py  Contains train loop and plotting code. 
train_scripts/run_experiment.py  Code for hyperparameter search (random search)

test_model.py  Code to predict and plot whole match
predict_on_testset.py  Predict on the whole test set, and save predictions to files


test_score.ipynb  Analyses predictions
analyze_results.ipynb  Analyse the results of the hyperparameter search

