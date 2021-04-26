# CS 6120 NLP: Models for Question Answering

The base code was cloned from https://github.com/minggg/squad
This base code is for [Stanford CS 224N Final Project of building a QA system](http://web.stanford.edu/class/cs224n/project/default-final-project-handout-squad-track.pdf).

We explored the task of models predicting answers to questions correctly
when given paragraphs and questions about a particular paragraph as input. 
This in a sense is very simlar to reading comprehension task.

## Dataset

### The SQuAD Data
The Stanford Question Answering Dataset (SQuAD) is a very popular dataset used in question answering.
We used SQuAD 2.0 found on the website https://rajpurkar.github.io/SQuAD-explorer/.

The paragraphs in SQuAD are from Wikipedia.
The questions and answers were crowdsourced using Amazon Mechanical Turk. There are around
150k questions in total, and roughly half of the questions cannot be answered using the provided
paragraph (this is new for SQuAD 2.0). However, if the question is answerable, the answer is a
chunk of text taken directly from the paragraph. This means that SQuAD systems don’t have to
generate the answer text – they just have to select the span of text in the paragraph that answers
the question.

We have the following splits:
- train (105,276 examples/questions): All taken from the official SQuAD 2.0 training set.
- dev (5,951 examples/questions): Roughly half of the official dev set, randomly selected.
- test (25,043 examples/questions): The remaining examples from the official train set.

The dataset can be found in the `data` folder as json format.

## Evaluation

In the official dev and test set of SQuAD dataset, every answerable SQuAD question has three answers
provided – each answer from a different crowd worker. The answers don’t always completely agree,
which is partly why ‘human performance’ on the SQuAD leaderboard is not 100%. Performance
is measured via two metrics: **Exact Match (EM) score** and **F1 score**.

- Exact Match is a binary measure (i.e. true/false) of whether the system output matches
the ground truth answer exactly. Stricter metric.
- F1 is a less strict metric – it is the harmonic mean of precision and recall.
- When evaluating on the dev or test sets, we take the maximum F1 and EM scores across
the three human-provided answers for that question. This makes evaluation more forgiving

Finally, the EM and F1 scores are averaged across the entire evaluation dataset to get the final
reported scores.

## Models

### Baseline Model - Bidirectional Attention Flow (BiDAF)

Based on the paper:
"Bidirectional Attention Flow for Machine Comprehension"
by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
(https://arxiv.org/abs/1611.01603).

Follows a high-level structure commonly found in SQuAD models:
- _Embedding layer_: Embed word indices to get word vectors.
- _Encoder layer_: Encode the embedded sequence.
- _Attention layer_: Apply an attention mechanism to the encoded sequence.
- _Model encoder layer_: Encode the sequence again.
- _Output layer_: Simple layer (e.g., fc + softmax) to get final outputs.

The original BiDAF model uses learned character-level word embeddings in addition to the word-level embeddings.
Unlike the original BiDAF model, the baseline implementation does not include a character-level embedding
layer.

### Improved BiDAF Model

The improved BiDAF model uses the learned **character-level word embeddings** like the original paper.
In addition, the model replaces the original attention layer with Coattention.
The **Coattention Layer** involves two-way attention between the context and the question.
However, unlike BiDAF, Coattention involves a second-level attention computation of attending
over representations that are themselves attention outputs.

Coattention Layer is based on the paper:
"Dynamic Coattention Networks For Question Answering"
by Caiming Xiong, Victor Zhong, Richard Socher
(https://arxiv.org/abs/1611.01604).

We have also created a model which combines the character-level embeddings and the coattention layer.
This is the combined model.

### How to use the BiDAF Model code

Make sure you do the following in the BiDAF folder.
`cd BiDAF`

#### Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code
  
4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.

#### Train

- To train BiDAF Baseline, run `python train.py -n baseline`
- To train BiDAF model with Character-level Embeddings, run `python train.py -n character`
- To train BiDAF model with Coattention Layer, run `python train.py -n coattention`
- To train the combined model, run `python train.py -n combined`

#### Test

Usage:

`python test.py --split SPLIT --load_path PATH --name NAME`

where
- SPLIT is either "dev" or "test"
- PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
- NAME is a name to identify the test run.

**NOTE**: `-n` is an alias for `--name` so `-n` can be used as well.

- To test BiDAF Baseline, run `python test.py --split test --load_path save/train/baseline-01/best.pth.tar -n baseline`
- To test BiDAF model with Character-level Embeddings, run `python test.py --split test --load_path save/train/character-01/best.pth.tar -n character`
- To test BiDAF model with Coattention Layer, run `python test.py --split test --load_path save/train/coattention-01/best.pth.tar -n coattention`
- To test the combined model, run `python test.py --split test --load_path save/train/combined-01/best.pth.tar -n combined`

### QANet Model

### BERT Model
