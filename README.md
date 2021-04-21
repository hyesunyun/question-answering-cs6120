# CS 6120 NLP: Models for Question Answering

The base code was cloned from https://github.com/minggg/squad
This base code is for Stanford CS 224N Final Project of building a QA system.

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

### QANet Model

### BERT Model

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
