# Building Transformers from scratch

Translation Model : Model will be able to translate one language to another

- building transformer from scratch using pytorch, building mdoel, build code to train, inference, and visualization of atttention scores


Module 1 - Input Embedding
- Takes in input and turns into Input Embedding
- Input Embeddings allows to convert original sentence into a vector of 512 dimensions

Original Sentence (tokens) -> Input IDs (position in the vocabulary) -> Embedding (vector of size 512)

Module 2 - Positional Encoding 
- Original sentence mapped to list of vectors by embeddings layer
- Now we want to convey to the model the position of each word inside the sentence and this is done by adding another Vector of the same size as the embedding so of size 512 that includes some special values that tells the model that this particular word occupies this position in the sentence 
- So we create these vectors called the Positional Embedding and we will add them to the embedding

Module 3 - Layer Normalization
- If you have a batch of n items
- each item will have some features
- for each item in the batch we calculate a mean and the variance independently from the other items of the batch and then we calculate the new values for each of them using their own mean and variance
- we also introduce parameters gamma and beta, one is multiplicative (gamma) and one is additive (beta) because we want the model to have the possiblity to amplify these values when we need the value to be amplified so the model will learn to multiply this gamma by these values in such a way to amplify the values that it wants to be amplified

Module 4 - Feed Forward
- fully connected layer that the model uses both in encoder and decoder
- this layer is two matrices that are multiplied by x one after another with a ReLU activation in between and a bias
- matrices d_model to d_ff and second is d_ff to d_model
- d_model = 512 and d_ff = 2048

Module 5 - Multi-Head Attention (most important and interesting)
- multi-head attention takes input of the encoder and uses it 3 times
- 1. query, 2. key, 3. value
- duplication of the input 3 times or same input applied 3 times
- we have our input sequence, length by d_model and transform into 3 matrices Q,K,V which are exactly the same as the input in this case for the encoder then multiply by W^Q, W^K, W^V results in new matrix of dimension sequence by d_model then split into h_matrices smaller bc number of head we want for this multihead attention, spliting along embedding dimension, each head will have access to the full sentence but a different part of the embedding of each word
- apply attention to each of the smaller matrices giving smaller matrices and combine then back(concatenation) of head_1 up to head_h and finally multiply by W^O to get multihead attention output, which is a matrix that has the same dimension as the input matrix

will also consider multibatching when we work with multiple sentences not just one sentence

Module 6 - Residual Connection (skip connection)
- techniques to help deep neural networks train better
- when you stack many layers in deep network problems such as vanishing gradients, hard to train, and degradation
- solution is instead of just passing data through layers, add a shortcut that skips the layer
- this helps the fradient flow directly backward through the skip connection
- if a layer dosen't need to do anything it can just learn to output zeros
- each layer learns what to add to the input, not replace it
In transformer architecture
 - residual connections appear twice in each encoder/decoder block

analogy: think of building a tower without residual connections as each floor built from scratch if the bottom floors are bad, top floors collapse, making it hard to fix problems at the bottom. Thus with residual connections each floor ADDS improvements to what's below, the ground floor is always accessible, can always "go back" to the original input and if a floor dosen't help, it can just do nothing

Module 7 - Encoder
- block Encoder is repeated N times where output of the previous is sent to the next one and the output of the last is sent to the decoder
- create this block with one Multi-Head Attention, two Add & Norm, one Feed Forward

Module 8 - Decoder
- 1 Masked Mult-Head Attention, 1 Multi-Head Attention, 1 Feed Forward, 3 Add & Norm for N  numbers of the decoder blocks, and 3 skip connections
- our current implemtation of multi-head attention class already takes in consideration of the mask so we dont need to redefine
- the middle Multi-head attention block takes in two encoder inputs(key and value) and one (query) coming from the decoders, which is called cross attention as we match two different objects together and matching them to calculate relationship between them opposite of self attention which inputs itself/one object for all 3 

Module 9 - Linear Layer
- expect to have output sequenced by d_model
- we want to map the words back into the vocab so we use the linear layer to convert the embedding into the position of the vocabulary
- in this case call it projecting layer

Tokenizer
- comes before the input embeddings
- goal is to create token/split sentence into single words BPE, wordlevel, sub word level, word part tokenizer, alot of differet ones
- we are going to use wordlevel, split sentence by space, each space defines boundary of a word
- each word mapped to a number, job is to build vocabulary of the numbers 
- can also create special tokens for transformer (padding, start of sentence, end of sentence)

self attention
- self attention is permutation invariant
- self attention requires no parameters up to now interaction between words has been driven by their embedding and the positional encodings. this will change later
- we expect values along the diagonal to be the highest
- if we dont want some positionas to interact, we can always set their value to -infinity before applying softmax in this matrix and the model will not learn those interactions

causal_mask
- 