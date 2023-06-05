## TL;DR

## Large Language Model (LLM)

+ Takes text inputs and predicts next words
+ Q&A - Use existing knowledge base to provide contextual response
+ Simulates Human like response

### One hot encoding 

One hot vector can be used to represent words in numerical values e.g. Corpus of five words *Gold, Silver, Salt, Diamond,* and *Coal* can be represented as 

<img width="400" alt="Vector" src="https://github.com/rajashtani/myGPT/assets/71159892/3d28fe23-ec28-4de3-95e2-e11a25fb0c3b.png">

The number of dimensions in the one hot vector is equal to the number of unique words in the corpus. 

One hot endoing doesn't get words any inherent meaning. It only tells the **Artificial Neural Network** that this unique word exists without telling it what it means not its relevance to other words. 

### Vector Visualization

Word vectors place words with similar meaning and/or relevance closer togetner in the vector space.
<img width="400" alt="2DVector" src="https://github.com/rajashtani/myGPT/assets/71159892/b56450d3-f65f-4f88-acaa-1aa178e858d0.png" >

One can determine the relevance of two words by computing the euclidean vector distance between thier respective vectors.

### High Dimension Vector

Words used in a similar context can be labeled to create multi dimension vector

<img width="400" alt="nDVector" src="https://github.com/rajashtani/myGPT/assets/71159892/8c9d6ee3-6da3-46ed-84b7-1640cffb474f.png" >

#### 3D Plot

<img width="400" alt="nDVector" src="https://github.com/rajashtani/myGPT/assets/71159892/62671cfc-6b85-4a15-b219-b2f8d1e3eda4.png" >

### Word Embedding

* Text Representation in n-dimensional space
* Similar meanings or occurring in similar contexts are closer to each other in the high-dimensional space aka distributed word representations 
depends on factors such as the size of the training dataset, the computational resources available
* There are NN Models to generate distributed word representations eg Word2vec, GloVe, BERT
* Vector Databases
  * Pinecone
  * Faiss
  * Weaviate
  * Chroma DB

## Traning your model

It is expensive and 
<img width="9050" alt="model" src="https://github.com/rajashtani/myGPT/assets/71159892/478479c6-0e84-442f-b22c-23dbdc4c799c.png" >

https://www.technologyreview.com/2019/06/06/239031/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes


## What do you need 
* Python 
* Open Source distribution
  * Huggingface Word embedding 
  * Langchain
* Vector Database 
  * Chroma DB
* Open Source LLM Models from GPT4ALL
  * LLama 
  * GPT-J
  * MPT
* Streamlit for rapid UI devlopment

And your documents 

