## TL;DR

## 

### One hot encoding 

One hot vector can be used to represent words in numerical values e.g. Corpus of five words *Gold, Silver, Salt, Diamond,* and *Coal* can be represented as 

<img width="400" alt="Vector" src="https://github.com/rajashtani/myGPT/assets/71159892/3d28fe23-ec28-4de3-95e2-e11a25fb0c3b.png">

The number of dimensions in the one hot vector is equal to the number of unique words in the corpus. 

One hot endcoing doesn't give words any inherent meaning. It only tells the **Artificial Neural Network** that this unique word exists without telling it what it means not its relevance to other words. 

### Word Vectors 

Word vectors place words with similar meaning and/or relevance closer togetner in the vector space e.g. in our corpus precious metals gold and silver can be placed closer in two dimension vector 

<img width="400" alt="2DVector" src="https://github.com/rajashtani/myGPT/assets/71159892/b56450d3-f65f-4f88-acaa-1aa178e858d0.png" >

One can determine the relevance of two words by computing the euclidean vector distance between thier respective vectors.

**High Dimension Vector**

Words used in a similar context are be labeled to create multi dimension vector

<img width="400" alt="nDVector" src="https://github.com/rajashtani/myGPT/assets/71159892/8c9d6ee3-6da3-46ed-84b7-1640cffb474f.png" >

**3 Dimension Vector Visualization**

<img width="400" alt="nDVector" src="https://github.com/rajashtani/myGPT/assets/71159892/62671cfc-6b85-4a15-b219-b2f8d1e3eda4.png" >

### Word Embedding

* Text Representation in n-dimensional space
* Similar meanings or occurring in similar contexts are closer to each other in the high-dimensional space aka distributed word representations 
depends on factors such as the size of the training dataset, the computational resources available
* Word embedding Algorithms
  * Embedding Layer
  * Word2Vec
  * GloVe
  * BERT
* Word Embedding can be stored in Vector Databases e.g.
  * Pinecone
  * Faiss
  * Weaviate
  * Chroma DB

### Large Language Model (LLM)

+ Takes text inputs and predicts next words
+ Q&A - Use existing knowledge base to provide contextual response
+ Simulates Human like response
+ Populor models
  * GPT (OpenAI)
  * BERT (Google)
  * BART (Facebook)

## Traning models

Training LLL model requires large dataset and compute resource. 

<img width="800" alt="Training" src="https://github.com/rajashtani/chicAssistant/assets/71159892/2504cc21-b711-4503-9160-b1c60a64478e" >

These models are costly to train and develop, both financially, due to the cost of hardware and electricity or cloud compute time, and environmentally, due to the carbon footprint required to fuel modern tensor processing hardware.
As MIT Technology Review training just one AI model can emit more than 626,00 pounds carbon dioxide.
<img width="9050" alt="model" src="https://github.com/rajashtani/myGPT/assets/71159892/478479c6-0e84-442f-b22c-23dbdc4c799c.png" >

https://www.technologyreview.com/2019/06/06/239031/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes


## App features

* Huggingface Word embedding(s) 
* Langchain
* Chroma DB Vector Database
* Open Source LLM Models from GPT4ALL
  * LLama 
  * GPT-J
  * MPT
* Streamlit
* run on mac pro

And lots of relevant documents 

