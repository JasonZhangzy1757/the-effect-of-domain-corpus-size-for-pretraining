# NationalSecurityBERT
XSC224u Final Project - Pre-training BERT on national security domain corpus

## Project Framework
### 1.Acquire natsec corpus for model pretraining
 - Ali will start scraping National Security archives to get an initial dataset for pre-processing testing - try to only include documents 1990 and later
 - Chris to make contact with repository POCs for large data pulls 

### 2.Decide on data preprocessing steps
How to tokenize
 - Jason to connect with Ankit regarding how to build a vocabulary and if not using that approach what would he suggest. 
 - Chris to connect with MSR author of PubMedBERT for ideas on this topic.  

### 3.Acquire training resources for model (GPUs)
 - Chris to take admin step to secure 16+ GPUs for pretraining
 - Jason to set up Github repo for Project 
    americanthinker  - Chris’ Github handle
    XXX - Ali's Github handle

### 4.Decide on model comparison measures
 a) Multi-class classification
 - Need to pre-process this dataset
 - Can score baseline on BERT after step i is completed. 
 b) Perplexity scores
 - Jason looking into how to measure/use Perplexity
 c) NER
 - Chris to start building NER dataset through Microsoft Cognitive Services model
 d) Q&A
 - ??? Need to connect with Prof. Potts about this idea

### 5.Pretrain models using three paradigms:
 a) BERT from scratch on natsec
 b) BERT with mixed domain
 c) BERT using exBERT process
 - Jason to assess feasibility of including the exBERT approach.

### 6.Finetune models on supervised datasets
 a) Classification task
 b) NER task
 c) Q&A task 

### 7.Benchmark models on testing sets 
 a) BERT from scratch
 b) BERT general with natsec
 c) BERT with exBERT


## Expected Timeframe

* March 11: Tokenization/Vocabulary scheme finalized + pretraining data secured
* March 20: Model pretraining completed (for at least one model approach)
* March 30: All experiments completed
* April 3: Final paper due
* April 8: Code completed

