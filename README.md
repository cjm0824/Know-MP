# Know-MP
Our goal is to improve few-shot relation extraction performance by combining meta-learning with prompting. Our proposed model Know-MP assigns template&encoder learning to the meta-learner and label words learning to base-learners, resplectively. We conduct extensive experiments on the widely-used relation extraction datasets: FewRel 1.0 and FewRel 2.0.

# Dataset
### FewRel 
A dataset for few-shot relation classification, containing 100 relations. Each statement has an entity pair and is annotated with the corresponding relation. The position of the entity pair is given, and the goal is to predict the correct relation based on the context. The 100 relations are split into 64, 16, and 20 for training, validation, and test, respectively. 
### FewRel 2.0
FewRel 2.0 proposes a domain adaptation challenge. Its training set is the same as FewRel 1.0, but the test set is from the biomedical domain and contains 25 relations with 100 instances each.

## Download 
You may download the FewRel 1.0 and FewRel 2.0 training data (JSON file named train_wiki.json or train.json) and validation data (JSON file named val_wiki.json or val.json and JSON file named val_pubmed.json)from https://github.com/thunlp/FewRel/tree/master/data or https://github.com/thunlp/MIML/tree/main/data (same dataset while marking entity positions under different conventions. Our word adopt the latter version.) The FewRel testing set is not publicly available for fair comparison, so you need to visit the benchmark website: https://thunlp.github.io/fewrel.html for testing on test set.

You need to put the source data into the corresponding folder of `Know-MP/data/{benchmark_name}`


# Candidate label words

### Format
+ `data/FewRel/P-info.json` provides for each relation, a list of alias, serving as candidate words. (from https://github.com/thunlp/MIML/tree/main/data )

+ `candidate_ebds.json` contains candidate word embeddings of each class. (you may run `data/word2ebd.py` to obtain candidate embeddings)


# Code
+ `train.py` contains fast-tuning model and meta-traning framework
+ `model.py` contains the overall model architechture.
+ `FewRel2_type.py` contains the entire process of obtaining the entity type of FewRel 2.0 valation set.
+ Run `main.py` to call the above two files and start meta-training.

# Citation
`@article{cui2024knowledge,
  title={Knowledge-enhanced meta-prompt for few-shot relation extraction},
  author={Cui, Jinman and Xu, Fu and Wang, Xinyang and Li, Yakun and Qu, Xiaolong and Yao, Lei and Li, Dongmei},
  journal={Computer Speech \& Language},
  pages={101762},
  year={2024},
  publisher={Elsevier}
}`
  
# Requirements
+ Pytorch>=0.4.1
+ Python3
+ numpy
+ transformers
+ json
+ apex (https://github.com/NVIDIA/apex)
