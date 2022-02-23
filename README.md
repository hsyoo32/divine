# DIVINE
- The repository is the implementation of [DIVINE](https://doi.org/10.1145/3488560.3498470)
Directed Network Embedding with Virtual Negative Edges
{Hyunsik Yoo, Yeon-Chang Lee,} Kijung Shin, Sang-Wook Kim
15th ACM International Conference on Web Search and Data Mining (WSDM), 2022

## Requirements
- python 35
- scikit-learn==0.21.3 (specific version for STNE)
- numpy
- tqdm
- networkx
- pandas

### For WRMF:
- tensorflow==1.13.1
- Cython

go to '*./NeuRec*' and compile the evaluator of cpp implementation with the following command line:
```bash
python setup.py build_ext --inplace
```

### For STNE:
- texttable

### For SIDE:
- OS: Only Mac OS and Linux are available for this code.
- tensorflow==1.1

#### (Please refer to the author's original README.md for more details of WRMF, STNE, and SIDE.)

## Usage

```bash
python divine.py --dataset GNU --emb_algo stne --lp_task LP-uniform --num_embed 128 --vne_algo wrmf --theta 0.5 --selection_strategy local
```
- vne_algo: Method for inferring degree of negativity
- selection_strategy: Strategey for selcting VNEs
- Theta: hyperparamter for determining the number of VNEs to be added
- dataset: input (unsigned) network
- emb_algo: (signed) network embedding method for learning node embeddings
- num_embed: dimensionality of embeddings
- lp_task: link prediction task type
