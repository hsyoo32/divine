# DIVINE
- The repository is the implementation of [DIVINE]()
Directed Network Embedding with Virtual Negative Edges
{Hyunsik Yoo, Yeon-Chang Lee,} Kijung Shin, Sang-Wook Kim
15th ACM International Conference on Web Search and Data Mining (WSDM), 2022

## Requirements
- python 35
- scikit-learn==0.21.3
- numpy
- tqdm

### For WRMF:
- python 35
- Cython
go to '*./NeuRec*' and compline the evaluator of cpp implementation with the following command line:
```bash
python setup.py build_ext --inplace
```
- tensorflow==1.13.1
- pandas

### For STNE:
- python 35
- scikit-learn==0.21.3
- texttable
- networkx
- tqdm

### For SIDE:
- OS: Only Mac OS and Linux are available for this code.
- python 35 
- tensorflow==1.1
- pandas
- networkx

#### Please refer to the author's original README.md for more details of WRMF, STNE, and SIDE.

## Usage

```bash
python divine.py --dataset GNU --emb_algo stne --lp_task LP-uniform --num_embed 128 --vne_algo wrmf --theta 0.5 --selection_strategy local
```
- vne_algo: Method for inferring degree of negativity
- selection_strategy: Strategey for selcting VNEs
- Theta: hyperparamter for determining the number of VNEs to be added
- dataset: input (unsigned) network
- emb_algo: (signed) network embedding method for learning node embeddings
- lp_task: link prediction task type