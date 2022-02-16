# SIDE: Feature Learning in Signed Directed Networks

## Authors
- Junghwan Kim (kjh900809@snu.ac.kr), Seoul National University
- Haekyu Park (hkpark627@snu.ac.kr), Seoul National University
- Ji-Eun Lee (dreamhunter@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

## Basic information
- Version: 1.0
- Date: 11 Jan 2018
- Main Contact: Junghwan Kim <br />
- This software is ***free of charge for research purposes***. For commercial purposes, please contact the authors.

## Overview

This package is an implementation of SIDE as proposed in the paper:
[SIDE: Feature Learning in Signed Directed Networks](https://datalab.snu.ac.kr/side/resources/SiDE.pdf).
SIDE algorithm learns a vector representation for nodes in (un)signed, (un)directed networks.
Please refer to the homepage of the project:
https://datalab.snu.ac.kr/side/.

## Requirements
 - python 3.5
 - numpy (We recommend you to use [Anaconda](https://anaconda.org/anaconda/numpy).)
 - [tensorflow (CPU)](https://github.com/tensorflow/tensorflow)
 - OS: Only Mac OS and Linux are available for this code.

## Installation of the source code
 - The source code is available at [the project homepage](https://datalab.snu.ac.kr/side/).
 - Or you can download the code through this [link](https://datalab.snu.ac.kr/side/resources/side.zip).

## Usage
### Input and output
- Input <br /> <br />
  Input graph files can be one of the following three file types: .mat, .gml and edgelist file.

  1. **.mat** file is a matlab adjacency matrix file. The data should be in the variable network.
  2. **.gml** file is a network data file supported by Graphlet, Pajek, yEd, LEDA and NetworkX.
  3. **edgelist** file is a list of edges formated as follows: node1, node2, and weight. 
    We do not specify file extension for edgelist file.
    Sign information is represented as weight in edgelist file.
  4. You can set the file type of input in an argument option ``--ftype``.
  
     * For **.mat** formatted input, please execute a command ``python main.py --ftype mat``.
     * For **.gml** formatted input, please execute a command ``python main.py --ftype gml``.
     * You do not need to give ``--ftype`` option for other file types.
 

 You can set an input graph file path as folllows. 
 
  1. You can give an argument with ``--network-file`` option to clarify what graph data is used as an input.
  2. For example, ``python main.py --network-file <A_PATH_OF_INPUT_GRAPH_DATA_FILE>``.
  3. Unless any ``--network-file`` option is given, a default dataset 
    [Read Highland tribes data](http://www.analytictech.com/ucinet/help/hs5101.htm) is applied.

- Output <br /> <br />
  The random walk simulation is saved in _./walk_ folder.
  The output file is saved in _./output_ folder. There are four output files: emb, emb2, bias and vocab files.
  
  1. emb and emb2 files contain target and neighborhood embedding for each node, respectively.
  2. bias files contain positive-in, negative-in, positive-out, negative-out bias for each node in the provided order.
  3. The order of node in the above three files is different from id-order of the graph to permit non-integer id in the graph.
     The order of node is saved in the file vocab file.

- Options <br /> <br />
  The list of command line options is available if you type: ``python main.py --help``.


### How to run
 1. Complie .cc codes
   * For mac users, please execute a command: ``make mac``.
   * For linux users, please execute a command: ``make linux``.
 2. Give parameters that you want (Optional)
    * We present a table of brief explanations on main parameters. 
     We recommend you to compare the following list and _Table of symbols_ (Table 1) 
     in our [paper](https://datalab.snu.ac.kr/side/resources/SiDE.pdf).

      Parameter   | Argument option        | Explanation 
      ------------|------------------------|-------------
      $$w$$       | --num-walks            | number of walks per node
      $$l$$       | --walk-length          | number of steps per walk
      $$k$$       | --window-size          | size of context window
      $$n$$       | --neg-sample-size      | number of noise sampling
      $$d$$       | --embed-dim            | dimension of embedding
      $$\lambda$$ | --regularization-param | regularization parameter

   * You can set main parameters as you want. 
     You can set the parameters in a command arguments as follows: <br />
     ``python main.py --<ARGUMENT_OPTION> <A_VALUE_YOU_WANT_SO_SET>``.
   * For example, if you want to set _w_ as 80 and _l_ as 40, you can execute a command: <br />
     ``python main.py --num-walks 80 --walk-length 40``.

   * You can see details on more optional parameters by execute a command ``python main.py --help``. 
   Then you can see the following list of the arguments.
     ```
     optional arguments:
     -h, --help               show this help message and exit
     --dataset                Dataset name
     --network-file           Input network file
     --ftype                  Input file format
     --sep                    Input separator
     --comment                Input file comment indicator
     --walk-path              Output path for walk
     --embed-path             Output path for embedding
     --directed               directed graph
     --weighted               weighted graph
     --signed                 signed graph
     --deg1                   parametrized walk
     --subsample              threshold ratio of the highest degree of nodes to be subsampled
     --mem-max-walks          Maximum number of walks to process in memory (default: 100000000)
     --num-walks              Number of walks per node (default: 80)
     --walk-length            Length of walks (default: 40)
     --window-size            Number of context to train (default: 5)
     --embed-dim              Dimension of embedding (default: 128)
     --neg-sample-size        Number of negative samples to train (default: 20)
     --regularization-param   Regularization parameter (default: 0.01)
     --batch-size             Size of batch to train in 1 iter (default: 16)
     --learning-rate          Learning rate (default: 0.025)
     --clip-norm              Gradient norm clipping (default: 5.0)
     --epochs-to-train        Number of epochs to train (default: 1)
     --summary-interval       Interval to update summary (default: 100)
     --save-interval          Interval to save checkpoint (default: 1000)
     ```


## A demo example of SIDE ([Read Highland tribes dataset](http://www.analytictech.com/ucinet/help/hs5101.htm))
We provide demo of our package applied to Read Highland tribes data.
To run SIDE demo, please execute the following command at the project home directory: ``python main.py``.
***Do not change the default arguments defined in main.py.***
For other example datasets, please refer to [the project homepage](https://datalab.snu.ac.kr/side/) 
and download the other datasets used in our paper.

* Demo input is ([Read Highland tribes dataset](http://www.analytictech.com/ucinet/help/hs5101.htm)) in _./graph/out.ucidata-gama_.
* Random walk sampling on the input graph is saved in a file _./walk/gama.walk_.
* Embedding learned from the random walk is saved in a directory _./output/_. The followings are name of the four output files.
  * target embedding: _gama.emb_
  * neighbor embedding: _gama.emb2_
  * bias factors: _gama.bias_
  * vocab files: _gama.vocab_

