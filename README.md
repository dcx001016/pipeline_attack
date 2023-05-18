# pipeline_attack

This is dcx's attack code against large model pipeline training.

## Setup:

* Create environment:
```
conda create -n acsgd python=3.8
conda activate acsgd
```
* Install PyTorch env:
```
pip3 install torch==2.0.1

# This depends on your CUDA version.
pip3 install cupy-cuda112==8.6.0
```
* Other dependencies:
```
pip3 install datasets==2.2.2
pip3 install transformers==4.19.2
pip3 install sentencepiece==0.1.99 # required by deberta
```