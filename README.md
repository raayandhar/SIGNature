# SIGNature
Sigmoid Pairwise Learning for AI-Generated Text Detection

## Setup 
You must first obtain the `SIGNature-data` by unzipping the attached zip file. This contains the data that is used to build that databse for the TFIA inference time method using KNNs. It also contains the evaluation splits and along with the CS162 dev sets and CS162 ethics dev set formatted appropriately for evaluation.

Replace the path at the top of `script/test.sh` with your path to the `SIGNature-data` and the model checkpoint:
```python
DATA_PATH="/path/to/your/SIGNature-data"
Model_PATH="/path/to/your/model_best.pth"
```

## Evaluation

The `script/test.sh` contains many commented out commands that runs various evaluations of our model on diferent evaluation splits. Uncomment the desired run, and run the command:

```bash
bash script/test.sh
```

To begin evaluation. The evaluation begins by embedding a smaller split of the training data to serve as the databse for the KNN inference, so this may take a while. All experiments in our report were done on 1x L40s GPU, occupying around 11 GB of VRAM, depending on the size of the evaluation split.

