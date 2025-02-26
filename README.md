# CSC-490

### Home Grown Dataset Collection

#### Download the Provided Code

Place the `Dataset_Audio_Collection.py` file into an empty folder on your desktop.

#### Install Dependencies

Before running the script, install the required dependencies by executing the following command:

```bash
pip install -r requirements.txt
```

This will ensure that all necessary packages are installed.

#### Run the Script
_If the needed packages are required, they will install automatically._

1. Enter your name when prompted.  (**_No spaces. Just your first name._**)
2. A window will appear showing the current sentence.  
3. Click **Record** to start recording, then **Stop** to end.  
4. Click **Next** to move to the next sentence.

>[!NOTE]
>If you fumble your words when recording, you can click stop and then re-record the sentence. (**_It will automatically overwrite the old recording._**)

#### To Check the Output

- Recorded audio is saved in the `audio` folder as `YourName_Sentence_X.wav`.
- The matching transcript is saved in the `transcripts` folder as `YourName_Sentence_X.txt`.

---
### CSV Creation Tool
To generate a CSV file from the collected dataset, run the following command:

```bash
python create_csv.py --audio_dir ./audio --transcript_dir ./transcripts --output_csv dataset.csv
```

This script will scan the `audio` and `transcripts` directories and generate a CSV file (`dataset.csv`) that maps audio files to their corresponding text transcripts.

---
### Training and Evaluation Scripts

#### Training the Model

Run the training script with the following command:

```bash
python train.py --train_csv train.csv --val_csv val.csv --model_dir whisper-small-finetuned
```

This will train the model using the dataset specified in `train.csv` and `val.csv`, saving the fine-tuned model in `whisper-small-finetuned`.

#### Evaluating the Model

To evaluate the trained model, execute:

```bash
python eval.py --model_dir ./whisper-small-finetuned --val_csv val.csv --num_samples -1 --metrics_dir eval_metrics
```

This script will compute the evaluation metrics and save the results in `eval_metrics`.

---
### Metrics Scripts

To export evaluation metrics:

```bash
python export_metrics.py --logdir whisper-small-finetuned/logs --output_dir metrics
```

To generate graphs from evaluation results:

```bash
python eval_export_metrics.py --predictions_csv ./eval_metrics/predictions.csv --output_dir ./eval_metrics/graphs
```

This will generate visualizations of model performance.

---
### Code Cleanup and Validation

- Ensure all scripts (`train.py`, `eval.py`, `export_metrics.py`, etc.) function correctly before running final experiments.
- Refactor code for clarity and efficiency after validation.
- Remove unused variables and redundant code to improve maintainability.

This README provides an overview of dataset collection, training, evaluation, and metric reporting for CSC-490's speech dataset project.
