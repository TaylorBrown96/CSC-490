# CSC-490 | Real-Time Medical Documentation​
The audio processing component of this project focuses on transcribing and analyzing spoken medical reports to enhance real-time procedure documentation. It utilizes natural language processing (NLP) to extract structured information from medic reports, ensuring accurate, time-stamped records of interventions.

#### Key features include:
- **Speech Recognition (SR):** Converts spoken words into text while handling background noise and specialized terminology.
- **Word Error Rate (WER) Analysis:** Evaluates transcription accuracy by comparing predictions to ground-truth data.
- **Integration with Video Processing:** Aligns audio transcripts with detected medical actions for multimodal documentation.

By fusing audio and visual data, this system improves automated medical logging in high-stress environments such as battlefield scenarios.



## Home Grown Dataset Collection

### Download the Provided Code

Place the `Dataset_Audio_Collection.py` file into an empty folder on your desktop.

### Install Dependencies

Before running the script, install the required dependencies by executing the following command:

```bash
pip install -r requirements.txt
```

This will ensure that all necessary packages are installed.

### Run the Script
_If the needed packages are required, they will install automatically._

1. Enter your name when prompted.  (**_No spaces. Just your first name._**)
2. A window will appear showing the current sentence.  
3. Click **Record** to start recording, then **Stop** to end.  
4. Click **Next** to move to the next sentence.

>[!NOTE]
>If you fumble your words when recording, you can click stop and then re-record the sentence. (**_It will automatically overwrite the old recording._**)

### To Check the Output

- Recorded audio is saved in the `audio` folder as `YourName_Sentence_X.wav`.
- The matching transcript is saved in the `transcripts` folder as `YourName_Sentence_X.txt`.
  
>[!NOTE]
>### Extracting The _Archive
>If you would like to use our repos dataset rather than gather your own, navigate to the `_Archive` folder and run the `assembleDataset.py` script and then move the `audio` and `transcripts` folders to the top level directory.


## CSV Creation Tool
To generate a set of CSV files from the collected dataset, run the following command:

```bash
python create_csv.py
```

This script will scan the `audio` and `transcripts` directories and generate CSV files named (`train.csv`, `test.csv`, and `val.csv`) that map the audio files to their corresponding text transcripts.

#### Editable Parameters
- `--audio_dir` (Default = `"./audio"`) – Path to the directory containing audio files.
- `--transcript_dir` (Default = `"./transcripts"`) – Path to the directory containing transcript files.
- `--train_csv` (Default = `"train.csv"`) – Name of the output CSV file for the training set.
- `--test_csv` (Default = `"test.csv"`) – Name of the output CSV file for the test set.
- `--val_csv` (Default = `"val.csv"`) – Name of the output CSV file for the validation set.
- `--train_split` (Default = `0.7`) – Proportion of data to use for the training set.
- `--test_split` (Default = `0.2`) – Proportion of data to use for the test set.
- `--val_split` (Default = `0.1`) – Proportion of data to use for the validation set.
- `--seed` (Default = `42`) – Random seed for reproducibility.


## Training and Evaluation Scripts

### Training the Model

Run the training script with the following command:

```bash
python train.py 
```

This will train the model using the dataset specified in `train.csv` and `test.csv`, saving the fine-tuned model in `whisper-small-finetuned` while also saving every 100 steps.

#### Editable Parameters
- `--train_csv` (Default = `"train.csv"`) – Path to the training dataset.
- `--eval_csv` (Default = `"test.csv"`) – Path to the evaluation dataset.
- `--output_dir` (Default = `"./whisper-small-finetuned"`) – Directory to save the fine-tuned model.
- `--num_train_epochs` (Default = `10`) – Number of training epochs.
- `--train_batch_size` (Default = `4`) – Training batch size per device.
- `--eval_batch_size` (Default = `4`) – Evaluation batch size per device.
- `--learning_rate` (Default = `1e-5`) – Learning rate.
- `--save_steps` (Default = `100`) – Save checkpoint every X steps.
- `--eval_steps` (Default = `100`) – Run evaluation every X steps.
- `--logging_steps` (Default = `50`) – Log every X steps.
- `--gradient_accumulation_steps` (Default = `1`) – Number of gradient accumulation steps.
- `--max_audio_length` (Default = `30` seconds) – Maximum audio length for truncation.

### Evaluating the Model

To evaluate the trained model, execute:

```bash
python eval.py --model_dir ./whisper-small-finetuned --metrics_dir ./metrics/eval
```

This script will compute the evaluation metrics and save the results in `eval_metrics`.

#### Editable Parameters
- `--model_dir` (Default = `"./whisper-small-finetuned"`) – Path to the directory containing the fine-tuned model.
- `--val_csv` (Default = `"val.csv"`) – Path to the validation CSV file.
- `--num_samples` (Default = `-1`) – Number of samples to evaluate from `val.csv` (`-1` evaluates all samples).
- `--metrics_dir` (Default = `"metrics"`) – Directory where metrics and predictions will be saved.


## Metrics Scripts

To generate training metrics:

```bash
python train_export_metrics.py --logdir whisper-small-finetuned/logs --output_dir ./metrics/train/graphs/
```

To generate graph for evaluation results:

```bash
python eval_export_metrics.py --predictions_csv ./metrics/predictions.csv --output_dir ./metrics/eval/graphs
```
---
### train_export_metrics.py  

#### Extracted Metrics  
- **Loss** – Measures how well the model’s predictions match the target values. Lower values indicate better performance.  
- **Accuracy** – Represents the percentage of correctly predicted outputs. Higher values indicate better performance.  
- **Learning Rate** – Tracks how the learning rate changes during training, often adjusted dynamically.  
- **Gradient Norm** – Measures the magnitude of model gradients, helping diagnose exploding/vanishing gradients.  
- **Evaluation Metrics (e.g., Validation Loss, Accuracy)** – Similar to training metrics but computed on validation data.  

#### What the Script Produces  
- Line plots of each metric against training steps.  
- Multiple event files are plotted separately to compare different runs.  
- Each metric (e.g., loss, accuracy) gets its own figure saved as a `.png`.  

### eval_export_metrics.py 

#### Extracted Metrics  
- Computes **Word Error Rate (WER)** for each sample in a CSV file containing ground-truth transcriptions and model predictions.  
- **WER Calculation:**  $$WER=\(\frac{S + D + I}{N}\)$$   
  Where:  
  - \( S \) = Number of substitutions (wrong words)  
  - \( D \) = Number of deletions (missing words)  
  - \( I \) = Number of insertions (extra words)  
  - \( N \) = Total number of words in the reference transcript  

#### What the Script Produces  
- A **bar chart** showing WER for each sample.  
- A **dashed red line** indicating the **average WER** across all samples.  
- A summary of the overall **average WER** printed in the console.  
