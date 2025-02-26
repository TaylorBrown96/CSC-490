# CSC-490

### Home Grown Dataset Collection

#### Download the provided code

Place the `Dataset_Audio_Collection.py` file into an empty folder on your desktop.

#### Run the Script
_If the needed packages are required they will install_

1. Enter your name when prompted.  (**_No spaces. Just your first name._**)
2. A window will appear showing the current sentence.  
3. Click **Record** to start recording, then **Stop** to end.  
4. Click **Next** to move to the next sentence.

>[!NOTE]
>If you fumble your words when recording you can click stop and then re-record the sentence. (**_It will automatically overwrite the old recording_**)

#### To Check The Output

- Recorded audio is saved in the `audio` folder as `YourName_Sentence_X.wav`.
- The matching transcript is saved in the `transcripts` folder as `YourName_Sentence_X.txt`.


---
### TODO
- Write instructions for csv creation tool, train and eval scrips
- Validate the train and eval scripts work as intended
- Clean up the code after validation


---
### Commands
`python export_metrics.py --logdir whisper-small-finetuned/logs --output_dir metrics`<br>
`python eval.py --model_dir ./whisper-small-finetuned --val_csv val.csv --num_samples -1 --metrics_dir eval_metrics`<br>
`python eval_export_metrics.py --predictions_csv ./eval_metrics\/predictions.csv --output_dir ./eval_metrics/graphs`
