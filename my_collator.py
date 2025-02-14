class MySpeechCollator:
    """
    Custom collator to handle:
      - Padding "input_features" for Whisper
      - Padding "labels" from the tokenizer
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Separate out the audio features and labels
        input_features_list = [f["input_features"] for f in features]
        label_list = [f["labels"] for f in features]

        # Pad the audio input features
        batch_inputs = self.processor.feature_extractor.pad(
            {"input_features": input_features_list}, 
            return_tensors="pt"
        )

        # Pad the labels
        batch_labels = self.processor.tokenizer.pad(
            {"input_ids": label_list}, 
            return_tensors="pt", 
            padding=True
        )

        # Return dict with properly padded "input_features" and "labels"
        return {
            "input_features": batch_inputs["input_features"],
            "labels": batch_labels["input_ids"],
        }
