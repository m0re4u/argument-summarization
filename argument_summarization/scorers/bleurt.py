import torch
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtSPTokenizer,
)


class BLEURTPyTorchScorer:
    def __init__(self):
        """
        BLEURT evaluation using a PyTorch implementation.
        """
        self.hf_model_name = "lucadiliello/BLEURT-20"
        self.config = BleurtConfig.from_pretrained(self.hf_model_name)
        self.model = BleurtForSequenceClassification.from_pretrained(self.hf_model_name)
        # Load BleurtSPTokenizer directly to avoid a warning from HuggingFace about mismatch in tokenizer names
        self.tokenizer = BleurtSPTokenizer.from_pretrained(self.hf_model_name)
        self.model.to("cuda")
        self.model.eval()

    def score(self, references, candidates):
        with torch.no_grad():
            inputs = self.tokenizer(
                references, candidates, padding="longest", return_tensors="pt"
            )
            inputs.to("cuda")
            res = self.model(**inputs).logits.flatten().tolist()
        return res
