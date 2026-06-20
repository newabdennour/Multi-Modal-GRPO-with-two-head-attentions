import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mgrpo.config import CLASSIFIER_PATH, N_MODALITIES, NORMAL_MODALITY, GROUP_SIZE, USE_TWO_ATTENTION_HEADS
from mgrpo.models.generator import module_device, to_device

class ModalityRouter:
    def __init__(self, generation_head=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_head = generation_head
        try:
            self.cls_tok = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH).to(self.device)
            self.classifier.eval()
            for p in self.classifier.parameters():
                p.requires_grad_(False)
        except Exception as exc:
            self.classifier = None
            self.cls_tok = None
            print(f"Prompt classifier unavailable. Falling back to neutral routing. Reason: {exc}")

    def classifier_distribution(self, user_prompt):
        probs = np.zeros(N_MODALITIES, dtype=np.float32)
        if self.classifier is None or self.cls_tok is None:
            probs[NORMAL_MODALITY] = 1.0
            return probs

        cls_device = module_device(self.classifier)
        inputs = self.cls_tok(user_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = to_device(inputs, cls_device)
        with torch.inference_mode():
            logits = self.classifier(**inputs).logits.float()[0]
            p = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)

        upto = min(N_MODALITIES, len(p))
        probs[:upto] = p[:upto]
        if probs.sum() <= 0:
            probs[NORMAL_MODALITY] = 1.0
        else:
            probs = probs / probs.sum()
        return probs

    def generation_attention_distribution(self, clf_probs):
        if not (USE_TWO_ATTENTION_HEADS and self.generation_head is not None):
            return clf_probs / (clf_probs.sum() + 1e-8), 0.0
        with torch.no_grad():
            dist, gate, _, _ = self.generation_head(torch.tensor(clf_probs, device=self.device))
        return dist.detach().cpu().numpy().astype(np.float32), float(gate.detach().cpu())

    def select_modalities(self, user_prompt, k=GROUP_SIZE):
        clf_probs = self.classifier_distribution(user_prompt)
        gen_scores, gen_gate = self.generation_attention_distribution(clf_probs)
        ranked = list(np.argsort(gen_scores)[::-1])
        chosen = []
        for mid in ranked:
            if int(mid) not in chosen:
                chosen.append(int(mid))
            if len(chosen) == k:
                break
        return chosen, clf_probs, gen_scores, gen_gate
