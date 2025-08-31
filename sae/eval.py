import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import yaml
import sys

from model import Autoencoder, TopK
from train_sae import TrainingConfig, create_model

class SAEEvaluator:
    def __init__(self, sae_path):
        self.reference_model, self.tokenizer = None, None
        
        state_dict = torch.load(sae_path)

        self.sae = Autoencoder.from_state_dict(state_dict)
        self.sae.eval().cuda()

        self.layer_idx = -1
        
    def analyze_vector(self, activation_vector):
        with torch.no_grad():
            if activation_vector.dim() == 1:
                activation_vector = activation_vector.unsqueeze(0)
            activation_vector = activation_vector.cuda()

            latents_pre_act, latents, reconstruction = self.sae(activation_vector)
            
            active_mask = latents > 0
            active_indices = torch.where(active_mask)[1]
            active_values = latents[active_mask]

            reconstruction_error = torch.nn.functional.mse_loss(
                reconstruction, activation_vector
            )
            cosine_similarity = torch.nn.functional.cosine_similarity(
                reconstruction.flatten(), activation_vector.flatten(), dim=0
            )
            return {
                "active_features": active_indices.cpu().tolist(),
                "feature_magnitudes": active_values.cpu().tolist(),
                "reconstruction_error": reconstruction_error.item(),
                "cosine_similarity": cosine_similarity.item(),
                "sparsity": (latents > 0).float().mean().item(),
            }


    def analyze_text(self, text, reference_model_name, layer_idx):
        self.reference_model = AutoModel.from_pretrained(reference_model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(reference_model_name)
        self.layer_idx = layer_idx
        pass

def main():
    sae_path, data_path = sys.argv[1], sys.argv[2]

    evaluator = SAEEvaluator(sae_path)
    
    print(f"Loading data from {data_path}")
    data = torch.load(data_path)
    
    print(f"Analyzing vector")
    results = evaluator.analyze_vector(data)

    print("Results:")
    print(f"Reconstruction error: {results['reconstruction_error']:.8f}")
    print(f"Cosine similarity: {results['cosine_similarity']:.8f}")
    print(f"Sparsity: {results['sparsity']:.8f}")
    print(f"Active features: {results['active_features']}")
    print(f"Feature magnitudes: {[f'{x:.4f}' for x in results['feature_magnitudes']]}")

if __name__ == "__main__":
    main()