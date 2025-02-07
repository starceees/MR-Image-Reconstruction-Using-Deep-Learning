import torch
import yaml
from dataloader2d import get_loaders
from unet2dmodel import UNet2D
from metrics import calculate_metrics, aggregate_metrics

def run_inference(model, loader, device):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            metrics = calculate_metrics(outputs, y)
            all_metrics.append(metrics)
    final_metrics = aggregate_metrics(all_metrics)
    print(f"Dice Score: {final_metrics['dice']:.4f}")
    print(f"Jaccard Index: {final_metrics['jaccard']:.4f}")
    return final_metrics

if __name__ == "__main__":
    # Load parameters from YAML.
    with open("parameters.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    loaders = get_loaders(config)
    test_loader = loaders['test']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the UNet model.
    model = UNet2D(in_channels=1,
                   out_channels=config['num_classes'],
                   base_filters=config['base_filters'],
                   bilinear=config['bilinear'])
    model.to(device)
    
    # Load the best model checkpoint.
    checkpoint_path = "checkpoints/best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    run_inference(model, test_loader, device)
