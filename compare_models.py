"""
Comprehensive Model Comparison Script

This script performs a complete evaluation and comparison of EfficientNet-B0
and DINOv3 ViT-S/14 models with all required metrics and visualizations.

Usage:
    python compare_models.py
"""

import os
import sys
import torch

from models import create_cnn_model, create_vit_model
from utils import (
    evaluate_model,
    print_evaluation_results,
    plot_comprehensive_evaluation,
    compare_models_comprehensive
)


def main():
    """Compare EfficientNet-B0 and DINOv3 ViT-S/14 models."""
    # Hardcoded configuration
    cnn_weights = 'checkpoints/cnn_best.pth'
    vit_weights = 'checkpoints/vit_best.pth'
    test_dir = 'data/test'
    output_dir = 'results/'
    batch_size = 32
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print(f"\nLoading test data from {test_dir}...")
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        print("Please provide a valid data directory with train/val/test subdirectories")
        sys.exit(1)
    
    from utils.data_loader import create_test_dataloader
    test_loader = create_test_dataloader(test_dir, batch_size=batch_size)
    
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Evaluate EfficientNet-B0
    print("\n" + "="*80)
    print("EVALUATING EfficientNet-B0")
    print("="*80)
    
    cnn_results = None
    if os.path.exists(cnn_weights):
        print(f"Loading weights from {cnn_weights}...")
        cnn_model = create_cnn_model(num_classes=2, pretrained=False)
        checkpoint = torch.load(cnn_weights, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            cnn_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            cnn_model.load_state_dict(checkpoint)
        
        cnn_model = cnn_model.to(device)
        cnn_model.eval()
        
        print("Evaluating EfficientNet-B0...")
        cnn_results = evaluate_model(cnn_model, test_loader, device)
        
        print("\nEfficientNet-B0 Results:")
        print_evaluation_results(cnn_results)
        
        # Generate plots
        cnn_dir = os.path.join(output_dir, 'cnn')
        plot_comprehensive_evaluation(cnn_results, 'EfficientNet-B0', cnn_dir)
    else:
        print(f"EfficientNet-B0 weights not found at {cnn_weights}. Skipping...")
    
    # Evaluate ViT (DINOv3)
    print("\n" + "="*80)
    print("EVALUATING DINOv3 ViT-S/14")
    print("="*80)
    
    vit_results = None
    if os.path.exists(vit_weights):
        print(f"Loading weights from {vit_weights}...")
        vit_model = create_vit_model(num_classes=2, pretrained=False)
        checkpoint = torch.load(vit_weights, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            vit_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            vit_model.load_state_dict(checkpoint)
        
        vit_model = vit_model.to(device)
        vit_model.eval()
        
        print("Evaluating DINOv3 ViT-S/14...")
        vit_results = evaluate_model(vit_model, test_loader, device)
        
        print("\nDINOv3 ViT-S/14 Results:")
        print_evaluation_results(vit_results)
        
        # Generate plots
        vit_dir = os.path.join(output_dir, 'vit')
        plot_comprehensive_evaluation(vit_results, 'DINOv3 ViT-S/14', vit_dir)
    else:
        print(f"DINOv3 ViT-S/14 weights not found at {vit_weights}. Skipping...")
    
    # Compare models
    if cnn_results and vit_results:
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        comparison_dir = os.path.join(output_dir, 'comparison')
        compare_models_comprehensive(
            cnn_results,
            vit_results,
            'EfficientNet-B0',
            'DINOv3 ViT-S/14',
            comparison_dir
        )
        
        # Save detailed results to file
        results_file = os.path.join(output_dir, 'comparison_results.txt')
        with open(results_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("EfficientNet-B0:\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {cnn_results['accuracy']:.4f} ({cnn_results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {cnn_results['precision']:.4f} ({cnn_results['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {cnn_results['recall']:.4f} ({cnn_results['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {cnn_results['f1_score']:.4f} ({cnn_results['f1_score']*100:.2f}%)\n")
            f.write(f"TP: {cnn_results['true_positive']}, TN: {cnn_results['true_negative']}\n")
            f.write(f"FP: {cnn_results['false_positive']}, FN: {cnn_results['false_negative']}\n\n")
            
            f.write("DINOv3 ViT-S/14:\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {vit_results['accuracy']:.4f} ({vit_results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {vit_results['precision']:.4f} ({vit_results['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {vit_results['recall']:.4f} ({vit_results['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {vit_results['f1_score']:.4f} ({vit_results['f1_score']*100:.2f}%)\n")
            f.write(f"TP: {vit_results['true_positive']}, TN: {vit_results['true_negative']}\n")
            f.write(f"FP: {vit_results['false_positive']}, FN: {vit_results['false_negative']}\n\n")
            
            accuracy_diff = (vit_results['accuracy'] - cnn_results['accuracy']) * 100
            winner = 'DINOv3 ViT-S/14' if accuracy_diff > 0 else 'EfficientNet-B0'
            f.write(f"Winner: {winner}\n")
            f.write(f"Accuracy difference: {accuracy_diff:+.2f}%\n")
        
        print(f"\n✓ Detailed results saved to {results_file}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nAll results and plots saved to: {output_dir}")
    print("\nGenerated files:")
    if cnn_results:
        print(f"  - EfficientNet-B0 plots: {output_dir}cnn/")
    if vit_results:
        print(f"  - DINOv3 ViT-S/14 plots: {output_dir}vit/")
    if cnn_results and vit_results:
        print(f"  - Comparison plots: {output_dir}comparison/")
        print(f"  - Results summary: {output_dir}comparison_results.txt")


if __name__ == "__main__":
    main()
