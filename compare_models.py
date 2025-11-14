"""
Comprehensive Model Comparison Script

This script performs a complete evaluation and comparison of FCNN (EfficientNet-B0)
and ViT (MobileViT-v2) models with all required metrics and visualizations.

Usage:
    python compare_models.py
"""

import os
import sys
import torch

from models import create_fcnn_model, create_vit_model
from utils import (
    evaluate_model,
    print_evaluation_results,
    plot_comprehensive_evaluation,
    compare_models_comprehensive
)


def main():
    """Compare FCNN and ViT models."""
    # Hardcoded configuration
    fcnn_weights = 'checkpoints/fcnn_best.pth'
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
    
    # Evaluate FCNN (EfficientNet-B0)
    print("\n" + "="*80)
    print("EVALUATING FCNN (EfficientNet-B0)")
    print("="*80)
    
    fcnn_results = None
    if os.path.exists(fcnn_weights):
        print(f"Loading weights from {fcnn_weights}...")
        fcnn_model = create_fcnn_model(num_classes=2, pretrained=False)
        checkpoint = torch.load(fcnn_weights, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            fcnn_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            fcnn_model.load_state_dict(checkpoint)
        
        fcnn_model = fcnn_model.to(device)
        fcnn_model.eval()
        
        print("Evaluating FCNN...")
        fcnn_results = evaluate_model(fcnn_model, test_loader, device)
        
        print("\nFCNN Results:")
        print_evaluation_results(fcnn_results)
        
        # Generate plots
        fcnn_dir = os.path.join(output_dir, 'fcnn')
        plot_comprehensive_evaluation(fcnn_results, 'FCNN', fcnn_dir)
    else:
        print(f"FCNN weights not found at {fcnn_weights}. Skipping...")
    
    # Evaluate ViT (MobileViT-v2)
    print("\n" + "="*80)
    print("EVALUATING ViT (MobileViT-v2)")
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
        
        print("Evaluating ViT...")
        vit_results = evaluate_model(vit_model, test_loader, device)
        
        print("\nViT Results:")
        print_evaluation_results(vit_results)
        
        # Generate plots
        vit_dir = os.path.join(output_dir, 'vit')
        plot_comprehensive_evaluation(vit_results, 'ViT', vit_dir)
    else:
        print(f"ViT weights not found at {vit_weights}. Skipping...")
    
    # Compare models
    if fcnn_results and vit_results:
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        comparison_dir = os.path.join(output_dir, 'comparison')
        compare_models_comprehensive(
            fcnn_results,
            vit_results,
            'FCNN',
            'ViT',
            comparison_dir
        )
        
        # Save detailed results to file
        results_file = os.path.join(output_dir, 'comparison_results.txt')
        with open(results_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("FCNN (EfficientNet-B0):\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {fcnn_results['accuracy']:.4f} ({fcnn_results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {fcnn_results['precision']:.4f} ({fcnn_results['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {fcnn_results['recall']:.4f} ({fcnn_results['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {fcnn_results['f1_score']:.4f} ({fcnn_results['f1_score']*100:.2f}%)\n")
            f.write(f"TP: {fcnn_results['true_positive']}, TN: {fcnn_results['true_negative']}\n")
            f.write(f"FP: {fcnn_results['false_positive']}, FN: {fcnn_results['false_negative']}\n\n")
            
            f.write("ViT (MobileViT-v2):\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {vit_results['accuracy']:.4f} ({vit_results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {vit_results['precision']:.4f} ({vit_results['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {vit_results['recall']:.4f} ({vit_results['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {vit_results['f1_score']:.4f} ({vit_results['f1_score']*100:.2f}%)\n")
            f.write(f"TP: {vit_results['true_positive']}, TN: {vit_results['true_negative']}\n")
            f.write(f"FP: {vit_results['false_positive']}, FN: {vit_results['false_negative']}\n\n")
            
            accuracy_diff = (vit_results['accuracy'] - fcnn_results['accuracy']) * 100
            winner = 'ViT' if accuracy_diff > 0 else 'FCNN'
            f.write(f"Winner: {winner}\n")
            f.write(f"Accuracy difference: {accuracy_diff:+.2f}%\n")
        
        print(f"\n✓ Detailed results saved to {results_file}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nAll results and plots saved to: {output_dir}")
    print("\nGenerated files:")
    if fcnn_results:
        print(f"  - FCNN plots: {output_dir}fcnn/")
    if vit_results:
        print(f"  - ViT plots: {output_dir}vit/")
    if fcnn_results and vit_results:
        print(f"  - Comparison plots: {output_dir}comparison/")
        print(f"  - Results summary: {output_dir}comparison_results.txt")


if __name__ == "__main__":
    main()
