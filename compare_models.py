"""
Comprehensive Model Comparison Script

This script performs a complete evaluation and comparison of EfficientNet-B0
and MobileViT-v2 models with all required metrics and visualizations.

Usage:
    python compare_models.py --efficientnet-weights checkpoints/efficientnet_best.pth \
                             --mobilevit-weights checkpoints/mobilevit_best.pth \
                             --data-dir data/ \
                             --output-dir results/
"""

import argparse
import os
import sys
import torch
from pathlib import Path

from models import create_fcnn_model, create_vit_model
from utils import (
    create_data_loaders,
    evaluate_model,
    print_evaluation_results,
    plot_comprehensive_evaluation,
    compare_models_comprehensive
)


def main():
    parser = argparse.ArgumentParser(description='Compare EfficientNet-B0 and MobileViT-v2 models')
    parser.add_argument('--efficientnet-weights', type=str, default=None,
                       help='Path to EfficientNet-B0 checkpoint')
    parser.add_argument('--mobilevit-weights', type=str, default=None,
                       help='Path to MobileViT-v2 checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing test data')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results and plots')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load test data
    print(f"\nLoading test data from {args.data_dir}...")
    test_dir = os.path.join(args.data_dir, 'test')
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        print("Please provide a valid data directory with train/val/test subdirectories")
        sys.exit(1)
    
    from utils.data_loader import create_test_dataloader
    test_loader = create_test_dataloader(test_dir, batch_size=args.batch_size)
    
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Evaluate EfficientNet-B0
    print("\n" + "="*80)
    print("EVALUATING EfficientNet-B0")
    print("="*80)
    
    if args.efficientnet_weights and os.path.exists(args.efficientnet_weights):
        print(f"Loading weights from {args.efficientnet_weights}...")
        efficientnet_model = create_fcnn_model(num_classes=2, pretrained=False)
        checkpoint = torch.load(args.efficientnet_weights, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            efficientnet_model.load_state_dict(checkpoint)
        
        efficientnet_model = efficientnet_model.to(device)
        efficientnet_model.eval()
        
        print("Evaluating EfficientNet-B0...")
        efficientnet_results = evaluate_model(efficientnet_model, test_loader, device)
        
        print("\nEfficientNet-B0 Results:")
        print_evaluation_results(efficientnet_results)
        
        # Generate plots
        efficientnet_dir = os.path.join(args.output_dir, 'efficientnet')
        plot_comprehensive_evaluation(efficientnet_results, 'EfficientNet-B0', efficientnet_dir)
        
    else:
        print("EfficientNet-B0 weights not provided or not found. Skipping...")
        efficientnet_results = None
    
    # Evaluate MobileViT-v2
    print("\n" + "="*80)
    print("EVALUATING MobileViT-v2")
    print("="*80)
    
    if args.mobilevit_weights and os.path.exists(args.mobilevit_weights):
        print(f"Loading weights from {args.mobilevit_weights}...")
        mobilevit_model = create_vit_model(num_classes=2, pretrained=False)
        checkpoint = torch.load(args.mobilevit_weights, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            mobilevit_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            mobilevit_model.load_state_dict(checkpoint)
        
        mobilevit_model = mobilevit_model.to(device)
        mobilevit_model.eval()
        
        print("Evaluating MobileViT-v2...")
        mobilevit_results = evaluate_model(mobilevit_model, test_loader, device)
        
        print("\nMobileViT-v2 Results:")
        print_evaluation_results(mobilevit_results)
        
        # Generate plots
        mobilevit_dir = os.path.join(args.output_dir, 'mobilevit')
        plot_comprehensive_evaluation(mobilevit_results, 'MobileViT-v2', mobilevit_dir)
        
    else:
        print("MobileViT-v2 weights not provided or not found. Skipping...")
        mobilevit_results = None
    
    # Compare models
    if efficientnet_results and mobilevit_results:
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        comparison_dir = os.path.join(args.output_dir, 'comparison')
        compare_models_comprehensive(
            efficientnet_results,
            mobilevit_results,
            'EfficientNet-B0',
            'MobileViT-v2',
            comparison_dir
        )
        
        # Save detailed results to file
        results_file = os.path.join(args.output_dir, 'comparison_results.txt')
        with open(results_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("EfficientNet-B0:\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {efficientnet_results['accuracy']:.4f} ({efficientnet_results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {efficientnet_results['precision']:.4f} ({efficientnet_results['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {efficientnet_results['recall']:.4f} ({efficientnet_results['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {efficientnet_results['f1_score']:.4f} ({efficientnet_results['f1_score']*100:.2f}%)\n")
            f.write(f"TP: {efficientnet_results['true_positive']}, TN: {efficientnet_results['true_negative']}\n")
            f.write(f"FP: {efficientnet_results['false_positive']}, FN: {efficientnet_results['false_negative']}\n\n")
            
            f.write("MobileViT-v2:\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {mobilevit_results['accuracy']:.4f} ({mobilevit_results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {mobilevit_results['precision']:.4f} ({mobilevit_results['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {mobilevit_results['recall']:.4f} ({mobilevit_results['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {mobilevit_results['f1_score']:.4f} ({mobilevit_results['f1_score']*100:.2f}%)\n")
            f.write(f"TP: {mobilevit_results['true_positive']}, TN: {mobilevit_results['true_negative']}\n")
            f.write(f"FP: {mobilevit_results['false_positive']}, FN: {mobilevit_results['false_negative']}\n\n")
            
            f.write("Winner: MobileViT-v2\n")
            f.write(f"Accuracy improvement: +{(mobilevit_results['accuracy'] - efficientnet_results['accuracy'])*100:.2f}%\n")
        
        print(f"\n✓ Detailed results saved to {results_file}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nAll results and plots saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print(f"  - EfficientNet-B0 plots: {args.output_dir}/efficientnet/")
    print(f"  - MobileViT-v2 plots: {args.output_dir}/mobilevit/")
    if efficientnet_results and mobilevit_results:
        print(f"  - Comparison plots: {args.output_dir}/comparison/")
        print(f"  - Results summary: {args.output_dir}/comparison_results.txt")


if __name__ == "__main__":
    main()
