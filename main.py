"""
main.py - Entry Point
Deteksi Retinopati Diabetik dengan MViTv2 + Explainability Hibrida

Script utama untuk menjalankan berbagai mode:
- train: Melatih model dari awal atau melanjutkan training
- evaluate: Mengevaluasi model pada test set
- explain: Menjalankan analisis explainability
- test_data: Memverifikasi dataloader dan preprocessing
"""

import argparse
import os
from config import validate_config

def main():
    parser = argparse.ArgumentParser(description="MViTv2 Diabetic Retinopathy Detection")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True,
        choices=["train", "evaluate", "explain", "test_data"],
        help="Mode eksekusi: train, evaluate, explain, atau test_data"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path ke checkpoint untuk melanjutkan training atau evaluasi"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5,
        help="Jumlah sampel per kelas untuk mode explain"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=None,
        help="Path ke satu gambar spesifik untuk dianalisis (mode explain)"
    )
    
    args = parser.parse_args()
    
    # Validasi konfigurasi sebelum mulai
    if not validate_config():
        print("Silakan perbaiki konfigurasi di config.py sebelum melanjutkan.")
        return
    
    # Eksekusi berdasarkan mode
    if args.mode == "train":
        from train import train
        train(resume_checkpoint=args.resume)
        
    elif args.mode == "evaluate":
        from evaluate import evaluate
        evaluate(checkpoint_path=args.resume)
        
    elif args.mode == "explain":
        from explainability import run_explainability, explain_single_image
        if args.image:
            explain_single_image(args.image, checkpoint_path=args.resume)
        else:
            run_explainability(
                checkpoint_path=args.resume, 
                num_samples_per_class=args.num_samples
            )
            
    elif args.mode == "test_data":
        from dataset import get_dataloaders
        print("\nMenjalankan test DataLoader...")
        get_dataloaders()
        print("\nTest DataLoader selesai!")

if __name__ == "__main__":
    main()
