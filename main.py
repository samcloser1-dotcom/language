import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Real-Time Sign Language Detection & Translation System")
    parser.add_argument("--collect", action="store_true", help="Launch interactive webcam dataset collection module")
    parser.add_argument("--train", action="store_true", help="Train classification model on dataset CSV")
    parser.add_argument("--sample-data", action="store_true", help="Generate synthetic sample dataset for testing")
    parser.add_argument("--gui", action="store_true", default=True, help="Launch Desktop GUI application (default)")

    args = parser.parse_args()

    if args.collect:
        from src.data_collection import run_data_collection
        run_data_collection()
    elif args.train:
        from src.train_model import train_sign_model
        train_sign_model()
    elif args.sample_data:
        from create_sample_data import generate_sample_dataset
        generate_sample_dataset()
    else:
        from app.gui import main as run_gui
        run_gui()

if __name__ == "__main__":
    main()
