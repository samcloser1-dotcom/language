import sys
import os

# Add project root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def main():
    # If running under Streamlit (e.g. Streamlit Cloud entry point set to main.py)
    try:
        import streamlit as st
        # Check if running within active Streamlit runtime context
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            import app.streamlit_app
            return
    except Exception:
        pass

    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Sign Language Detection & Translation System")
    parser.add_argument("--collect", action="store_true", help="Launch interactive webcam dataset collection module")
    parser.add_argument("--train", action="store_true", help="Train classification model on dataset CSV")
    parser.add_argument("--sample-data", action="store_true", help="Generate synthetic sample dataset for testing")
    parser.add_argument("--gui", action="store_true", default=False, help="Launch Desktop GUI application")

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
        try:
            from app.gui import main as run_gui
            run_gui()
        except (ImportError, ModuleNotFoundError, Exception) as e:
            # Tkinter not available or running in cloud environment — load Streamlit web app
            print(f"[Launcher] Tkinter GUI unavailable ({e}). Falling back to Streamlit app...")
            import app.streamlit_app

if __name__ == "__main__":
    main()

