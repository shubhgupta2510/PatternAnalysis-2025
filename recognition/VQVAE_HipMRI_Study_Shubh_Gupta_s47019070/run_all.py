"""
Complete VQ-VAE Pipeline Script
Author: Shubh Gupta (s47019070)
Description: Master script to run the entire VQ-VAE training and evaluation pipeline
"""

import sys
import os
import argparse
from pathlib import Path
import subprocess

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")
    
    
def print_step(step_num, total_steps, description):
    """Print step information."""
    print(f"\n{'─' * 80}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'─' * 80}\n")

def run_command(description, command, python_script=False):
    """Run a command and handle errors."""
    print(f"Running: {description}...")
    
    if python_script:
        # Use the virtual environment Python
        venv_python = "/Users/shubhgupta/Library/CloudStorage/GoogleDrive-shubhgupta2510@gmail.com/My Drive/Shubh Gupta/University/YEAR 5/SEM 2/COMP3710/PatternAnalysis-2025/.venv/bin/python"
        full_command = [venv_python] + command
    else:
        full_command = command
    
    try:
        result = subprocess.run(
            full_command,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during {description}: {e}")
        return False


def check_setup():
    """Verify setup before running pipeline."""
    print_step(1, 6, "Verifying Setup")
    
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    # Check if data directory exists
    data_dir = Path("keras_slices_data")
    if not data_dir.exists():
        print("✗ Error: keras_slices_data directory not found!")
        print(f"  Expected location: {data_dir.absolute()}")
        print(f"  Current directory: {Path.cwd()}")
        return False
    
    # Check for required files
    required_files = ["modules.py", "dataset.py", "train.py", "predict.py", "utils.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"✗ Error: Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✓ All required files found")
    print(f"✓ Data directory exists with subdirectories:")
    
    for subdir in ["keras_slices_train", "keras_slices_validate", "keras_slices_test"]:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            num_files = len(list(subdir_path.glob("*.nii.gz")))
            print(f"    - {subdir}: {num_files} files")
    
    return True

def run_test_setup():
    """Run setup verification."""
    print_step(2, 6, "Running Setup Verification")
    return run_command("Setup verification", ["test_setup.py"], python_script=True)

def train_model(quick_mode=False):
    """Train the VQ-VAE model."""
    print_step(3, 6, "Training VQ-VAE Model")
    
    if quick_mode:
        print("Quick mode enabled - training will use reduced epochs")
        print("   (Modify train.py config to set epochs=10 for quick testing)")
    
    print(" This may take 2-4 hours on CPU...")
    print("   The model will automatically:")
    print("   - Train for up to 100 epochs")
    print("   - Save the best model based on validation loss")
    print("   - Stop early if no improvement for 15 epochs")
    print("")
    
    return run_command("Model training", ["train.py"], python_script=True)

def run_predictions():
    """Run predictions and generate visualizations."""
    print_step(4, 6, "Generating Predictions and Visualizations")
    
    print("This will:")
    print("  - Load the best trained model")
    print("  - Evaluate on test set")
    print("  - Calculate SSIM scores")
    print("  - Generate all visualizations")
    print("")
    
    return run_command("Prediction and evaluation", ["predict.py"], python_script=True)

def create_diagrams():
    """Create architecture diagrams."""
    print_step(5, 6, "Creating Architecture Diagrams")
    return run_command("Diagram generation", ["create_diagrams.py"], python_script=True)

def summarize_results():
    """Summarize all results."""
    print_step(6, 6, "Summarizing Results")
    
    print("\nRESULTS SUMMARY")
    print("=" * 80)
    
    # Check for saved models
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.keras"))
        print(f"\n✓ Models saved: {len(model_files)}")
        for model_file in sorted(model_files):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"    - {model_file.name} ({size_mb:.1f} MB)")
    
    # Check for results
    results_dir = Path("results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.png"))
        print(f"\n✓ Visualizations created: {len(result_files)}")
        for result_file in sorted(result_files):
            print(f"    - {result_file.name}")
    
    # Check for logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        png_files = list(logs_dir.glob("*.png"))
        if png_files:
            print(f"\n✓ Training plots: {len(png_files)}")
            for png in sorted(png_files):
                print(f"    - {png.name}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nOutput Locations:")
    print(f"    Models:         {models_dir.absolute()}")
    print(f"    Results:        {results_dir.absolute()}")
    print(f"    Training Logs:  {logs_dir.absolute()}")
    print("\nNext Steps:")
    print("    - Review visualizations in results/")
    print("    - Check SSIM scores in terminal output")
    print("    - View training curves in logs/")
    print("    - Use TensorBoard: tensorboard --logdir logs/")
    print("")

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="VQ-VAE Complete Pipeline - Train and Evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                    # Run complete pipeline
  python run_all.py --quick            # Quick mode (for testing)
  python run_all.py --skip-train       # Skip training, only predict
  python run_all.py --train-only       # Only train, skip predictions
  python run_all.py --no-verify        # Skip setup verification
        """
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced epochs (for testing)"
    )
    
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (assumes model already exists)"
    )
    
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train the model, skip predictions"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip setup verification step"
    )
    
    parser.add_argument(
        "--no-diagrams",
        action="store_true",
        help="Skip diagram generation"
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print_header("VQ-VAE Complete Pipeline")
    print("Author: Shubh Gupta (s47019070)")
    print("Project: HipMRI Study Generative Model")
    print("=" * 80)
    
    # Track success of each step
    all_success = True
    
    # Step 1: Check setup
    if not args.no_verify:
        if not check_setup():
            print("\nSetup verification failed. Please fix issues and try again.")
            return 1
    
    # Step 2: Run test setup
    if not args.no_verify:
        if not run_test_setup():
            print("\nSetup test failed, but continuing anyway...")
    
    # Step 3: Train model
    if not args.skip_train:
        if not train_model(quick_mode=args.quick):
            print("\nTraining failed!")
            all_success = False
            if not args.train_only:
                print("Cannot proceed to predictions without a trained model.")
                return 1
    else:
        print_step(3, 6, "Skipping Training (as requested)")
        print("✓ Using existing model")
    
    # Step 4: Run predictions
    if not args.train_only and all_success:
        if not run_predictions():
            print("\nPrediction failed!")
            all_success = False
    elif args.train_only:
        print_step(4, 6, "Skipping Predictions (train-only mode)")
    
    # Step 5: Create diagrams (optional)
    if not args.no_diagrams and not args.train_only:
        if not create_diagrams():
            print("\nDiagram generation failed, but continuing...")
    
    # Step 6: Summarize
    if all_success:
        summarize_results()
        return 0
    else:
        print("\nPipeline completed with errors. Please review output above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        print("Partial results may be available in models/ and results/ directories.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
