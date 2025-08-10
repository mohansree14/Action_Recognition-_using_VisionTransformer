# main.py
# Sequential execution of src scripts for HMDB Video Action Recognition

import sys
import os
import argparse
import subprocess
import time
from datetime import datetime

def print_banner():
    print("=" * 70)
    print("🎬 HMDB Video Action Recognition - Sequential Script Runner")
    print("=" * 70)

def log_step(step_name, status="START"):
    """Log each step with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if status == "START":
        print(f"\n{'='*50}")
        print(f"� [{timestamp}] STARTING: {step_name}")
        print(f"{'='*50}")
    elif status == "SUCCESS":
        print(f"\n✅ [{timestamp}] COMPLETED: {step_name}")
    elif status == "ERROR":
        print(f"\n❌ [{timestamp}] FAILED: {step_name}")
    elif status == "SKIP":
        print(f"\n⏭️  [{timestamp}] SKIPPED: {step_name}")

def run_script(script_path, script_name, args=None):
    """Run a script and return success status"""
    log_step(script_name, "START")
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    try:
        # Show the command being run
        print(f"📋 Command: {' '.join(cmd)}")
        print("-" * 50)
        
        # Run the script with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        
        if rc == 0:
            log_step(script_name, "SUCCESS")
            return True
        else:
            log_step(script_name, "ERROR")
            print(f"Exit code: {rc}")
            return False
            
    except Exception as e:
        log_step(script_name, "ERROR")
        print(f"Exception: {e}")
        return False

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "src/frame_preprocess.py",
        "src/train.py", 
        "src/evaluate.py",
        "configs/coursework_config.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files found")
    return True

def run_all_scripts(model_type="timesformer", skip_preprocess=False, skip_train=False, skip_eval=False):
    """Run all scripts in sequence"""
    print_banner()
    print(f"🎯 Target model: {model_type.upper()}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Exiting.")
        return False
    
    total_steps = 3
    current_step = 0
    failed_steps = []
    
    # Step 1: Frame preprocessing
    current_step += 1
    if skip_preprocess:
        log_step(f"Step {current_step}/{total_steps}: Frame Preprocessing", "SKIP")
        print("User requested to skip preprocessing")
    else:
        # Check if already processed
        processed_path = os.path.join("results", "HMDB_simp_processed")
        if os.path.exists(processed_path):
            log_step(f"Step {current_step}/{total_steps}: Frame Preprocessing", "SKIP")
            print("Processed data already exists")
        else:
            success = run_script("src/frame_preprocess.py", f"Step {current_step}/{total_steps}: Frame Preprocessing")
            if not success:
                failed_steps.append("Frame Preprocessing")
    
    # Step 2: Model training
    current_step += 1
    if skip_train:
        log_step(f"Step {current_step}/{total_steps}: Model Training", "SKIP")
        print("User requested to skip training")
    else:
        train_args = ["--model", model_type, "--use-processed"]
        success = run_script("src/train.py", f"Step {current_step}/{total_steps}: Model Training ({model_type.upper()})", train_args)
        if not success:
            failed_steps.append("Model Training")
    
    # Step 3: Model evaluation
    current_step += 1
    if skip_eval:
        log_step(f"Step {current_step}/{total_steps}: Model Evaluation", "SKIP")
        print("User requested to skip evaluation")
    else:
        success = run_script("src/evaluate.py", f"Step {current_step}/{total_steps}: Model Evaluation")
        if not success:
            failed_steps.append("Model Evaluation")
    
    # Summary
    print("\n" + "="*70)
    print("📊 EXECUTION SUMMARY")
    print("="*70)
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_steps:
        print(f"❌ Failed steps ({len(failed_steps)}):")
        for step in failed_steps:
            print(f"   - {step}")
        return False
    else:
        print("✅ All steps completed successfully!")
        return True

def run_single_script(script_name, args=None):
    """Run a single script from src folder"""
    script_mapping = {
        "preprocess": ("src/frame_preprocess.py", "Frame Preprocessing"),
        "train": ("src/train.py", "Model Training"),
        "evaluate": ("src/evaluate.py", "Model Evaluation"),
        "dataset": ("src/dataset.py", "Dataset Test"),
        "utils": ("src/utils.py", "Utilities")
    }
    
    if script_name not in script_mapping:
        print(f"❌ Unknown script: {script_name}")
        print(f"Available scripts: {', '.join(script_mapping.keys())}")
        return False
    
    script_path, display_name = script_mapping[script_name]
    
    print_banner()
    return run_script(script_path, display_name, args)

def show_help():
    """Show usage instructions"""
    print_banner()
    print("\n📚 Usage Instructions:")
    
    print("\n🔄 Run all scripts in sequence:")
    print("   python main.py --run-all [timesformer|vit|slowfast]")
    print("   python main.py --run-all  # defaults to timesformer")
    
    print("\n🎯 Run single script:")
    print("   python main.py --script preprocess")
    print("   python main.py --script train --model timesformer")
    print("   python main.py --script evaluate")
    
    print("\n⏭️  Skip specific steps:")
    print("   python main.py --run-all --skip-preprocess")
    print("   python main.py --run-all --skip-train")
    print("   python main.py --run-all --skip-eval")
    
    print("\n� Available scripts:")
    print("   - preprocess: Frame preprocessing with augmentations")
    print("   - train: Model training")
    print("   - evaluate: Model evaluation")
    print("   - dataset: Test dataset loading")
    print("   - utils: Utility functions")
    
    print("\n� Check requirements:")
    print("   python main.py --check")

def main():
    parser = argparse.ArgumentParser(description='HMDB Video Action Recognition - Sequential Script Runner')
    
    # Main actions
    parser.add_argument('--run-all', nargs='?', const='timesformer', 
                       help='Run all scripts in sequence (timesformer|vit|slowfast)')
    parser.add_argument('--script', type=str, 
                       help='Run single script (preprocess|train|evaluate|dataset|utils)')
    parser.add_argument('--check', action='store_true', 
                       help='Check if all required files exist')
    
    # Options for run-all
    parser.add_argument('--skip-preprocess', action='store_true', 
                       help='Skip preprocessing step')
    parser.add_argument('--skip-train', action='store_true', 
                       help='Skip training step')
    parser.add_argument('--skip-eval', action='store_true', 
                       help='Skip evaluation step')
    
    # Script arguments
    parser.add_argument('--model', type=str, default='timesformer',
                       help='Model type for training (timesformer|vit|slowfast)')
    parser.add_argument('--use-processed', action='store_true',
                       help='Use processed data for training')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_scripts(
            model_type=args.run_all,
            skip_preprocess=args.skip_preprocess,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval
        )
    elif args.script:
        # Prepare script arguments
        script_args = []
        if args.script == 'train':
            script_args.extend(['--model', args.model])
            if args.use_processed:
                script_args.append('--use-processed')
        
        run_single_script(args.script, script_args)
    elif args.check:
        print_banner()
        check_requirements()
    else:
        show_help()

if __name__ == "__main__":
    main()
