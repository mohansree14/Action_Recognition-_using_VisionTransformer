import matplotlib.pyplot as plt
import numpy as np
import re

def parse_log_file(log_file_path):
    """Parse training log file to extract loss and accuracy data"""
    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match log lines
    pattern = r'Epoch (\d+)/\d+: Train Loss = ([\d.]+), Val Loss = ([\d.]+), Val Accuracy = ([\d.]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        epoch = int(match[0])
        train_loss = float(match[1])
        val_loss = float(match[2])
        val_acc = float(match[3])
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    return epochs, train_losses, val_losses, val_accuracies

# Parse both log files
lr_0001_sgd = "Results/lr - 0.001/SGD/timesformer_lr000100_f8_s32_20250805_134346.log"
lr_0005_sgd = "Results/lr - 0.0005/SGD/timesformer_lr000050_f8_s32_20250804_202022.log"

# Parse data
epochs1, train_loss1, val_loss1, val_acc1 = parse_log_file(lr_0001_sgd)
epochs2, train_loss2, val_loss2, val_acc2 = parse_log_file(lr_0005_sgd)

# Create single professional graph for report
plt.figure(figsize=(12, 8))

# Plot training and validation loss for both learning rates
plt.plot(epochs1, train_loss1, 'b-', linewidth=3, label='Training Loss (LR=0.001)', alpha=0.8)
plt.plot(epochs1, val_loss1, 'b--', linewidth=3, label='Validation Loss (LR=0.001)', alpha=0.8)
plt.plot(epochs2, train_loss2, 'r-', linewidth=3, label='Training Loss (LR=0.0005)', alpha=0.8)
plt.plot(epochs2, val_loss2, 'r--', linewidth=3, label='Validation Loss (LR=0.0005)', alpha=0.8)

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=14, fontweight='bold')
plt.title('Timesformer Training: SGD Optimizer Learning Rate Comparison', 
          fontsize=16, fontweight='bold', pad=20)

plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
plt.yscale('log')

# Set professional styling
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add subtle background color
plt.gca().set_facecolor('#fafafa')

# Improve margins
plt.tight_layout(pad=2.0)

# Save the plot
output_path = "Results/SGD_Loss_Curves_Report.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Professional loss curve saved to: {output_path}")

# Don't show interactive plot for report
# plt.show()
