import torch
import whisper

def check_mps_availability():
    """Check if MPS is available and properly working."""
    if not torch.backends.mps.is_available():
        return False, "MPS not available on this system"
    
    if not torch.backends.mps.is_built():
        return False, "PyTorch not built with MPS support"
    
    try:
        # Test basic tensor operations
        x = torch.randn(2, 3).to("mps")
        y = x + 2
        z = y * y
        z = z.to("cpu")  # Test moving back to CPU
        return True, "MPS is working properly"
    except Exception as e:
        return False, f"MPS runtime check failed: {str(e)}"

# Check MPS status
mps_available, mps_message = check_mps_availability()
if mps_available:
    device = "mps"
    print(f"✅ Using MPS (Apple GPU): {mps_message}")
else:
    device = "cpu"
    print(f"⚠️ Using CPU: {mps_message}")

# Print PyTorch version for debugging
print(f"PyTorch version: {torch.__version__}")

print(f"\nLoading Whisper model on {device}...")
model = whisper.load_model("base")

if device == "mps":
    try:
        # Move model to MPS device
        model = model.to(device)
        
        # Verify model device
        first_param_device = next(model.parameters()).device
        print(f"Model parameters are on: {first_param_device}")
        
        if str(first_param_device) == "mps":
            print("✅ Model successfully moved to MPS")
        else:
            print("⚠️ Model is not on MPS device, falling back to CPU")
            device = "cpu"
            model = model.to("cpu")
    except Exception as e:
        print(f"⚠️ Error moving model to MPS: {str(e)}")
        print("⚠️ Falling back to CPU")
        device = "cpu"
        model = model.to("cpu")

print(f"\nModel loaded successfully on {device}!") 