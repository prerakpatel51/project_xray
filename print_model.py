
import sys
from pathlib import Path

# Add project modules to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.vision_mamba import create_vision_mamba_model

# Create a 'base' model instance
model = create_vision_mamba_model(model_size="base")

# Print the model architecture
print(model)
