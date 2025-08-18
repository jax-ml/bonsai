# Whisper Model Installation Guide

This guide will help you set up and run the Whisper model implementation in JAX NNX.

## Prerequisites

### Required Dependencies

Install the following packages:

```bash
# Core JAX ecosystem
pip install jax jaxlib

# Flax with NNX support
pip install flax

# Transformers for tokenization
pip install transformers

# HuggingFace Hub for model downloading
pip install huggingface_hub

# Safetensors for model loading
pip install safetensors

# NumPy for numerical operations
pip install numpy
```

### Optional Dependencies

For audio processing and visualization:

```bash
# Audio processing
pip install librosa soundfile

# Visualization
pip install matplotlib seaborn

# Jupyter for notebooks
pip install jupyter
```

### GPU Support (Optional)

For GPU acceleration:

```bash
# CUDA support (if you have NVIDIA GPU)
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Or for ROCm (AMD GPU)
pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
```

## Quick Start

1. **Install dependencies** (see above)

2. **Run the model**:
   ```bash
   python -m bonsai.models.whisper.tests.run_model
   ```

3. **Open the notebook**:
   ```bash
   jupyter notebook bonsai/models/whisper/tests/Whisper_speech_recognition_example.ipynb
   ```

## Model Download

The first time you run the model, it will automatically download the Whisper weights from HuggingFace. The model will be cached in `/tmp/models-bonsai/whisper-tiny/`.

To use a different model size:

```python
from bonsai.models.whisper import modeling, params

# Choose model size
config = modeling.WhisperConfig.whisper_base()  # or small, medium, large
model = params.load_whisper_model("openai/whisper-base", config)
```

## Troubleshooting

### Common Issues

1. **JAX not found**: Make sure you have JAX installed correctly
   ```bash
   pip install jax jaxlib
   ```

2. **Flax NNX not available**: Ensure you have the latest Flax version
   ```bash
   pip install --upgrade flax
   ```

3. **CUDA errors**: Check your CUDA installation and JAX CUDA compatibility
   ```bash
   python -c "import jax; print(jax.devices())"
   ```

4. **Memory issues**: Use a smaller model or reduce batch size
   ```python
   config = modeling.WhisperConfig.whisper_tiny()  # Smallest model
   ```

### Verification

Run the structure validation test:

```bash
python bonsai/models/whisper/tests/test_structure.py
```

This will verify that all components are properly installed and accessible.

## Performance Tips

1. **Use JIT compilation**: The model automatically uses JAX JIT for optimal performance

2. **GPU acceleration**: Install JAX with CUDA support for faster inference

3. **Batch processing**: Process multiple audio files together for better throughput

4. **Model size**: Choose the appropriate model size for your use case:
   - Tiny: Fastest, good for real-time applications
   - Base: Good balance of speed and accuracy
   - Small/Medium/Large: Higher accuracy, slower inference

## Examples

See the following files for usage examples:

- `tests/run_model.py`: Basic model execution
- `tests/Whisper_speech_recognition_example.ipynb`: Comprehensive notebook with examples
- `README.md`: Detailed documentation and API reference

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your dependencies are correctly installed
3. Run the structure validation test
4. Check the JAX and Flax documentation for platform-specific issues
