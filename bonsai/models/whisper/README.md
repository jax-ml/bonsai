# Whisper in JAX NNX (In Progress)

This directory contains a **JAX + NNX** implementation of the [OpenAI Whisper speech recognition model](https://github.com/openai/whisper), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.

Whisper is a general-purpose speech recognition model that can transcribe audio in multiple languages and perform various speech recognition tasks including transcription, translation, and language identification.

## Model Status

Complete:
1. Correct parameter loading. Testing so far suggests this is correct. 
2. Attention mechanism using the `nnx.MultiheadAttention` layer. Starting with this implementation to reduce implementation complexity for now (e.g. caching taken care of). 
3. Encoder layer passes tests. Decoder layer tests are done without caching. 


Remaining:
1. Finish implementing and numerically testing the model for 30 second audio inputs. 
    1. Test the decoder layer with caching. 
    2. Implement the full model forward pass.
    3. Add tests on real audio data. 
2. Add chunking to deal with larger audio segments. 
3. Create notebook and `run_model.py` file to demonstrate how to use the model and it's efficiency.   
4. Implement sharding to improve performance with larger models. 

