# CLI

## `python3 -m bonsai --help`
```bash
usage: python3 -m bonsai [-h] [--version] [-s SEARCH] {ls,run} ...

Bonsai is a minimal, lightweight JAX implementation of popular models.

positional arguments:
  {ls,run}
    ls                  List installed models
    run                 Run specified model

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -s SEARCH, --search SEARCH
                        Alternative filepath(s) or fully-qualified name (FQN)
                        to use models from.
```

## `python3 -m bonsai ls --help`
```bash
usage: python3 -m bonsai ls [-h]

options:
  -h, --help  show this help message and exit
```

## `python3 -m bonsai run --help`
```bash
usage: python3 -m bonsai run [-h] [-n MODEL_NAME] [-p PATH_ROOT]

options:
  -h, --help            show this help message and exit
  -n MODEL_NAME, --model-name MODEL_NAME
                        Model name
  -p PATH_ROOT, --path-root PATH_ROOT
                        Path root (for model.safetensors directory
                        &etc.).Defaults to "/tmp/models-bonsai/${model_name}"
```
