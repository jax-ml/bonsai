# Contributing to Bonsai

We love contributions! Bonsai aims to be a community-driven collection of high-quality JAX NNX model implementations. Whether you're fixing a bug, [adding a new model](#contributing-a-model), improving documentation, or proposing new features, your help is greatly appreciated.

Please take a moment to review this document to understand how to contribute effectively.


## Ways to Contribute

There are many ways you can contribute to Bonsai:

1.  **Reporting Bugs:** If you find a bug, please open an issue describing the problem clearly.
2.  **Suggesting Enhancements:** Have an idea for a new model, a better way to structure code, or a useful feature? Open an issue to discuss it.
3.  **Writing Code:**
    * **Fixing Bugs:** Submit a pull request with a fix for an existing bug.
    * **Adding New Models:** [Implement a new model](#contributing-a-model) using JAX NNX.
    * **Improving Existing Models:** Enhance performance, add features, or refactor existing model implementations.
    * **Writing Tests:** Improve code coverage by adding new tests.
4.  **Improving Documentation:** Enhance the `README.md` files, add clearer explanations, or create new guides.
5.  **Community Engagement:** Answer questions, help other users, and share your experiences.

## Contributing code using pull requests

> [!TIP]
> All submissions to Google Open Source projects need to follow Google’s Contributor License Agreement (CLA), in which contributors agree that their contribution is an original work of authorship. This doesn’t prohibit the use of coding assistance tools, but what’s submitted does need to be a contributor’s original creation.

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Sign the [Google Contributor License Agreement (CLA)](https://cla.developers.google.com/).
   For more information, please see [Bonsai Pull Request checklist](#bonsai-pull-request-checklist).

2. Fork the Bonsai repository by clicking the **Fork** button on the
   [repository page](https://www.github.com/jax-ml/bonsai). This creates
   a copy of the Bonsai repository in your own account.

3. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/bonsai
   cd bonsai
   pip install -e ".[dev]"
   ```

4. Add the Bonsai repo as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git remote add upstream https://www.github.com/jax-ml/bonsai
   ```

5. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

   And implement your changes using your favorite editor (for example:
   [Visual Studio Code](https://code.visualstudio.com/)).

6. Once you are satisfied with your change, create a commit as follows (
   [how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -m "Your commit message"
   ```

   Then sync your code with the main repo:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Finally, push your commit on your development branch and create a remote
   branch in your fork that you can use to create a pull request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```

   Please ensure your contribution is a single commit (see [single change commits](https://docs.jax.dev/en/latest/contributing.html#single-change-commits))

7. Create a pull request from the Bonsai repository and send it for review.
    Check the [PR checklist](#bonsai-pull-request-checklist) for considerations when preparing your PR, and
    consult [GitHub Help](https://help.github.com/articles/about-pull-requests/)
    if you need more information on using pull requests.

## Bonsai Pull Request checklist

### Google contributor license agreement

Contributions to this project must be accompanied by a Google Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again. If you're not certain whether you've signed a CLA, you can open your PR
and our friendly CI bot will check for you.

### Linting and type-checking

Bonsai uses [ruff](https://docs.astral.sh/ruff/) to statically test code quality; the
easiest way to run these checks locally is via the
[pre-commit](https://pre-commit.com/) framework via the following:

Install pre-commit. Alternatively, install the dev packages via `pip install -e ".[dev]"`
```bash
pip install pre-commit
```

Install the pre-commit hook. This will automatically run when you create a commit.
```bash
pre-commit install
```

Alternatively, manually run a pre-commit hook.
```bash
pre-commit run --all-files
```

### Single-change commits and pull requests

We follow [jax's convention](https://docs.jax.dev/en/latest/contributing.html#single-change-commits-and-pull-requests) in single-change commits in pull requests.
A git commit ought to be a self-contained, single change with a descriptive
message. This helps with review and with identifying or reverting changes if
issues are uncovered later on.

**Pull requests typically comprise a single git commit.** (In some cases, for
instance for large refactors or internal rewrites, they may contain several.)
In preparing a pull request for review, you may need to squash together
multiple commits. We ask that you do this prior to sending the PR for review if
possible. The `git rebase -i` command might be useful to this end.


## Contributing a model

We welcome contribution of new models that may be beneficial for the JAX community

1. Check if your model is [already being worked on](https://github.com/jax-ml/bonsai/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22Model%20request%22). If not, add a [Model Request](https://github.com/jax-ml/bonsai/issues/new?template=model-request---enter-model-name-here-.md) with the details about the model.

2. Follow the [guidelines](#contributing-code-using-pull-requests) for forking the repository and making contributions.

3. Create a new directory in [bonsai/models](bonsai/models/). Make sure to follow these designs:

    ```
    ├── models/
    │   └── your_model_name/
    │       ├── tests
    │       │   ├── __init__.py
    │       │   ├── model_validation_colab.ipynb
    │       │   ├── run_model.py
    │       │   └── test_outputs.py
    │       ├── README.md
    │       ├── modeling.py
    │       └── params.py
    ```

    Refer to the following for each of the components:
    * `README.md`: Please include the original model source and `Tested on` matrix for supported configurations confirmation on different hardwares.
    * `modeling.py`: This is your entire model. We aim to have [single-model, single-file policy](https://huggingface.co/blog/transformers-design-philosophy).
      * There can be [exceptions](bonsai/models/sam2) due to model architecture.
      * Make sure your explicit `forward` pass is properly jitted.
      * Maximize the [120 char limit](pyproject.toml#L61) for shorter, concise and easily readable code.
    * `params.py`: Functions for supporting conversion of checkpoints.
    * `tests/`: Make sure the contributed model has reasonable performance and correct quality.
      * Run [JAX profiling](https://docs.jax.dev/en/latest/profiling.html#viewing-the-trace) (i.e. `xprof --port 8791 /tmp/profile-data`) to make sure the model code fully utilizes benefits of JAX's [jit capabilities](https://docs.jax.dev/en/latest/jit-compilation.html).
      * Add a validation colab ([SAM2 example](bonsai/models/sam2/tests/SAM2_image_predictor_example.ipynb)) to make sure the model functions properly.
      * Add a few tests to `test_outputs.py` to confirm that model forward passes are consistent with a reference implementation.

   See an example model directory in [Qwen3](bonsai/models/qwen3).
