# 🤝 Contributing to Bonsai

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

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Sign the [Google Contributor License Agreement (CLA)](https://cla.developers.google.com/).
   For more information, please see [Bonsai Pull Request checklist](#bonsai-pull-request-checklist).

2. Fork the Bonsai repository by clicking the **Fork** button on the
   [repository page](http://www.github.com/jax-ml/bonsai). This creates
   a copy of the Bonsai repository in your own account.

3. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/bonsai
   cd bonsai
   pip install -r build/test-requirements.txt  # Installs all testing requirements.
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

   Please ensure your contribution is a single commit (see {ref}`single-change-commits`)

7. Create a pull request from the Bonsai repository and send it for review.
    Check the {ref}`pr-checklist` for considerations when preparing your PR, and
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

#### 1. Install `pre-commit`.
```bash
pip install pre-commit
```

#### 2. Install `pre-commit` hooks.
Navigate to your `bonsai` repository root and install the hooks. This sets up the checks to run automatically before every `git commit`.

```bash
pre-commit install
```

Ruff is configured to automatically fix many linting and formatting issues. If Ruff makes any changes to your files, `pre-commit` will stop the commit, stage the fixed files, and notify you. You'll then need to git add the modified files and git commit again.

#### 3. Manual checks (Optional, but recommended)

While `pre-commit` handles checks on commit, you might want to manually run all checks across your entire codebase at any time (e.g., after pulling new changes, or before starting a significant refactor).

* To run all configured pre-commit hooks against all files (not just staged ones):

    ```bash
    pre-commit run --all-files
    ```

* To run Ruff's linting and auto-fix capabilities across your entire project, referencing the rules in pyproject.toml:

    ```bash
    ruff check --fix
    ```
    This command will lint all Python files, applying any auto-fixable rules. It's a good habit to run this proactively.

* To run Ruff's formatting capabilities across your entire project, referencing the style defined in pyproject.toml:
    ```bash
    ruff format .
    ```

Please ensure your code passes all linting, formatting, and type-checking checks before submitting a pull request. This helps maintain a clean and reliable codebase for everyone.


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
    │       │   └── run_model.py
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
   
   See an example model directory in [Qwen3](bonsai/models/qwen3).