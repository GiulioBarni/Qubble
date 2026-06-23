# Contributing to Qubble

Qubble is the research code accompanying *Globally Charged Vacuum Decay*
([arXiv:2606.21653](https://arxiv.org/abs/2606.21653)) by G. Barni and J. R. Espinosa. It is primarily a
**publication reproducibility companion**, not a general-purpose finite-density
tunnelling package. Contributions are welcome, but please read this short note
before opening an issue or a pull request.

## Scope

Qubble is built around a specific scalar model and a specific numerical
strategy (a real doubled-field Euclidean saddle with twisted boundary
conditions, plus a post-tunnelling Minkowski continuation). Pull requests that
extend the code in directions consistent with this scope are welcome — for
example:

- additional diagnostics or sanity checks,
- documentation improvements,
- numerical-precision improvements that do not change physics conventions,
- tests that lock in current behaviour,
- portability fixes (paths, OS-specific issues, dependency pinning).

Pull requests that significantly refactor or rewrite the scientific algorithms
are unlikely to be merged without prior discussion in an issue, because the
present implementation is the one referenced in the paper.

## Reporting bugs

Please open a [GitHub issue](https://github.com/GiulioBarni/Qubble/issues)
including:

1. **Environment**
   - operating system,
   - Python version,
   - relevant package versions (run `pip freeze` or `conda list`).
2. **What you ran** — the notebook section, function call, or script, with
   any non-default parameters.
3. **Full traceback** — the complete Python error, not just the last line.
4. **Minimal reproduction** — ideally a short snippet or a notebook cell that
   reproduces the issue starting from a fresh kernel.
5. **Expected vs. observed behaviour**.

## Pull requests

- Open an issue first if the change is non-trivial, so we can agree on scope.
- Keep PRs focused: one logical change per PR.
- Run the test suite locally before submitting:
  ```bash
  pytest Bubble_finder/tests
  pytest Q_ball_finder/tests
  ```
- Do not commit large generated artefacts (cached `.npz`/`.npy`/`.h5`
  solutions, large notebook outputs, multi-megabyte figures that are not paper
  figures). The `.gitignore` already excludes the standard ones; if your
  workflow produces new heavy outputs, please discuss in the issue how to
  archive them externally (e.g. on Zenodo).
- Notebooks in this repository are part of the paper's reproduction pipeline.
  When editing a notebook, please:
  - keep cell outputs that are referenced by the paper figures,
  - avoid committing throwaway debugging output,
  - check that the notebook still runs end-to-end on a clean kernel.

## Coding style

- Follow the existing module style (NumPy-style docstrings, English comments,
  no inline narration of obvious code).
- Do not rename internal functions, classes, or modules without prior
  discussion: the names are referenced in the paper's appendices and in
  cached results.

## Code of conduct

Be kind, be precise, and assume good faith. Discussions should stay focused on
the science and on the code.
