# ReFrame

ReFrame is a new framework for writing regression tests for HPC systems.
The goal of this framework is to abstract away the complexity of the interactions with the system, separating the logic of a regression test from the low-level details, which pertain to the system configuration and setup.
This allows users to write easily portable regression tests, focusing only on the functionality.

Regression tests in ReFrame are simple Python classes that specify the basic parameters of the test.
The framework will load the test and will send it down a well-defined pipeline that will take care of its execution.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

Writing system regression tests in a high-level modern programming language, like Python, poses a great advantage in organizing and maintaining the tests.
Users can create their own test hierarchies, create test factories for generating multiple tests at the same time and also customize them in a simple and expressive way.


## Documentation

The official documentation is maintained [here](https://eth-cscs.github.io/reframe/index.html).

### Manually generate the documentation

In order to generate the documentation yourself, these are the necessary steps:

1. Install [pandoc](https://pandoc.org).
2. Install the Python requirements (you can do that from within a virtual environment):
   ```
   pip install -r docs/requirements.txt
   ```

Generate the documentation:
```
make -C docs
```

And view it by opening `docs/html/index.html`.

If you want to view also the old documentation, you should first do the following:

```
cd docs/html
python -m http.server # or python -m SimpleHTTPServer for Python 2
```

You can can now view all the documentation (new and old) by opening `localhost:8000` in your browser.
