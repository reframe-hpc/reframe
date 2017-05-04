# Folder Structure

ReFrame allows the users to organize their regression tests in any way that is the most convenient for their needs.
The only soft requirement imposed by the framework is that a `src/}`folder should be present at the same level as the test's source file. This is just the default behavior. The users may override this by redefining the `self.sourcesdir` variable in their tests.

Users can group together related tests in a common directory sharing the same `src/` folder as in the `foobar` family of tests in the following example.
This sharing can eliminate duplication at the level of regression test resources, which can prove beneficial in maintaining a large regression test suite.
For run-only regression tests the `src/` directory can be empty or contain other resources relative to the test, e.g., input files.


The following directory structure visualizes the organization concepts described:
```bash
mychecks/
   compile/
      helloworld/
          helloworld.py
          src/
             helloworld.c
      foobar/
          bar.py
          foo.py
          src/
             bar.c
             foo.c
   apps/
      prog1/
          src/
          prog1.py
      prog2/
          src/
              input.txt
          prog2.py
```