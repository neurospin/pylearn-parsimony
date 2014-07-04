General guidelines
==================

- Functions are better if they are short (try to keep them less than 25 lines, and absolutely less than 50).
- Make functions and classes, or code in general, as clear as possible. Use explicit and descriptive names as far as possible.
- Some comments are very important, for example when choices have been made or if something is an opition.
- Use standard comment prefixes, that are easy to search, and which are adequate for: TODO, FIXME, WARNING, ERROR, BUG, XXX, etc.
- Boy scout rule: Leave the campground cleaner than you found it.
- Write your method from top to down. For example: if a method A use a method B, write A before you write B.
- Class order: top -> down, high level code -> low level code.
- Don't leave prints (writes to stdout).
- Don't optimise prematurely. Make it work well, before you spend time making it fast.


Python code format
==================

- Pep 8. As far as possible.
- Never go beyond 80 characters per line. If you cannot break the line before 80 characters, well then you or someone else have done something wrong.
- Keep modules short and purposeful.
- All the fields of an instance should be explictly declared in the function "__init__" or in methods that have the prefix "_init_". Avoid adding fields run-time.
- Never "import *".
- Do not use relative imports. When we move to Python 3, this will be relevant.
- First import standard python modules, add one empty line and then import other non-standard python modules, add another empty line and then import library modules.
- Use coherent names when importing modules, and use the same thoughout. E.g. "import numpy as np", "import very.long.imported.module.path as path".
- Private fields, classes, functions, etc. use an underscore prefix. Such as "_private_field = 3.141592653589".
- Lists, tuples, modules or datatypes with multiples members should be names in plural.

Docstring format
----------------

Use PEP 257 convention. Here is a useful example using the same style as numpy:

    """
    Descriptive explanation of the function/class.

    Parameters
    ----------
    a : Datatype. Descriptive text that explains the parameter. Then a blank line.

    b : List or tuple. Descriptive texts that span multiple lines should be broken
            to be less than 80 characters wide. The second, third and so on, lines
            of the same descriptive text should be indented two tabs or eight
            spaces.

    c : Int or Float. Note that the variable name is preceded by a space, such
            that we have "name : Datatype."

    Returns
    -------
    out1 : Datatype. A descriptive text for the output parameters. Then a blank
            line.
    
    out2 : Datatype. Note that the same rules regarding lines that wrap are used
            here as well.
    
    See Also
    --------
    other_function : Short description of what the other function does.
    or_other_class : Or of what the other class does.
    
    Examples
    --------
    >>> s = "Give doctest examples here."
    >>> s
    'Give doctest examples here.'
    >>> such_as = np.random.rand(2, 3)
    >>> such_as
    array([[ 0.51982512,  0.84506951,  0.31517025],
           [ 0.81975035,  0.6654892 ,  0.78142793]])
    """

You may omit any parts you feel are not relevant for your function.

Variable naming
---------------
Matrix: Upper case. For instance
```python
import numpy as np
X = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```

Vector with _p_ dimensions. Lower case. Shape of is (p, 1) to avoid numpy
broadcasting. For instance
```python
y = np.random.random((p, 1))
```

Commit Message
==============

Git format
----------

- Write proper sentences.
- Be descriptive.
- Use several lines if necessary, wrap at 80 characters.

Possible tag list for the first line
------------------------------------

- **ENH**: When adding or improving an existing or new class in term of capabilities,
- **COMP**: When fixing a compilation error or warning,
- **DOC**: When starting or improving the class documentation,
- **STYLE**: When enhancing the comments or coding style without any effect on the class behaviour,
- **REFAC**: When refactoring without adding new capabilities,
- **BUG**: When fixing a bug (if the bug is identified in the tracker, please add the reference),
- **INST**: When fixing issues related to installation,
- **PERF**: When improving performance,
- **TEST**: When adding or modifying a test,
- **WRG**: When correcting a warning.
