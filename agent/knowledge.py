"""The RAG knowledge base: short, verified notes on Python errors and concepts.

Kept as Python data (not loose files) so it is bundled into agent.exe by
PyInstaller with no extra configuration.
"""

DOCUMENTS: list[dict[str, str]] = [
    {
        "title": "NameError: name 'x' is not defined",
        "text": "A NameError means Python cannot find a variable or function with that name. "
        "Common causes: a typo in the name, using a variable before assigning it, "
        "forgetting to import a module, or using a variable defined inside a function "
        "from outside it. Fix: check the spelling and make sure the name is assigned or "
        "imported before the line that uses it.",
    },
    {
        "title": "TypeError: can only concatenate str (not 'int') to str",
        "text": "You cannot add a string and a number with +. Convert first: "
        "str(age) to turn a number into text, or int(text) to turn text into a number. "
        "Better: use an f-string, e.g. f'I am {age} years old'.",
    },
    {
        "title": "IndentationError and TabError",
        "text": "Python uses indentation to define blocks (after if, for, while, def, class). "
        "Every line in the same block must have the same indentation. Use 4 spaces per "
        "level and never mix tabs and spaces. 'expected an indented block' means a line "
        "ending with a colon is not followed by an indented line.",
    },
    {
        "title": "SyntaxError: invalid syntax",
        "text": "A SyntaxError means the code is not valid Python. Common causes: a missing "
        "colon at the end of if/for/def lines, unclosed brackets or quotes, using = instead "
        "of == in a condition, or a missing comma in a list. The error often points to the "
        "line after the real mistake, so check the line above too.",
    },
    {
        "title": "IndexError: list index out of range",
        "text": "Lists are indexed from 0 to len(list) - 1. Accessing my_list[len(my_list)] "
        "raises IndexError. In loops prefer 'for item in my_list' or "
        "'for i, item in enumerate(my_list)' instead of manual index arithmetic.",
    },
    {
        "title": "KeyError in dictionaries",
        "text": "A KeyError means the key does not exist in the dictionary. Use 'key in d' to "
        "check first, or d.get(key, default) to return a default value instead of an error.",
    },
    {
        "title": "AttributeError: object has no attribute",
        "text": "An AttributeError means the object does not have that method or attribute. "
        "Common causes: a typo (e.g. .apend instead of .append), calling a list method on "
        "a string, or a variable that is None because a function returned nothing. Use "
        "type(x) and dir(x) to inspect the object.",
    },
    {
        "title": "ZeroDivisionError",
        "text": "Dividing by zero raises ZeroDivisionError. Check the divisor before dividing "
        "(if count != 0) or catch the exception with try/except ZeroDivisionError.",
    },
    {
        "title": "ValueError: invalid literal for int()",
        "text": "int('abc') or int('3.5') raises ValueError because the text is not a whole "
        "number. Validate input with str.isdigit(), use float() for decimals, or wrap the "
        "conversion in try/except ValueError and ask the user again.",
    },
    {
        "title": "ModuleNotFoundError / ImportError",
        "text": "ModuleNotFoundError means the package is not installed in the Python you are "
        "running. Install it with 'python -m pip install package-name' using the same "
        "Python interpreter. Also make sure your own file is not named like the module "
        "(for example random.py hides the real random module).",
    },
    {
        "title": "Functions and return values",
        "text": "A function returns None unless it has a return statement. print() shows a "
        "value on screen but does not return it. If you need the result later, use return "
        "and store it: result = add(2, 3).",
    },
    {
        "title": "Mutable default arguments",
        "text": "Never use a mutable object such as a list as a default argument "
        "(def f(items=[])). The same list is shared across calls. Use items=None and "
        "create the list inside: if items is None: items = [].",
    },
    {
        "title": "Loops: for, while, range",
        "text": "range(n) gives 0 .. n-1. range(start, stop, step) stops before stop. A while "
        "loop needs a condition that eventually becomes False, otherwise it runs forever. "
        "Use break to exit early and continue to skip to the next iteration.",
    },
    {
        "title": "Lists vs tuples vs sets vs dicts",
        "text": "list: ordered, changeable, allows duplicates. tuple: ordered, unchangeable. "
        "set: unordered, unique items, fast membership checks. dict: key-value pairs with "
        "unique keys and fast lookup by key.",
    },
    {
        "title": "List comprehensions",
        "text": "A list comprehension builds a list in one line: "
        "squares = [x * x for x in range(10) if x % 2 == 0]. It is equivalent to a for loop "
        "with append, but shorter. Keep it simple; use a normal loop if it gets complex.",
    },
    {
        "title": "Exceptions: try / except / finally",
        "text": "Wrap code that may fail in try. Catch specific exceptions "
        "(except ValueError:) rather than a bare except. finally always runs, which is "
        "useful for cleanup. Use raise to signal an error from your own function.",
    },
    {
        "title": "Reading and writing files",
        "text": "Use a with block so the file is closed automatically: "
        "with open('data.txt', encoding='utf-8') as f: text = f.read(). Mode 'w' overwrites, "
        "'a' appends. FileNotFoundError means the path is wrong relative to the current "
        "working directory.",
    },
    {
        "title": "Classes and self",
        "text": "Methods receive the instance as the first parameter, conventionally named self. "
        "__init__ runs when an object is created and sets its attributes "
        "(self.name = name). Forgetting self in a method definition causes "
        "'takes 0 positional arguments but 1 was given'.",
    },
    {
        "title": "Comparing values: == vs is",
        "text": "== compares values; 'is' checks whether two names point to the same object. "
        "Use 'is' only for None (if x is None). Use == for numbers and strings.",
    },
    {
        "title": "Floating point precision",
        "text": "0.1 + 0.2 gives 0.30000000000000004 because floats are binary approximations. "
        "Use round(x, 2) for display, math.isclose(a, b) to compare floats, or the decimal "
        "module for money.",
    },
    {
        "title": "Reading a traceback",
        "text": "Read a traceback from the bottom up. The last line is the error type and "
        "message. The lines above show the call chain; the last 'File ..., line N' that "
        "points to your own file is usually where to look.",
    },
    {
        "title": "RecursionError: maximum recursion depth exceeded",
        "text": "A recursive function must have a base case that stops the recursion, and each "
        "call must move closer to it. RecursionError usually means the base case is missing "
        "or never reached.",
    },
]
