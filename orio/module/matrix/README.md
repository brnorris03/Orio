The parser is configured in `parse.py`, in the `setup` method. To regenerate the parser tables, 
simply parse a file and specify the --regen option at the end of the command line, e.g., in
the `example/` subdirectory:

	../mparser.py atax.m --regen

This will create the `lextab.py` and  `parsetab.py` file and related tables in the correct directory (`orio/module/matrix`).

To regenerate the AST class, run `_build_ast.py`.