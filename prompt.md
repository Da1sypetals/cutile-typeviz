Refactor this file into standalone functions:
- dump_cutileir
- dump_bytecode
- dump_typechecked_ir
- get_function_repr

Only these functions, NOTHING MORE, NOTHING LESS.

and fix everything that calls into those.
every function should return str if applicable.

Note:
USE ENGLISH.
Do NOT add a readme or document or test.
