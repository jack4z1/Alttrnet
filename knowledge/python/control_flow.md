# Python Control Flow — Quick Reference

*Sample document for testing the ALTTRNET ingestion pipeline.*

## Conditionals

Use if / elif / else to run different branches depending on a
condition. The match statement is a pattern-based alternative.

## Loops

- `for` iterates over the items of a sequence.
- `while` repeats while a condition is true.
- `range(start, stop, step)` produces a sequence of numbers; the stop
  value is excluded.

## Loop control

- `break` exits the innermost loop.
- `continue` jumps to the next iteration.
- `pass` is a no-op placeholder.
- A loop's `else` clause runs when the loop ends without a `break`.

## Code example

```python
for i in range(1, 6):
    if i == 3:
        continue
    if i == 5:
        break
    print(i)
```
