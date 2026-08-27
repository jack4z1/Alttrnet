# Python Tutorial — Sample: Control Flow

*Sample document for testing the ALTTRNET ingestion pipeline. This is
not the official Python documentation; replace with real content.*

## Introduction

Control flow statements are the building blocks that let a program
decide what to do next. Python provides a small but expressive set of
these statements, and understanding them is essential before writing
any non-trivial program. This tutorial covers conditional execution,
loops, and a few statements that appear inside loops.

## The if statement

The most fundamental control flow statement is the if statement. It
evaluates a condition and, when the condition is true, executes the
indented block that follows it. You can chain several conditions with
elif and provide a fallback with else. Only the first matching branch
runs, which keeps the logic easy to follow.

```python
value = 7
if value < 0:
    print("negative")
elif value == 0:
    print("zero")
else:
    print("positive")
```

The keyword elif is short for else if. A long chain of elif branches
can sometimes be replaced by a match statement, which compares a value
against several patterns at once. The match statement is useful when
you are comparing the same value to several constants or checking for
specific shapes of data.

## The while loop

A while loop repeats a block as long as its condition stays true. The
condition is checked before every iteration, so if it is false from the
start the body never runs. Infinite loops happen when the condition
never becomes false; make sure the body makes progress toward ending
the loop.

```python
count = 0
while count < 3:
    print(count)
    count += 1
```

## The for loop

The for statement iterates over the items of any sequence, such as a
list, a string, or the range of numbers produced by the range function.
Unlike loops in some other languages, the Python for loop does not
track an index by itself; it visits each element in order. If you do
need an index, combine range with len, or use enumerate.

```python
names = ["ada", "grace", "linus"]
for name in names:
    print(name)
```

The range function produces an arithmetic progression. The end point is
never included: range(5) produces five values starting at zero. You can
supply a start value and a step, and the step may be negative. The
object returned by range is iterable but is not a list, which saves
memory for large ranges.

## Break, continue and pass

The break statement exits the innermost enclosing loop immediately.
The continue statement skips the rest of the current iteration and
moves to the next one. The pass statement does nothing at all; it is a
placeholder for code that is not written yet but must exist
syntactically, for example an empty function body.

```python
for number in range(10):
    if number % 2 == 0:
        continue
    if number == 7:
        break
    print(number)
```

## Else clauses on loops

A for or while loop may carry an else clause that runs when the loop
finishes without hitting a break. This is a convenient way to search
for something: the else branch fires when the search failed. The else
clause belongs to the loop, not to any if statement inside the loop
body.

```python
for divisor in range(2, number):
    if number % divisor == 0:
        print("not prime")
        break
else:
    print("prime")
```

## Match statements

The match statement compares an expression to a series of patterns.
Only the first pattern that matches executes, and a pattern can bind
parts of the value to variables. Patterns can combine literals with the
or operator, unpack tuples, match against class attributes, and add a
guard condition with an if clause inside the pattern.

```python
def describe(point):
    match point:
        case (0, 0):
            return "origin"
        case (x, y):
            return f"x={x}, y={y}"
```

## Nested loops

Loops may be nested inside other loops. An inner loop runs to
completion for every iteration of the outer loop. The break statement
inside the inner loop only exits the innermost enclosing loop, so a
single break cannot terminate several nested loops at once. The
continue statement behaves the same way: it skips to the next
iteration of the innermost loop.

## Summary

Control flow gives a program its shape. The if and match statements
choose between branches, while and for loops repeat work, and break,
continue and pass refine what happens inside a loop. The else clause
on a loop is an elegant way to detect that a search found nothing.
Together these statements cover almost every decision a small program
needs to make.
