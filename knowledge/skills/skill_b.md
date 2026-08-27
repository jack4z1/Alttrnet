# Skill B — Explaining an Error Message

*Sample document for testing the ALTTRNET ingestion pipeline.*

## Purpose

This skill explains a compiler or runtime error message in plain
language and suggests a fix.

## When to use it

Use this skill when the user pastes an error message and asks what it
means or how to fix it.

## Steps

1. Identify the error type and the line it points at.
2. Read the code around that line.
3. Explain the cause in one or two sentences without jargon.
4. Suggest the smallest fix that addresses the cause.
5. Show a corrected code snippet.

## Constraints

- Do not guess; if the cause is unclear, say so.
- Keep the explanation shorter than the original message when possible.
- Only suggest fixes you can justify from the code shown.
