from __future__ import annotations


BENCHMARK_PROMPTS = [
    "def ",
    "theorem ",
    "by ",
    "structure ",
    "inductive ",
    "instance ",
]


BENCHMARK_SNIPPETS = [
    "def addOne (n : Nat) : Nat :=\n  n + 1",
    "theorem add_zero_right (n : Nat) : n + 0 = n := by\n  simp",
    "structure Point where\n  x : Float\n  y : Float",
    "inductive Flag where\n  | on\n  | off",
    "instance : Inhabited Flag where\n  default := Flag.off",
]
