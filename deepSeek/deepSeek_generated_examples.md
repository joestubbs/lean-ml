# Python To Lean OEIS Examples Using DeepSeek Prover V2 7B
In this file, we collect some specific examples of using DeepSeek Prover V2 7B (DSP)
with few shot learning to generate Lean versions of Python code. 

In all examples below, DSP was given three examples of Python to Lean 
translations. See the prompts defined in the deepSeek_prover.py script. 


## A001221

*Number of distinct primes dividing n.*

Python:
```
from sympy.ntheory import primefactors
print([len(primefactors(n)) for n in range(1, 1001)])
```

DSP:
```
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Factorization.Basic

def distinctPrimeFactors (n : ℕ) : ℕ :=
  (Nat.factorization n).support.card
```

*Discussion:*
The sequence values are correct but they are off by 1, as the 
OEIS sequence uses an offset of `1`. 


## A000720

*pi(n), the number of primes <= n. Sometimes called PrimePi(n) to distinguish it from the number 3.14159...*

Python: 
```
from sympy.ntheory import primefactors
print([len(primefactors(n)) for n in range(1, 1001)])
```

DSP:
```
def primePi (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (fun x => Nat.Prime x).length
```
The code is syntactically invalid, with the error, "invalid field notation".


*Discussion:* The code is very close to being correct; the 
issue is that the outer `length` is not being applied to the
entire `filter` result. This small code change fixes
the issue:

```
def primePiFix (n : ℕ) : ℕ :=
  ((List.range (n + 1)).filter (fun x => Nat.Prime x)).length
```
This function almost generates the sequence, but the indexing
is off by 1. A complete fix is as follows:

```
def primePiFix2 (n : ℕ) : ℕ :=
  ((List.range (n + 2)).filter (fun x => Nat.Prime x)).length
```

## A001222

*Number of prime divisors of n counted with multiplicity (also called big omega of n, bigomega(n) or Omega(n))*

Python:

```
from sympy import primeomega
def a(n): return primeomega(n)
print([a(n) for n in range(1, 112)])
```

DSP:

```
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.PrimeCounting

-- Define the Lean function to compute the number of prime divisors of n with multiplicity
def a (n : ℕ) : ℕ := (Nat.factorization n).sum (fun p _ => 1)
```

In this example, the generated Lean type checks but it does 
not produce the correct sequence. Note that the use of `p` produces an 
unused variable warning. 


## Correct Examples

The following examples were generated correctly. 


### A000142: Factorial numbers

*Factorial numbers, n! = 1*2*3*4*...*n (order of symmetric group S_n, number of permutations of n letters)*


Python:
```
for i in range(1, 1000):
    y = i
    for j in range(1, i):
       y *= i - j
    print(y, "
")
```

DSP:

```
def factorial (n : ℕ) : ℕ :=
  Nat.factorial n
```