/- 
  MicroGPT.lean — A Formally Typed Micro Transformer in Lean 4

  Implements: matmul, softmax, layer norm, single-head attention, MLP,
  residual connections, and a complete forward pass.

  Key advantage over PyTorch: tensor dimensions are checked at compile time.
  Shape mismatches are caught by the type system, not at runtime.

  To run:   lean --run MicroGPT.lean
  To check: lean MicroGPT.lean
-/

-- ============================================================
-- SECTION 1: Dependently-Typed Tensor Foundations
-- ============================================================

/-- A vector of known length. Dimensions enforced at compile time. -/
structure Vec (n : Nat) where
  data : Array Float
  h_size : data.size = n

/-- A matrix of known dimensions. Shape mismatches won't compile. -/
structure Mat (rows cols : Nat) where
  data : Array (Array Float)
  h_rows : data.size = rows
  h_cols : ∀ (i : Fin rows), (data[i.val]'(by simp [h_rows])).size = cols

/-- A sequence of vectors with known length and hidden size. -/
structure TokenSeq (len dim : Nat) where
  data : Array (Vec dim)
  h_size : data.size = len

-- ============================================================
-- SECTION 2: Vector Operations
-- ============================================================

/-- Create a vector from a generating function. -/
def Vec.ofFn (n : Nat) (f : Fin n → Float) : Vec n :=
  ⟨Array.ofFn f, by simp⟩

/-- Create a zero vector of length `n`. -/
def Vec.zeros (n : Nat) : Vec n :=
  Vec.ofFn n (fun _ => 0.0)

/-- Get element at index. -/
def Vec.get (v : Vec n) (i : Fin n) : Float :=
  v.data[i.val]'(by simp [v.h_size])

/-- Map a function over each element. -/
def Vec.map (v : Vec n) (f : Float → Float) : Vec n :=
  Vec.ofFn n (fun i => f (v.get i))

/-- Element-wise addition. -/
def Vec.add (a b : Vec n) : Vec n :=
  Vec.ofFn n (fun i => a.get i + b.get i)

/-- Element-wise multiplication. -/
def Vec.mul (a b : Vec n) : Vec n :=
  Vec.ofFn n (fun i => a.get i * b.get i)

/-- Scalar multiplication. -/
def Vec.scale (v : Vec n) (s : Float) : Vec n :=
  Vec.map v (fun x => x * s)

/-- Element-wise subtraction. -/
def Vec.sub (a b : Vec n) : Vec n :=
  Vec.ofFn n (fun i => a.get i - b.get i)

/-- Argmax over a non-empty vector. -/
def Vec.argmax (v : Vec n) (h : 0 < n) : Fin n :=
  let start : Fin n := ⟨0, h⟩
  (List.finRange n).foldl (fun best i => if v.get i > v.get best then i else best) start

/-- Dot product of two vectors. -/
def Vec.dot (a b : Vec n) : Float :=
  let prods := Array.ofFn (fun i : Fin n => a.get i * b.get i)
  prods.foldl (fun acc x => acc + x) 0.0

/-- Sum of all elements. -/
def Vec.sum (v : Vec n) : Float :=
  v.data.foldl (fun acc x => acc + x) 0.0

/-- Convert a vector to a list. -/
def Vec.toList (v : Vec n) : List Float :=
  v.data.toList

-- ============================================================
-- SECTION 3: Matrix and Sequence Operations
-- ============================================================

/-- Create a matrix from a generating function. -/
def Mat.ofFn (rows cols : Nat) (f : Fin rows → Fin cols → Float) : Mat rows cols :=
  ⟨Array.ofFn (fun i : Fin rows => Array.ofFn (fun j : Fin cols => f i j)),
    by simp,
    by intro i; simp⟩

/-- Create a zero matrix. -/
def Mat.zeros (rows cols : Nat) : Mat rows cols :=
  Mat.ofFn rows cols (fun _ _ => 0.0)

/-- Create an identity matrix. -/
def Mat.identity (n : Nat) : Mat n n :=
  Mat.ofFn n n (fun i j => if i.val = j.val then 1.0 else 0.0)

/-- Get matrix entry at `(row, col)`. -/
def Mat.get (m : Mat rows cols) (i : Fin rows) (j : Fin cols) : Float :=
  let row := m.data[i.val]'(by simp [m.h_rows])
  row[j.val]'(by simp [row, m.h_cols i])

/-- Read a row as a typed vector. -/
def Mat.row (m : Mat rows cols) (i : Fin rows) : Vec cols :=
  let row := m.data[i.val]'(by simp [m.h_rows])
  ⟨row, m.h_cols i⟩

/-- Read a column as a typed vector. -/
def Mat.col (m : Mat rows cols) (j : Fin cols) : Vec rows :=
  Vec.ofFn rows (fun i => m.get i j)

/-- Matrix-vector multiplication: `Mat(m,n) × Vec(n) → Vec(m)`. -/
def Mat.mulVec (m : Mat rows cols) (v : Vec cols) : Vec rows :=
  Vec.ofFn rows (fun i => Vec.dot (m.row i) v)

/-- Matrix-matrix multiplication: `Mat(m,n) × Mat(n,p) → Mat(m,p)`. -/
def Mat.mulMat (a : Mat m n) (b : Mat n p) : Mat m p :=
  Mat.ofFn m p (fun i j => Vec.dot (a.row i) (b.col j))

/-- Transpose: `Mat(m,n) → Mat(n,m)`. -/
def Mat.transpose (m : Mat rows cols) : Mat cols rows :=
  Mat.ofFn cols rows (fun j i => m.get i j)

/-- Map a function over each matrix entry. -/
def Mat.map (m : Mat rows cols) (f : Float → Float) : Mat rows cols :=
  Mat.ofFn rows cols (fun i j => f (m.get i j))

/-- Element-wise matrix addition. -/
def Mat.add (a b : Mat rows cols) : Mat rows cols :=
  Mat.ofFn rows cols (fun i j => a.get i j + b.get i j)

/-- Element-wise matrix subtraction. -/
def Mat.sub (a b : Mat rows cols) : Mat rows cols :=
  Mat.ofFn rows cols (fun i j => a.get i j - b.get i j)

/-- Scalar multiplication of a matrix. -/
def Mat.scale (m : Mat rows cols) (s : Float) : Mat rows cols :=
  Mat.map m (fun x => x * s)

/-- Flatten a matrix row-major into a list. -/
def Mat.toList (m : Mat rows cols) : List Float :=
  ((List.finRange rows).map (fun i => (m.row i).toList)).foldr List.append []

/-- Build a fixed-length sequence from a generating function. -/
def TokenSeq.ofFn (len dim : Nat) (f : Fin len → Vec dim) : TokenSeq len dim :=
  let data := Array.ofFn f
  ⟨data, by simp [data]⟩

/-- Read an element from a fixed-length sequence. -/
def TokenSeq.get (s : TokenSeq len dim) (i : Fin len) : Vec dim :=
  s.data[i.val]'(by simp [s.h_size])

/-- Map over a fixed-length sequence. -/
def TokenSeq.map (s : TokenSeq len dim) (f : Vec dim → Vec dim') : TokenSeq len dim' :=
  TokenSeq.ofFn len dim' (fun i => f (s.get i))

/-- Replace an element at a given index. -/
def TokenSeq.set (s : TokenSeq len dim) (idx : Fin len) (value : Vec dim) : TokenSeq len dim :=
  TokenSeq.ofFn len dim (fun i => if i = idx then value else s.get i)

/-- Element-wise addition of two fixed-length sequences. -/
def TokenSeq.add (a b : TokenSeq len dim) : TokenSeq len dim :=
  TokenSeq.ofFn len dim (fun i => Vec.add (a.get i) (b.get i))

-- ============================================================
-- SECTION 4: Activation Functions
-- ============================================================

/-- Numerically stable softmax. -/
def softmax (v : Vec n) : Vec n :=
  let maxVal :=
    if h : 0 < n then
      v.get ⟨0, h⟩
    else
      0.0
  let exps := v.map (fun x => Float.exp (x - maxVal))
  let sumExps := Vec.sum exps
  let denom := if sumExps == 0.0 then 1.0 else sumExps
  exps.map (fun x => x / denom)

/-- ReLU activation. -/
def relu (v : Vec n) : Vec n :=
  v.map (fun x => if x > 0.0 then x else 0.0)

/-- GELU activation with the common tanh approximation. -/
def gelu (v : Vec n) : Vec n :=
  v.map fun x =>
    let cubic := x * x * x
    let cdf := 0.5 * (1.0 + Float.tanh (0.7978845608 * (x + 0.044715 * cubic)))
    x * cdf

-- ============================================================
-- SECTION 4.5: Forward-Mode Autodiff
-- ============================================================

/-- A scalar value paired with its forward derivative. -/
structure DualFloat where
  val : Float
  deriv : Float

/-- Constant dual number. -/
def DualFloat.const (x : Float) : DualFloat :=
  ⟨x, 0.0⟩

/-- Variable dual number with seed derivative 1. -/
def DualFloat.var (x : Float) : DualFloat :=
  ⟨x, 1.0⟩

/-- Forward-mode addition. -/
def DualFloat.add (a b : DualFloat) : DualFloat :=
  ⟨a.val + b.val, a.deriv + b.deriv⟩

/-- Forward-mode subtraction. -/
def DualFloat.sub (a b : DualFloat) : DualFloat :=
  ⟨a.val - b.val, a.deriv - b.deriv⟩

/-- Forward-mode negation. -/
def DualFloat.neg (a : DualFloat) : DualFloat :=
  ⟨-a.val, -a.deriv⟩

/-- Forward-mode multiplication. -/
def DualFloat.mul (a b : DualFloat) : DualFloat :=
  ⟨a.val * b.val, a.deriv * b.val + a.val * b.deriv⟩

/-- Forward-mode division. -/
def DualFloat.div (a b : DualFloat) : DualFloat :=
  let denom := b.val * b.val
  ⟨a.val / b.val, (a.deriv * b.val - a.val * b.deriv) / denom⟩

/-- Forward-mode exponential. -/
def DualFloat.exp (x : DualFloat) : DualFloat :=
  let e := Float.exp x.val
  ⟨e, e * x.deriv⟩

/-- Forward-mode logarithm. -/
def DualFloat.log (x : DualFloat) : DualFloat :=
  ⟨Float.log x.val, x.deriv / x.val⟩

/-- Forward-mode square root. -/
def DualFloat.sqrt (x : DualFloat) : DualFloat :=
  let root := Float.sqrt x.val
  ⟨root, x.deriv / (2.0 * root)⟩

/-- Forward-mode hyperbolic tangent. -/
def DualFloat.tanh (x : DualFloat) : DualFloat :=
  let t := Float.tanh x.val
  ⟨t, (1.0 - t * t) * x.deriv⟩

instance : Add DualFloat where
  add := DualFloat.add

instance : Sub DualFloat where
  sub := DualFloat.sub

instance : Neg DualFloat where
  neg := DualFloat.neg

instance : Mul DualFloat where
  mul := DualFloat.mul

instance : Div DualFloat where
  div := DualFloat.div

/-- GELU lifted to dual numbers. -/
def dualGeluScalar (x : DualFloat) : DualFloat :=
  let x2 := x * x
  let x3 := x2 * x
  let inner := x + DualFloat.const 0.044715 * x3
  let tanhArg := DualFloat.const 0.7978845608 * inner
  let cdf := DualFloat.const 0.5 * (DualFloat.const 1.0 + DualFloat.tanh tanhArg)
  x * cdf

/-- Extract the derivative of a scalar function at a point via forward-mode AD. -/
def forwardDerivative (f : DualFloat → DualFloat) (x : Float) : Float :=
  (f (DualFloat.var x)).deriv

/-- Symmetric finite-difference derivative estimate. -/
def finiteDiff (f : Float → Float) (x : Float) (eps : Float := 1.0e-5) : Float :=
  let step := if eps <= 0.0 then 1.0e-5 else eps
  (f (x + step) - f (x - step)) / (2.0 * step)

/-- Symmetric finite-difference estimate of the partial derivative in `x`. -/
def finiteDiffX (f : Float → Float → Float) (x y : Float) (eps : Float := 1.0e-5) : Float :=
  let step := if eps <= 0.0 then 1.0e-5 else eps
  (f (x + step) y - f (x - step) y) / (2.0 * step)

/-- Symmetric finite-difference estimate of the partial derivative in `y`. -/
def finiteDiffY (f : Float → Float → Float) (x y : Float) (eps : Float := 1.0e-5) : Float :=
  let step := if eps <= 0.0 then 1.0e-5 else eps
  (f x (y + step) - f x (y - step)) / (2.0 * step)

/-- Read a float from a list, defaulting to `0.0` if out of bounds. -/
def getListFloat : List Float → Nat → Float
  | [], _ => 0.0
  | x :: _, 0 => x
  | _ :: xs, n + 1 => getListFloat xs n

/-- Replace one float in a list, ignoring out-of-bounds writes. -/
def setListFloat : List Float → Nat → Float → List Float
  | [], _, _ => []
  | _ :: xs, 0, value => value :: xs
  | x :: xs, n + 1, value => x :: setListFloat xs n value

/-- Symmetric finite-difference estimate of a partial derivative in a flat input list. -/
def finiteDiffList (f : List Float → Float) (xs : List Float) (idx : Nat) (eps : Float := 1.0e-5) : Float :=
  let step := if eps <= 0.0 then 1.0e-5 else eps
  let x := getListFloat xs idx
  let plus := setListFloat xs idx (x + step)
  let minus := setListFloat xs idx (x - step)
  (f plus - f minus) / (2.0 * step)

/-- Rebuild a typed vector from a flat float list, defaulting missing entries to `0.0`. -/
def vecOfList (n : Nat) (xs : List Float) : Vec n :=
  Vec.ofFn n (fun i => getListFloat xs i.val)

/-- Rebuild a typed matrix from a row-major float list, defaulting missing entries to `0.0`. -/
def matOfList (rows cols : Nat) (xs : List Float) : Mat rows cols :=
  Mat.ofFn rows cols (fun i j => getListFloat xs (i.val * cols + j.val))

/-- Numerically stable softmax over a float list. -/
def softmaxList (xs : List Float) : List Float :=
  let maxVal := xs.foldl max 0.0
  let exps := xs.map (fun x => Float.exp (x - maxVal))
  let total := exps.foldl (fun acc x => acc + x) 0.0
  let denom := if total == 0.0 then 1.0 else total
  exps.map (fun x => x / denom)

/-- Cross-entropy loss from float logits and a target index. -/
def crossEntropyLogitsList (logits : List Float) (target : Nat) : Float :=
  let probs := softmaxList logits
  let targetProb := getListFloat probs target
  let clipped := if targetProb <= 1.0e-9 then 1.0e-9 else targetProb
  0.0 - Float.log clipped

/-- Absolute value helper for reporting derivative error. -/
def floatAbs (x : Float) : Float :=
  if x < 0.0 then -x else x

-- ============================================================
-- SECTION 4.6: Reverse-Mode Autodiff
-- ============================================================

/-- Primitive scalar operations recorded on a reverse-mode tape. -/
inductive ScalarOp where
  | input
  | const
  | add (lhs rhs : Nat)
  | sub (lhs rhs : Nat)
  | mul (lhs rhs : Nat)
  | div (lhs rhs : Nat)
  | exp (arg : Nat)
  | log (arg : Nat)
  | sqrt (arg : Nat)
  | tanh (arg : Nat)

deriving Inhabited

/-- One reverse-mode tape entry with its forward value. -/
structure TapeEntry where
  val : Float
  op : ScalarOp

deriving Inhabited

/-- A scalar tracked by index inside the reverse-mode tape. -/
structure TracedFloat where
  idx : Nat
  val : Float

deriving Inhabited

abbrev Tape := Array TapeEntry
abbrev TraceM := StateM Tape

/-- Append one entry to the tape and return its traced handle. -/
def pushEntry (entry : TapeEntry) : TraceM TracedFloat := do
  let tape ← get
  let idx := tape.size
  set (tape.push entry)
  pure ⟨idx, entry.val⟩

/-- Reverse-mode input variable. -/
def traceInput (x : Float) : TraceM TracedFloat :=
  pushEntry ⟨x, ScalarOp.input⟩

/-- Reverse-mode constant. -/
def traceConst (x : Float) : TraceM TracedFloat :=
  pushEntry ⟨x, ScalarOp.const⟩

/-- Reverse-mode addition. -/
def traceAdd (a b : TracedFloat) : TraceM TracedFloat :=
  pushEntry ⟨a.val + b.val, ScalarOp.add a.idx b.idx⟩

/-- Reverse-mode subtraction. -/
def traceSub (a b : TracedFloat) : TraceM TracedFloat :=
  pushEntry ⟨a.val - b.val, ScalarOp.sub a.idx b.idx⟩

/-- Reverse-mode multiplication. -/
def traceMul (a b : TracedFloat) : TraceM TracedFloat :=
  pushEntry ⟨a.val * b.val, ScalarOp.mul a.idx b.idx⟩

/-- Reverse-mode division. -/
def traceDiv (a b : TracedFloat) : TraceM TracedFloat :=
  pushEntry ⟨a.val / b.val, ScalarOp.div a.idx b.idx⟩

/-- Reverse-mode exponential. -/
def traceExp (x : TracedFloat) : TraceM TracedFloat :=
  let value := Float.exp x.val
  pushEntry ⟨value, ScalarOp.exp x.idx⟩

/-- Reverse-mode logarithm. -/
def traceLog (x : TracedFloat) : TraceM TracedFloat :=
  pushEntry ⟨Float.log x.val, ScalarOp.log x.idx⟩

/-- Reverse-mode square root. -/
def traceSqrt (x : TracedFloat) : TraceM TracedFloat :=
  pushEntry ⟨Float.sqrt x.val, ScalarOp.sqrt x.idx⟩

/-- Reverse-mode hyperbolic tangent. -/
def traceTanh (x : TracedFloat) : TraceM TracedFloat :=
  let value := Float.tanh x.val
  pushEntry ⟨value, ScalarOp.tanh x.idx⟩

/-- Accumulate gradient mass at a tape slot. -/
def accumGrad (grads : Array Float) (idx : Nat) (delta : Float) : Array Float :=
  grads.set! idx (grads[idx]! + delta)

/-- One reverse-mode backpropagation step for a tape entry. -/
def backwardStep (tape : Tape) (idx : Nat) (grads : Array Float) : Array Float :=
  let grad := grads[idx]!
  let entry := tape[idx]!
  match TapeEntry.op entry with
  | ScalarOp.input => grads
  | ScalarOp.const => grads
  | ScalarOp.add lhs rhs =>
      let grads := accumGrad grads lhs grad
      accumGrad grads rhs grad
  | ScalarOp.sub lhs rhs =>
      let grads := accumGrad grads lhs grad
      accumGrad grads rhs (0.0 - grad)
  | ScalarOp.mul lhs rhs =>
      let lhsVal := (tape[lhs]!).val
      let rhsVal := (tape[rhs]!).val
      let grads := accumGrad grads lhs (grad * rhsVal)
      accumGrad grads rhs (grad * lhsVal)
  | ScalarOp.div lhs rhs =>
      let lhsVal := (tape[lhs]!).val
      let rhsVal := (tape[rhs]!).val
      let grads := accumGrad grads lhs (grad / rhsVal)
      let rhsGrad := grad * (0.0 - lhsVal / (rhsVal * rhsVal))
      accumGrad grads rhs rhsGrad
  | ScalarOp.exp arg =>
      accumGrad grads arg (grad * entry.val)
  | ScalarOp.log arg =>
      let argVal := (tape[arg]!).val
      accumGrad grads arg (grad / argVal)
  | ScalarOp.sqrt arg =>
      accumGrad grads arg (grad / (2.0 * entry.val))
  | ScalarOp.tanh arg =>
      accumGrad grads arg (grad * (1.0 - entry.val * entry.val))

/-- Reverse-mode backward pass seeded with output gradient 1. -/
def backwardTape (tape : Tape) (output : TracedFloat) : Array Float :=
  let init := (Array.replicate tape.size 0.0).set! output.idx 1.0
  let rec go : Nat → Array Float → Array Float
    | 0, grads => grads
    | remaining + 1, grads =>
        let grads' := backwardStep tape remaining grads
        go remaining grads'
  go tape.size init

/-- Extract the reverse-mode derivative of a scalar function at a point. -/
def reverseDerivative (f : TracedFloat → TraceM TracedFloat) (x : Float) : Float :=
  let ((input, output), tape) := Id.run <| (do
    let input ← traceInput x
    let output ← f input
    pure (input, output)).run #[]
  let grads := backwardTape tape output
  grads[input.idx]!

/-- Reverse-mode gradient for a scalar function of two scalar inputs. -/
def reverseGradient2 (f : TracedFloat → TracedFloat → TraceM TracedFloat)
    (x y : Float) : Float × Float :=
  let (((inputX, inputY), output), tape) := Id.run <| (do
    let inputX ← traceInput x
    let inputY ← traceInput y
    let output ← f inputX inputY
    pure ((inputX, inputY), output)).run #[]
  let grads := backwardTape tape output
  (grads[inputX.idx]!, grads[inputY.idx]!)

/-- GELU lifted to traced reverse-mode scalars. -/
def traceGeluScalar (x : TracedFloat) : TraceM TracedFloat := do
  let x2 ← traceMul x x
  let x3 ← traceMul x2 x
  let c1 ← traceConst 0.044715
  let cubicTerm ← traceMul c1 x3
  let inner ← traceAdd x cubicTerm
  let c2 ← traceConst 0.7978845608
  let tanhArg ← traceMul c2 inner
  let tanhOut ← traceTanh tanhArg
  let one ← traceConst 1.0
  let shifted ← traceAdd one tanhOut
  let half ← traceConst 0.5
  let cdf ← traceMul half shifted
  traceMul x cdf

/-- Reverse-mode dot product over traced scalar lists. -/
def traceDotList : List TracedFloat → List TracedFloat → TraceM TracedFloat
  | [], _ => traceConst 0.0
  | _, [] => traceConst 0.0
  | a :: as, b :: bs => do
      let prod ← traceMul a b
      let tail ← traceDotList as bs
      traceAdd prod tail

/-- Reverse-mode sum over a traced scalar list. -/
def traceSumList : List TracedFloat → TraceM TracedFloat
  | [] => traceConst 0.0
  | x :: xs => do
      let tail ← traceSumList xs
      traceAdd x tail

/-- Reverse-mode matrix-vector product over traced scalar lists. -/
def traceMatVecList : List (List TracedFloat) → List TracedFloat → TraceM (List TracedFloat)
  | [], _ => pure []
  | row :: rows, vec => do
      let head ← traceDotList row vec
      let tail ← traceMatVecList rows vec
      pure (head :: tail)

/-- Trace a list of scalar inputs onto the reverse-mode tape. -/
def traceInputs : List Float → TraceM (List TracedFloat)
  | [] => pure []
  | x :: xs => do
      let head ← traceInput x
      let tail ← traceInputs xs
      pure (head :: tail)

/-- Get a traced scalar at a list index, with a small positive fallback. -/
def traceGetOrEps : List TracedFloat → Nat → TraceM TracedFloat
  | [], _ => traceConst 1.0e-9
  | x :: _, 0 => pure x
  | _ :: xs, n + 1 => traceGetOrEps xs n

/-- Reverse-mode exponentiation over a traced scalar list. -/
def traceExpList : List TracedFloat → TraceM (List TracedFloat)
  | [] => pure []
  | x :: xs => do
      let head ← traceExp x
      let tail ← traceExpList xs
      pure (head :: tail)

/-- Reverse-mode normalization of a traced list by a shared denominator. -/
def traceNormalizeList : List TracedFloat → TracedFloat → TraceM (List TracedFloat)
  | [], _ => pure []
  | x :: xs, denom => do
      let head ← traceDiv x denom
      let tail ← traceNormalizeList xs denom
      pure (head :: tail)

/-- Reverse-mode softmax over a traced scalar list. -/
def traceSoftmaxList (logits : List TracedFloat) : TraceM (List TracedFloat) := do
  let exps ← traceExpList logits
  let total ← traceSumList exps
  traceNormalizeList exps total

/-- Reverse-mode cross-entropy from traced logits and a target index. -/
def traceCrossEntropyLogits (logits : List TracedFloat) (target : Nat) : TraceM TracedFloat := do
  let probs ← traceSoftmaxList logits
  let targetProb ← traceGetOrEps probs target
  let logged ← traceLog targetProb
  let zero ← traceConst 0.0
  traceSub zero logged

/-- Reverse-mode gradient for a scalar function of a flat input list. -/
def reverseGradientList (f : List TracedFloat → TraceM TracedFloat) (xs : List Float) : List Float :=
  let ((inputs, output), tape) := Id.run <| (do
    let inputs ← traceInputs xs
    let output ← f inputs
    pure (inputs, output)).run #[]
  let grads := backwardTape tape output
  inputs.map (fun input => grads[input.idx]!)

/-- Trace a list of constants onto the reverse-mode tape. -/
def traceConsts : List Float → TraceM (List TracedFloat)
  | [] => pure []
  | x :: xs => do
      let head ← traceConst x
      let tail ← traceConsts xs
      pure (head :: tail)

/-- Read a traced scalar from a list by index. -/
def getListTraced? : List TracedFloat → Nat → Option TracedFloat
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, n + 1 => getListTraced? xs n

/-- Get a traced scalar from a flat parameter list or fall back to zero. -/
def traceGetOrZero (xs : List TracedFloat) (idx : Nat) : TraceM TracedFloat :=
  match getListTraced? xs idx with
  | some x => pure x
  | none => traceConst 0.0

/-- A reverse-mode vector whose length is checked at compile time. -/
structure TracedVec (n : Nat) where
  data : Array TracedFloat
  h_size : data.size = n

instance : Inhabited (TracedVec n) where
  default := ⟨Array.ofFn (fun (_ : Fin n) => default), by simp⟩

/-- A reverse-mode matrix whose shape is checked at compile time. -/
structure TracedMat (rows cols : Nat) where
  data : Array (Array TracedFloat)
  h_rows : data.size = rows
  h_cols : ∀ (i : Fin rows), (data[i.val]'(by simp [h_rows])).size = cols

/-- Build a typed traced vector from a generator. -/
def TracedVec.ofFn (n : Nat) (f : Fin n → TracedFloat) : TracedVec n :=
  ⟨Array.ofFn f, by simp⟩

/-- Read a traced vector entry. -/
def TracedVec.get (v : TracedVec n) (i : Fin n) : TracedFloat :=
  v.data[i.val]'(by simp [v.h_size])

/-- Convert a float vector into a constant traced vector. -/
def traceConstVec (v : Vec n) : TraceM (TracedVec n) := do
  let traced ← traceConsts v.toList
  pure <| TracedVec.ofFn n (fun i =>
    match getListTraced? traced i.val with
    | some x => x
    | none => panic! "traceConstVec: impossible index")

/-- Convert a float matrix into a constant traced matrix. -/
def traceConstMat (m : Mat rows cols) : TraceM (TracedMat rows cols) := do
  let traced ← traceConsts m.toList
  pure <|
    ⟨Array.ofFn (fun (i : Fin rows) =>
        Array.ofFn (fun (j : Fin cols) =>
          match getListTraced? traced (i.val * cols + j.val) with
          | some x => x
          | none => panic! "traceConstMat: impossible index")),
      by simp,
      by intro i; simp⟩

/-- Read a traced vector from a list by index. -/
def getListTracedVec? {n : Nat} : List (TracedVec n) → Nat → Option (TracedVec n)
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, k + 1 => getListTracedVec? xs k

/-- Build a typed traced matrix from a generator. -/
def TracedMat.ofFn (rows cols : Nat) (f : Fin rows → Fin cols → TracedFloat) : TracedMat rows cols :=
  ⟨Array.ofFn (fun i : Fin rows => Array.ofFn (fun j : Fin cols => f i j)),
    by simp,
    by intro i; simp⟩

/-- Read a traced matrix entry. -/
def TracedMat.get (m : TracedMat rows cols) (i : Fin rows) (j : Fin cols) : TracedFloat :=
  let row := m.data[i.val]'(by simp [m.h_rows])
  row[j.val]'(by simp [row, m.h_cols i])

/-- Convert a flat parameter list into a typed traced matrix. -/
def traceMatFromFlat (rows cols : Nat) (params : List TracedFloat) : TracedMat rows cols :=
  TracedMat.ofFn rows cols (fun i j =>
    match getListTraced? params (i.val * cols + j.val) with
    | some x => x
    | none => panic! "traceMatFromFlat: impossible index")

/-- Convert a flat parameter list into a typed traced matrix starting at an offset. -/
def traceMatFromFlatOffset (rows cols offset : Nat) (params : List TracedFloat) : TracedMat rows cols :=
  TracedMat.ofFn rows cols (fun i j =>
    match getListTraced? params (offset + i.val * cols + j.val) with
    | some x => x
    | none => panic! "traceMatFromFlatOffset: impossible index")

/-- Convert a flat parameter list into a typed traced vector starting at an offset. -/
def traceVecFromFlat (n offset : Nat) (params : List TracedFloat) : TracedVec n :=
  TracedVec.ofFn n (fun i =>
    match getListTraced? params (offset + i.val) with
    | some x => x
    | none => panic! "traceVecFromFlat: impossible index")

/-- Reverse-mode dot product over typed traced vectors. -/
def traceDotVec (a b : TracedVec n) : TraceM TracedFloat := do
  let zero ← traceConst 0.0
  (List.finRange n).foldlM (fun acc i => do
    let prod ← traceMul (a.get i) (b.get i)
    traceAdd acc prod
  ) zero

/-- Reverse-mode element-wise addition over typed traced vectors. -/
def traceAddVec (a b : TracedVec n) : TraceM (TracedVec n) := do
  let entries ← (List.finRange n).mapM (fun i => traceAdd (a.get i) (b.get i))
  pure <| TracedVec.ofFn n (fun i =>
    match getListTraced? entries i.val with
    | some x => x
    | none => panic! "traceAddVec: impossible index")

/-- Reverse-mode element-wise subtraction over typed traced vectors. -/
def traceSubVec (a b : TracedVec n) : TraceM (TracedVec n) := do
  let entries ← (List.finRange n).mapM (fun i => traceSub (a.get i) (b.get i))
  pure <| TracedVec.ofFn n (fun i =>
    match getListTraced? entries i.val with
    | some x => x
    | none => panic! "traceSubVec: impossible index")

/-- Reverse-mode element-wise multiplication over typed traced vectors. -/
def traceMulVec (a b : TracedVec n) : TraceM (TracedVec n) := do
  let entries ← (List.finRange n).mapM (fun i => traceMul (a.get i) (b.get i))
  pure <| TracedVec.ofFn n (fun i =>
    match getListTraced? entries i.val with
    | some x => x
    | none => panic! "traceMulVec: impossible index")

/-- Reverse-mode vector divided by a shared scalar. -/
def traceDivVecScalar (v : TracedVec n) (s : TracedFloat) : TraceM (TracedVec n) := do
  let entries ← (List.finRange n).mapM (fun i => traceDiv (v.get i) s)
  pure <| TracedVec.ofFn n (fun i =>
    match getListTraced? entries i.val with
    | some x => x
    | none => panic! "traceDivVecScalar: impossible index")

/-- Reverse-mode vector scaled by a shared scalar. -/
def traceScaleVec (v : TracedVec n) (s : TracedFloat) : TraceM (TracedVec n) := do
  let entries ← (List.finRange n).mapM (fun i => traceMul (v.get i) s)
  pure <| TracedVec.ofFn n (fun i =>
    match getListTraced? entries i.val with
    | some x => x
    | none => panic! "traceScaleVec: impossible index")

/-- Reverse-mode sum of a typed traced vector. -/
def traceSumVec (v : TracedVec n) : TraceM TracedFloat := do
  let zero ← traceConst 0.0
  (List.finRange n).foldlM (fun acc i => traceAdd acc (v.get i)) zero

/-- Reverse-mode matrix-vector product over typed traced tensors. -/
def traceMatVec (m : TracedMat rows cols) (v : TracedVec cols) : TraceM (TracedVec rows) := do
  let outputs ← (List.finRange rows).mapM (fun i =>
    traceDotVec
      (TracedVec.ofFn cols (fun j => m.get i j))
      v)
  pure <| TracedVec.ofFn rows (fun i =>
    match getListTraced? outputs i.val with
    | some x => x
    | none => panic! "traceMatVec: impossible index")

/-- Reverse-mode softmax over a typed traced vector. -/
def traceSoftmaxVec (logits : TracedVec n) : TraceM (TracedVec n) := do
  let probs ← traceSoftmaxList ((List.finRange n).map (fun i => logits.get i))
  pure <| TracedVec.ofFn n (fun i =>
    match getListTraced? probs i.val with
    | some x => x
    | none => panic! "traceSoftmaxVec: impossible index")

/-- Reverse-mode cross-entropy from typed traced logits. -/
def traceCrossEntropyVec (logits : TracedVec n) (target : Fin n) : TraceM TracedFloat :=
  traceCrossEntropyLogits ((List.finRange n).map (fun i => logits.get i)) target.val

/-- GELU lifted over typed traced vectors. -/
def traceGeluVec (v : TracedVec n) : TraceM (TracedVec n) := do
  let entries ← (List.finRange n).mapM (fun i => traceGeluScalar (v.get i))
  pure <| TracedVec.ofFn n (fun i =>
    match getListTraced? entries i.val with
    | some x => x
    | none => panic! "traceGeluVec: impossible index")

/-- Reverse-mode layer norm with traced gamma/beta and a constant epsilon. -/
def traceLayerNorm (gamma beta input : TracedVec dim) (eps : Float := 1.0e-5) :
    TraceM (TracedVec dim) := do
  let sum ← traceSumVec input
  let denomVal := if dim == 0 then 1.0 else Float.ofNat dim
  let denom ← traceConst denomVal
  let mean ← traceDiv sum denom
  let meanVec := TracedVec.ofFn dim (fun _ => mean)
  let diffs ← traceSubVec input meanVec
  let sqDiffs ← traceMulVec diffs diffs
  let varianceSum ← traceSumVec sqDiffs
  let variance ← traceDiv varianceSum denom
  let epsConst ← traceConst eps
  let varianceEps ← traceAdd variance epsConst
  let std ← traceSqrt varianceEps
  let normalized ← traceDivVecScalar diffs std
  let scaled ← traceMulVec gamma normalized
  traceAddVec scaled beta

-- ============================================================
-- SECTION 5: Layer Normalization
-- ============================================================

/-- Layer normalization with learnable parameters. -/
structure LayerNormParams (dim : Nat) where
  gamma : Vec dim
  beta  : Vec dim

/-- Initialize gamma to ones and beta to zeros. -/
def LayerNormParams.ones (dim : Nat) : LayerNormParams dim :=
  { gamma := Vec.ofFn dim (fun _ => 1.0)
  , beta := Vec.zeros dim
  }

/-- Convert a natural number to a denominator-safe float. -/
def safeNatDenom (n : Nat) : Float :=
  if n == 0 then 1.0 else Float.ofNat n

/-- Layer normalization. -/
def layerNorm (params : LayerNormParams dim) (v : Vec dim) (eps : Float := 1e-5) : Vec dim :=
  let denom := safeNatDenom dim
  let mean := Vec.sum v / denom
  let diffs := v.map (fun x => x - mean)
  let variance := Vec.sum (Vec.mul diffs diffs) / denom
  let std := Float.sqrt (variance + eps)
  let normalized := diffs.map (fun x => x / std)
  Vec.add (Vec.mul params.gamma normalized) params.beta

/-- Fixed sinusoidal encoding for a single position. -/
def sinusoidalPosition (dim pos : Nat) : Vec dim :=
  Vec.ofFn dim fun i =>
    let pairIdx := i.val / 2
    let exponent := Float.ofNat (2 * pairIdx) / safeNatDenom dim
    let angle := Float.ofNat pos / Float.pow 10000.0 exponent
    if i.val % 2 = 0 then Float.sin angle else Float.cos angle

/-- Fixed sinusoidal encoding for a whole sequence length. -/
def sinusoidalEncoding (len dim : Nat) : TokenSeq len dim :=
  TokenSeq.ofFn len dim (fun i => sinusoidalPosition dim i.val)

-- ============================================================
-- SECTION 6: Attention Mechanism
-- ============================================================

/-- Single attention head with dimensionally-typed weights. -/
structure AttentionHead (d_model d_head : Nat) where
  wq : Mat d_head d_model
  wk : Mat d_head d_model
  wv : Mat d_head d_model

/-- Multi-head attention with a compile-time split `d_model = n_heads * d_head`. -/
structure MultiHeadAttention (d_model n_heads d_head : Nat) where
  heads : Fin n_heads → AttentionHead d_model d_head
  wo : Mat d_model (n_heads * d_head)
  h_split : d_model = n_heads * d_head
  h_headDim : 0 < d_head

/-- Large negative value used to zero-out masked attention weights after softmax. -/
def causalMaskPenalty : Float := -1.0e9

/-- Scaled dot-product attention for one head over a fixed-length sequence. -/
def attention (head : AttentionHead d_model d_head)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (causal : Bool := false) : Vec d_head :=
  let queryToken := sequence.get queryIdx
  let q := Mat.mulVec head.wq queryToken

  let keys : TokenSeq seqLen d_head := TokenSeq.map sequence (Mat.mulVec head.wk)
  let values : TokenSeq seqLen d_head := TokenSeq.map sequence (Mat.mulVec head.wv)

  let scaleRaw := Float.sqrt (Float.ofNat d_head)
  let scale := if scaleRaw == 0.0 then 1.0 else scaleRaw
  let scores : Vec seqLen :=
    Vec.ofFn seqLen fun i =>
      let raw := Vec.dot q (keys.get i) / scale
      if causal then
        if queryIdx.val < i.val then causalMaskPenalty else raw
      else
        raw
  let attnWeights := softmax scores

  let contributions : TokenSeq seqLen d_head :=
    TokenSeq.ofFn seqLen d_head (fun i => Vec.scale (values.get i) (attnWeights.get i))
  contributions.data.foldl (fun acc v => Vec.add acc v) (Vec.zeros d_head)

/-- Concatenate per-head outputs into a model-width vector. -/
def concatHeadOutputs (outputs : Fin n_heads → Vec d_head) (h_headDim : 0 < d_head) :
    Vec (n_heads * d_head) :=
  Vec.ofFn (n_heads * d_head) fun i =>
    let headIdx : Fin n_heads :=
      ⟨i.val / d_head, by
        rw [Nat.div_lt_iff_lt_mul h_headDim]
        exact i.isLt⟩
    let offset : Fin d_head :=
      ⟨i.val % d_head, Nat.mod_lt _ h_headDim⟩
    (outputs headIdx).get offset

/-- Multi-head attention = per-head attention, concatenate, output projection. -/
def multiHeadAttention (mha : MultiHeadAttention d_model n_heads d_head)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (causal : Bool := false) : Vec d_model :=
  let concatenated :=
    concatHeadOutputs (fun h => attention (mha.heads h) sequence queryIdx causal) mha.h_headDim
  Mat.mulVec mha.wo concatenated

-- ============================================================
-- SECTION 7: Transformer Layer (Full)
-- ============================================================

/-- Complete transformer layer: attention + MLP + residuals + layer norm. -/
structure TransformerLayer (d_model n_heads d_head d_ff : Nat) where
  ln1     : LayerNormParams d_model
  attn    : MultiHeadAttention d_model n_heads d_head
  ln2     : LayerNormParams d_model
  mlp_fc1 : Mat d_ff d_model
  mlp_fc2 : Mat d_model d_ff

/-- MLP sub-layer: up-project → GELU → down-project. -/
def mlpForward (fc1 : Mat d_ff d_model) (fc2 : Mat d_model d_ff)
    (x : Vec d_model) : Vec d_model :=
  let hidden := gelu (Mat.mulVec fc1 x)
  Mat.mulVec fc2 hidden

/-- Full transformer layer forward pass for one token position. -/
def transformerForward (layer : TransformerLayer d_model n_heads d_head d_ff)
    (sequence : TokenSeq seqLen d_model)
    (tokenIdx : Fin seqLen) : Vec d_model :=
  let x := sequence.get tokenIdx

  let normedSeq := TokenSeq.map sequence (layerNorm layer.ln1)
  let attnOut := multiHeadAttention layer.attn normedSeq tokenIdx true
  let x' := Vec.add x attnOut

  let normed2 := layerNorm layer.ln2 x'
  let mlpOut := mlpForward layer.mlp_fc1 layer.mlp_fc2 normed2
  Vec.add x' mlpOut

/-- Apply a transformer layer to every token position in a sequence. -/
def applyLayer (layer : TransformerLayer d_model n_heads d_head d_ff)
    (sequence : TokenSeq seqLen d_model) : TokenSeq seqLen d_model :=
  TokenSeq.ofFn seqLen d_model (fun i => transformerForward layer sequence i)

-- ============================================================
-- SECTION 8: Full Model
-- ============================================================

/-- MicroGPT model configuration. -/
structure GPTConfig where
  d_model  : Nat := 64
  n_heads  : Nat := 4
  d_head   : Nat := 16
  d_ff     : Nat := 256
  n_layers : Nat := 2
  vocab    : Nat := 256

/-- Complete MicroGPT model. -/
structure MicroGPT (cfg : GPTConfig) where
  embedding : Mat cfg.d_model cfg.vocab
  layers    : List (TransformerLayer cfg.d_model cfg.n_heads cfg.d_head cfg.d_ff)
  h_layers  : layers.length = cfg.n_layers
  ln_final  : LayerNormParams cfg.d_model
  unembed   : Mat cfg.vocab cfg.d_model

/-- Embed a token list into a fixed-length sequence of vectors. -/
def embedTokens (model : MicroGPT cfg) (tokenIds : List (Fin cfg.vocab)) :
    TokenSeq tokenIds.length cfg.d_model :=
  TokenSeq.ofFn tokenIds.length cfg.d_model fun i =>
    let tokId := tokenIds.get i
    Vec.ofFn cfg.d_model fun rowIdx =>
      model.embedding.get rowIdx tokId

/-- Input sequence seen by the transformer stack: embeddings plus positional encodings. -/
def MicroGPT.inputSequence (model : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab)) : TokenSeq tokenIds.length cfg.d_model :=
  let embedded := embedTokens model tokenIds
  TokenSeq.add embedded (sinusoidalEncoding tokenIds.length cfg.d_model)

/-- Final transformer block output before the learned output head. -/
def MicroGPT.preFinalState (model : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab))
    (predictIdx : Fin tokenIds.length) : Vec cfg.d_model :=
  let finalSeq := model.layers.foldl (fun seq layer => applyLayer layer seq) (model.inputSequence tokenIds)
  finalSeq.get predictIdx

/-- Final normalized hidden state before unembedding. -/
def MicroGPT.hiddenState (model : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab))
    (predictIdx : Fin tokenIds.length) : Vec cfg.d_model :=
  layerNorm model.ln_final (model.preFinalState tokenIds predictIdx)

/-- Forward pass through the full model, returning vocabulary logits. -/
def MicroGPT.forward (model : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab))
    (predictIdx : Fin tokenIds.length) : Vec cfg.vocab :=
  Mat.mulVec model.unembed (model.hiddenState tokenIds predictIdx)

/-- Inference helper returning probabilities over the vocabulary. -/
def MicroGPT.forwardProbs (model : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab))
    (predictIdx : Fin tokenIds.length) : Vec cfg.vocab :=
  softmax (model.forward tokenIds predictIdx)

/-- Cross-entropy loss computed from logits and a target token. -/
def crossEntropyLoss (logits : Vec vocab) (target : Fin vocab) : Float :=
  let probs := softmax logits
  let targetProb := probs.get target
  let clipped := if targetProb <= 1.0e-9 then 1.0e-9 else targetProb
  0.0 - Float.log clipped

/-- A tiny trainable linear classifier used to connect AD to optimization. -/
structure LinearClassifier (inputDim vocab : Nat) where
  weight : Mat vocab inputDim
  bias : Vec vocab

/-- Typed gradients for the linear classifier parameters. -/
structure LinearClassifierGrads (inputDim vocab : Nat) where
  weight : Mat vocab inputDim
  bias : Vec vocab

/-- Linear classifier forward pass: `Wx + b`. -/
def linearClassifierForward (model : LinearClassifier inputDim vocab)
    (input : Vec inputDim) : Vec vocab :=
  Vec.add (Mat.mulVec model.weight input) model.bias

/-- Cross-entropy loss for the linear classifier. -/
def linearClassifierLoss (model : LinearClassifier inputDim vocab)
    (input : Vec inputDim) (target : Fin vocab) : Float :=
  crossEntropyLoss (linearClassifierForward model input) target

/-- Flatten linear classifier parameters row-major for tape-based reverse-mode. -/
def flattenLinearClassifier (model : LinearClassifier inputDim vocab) : List Float :=
  model.weight.toList ++ model.bias.toList

/-- Rebuild a linear classifier from row-major flattened parameters. -/
def linearClassifierOfList (inputDim vocab : Nat) (xs : List Float) :
    LinearClassifier inputDim vocab :=
  let weightCount := vocab * inputDim
  { weight := matOfList vocab inputDim (xs.take weightCount)
  , bias := vecOfList vocab (xs.drop weightCount)
  }

/-- Rebuild typed linear-classifier gradients from a flattened parameter gradient list. -/
def linearClassifierGradsOfList (inputDim vocab : Nat) (xs : List Float) :
    LinearClassifierGrads inputDim vocab :=
  let weightCount := vocab * inputDim
  { weight := matOfList vocab inputDim (xs.take weightCount)
  , bias := vecOfList vocab (xs.drop weightCount)
  }

/-- Flatten typed linear-classifier gradients for comparison/debug printing. -/
def flattenLinearClassifierGrads (grads : LinearClassifierGrads inputDim vocab) : List Float :=
  grads.weight.toList ++ grads.bias.toList

/-- Trace the logits of a linear classifier whose parameters live in a flat tape input list. -/
def traceLinearClassifierLogitsFlat (vocab : Nat) (input : Vec inputDim)
    (params : List TracedFloat) : TraceM (List TracedFloat) := do
  let inputVec ← traceConstVec input
  let weight := traceMatFromFlat vocab inputDim params
  let bias := traceVecFromFlat vocab (vocab * inputDim) params
  let logits ← traceMatVec weight inputVec
  let shifted ← traceAddVec logits bias
  pure ((List.finRange vocab).map (fun i => shifted.get i))

/-- Trace the linear classifier loss with respect to flattened parameters. -/
def traceLinearClassifierLossFlat (input : Vec inputDim) (target : Fin vocab)
    (params : List TracedFloat) : TraceM TracedFloat := do
  let logits ← traceLinearClassifierLogitsFlat vocab input params
  traceCrossEntropyLogits logits target.val

/-- Reverse-mode gradient of linear classifier loss with typed parameter output. -/
def reverseGradientLinearClassifier (model : LinearClassifier inputDim vocab)
    (input : Vec inputDim) (target : Fin vocab) :
    LinearClassifierGrads inputDim vocab :=
  let flatParams := flattenLinearClassifier model
  let flatGrads := reverseGradientList (traceLinearClassifierLossFlat input target) flatParams
  linearClassifierGradsOfList inputDim vocab flatGrads

/-- Finite-difference gradient of linear classifier loss with typed parameter output. -/
def finiteDiffLinearClassifier (model : LinearClassifier inputDim vocab)
    (input : Vec inputDim) (target : Fin vocab) :
    LinearClassifierGrads inputDim vocab :=
  let flatParams := flattenLinearClassifier model
  let lossFromFlat := fun xs => linearClassifierLoss (linearClassifierOfList inputDim vocab xs) input target
  let flatGrads := (List.range flatParams.length).map (fun idx => finiteDiffList lossFromFlat flatParams idx)
  linearClassifierGradsOfList inputDim vocab flatGrads

/-- One SGD step on the linear classifier parameters. -/
def sgdStepLinearClassifier (model : LinearClassifier inputDim vocab)
    (grads : LinearClassifierGrads inputDim vocab) (lr : Float) :
    LinearClassifier inputDim vocab :=
  { weight := Mat.sub model.weight (Mat.scale grads.weight lr)
  , bias := Vec.sub model.bias (Vec.scale grads.bias lr)
  }

/-- Repeated SGD steps on a single training example. -/
def trainLinearClassifierSteps (model : LinearClassifier inputDim vocab)
    (input : Vec inputDim) (target : Fin vocab) (lr : Float) : Nat → LinearClassifier inputDim vocab
  | 0 => model
  | steps + 1 =>
      let grads := reverseGradientLinearClassifier model input target
      trainLinearClassifierSteps (sgdStepLinearClassifier model grads lr) input target lr steps

/-- Zero-initialized linear classifier. -/
def zeroLinearClassifier (inputDim vocab : Nat) : LinearClassifier inputDim vocab :=
  { weight := Mat.zeros vocab inputDim
  , bias := Vec.zeros vocab
  }

/-- Train a fresh output head on top of a frozen MicroGPT hidden state. -/
def trainFrozenOutputHead (model : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab))
    (predictIdx : Fin tokenIds.length)
    (target : Fin cfg.vocab)
    (lr : Float) (steps : Nat) : LinearClassifier cfg.d_model cfg.vocab :=
  let hidden := model.hiddenState tokenIds predictIdx
  trainLinearClassifierSteps (zeroLinearClassifier cfg.d_model cfg.vocab) hidden target lr steps

/-- Trainable model output head: final layer norm plus unembedding matrix. -/
structure OutputHeadParams (d_model vocab : Nat) where
  ln : LayerNormParams d_model
  unembed : Mat vocab d_model

/-- Typed gradients for the model output head. -/
structure OutputHeadGrads (d_model vocab : Nat) where
  ln : LayerNormParams d_model
  unembed : Mat vocab d_model

/-- Extract the trainable output head from a model. -/
def outputHeadFromModel (model : MicroGPT cfg) : OutputHeadParams cfg.d_model cfg.vocab :=
  { ln := model.ln_final
  , unembed := model.unembed
  }

/-- Replace the trainable output head of a model. -/
def MicroGPT.withOutputHead (model : MicroGPT cfg)
    (head : OutputHeadParams cfg.d_model cfg.vocab) : MicroGPT cfg :=
  { embedding := model.embedding
  , layers := model.layers
  , h_layers := model.h_layers
  , ln_final := head.ln
  , unembed := head.unembed
  }

/-- Output-head forward pass: layer norm followed by unembedding. -/
def outputHeadForward (params : OutputHeadParams d_model vocab)
    (state : Vec d_model) : Vec vocab :=
  Mat.mulVec params.unembed (layerNorm params.ln state)

/-- Cross-entropy loss for the trainable output head. -/
def outputHeadLoss (params : OutputHeadParams d_model vocab)
    (state : Vec d_model) (target : Fin vocab) : Float :=
  crossEntropyLoss (outputHeadForward params state) target

/-- Flatten output-head parameters into `gamma ++ beta ++ unembed`. -/
def flattenOutputHeadParams (params : OutputHeadParams d_model vocab) : List Float :=
  params.ln.gamma.toList ++ params.ln.beta.toList ++ params.unembed.toList

/-- Rebuild output-head parameters from `gamma ++ beta ++ unembed`. -/
def outputHeadParamsOfList (d_model vocab : Nat) (xs : List Float) : OutputHeadParams d_model vocab :=
  let gamma := vecOfList d_model xs
  let beta := vecOfList d_model (xs.drop d_model)
  let weight := matOfList vocab d_model (xs.drop (2 * d_model))
  { ln := { gamma := gamma, beta := beta }
  , unembed := weight
  }

/-- Rebuild typed output-head gradients from a flattened gradient list. -/
def outputHeadGradsOfList (d_model vocab : Nat) (xs : List Float) : OutputHeadGrads d_model vocab :=
  let gamma := vecOfList d_model xs
  let beta := vecOfList d_model (xs.drop d_model)
  let weight := matOfList vocab d_model (xs.drop (2 * d_model))
  { ln := { gamma := gamma, beta := beta }
  , unembed := weight
  }

/-- Flatten typed output-head gradients for comparison/debug printing. -/
def flattenOutputHeadGrads (grads : OutputHeadGrads d_model vocab) : List Float :=
  grads.ln.gamma.toList ++ grads.ln.beta.toList ++ grads.unembed.toList

/-- Trace output-head logits with trainable `layerNorm + unembed` parameters. -/
def traceOutputHeadLogitsFlat (d_model vocab : Nat) (state : Vec d_model)
    (params : List TracedFloat) : TraceM (TracedVec vocab) := do
  let gamma := traceVecFromFlat d_model 0 params
  let beta := traceVecFromFlat d_model d_model params
  let unembed := traceMatFromFlatOffset vocab d_model (2 * d_model) params
  let stateVec ← traceConstVec state
  let normed ← traceLayerNorm gamma beta stateVec
  traceMatVec unembed normed

/-- Trace output-head loss with trainable `layerNorm + unembed` parameters. -/
def traceOutputHeadLossFlat (state : Vec d_model) (target : Fin vocab)
    (params : List TracedFloat) : TraceM TracedFloat := do
  let logits ← traceOutputHeadLogitsFlat d_model vocab state params
  traceCrossEntropyVec logits target

/-- Reverse-mode gradient of the output-head loss with typed parameter output. -/
def reverseGradientOutputHead (params : OutputHeadParams d_model vocab)
    (state : Vec d_model) (target : Fin vocab) : OutputHeadGrads d_model vocab :=
  let flatParams := flattenOutputHeadParams params
  let flatGrads := reverseGradientList (traceOutputHeadLossFlat state target) flatParams
  outputHeadGradsOfList d_model vocab flatGrads

/-- Finite-difference gradient of the output-head loss with typed parameter output. -/
def finiteDiffOutputHead (params : OutputHeadParams d_model vocab)
    (state : Vec d_model) (target : Fin vocab) : OutputHeadGrads d_model vocab :=
  let flatParams := flattenOutputHeadParams params
  let lossFromFlat := fun xs => outputHeadLoss (outputHeadParamsOfList d_model vocab xs) state target
  let flatGrads := (List.range flatParams.length).map (fun idx => finiteDiffList lossFromFlat flatParams idx)
  outputHeadGradsOfList d_model vocab flatGrads

/-- One SGD step on the model output head. -/
def sgdStepOutputHead (params : OutputHeadParams d_model vocab)
    (grads : OutputHeadGrads d_model vocab) (lr : Float) : OutputHeadParams d_model vocab :=
  { ln :=
      { gamma := Vec.sub params.ln.gamma (Vec.scale grads.ln.gamma lr)
      , beta := Vec.sub params.ln.beta (Vec.scale grads.ln.beta lr)
      }
  , unembed := Mat.sub params.unembed (Mat.scale grads.unembed lr)
  }

/-- Repeated SGD steps on a frozen transformer state for the model output head. -/
def trainOutputHeadSteps (params : OutputHeadParams d_model vocab)
    (state : Vec d_model) (target : Fin vocab) (lr : Float) : Nat → OutputHeadParams d_model vocab
  | 0 => params
  | steps + 1 =>
      let grads := reverseGradientOutputHead params state target
      trainOutputHeadSteps (sgdStepOutputHead params grads lr) state target lr steps

/-- Train the actual model output head on a frozen transformer state. -/
def trainModelOutputHead (model : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab))
    (predictIdx : Fin tokenIds.length)
    (target : Fin cfg.vocab)
    (lr : Float) (steps : Nat) : MicroGPT cfg :=
  let state := model.preFinalState tokenIds predictIdx
  let trainedHead := trainOutputHeadSteps (outputHeadFromModel model) state target lr steps
  model.withOutputHead trainedHead

/-- MLP block parameters: pre-MLP layer norm plus two linear maps. -/
structure MLPBlockParams (d_model d_ff : Nat) where
  ln : LayerNormParams d_model
  fc1 : Mat d_ff d_model
  fc2 : Mat d_model d_ff

/-- Typed gradients for the MLP block parameters. -/
structure MLPBlockGrads (d_model d_ff : Nat) where
  ln : LayerNormParams d_model
  fc1 : Mat d_ff d_model
  fc2 : Mat d_model d_ff

/-- MLP block forward pass with residual connection. -/
def mlpBlockForward (params : MLPBlockParams d_model d_ff) (state : Vec d_model) : Vec d_model :=
  let normed := layerNorm params.ln state
  let hidden := gelu (Mat.mulVec params.fc1 normed)
  let mlpOut := Mat.mulVec params.fc2 hidden
  Vec.add state mlpOut

/-- Cross-entropy loss for an MLP block feeding a fixed output head. -/
def mlpBlockLoss (params : MLPBlockParams d_model d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (state : Vec d_model) (target : Fin vocab) : Float :=
  outputHeadLoss outputHead (mlpBlockForward params state) target

/-- Flatten MLP block parameters into `gamma ++ beta ++ fc1 ++ fc2`. -/
def flattenMLPBlockParams (params : MLPBlockParams d_model d_ff) : List Float :=
  params.ln.gamma.toList ++ params.ln.beta.toList ++ params.fc1.toList ++ params.fc2.toList

/-- Rebuild MLP block parameters from `gamma ++ beta ++ fc1 ++ fc2`. -/
def mlpBlockParamsOfList (d_model d_ff : Nat) (xs : List Float) : MLPBlockParams d_model d_ff :=
  let gamma := vecOfList d_model xs
  let beta := vecOfList d_model (xs.drop d_model)
  let fc1 := matOfList d_ff d_model (xs.drop (2 * d_model))
  let fc2 := matOfList d_model d_ff (xs.drop (2 * d_model + d_ff * d_model))
  { ln := { gamma := gamma, beta := beta }
  , fc1 := fc1
  , fc2 := fc2
  }

/-- Rebuild typed MLP block gradients from a flat gradient list. -/
def mlpBlockGradsOfList (d_model d_ff : Nat) (xs : List Float) : MLPBlockGrads d_model d_ff :=
  let gamma := vecOfList d_model xs
  let beta := vecOfList d_model (xs.drop d_model)
  let fc1 := matOfList d_ff d_model (xs.drop (2 * d_model))
  let fc2 := matOfList d_model d_ff (xs.drop (2 * d_model + d_ff * d_model))
  { ln := { gamma := gamma, beta := beta }
  , fc1 := fc1
  , fc2 := fc2
  }

/-- Flatten typed MLP block gradients for comparison/debug printing. -/
def flattenMLPBlockGrads (grads : MLPBlockGrads d_model d_ff) : List Float :=
  grads.ln.gamma.toList ++ grads.ln.beta.toList ++ grads.fc1.toList ++ grads.fc2.toList

/-- Trace a fixed output head applied to a traced state. -/
def traceOutputHeadForwardConst (head : OutputHeadParams d_model vocab)
    (state : TracedVec d_model) : TraceM (TracedVec vocab) := do
  let gamma ← traceConstVec head.ln.gamma
  let beta ← traceConstVec head.ln.beta
  let unembed ← traceConstMat head.unembed
  let normed ← traceLayerNorm gamma beta state
  traceMatVec unembed normed

/-- Trace MLP block output starting from an already traced state. -/
def traceMLPBlockForwardTraced (d_model d_ff : Nat)
    (state : TracedVec d_model)
    (params : List TracedFloat) : TraceM (TracedVec d_model) := do
  let gamma := traceVecFromFlat d_model 0 params
  let beta := traceVecFromFlat d_model d_model params
  let fc1 := traceMatFromFlatOffset d_ff d_model (2 * d_model) params
  let fc2 := traceMatFromFlatOffset d_model d_ff (2 * d_model + d_ff * d_model) params
  let normed ← traceLayerNorm gamma beta state
  let hiddenPre ← traceMatVec fc1 normed
  let hidden ← traceGeluVec hiddenPre
  let mlpOut ← traceMatVec fc2 hidden
  traceAddVec state mlpOut

/-- Trace MLP block output with trainable `ln + fc1 + fc2` parameters. -/
def traceMLPBlockForwardFlat (d_model d_ff : Nat) (state : Vec d_model)
    (params : List TracedFloat) : TraceM (TracedVec d_model) := do
  let stateVec ← traceConstVec state
  traceMLPBlockForwardTraced d_model d_ff stateVec params

/-- Trace MLP block loss against a fixed output head. -/
def traceMLPBlockLossFlat {d_model d_ff vocab : Nat} (state : Vec d_model)
    (outputHead : OutputHeadParams d_model vocab)
    (target : Fin vocab)
    (params : List TracedFloat) : TraceM TracedFloat := do
  let newState ← traceMLPBlockForwardFlat d_model d_ff state params
  let logits ← traceOutputHeadForwardConst outputHead newState
  traceCrossEntropyVec logits target

/-- Reverse-mode gradient of the MLP block loss with typed parameter output. -/
def reverseGradientMLPBlock (params : MLPBlockParams d_model d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (state : Vec d_model)
    (target : Fin vocab) : MLPBlockGrads d_model d_ff :=
  let flatParams := flattenMLPBlockParams params
  let flatGrads := reverseGradientList (fun xs => traceMLPBlockLossFlat (d_ff := d_ff) state outputHead target xs) flatParams
  mlpBlockGradsOfList d_model d_ff flatGrads

/-- Finite-difference gradient of the MLP block loss with typed parameter output. -/
def finiteDiffMLPBlock (params : MLPBlockParams d_model d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (state : Vec d_model)
    (target : Fin vocab) : MLPBlockGrads d_model d_ff :=
  let flatParams := flattenMLPBlockParams params
  let lossFromFlat := fun xs => mlpBlockLoss (mlpBlockParamsOfList d_model d_ff xs) outputHead state target
  let flatGrads := (List.range flatParams.length).map (fun idx => finiteDiffList lossFromFlat flatParams idx)
  mlpBlockGradsOfList d_model d_ff flatGrads

/-- One SGD step on the MLP block parameters. -/
def sgdStepMLPBlock (params : MLPBlockParams d_model d_ff)
    (grads : MLPBlockGrads d_model d_ff) (lr : Float) : MLPBlockParams d_model d_ff :=
  { ln :=
      { gamma := Vec.sub params.ln.gamma (Vec.scale grads.ln.gamma lr)
      , beta := Vec.sub params.ln.beta (Vec.scale grads.ln.beta lr)
      }
  , fc1 := Mat.sub params.fc1 (Mat.scale grads.fc1 lr)
  , fc2 := Mat.sub params.fc2 (Mat.scale grads.fc2 lr)
  }

/-- Repeated SGD steps on an MLP block over one frozen state example. -/
def trainMLPBlockSteps (params : MLPBlockParams d_model d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (state : Vec d_model)
    (target : Fin vocab)
    (lr : Float) : Nat → MLPBlockParams d_model d_ff
  | 0 => params
  | steps + 1 =>
      let grads := reverseGradientMLPBlock params outputHead state target
      trainMLPBlockSteps (sgdStepMLPBlock params grads lr) outputHead state target lr steps

/-- Single-head attention block parameters: pre-attention layer norm plus projections. -/
structure AttentionBlockParams (d_model d_head : Nat) where
  ln : LayerNormParams d_model
  wq : Mat d_head d_model
  wk : Mat d_head d_model
  wv : Mat d_head d_model
  wo : Mat d_model d_head

/-- Typed gradients for the single-head attention block parameters. -/
structure AttentionBlockGrads (d_model d_head : Nat) where
  ln : LayerNormParams d_model
  wq : Mat d_head d_model
  wk : Mat d_head d_model
  wv : Mat d_head d_model
  wo : Mat d_model d_head

/-- Single-head attention block forward pass with residual connection. -/
def attentionBlockForward (params : AttentionBlockParams d_model d_head)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (causal : Bool := true) : Vec d_model :=
  let x := sequence.get queryIdx
  let normedSeq := TokenSeq.map sequence (layerNorm params.ln)
  let q := Mat.mulVec params.wq (normedSeq.get queryIdx)
  let keys := TokenSeq.map normedSeq (Mat.mulVec params.wk)
  let values := TokenSeq.map normedSeq (Mat.mulVec params.wv)
  let scaleRaw := Float.sqrt (Float.ofNat d_head)
  let scale := if scaleRaw == 0.0 then 1.0 else scaleRaw
  let scores : Vec seqLen :=
    Vec.ofFn seqLen fun i =>
      let raw := Vec.dot q (keys.get i) / scale
      if causal && queryIdx.val < i.val then causalMaskPenalty else raw
  let weights := softmax scores
  let context := (List.finRange seqLen).foldl
    (fun acc i => Vec.add acc (Vec.scale (values.get i) (weights.get i)))
    (Vec.zeros d_head)
  let attnOut := Mat.mulVec params.wo context
  Vec.add x attnOut

/-- Cross-entropy loss for the attention block feeding a fixed output head. -/
def attentionBlockLoss (params : AttentionBlockParams d_model d_head)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab) : Float :=
  outputHeadLoss outputHead (attentionBlockForward params sequence queryIdx) target

/-- Flatten attention block parameters into `gamma ++ beta ++ wq ++ wk ++ wv ++ wo`. -/
def flattenAttentionBlockParams (params : AttentionBlockParams d_model d_head) : List Float :=
  params.ln.gamma.toList ++ params.ln.beta.toList ++
    params.wq.toList ++ params.wk.toList ++ params.wv.toList ++ params.wo.toList

/-- Rebuild attention block parameters from `gamma ++ beta ++ wq ++ wk ++ wv ++ wo`. -/
def attentionBlockParamsOfList (d_model d_head : Nat) (xs : List Float) :
    AttentionBlockParams d_model d_head :=
  let gamma := vecOfList d_model xs
  let beta := vecOfList d_model (xs.drop d_model)
  let wq := matOfList d_head d_model (xs.drop (2 * d_model))
  let wk := matOfList d_head d_model (xs.drop (2 * d_model + d_head * d_model))
  let wv := matOfList d_head d_model (xs.drop (2 * d_model + 2 * d_head * d_model))
  let wo := matOfList d_model d_head (xs.drop (2 * d_model + 3 * d_head * d_model))
  { ln := { gamma := gamma, beta := beta }
  , wq := wq
  , wk := wk
  , wv := wv
  , wo := wo
  }

/-- Rebuild typed attention block gradients from a flat gradient list. -/
def attentionBlockGradsOfList (d_model d_head : Nat) (xs : List Float) :
    AttentionBlockGrads d_model d_head :=
  let gamma := vecOfList d_model xs
  let beta := vecOfList d_model (xs.drop d_model)
  let wq := matOfList d_head d_model (xs.drop (2 * d_model))
  let wk := matOfList d_head d_model (xs.drop (2 * d_model + d_head * d_model))
  let wv := matOfList d_head d_model (xs.drop (2 * d_model + 2 * d_head * d_model))
  let wo := matOfList d_model d_head (xs.drop (2 * d_model + 3 * d_head * d_model))
  { ln := { gamma := gamma, beta := beta }
  , wq := wq
  , wk := wk
  , wv := wv
  , wo := wo
  }

/-- Flatten typed attention block gradients for comparison/debug printing. -/
def flattenAttentionBlockGrads (grads : AttentionBlockGrads d_model d_head) : List Float :=
  grads.ln.gamma.toList ++ grads.ln.beta.toList ++
    grads.wq.toList ++ grads.wk.toList ++ grads.wv.toList ++ grads.wo.toList

/-- Trace single-head attention block output with trainable parameters on a frozen sequence. -/
def traceAttentionBlockForwardFlat (d_model d_head seqLen : Nat)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (params : List TracedFloat)
    (causal : Bool := true) : TraceM (TracedVec d_model) := do
  let gamma := traceVecFromFlat d_model 0 params
  let beta := traceVecFromFlat d_model d_model params
  let wq := traceMatFromFlatOffset d_head d_model (2 * d_model) params
  let wk := traceMatFromFlatOffset d_head d_model (2 * d_model + d_head * d_model) params
  let wv := traceMatFromFlatOffset d_head d_model (2 * d_model + 2 * d_head * d_model) params
  let wo := traceMatFromFlatOffset d_model d_head (2 * d_model + 3 * d_head * d_model) params

  let normedSeq ← (List.finRange seqLen).mapM fun i => do
    let tok ← traceConstVec (sequence.get i)
    traceLayerNorm gamma beta tok

  let queryNorm :=
    match getListTracedVec? normedSeq queryIdx.val with
    | some x => x
    | none => panic! "traceAttentionBlockForwardFlat: impossible query index"
  let q ← traceMatVec wq queryNorm

  let keys ← (List.finRange seqLen).mapM fun i =>
    match getListTracedVec? normedSeq i.val with
    | some tok => traceMatVec wk tok
    | none => panic! "traceAttentionBlockForwardFlat: impossible key index"
  let values ← (List.finRange seqLen).mapM fun i =>
    match getListTracedVec? normedSeq i.val with
    | some tok => traceMatVec wv tok
    | none => panic! "traceAttentionBlockForwardFlat: impossible value index"

  let scaleRaw := Float.sqrt (Float.ofNat d_head)
  let scale := if scaleRaw == 0.0 then 1.0 else scaleRaw
  let scaleConst ← traceConst scale
  let scoreEntries ← (List.finRange seqLen).mapM fun i => do
    let key :=
      match getListTracedVec? keys i.val with
      | some x => x
      | none => panic! "traceAttentionBlockForwardFlat: impossible score index"
    let rawDot ← traceDotVec q key
    let raw ← traceDiv rawDot scaleConst
    if causal && queryIdx.val < i.val then traceConst causalMaskPenalty else pure raw
  let scores := TracedVec.ofFn seqLen (fun i =>
    match getListTraced? scoreEntries i.val with
    | some x => x
    | none => panic! "traceAttentionBlockForwardFlat: impossible score retrieval")
  let weights ← traceSoftmaxVec scores

  let zeroCtx ← traceConstVec (Vec.zeros d_head)
  let context ← (List.finRange seqLen).foldlM (fun acc i => do
    let value :=
      match getListTracedVec? values i.val with
      | some x => x
      | none => panic! "traceAttentionBlockForwardFlat: impossible context value"
    let scaled ← traceScaleVec value (weights.get i)
    traceAddVec acc scaled
  ) zeroCtx
  let attnOut ← traceMatVec wo context
  let x ← traceConstVec (sequence.get queryIdx)
  traceAddVec x attnOut

/-- Trace attention block loss against a fixed output head. -/
def traceAttentionBlockLossFlat {d_model d_head seqLen vocab : Nat}
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (outputHead : OutputHeadParams d_model vocab)
    (target : Fin vocab)
    (params : List TracedFloat) : TraceM TracedFloat := do
  let state ← traceAttentionBlockForwardFlat d_model d_head seqLen sequence queryIdx params
  let logits ← traceOutputHeadForwardConst outputHead state
  traceCrossEntropyVec logits target

/-- Reverse-mode gradient of the attention block loss with typed parameter output. -/
def reverseGradientAttentionBlock (params : AttentionBlockParams d_model d_head)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab) : AttentionBlockGrads d_model d_head :=
  let flatParams := flattenAttentionBlockParams params
  let flatGrads := reverseGradientList
    (fun xs => traceAttentionBlockLossFlat (d_head := d_head) sequence queryIdx outputHead target xs)
    flatParams
  attentionBlockGradsOfList d_model d_head flatGrads

/-- Finite-difference gradient of the attention block loss with typed parameter output. -/
def finiteDiffAttentionBlock (params : AttentionBlockParams d_model d_head)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab) : AttentionBlockGrads d_model d_head :=
  let flatParams := flattenAttentionBlockParams params
  let lossFromFlat := fun xs =>
    attentionBlockLoss (attentionBlockParamsOfList d_model d_head xs) outputHead sequence queryIdx target
  let flatGrads := (List.range flatParams.length).map (fun idx => finiteDiffList lossFromFlat flatParams idx)
  attentionBlockGradsOfList d_model d_head flatGrads

/-- One SGD step on the attention block parameters. -/
def sgdStepAttentionBlock (params : AttentionBlockParams d_model d_head)
    (grads : AttentionBlockGrads d_model d_head) (lr : Float) : AttentionBlockParams d_model d_head :=
  { ln :=
      { gamma := Vec.sub params.ln.gamma (Vec.scale grads.ln.gamma lr)
      , beta := Vec.sub params.ln.beta (Vec.scale grads.ln.beta lr)
      }
  , wq := Mat.sub params.wq (Mat.scale grads.wq lr)
  , wk := Mat.sub params.wk (Mat.scale grads.wk lr)
  , wv := Mat.sub params.wv (Mat.scale grads.wv lr)
  , wo := Mat.sub params.wo (Mat.scale grads.wo lr)
  }

/-- Repeated SGD steps on an attention block over one frozen sequence example. -/
def trainAttentionBlockSteps (params : AttentionBlockParams d_model d_head)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab)
    (lr : Float) : Nat → AttentionBlockParams d_model d_head
  | 0 => params
  | steps + 1 =>
      let grads := reverseGradientAttentionBlock params outputHead sequence queryIdx target
      trainAttentionBlockSteps (sgdStepAttentionBlock params grads lr) outputHead sequence queryIdx target lr steps

/-- Combined trainable transformer block: attention block followed by MLP block. -/
structure TransformerBlockTrainParams (d_model d_head d_ff : Nat) where
  attn : AttentionBlockParams d_model d_head
  mlp  : MLPBlockParams d_model d_ff

/-- Typed gradients for the combined transformer block. -/
structure TransformerBlockTrainGrads (d_model d_head d_ff : Nat) where
  attn : AttentionBlockGrads d_model d_head
  mlp  : MLPBlockGrads d_model d_ff

/-- Full transformer block forward pass with fixed sequence input. -/
def transformerBlockTrainForward (params : TransformerBlockTrainParams d_model d_head d_ff)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (causal : Bool := true) : Vec d_model :=
  let afterAttn := attentionBlockForward params.attn sequence queryIdx causal
  mlpBlockForward params.mlp afterAttn

/-- Cross-entropy loss for the trainable transformer block feeding a fixed output head. -/
def transformerBlockTrainLoss (params : TransformerBlockTrainParams d_model d_head d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab) : Float :=
  outputHeadLoss outputHead (transformerBlockTrainForward params sequence queryIdx) target

/-- Flatten combined transformer block parameters. -/
def flattenTransformerBlockTrainParams (params : TransformerBlockTrainParams d_model d_head d_ff) : List Float :=
  flattenAttentionBlockParams params.attn ++ flattenMLPBlockParams params.mlp

/-- Rebuild combined transformer block parameters from a flat list. -/
def transformerBlockTrainParamsOfList (d_model d_head d_ff : Nat) (xs : List Float) :
    TransformerBlockTrainParams d_model d_head d_ff :=
  let attnCount := 2 * d_model + 4 * d_head * d_model
  { attn := attentionBlockParamsOfList d_model d_head (xs.take attnCount)
  , mlp := mlpBlockParamsOfList d_model d_ff (xs.drop attnCount)
  }

/-- Rebuild combined transformer block gradients from a flat list. -/
def transformerBlockTrainGradsOfList (d_model d_head d_ff : Nat) (xs : List Float) :
    TransformerBlockTrainGrads d_model d_head d_ff :=
  let attnCount := 2 * d_model + 4 * d_head * d_model
  { attn := attentionBlockGradsOfList d_model d_head (xs.take attnCount)
  , mlp := mlpBlockGradsOfList d_model d_ff (xs.drop attnCount)
  }

/-- Flatten combined transformer block gradients for comparison/debug printing. -/
def flattenTransformerBlockTrainGrads (grads : TransformerBlockTrainGrads d_model d_head d_ff) : List Float :=
  flattenAttentionBlockGrads grads.attn ++ flattenMLPBlockGrads grads.mlp

/-- Trace the full transformer block output on a frozen sequence. -/
def traceTransformerBlockTrainForwardFlat (d_model d_head d_ff seqLen : Nat)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (params : List TracedFloat) : TraceM (TracedVec d_model) := do
  let attnCount := 2 * d_model + 4 * d_head * d_model
  let afterAttn ←
    traceAttentionBlockForwardFlat d_model d_head seqLen sequence queryIdx (params.take attnCount)
  traceMLPBlockForwardTraced d_model d_ff afterAttn (params.drop attnCount)

/-- Trace the full transformer block loss against a fixed output head. -/
def traceTransformerBlockTrainLossFlat {d_model d_head d_ff seqLen vocab : Nat}
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (outputHead : OutputHeadParams d_model vocab)
    (target : Fin vocab)
    (params : List TracedFloat) : TraceM TracedFloat := do
  let state ← traceTransformerBlockTrainForwardFlat d_model d_head d_ff seqLen sequence queryIdx params
  let logits ← traceOutputHeadForwardConst outputHead state
  traceCrossEntropyVec logits target

/-- Reverse-mode gradient of the full transformer block loss with typed parameter output. -/
def reverseGradientTransformerBlockTrain (params : TransformerBlockTrainParams d_model d_head d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab) : TransformerBlockTrainGrads d_model d_head d_ff :=
  let flatParams := flattenTransformerBlockTrainParams params
  let flatGrads := reverseGradientList
    (fun xs => traceTransformerBlockTrainLossFlat (d_head := d_head) (d_ff := d_ff) sequence queryIdx outputHead target xs)
    flatParams
  transformerBlockTrainGradsOfList d_model d_head d_ff flatGrads

/-- Finite-difference gradient of the full transformer block loss with typed parameter output. -/
def finiteDiffTransformerBlockTrain (params : TransformerBlockTrainParams d_model d_head d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab) : TransformerBlockTrainGrads d_model d_head d_ff :=
  let flatParams := flattenTransformerBlockTrainParams params
  let lossFromFlat := fun xs =>
    transformerBlockTrainLoss (transformerBlockTrainParamsOfList d_model d_head d_ff xs) outputHead sequence queryIdx target
  let flatGrads := (List.range flatParams.length).map (fun idx => finiteDiffList lossFromFlat flatParams idx)
  transformerBlockTrainGradsOfList d_model d_head d_ff flatGrads

/-- One SGD step on the full transformer block parameters. -/
def sgdStepTransformerBlockTrain (params : TransformerBlockTrainParams d_model d_head d_ff)
    (grads : TransformerBlockTrainGrads d_model d_head d_ff) (lr : Float) :
    TransformerBlockTrainParams d_model d_head d_ff :=
  { attn := sgdStepAttentionBlock params.attn grads.attn lr
  , mlp := sgdStepMLPBlock params.mlp grads.mlp lr
  }

/-- Repeated SGD steps on a frozen-sequence transformer block example. -/
def trainTransformerBlockTrainSteps (params : TransformerBlockTrainParams d_model d_head d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab)
    (lr : Float) : Nat → TransformerBlockTrainParams d_model d_head d_ff
  | 0 => params
  | steps + 1 =>
      let grads := reverseGradientTransformerBlockTrain params outputHead sequence queryIdx target
      trainTransformerBlockTrainSteps
        (sgdStepTransformerBlockTrain params grads lr)
        outputHead sequence queryIdx target lr steps

/-- Flat parameter count for one attention head (`wq ++ wk ++ wv`). -/
def attentionHeadParamCount (d_model d_head : Nat) : Nat :=
  3 * d_head * d_model

/-- Flatten a single attention head. -/
def flattenAttentionHeadParams (head : AttentionHead d_model d_head) : List Float :=
  head.wq.toList ++ head.wk.toList ++ head.wv.toList

/-- Rebuild a single attention head from a flat list starting at `offset`. -/
def attentionHeadOfFlatOffset (d_model d_head offset : Nat) (xs : List Float) :
    AttentionHead d_model d_head :=
  let base := d_head * d_model
  { wq := matOfList d_head d_model (xs.drop offset)
  , wk := matOfList d_head d_model (xs.drop (offset + base))
  , wv := matOfList d_head d_model (xs.drop (offset + 2 * base))
  }

/-- Concatenate typed traced head outputs into one traced model-width vector. -/
def traceConcatHeadOutputs (outputs : Fin n_heads → TracedVec d_head) (h_headDim : 0 < d_head) :
    TracedVec (n_heads * d_head) :=
  TracedVec.ofFn (n_heads * d_head) fun i =>
    let headIdx : Fin n_heads :=
      ⟨i.val / d_head, by
        rw [Nat.div_lt_iff_lt_mul h_headDim]
        exact i.isLt⟩
    let offset : Fin d_head :=
      ⟨i.val % d_head, Nat.mod_lt _ h_headDim⟩
    (outputs headIdx).get offset

/-- Trace one attention head over an already normalized sequence. -/
def traceAttentionHeadOnNormedSeq (d_model d_head seqLen : Nat)
    (normedSeq : List (TracedVec d_model))
    (queryIdx : Fin seqLen)
    (wq wk wv : TracedMat d_head d_model)
    (causal : Bool := true) : TraceM (TracedVec d_head) := do
  let queryNorm :=
    match getListTracedVec? normedSeq queryIdx.val with
    | some x => x
    | none => panic! "traceAttentionHeadOnNormedSeq: impossible query index"
  let q ← traceMatVec wq queryNorm

  let keys ← (List.finRange seqLen).mapM fun i =>
    match getListTracedVec? normedSeq i.val with
    | some tok => traceMatVec wk tok
    | none => panic! "traceAttentionHeadOnNormedSeq: impossible key index"
  let values ← (List.finRange seqLen).mapM fun i =>
    match getListTracedVec? normedSeq i.val with
    | some tok => traceMatVec wv tok
    | none => panic! "traceAttentionHeadOnNormedSeq: impossible value index"

  let scaleRaw := Float.sqrt (Float.ofNat d_head)
  let scale := if scaleRaw == 0.0 then 1.0 else scaleRaw
  let scaleConst ← traceConst scale
  let scoreEntries ← (List.finRange seqLen).mapM fun i => do
    let key :=
      match getListTracedVec? keys i.val with
      | some x => x
      | none => panic! "traceAttentionHeadOnNormedSeq: impossible score index"
    let rawDot ← traceDotVec q key
    let raw ← traceDiv rawDot scaleConst
    if causal && queryIdx.val < i.val then traceConst causalMaskPenalty else pure raw
  let scores := TracedVec.ofFn seqLen (fun i =>
    match getListTraced? scoreEntries i.val with
    | some x => x
    | none => panic! "traceAttentionHeadOnNormedSeq: impossible score retrieval")
  let weights ← traceSoftmaxVec scores

  let zeroCtx ← traceConstVec (Vec.zeros d_head)
  (List.finRange seqLen).foldlM (fun acc i => do
    let value :=
      match getListTracedVec? values i.val with
      | some x => x
      | none => panic! "traceAttentionHeadOnNormedSeq: impossible context value"
    let scaled ← traceScaleVec value (weights.get i)
    traceAddVec acc scaled
  ) zeroCtx

/-- Full trainable multi-head transformer layer mirroring the actual model layer shape. -/
structure FullTransformerLayerTrainParams (d_model n_heads d_head d_ff : Nat) where
  ln1 : LayerNormParams d_model
  heads : Fin n_heads → AttentionHead d_model d_head
  wo : Mat d_model (n_heads * d_head)
  h_split : d_model = n_heads * d_head
  h_headDim : 0 < d_head
  ln2 : LayerNormParams d_model
  fc1 : Mat d_ff d_model
  fc2 : Mat d_model d_ff

/-- Typed gradients for the full trainable multi-head transformer layer. -/
structure FullTransformerLayerTrainGrads (d_model n_heads d_head d_ff : Nat) where
  ln1 : LayerNormParams d_model
  heads : Fin n_heads → AttentionHead d_model d_head
  wo : Mat d_model (n_heads * d_head)
  ln2 : LayerNormParams d_model
  fc1 : Mat d_ff d_model
  fc2 : Mat d_model d_ff

/-- Parameter count for the attention sublayer of a full multi-head transformer layer. -/
def fullTransformerLayerAttnParamCount (d_model n_heads d_head : Nat) : Nat :=
  2 * d_model + n_heads * attentionHeadParamCount d_model d_head + d_model * (n_heads * d_head)

/-- Forward pass of a full trainable multi-head transformer layer. -/
def fullTransformerLayerTrainForward (params : FullTransformerLayerTrainParams d_model n_heads d_head d_ff)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (causal : Bool := true) : Vec d_model :=
  let x := sequence.get queryIdx
  let normedSeq := TokenSeq.map sequence (layerNorm params.ln1)
  let attnOut :=
    multiHeadAttention
      { heads := params.heads
      , wo := params.wo
      , h_split := params.h_split
      , h_headDim := params.h_headDim
      }
      normedSeq
      queryIdx
      causal
  let x' := Vec.add x attnOut
  let normed2 := layerNorm params.ln2 x'
  let hidden := gelu (Mat.mulVec params.fc1 normed2)
  let mlpOut := Mat.mulVec params.fc2 hidden
  Vec.add x' mlpOut

/-- Cross-entropy loss for the full trainable multi-head transformer layer feeding a fixed output head. -/
def fullTransformerLayerTrainLoss (params : FullTransformerLayerTrainParams d_model n_heads d_head d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab) : Float :=
  outputHeadLoss outputHead (fullTransformerLayerTrainForward params sequence queryIdx) target

/-- Flatten full trainable multi-head transformer layer parameters. -/
def flattenFullTransformerLayerTrainParams (params : FullTransformerLayerTrainParams d_model n_heads d_head d_ff) :
    List Float :=
  let headParams :=
    ((List.finRange n_heads).map fun i => flattenAttentionHeadParams (params.heads i)).foldr List.append []
  params.ln1.gamma.toList ++
    params.ln1.beta.toList ++
    headParams ++
    params.wo.toList ++
    params.ln2.gamma.toList ++
    params.ln2.beta.toList ++
    params.fc1.toList ++
    params.fc2.toList

/-- Rebuild full trainable multi-head transformer layer parameters from a flat list. -/
def fullTransformerLayerTrainParamsOfList (d_model n_heads d_head d_ff : Nat)
    (h_split : d_model = n_heads * d_head)
    (h_headDim : 0 < d_head)
    (xs : List Float) : FullTransformerLayerTrainParams d_model n_heads d_head d_ff :=
  let ln1Gamma := vecOfList d_model xs
  let ln1Beta := vecOfList d_model (xs.drop d_model)
  let headStart := 2 * d_model
  let headCount := attentionHeadParamCount d_model d_head
  let woStart := headStart + n_heads * headCount
  let ln2Start := woStart + d_model * (n_heads * d_head)
  let fc1Start := ln2Start + 2 * d_model
  let fc2Start := fc1Start + d_ff * d_model
  { ln1 := { gamma := ln1Gamma, beta := ln1Beta }
  , heads := fun i => attentionHeadOfFlatOffset d_model d_head (headStart + i.val * headCount) xs
  , wo := matOfList d_model (n_heads * d_head) (xs.drop woStart)
  , h_split := h_split
  , h_headDim := h_headDim
  , ln2 :=
      { gamma := vecOfList d_model (xs.drop ln2Start)
      , beta := vecOfList d_model (xs.drop (ln2Start + d_model))
      }
  , fc1 := matOfList d_ff d_model (xs.drop fc1Start)
  , fc2 := matOfList d_model d_ff (xs.drop fc2Start)
  }

/-- Rebuild full trainable multi-head transformer layer gradients from a flat list. -/
def fullTransformerLayerTrainGradsOfList (d_model n_heads d_head d_ff : Nat)
    (xs : List Float) : FullTransformerLayerTrainGrads d_model n_heads d_head d_ff :=
  let ln1Gamma := vecOfList d_model xs
  let ln1Beta := vecOfList d_model (xs.drop d_model)
  let headStart := 2 * d_model
  let headCount := attentionHeadParamCount d_model d_head
  let woStart := headStart + n_heads * headCount
  let ln2Start := woStart + d_model * (n_heads * d_head)
  let fc1Start := ln2Start + 2 * d_model
  let fc2Start := fc1Start + d_ff * d_model
  { ln1 := { gamma := ln1Gamma, beta := ln1Beta }
  , heads := fun i => attentionHeadOfFlatOffset d_model d_head (headStart + i.val * headCount) xs
  , wo := matOfList d_model (n_heads * d_head) (xs.drop woStart)
  , ln2 :=
      { gamma := vecOfList d_model (xs.drop ln2Start)
      , beta := vecOfList d_model (xs.drop (ln2Start + d_model))
      }
  , fc1 := matOfList d_ff d_model (xs.drop fc1Start)
  , fc2 := matOfList d_model d_ff (xs.drop fc2Start)
  }

/-- Flatten typed gradients for the full trainable multi-head transformer layer. -/
def flattenFullTransformerLayerTrainGrads (grads : FullTransformerLayerTrainGrads d_model n_heads d_head d_ff) :
    List Float :=
  let headGrads :=
    ((List.finRange n_heads).map fun i => flattenAttentionHeadParams (grads.heads i)).foldr List.append []
  grads.ln1.gamma.toList ++
    grads.ln1.beta.toList ++
    headGrads ++
    grads.wo.toList ++
    grads.ln2.gamma.toList ++
    grads.ln2.beta.toList ++
    grads.fc1.toList ++
    grads.fc2.toList

/-- Trace the full trainable multi-head transformer layer on a frozen input sequence. -/
def traceFullTransformerLayerTrainForwardFlat (d_model n_heads d_head d_ff seqLen : Nat)
    (h_headDim : 0 < d_head)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (params : List TracedFloat)
    (causal : Bool := true) : TraceM (TracedVec d_model) := do
  let gamma1 := traceVecFromFlat d_model 0 params
  let beta1 := traceVecFromFlat d_model d_model params
  let headStart := 2 * d_model
  let headCount := attentionHeadParamCount d_model d_head
  let woStart := headStart + n_heads * headCount
  let attnCount := fullTransformerLayerAttnParamCount d_model n_heads d_head

  let normedSeq ← (List.finRange seqLen).mapM fun i => do
    let tok ← traceConstVec (sequence.get i)
    traceLayerNorm gamma1 beta1 tok

  let headOutputs ← (List.finRange n_heads).mapM fun h => do
    let offset := headStart + h.val * headCount
    let wq := traceMatFromFlatOffset d_head d_model offset params
    let wk := traceMatFromFlatOffset d_head d_model (offset + d_head * d_model) params
    let wv := traceMatFromFlatOffset d_head d_model (offset + 2 * d_head * d_model) params
    traceAttentionHeadOnNormedSeq d_model d_head seqLen normedSeq queryIdx wq wk wv causal

  let concatenated :=
    traceConcatHeadOutputs
      (fun h =>
        match getListTracedVec? headOutputs h.val with
        | some x => x
        | none => panic! "traceFullTransformerLayerTrainForwardFlat: impossible head output")
      h_headDim
  let wo := traceMatFromFlatOffset d_model (n_heads * d_head) woStart params
  let attnOut ← traceMatVec wo concatenated
  let x ← traceConstVec (sequence.get queryIdx)
  let afterAttn ← traceAddVec x attnOut
  traceMLPBlockForwardTraced d_model d_ff afterAttn (params.drop attnCount)

/-- Trace the full trainable multi-head transformer layer loss against a fixed output head. -/
def traceFullTransformerLayerTrainLossFlat {d_model n_heads d_head d_ff seqLen vocab : Nat}
    (h_headDim : 0 < d_head)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (outputHead : OutputHeadParams d_model vocab)
    (target : Fin vocab)
    (params : List TracedFloat) : TraceM TracedFloat := do
  let state ← traceFullTransformerLayerTrainForwardFlat d_model n_heads d_head d_ff seqLen h_headDim sequence queryIdx params
  let logits ← traceOutputHeadForwardConst outputHead state
  traceCrossEntropyVec logits target

/-- Reverse-mode gradient of the full trainable multi-head transformer layer loss. -/
def reverseGradientFullTransformerLayerTrain (params : FullTransformerLayerTrainParams d_model n_heads d_head d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab) : FullTransformerLayerTrainGrads d_model n_heads d_head d_ff :=
  let flatParams := flattenFullTransformerLayerTrainParams params
  let flatGrads := reverseGradientList
    (fun xs =>
      traceFullTransformerLayerTrainLossFlat
        (d_model := d_model)
        (n_heads := n_heads)
        (d_head := d_head)
        (d_ff := d_ff)
        (seqLen := seqLen)
        (vocab := vocab)
        params.h_headDim
        sequence
        queryIdx
        outputHead
        target
        xs)
    flatParams
  fullTransformerLayerTrainGradsOfList d_model n_heads d_head d_ff flatGrads

/-- Finite-difference gradient of the full trainable multi-head transformer layer loss. -/
def finiteDiffFullTransformerLayerTrain (params : FullTransformerLayerTrainParams d_model n_heads d_head d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab) : FullTransformerLayerTrainGrads d_model n_heads d_head d_ff :=
  let flatParams := flattenFullTransformerLayerTrainParams params
  let lossFromFlat := fun xs =>
    fullTransformerLayerTrainLoss
      (fullTransformerLayerTrainParamsOfList d_model n_heads d_head d_ff params.h_split params.h_headDim xs)
      outputHead
      sequence
      queryIdx
      target
  let flatGrads := (List.range flatParams.length).map (fun idx => finiteDiffList lossFromFlat flatParams idx)
  fullTransformerLayerTrainGradsOfList d_model n_heads d_head d_ff flatGrads

/-- One SGD step on a full trainable multi-head transformer layer. -/
def sgdStepFullTransformerLayerTrain (params : FullTransformerLayerTrainParams d_model n_heads d_head d_ff)
    (grads : FullTransformerLayerTrainGrads d_model n_heads d_head d_ff) (lr : Float) :
    FullTransformerLayerTrainParams d_model n_heads d_head d_ff :=
  { ln1 :=
      { gamma := Vec.sub params.ln1.gamma (Vec.scale grads.ln1.gamma lr)
      , beta := Vec.sub params.ln1.beta (Vec.scale grads.ln1.beta lr)
      }
  , heads := fun i =>
      let head := params.heads i
      let grad := grads.heads i
      { wq := Mat.sub head.wq (Mat.scale grad.wq lr)
      , wk := Mat.sub head.wk (Mat.scale grad.wk lr)
      , wv := Mat.sub head.wv (Mat.scale grad.wv lr)
      }
  , wo := Mat.sub params.wo (Mat.scale grads.wo lr)
  , h_split := params.h_split
  , h_headDim := params.h_headDim
  , ln2 :=
      { gamma := Vec.sub params.ln2.gamma (Vec.scale grads.ln2.gamma lr)
      , beta := Vec.sub params.ln2.beta (Vec.scale grads.ln2.beta lr)
      }
  , fc1 := Mat.sub params.fc1 (Mat.scale grads.fc1 lr)
  , fc2 := Mat.sub params.fc2 (Mat.scale grads.fc2 lr)
  }

/-- Repeated SGD steps on a full trainable multi-head transformer layer example. -/
def trainFullTransformerLayerTrainSteps (params : FullTransformerLayerTrainParams d_model n_heads d_head d_ff)
    (outputHead : OutputHeadParams d_model vocab)
    (sequence : TokenSeq seqLen d_model)
    (queryIdx : Fin seqLen)
    (target : Fin vocab)
    (lr : Float) : Nat → FullTransformerLayerTrainParams d_model n_heads d_head d_ff
  | 0 => params
  | steps + 1 =>
      let grads := reverseGradientFullTransformerLayerTrain params outputHead sequence queryIdx target
      trainFullTransformerLayerTrainSteps
        (sgdStepFullTransformerLayerTrain params grads lr)
        outputHead sequence queryIdx target lr steps

/-- Total parameter count for one full trainable transformer layer. -/
def fullTransformerLayerParamCount (d_model n_heads d_head d_ff : Nat) : Nat :=
  fullTransformerLayerAttnParamCount d_model n_heads d_head + 2 * d_model + d_ff * d_model + d_model * d_ff

/-- Flatten an actual transformer layer using the full trainable layer layout. -/
def flattenTransformerLayerParams (layer : TransformerLayer d_model n_heads d_head d_ff) : List Float :=
  let headParams :=
    ((List.finRange n_heads).map fun i =>
      let head := layer.attn.heads i
      flattenAttentionHeadParams head).foldr List.append []
  layer.ln1.gamma.toList ++
    layer.ln1.beta.toList ++
    headParams ++
    layer.attn.wo.toList ++
    layer.ln2.gamma.toList ++
    layer.ln2.beta.toList ++
    layer.mlp_fc1.toList ++
    layer.mlp_fc2.toList

/-- Rebuild an actual transformer layer from flat parameters using a template layer for structural proofs. -/
def transformerLayerOfFlatLike (template : TransformerLayer d_model n_heads d_head d_ff) (xs : List Float) :
    TransformerLayer d_model n_heads d_head d_ff :=
  let params :=
    fullTransformerLayerTrainParamsOfList
      d_model
      n_heads
      d_head
      d_ff
      template.attn.h_split
      template.attn.h_headDim
      xs
  { ln1 := params.ln1
  , attn :=
      { heads := params.heads
      , wo := params.wo
      , h_split := params.h_split
      , h_headDim := params.h_headDim
      }
  , ln2 := params.ln2
  , mlp_fc1 := params.fc1
  , mlp_fc2 := params.fc2
  }

/-- Trace output-head logits from trainable flat parameters and an already traced hidden state. -/
def traceOutputHeadForwardTracedFlat (d_model vocab : Nat)
    (state : TracedVec d_model)
    (params : List TracedFloat) : TraceM (TracedVec vocab) := do
  let gamma := traceVecFromFlat d_model 0 params
  let beta := traceVecFromFlat d_model d_model params
  let unembed := traceMatFromFlatOffset vocab d_model (2 * d_model) params
  let normed ← traceLayerNorm gamma beta state
  traceMatVec unembed normed

/-- Trace the full trainable multi-head transformer layer on an already traced sequence. -/
def traceFullTransformerLayerTrainForwardTraced (d_model n_heads d_head d_ff seqLen : Nat)
    (h_headDim : 0 < d_head)
    (sequence : List (TracedVec d_model))
    (queryIdx : Fin seqLen)
    (params : List TracedFloat)
    (causal : Bool := true) : TraceM (TracedVec d_model) := do
  let gamma1 := traceVecFromFlat d_model 0 params
  let beta1 := traceVecFromFlat d_model d_model params
  let headStart := 2 * d_model
  let headCount := attentionHeadParamCount d_model d_head
  let woStart := headStart + n_heads * headCount
  let attnCount := fullTransformerLayerAttnParamCount d_model n_heads d_head

  let normedSeq ← (List.finRange seqLen).mapM fun i => do
    let tok :=
      match getListTracedVec? sequence i.val with
      | some x => x
      | none => panic! "traceFullTransformerLayerTrainForwardTraced: impossible token index"
    traceLayerNorm gamma1 beta1 tok

  let headOutputs ← (List.finRange n_heads).mapM fun h => do
    let offset := headStart + h.val * headCount
    let wq := traceMatFromFlatOffset d_head d_model offset params
    let wk := traceMatFromFlatOffset d_head d_model (offset + d_head * d_model) params
    let wv := traceMatFromFlatOffset d_head d_model (offset + 2 * d_head * d_model) params
    traceAttentionHeadOnNormedSeq d_model d_head seqLen normedSeq queryIdx wq wk wv causal

  let concatenated :=
    traceConcatHeadOutputs
      (fun h =>
        match getListTracedVec? headOutputs h.val with
        | some x => x
        | none => panic! "traceFullTransformerLayerTrainForwardTraced: impossible head output")
      h_headDim
  let wo := traceMatFromFlatOffset d_model (n_heads * d_head) woStart params
  let attnOut ← traceMatVec wo concatenated
  let x :=
    match getListTracedVec? sequence queryIdx.val with
    | some tok => tok
    | none => panic! "traceFullTransformerLayerTrainForwardTraced: impossible query token"
  let afterAttn ← traceAddVec x attnOut
  traceMLPBlockForwardTraced d_model d_ff afterAttn (params.drop attnCount)

/-- Apply one trainable full transformer layer to every position in a traced sequence. -/
def traceApplyFullTransformerLayerTraced (d_model n_heads d_head d_ff seqLen : Nat)
    (h_headDim : 0 < d_head)
    (sequence : List (TracedVec d_model))
    (params : List TracedFloat)
    (causal : Bool := true) : TraceM (List (TracedVec d_model)) :=
  (List.finRange seqLen).mapM fun i =>
    traceFullTransformerLayerTrainForwardTraced d_model n_heads d_head d_ff seqLen h_headDim sequence i params causal

/-- Encode a string as UTF-8 bytes, one token per byte. -/
def encodeUTF8Bytes (text : String) : List (Fin 256) :=
  let bytes := text.toUTF8
  (List.finRange bytes.size).map fun i =>
    let b := bytes.get i.val
    ⟨b.toNat, by simpa [UInt8.size] using b.toNat_lt⟩

/-- Decode a UTF-8 byte token stream back to a string, if valid. -/
def decodeUTF8Bytes? (tokens : List (Fin 256)) : Option String :=
  let bytes := tokens.foldl (fun acc tok => acc.push (UInt8.ofNat tok.val)) ByteArray.empty
  String.fromUTF8? bytes

/-- Greedy token generation using argmax over logits. -/
def generateArgmax (model : MicroGPT cfg) (h_vocab : 0 < cfg.vocab)
    (prompt : List (Fin cfg.vocab)) (steps : Nat) : List (Fin cfg.vocab) :=
  let rec go (current : List (Fin cfg.vocab)) : Nat → List (Fin cfg.vocab)
    | 0 => current
    | remaining + 1 =>
        match current with
        | [] => []
        | x :: xs =>
            let current := x :: xs
            have h_len : 0 < current.length := by simp [current]
            let predictIdx : Fin current.length :=
              ⟨current.length - 1, by simpa using Nat.sub_lt h_len (by decide : 0 < 1)⟩
            let next := Vec.argmax (model.forward current predictIdx) h_vocab
            go (current ++ [next]) remaining
  go prompt steps

/-- Probability pairs `(token, prob)` for a vocabulary vector. -/
def indexedProbs (probs : Vec n) : List (Fin n × Float) :=
  (List.finRange n).map (fun i => (i, probs.get i))

/-- Insert a token-probability pair into a list sorted by descending probability. -/
def insertProbDesc (entry : Fin n × Float) : List (Fin n × Float) → List (Fin n × Float)
  | [] => [entry]
  | head :: tail =>
      if entry.2 > head.2 then
        entry :: head :: tail
      else
        head :: insertProbDesc entry tail

/-- Sort token-probability pairs by descending probability. -/
def sortProbsDesc (entries : List (Fin n × Float)) : List (Fin n × Float) :=
  entries.foldl (fun acc entry => insertProbDesc entry acc) []

/-- Temperature-scaled probabilities from logits. -/
def temperatureProbs (logits : Vec n) (temperature : Float) : Vec n :=
  let safeTemp := if temperature <= 0.0 then 1.0 else temperature
  softmax (logits.map (fun x => x / safeTemp))

/-- Keep only the top-k probabilities and renormalize them. -/
def topKProbs (probs : Vec n) (k : Nat) : List (Fin n × Float) :=
  let ranked := sortProbsDesc (indexedProbs probs)
  let kept := ranked.take (if k == 0 then 1 else k)
  let total := kept.foldl (fun acc entry => acc + entry.2) 0.0
  let denom := if total == 0.0 then 1.0 else total
  kept.map (fun entry => (entry.1, entry.2 / denom))

/-- Sample one token from a non-empty probability list. -/
def sampleFromPairs (fallback : Fin n) (pairs : List (Fin n × Float)) : IO (Fin n) := do
  let drawNat ← IO.rand 0 1000000
  let draw := Float.ofNat drawNat / 1000000.0
  let rec pick (remaining : List (Fin n × Float)) (mass : Float) : Fin n :=
    match remaining with
    | [] => fallback
    | (tok, prob) :: tail =>
        let mass' := mass + prob
        if draw <= mass' then tok else pick tail mass'
  pure <| pick pairs 0.0

/-- Temperature sampling over the full vocabulary. -/
def sampleTemperature (logits : Vec n) (h_vocab : 0 < n) (temperature : Float) : IO (Fin n) := do
  let probs := temperatureProbs logits temperature
  sampleFromPairs (Vec.argmax probs h_vocab) (indexedProbs probs)

/-- Top-k sampling after temperature scaling. -/
def sampleTopK (logits : Vec n) (h_vocab : 0 < n) (temperature : Float) (k : Nat) : IO (Fin n) := do
  let probs := temperatureProbs logits temperature
  sampleFromPairs (Vec.argmax probs h_vocab) (topKProbs probs k)

/-- Autoregressive generation with temperature and optional top-k filtering. -/
def generateSampled (model : MicroGPT cfg) (h_vocab : 0 < cfg.vocab)
    (prompt : List (Fin cfg.vocab)) (steps : Nat)
    (temperature : Float := 1.0) (topK : Option Nat := none) : IO (List (Fin cfg.vocab)) := do
  let rec go (current : List (Fin cfg.vocab)) : Nat → IO (List (Fin cfg.vocab))
    | 0 => pure current
    | remaining + 1 =>
        match current with
        | [] => pure []
        | x :: xs => do
            let current := x :: xs
            have h_len : 0 < current.length := by simp [current]
            let predictIdx : Fin current.length :=
              ⟨current.length - 1, by simpa using Nat.sub_lt h_len (by decide : 0 < 1)⟩
            let logits := model.forward current predictIdx
            let next ←
              match topK with
              | some k => sampleTopK logits h_vocab temperature k
              | none => sampleTemperature logits h_vocab temperature
            go (current ++ [next]) remaining
  go prompt steps

-- ============================================================
-- SECTION 9: Verification Notes
-- ============================================================

/- Softmax should sum to approximately 1.0; proving this would require
   substantial floating-point reasoning beyond this executable example. -/

/- Residual dimension safety is encoded directly in the type of `Vec.add`:
   `Vec.add : Vec d → Vec d → Vec d`. -/

/- Matrix multiplication shape safety is encoded directly in the type of
   `Mat.mulVec : Mat m n → Vec n → Vec m`. -/

-- ============================================================
-- SECTION 10: Demo / Smoke Test
-- ============================================================

/-- Create a tiny model configuration for experimentation. -/
def mkTinyConfig : GPTConfig :=
  { d_model := 4, n_heads := 2, d_head := 2, d_ff := 8, n_layers := 1, vocab := 3 }

/-- Deterministic small scalar used to initialize larger typed fixtures without randomness. -/
def patternedWeight (seed row col : Nat) (scale : Float := 0.02) : Float :=
  let raw := (seed + 17 * (row + 1) + 31 * (col + 1)) % 19
  (Float.ofNat raw - 9.0) * scale

/-- Deterministic small vector initializer. -/
def patternedVec (n seed : Nat) (scale : Float := 0.02) : Vec n :=
  Vec.ofFn n fun i => patternedWeight seed i.val 0 scale

/-- Deterministic small matrix initializer. -/
def patternedMat (rows cols seed : Nat) (scale : Float := 0.02) : Mat rows cols :=
  Mat.ofFn rows cols fun i j => patternedWeight seed i.val j.val scale

/-- Quick test: vector operations. -/
def testVecOps : IO Unit := do
  let v1 : Vec 3 := Vec.ofFn 3 fun i =>
    match i.val with
    | 0 => 1.0
    | 1 => 2.0
    | _ => 3.0

  let v2 : Vec 3 := Vec.ofFn 3 fun i =>
    match i.val with
    | 0 => 4.0
    | 1 => 5.0
    | _ => 6.0

  let sum := Vec.add v1 v2
  IO.println s!"Vec add: {sum.data.toList}"

  let dot := Vec.dot v1 v2
  IO.println s!"Dot product: {dot}"

  let sm := softmax v1
  IO.println s!"Softmax: {sm.data.toList}"
  IO.println s!"Softmax sum: {Vec.sum sm}"

  let normed := layerNorm (LayerNormParams.ones 3) v1
  IO.println s!"LayerNorm: {normed.data.toList}"

/-- Print forward/reverse AD derivatives vs finite differences for one scalar function. -/
def reportDerivative (label : String)
    (fDual : DualFloat → DualFloat)
    (fTrace : TracedFloat → TraceM TracedFloat)
    (fFloat : Float → Float)
    (x : Float) : IO Unit := do
  let forward := forwardDerivative fDual x
  let reverse := reverseDerivative fTrace x
  let numeric := finiteDiff fFloat x
  let errForward := floatAbs (forward - numeric)
  let errReverse := floatAbs (reverse - numeric)
  IO.println s!"{label}: forward={forward}, reverse={reverse}, finiteDiff={numeric}, errF={errForward}, errR={errReverse}"

/-- Print reverse-mode two-variable gradients vs finite-difference partials. -/
def reportGradient2 (label : String)
    (fTrace : TracedFloat → TracedFloat → TraceM TracedFloat)
    (fFloat : Float → Float → Float)
    (x y : Float) : IO Unit := do
  let (dx, dy) := reverseGradient2 fTrace x y
  let ndx := finiteDiffX fFloat x y
  let ndy := finiteDiffY fFloat x y
  let errX := floatAbs (dx - ndx)
  let errY := floatAbs (dy - ndy)
  IO.println s!"{label}: dx={dx}, dy={dy}, finiteDx={ndx}, finiteDy={ndy}, errX={errX}, errY={errY}"

/-- Print reverse-mode flat-list gradients vs finite-difference partials. -/
def reportGradientList (label : String)
    (fTrace : List TracedFloat → TraceM TracedFloat)
    (fFloat : List Float → Float)
    (xs : List Float) : IO Unit := do
  let reverse := reverseGradientList fTrace xs
  let numeric := (List.range xs.length).map (fun idx => finiteDiffList fFloat xs idx)
  let errors :=
    List.zipWith (fun r n => floatAbs (r - n)) reverse numeric
  IO.println s!"{label}: reverse={reverse}, finite={numeric}, absErr={errors}"

/-- Quick test: forward-mode autodiff against finite differences. -/
def testAutodiff : IO Unit := do
  let quadraticDual : DualFloat → DualFloat := fun x =>
    x * x + DualFloat.const 3.0 * x + DualFloat.const 2.0
  let quadraticTrace : TracedFloat → TraceM TracedFloat := fun x => do
    let x2 ← traceMul x x
    let c3 ← traceConst 3.0
    let threeX ← traceMul c3 x
    let partialSum ← traceAdd x2 threeX
    let c2 ← traceConst 2.0
    traceAdd partialSum c2
  let quadraticFloat : Float → Float := fun x => x * x + 3.0 * x + 2.0
  reportDerivative "quadratic @ x=2" quadraticDual quadraticTrace quadraticFloat 2.0

  let expDual : DualFloat → DualFloat := DualFloat.exp
  let expTrace : TracedFloat → TraceM TracedFloat := traceExp
  let expFloat : Float → Float := Float.exp
  reportDerivative "exp @ x=1" expDual expTrace expFloat 1.0

  let tanhDual : DualFloat → DualFloat := DualFloat.tanh
  let tanhTrace : TracedFloat → TraceM TracedFloat := traceTanh
  let tanhFloat : Float → Float := Float.tanh
  reportDerivative "tanh @ x=0.5" tanhDual tanhTrace tanhFloat 0.5

  let geluDual : DualFloat → DualFloat := dualGeluScalar
  let geluTrace : TracedFloat → TraceM TracedFloat := traceGeluScalar
  let geluFloat : Float → Float := fun x =>
    let cubic := x * x * x
    let cdf := 0.5 * (1.0 + Float.tanh (0.7978845608 * (x + 0.044715 * cubic)))
    x * cdf
  reportDerivative "gelu @ x=0.7" geluDual geluTrace geluFloat 0.7

  let dotTrace : TracedFloat → TracedFloat → TraceM TracedFloat := fun x y => do
    let two ← traceConst 2.0
    let three ← traceConst 3.0
    traceDotList [x, two] [three, y]
  let dotFloat : Float → Float → Float := fun x y => x * 3.0 + 2.0 * y
  reportGradient2 "dot([x,2],[3,y]) @ (1,4)" dotTrace dotFloat 1.0 4.0

  let matvecTrace : List TracedFloat → TraceM TracedFloat := fun inputs =>
    match inputs with
    | [w11, w12, w21, w22, x1, x2] => do
        let outputs ← traceMatVecList [[w11, w12], [w21, w22]] [x1, x2]
        traceSumList outputs
    | _ => traceConst 0.0
  let matvecFloat : List Float → Float := fun inputs =>
    match inputs with
    | [w11, w12, w21, w22, x1, x2] =>
        (w11 * x1 + w12 * x2) + (w21 * x1 + w22 * x2)
    | _ => 0.0
  reportGradientList "sum(matvec(W,x)) @ [1,2,3,4,5,6]" matvecTrace matvecFloat [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

  let xentTrace : List TracedFloat → TraceM TracedFloat := fun logits =>
    traceCrossEntropyLogits logits 0
  let xentFloat : List Float → Float := fun logits =>
    crossEntropyLogitsList logits 0
  reportGradientList "crossEntropy(logits,target=0) @ [1,0,-1]" xentTrace xentFloat [1.0, 0.0, -1.0]

/-- Quick test: matrix-vector multiplication. -/
def testMatVec : IO Unit := do
  let m : Mat 2 3 := Mat.ofFn 2 3 fun i j =>
    match i.val, j.val with
    | 0, 0 => 1.0
    | 0, 1 => 2.0
    | 0, _ => 3.0
    | 1, 0 => 4.0
    | 1, 1 => 5.0
    | _, _ => 6.0

  let v : Vec 3 := Vec.ofFn 3 fun _ => 1.0
  let result := Mat.mulVec m v
  IO.println s!"MatVec: {result.data.toList}"
  IO.println "Expected approximately: [6.0, 15.0]"

/-- Quick test: deterministic single-head attention. -/
def testAttention : IO Unit := do
  let head : AttentionHead 2 2 :=
    { wq := Mat.identity 2
    , wk := Mat.identity 2
    , wv := Mat.identity 2
    }

  let sequence : TokenSeq 2 2 :=
    TokenSeq.ofFn 2 2 fun i =>
      match i.val with
      | 0 => Vec.ofFn 2 fun j => if j.val = 0 then 1.0 else 0.0
      | _ => Vec.ofFn 2 fun j => if j.val = 1 then 1.0 else 0.0

  let output := attention head sequence ⟨0, by decide⟩
  IO.println s!"Attention output: {output.data.toList}"
  IO.println "Expected approximately: [0.6698, 0.3302]"

  let causalOutput := attention head sequence ⟨0, by decide⟩ true
  IO.println s!"Causal attention output: {causalOutput.data.toList}"
  IO.println "Expected approximately: [1.0, 0.0]"

/-- Quick test: deterministic multi-head attention with two 1D heads. -/
def testMultiHeadAttention : IO Unit := do
  let head0 : AttentionHead 2 1 :=
    { wq := Mat.ofFn 1 2 fun _ j => if j.val = 0 then 1.0 else 0.0
    , wk := Mat.ofFn 1 2 fun _ j => if j.val = 0 then 1.0 else 0.0
    , wv := Mat.ofFn 1 2 fun _ j => if j.val = 0 then 1.0 else 0.0
    }
  let head1 : AttentionHead 2 1 :=
    { wq := Mat.ofFn 1 2 fun _ j => if j.val = 1 then 1.0 else 0.0
    , wk := Mat.ofFn 1 2 fun _ j => if j.val = 1 then 1.0 else 0.0
    , wv := Mat.ofFn 1 2 fun _ j => if j.val = 1 then 1.0 else 0.0
    }
  let mha : MultiHeadAttention 2 2 1 :=
    { heads := fun h => if h.val = 0 then head0 else head1
    , wo := Mat.identity 2
    , h_split := by decide
    , h_headDim := by decide
    }
  let sequence : TokenSeq 2 2 :=
    TokenSeq.ofFn 2 2 fun i =>
      match i.val with
      | 0 => Vec.ofFn 2 fun j => if j.val = 0 then 1.0 else 0.0
      | _ => Vec.ofFn 2 fun j => if j.val = 1 then 1.0 else 0.0

  let output := multiHeadAttention mha sequence ⟨0, by decide⟩
  IO.println s!"Multi-head output: {output.data.toList}"
  IO.println "Expected approximately: [0.7311, 0.5000]"

  let causalOutput := multiHeadAttention mha sequence ⟨0, by decide⟩ true
  IO.println s!"Causal multi-head output: {causalOutput.data.toList}"
  IO.println "Expected approximately: [1.0, 0.0]"

/-- Quick test: sinusoidal positional encoding. -/
def testPositionalEncoding : IO Unit := do
  let pos := sinusoidalEncoding 2 4
  IO.println s!"Positional encoding pos0: {(pos.get ⟨0, by decide⟩).data.toList}"
  IO.println s!"Positional encoding pos1: {(pos.get ⟨1, by decide⟩).data.toList}"
  IO.println "Expected pos0 approximately: [0.0, 1.0, 0.0, 1.0]"

/-- Quick test: UTF-8 byte tokenizer round-trip. -/
def testTokenizer : IO Unit := do
  let text := "Lean 4"
  let tokens := encodeUTF8Bytes text
  let roundTrip := decodeUTF8Bytes? tokens
  IO.println s!"UTF-8 byte tokens: {tokens.map (fun tok => tok.val)}"
  IO.println s!"Decoded text: {roundTrip.getD "<invalid utf8>"}"
  IO.println "Expected decoded text: Lean 4"

/-- A tiny zeroed transformer layer used for deterministic model tests. -/
def mkTinyLayer : TransformerLayer 2 2 1 2 :=
  { ln1 := LayerNormParams.ones 2
  , attn :=
      { heads := fun _ =>
          { wq := Mat.zeros 1 2
          , wk := Mat.zeros 1 2
          , wv := Mat.zeros 1 2
          }
      , wo := Mat.identity 2
      , h_split := by decide
      , h_headDim := by decide
      }
  , ln2 := LayerNormParams.ones 2
  , mlp_fc1 := Mat.zeros 2 2
  , mlp_fc2 := Mat.zeros 2 2
  }

/-- A deterministic tiny model for end-to-end forward testing. -/
def mkTinyModel : MicroGPT { d_model := 2, n_heads := 2, d_head := 1, d_ff := 2, n_layers := 1, vocab := 3 } :=
  { embedding := Mat.ofFn 2 3 fun i j =>
      match i.val, j.val with
      | 0, 0 => 1.0
      | 0, 1 => 0.0
      | 0, _ => 1.0
      | 1, 0 => 0.0
      | 1, 1 => 1.0
      | _, _ => 1.0
  , layers := [mkTinyLayer]
  , h_layers := rfl
  , ln_final :=
      { gamma := Vec.ofFn 2 (fun _ => 1.0)
      , beta := Vec.ofFn 2 fun i => if i.val = 0 then 1.0 else -1.0
      }
  , unembed := Mat.ofFn 3 2 fun i j =>
      match i.val, j.val with
      | 0, 0 => 1.0
      | 0, _ => 0.0
      | 1, 0 => 0.0
      | 1, _ => 1.0
      | _, 0 => 1.0
      | _, _ => 1.0
  }

/-- Quick test: full model logits and probabilities. -/
def testModelForward : IO Unit := do
  let model := mkTinyModel
  let tokenIds : List (Fin 3) := [⟨0, by decide⟩, ⟨1, by decide⟩]
  let predictIdx : Fin tokenIds.length := ⟨0, by decide⟩
  let logits := model.forward tokenIds predictIdx
  let probs := model.forwardProbs tokenIds predictIdx
  let loss := crossEntropyLoss logits ⟨0, by decide⟩
  IO.println s!"Model logits: {logits.data.toList}"
  IO.println s!"Model probs: {probs.data.toList}"
  IO.println s!"Model probs sum: {Vec.sum probs}"
  IO.println s!"Cross-entropy loss(target=0): {loss}"
  IO.println "Expected logits approximately: [1.0, -1.0, 0.0]"
  IO.println "Expected loss approximately: 0.4076"

/-- A tiny linear classifier for typed gradient and SGD tests. -/
def mkTinyLinearClassifier : LinearClassifier 2 3 :=
  { weight := Mat.zeros 3 2
  , bias := Vec.zeros 3
  }

/-- A tiny nonzero MLP block so gradients flow through both linear maps. -/
def mkTinyMLPBlock : MLPBlockParams 2 2 :=
  { ln :=
      { gamma := Vec.ofFn 2 (fun _ => 1.0)
      , beta := Vec.zeros 2
      }
  , fc1 := Mat.ofFn 2 2 fun i j =>
      match i.val, j.val with
      | 0, 0 => 0.2
      | 0, _ => -0.1
      | 1, 0 => 0.1
      | _, _ => 0.3
  , fc2 := Mat.ofFn 2 2 fun i j =>
      match i.val, j.val with
      | 0, 0 => 0.15
      | 0, _ => -0.05
      | 1, 0 => 0.05
      | _, _ => 0.10
  }

/-- A tiny nonzero single-head attention block so gradients flow through q/k/v/o. -/
def mkTinyAttentionBlock : AttentionBlockParams 2 2 :=
  { ln :=
      { gamma := Vec.ofFn 2 (fun _ => 1.0)
      , beta := Vec.zeros 2
      }
  , wq := Mat.ofFn 2 2 fun i j =>
      match i.val, j.val with
      | 0, 0 => 0.25
      | 0, _ => -0.10
      | 1, 0 => 0.05
      | _, _ => 0.20
  , wk := Mat.ofFn 2 2 fun i j =>
      match i.val, j.val with
      | 0, 0 => 0.15
      | 0, _ => 0.05
      | 1, 0 => -0.05
      | _, _ => 0.25
  , wv := Mat.ofFn 2 2 fun i j =>
      match i.val, j.val with
      | 0, 0 => 0.30
      | 0, _ => -0.20
      | 1, 0 => 0.10
      | _, _ => 0.15
  , wo := Mat.ofFn 2 2 fun i j =>
      match i.val, j.val with
      | 0, 0 => 0.20
      | 0, _ => -0.05
      | 1, 0 => 0.10
      | _, _ => 0.05
  }

/-- A sharp fixed output head for frozen-state attention/full-layer training tests. -/
def mkSharpOutputHead : OutputHeadParams 2 3 :=
  { ln := LayerNormParams.ones 2
  , unembed := Mat.ofFn 3 2 fun i j =>
      match i.val, j.val with
      | 0, 0 => 2.0
      | 0, _ => -1.0
      | 1, 0 => -1.0
      | 1, _ => 2.0
      | _, 0 => 0.5
      | _, _ => 0.5
  }

/-- A tiny trainable full transformer block fixture. -/
def mkTinyTransformerBlockTrain : TransformerBlockTrainParams 2 2 2 :=
  { attn := mkTinyAttentionBlock
  , mlp := mkTinyMLPBlock
  }

/-- Tiny token training example with a typed prediction position. -/
structure TokenExample (vocab : Nat) where
  tokens : List (Fin vocab)
  predictIdx : Fin tokens.length
  target : Fin vocab

/-- Last-token prediction index for a non-empty token list. -/
def lastPredictIdx (tokens : List α) (h : 0 < tokens.length) : Fin tokens.length :=
  ⟨tokens.length - 1, by simpa using Nat.sub_lt h (by decide : 0 < 1)⟩

/-- Repeat a list of strings `n` times. -/
def repeatStrings : Nat → List String → List String
  | 0, _ => []
  | n + 1, xs => xs ++ repeatStrings n xs

/-- Number of times to oversample the model's own source in the curriculum corpus. -/
def selfSourceRepeatCount : Nat := 4

/-- Short contextual Lean proof snippets in the style of Mathlib tactics. -/
def curatedMathlibStyleSnippets : List String :=
  [ "example : (1 : Nat) = 1 := by decide"
  , "example (n : Nat) : n = n := by rfl"
  , "example (a b : Nat) : a + b = b + a := by omega"
  , "example (xs : List Nat) : xs.length = xs.reverse.reverse.length := by simp"
  , "theorem add_zero_right (n : Nat) : n + 0 = n := by simp"
  ]

/-- Small general Lean 4 snippets emphasizing structures, instances, and functions. -/
def curatedLeanProgramSnippets : List String :=
  [ "structure Point where\n  x : Float\n  y : Float"
  , "def swapPair (p : Nat × Nat) : Nat × Nat :=\n  (p.2, p.1)"
  , "inductive Flag where\n  | on\n  | off"
  , "instance : Inhabited Flag where\n  default := Flag.off"
  , "def mapOption (f : Nat → Nat) : Option Nat → Option Nat\n  | none => none\n  | some n => some (f n)"
  ]

/-- Load the model's own source text, falling back to a small placeholder if unavailable. -/
def loadSelfSourceText : IO String := do
  try
    IO.FS.readFile "MicroGPT.lean"
  catch _ =>
    pure "/- self source unavailable -/"

/-- Build the prioritized training text corpus: self source, tactics/proofs, then small programs. -/
def buildPriorityTrainingTexts (selfSource : String) : List String :=
  repeatStrings selfSourceRepeatCount [selfSource] ++
    curatedMathlibStyleSnippets ++
    curatedLeanProgramSnippets

/-- Load the prioritized training text corpus from disk and curated snippets. -/
def loadPriorityTrainingTexts : IO (List String) := do
  let selfSource ← loadSelfSourceText
  pure <| buildPriorityTrainingTexts selfSource

/-- Build byte-level next-token examples from a UTF-8 token stream. -/
def byteTokenWindow (tokens : Array (Fin 256)) (start context : Nat) : List (Fin 256) :=
  (List.finRange context).map (fun i => tokens[start + i.val]!)

/-- Build byte-level next-token examples from a UTF-8 token stream. -/
def byteTokenExamplesFromStream (context : Nat) (tokens : List (Fin 256)) : List (TokenExample 256) :=
  if h_ctx : 0 < context then
    let tokenArray := tokens.toArray
    let reversed :=
      (List.range tokenArray.size).foldl (fun acc start =>
        if h : start + context < tokenArray.size then
          let window := byteTokenWindow tokenArray start context
          let target := tokenArray[start + context]!
          have h_window_len : window.length = context := by
            simp [window, byteTokenWindow]
          have h_window_pos : 0 < window.length := by
            rw [h_window_len]
            exact h_ctx
          { tokens := window
          , predictIdx := lastPredictIdx window h_window_pos
          , target := target
          } :: acc
        else
          acc
      ) []
    reversed.reverse
  else
    []

/-- Build byte-level next-token examples from one UTF-8 text. -/
def byteTokenExamplesFromText (context : Nat) (text : String) : List (TokenExample 256) :=
  byteTokenExamplesFromStream context (encodeUTF8Bytes text)

/-- Build byte-level next-token examples from many UTF-8 texts. -/
def byteTokenExamplesFromTexts (context : Nat) (texts : List String) : List (TokenExample 256) :=
  texts.foldr (fun text acc => byteTokenExamplesFromText context text ++ acc) []

/-- Number of byte-level next-token examples contributed by one UTF-8 text. -/
def byteTokenExampleCountFromText (context : Nat) (text : String) : Nat :=
  let tokenCount := (encodeUTF8Bytes text).length
  if context < tokenCount then tokenCount - context else 0

/-- Number of byte-level next-token examples contributed by many UTF-8 texts. -/
def byteTokenExampleCountFromTexts (context : Nat) (texts : List String) : Nat :=
  texts.foldl (fun acc text => acc + byteTokenExampleCountFromText context text) 0

/-- Take only the first `limit` byte-level next-token examples from many UTF-8 texts. -/
def takeByteTokenExamplesFromTexts (context limit : Nat) (texts : List String) : List (TokenExample 256) :=
  let rec go (remaining : Nat) (rest : List String) : List (TokenExample 256) :=
    match remaining, rest with
    | 0, _ => []
    | _, [] => []
    | remaining + 1, text :: tail =>
        let current := byteTokenExamplesFromText context text
        let taken := current.take (remaining + 1)
        if taken.length < remaining + 1 then
          taken ++ go (remaining + 1 - taken.length) tail
        else
          taken
  go limit texts

/-- Deterministic train/validation split. Every `holdoutEvery`-th element goes to validation. -/
structure DatasetSplit (α : Type) where
  train : List α
  valid : List α

/-- Split a dataset deterministically by index. -/
def splitDatasetEvery {α : Type} (holdoutEvery : Nat) (xs : List α) : DatasetSplit α :=
  let rec go : Nat → List α → DatasetSplit α
    | _, [] => { train := [], valid := [] }
    | idx, x :: rest =>
        let tail := go (idx + 1) rest
        if holdoutEvery > 0 && idx % holdoutEvery = 0 then
          { train := tail.train, valid := x :: tail.valid }
        else
          { train := x :: tail.train, valid := tail.valid }
  go 0 xs

/-- Split texts by file before converting them into byte-token examples. -/
def splitByteTokenExamplesFromTexts (context holdoutEvery : Nat) (texts : List String) :
    DatasetSplit (TokenExample 256) :=
  let split := splitDatasetEvery holdoutEvery texts
  { train := byteTokenExamplesFromTexts context split.train
  , valid := byteTokenExamplesFromTexts context split.valid
  }

/-- Split texts by file and take only the first needed byte-token examples from each side. -/
def takeSplitByteTokenExamplesFromTexts
    (context holdoutEvery nTrain nValid : Nat) (texts : List String) :
    DatasetSplit (TokenExample 256) :=
  let split := splitDatasetEvery holdoutEvery texts
  { train := takeByteTokenExamplesFromTexts context nTrain split.train
  , valid := takeByteTokenExamplesFromTexts context nValid split.valid
  }

/-- Count byte-token training/validation examples after splitting texts by file. -/
def countSplitByteTokenExamplesFromTexts (context holdoutEvery : Nat) (texts : List String) :
    Nat × Nat :=
  let split := splitDatasetEvery holdoutEvery texts
  (byteTokenExampleCountFromTexts context split.train, byteTokenExampleCountFromTexts context split.valid)

/-- Deterministically batch a list into chunks of size `batchSize`. -/
def batchesOf {α : Type} (batchSize : Nat) (xs : List α) : List (List α) :=
  if 0 < batchSize then
    (List.range ((xs.length + batchSize - 1) / batchSize)).map fun i =>
      xs.drop (i * batchSize) |>.take batchSize
  else
    [xs]

/-- Mean loss over a token dataset for an arbitrary model. -/
def tokenDatasetLoss (model : MicroGPT cfg) (dataset : List (TokenExample cfg.vocab)) : Float :=
  let total := dataset.foldl (fun acc ex => acc + crossEntropyLoss (model.forward ex.tokens ex.predictIdx) ex.target) 0.0
  let denom := if dataset.length == 0 then 1.0 else Float.ofNat dataset.length
  total / denom

/-- One log entry for a training epoch. -/
structure TrainingEpochLog where
  epoch : Nat
  trainLoss : Float
  validLoss : Float

/-- Replace the element at index `idx`, keeping list length unchanged. -/
def listReplaceAt (xs : List α) (idx : Nat) (value : α) : List α :=
  match xs, idx with
  | [], _ => []
  | _ :: rest, 0 => value :: rest
  | x :: rest, idx + 1 => x :: listReplaceAt rest idx value

/-- Replacing an element keeps list length unchanged. -/
theorem listReplaceAt_length (xs : List α) (idx : Nat) (value : α) :
    (listReplaceAt xs idx value).length = xs.length := by
  induction xs generalizing idx with
  | nil =>
      cases idx <;> rfl
  | cons x rest ih =>
      cases idx with
      | zero => rfl
      | succ idx =>
          simp [listReplaceAt, ih]

/-- Apply all but the last transformer layer in order. -/
def applyAllButLastLayers (layers : List (TransformerLayer d_model n_heads d_head d_ff))
    (sequence : TokenSeq seqLen d_model) : TokenSeq seqLen d_model :=
  match layers with
  | [] => sequence
  | [_] => sequence
  | layer :: rest => applyAllButLastLayers rest (applyLayer layer sequence)

/-- Get a transformer layer by a config-checked index. -/
def MicroGPT.getLayer (model : MicroGPT cfg) (layerIdx : Fin cfg.n_layers) :
    TransformerLayer cfg.d_model cfg.n_heads cfg.d_head cfg.d_ff :=
  model.layers.get ⟨layerIdx.val, by
    rw [model.h_layers]
    exact layerIdx.isLt⟩

/-- Replace one transformer layer inside a model while preserving the layer-count invariant. -/
def MicroGPT.withLayer (model : MicroGPT cfg) (layerIdx : Fin cfg.n_layers)
    (layer : TransformerLayer cfg.d_model cfg.n_heads cfg.d_head cfg.d_ff) : MicroGPT cfg :=
  let newLayers := listReplaceAt model.layers layerIdx.val layer
  { embedding := model.embedding
  , layers := newLayers
  , h_layers := by
      rw [listReplaceAt_length, model.h_layers]
  , ln_final := model.ln_final
  , unembed := model.unembed
  }

/-- Config-checked index of the last layer in a non-empty stack. -/
def lastLayerIdx (n : Nat) (h_nonempty : 0 < n) : Fin n :=
  ⟨n - 1, by simpa using Nat.sub_lt h_nonempty (by decide : 0 < 1)⟩

/-- Extract the final trainable layer input sequence from an actual model. -/
def MicroGPT.lastLayerInputSequence (model : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab)) : TokenSeq tokenIds.length cfg.d_model :=
  applyAllButLastLayers model.layers (model.inputSequence tokenIds)

/-- Convert an actual transformer layer into the generic trainable full-layer scaffold. -/
def fullTransformerLayerTrainOfLayer (layer : TransformerLayer d_model n_heads d_head d_ff) :
    FullTransformerLayerTrainParams d_model n_heads d_head d_ff :=
  { ln1 := layer.ln1
  , heads := layer.attn.heads
  , wo := layer.attn.wo
  , h_split := layer.attn.h_split
  , h_headDim := layer.attn.h_headDim
  , ln2 := layer.ln2
  , fc1 := layer.mlp_fc1
  , fc2 := layer.mlp_fc2
  }

/-- Convert the generic trainable full-layer scaffold back into an actual transformer layer. -/
def layerOfFullTransformerLayerTrain (params : FullTransformerLayerTrainParams d_model n_heads d_head d_ff) :
    TransformerLayer d_model n_heads d_head d_ff :=
  { ln1 := params.ln1
  , attn :=
      { heads := params.heads
      , wo := params.wo
      , h_split := params.h_split
      , h_headDim := params.h_headDim
      }
  , ln2 := params.ln2
  , mlp_fc1 := params.fc1
  , mlp_fc2 := params.fc2
  }

/-- Train the last actual transformer layer of a non-empty model on one token example. -/
def trainModelLastLayerOnExample (model : MicroGPT cfg)
    (h_nonempty : 0 < cfg.n_layers)
    (ex : TokenExample cfg.vocab)
    (lr : Float) (steps : Nat) : MicroGPT cfg :=
  let layerIdx := lastLayerIdx cfg.n_layers h_nonempty
  let params := fullTransformerLayerTrainOfLayer (model.getLayer layerIdx)
  let sequence := model.lastLayerInputSequence ex.tokens
  let trained :=
    trainFullTransformerLayerTrainSteps
      params
      (outputHeadFromModel model)
      sequence
      ex.predictIdx
      ex.target
      lr
      steps
  model.withLayer layerIdx (layerOfFullTransformerLayerTrain trained)

/-- One batched epoch of last-layer training for a non-empty model. -/
def trainModelLastLayerEpoch (model : MicroGPT cfg)
    (h_nonempty : 0 < cfg.n_layers)
    (dataset : List (TokenExample cfg.vocab))
    (batchSize : Nat)
    (lr : Float) (stepsPerExample : Nat) : MicroGPT cfg :=
  (batchesOf batchSize dataset).foldl
    (fun current batch =>
      batch.foldl (fun batchModel ex => trainModelLastLayerOnExample batchModel h_nonempty ex lr stepsPerExample) current)
    model

/-- Batched last-layer training with epoch-wise train/validation logging. -/
def trainModelLastLayerEpochsLogged (model : MicroGPT cfg)
    (h_nonempty : 0 < cfg.n_layers)
    (split : DatasetSplit (TokenExample cfg.vocab))
    (batchSize : Nat)
    (lr : Float) (stepsPerExample epochs : Nat) :
    MicroGPT cfg × List TrainingEpochLog :=
  let rec go (current : MicroGPT cfg) (epoch : Nat) : Nat → MicroGPT cfg × List TrainingEpochLog
    | 0 => (current, [])
    | remaining + 1 =>
        let trained := trainModelLastLayerEpoch current h_nonempty split.train batchSize lr stepsPerExample
        let entry :=
          { epoch := epoch
          , trainLoss := tokenDatasetLoss trained split.train
          , validLoss := tokenDatasetLoss trained split.valid
          }
        let (finalModel, logs) := go trained (epoch + 1) remaining
        (finalModel, entry :: logs)
  go model 1 epochs

/-- Flatten all trainable MicroGPT parameters into one list. -/
def flattenMicroGPTParams (model : MicroGPT cfg) : List Float :=
  let layerParams :=
    ((List.finRange cfg.n_layers).map fun i => flattenTransformerLayerParams (model.getLayer i)).foldr List.append []
  model.embedding.toList ++ layerParams ++ flattenOutputHeadParams (outputHeadFromModel model)

/-- Total trainable parameter count for a MicroGPT configuration. -/
def microGPTParamCount (cfg : GPTConfig) : Nat :=
  cfg.d_model * cfg.vocab +
    cfg.n_layers * fullTransformerLayerParamCount cfg.d_model cfg.n_heads cfg.d_head cfg.d_ff +
    (2 * cfg.d_model + cfg.vocab * cfg.d_model)

/-- Rebuild a MicroGPT from flat parameters using an existing model for structural proofs. -/
def microGPTOfFlatLike (template : MicroGPT cfg) (xs : List Float) : MicroGPT cfg :=
  let embedCount := cfg.d_model * cfg.vocab
  let layerCount := fullTransformerLayerParamCount cfg.d_model cfg.n_heads cfg.d_head cfg.d_ff
  let headStart := embedCount + cfg.n_layers * layerCount
  let layers :=
    (List.finRange cfg.n_layers).map fun i =>
      let offset := embedCount + i.val * layerCount
      transformerLayerOfFlatLike (template.getLayer i) (xs.drop offset)
  let h_layers : layers.length = cfg.n_layers := by
    simp [layers]
  let head := outputHeadParamsOfList cfg.d_model cfg.vocab (xs.drop headStart)
  { embedding := matOfList cfg.d_model cfg.vocab xs
  , layers := layers
  , h_layers := h_layers
  , ln_final := head.ln
  , unembed := head.unembed
  }

/-- Subtract a scaled gradient list from a parameter list. -/
def sgdStepFlat (params grads : List Float) (lr : Float) : List Float :=
  List.zipWith (fun p g => p - lr * g) params grads

/-- Adam optimizer state over a flat parameter vector. -/
structure AdamState where
  m : List Float
  v : List Float
  step : Nat

/-- Zero-initialized Adam state for a flat parameter vector. -/
def initAdamState (paramCount : Nat) : AdamState :=
  { m := List.replicate paramCount 0.0
  , v := List.replicate paramCount 0.0
  , step := 0
  }

/-- One flat Adam update, returning updated parameters and optimizer state. -/
def adamStepFlat (params grads : List Float) (state : AdamState)
    (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999) (eps : Float := 1.0e-8) :
    List Float × AdamState :=
  let step := state.step + 1
  let oneMinusBeta1 := 1.0 - beta1
  let oneMinusBeta2 := 1.0 - beta2
  let bias1 := 1.0 - Float.pow beta1 (Float.ofNat step)
  let bias2 := 1.0 - Float.pow beta2 (Float.ofNat step)
  let rec go :
      List Float → List Float → List Float → List Float →
      List Float → List Float → List Float →
      (List Float × List Float × List Float)
    | [], [], [], [], accParams, accM, accV => (accParams.reverse, accM.reverse, accV.reverse)
    | p :: ps, g :: gs, mPrev :: ms, vPrev :: vs, accParams, accM, accV =>
        let m := beta1 * mPrev + oneMinusBeta1 * g
        let v := beta2 * vPrev + oneMinusBeta2 * g * g
        let mHat := if bias1 == 0.0 then m else m / bias1
        let vHat := if bias2 == 0.0 then v else v / bias2
        let denom := Float.sqrt vHat + eps
        let p' := p - lr * mHat / denom
        go ps gs ms vs (p' :: accParams) (m :: accM) (v :: accV)
    | _, _, _, _, accParams, accM, accV => (accParams.reverse, accM.reverse, accV.reverse)
  let (newParams, newM, newV) := go params grads state.m state.v [] [] []
  (newParams, { m := newM, v := newV, step := step })

/-- Trace the final hidden state of a full trainable MicroGPT from flat parameters. -/
def traceMicroGPTHiddenStateFlatLike (template : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab))
    (predictIdx : Fin tokenIds.length)
    (params : List TracedFloat) : TraceM (TracedVec cfg.d_model) := do
  let embedCount := cfg.d_model * cfg.vocab
  let layerCount := fullTransformerLayerParamCount cfg.d_model cfg.n_heads cfg.d_head cfg.d_ff
  let embedding := traceMatFromFlatOffset cfg.d_model cfg.vocab 0 params
  let inputSeq ← (List.finRange tokenIds.length).mapM fun i => do
    let tokId := tokenIds.get i
    let embedded := TracedVec.ofFn cfg.d_model (fun rowIdx => embedding.get rowIdx tokId)
    let pos ← traceConstVec (sinusoidalPosition cfg.d_model i.val)
    traceAddVec embedded pos
  let finalSeq ← (List.finRange cfg.n_layers).foldlM (fun currentSeq layerIdx => do
    let offset := embedCount + layerIdx.val * layerCount
    let templateLayer := template.getLayer layerIdx
    traceApplyFullTransformerLayerTraced
      cfg.d_model
      cfg.n_heads
      cfg.d_head
      cfg.d_ff
      tokenIds.length
      templateLayer.attn.h_headDim
      currentSeq
      (params.drop offset)
      true
  ) inputSeq
  match getListTracedVec? finalSeq predictIdx.val with
  | some x => pure x
  | none => panic! "traceMicroGPTHiddenStateFlatLike: impossible prediction index"

/-- Trace full-model logits from flat trainable parameters. -/
def traceMicroGPTLogitsFlatLike (template : MicroGPT cfg)
    (tokenIds : List (Fin cfg.vocab))
    (predictIdx : Fin tokenIds.length)
    (params : List TracedFloat) : TraceM (TracedVec cfg.vocab) := do
  let embedCount := cfg.d_model * cfg.vocab
  let layerCount := fullTransformerLayerParamCount cfg.d_model cfg.n_heads cfg.d_head cfg.d_ff
  let headStart := embedCount + cfg.n_layers * layerCount
  let hidden ← traceMicroGPTHiddenStateFlatLike template tokenIds predictIdx params
  traceOutputHeadForwardTracedFlat cfg.d_model cfg.vocab hidden (params.drop headStart)

/-- Trace full-model cross-entropy loss from flat trainable parameters. -/
def traceMicroGPTLossFlatLike (template : MicroGPT cfg)
    (ex : TokenExample cfg.vocab)
    (params : List TracedFloat) : TraceM TracedFloat := do
  let logits ← traceMicroGPTLogitsFlatLike template ex.tokens ex.predictIdx params
  traceCrossEntropyVec logits ex.target

/-- Reverse-mode gradient of the full MicroGPT loss with respect to all trainable parameters. -/
def reverseGradientMicroGPT (model : MicroGPT cfg) (ex : TokenExample cfg.vocab) : List Float :=
  reverseGradientList (traceMicroGPTLossFlatLike model ex) (flattenMicroGPTParams model)

/-- Finite-difference gradient of the full MicroGPT loss with respect to all trainable parameters. -/
def finiteDiffMicroGPT (model : MicroGPT cfg) (ex : TokenExample cfg.vocab) : List Float :=
  let flatParams := flattenMicroGPTParams model
  let lossFromFlat := fun xs => crossEntropyLoss ((microGPTOfFlatLike model xs).forward ex.tokens ex.predictIdx) ex.target
  (List.range flatParams.length).map (fun idx => finiteDiffList lossFromFlat flatParams idx)

/-- One full-model SGD step over all trainable MicroGPT parameters. -/
def sgdStepMicroGPT (model : MicroGPT cfg) (flatGrads : List Float) (lr : Float) : MicroGPT cfg :=
  microGPTOfFlatLike model (sgdStepFlat (flattenMicroGPTParams model) flatGrads lr)

/-- One full-model Adam step over all trainable MicroGPT parameters. -/
def adamStepMicroGPT (model : MicroGPT cfg) (state : AdamState) (flatGrads : List Float)
    (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999) (eps : Float := 1.0e-8) :
    MicroGPT cfg × AdamState :=
  let (newParams, newState) := adamStepFlat (flattenMicroGPTParams model) flatGrads state lr beta1 beta2 eps
  (microGPTOfFlatLike model newParams, newState)

/-- Repeated full-model SGD steps on a single token example. -/
def trainMicroGPTOnExample (model : MicroGPT cfg) (ex : TokenExample cfg.vocab) (lr : Float) : Nat → MicroGPT cfg
  | 0 => model
  | steps + 1 =>
      let grads := reverseGradientMicroGPT model ex
      trainMicroGPTOnExample (sgdStepMicroGPT model grads lr) ex lr steps

/-- Repeated full-model Adam steps on a single token example. -/
def trainMicroGPTOnExampleAdam (model : MicroGPT cfg) (state : AdamState)
    (ex : TokenExample cfg.vocab)
    (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999) (eps : Float := 1.0e-8) :
    Nat → MicroGPT cfg × AdamState
  | 0 => (model, state)
  | steps + 1 =>
      let grads := reverseGradientMicroGPT model ex
      let (nextModel, nextState) := adamStepMicroGPT model state grads lr beta1 beta2 eps
      trainMicroGPTOnExampleAdam nextModel nextState ex lr beta1 beta2 eps steps

/-- One full-model epoch over a token dataset. -/
def trainMicroGPTEpoch (model : MicroGPT cfg)
    (dataset : List (TokenExample cfg.vocab))
    (batchSize : Nat)
    (lr : Float) (stepsPerExample : Nat) : MicroGPT cfg :=
  (batchesOf batchSize dataset).foldl
    (fun current batch =>
      batch.foldl (fun batchModel ex => trainMicroGPTOnExample batchModel ex lr stepsPerExample) current)
    model

/-- One full-model Adam epoch over a token dataset. -/
def trainMicroGPTEpochAdam (model : MicroGPT cfg) (state : AdamState)
    (dataset : List (TokenExample cfg.vocab))
    (batchSize : Nat)
    (lr : Float) (stepsPerExample : Nat)
    (beta1 : Float := 0.9) (beta2 : Float := 0.999) (eps : Float := 1.0e-8) :
    MicroGPT cfg × AdamState :=
  (batchesOf batchSize dataset).foldl
    (fun (current : MicroGPT cfg × AdamState) batch =>
      batch.foldl
        (fun (inner : MicroGPT cfg × AdamState) ex =>
          let (batchModel, batchState) := inner
          trainMicroGPTOnExampleAdam batchModel batchState ex lr beta1 beta2 eps stepsPerExample)
        current)
    (model, state)

/-- Full-model batched training with epoch-wise train/validation logging. -/
def trainMicroGPTEpochsLogged (model : MicroGPT cfg)
    (split : DatasetSplit (TokenExample cfg.vocab))
    (batchSize : Nat)
    (lr : Float) (stepsPerExample epochs : Nat) :
    MicroGPT cfg × List TrainingEpochLog :=
  let rec go (current : MicroGPT cfg) (epoch : Nat) : Nat → MicroGPT cfg × List TrainingEpochLog
    | 0 => (current, [])
    | remaining + 1 =>
        let trained := trainMicroGPTEpoch current split.train batchSize lr stepsPerExample
        let entry : TrainingEpochLog :=
          { epoch := epoch
          , trainLoss := tokenDatasetLoss trained split.train
          , validLoss := tokenDatasetLoss trained split.valid
          }
        let (finalModel, logs) := go trained (epoch + 1) remaining
        (finalModel, entry :: logs)
  go model 1 epochs

/-- Full-model Adam training with epoch-wise train/validation logging. -/
def trainMicroGPTEpochsLoggedAdam (model : MicroGPT cfg)
    (split : DatasetSplit (TokenExample cfg.vocab))
    (batchSize : Nat)
    (lr : Float) (stepsPerExample epochs : Nat)
    (beta1 : Float := 0.9) (beta2 : Float := 0.999) (eps : Float := 1.0e-8) :
    MicroGPT cfg × AdamState × List TrainingEpochLog :=
  let rec go (current : MicroGPT cfg) (state : AdamState) (epoch : Nat) :
      Nat → MicroGPT cfg × AdamState × List TrainingEpochLog
    | 0 => (current, state, [])
    | remaining + 1 =>
        let (trained, nextState) := trainMicroGPTEpochAdam current state split.train batchSize lr stepsPerExample beta1 beta2 eps
        let entry : TrainingEpochLog :=
          { epoch := epoch
          , trainLoss := tokenDatasetLoss trained split.train
          , validLoss := tokenDatasetLoss trained split.valid
          }
        let (finalModel, finalState, logs) := go trained nextState (epoch + 1) remaining
        (finalModel, finalState, entry :: logs)
  go model (initAdamState (microGPTParamCount cfg)) 1 epochs

/-- Encode a `UInt64` as 8 little-endian bytes. -/
def encodeUInt64LE (x : UInt64) : ByteArray :=
  (List.range 8).foldl (fun acc i =>
    let shifted := UInt64.shiftRight x (UInt64.ofNat (8 * i))
    acc.push shifted.toUInt8
  ) ByteArray.empty

/-- Decode 8 little-endian bytes from a byte array offset. -/
def decodeUInt64LE? (bytes : ByteArray) (offset : Nat) : Option UInt64 :=
  if offset + 8 ≤ bytes.size then
    let natVal :=
      (List.range 8).foldl (fun acc i =>
        let byte := (bytes.get! (offset + i)).toNat
        acc + Nat.shiftLeft byte (8 * i)
      ) 0
    some (UInt64.ofNat natVal)
  else
    none

/-- Encode a list of floats as raw IEEE-754 bits. -/
def encodeFloatList (xs : List Float) : ByteArray :=
  xs.foldl (fun acc x =>
    let chunk := encodeUInt64LE (Float.toBits x)
    (List.finRange chunk.size).foldl (fun inner i => inner.push (chunk.get! i.val)) acc
  ) ByteArray.empty

/-- Decode a fixed number of floats from raw IEEE-754 bytes. -/
def decodeFloatList? (count : Nat) (bytes : ByteArray) (offset : Nat := 0) : Option (List Float) :=
  if offset + 8 * count ≤ bytes.size then
    some <|
      (List.range count).map fun i =>
        match decodeUInt64LE? bytes (offset + 8 * i) with
        | some bits => Float.ofBits bits
        | none => 0.0
  else
    none

/-- Save a MicroGPT checkpoint as `[paramCount:u64][float_bits...]`. -/
def saveMicroGPTCheckpoint (path : System.FilePath) (model : MicroGPT cfg) : IO Unit := do
  let flat := flattenMicroGPTParams model
  let header := encodeUInt64LE (UInt64.ofNat flat.length)
  let payload := encodeFloatList flat
  let bytes :=
    (List.finRange payload.size).foldl (fun acc i => acc.push (payload.get! i.val)) header
  IO.FS.writeBinFile path bytes

/-- Load a MicroGPT checkpoint from `[paramCount:u64][float_bits...]`, validated against the target config. -/
def loadMicroGPTCheckpoint (template : MicroGPT cfg) (path : System.FilePath) :
    IO (Except String (MicroGPT cfg)) := do
  let bytes ← IO.FS.readBinFile path
  match decodeUInt64LE? bytes 0 with
  | none => pure <| Except.error "checkpoint missing parameter-count header"
  | some countBits =>
      let count := countBits.toNat
      let expected := microGPTParamCount cfg
      if count != expected then
        pure <| Except.error s!"checkpoint param count mismatch: file={count}, expected={expected}"
      else
        match decodeFloatList? count bytes 8 with
        | none => pure <| Except.error "checkpoint payload truncated"
        | some flat => pure <| Except.ok (microGPTOfFlatLike template flat)

/-- Convert a trainable block fixture into an actual single-layer transformer layer. -/
def tinyLayerOfTrainBlock (params : TransformerBlockTrainParams 2 2 2) : TransformerLayer 2 1 2 2 :=
  { ln1 := params.attn.ln
  , attn :=
      { heads := fun _ =>
          { wq := params.attn.wq
          , wk := params.attn.wk
          , wv := params.attn.wv
          }
      , wo := params.attn.wo
      , h_split := by simp
      , h_headDim := by decide
      }
  , ln2 := params.mlp.ln
  , mlp_fc1 := params.mlp.fc1
  , mlp_fc2 := params.mlp.fc2
  }

/-- Extract the trainable block parameters from a single-layer single-head transformer layer. -/
def tinyTrainBlockOfLayer (layer : TransformerLayer 2 1 2 2) : TransformerBlockTrainParams 2 2 2 :=
  let head := layer.attn.heads ⟨0, by decide⟩
  { attn :=
      { ln := layer.ln1
      , wq := head.wq
      , wk := head.wk
      , wv := head.wv
      , wo := layer.attn.wo
      }
  , mlp :=
      { ln := layer.ln2
      , fc1 := layer.mlp_fc1
      , fc2 := layer.mlp_fc2
      }
  }

/-- A tiny single-head MicroGPT whose input sequence matches the frozen block-training fixture on `[0,1]`. -/
def mkTinySingleHeadModel :
    MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 } :=
  { embedding := Mat.ofFn 2 3 fun i j =>
      match i.val, j.val with
      | 0, 0 => 2.0
      | 1, 0 => -2.0
      | 0, 1 => -0.5914709848
      | 1, 1 => 0.9596976941
      | 0, _ => 0.5
      | _, _ => -0.5
  , layers := [tinyLayerOfTrainBlock mkTinyTransformerBlockTrain]
  , h_layers := rfl
  , ln_final := mkSharpOutputHead.ln
  , unembed := mkSharpOutputHead.unembed
  }

/-- Extract the single trainable transformer block from the tiny single-head model. -/
def tinySingleHeadModelBlock
    (model : MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 }) :
    TransformerBlockTrainParams 2 2 2 :=
  match model.layers with
  | [layer] => tinyTrainBlockOfLayer layer
  | _ => mkTinyTransformerBlockTrain

/-- Replace the single trainable transformer block inside the tiny single-head model. -/
def withTinySingleHeadModelBlock
    (model : MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 })
    (params : TransformerBlockTrainParams 2 2 2) :
    MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 } :=
  { embedding := model.embedding
  , layers := [tinyLayerOfTrainBlock params]
  , h_layers := rfl
  , ln_final := model.ln_final
  , unembed := model.unembed
  }

/-- Loss of the tiny single-head model on one token example. -/
def tinySingleHeadModelLoss
    (model : MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 })
    (ex : TokenExample 3) : Float :=
  crossEntropyLoss (model.forward ex.tokens ex.predictIdx) ex.target

/-- Train the actual transformer block inside the tiny single-head model on one example. -/
def trainTinySingleHeadModelOnExample
    (model : MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 })
    (ex : TokenExample 3)
    (lr : Float) (steps : Nat) :
    MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 } :=
  let block := tinySingleHeadModelBlock model
  let trained :=
    trainTransformerBlockTrainSteps
      block
      (outputHeadFromModel model)
      (model.inputSequence ex.tokens)
      ex.predictIdx
      ex.target
      lr
      steps
  withTinySingleHeadModelBlock model trained

/-- Mean loss over a tiny token dataset. -/
def tinyTokenDatasetLoss
    (model : MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 })
    (dataset : List (TokenExample 3)) : Float :=
  let total := dataset.foldl (fun acc ex => acc + tinySingleHeadModelLoss model ex) 0.0
  let denom := if dataset.length == 0 then 1.0 else Float.ofNat dataset.length
  total / denom

/-- One training epoch over a tiny token dataset. -/
def trainTinyTokenDatasetEpoch
    (model : MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 })
    (dataset : List (TokenExample 3))
    (lr : Float) (stepsPerExample : Nat) :
    MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 } :=
  dataset.foldl (fun current ex => trainTinySingleHeadModelOnExample current ex lr stepsPerExample) model

/-- Repeated training epochs over a tiny token dataset. -/
def trainTinyTokenDatasetEpochs
    (model : MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 })
    (dataset : List (TokenExample 3))
    (lr : Float) (stepsPerExample : Nat) : Nat →
    MicroGPT { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 }
  | 0 => model
  | epochs + 1 =>
      trainTinyTokenDatasetEpochs (trainTinyTokenDatasetEpoch model dataset lr stepsPerExample) dataset lr stepsPerExample epochs

/-- Tiny dataset used to exercise actual-model token training. -/
def tinyTokenDataset : List (TokenExample 3) :=
  [ { tokens := [⟨0, by decide⟩, ⟨1, by decide⟩]
    , predictIdx := ⟨1, by decide⟩
    , target := ⟨0, by decide⟩
    }
  ]

/-- A nonzero multi-head transformer layer used to exercise generic last-layer training. -/
def mkTinyTrainableMultiHeadLayer : TransformerLayer 2 2 1 2 :=
  let head0 : AttentionHead 2 1 :=
    { wq := Mat.ofFn 1 2 fun _ j => if j.val = 0 then 1.0 else 0.0
    , wk := Mat.ofFn 1 2 fun _ j => if j.val = 0 then 1.0 else 0.0
    , wv := Mat.ofFn 1 2 fun _ j => if j.val = 0 then 1.0 else 0.0
    }
  let head1 : AttentionHead 2 1 :=
    { wq := Mat.ofFn 1 2 fun _ j => if j.val = 1 then 1.0 else 0.0
    , wk := Mat.ofFn 1 2 fun _ j => if j.val = 1 then 1.0 else 0.0
    , wv := Mat.ofFn 1 2 fun _ j => if j.val = 1 then 1.0 else 0.0
    }
  { ln1 := LayerNormParams.ones 2
  , attn :=
      { heads := fun h => if h.val = 0 then head0 else head1
      , wo := Mat.identity 2
      , h_split := by decide
      , h_headDim := by decide
      }
  , ln2 := mkTinyMLPBlock.ln
  , mlp_fc1 := mkTinyMLPBlock.fc1
  , mlp_fc2 := mkTinyMLPBlock.fc2
  }

/-- A two-layer multi-head model used to exercise generic actual-model last-layer training. -/
def mkTinyTwoLayerModel :
    MicroGPT { d_model := 2, n_heads := 2, d_head := 1, d_ff := 2, n_layers := 2, vocab := 3 } :=
  { embedding := mkTinySingleHeadModel.embedding
  , layers := [mkTinyLayer, mkTinyTrainableMultiHeadLayer]
  , h_layers := rfl
  , ln_final := mkSharpOutputHead.ln
  , unembed := mkSharpOutputHead.unembed
  }

/-- Tiny dataset for the generic multi-head actual-model last-layer trainer. -/
def tinyTwoLayerTokenDataset : List (TokenExample 3) :=
  [ { tokens := [⟨0, by decide⟩, ⟨1, by decide⟩]
    , predictIdx := ⟨1, by decide⟩
    , target := ⟨0, by decide⟩
    }
  , { tokens := [⟨1, by decide⟩, ⟨0, by decide⟩]
    , predictIdx := ⟨1, by decide⟩
    , target := ⟨1, by decide⟩
    }
  ]

/-- A small CPU-friendly byte-level configuration for real `vocab = 256` training smoke tests. -/
def mkByteConfig : GPTConfig :=
  { d_model := 16, n_heads := 1, d_head := 16, d_ff := 32, n_layers := 1, vocab := 256 }

/-- Generic patterned attention head used to build deterministic byte-model templates. -/
def mkPatternedAttentionHead (d_model d_head seed : Nat) : AttentionHead d_model d_head :=
  { wq := patternedMat d_head d_model (seed + 100)
  , wk := patternedMat d_head d_model (seed + 200)
  , wv := patternedMat d_head d_model (seed + 300)
  }

/-- Generic patterned transformer layer used to build deterministic byte-model templates. -/
def mkPatternedByteLayer (cfg : GPTConfig) (seed : Nat)
    (h_split : cfg.d_model = cfg.n_heads * cfg.d_head)
    (h_headDim : 0 < cfg.d_head) :
    TransformerLayer cfg.d_model cfg.n_heads cfg.d_head cfg.d_ff :=
  { ln1 := LayerNormParams.ones cfg.d_model
  , attn :=
      { heads := fun h => mkPatternedAttentionHead cfg.d_model cfg.d_head (seed + h.val * 17)
      , wo := patternedMat cfg.d_model (cfg.n_heads * cfg.d_head) (seed + 400)
      , h_split := h_split
      , h_headDim := h_headDim
      }
  , ln2 := LayerNormParams.ones cfg.d_model
  , mlp_fc1 := patternedMat cfg.d_ff cfg.d_model (seed + 500)
  , mlp_fc2 := patternedMat cfg.d_model cfg.d_ff (seed + 600)
  }

/-- Generic deterministic byte-level MicroGPT template for arbitrary byte-model configs. -/
def mkPatternedByteModel (cfg : GPTConfig)
    (h_split : cfg.d_model = cfg.n_heads * cfg.d_head)
    (h_headDim : 0 < cfg.d_head) : MicroGPT cfg :=
  let layers :=
    (List.finRange cfg.n_layers).map fun i =>
      mkPatternedByteLayer cfg (1000 + i.val * 100) h_split h_headDim
  let h_layers : layers.length = cfg.n_layers := by
    simp [layers]
  { embedding := patternedMat cfg.d_model cfg.vocab 10
  , layers := layers
  , h_layers := h_layers
  , ln_final := LayerNormParams.ones cfg.d_model
  , unembed := patternedMat cfg.vocab cfg.d_model 700
  }

/-- Deterministic single-layer byte-level transformer fixture. -/
def mkByteLayer : TransformerLayer 16 1 16 32 :=
  { ln1 := LayerNormParams.ones 16
  , attn :=
      { heads := fun h =>
          { wq := patternedMat 16 16 (100 + h.val * 11)
          , wk := patternedMat 16 16 (200 + h.val * 11)
          , wv := patternedMat 16 16 (300 + h.val * 11)
          }
      , wo := patternedMat 16 16 400
      , h_split := by decide
      , h_headDim := by decide
      }
  , ln2 := LayerNormParams.ones 16
  , mlp_fc1 := patternedMat 32 16 500
  , mlp_fc2 := patternedMat 16 32 600
  }

/-- Deterministic byte-level MicroGPT fixture with real `vocab = 256`. -/
def mkByteModel : MicroGPT mkByteConfig :=
  mkPatternedByteModel mkByteConfig (by decide) (by decide)

/-- Short self-source excerpt to keep the byte-model smoke test fast. -/
def microGPTSelfExcerpt : String :=
  "structure Vec (n : Nat) where\n  data : Array Float\n\nstructure Mat (rows cols : Nat) where\n  data : Array (Array Float)"

/-- A small mixed-source byte-level corpus used for fast training smoke tests. -/
def buildSmallByteTrainingTexts : List String :=
  [microGPTSelfExcerpt] ++ curatedMathlibStyleSnippets.take 2 ++ curatedLeanProgramSnippets.take 1

/-- Keep only the first `nTrain` and `nValid` examples from a split. -/
def trimDatasetSplit (nTrain nValid : Nat) (split : DatasetSplit α) : DatasetSplit α :=
  { train := split.train.take nTrain
  , valid := split.valid.take nValid
  }

/-- Exact float-bit equality for two flat parameter lists. -/
def floatBitsEqualList (xs ys : List Float) : Bool :=
  xs.length == ys.length &&
    (List.zipWith (fun x y => Float.toBits x == Float.toBits y) xs ys).foldl (fun ok same => ok && same) true

/-- Quick test: typed reverse-mode gradients and SGD loss decrease. -/
def testLinearClassifierTraining : IO Unit := do
  let model := mkTinyLinearClassifier
  let input : Vec 2 := Vec.ofFn 2 fun i => if i.val = 0 then 1.0 else -1.0
  let target : Fin 3 := ⟨0, by decide⟩

  let reverse := reverseGradientLinearClassifier model input target
  let numeric := finiteDiffLinearClassifier model input target
  let reverseFlat := flattenLinearClassifierGrads reverse
  let numericFlat := flattenLinearClassifierGrads numeric
  let errors := List.zipWith (fun r n => floatAbs (r - n)) reverseFlat numericFlat

  let lossBefore := linearClassifierLoss model input target
  let oneStep := sgdStepLinearClassifier model reverse 0.5
  let lossAfterOne := linearClassifierLoss oneStep input target
  let trained := trainLinearClassifierSteps model input target 0.5 10
  let lossAfterTen := linearClassifierLoss trained input target

  IO.println s!"Linear classifier reverse grads: {reverseFlat}"
  IO.println s!"Linear classifier finite grads:  {numericFlat}"
  IO.println s!"Linear classifier abs errors:   {errors}"
  IO.println s!"Linear classifier loss: before={lossBefore}, after1={lossAfterOne}, after10={lossAfterTen}"
  IO.println "Expected: gradients match finite differences and loss decreases after SGD"

/-- Quick test: frozen-feature output-head training on top of MicroGPT hidden states. -/
def testFrozenFeatureTraining : IO Unit := do
  let model := mkTinyModel
  let tokenIds : List (Fin 3) := [⟨0, by decide⟩, ⟨1, by decide⟩]
  let predictIdx : Fin tokenIds.length := ⟨0, by decide⟩
  let target : Fin 3 := ⟨0, by decide⟩
  let hidden := model.hiddenState tokenIds predictIdx
  let initialHead := zeroLinearClassifier 2 3
  let initialLoss := linearClassifierLoss initialHead hidden target
  let trainedHead := trainFrozenOutputHead model tokenIds predictIdx target 0.5 10
  let finalLoss := linearClassifierLoss trainedHead hidden target
  let trainedLogits := linearClassifierForward trainedHead hidden

  IO.println s!"Frozen hidden state: {hidden.data.toList}"
  IO.println s!"Frozen-head logits after training: {trainedLogits.data.toList}"
  IO.println s!"Frozen-head loss: before={initialLoss}, after10={finalLoss}"
  IO.println "Expected: frozen-feature head training lowers loss on the model-derived example"

/-- Quick test: train the actual model output head (`ln_final + unembed`) on frozen transformer states. -/
def testModelOutputHeadTraining : IO Unit := do
  let model := mkTinyModel
  let tokenIds : List (Fin 3) := [⟨0, by decide⟩, ⟨1, by decide⟩]
  let predictIdx : Fin tokenIds.length := ⟨0, by decide⟩
  let target : Fin 3 := ⟨0, by decide⟩
  let state := model.preFinalState tokenIds predictIdx
  let params := outputHeadFromModel model
  let reverse := reverseGradientOutputHead params state target
  let numeric := finiteDiffOutputHead params state target
  let reverseFlat := flattenOutputHeadGrads reverse
  let numericFlat := flattenOutputHeadGrads numeric
  let errors := List.zipWith (fun r n => floatAbs (r - n)) reverseFlat numericFlat

  let initialLoss := outputHeadLoss params state target
  let trainedModel := trainModelOutputHead model tokenIds predictIdx target 0.1 10
  let finalLoss := crossEntropyLoss (trainedModel.forward tokenIds predictIdx) target

  IO.println s!"Output-head reverse grads: {reverseFlat}"
  IO.println s!"Output-head finite grads:  {numericFlat}"
  IO.println s!"Output-head abs errors:   {errors}"
  IO.println s!"Output-head loss: before={initialLoss}, after10={finalLoss}"
  IO.println "Expected: typed output-head gradients match finite differences and model loss decreases"

/-- Quick test: train a frozen-state MLP block against the model's fixed output head. -/
def testMLPBlockTraining : IO Unit := do
  let model := mkTinyModel
  let tokenIds : List (Fin 3) := [⟨0, by decide⟩, ⟨1, by decide⟩]
  let predictIdx : Fin tokenIds.length := ⟨0, by decide⟩
  let target : Fin 3 := ⟨0, by decide⟩
  let state := model.preFinalState tokenIds predictIdx
  let outputHead := outputHeadFromModel model
  let params := mkTinyMLPBlock
  let reverse := reverseGradientMLPBlock params outputHead state target
  let numeric := finiteDiffMLPBlock params outputHead state target
  let reverseFlat := flattenMLPBlockGrads reverse
  let numericFlat := flattenMLPBlockGrads numeric
  let errors := List.zipWith (fun r n => floatAbs (r - n)) reverseFlat numericFlat

  let initialLoss := mlpBlockLoss params outputHead state target
  let trained := trainMLPBlockSteps params outputHead state target 0.05 10
  let finalLoss := mlpBlockLoss trained outputHead state target
  let trainedState := mlpBlockForward trained state

  IO.println s!"MLP reverse grads: {reverseFlat}"
  IO.println s!"MLP finite grads:  {numericFlat}"
  IO.println s!"MLP abs errors:   {errors}"
  IO.println s!"MLP block state after training: {trainedState.data.toList}"
  IO.println s!"MLP block loss: before={initialLoss}, after10={finalLoss}"
  IO.println "Expected: MLP gradients match finite differences and frozen-state loss decreases"

/-- Quick test: train a frozen-sequence attention block against a fixed output head. -/
def testAttentionBlockTraining : IO Unit := do
  let sequence : TokenSeq 2 2 :=
    TokenSeq.ofFn 2 2 fun i =>
      match i.val with
      | 0 => Vec.ofFn 2 fun j => if j.val = 0 then 2.0 else -1.0
      | _ => Vec.ofFn 2 fun j => if j.val = 0 then 0.25 else 1.5
  let queryIdx : Fin 2 := ⟨1, by decide⟩
  let target : Fin 3 := ⟨0, by decide⟩
  let outputHead := mkSharpOutputHead
  let params := mkTinyAttentionBlock
  let reverse := reverseGradientAttentionBlock params outputHead sequence queryIdx target
  let numeric := finiteDiffAttentionBlock params outputHead sequence queryIdx target
  let reverseFlat := flattenAttentionBlockGrads reverse
  let numericFlat := flattenAttentionBlockGrads numeric
  let errors := List.zipWith (fun r n => floatAbs (r - n)) reverseFlat numericFlat

  let initialLoss := attentionBlockLoss params outputHead sequence queryIdx target
  let trained := trainAttentionBlockSteps params outputHead sequence queryIdx target 100000.0 10
  let finalLoss := attentionBlockLoss trained outputHead sequence queryIdx target
  let trainedState := attentionBlockForward trained sequence queryIdx

  IO.println s!"Attention reverse grads: {reverseFlat}"
  IO.println s!"Attention finite grads:  {numericFlat}"
  IO.println s!"Attention abs errors:   {errors}"
  IO.println s!"Attention block state after training: {trainedState.data.toList}"
  IO.println s!"Attention block loss: before={initialLoss}, after10={finalLoss}"
  IO.println "Expected: attention gradients match finite differences and frozen-sequence loss decreases"

/-- Quick test: train a full frozen-sequence transformer block against a fixed output head. -/
def testTransformerBlockTraining : IO Unit := do
  let sequence : TokenSeq 2 2 :=
    TokenSeq.ofFn 2 2 fun i =>
      match i.val with
      | 0 => Vec.ofFn 2 fun j => if j.val = 0 then 2.0 else -1.0
      | _ => Vec.ofFn 2 fun j => if j.val = 0 then 0.25 else 1.5
  let queryIdx : Fin 2 := ⟨1, by decide⟩
  let target : Fin 3 := ⟨0, by decide⟩
  let outputHead := mkSharpOutputHead
  let params := mkTinyTransformerBlockTrain
  let reverse := reverseGradientTransformerBlockTrain params outputHead sequence queryIdx target
  let numeric := finiteDiffTransformerBlockTrain params outputHead sequence queryIdx target
  let reverseFlat := flattenTransformerBlockTrainGrads reverse
  let numericFlat := flattenTransformerBlockTrainGrads numeric
  let errors := List.zipWith (fun r n => floatAbs (r - n)) reverseFlat numericFlat

  let initialLoss := transformerBlockTrainLoss params outputHead sequence queryIdx target
  let trained := trainTransformerBlockTrainSteps params outputHead sequence queryIdx target 100000.0 10
  let finalLoss := transformerBlockTrainLoss trained outputHead sequence queryIdx target
  let trainedState := transformerBlockTrainForward trained sequence queryIdx

  IO.println s!"Transformer-block reverse grads: {reverseFlat}"
  IO.println s!"Transformer-block finite grads:  {numericFlat}"
  IO.println s!"Transformer-block abs errors:   {errors}"
  IO.println s!"Transformer-block state after training: {trainedState.data.toList}"
  IO.println s!"Transformer-block loss: before={initialLoss}, after10={finalLoss}"
  IO.println "Expected: full-block gradients match finite differences and frozen-sequence loss decreases"

/-- Quick test: train the actual layer inside a single-head MicroGPT model on one token example. -/
def testActualModelLayerTraining : IO Unit := do
  let model := mkTinySingleHeadModel
  let ex : TokenExample 3 :=
    { tokens := [⟨0, by decide⟩, ⟨1, by decide⟩]
    , predictIdx := ⟨1, by decide⟩
    , target := ⟨0, by decide⟩
    }
  let lossBefore := tinySingleHeadModelLoss model ex
  let trained := trainTinySingleHeadModelOnExample model ex 100000.0 10
  let lossAfter := tinySingleHeadModelLoss trained ex
  let logits := trained.forward ex.tokens ex.predictIdx

  IO.println s!"Actual-model logits after training: {logits.data.toList}"
  IO.println s!"Actual-model loss: before={lossBefore}, after10={lossAfter}"
  IO.println "Expected: actual MicroGPT layer training lowers model loss on the token example"

/-- Quick test: generic full multi-head layer gradients match finite differences and SGD lowers loss. -/
def testFullTransformerLayerTraining : IO Unit := do
  let model := mkTinyTwoLayerModel
  let ex : TokenExample 3 :=
    { tokens := [⟨0, by decide⟩, ⟨1, by decide⟩]
    , predictIdx := ⟨1, by decide⟩
    , target := ⟨0, by decide⟩
    }
  let layer := model.getLayer ⟨1, by decide⟩
  let params := fullTransformerLayerTrainOfLayer layer
  let sequence := model.lastLayerInputSequence ex.tokens
  let outputHead := outputHeadFromModel model
  let reverse := reverseGradientFullTransformerLayerTrain params outputHead sequence ex.predictIdx ex.target
  let numeric := finiteDiffFullTransformerLayerTrain params outputHead sequence ex.predictIdx ex.target
  let reverseFlat := flattenFullTransformerLayerTrainGrads reverse
  let numericFlat := flattenFullTransformerLayerTrainGrads numeric
  let errors := List.zipWith (fun r n => floatAbs (r - n)) reverseFlat numericFlat
  let lossBefore := fullTransformerLayerTrainLoss params outputHead sequence ex.predictIdx ex.target
  let trained := trainFullTransformerLayerTrainSteps params outputHead sequence ex.predictIdx ex.target 10000.0 10
  let lossAfter := fullTransformerLayerTrainLoss trained outputHead sequence ex.predictIdx ex.target

  IO.println s!"Full-layer reverse grads: {reverseFlat}"
  IO.println s!"Full-layer finite grads:  {numericFlat}"
  IO.println s!"Full-layer abs errors:   {errors}"
  IO.println s!"Full-layer loss: before={lossBefore}, after10={lossAfter}"
  IO.println "Expected: generic multi-head layer gradients match finite differences and loss decreases"

/-- Quick test: generic actual-model last-layer training works for a non-empty multi-head model. -/
def testGenericLastLayerTraining : IO Unit := do
  let model := mkTinyTwoLayerModel
  let ex : TokenExample 3 :=
    { tokens := [⟨0, by decide⟩, ⟨1, by decide⟩]
    , predictIdx := ⟨1, by decide⟩
    , target := ⟨0, by decide⟩
    }
  let lossBefore := tokenDatasetLoss model [ex]
  let trained := trainModelLastLayerOnExample model (by decide : 0 < 2) ex 10000.0 10
  let lossAfter := tokenDatasetLoss trained [ex]
  let logits := trained.forward ex.tokens ex.predictIdx

  IO.println s!"Generic last-layer logits after training: {logits.data.toList}"
  IO.println s!"Generic last-layer loss: before={lossBefore}, after10={lossAfter}"
  IO.println "Expected: generic last-layer training lowers loss on the actual multi-head model"

/-- Quick test: full-model reverse gradients match finite differences and SGD lowers loss. -/
def testFullModelTraining : IO Unit := do
  let model := mkTinySingleHeadModel
  let ex : TokenExample 3 :=
    { tokens := [⟨0, by decide⟩, ⟨1, by decide⟩]
    , predictIdx := ⟨1, by decide⟩
    , target := ⟨0, by decide⟩
    }
  let reverse := reverseGradientMicroGPT model ex
  let numeric := finiteDiffMicroGPT model ex
  let errors := List.zipWith (fun r n => floatAbs (r - n)) reverse numeric
  let lossBefore := tinySingleHeadModelLoss model ex
  let trained := trainMicroGPTOnExample model ex 0.01 10
  let lossAfter := tinySingleHeadModelLoss trained ex

  IO.println s!"Full-model param count: {microGPTParamCount { d_model := 2, n_heads := 1, d_head := 2, d_ff := 2, n_layers := 1, vocab := 3 }}"
  IO.println s!"Full-model reverse grads: {reverse}"
  IO.println s!"Full-model finite grads:  {numeric}"
  IO.println s!"Full-model abs errors:   {errors}"
  IO.println s!"Full-model loss: before={lossBefore}, after10={lossAfter}"
  IO.println "Expected: full-model gradients match finite differences and loss decreases"

/-- Quick test: full-model Adam lowers loss on the same tiny actual-model example. -/
def testFullModelTrainingAdam : IO Unit := do
  let model := mkTinySingleHeadModel
  let ex : TokenExample 3 :=
    { tokens := [⟨0, by decide⟩, ⟨1, by decide⟩]
    , predictIdx := ⟨1, by decide⟩
    , target := ⟨0, by decide⟩
    }
  let lossBefore := tinySingleHeadModelLoss model ex
  let (trained, state) := trainMicroGPTOnExampleAdam model (initAdamState (flattenMicroGPTParams model).length) ex 0.01 0.9 0.999 1.0e-8 10
  let lossAfter := tinySingleHeadModelLoss trained ex
  IO.println s!"Full-model Adam steps: {state.step}"
  IO.println s!"Full-model Adam loss: before={lossBefore}, after10={lossAfter}"
  IO.println "Expected: Adam lowers full-model loss on the tiny actual-model example"

/-- Quick test: tiny token-dataset epoch loop over an actual MicroGPT model. -/
def testTinyTokenTrainingLoop : IO Unit := do
  let model := mkTinySingleHeadModel
  let lossBefore := tinyTokenDatasetLoss model tinyTokenDataset
  let trained := trainTinyTokenDatasetEpochs model tinyTokenDataset 100000.0 10 3
  let lossAfter := tinyTokenDatasetLoss trained tinyTokenDataset

  IO.println s!"Tiny token dataset mean loss: before={lossBefore}, after3epochs={lossAfter}"
  IO.println "Expected: tiny tokenized training loop lowers average dataset loss"

/-- Quick test: batched train/validation logs for generic last-layer training. -/
def testGenericTrainingLogs : IO Unit := do
  let model := mkTinyTwoLayerModel
  let split := splitDatasetEvery 2 tinyTwoLayerTokenDataset
  let lossBeforeTrain := tokenDatasetLoss model split.train
  let lossBeforeValid := tokenDatasetLoss model split.valid
  let (trained, logs) := trainModelLastLayerEpochsLogged model (by decide : 0 < 2) split 2 100000.0 5 3
  let lossAfterTrain := tokenDatasetLoss trained split.train
  let lossAfterValid := tokenDatasetLoss trained split.valid

  IO.println s!"Generic training split sizes: train={split.train.length}, valid={split.valid.length}"
  IO.println s!"Generic training loss: trainBefore={lossBeforeTrain}, trainAfter={lossAfterTrain}"
  IO.println s!"Generic validation loss: validBefore={lossBeforeValid}, validAfter={lossAfterValid}"
  IO.println s!"Generic epoch logs: {logs.map (fun log => (log.epoch, log.trainLoss, log.validLoss))}"
  IO.println "Expected: batched last-layer training produces loss logs and lowers training loss"

/-- Quick test: full-model batched train/validation logs over a tiny actual-model dataset. -/
def testFullModelTrainingLogs : IO Unit := do
  let model := mkTinyTwoLayerModel
  let split := splitDatasetEvery 2 tinyTwoLayerTokenDataset
  let lossBeforeTrain := tokenDatasetLoss model split.train
  let lossBeforeValid := tokenDatasetLoss model split.valid
  let (trained, logs) := trainMicroGPTEpochsLogged model split 2 0.01 1 3
  let lossAfterTrain := tokenDatasetLoss trained split.train
  let lossAfterValid := tokenDatasetLoss trained split.valid

  IO.println s!"Full-model training split sizes: train={split.train.length}, valid={split.valid.length}"
  IO.println s!"Full-model train loss: before={lossBeforeTrain}, after={lossAfterTrain}"
  IO.println s!"Full-model valid loss: before={lossBeforeValid}, after={lossAfterValid}"
  IO.println s!"Full-model epoch logs: {logs.map (fun log => (log.epoch, log.trainLoss, log.validLoss))}"
  IO.println "Expected: full-model training produces epoch logs and lowers at least the train loss"

/-- Quick test: connect the byte-token pipeline to a real `vocab = 256` model and train its layer. -/
def testByteModelTraining : IO Unit := do
  let texts := buildSmallByteTrainingTexts
  let split := trimDatasetSplit 1 1 (splitByteTokenExamplesFromTexts 8 3 texts)
  let model := mkByteModel
  let exampleCount := split.train.length + split.valid.length
  let firstTarget :=
    match split.train with
    | [] => none
    | ex :: _ => some ex.target.val
  let (lossBeforeTrain, lossAfterTrain, lossBeforeValid, lossAfterValid, logs) :=
    match split.train with
    | [] => (0.0, 0.0, tokenDatasetLoss model split.valid, tokenDatasetLoss model split.valid, ([] : List TrainingEpochLog))
    | ex :: _ =>
        let validBefore := tokenDatasetLoss model split.valid
        let trained := trainModelLastLayerOnExample model (by decide : 0 < 1) ex 0.05 2
        let trainBefore := crossEntropyLoss (model.forward ex.tokens ex.predictIdx) ex.target
        let trainAfter := crossEntropyLoss (trained.forward ex.tokens ex.predictIdx) ex.target
        let validAfter := tokenDatasetLoss trained split.valid
        let logs : List TrainingEpochLog :=
          [ { epoch := 1, trainLoss := trainBefore, validLoss := validBefore }
          , { epoch := 2, trainLoss := trainAfter, validLoss := validAfter }
          ]
        (trainBefore, trainAfter, validBefore, validAfter, logs)

  IO.println s!"Byte-model config params: {microGPTParamCount mkByteConfig}"
  IO.println s!"Byte-model split sizes: train={split.train.length}, valid={split.valid.length}, total={exampleCount}"
  IO.println s!"Byte-model first train target byte: {firstTarget.getD 0}"
  IO.println s!"Byte-model train loss: before={lossBeforeTrain}, after={lossAfterTrain}"
  IO.println s!"Byte-model valid loss: before={lossBeforeValid}, after={lossAfterValid}"
  IO.println s!"Byte-model epoch logs: {logs.map (fun log => (log.epoch, log.trainLoss, log.validLoss))}"
  IO.println "Expected: real byte-token examples train a real vocab-256 model and lower train loss"

/-- Quick test: overfit one real byte-token example until the target byte becomes the argmax. -/
def testByteModelOverfit : IO Unit := do
  let texts := buildSmallByteTrainingTexts
  let split := trimDatasetSplit 1 0 (splitByteTokenExamplesFromTexts 8 3 texts)
  let model := mkByteModel
  match split.train with
  | [] =>
      IO.println "Byte-model overfit skipped: no training example"
  | ex :: _ =>
      let logitsBefore := model.forward ex.tokens ex.predictIdx
      let predBefore := Vec.argmax logitsBefore (by decide : 0 < 256)
      let lossBefore := crossEntropyLoss logitsBefore ex.target
      let trained := trainModelLastLayerOnExample model (by decide : 0 < 1) ex 0.2 8
      let logitsAfter := trained.forward ex.tokens ex.predictIdx
      let predAfter := Vec.argmax logitsAfter (by decide : 0 < 256)
      let lossAfter := crossEntropyLoss logitsAfter ex.target
      IO.println s!"Byte-model overfit target byte: {ex.target.val}"
      IO.println s!"Byte-model overfit pred before: {predBefore.val}"
      IO.println s!"Byte-model overfit pred after:  {predAfter.val}"
      IO.println s!"Byte-model overfit loss: before={lossBefore}, after8={lossAfter}"
      IO.println s!"Byte-model overfit success: {predAfter.val == ex.target.val}"
      IO.println "Expected: one real byte example becomes the argmax after training"

/-- Quick test: binary checkpoint round-trip preserves the exact flat parameter bits. -/
def testCheckpointSerialization : IO Unit := do
  let model := mkTinyTwoLayerModel
  let path : System.FilePath := "/tmp/microgpt-checkpoint.bin"
  let flatBefore := flattenMicroGPTParams model
  saveMicroGPTCheckpoint path model
  let loadedResult ← loadMicroGPTCheckpoint model path
  match loadedResult with
  | Except.error err =>
      IO.println s!"Checkpoint load failed: {err}"
  | Except.ok loaded =>
      let flatAfter := flattenMicroGPTParams loaded
      let bitsEqual := floatBitsEqualList flatBefore flatAfter
      let logitsBefore := model.forward [⟨0, by decide⟩, ⟨1, by decide⟩] ⟨1, by decide⟩
      let logitsAfter := loaded.forward [⟨0, by decide⟩, ⟨1, by decide⟩] ⟨1, by decide⟩
      IO.println s!"Checkpoint param count: {flatBefore.length}"
      IO.println s!"Checkpoint exact bit round-trip: {bitsEqual}"
      IO.println s!"Checkpoint logits before: {logitsBefore.data.toList}"
      IO.println s!"Checkpoint logits after:  {logitsAfter.data.toList}"
      IO.println "Expected: checkpoint save/load preserves exact parameter bits"

/-- A short decoded rendering of byte tokens, falling back to raw byte values if needed. -/
def renderByteTokens (tokens : List (Fin 256)) : String :=
  match decodeUTF8Bytes? tokens with
  | some text => text
  | none => s!"<bytes {tokens.map (fun tok => tok.val)}>"

/-- Per-step loss log for a bounded optimizer experiment. -/
structure TrainingStepLog where
  step : Nat
  trainLoss : Float
  validLoss : Float

/-- Run a bounded full-model Adam loop over byte-token examples, logging every `logEvery` steps. -/
def trainByteModelAdamStepsLogged
    (model : MicroGPT mkByteConfig)
    (state : AdamState)
    (trainExamples : List (TokenExample 256))
    (trainEval validEval : List (TokenExample 256))
    (lr : Float) (logEvery : Nat) :
    MicroGPT mkByteConfig × AdamState × List TrainingStepLog :=
  let rec go
      (current : MicroGPT mkByteConfig)
      (currentState : AdamState)
      (step : Nat)
      (remaining : List (TokenExample 256))
      (logs : List TrainingStepLog) :
      MicroGPT mkByteConfig × AdamState × List TrainingStepLog :=
    match remaining with
    | [] => (current, currentState, logs.reverse)
    | ex :: rest =>
        let (nextModel, nextState) := trainMicroGPTOnExampleAdam current currentState ex lr 0.9 0.999 1.0e-8 1
        let nextStep := step + 1
        let shouldLog := logEvery != 0 && (nextStep % logEvery == 0 || rest.isEmpty)
        let nextLogs :=
          if shouldLog then
            let entry : TrainingStepLog :=
              { step := nextStep
              , trainLoss := tokenDatasetLoss nextModel trainEval
              , validLoss := tokenDatasetLoss nextModel validEval
              }
            entry :: logs
          else
            logs
        go nextModel nextState nextStep rest nextLogs
  go model state 0 trainExamples []

/-- Focused experiment: train the real byte model with Adam on a prefix of the full priority corpus. -/
def runByteModelAdamExperimentCore
    (trainSteps trainEvalCount validEvalCount : Nat)
    (label : String) : IO Unit := do
  let texts ← loadPriorityTrainingTexts
  let (fullTrainCount, fullValidCount) := countSplitByteTokenExamplesFromTexts 8 5 texts
  let split := takeSplitByteTokenExamplesFromTexts 8 5 (Nat.max trainSteps trainEvalCount) validEvalCount texts
  let trainWindow := split.train.take trainSteps
  let trainEval := split.train.take trainEvalCount
  let validEval := split.valid.take validEvalCount
  let model := mkByteModel
  let state := initAdamState (microGPTParamCount mkByteConfig)
  let lossBeforeTrain := tokenDatasetLoss model trainEval
  let lossBeforeValid := tokenDatasetLoss model validEval
  let (trained, finalState, logs) :=
    trainByteModelAdamStepsLogged model state trainWindow trainEval validEval 0.005 10
  let lossAfterTrain := tokenDatasetLoss trained trainEval
  let lossAfterValid := tokenDatasetLoss trained validEval
  let prompts := ["def ", "theorem ", "by "]

  IO.println s!"=== Byte-Model Adam Experiment ({label}) ==="
  IO.println s!"Full corpus texts: {texts.length}"
  IO.println s!"Full corpus example counts(context=8): train={fullTrainCount}, valid={fullValidCount}"
  IO.println s!"Streamed subset sizes: train={split.train.length}, valid={split.valid.length}"
  IO.println s!"Train window steps: {trainWindow.length}"
  IO.println s!"Eval window sizes: train={trainEval.length}, valid={validEval.length}"
  IO.println s!"Byte-model param count: {microGPTParamCount mkByteConfig}"
  IO.println s!"Adam state step: {finalState.step}"
  IO.println s!"Adam train loss: before={lossBeforeTrain}, after={lossAfterTrain}"
  IO.println s!"Adam valid loss: before={lossBeforeValid}, after={lossAfterValid}"
  IO.println s!"Adam step logs: {logs.map (fun log => (log.step, log.trainLoss, log.validLoss))}"
  for prompt in prompts do
    let promptTokens := encodeUTF8Bytes prompt
    let generated := generateArgmax trained (by decide : 0 < 256) promptTokens 8
    let completion := generated.drop promptTokens.length
    IO.println s!"Prompt: {prompt}"
    IO.println s!"Completion: {renderByteTokens completion}"
    IO.println s!"Full sample: {renderByteTokens generated}"
  IO.println "Expected: losses trend down over 100+ Adam steps and samples start to resemble byte-level Lean syntax"

/-- Full staged experiment configuration. -/
def runByteModelAdamExperiment : IO Unit :=
  runByteModelAdamExperimentCore 120 32 32 "full"

/-- Cheaper native-only probe to finish quickly enough for inspection. -/
def runByteModelAdamQuickExperiment : IO Unit :=
  runByteModelAdamExperimentCore 1 1 1 "quick"

/-- Tiny train-only timing probe for estimating byte-model Adam throughput on CPU. -/
def runByteModelAdamTimingProbe : IO Unit := do
  let texts ← loadPriorityTrainingTexts
  let (fullTrainCount, fullValidCount) := countSplitByteTokenExamplesFromTexts 8 5 texts
  let split := takeSplitByteTokenExamplesFromTexts 8 5 2 1 texts
  let model := mkByteModel
  let state := initAdamState (microGPTParamCount mkByteConfig)
  let lossBeforeTrain := tokenDatasetLoss model split.train
  let lossBeforeValid := tokenDatasetLoss model split.valid
  let (trained, finalState, logs) :=
    trainByteModelAdamStepsLogged model state split.train split.train split.valid 0.005 1
  let lossAfterTrain := tokenDatasetLoss trained split.train
  let lossAfterValid := tokenDatasetLoss trained split.valid
  IO.println "=== Byte-Model Adam Timing Probe ==="
  IO.println s!"Full corpus example counts(context=8): train={fullTrainCount}, valid={fullValidCount}"
  IO.println s!"Train examples: {split.train.length}"
  IO.println s!"Valid examples: {split.valid.length}"
  IO.println s!"Adam state step: {finalState.step}"
  IO.println s!"Train loss: before={lossBeforeTrain}, after={lossAfterTrain}"
  IO.println s!"Valid loss: before={lossBeforeValid}, after={lossAfterValid}"
  IO.println s!"Step logs: {logs.map (fun log => (log.step, log.trainLoss, log.validLoss))}"

/-- Fast counter for the full byte-token split used by the native experiment. -/
def runByteTokenCountProbe : IO Unit := do
  let texts ← loadPriorityTrainingTexts
  let (trainCount, validCount) := countSplitByteTokenExamplesFromTexts 8 5 texts
  IO.println "=== Byte-Token Count Probe ==="
  IO.println s!"Priority corpus texts: {texts.length}"
  IO.println s!"Full corpus example counts(context=8): train={trainCount}, valid={validCount}, total={trainCount + validCount}"

/-- Parse a natural number CLI argument. -/
def parseNatArg (label value : String) : IO Nat :=
  match value.toNat? with
  | some n => pure n
  | none => throw <| IO.userError s!"invalid {label}: {value}"

/-- Parse a comma-separated token list. -/
def parseTokenCsv (value : String) : IO (List Nat) := do
  if value.isEmpty then
    pure []
  else
    (value.splitOn ",").mapM fun token => parseNatArg "token" token.trimAscii.toString

/-- Parse a `GPTConfig` from CLI arguments. -/
def parseGPTConfigArgs
    (dModelValue nHeadsValue dHeadValue dFfValue nLayersValue vocabValue : String) : IO GPTConfig := do
  let dModel ← parseNatArg "d_model" dModelValue
  let nHeads ← parseNatArg "n_heads" nHeadsValue
  let dHead ← parseNatArg "d_head" dHeadValue
  let dFf ← parseNatArg "d_ff" dFfValue
  let nLayers ← parseNatArg "n_layers" nLayersValue
  let vocab ← parseNatArg "vocab" vocabValue
  pure
    { d_model := dModel
    , n_heads := nHeads
    , d_head := dHead
    , d_ff := dFf
    , n_layers := nLayers
    , vocab := vocab
    }

/-- Convert token naturals into bounded vocab indices. -/
def finListOfNatList (vocab : Nat) (values : List Nat) : IO (List (Fin vocab)) :=
  values.mapM fun n =>
    if h : n < vocab then
      pure ⟨n, h⟩
    else
      throw <| IO.userError s!"token {n} out of range for vocab {vocab}"

/-- Common parity output format for Python-side parsing. -/
def printParityEvalResult {vocab : Nat} (logits : Vec vocab) (loss : Float) : IO Unit := do
  IO.println s!"logits={logits.data.toList}"
  IO.println s!"loss={loss}"

/-- Save one of the built-in fixture models as a Lean-compatible flat checkpoint. -/
def saveParityFixture (fixture : String) (path : System.FilePath) : IO Unit := do
  match fixture with
  | "tiny-single-head" => saveMicroGPTCheckpoint path mkTinySingleHeadModel
  | "tiny-two-layer" => saveMicroGPTCheckpoint path mkTinyTwoLayerModel
  | "byte" => saveMicroGPTCheckpoint path mkByteModel
  | _ => throw <| IO.userError s!"unknown fixture: {fixture}"

/-- Save a patterned byte-model template for an arbitrary valid config. -/
def saveParityConfig (cfg : GPTConfig) (path : System.FilePath) : IO Unit := do
  if h_split : cfg.d_model = cfg.n_heads * cfg.d_head then
    if h_headDim : 0 < cfg.d_head then
      saveMicroGPTCheckpoint path (mkPatternedByteModel cfg h_split h_headDim)
    else
      throw <| IO.userError s!"invalid config: d_head must be positive, got {cfg.d_head}"
  else
    throw <| IO.userError
      s!"invalid config: expected d_model = n_heads * d_head, got {cfg.d_model} != {cfg.n_heads * cfg.d_head}"

/-- Print a built-in default parity example for one fixture. -/
def printParityDefaultCase (fixture : String) : IO Unit := do
  match fixture with
  | "tiny-single-head" =>
      let tokens : List (Fin 3) := [⟨0, by decide⟩, ⟨1, by decide⟩]
      IO.println s!"tokens={tokens.map (fun tok => tok.val)}"
      IO.println "predict_idx=1"
      IO.println "target=0"
  | "tiny-two-layer" =>
      let tokens : List (Fin 3) := [⟨0, by decide⟩, ⟨1, by decide⟩]
      IO.println s!"tokens={tokens.map (fun tok => tok.val)}"
      IO.println "predict_idx=1"
      IO.println "target=0"
  | "byte" =>
      let texts := buildSmallByteTrainingTexts
      let split := trimDatasetSplit 1 1 (splitByteTokenExamplesFromTexts 8 3 texts)
      match split.train with
      | ex :: _ =>
          IO.println s!"tokens={ex.tokens.map (fun tok => tok.val)}"
          IO.println s!"predict_idx={ex.predictIdx.val}"
          IO.println s!"target={ex.target.val}"
      | [] =>
          throw <| IO.userError "byte fixture example unavailable"
  | _ => throw <| IO.userError s!"unknown fixture: {fixture}"

/-- Evaluate one fixture model, optionally after loading an external checkpoint. -/
def evalParityFixture (fixture checkpointValue tokensCsv predictValue targetValue : String) : IO Unit := do
  let checkpoint :=
    if checkpointValue = "-" then
      none
    else
      some (System.FilePath.mk checkpointValue)
  let tokenValues ← parseTokenCsv tokensCsv
  let predictIdxNat ← parseNatArg "predictIdx" predictValue
  let targetNat ← parseNatArg "target" targetValue
  match fixture with
  | "tiny-single-head" =>
      let template := mkTinySingleHeadModel
      let model ←
        match checkpoint with
        | none => pure template
        | some path =>
            match (← loadMicroGPTCheckpoint template path) with
            | Except.ok loaded => pure loaded
            | Except.error err => throw <| IO.userError err
      let tokens ← finListOfNatList 3 tokenValues
      if tokens.isEmpty then
        throw <| IO.userError "token sequence must be non-empty"
      else if h_pred : predictIdxNat < tokens.length then
        if h_target : targetNat < 3 then
          let idx : Fin tokens.length := ⟨predictIdxNat, h_pred⟩
          let target : Fin 3 := ⟨targetNat, h_target⟩
          let logits := model.forward tokens idx
          printParityEvalResult logits (crossEntropyLoss logits target)
        else
          throw <| IO.userError s!"target {targetNat} out of range for vocab 3"
      else
        throw <| IO.userError s!"predict index {predictIdxNat} out of range for length {tokens.length}"
  | "tiny-two-layer" =>
      let template := mkTinyTwoLayerModel
      let model ←
        match checkpoint with
        | none => pure template
        | some path =>
            match (← loadMicroGPTCheckpoint template path) with
            | Except.ok loaded => pure loaded
            | Except.error err => throw <| IO.userError err
      let tokens ← finListOfNatList 3 tokenValues
      if tokens.isEmpty then
        throw <| IO.userError "token sequence must be non-empty"
      else if h_pred : predictIdxNat < tokens.length then
        if h_target : targetNat < 3 then
          let idx : Fin tokens.length := ⟨predictIdxNat, h_pred⟩
          let target : Fin 3 := ⟨targetNat, h_target⟩
          let logits := model.forward tokens idx
          printParityEvalResult logits (crossEntropyLoss logits target)
        else
          throw <| IO.userError s!"target {targetNat} out of range for vocab 3"
      else
        throw <| IO.userError s!"predict index {predictIdxNat} out of range for length {tokens.length}"
  | "byte" =>
      let template := mkByteModel
      let model ←
        match checkpoint with
        | none => pure template
        | some path =>
            match (← loadMicroGPTCheckpoint template path) with
            | Except.ok loaded => pure loaded
            | Except.error err => throw <| IO.userError err
      let tokens ← finListOfNatList 256 tokenValues
      if tokens.isEmpty then
        throw <| IO.userError "token sequence must be non-empty"
      else if h_pred : predictIdxNat < tokens.length then
        if h_target : targetNat < 256 then
          let idx : Fin tokens.length := ⟨predictIdxNat, h_pred⟩
          let target : Fin 256 := ⟨targetNat, h_target⟩
          let logits := model.forward tokens idx
          printParityEvalResult logits (crossEntropyLoss logits target)
        else
          throw <| IO.userError s!"target {targetNat} out of range for vocab 256"
      else
        throw <| IO.userError s!"predict index {predictIdxNat} out of range for length {tokens.length}"
  | _ =>
      throw <| IO.userError s!"unknown fixture: {fixture}"

/-- Evaluate an arbitrary byte-model config, optionally after loading an external checkpoint. -/
def evalParityConfig (cfg : GPTConfig)
    (checkpointValue tokensCsv predictValue targetValue : String) : IO Unit := do
  let checkpoint :=
    if checkpointValue = "-" then
      none
    else
      some (System.FilePath.mk checkpointValue)
  let tokenValues ← parseTokenCsv tokensCsv
  let predictIdxNat ← parseNatArg "predictIdx" predictValue
  let targetNat ← parseNatArg "target" targetValue
  if h_split : cfg.d_model = cfg.n_heads * cfg.d_head then
    if h_headDim : 0 < cfg.d_head then
      let template := mkPatternedByteModel cfg h_split h_headDim
      let model ←
        match checkpoint with
        | none => pure template
        | some path =>
            match (← loadMicroGPTCheckpoint template path) with
            | Except.ok loaded => pure loaded
            | Except.error err => throw <| IO.userError err
      let tokens ← finListOfNatList cfg.vocab tokenValues
      if tokens.isEmpty then
        throw <| IO.userError "token sequence must be non-empty"
      else if h_pred : predictIdxNat < tokens.length then
        if h_target : targetNat < cfg.vocab then
          let idx : Fin tokens.length := ⟨predictIdxNat, h_pred⟩
          let target : Fin cfg.vocab := ⟨targetNat, h_target⟩
          let logits := model.forward tokens idx
          printParityEvalResult logits (crossEntropyLoss logits target)
        else
          throw <| IO.userError s!"target {targetNat} out of range for vocab {cfg.vocab}"
      else
        throw <| IO.userError s!"predict index {predictIdxNat} out of range for length {tokens.length}"
    else
      throw <| IO.userError s!"invalid config: d_head must be positive, got {cfg.d_head}"
  else
    throw <| IO.userError
      s!"invalid config: expected d_model = n_heads * d_head, got {cfg.d_model} != {cfg.n_heads * cfg.d_head}"

/-- Quick test: prioritized byte-token corpus and next-token example builder. -/
def testByteTokenDataPipeline : IO Unit := do
  let texts ← loadPriorityTrainingTexts
  let examples := byteTokenExamplesFromTexts 16 texts
  let split := splitByteTokenExamplesFromTexts 16 5 texts
  let batches := batchesOf 64 split.train
  IO.println s!"Priority corpus texts: {texts.length}"
  IO.println s!"Self-source repeats: {selfSourceRepeatCount}"
  IO.println s!"Mathlib-style snippets: {curatedMathlibStyleSnippets.length}"
  IO.println s!"Lean program snippets: {curatedLeanProgramSnippets.length}"
  IO.println s!"Byte-token examples(context=16): {examples.length}"
  IO.println s!"Byte-token split sizes: train={split.train.length}, valid={split.valid.length}"
  IO.println s!"Byte-token batch count(batch=64): {batches.length}"
  match examples with
  | [] =>
      IO.println "No byte-token examples generated"
  | ex :: _ =>
      let prefixBytes := ex.tokens.map (fun tok => tok.val)
      IO.println s!"First example prefix bytes: {prefixBytes}"
      IO.println s!"First example target byte: {ex.target.val}"
      IO.println s!"First example decoded prefix: {(decodeUTF8Bytes? ex.tokens).getD "<non-utf8-prefix>"}"
  IO.println "Expected: prioritized corpus loads and yields non-empty byte-token next-token examples"

/-- Quick test: greedy generation loop on the tiny model. -/
def testGeneration : IO Unit := do
  let model := mkTinyModel
  let prompt : List (Fin 3) := [⟨0, by decide⟩, ⟨1, by decide⟩]
  let generated := generateArgmax model (by decide : 0 < 3) prompt 2
  IO.println s!"Generated token ids: {generated.map (fun tok => tok.val)}"
  IO.println "Expected greedy continuation: [0, 1, 0, 0]"

/-- Quick test: temperature scaling and top-k filtering. -/
def testSampling : IO Unit := do
  let logits : Vec 3 := Vec.ofFn 3 fun i =>
    match i.val with
    | 0 => 1.0
    | 1 => 0.0
    | _ => -1.0
  let tempProbs := temperatureProbs logits 0.5
  let top2 := topKProbs (softmax logits) 2
  IO.println s!"Temperature probs (temp=0.5): {tempProbs.data.toList}"
  IO.println s!"Top-2 probs: {top2.map (fun entry => (entry.1.val, entry.2))}"
  IO.println "Expected temp=0.5 probs approximately: [0.8668, 0.1173, 0.0159]"

  let model := mkTinyModel
  let prompt : List (Fin 3) := [⟨0, by decide⟩, ⟨1, by decide⟩]
  let sampled ← generateSampled model (by decide : 0 < 3) prompt 2 1.0 (some 1)
  IO.println s!"Top-k=1 sampled ids: {sampled.map (fun tok => tok.val)}"
  IO.println "Expected deterministic top-k=1 continuation: [0, 1, 0, 0]"

def runSmokeSuite : IO Unit := do
  IO.println "=== MicroGPT Lean 4 — Smoke Tests ==="
  IO.println ""
  IO.println "--- Vector Operations ---"
  testVecOps
  IO.println ""
  IO.println "--- Forward-Mode AD ---"
  testAutodiff
  IO.println ""
  IO.println "--- Matrix-Vector Multiply ---"
  testMatVec
  IO.println ""
  IO.println "--- Attention ---"
  testAttention
  IO.println ""
  IO.println "--- Multi-Head Attention ---"
  testMultiHeadAttention
  IO.println ""
  IO.println "--- Positional Encoding ---"
  testPositionalEncoding
  IO.println ""
  IO.println "--- UTF-8 Tokenizer ---"
  testTokenizer
  IO.println ""
  IO.println "--- Full Model Forward ---"
  testModelForward
  IO.println ""
  IO.println "--- Typed Training Scaffold ---"
  testLinearClassifierTraining
  IO.println ""
  IO.println "--- Frozen-Feature Head Training ---"
  testFrozenFeatureTraining
  IO.println ""
  IO.println "--- Model Output-Head Training ---"
  testModelOutputHeadTraining
  IO.println ""
  IO.println "--- MLP Block Training ---"
  testMLPBlockTraining
  IO.println ""
  IO.println "--- Attention Block Training ---"
  testAttentionBlockTraining
  IO.println ""
  IO.println "--- Full Transformer Block Training ---"
  testTransformerBlockTraining
  IO.println ""
  IO.println "--- Actual Model Layer Training ---"
  testActualModelLayerTraining
  IO.println ""
  IO.println "--- Generic Full-Layer Training ---"
  testFullTransformerLayerTraining
  IO.println ""
  IO.println "--- Generic Last-Layer Training ---"
  testGenericLastLayerTraining
  IO.println ""
  IO.println "--- Full-Model Training ---"
  testFullModelTraining
  IO.println ""
  IO.println "--- Full-Model Adam Training ---"
  testFullModelTrainingAdam
  IO.println ""
  IO.println "--- Tiny Token Training Loop ---"
  testTinyTokenTrainingLoop
  IO.println ""
  IO.println "--- Generic Training Logs ---"
  testGenericTrainingLogs
  IO.println ""
  IO.println "--- Full-Model Training Logs ---"
  testFullModelTrainingLogs
  IO.println ""
  IO.println "--- Byte-Model Training ---"
  testByteModelTraining
  IO.println ""
  IO.println "--- Checkpoint Serialization ---"
  testCheckpointSerialization
  IO.println ""
  IO.println "--- Byte-Token Data Pipeline ---"
  testByteTokenDataPipeline
  IO.println ""
  IO.println "--- Greedy Generation ---"
  testGeneration
  IO.println ""
  IO.println "--- Sampling ---"
  testSampling
  IO.println ""
  IO.println "=== All core definitions loaded ==="
  IO.println "=== Shape safety is enforced at compile time ==="
  IO.println ""
  IO.println "Next steps:"
  IO.println "  1. Train larger generic models with Adam instead of plain SGD"
  IO.println "  2. Scale the byte-token run beyond the tiny smoke dataset"
  IO.println "  3. Add checkpoint metadata/versioning and resume-from-checkpoint training"
  IO.println "  4. Learned positional embeddings or KV-cache"
  IO.println "  5. Prove core tensor and AD invariants"
  IO.println "SMOKE_STATUS=PASS"

def main (args : List String) : IO Unit := do
  match args with
  | ["experiment-byte-adam"] => runByteModelAdamExperiment
  | ["experiment-byte-adam-quick"] => runByteModelAdamQuickExperiment
  | ["experiment-byte-adam-timing"] => runByteModelAdamTimingProbe
  | ["experiment-byte-counts"] => runByteTokenCountProbe
  | ["parity-save", fixture, path] => saveParityFixture fixture (System.FilePath.mk path)
  | ["parity-save-config", dModel, nHeads, dHead, dFf, nLayers, vocab, path] => do
      let cfg ← parseGPTConfigArgs dModel nHeads dHead dFf nLayers vocab
      saveParityConfig cfg (System.FilePath.mk path)
  | ["parity-eval", fixture, checkpoint, tokensCsv, predictIdx, target] =>
      evalParityFixture fixture checkpoint tokensCsv predictIdx target
  | ["parity-eval-config", dModel, nHeads, dHead, dFf, nLayers, vocab, checkpoint, tokensCsv, predictIdx, target] => do
      let cfg ← parseGPTConfigArgs dModel nHeads dHead dFf nLayers vocab
      evalParityConfig cfg checkpoint tokensCsv predictIdx target
  | ["parity-default-case", fixture] => printParityDefaultCase fixture
  | _ => runSmokeSuite
