(* ::Package:: *)

(* RunFromPythonTest_fixed2.wl
   Robust generator of a python lambda from a Mathematica expression.
   - Clears globals
   - Forces string conversions
   - Uses Module to avoid name collisions
*)

(* remove any previous definitions that might conflict *)
Remove["Global`*"];

Module[
  {
    (* internal names *)
    integrand, intExpr, simpl,
    inputformStr, pyBody, pyBody2, pyLambdaStr,
    replacements
  },

  (* 1) define and integrate *)
  integrand[x_, y_] := x^2 + Cos[2 y];
  intExpr = Integrate[integrand[x, y], {x, 0, 5}];
  simpl = FullSimplify[intExpr];

  (* 2) produce input-form string (guaranteed string) *)
  inputformStr = ToString[simpl, InputForm];  (* e.g. "125/3 + 5*Cos[2*y]" *)

  (* 3) replacements from Mathematica notation to Python math functions *)
  replacements = {
    "Cos[" -> "math.cos(",
    "Sin[" -> "math.sin(",
    "Tan[" -> "math.tan(",
    "Exp[" -> "math.exp(",
    "Log[" -> "math.log(",
    "Sqrt[" -> "math.sqrt(",
    "Pi" -> "math.pi"
  };

  (* 4) convert the InputForm string using StringReplace (operates on strings) *)
  pyBody = StringReplace[inputformStr, replacements];

  (* 5) fix bracket and power notation in the string *)
  pyBody2 = StringReplace[pyBody, {"]" -> ")", "[" -> "(", "^" -> "**"}];

  (* 6) final python lambda string *)
  pyLambdaStr = "lambda y: " <> ToString[pyBody2, InputForm];

  (* 7) Debugging prints (one well-formatted PY_LAMBDA line) *)
  Print["RESULT_CFORM=" <> ToString[CForm[simpl], InputForm]];
  Print["PY_LAMBDA_HEAD=" <> ToString[Head[pyLambdaStr], InputForm]];
  Print["PY_LAMBDA=" <> pyLambdaStr];
  Print["RESULT_INPUTFORM=" <> inputformStr];

] (* end Module *)

