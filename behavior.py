import sys
import time

import sympy as sp
from sympy import Q

MAX_SIMPLIFICATION_ITERATIONS = 10

# x,y,z,w are interchangeable with any of tp, fp, fn, tn

x = sp.Symbol('x', integer=True, nonnegative=True)
y = sp.Symbol('y', integer=True, nonnegative=True)
z = sp.Symbol('z', integer=True, nonnegative=True)

n = sp.Symbol('n', integer=True, positive=True)

# marginal_benefit is easier to solve with w as a function of n,x,y,z
w = n - x - y - z
assumptions = Q.nonnegative(w)

tp, fp, fn, tn = x, y, z, w
print(f"Symbols: tp={tp}, fp={fp}, fn={fn}, tn={tn}, n={n}.")
predicted_positive = tp + fp
predicted_negative = fn + tn
actual_positive = tp + fn
actual_negative = fp + tn


def expr_info(expression, name, verbose=False):
    """
    Prints information about an expression.
    Not needed for any simplification purposes, just for debugging.
    """
    iteration_str = f"{name}: "
    if not verbose:
        iteration_str = "  " + iteration_str
    print(f"{iteration_str}", end='\n' if verbose else '')
    print(f"  Expression: {expression}")
    if verbose:
        symbol_atoms = [atom for atom in expression.atoms() if atom.is_symbol]
        # Our goal is to express in terms of n
        print(f"  Symbol Atoms: {symbol_atoms}")
        print(f"  Count of Basic Ops: {expression.count_ops()}")
        print(f"  Length of srepr: {len(sp.srepr(expression))}")
        if assumptions.func in (sp.And, sp.Or):
            # If assumptions are compound, break them down and check each one
            individual_assumptions = [arg for arg in assumptions.args]
        else:
            # If it's a singular assumption, just check it directly
            individual_assumptions = [assumptions]
            if not all(sp.ask(assumption) for assumption in individual_assumptions):
                print("  Warning: Assumptions not satisfied.", file=sys.stderr)


def simplest(expr, assumptions, verbose=False, max_iterations=MAX_SIMPLIFICATION_ITERATIONS):
    """
    Simplifies an expression as much as possible by looping until it stops changing,
     or until it reaches the maximum number of iterations.
    """
    iter = 0
    expr_info(expr, "Initial Expression", verbose=verbose)
    for iter in range(1, max_iterations + 1):
        if expr.has(sp.Sum) or expr.has(sp.Product):
            expr = sp.simplify(expr, assumptions=assumptions)
            expr_info(expr, "After Sum or Product Simplification", verbose=verbose)
            expr = expr.doit()
            if verbose:
                expr_info(expr, "After Sum or Product Evaluation")

        expr2 = sp.simplify(expr.doit(), assumptions=assumptions, )
        if expr2 == expr:
            break
        expr_info(expr2, f"Iteration {iter}", verbose=verbose)
        expr = expr2
    if iter == max_iterations:
        print(f"Warning: Maximum number of iterations reached.", file=sys.stderr)
    if verbose and expr.has(sp.Sum):
        print("Warning: expression still has a sum.", file=sys.stderr)
    return expr


def over_all_sums(expr, assumptions, title, verbose=False):
    """
    Takes the expression (metric) and simplifies it with the three summations.
    Recall these three summations are to iterate over all possible values of x,y,z,w.
    """
    print(f"{title}: Simplifying with the two innermost summations.")
    dim2_sum = sp.Sum(expr, (fn, 0, n - tp - fp))
    dim1_sum = sp.Sum(dim2_sum, (fp, 0, n - tp))
    dim1_sum = simplest(dim1_sum, assumptions, verbose=verbose)
    print(f"{title}: Simplifying with the outermost sum.")
    dim0_sum = sp.Sum(dim1_sum, (tp, 0, n))
    dim0_sum = simplest(dim0_sum, assumptions, verbose=verbose)
    return dim0_sum


def calculate_variance_function(expr, mean: float = 0, additional_assumptions=None, title="Custom", verbose=False):
    """
    Calculates the variance of a function
    WARNING: Not guaranteed to find an expression that is computable in constant time.
    """
    global assumptions
    if additional_assumptions is not None:
        assumptions = sp.And(assumptions, additional_assumptions)

    title += " Metric"
    print(f"\n{title}: Simplifying the expression within all sums.")
    expr = (expr - mean) ** 2
    expr = simplest(expr, assumptions, verbose=verbose)

    all_sums = over_all_sums(expr, assumptions, title, verbose=verbose)

    # Note that I've tried to simplify summations separately. I don't recall any benefits.
    print(f"{title}: Simplifying Variance.")
    variance = all_sums / ((n + 1) * (n + 2) * (n + 3) / 6)
    variance = simplest(variance, assumptions, verbose=verbose)
    print(f"Variance for {title}:", variance)

    variance_limit = sp.limit(variance, n, sp.oo)
    print(f"Variance limit for {title}:", variance_limit)
    return variance, variance_limit


def calculate_mean_function(expr, additional_assumptions=None, title="Custom", verbose=False):
    """
    Calculates the mean of a function
    WARNING: Not guaranteed to find an expression that is computable in constant time.
    """
    global assumptions
    if additional_assumptions is not None:
        assumptions = sp.And(assumptions, additional_assumptions)

    title += " Metric"
    print(f"\n{title}: Simplifying the expression within all sums.")
    expr = simplest(expr, assumptions, verbose=verbose)

    all_sums = over_all_sums(expr, assumptions, title, verbose=verbose)

    print(f"{title}: Simplifying Mean.")
    mean = all_sums / ((n + 1) * (n + 2) * (n + 3) / 6)
    mean = simplest(mean, assumptions, verbose=verbose)
    print(f"Mean for {title}:", mean)

    mean_limit = sp.limit(mean, n, sp.oo)
    print(f"Mean limit for {title}:", mean_limit)
    return mean, mean_limit


if __name__ == "__main__":
    marginal_benefit = (fp - fn) / n
    marginal_benefit_mean, marginal_benefit_mean_limit = calculate_mean_function(
        marginal_benefit, title="Marginal Benefit"
    )
    assert marginal_benefit_mean == 0.
    marginal_benefit_variance, marginal_benefit_variance_limit = calculate_variance_function(
        marginal_benefit,
        mean=marginal_benefit_mean,
        title="Marginal Benefit"
    )

    def print_summary(title: str, variance_expr, variance_limit, mean_expr, mean_limit):
        print(f"{title}:")
        print(f"  Mean: {mean_expr}")
        print(f"  Mean Limit: {mean_limit} = {mean_limit.evalf():.6f}")
        print(f"  Variance: {variance_expr}")
        print(f"  Variance Limit: {variance_limit} = {variance_limit.evalf():.6f}")
        std_limit = sp.sqrt(variance_limit)
        print(f"  Standard Deviation Limit: {std_limit.evalf():.6f}")


    print("=====================================================================")
    print("Summary of Variances:")
    # print(f"fp = {fp}, fn = {fn}, tp = {tp}, tn = {tn}, n = {n}")
    print_summary(
        "Marginal Benefit",
        marginal_benefit_variance,
        marginal_benefit_variance_limit,
        mean_expr=marginal_benefit_mean,
        mean_limit=marginal_benefit_mean_limit
    )
