import os
import sympy as sp


def transform_expression(equation):
    # Replace ** with ^ for easier parsing with sympy
    modified_eq = equation.replace('**', '^')

    # Parse the equation to a sympy expression
    expr = sp.sympify(modified_eq)

    # Recursive function to replace exponents with pow
    def replace_pow(expr):
        if expr.is_Pow:
            base, exponent = expr.args
            return sp.Function('pow')(replace_pow(base), replace_pow(exponent))
        elif expr.is_Function:
            return expr.func(*[replace_pow(arg) for arg in expr.args])
        elif expr.is_Add or expr.is_Mul:
            return expr.func(*[replace_pow(arg) for arg in expr.args])
        else:
            return expr

    transformed_expr = replace_pow(expr)

    return str(transformed_expr)

import os

def generate_arduino_function_new(formula_strings, input_size, output_file):

    processed = []

    for formula in formula_strings:

        formula = transform_expression(formula)

        formula = formula.replace('sin', 'std::sin')
        formula = formula.replace('cos', 'std::cos')
        formula = formula.replace('tan', 'std::tan')
        formula = formula.replace('sqrt', 'std::sqrt')
        formula = formula.replace('exp', 'std::exp')
        formula = formula.replace('log', 'std::log')
        formula = formula.replace('tanh', 'std::tanh')
        formula = formula.replace('abs', 'std::abs')
        formula = formula.replace('pow', 'std::pow')

        processed.append(formula)

    num_outputs = len(processed)

    header = """
#ifndef KAN_MODEL_H
#define KAN_MODEL_H

#include <cmath>

"""

    # ==============================
    # gerar funções individuais
    # ==============================

    for i, formula in enumerate(processed):

        header += f"\ninline float score_{i}("

        for j in range(1, input_size + 1):
            if j == input_size:
                header += f"float x_{j})"
            else:
                header += f"float x_{j}, "

        header += "{\n"
        header += f"    return ({formula});\n"
        header += "}\n"

    # ==============================
    # função predict
    # ==============================

    header += "\nint predict("

    for j in range(1, input_size + 1):
        if j == input_size:
            header += f"float x_{j})"
        else:
            header += f"float x_{j}, "

    header += "\n{\n"

    for i in range(num_outputs):
        header += f"    float s{i} = score_{i}("

        for j in range(1, input_size + 1):
            if j == input_size:
                header += f"x_{j});\n"
            else:
                header += f"x_{j}, "

    header += """
    float max_score = s0;
    int max_index = 0;
"""

    for i in range(1, num_outputs):
        header += f"""
    if(s{i} > max_score) {{
        max_score = s{i};
        max_index = {i};
    }}
"""

    header += """
    return max_index;
}

#endif
"""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        f.write(header)

    return header