import re
from sympy import simplify, sympify, Eq, symbols
from sympy.parsing.latex import parse_latex
from sympy.core.relational import Relational
from format_converters import MathFormatConverter

x = symbols('x')  # default symbol
converter = MathFormatConverter()

def remove_boxed(expr: str) -> str:
    """Remove all occurrences of \boxed{...} with proper handling of nested braces and outer delimiters."""
    # First, handle outer LaTeX delimiters like \[ \] or $ $
    expr = re.sub(r'^\\\[(.+?)\\\]$', r'\1', expr)
    expr = re.sub(r'^\$(.+?)\$$', r'\1', expr)
    
    # Then handle boxed command
    out = ""
    i = 0
    while i < len(expr):
        if expr[i:].startswith("\\boxed{"):
            j = i + len("\\boxed{")
            stack = 1
            start_inner = j
            while j < len(expr) and stack > 0:
                if expr[j] == '{':
                    stack += 1
                elif expr[j] == '}':
                    stack -= 1
                j += 1
            # Append the inner content (exclude the matching closing brace)
            out += expr[start_inner:j-1]
            i = j
        else:
            out += expr[i]
            i += 1
    return out

def normalize_units(unit_str: str) -> str:
    """Normalize unit string to a standard form for comparison"""
    if not unit_str:
        return ""
    
    # Remove spaces in common compound units
    unit_str = re.sub(r'(\w+)/(\w+)', r'\1/\2', unit_str)
    
    # Common unit equivalence mappings
    equivalences = {
        # Basic metric prefixes 
        "km": "1000 m",
        "cm": "0.01 m",
        "mm": "0.001 m",
        "g": "0.001 kg",
        "mg": "0.000001 kg",
        
        # Common derived units
        "N": "kg m/s^2",
        "J": "kg m^2/s^2",
        "W": "kg m^2/s^3",
        "Pa": "kg/(m s^2)",
        
        # Volume units
        "L": "0.001 m^3",
        "mL": "0.000001 m^3",
        
        # Time units
        "min": "60 s",
        "h": "3600 s",
        "hr": "3600 s",
        "hour": "3600 s",
        "hours": "3600 s",
    }
    
    # Handle exponents in different formats 
    # Convert m^2 -> m²
    unit_str = re.sub(r'(\w+)\^(-?\d+)', r'\1\2', unit_str)
    # Convert m² -> m^2 for standardization
    unit_str = unit_str.replace('²', '^2').replace('³', '^3')
    
    # Handle divisions like kg/m^2 vs kg m^-2
    unit_str = re.sub(r'(\w+)/(\w+)\^(\d+)', r'\1 \2^-\3', unit_str)
    
    # Simplistic normalization for now - can be expanded later
    # with a proper unit conversion library if needed
    return unit_str

def extract_units(expr: str) -> tuple:
    """Extract units from a mathematical expression and return (expression_without_units, units)"""
    # Common LaTeX unit patterns
    unit_patterns = [
        (r'\\mathrm\{([^}]+)\}', r'\1'),
        (r'\\text\{([^}]+)\}', r'\1'),
        (r'\\,\\mathrm\{([^}]+)\}', r'\1'),
        (r'\\,\\text\{([^}]+)\}', r'\1')
    ]
    
    # Store the extracted units
    units = []
    
    # Extract all units from the expression
    for pattern, replacement in unit_patterns:
        matches = re.findall(pattern, expr)
        units.extend(matches)
        # Remove the unit markers but preserve the unit text for later comparison
        expr = re.sub(pattern, '', expr)
    
    # Join all units into a standardized string
    unit_str = ' '.join(units).strip()
    
    # Normalize the unit string for comparison
    normalized_unit_str = normalize_units(unit_str)
    
    return expr, normalized_unit_str

def preprocess_math_expr(expr: str) -> str:
    """Preprocess mathematical expression to standardize format"""
    expr = expr.strip()
    
    # Extract units but only remove the markup, store the units separately
    expr, _ = extract_units(expr)
    
    # Check for text-only content inside boxed or LaTeX delimiters
    text_patterns = [
        r'\\boxed\s*\{\s*\\text\s*\{([^}]+)\}\s*\}',
        r'\$+\s*\\boxed\s*\{\s*\\text\s*\{([^}]+)\}\s*\}\s*\$+',
        r'\\boxed\s*\{([^}]+)\}',
        r'\$+\s*\\boxed\s*\{([^}]+)\}\s*\$+',
        r'\$+\s*\\text\s*\{([^}]+)\}\s*\$+',
        r'\\text\s*\{([^}]+)\}'
    ]
    
    for pattern in text_patterns:
        match = re.search(pattern, expr)
        if match:
            # This is a text answer, not a mathematical expression
            return match.group(1).replace('.', '')
    
    # Remove LaTeX \boxed command using the helper function
    expr = remove_boxed(expr)
    
    # Try to detect the format
    format_detected = detect_format(expr)
    
    # Convert to SymPy format
    try:
        return str(converter.convert_to_sympy(expr, format_detected))
    except:
        # Fallback to original preprocessing if conversion fails
        return original_preprocess(expr)

def detect_format(expr: str) -> str:
    """Detect the format of the mathematical expression"""
    # Check for LaTeX
    if '\\' in expr or '$' in expr:
        return 'latex'
    
    # Check for MathML
    if '<math' in expr or '<MathML' in expr:
        return 'mathml'
    
    # Check for Wolfram/Mathematica
    if '[' in expr and ']' in expr and not expr.startswith('['):
        return 'wolfram'
    
    # Check for MATLAB
    if '.*' in expr or './' in expr:
        return 'matlab'
    
    # Check for R
    if '%%' in expr:
        return 'r'
    
    # Check for Julia
    if '.*' in expr and not '.*.' in expr:
        return 'julia'
    
    # Check for Excel
    if '=' in expr and not '==' in expr:
        return 'excel'
    
    # Check for AsciiMath
    if 'sqrt' in expr and not '\\sqrt' in expr:
        return 'asciimath'
    
    # Check for Unicode math
    if any(char in expr for char in ['ʸ', '⁄', '·', ' ', ' ', ' ', ' ']):
        return 'unicode'
    
    # Default to Python format
    return 'python'

def original_preprocess(expr: str) -> str:
    """Original preprocessing function as fallback"""
    # Replace common Unicode symbols
    expr = expr.replace('∈', 'in')
    expr = expr.replace('−', '-')
    expr = expr.replace('–', '-')
    expr = expr.replace('÷', '/')
    expr = expr.replace('×', '*')
    expr = expr.replace('⋅', '*')
    expr = expr.replace('·', '*')  # Middle dot
    expr = expr.replace('⁄', '/')  # Fraction slash
    expr = expr.replace('ʸ', '**y')  # Superscript y
    expr = expr.replace(' ', ' ')  # Narrow no-break space
    expr = expr.replace(' ', ' ')  # En space
    expr = expr.replace(' ', ' ')  # Thin space
    expr = expr.replace(' ', '')   # Hair space
    expr = expr.replace('π', 'pi')
    expr = expr.replace('∞', 'oo')
    expr = expr.replace('≠', '!=')
    expr = expr.replace('≤', '<=')
    expr = expr.replace('≥', '>=')
    expr = expr.replace('≈', '~=')
    expr = expr.replace('∝', '~')
    expr = expr.replace('∩', '&')
    expr = expr.replace('∪', '|')
    expr = expr.replace('∅', 'EmptySet')
    expr = expr.replace('ℕ', 'Naturals')
    expr = expr.replace('ℤ', 'Integers')
    expr = expr.replace('ℚ', 'Rationals')
    expr = expr.replace('ℝ', 'Reals')
    expr = expr.replace('ℂ', 'Complexes')
    
    # Handle absolute value notations like |A| = 1
    # Replace |x| with abs(x)
    expr = re.sub(r'\|([^|]+)\|', r'abs(\1)', expr)
    
    # Handle LaTeX absolute value notation
    expr = re.sub(r'\\left\|([^|]+)\\right\|', r'abs(\1)', expr)
    expr = re.sub(r'\\lvert([^|]+)\\rvert', r'abs(\1)', expr)
    
    # First, extract all derivative patterns and replace with unique markers
    derivative_markers = {}
    derivative_count = 0
    
    # Handle derivatives first to prevent interference with other replacements
    derivative_pattern = r'd([a-zA-Z])/d([a-zA-Z])'
    for match in re.finditer(derivative_pattern, expr):
        original = match.group(0)
        replacement = f"DERIV_{derivative_count}"
        derivative_markers[replacement] = f"Derivative({match.group(1)}, {match.group(2)})"
        expr = expr.replace(original, replacement, 1)
        derivative_count += 1
    
    # Handle function calls before other operations
    # Add parentheses around variables after log/ln, sin, cos, etc.
    function_names = ["log", "ln", "sin", "cos", "tan", "cot", "sec", "csc", "exp", "sqrt"]
    for func in function_names:
        # Replace func x with func(x)
        expr = re.sub(rf'{func}\s*([a-zA-Z0-9]+)', rf'{func}(\1)', expr)
    
    # Handle fractions with Unicode fraction slash
    expr = re.sub(r'(\d+)[⁄](\d+)', r'(\1)/(\2)', expr)
    
    # Handle exponents with Unicode superscripts
    expr = re.sub(r'([a-zA-Z])ʸ', r'\1**y', expr)
    
    # Handle natural log
    expr = expr.replace('ln', 'log')
    
    # Handle LaTeX exponents with curly braces - needs to be done before regular exponents
    expr = re.sub(r'(\d+|\w+)\^\{([^}]+)\}', r'\1**(\2)', expr)
    
    # Handle caret exponents - needs to be done early
    # Handle complex exponents inside parentheses: x^(y+1)
    expr = re.sub(r'([a-zA-Z0-9])\^\(([^)]+)\)', r'\1**(\2)', expr)
    # Handle simple exponents: x^y
    expr = re.sub(r'([a-zA-Z0-9])\^([-+]?\d+(?:\.\d+)?)', r'\1**\2', expr)
    
    # Normalize square root representations
    expr = expr.replace('√(', 'sqrt(')
    expr = re.sub(r'√(\d+(?:\.\d+)?)', r'sqrt(\1)', expr)
    expr = re.sub(r'\bsqrt(\d+(?:\.\d+)?)\b', r'sqrt(\1)', expr)
    
    # Insert explicit multiplication
    expr = re.sub(r'(\d)(?=sqrt\()', r'\1*', expr)
    
    # Handle ± symbols
    expr = re.sub(r'(\\pm|±)\s*(\d+)', r'\2 or -\2', expr)
    
    # Handle text commands in LaTeX
    expr = re.sub(r'\\text\s*{\s*or\s*}', 'or', expr)
    
    # Fix spacing
    expr = re.sub(r'\s+', ' ', expr)
    
    # Remove extra spaces around operators
    expr = re.sub(r'\s*([+\-*/=])\s*', r'\1', expr)
    
    # Handle brackets
    expr = expr.replace('[', '(')
    expr = expr.replace(']', ')')
    
    # Create a list of tokens to skip in implicit multiplication
    protected_terms = function_names + [f"DERIV_{i}" for i in range(derivative_count)]
    
    # Handle implicit multiplication for variables next to each other
    # First, find all single-letter variables
    variables = ["x", "y", "z", "t", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "u", "v", "w"]
    
    # Add explicit multiplication between adjacent variables
    for i, var1 in enumerate(variables):
        for var2 in variables:
            if var1 != var2:  # Don't replace xx with x*x
                expr = re.sub(rf'(?<![a-zA-Z0-9_]){var1}(?![a-zA-Z0-9_(\[])(?=[{var2}])', f'{var1}*', expr)
    
    # Handle implicit multiplication more generally
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)  # 2x -> 2*x
    expr = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr)  # x2 -> x*2
    
    # Find all multi-letter identifiers/words in the expression
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
    words = sorted(set(words), key=len, reverse=True)  # Sort by length to avoid partial replacements
    
    # For each word that's not a function name or variable, add asterisk before/after if adjacent to a variable
    for word in words:
        if word not in protected_terms and word not in variables and len(word) > 1:
            for var in variables:
                expr = re.sub(rf'\b{word}{var}\b', f'{word}*{var}', expr)
                expr = re.sub(rf'\b{var}{word}\b', f'{var}*{word}', expr)
    
    # Restore all derivative markers
    for marker, replacement in derivative_markers.items():
        expr = expr.replace(marker, replacement)
    
    # Handle negative exponents
    expr = re.sub(r'(\w+)\^\(-(\d+)\)', r'\1**(-\2)', expr)
    
    # Final check for log without parentheses
    expr = re.sub(r'log\s+([a-zA-Z0-9]+)', r'log(\1)', expr)
    
    return expr

def parse_expr_flex(expr_str: str):
    """Flexible expression parser that handles multiple formats"""
    expr_str = preprocess_math_expr(expr_str)
    
    # Check for text-only answers with semantic categories
    expr_lower = expr_str.lower()
    
    # Define semantic categories with variations
    no_solution_texts = ["no solution", "no solutions", "empty set", "null set", "∅", "{}", "emptyset"]
    infinite_solution_texts = ["infinitely many", "infinite solutions", "infinite number of solutions", "all real numbers"]
    
    # Check if the expression belongs to a semantic category
    if any(term in expr_lower for term in no_solution_texts):
        return "No solutions"
    if any(term in expr_lower for term in infinite_solution_texts):
        return "Infinitely many solutions"
    
    # Handle equations with absolute values: x=|n| or |n|=x
    abs_eq_pattern = r'^([a-zA-Z])\s*=\s*abs\(([^)]+)\)$|^abs\(([^)]+)\)\s*=\s*([a-zA-Z])$'
    abs_match = re.search(abs_eq_pattern, expr_str)
    if abs_match:
        try:
            from sympy import Abs, symbols, Eq
            groups = abs_match.groups()
            if groups[0] is not None:  # x=|n| format
                var = symbols(groups[0])
                inner = sympify(groups[1])
                return Eq(var, Abs(inner))
            else:  # |n|=x format
                var = symbols(groups[3])
                inner = sympify(groups[2])
                return Eq(Abs(inner), var)
        except Exception as e:
            print(f"Failed to parse absolute value equation: {e}")
    
    # Handle absolute value equations: abs(X)=n
    abs_eq_pattern = r'abs\(([^)]+)\)\s*=\s*(.+)'
    abs_match = re.search(abs_eq_pattern, expr_str)
    if abs_match:
        try:
            from sympy import Abs, symbols, Eq
            var_name = abs_match.group(1).strip()
            rhs = abs_match.group(2).strip()
            
            # Create symbols for variable if it's a letter
            if len(var_name) == 1 and var_name.isalpha():
                var = symbols(var_name)
            else:
                var = sympify(var_name)
                
            rhs_expr = sympify(rhs)
            return Eq(Abs(var), rhs_expr)
        except Exception as e:
            print(f"Failed to parse absolute value equation: {e}")
    
    # Handle comma-separated equations (multiple solutions)
    if ',' in expr_str:
        try:
            parts = [p.strip() for p in expr_str.split(',')]
            return [parse_expr_flex(p) for p in parts]
        except Exception as e:
            print(f"Failed to parse comma-separated expressions: {e}")
    
    # Try multiple parsing approaches
    exceptions = []
    
    # Handle common LaTeX derivative notations like \frac{dy}{dx}
    if '\\frac{d' in expr_str or 'dy/dx' in expr_str:
        try:
            if '=' in expr_str:
                lhs, rhs = expr_str.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                # Check for derivative notation patterns
                frac_pattern = r'\\frac{d([a-zA-Z])}{d([a-zA-Z])}'
                direct_pattern = r'd([a-zA-Z])/d([a-zA-Z])'
                
                frac_match = re.search(frac_pattern, lhs)
                direct_match = re.search(direct_pattern, lhs)
                
                if frac_match:
                    dep_var, indep_var = frac_match.groups()
                    from sympy import Derivative, symbols
                    y = symbols(dep_var.strip())
                    x = symbols(indep_var.strip())
                    lhs_expr = Derivative(y, x)
                elif direct_match:
                    dep_var, indep_var = direct_match.groups()
                    from sympy import Derivative, symbols
                    y = symbols(dep_var.strip())
                    x = symbols(indep_var.strip())
                    lhs_expr = Derivative(y, x)
                else:
                    raise ValueError("Derivative pattern not recognized in: " + lhs)
                
                # Parse RHS
                try:
                    rhs_expr = sympify(rhs)
                    return Eq(lhs_expr, rhs_expr)
                except Exception as e:
                    # If RHS parsing fails, try LaTeX parsing
                    try:
                        rhs_expr = parse_latex(rhs)
                        return Eq(lhs_expr, rhs_expr)
                    except:
                        # Final attempt - clean up and try again
                        rhs = rhs.replace('log x', 'log(x)')
                        rhs_expr = sympify(rhs)
                        return Eq(lhs_expr, rhs_expr)
        except Exception as e:
            exceptions.append(f"Derivative parsing failed: {str(e)}")
    
    # Handle derivatives with Derivative notation
    if 'Derivative' in expr_str:
        try:
            # Split into LHS and RHS
            if '=' in expr_str:
                lhs, rhs = expr_str.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                # Parse derivative on LHS
                if 'Derivative' in lhs:
                    # Extract variables from Derivative(y, x)
                    match = re.match(r'Derivative\(([^,]+),\s*([^)]+)\)', lhs)
                    if match:
                        dep_var, indep_var = match.groups()
                        from sympy import Derivative, symbols
                        y = symbols(dep_var.strip())
                        x = symbols(indep_var.strip())
                        lhs_expr = Derivative(y, x)
                        
                        # Parse RHS
                        try:
                            rhs_expr = sympify(rhs)
                            return Eq(lhs_expr, rhs_expr)
                        except Exception as e:
                            # If RHS parsing fails, we'll try cleaning it up a bit
                            rhs = rhs.replace('log x', 'log(x)')
                            rhs_expr = sympify(rhs)
                            return Eq(lhs_expr, rhs_expr)
            
            # If we get here, we have Derivative but not in the expected format
            # Try to sympify the whole expression
            return sympify(expr_str)
        except Exception as e:
            exceptions.append(f"Derivative parsing failed: {str(e)}")
    
    # For LaTeX expressions
    if '\\' in expr_str:
        try:
            return parse_latex(expr_str)
        except Exception as e:
            exceptions.append(f"LaTeX parsing failed: {str(e)}")
    
    # Try direct sympify
    try:
        return sympify(expr_str)
    except Exception as e:
        exceptions.append(f"Sympify failed: {str(e)}")
    
    # Try with additional parsing techniques
    try:
        from sympy import sin, cos, tan, cot, sec, csc, log, ln, binomial, Sum, Integral, limit
        from sympy import EmptySet, Naturals, Integers, Rationals, Reals, Complexes
        from sympy import I, Derivative
        
        # Handle common issues like "log x" instead of "log(x)"
        expr_str = re.sub(r'log\s+([a-zA-Z])', r'log(\1)', expr_str)
        
        return eval(expr_str, {
            "sympy": __import__("sympy"),
            "sin": sin, "cos": cos, "tan": tan, "cot": cot,
            "sec": sec, "csc": csc, "log": log, "ln": ln,
            "binomial": binomial, "Sum": Sum, "Integral": Integral,
            "limit": limit, "I": I, "EmptySet": EmptySet,
            "Naturals": Naturals, "Integers": Integers,
            "Rationals": Rationals, "Reals": Reals,
            "Complexes": Complexes, "Derivative": Derivative
        })
    except Exception as e:
        exceptions.append(f"Additional parsing failed: {str(e)}")
    
    # If we got here, all methods failed
    error_msg = "\n".join(exceptions)
    raise ValueError(f"Unable to parse: '{expr_str}'\nErrors:\n{error_msg}")

def extract_solutions(expr_str: str):
    expr_str = preprocess_math_expr(expr_str)

    # symbolic set: S = {1, 2}
    match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{(.+?)\}$', expr_str)
    if match:
        varname, content = match.groups()
        parts = [p.strip() for p in content.split(',')]
        try:
            values = set(simplify(parse_expr_flex(p)) for p in parts)
            return varname, values
        except Exception as e:
            print(f"Failed to parse symbolic set: {content}\n{e}")
            return None

    # x in {1, 2}
    match = re.match(r'([a-zA-Z]+)\s+in\s+\{(.+?)\}', expr_str)
    if match:
        var = symbols(match.group(1))
        values = match.group(2).split(',')
        return None, set(Eq(var, simplify(v.strip())) for v in values)

    # {x = 1, x = 2} or {1, 2}
    if expr_str.startswith('{') and expr_str.endswith('}'):
        contents = expr_str[1:-1]
        parts = contents.split(',')
        values = []
        for part in parts:
            try:
                parsed = parse_expr_flex(part.strip())
                if isinstance(parsed, Relational):
                    values.append(simplify(parsed))
                else:
                    values.append(simplify(parsed))
            except Exception:
                continue
        return None, set(values)

    # x = 1 or x = 2
    parts = re.split(r'\s+or\s+', expr_str)
    eqs = []
    for part in parts:
        try:
            parsed = parse_expr_flex(part)
            if isinstance(parsed, Relational):
                eqs.append(simplify(parsed))
            else:
                eqs.append(Eq(x, simplify(parsed)))
        except Exception:
            continue
    return None, set(eqs)

def are_math_equivalent_sets(a_str: str, b_str: str) -> bool:
    """Compare two mathematical expressions for equivalence"""
    try:
        # Special case for fractions vs decimals (e.g., 1/2 vs 0.5)
        # This checks if both inputs can be directly converted to floats and compared
        try:
            # Check if the inputs look like simple numbers (no variables)
            if all(c in "0123456789./-" for c in a_str.strip()) and all(c in "0123456789./-" for c in b_str.strip()):
                from fractions import Fraction
                
                # Try to convert both to Fraction objects
                try:
                    a_frac = Fraction(a_str.strip())
                    b_frac = Fraction(b_str.strip())
                    if a_frac == b_frac:
                        print(f"Direct fraction comparison: {a_frac} == {b_frac}")
                        return True
                except ValueError:
                    # If Fraction conversion fails, try float
                    try:
                        a_float = float(eval(a_str.strip()))
                        b_float = float(eval(b_str.strip()))
                        if abs(a_float - b_float) < 1e-10:
                            print(f"Direct float comparison: {a_float} ≈ {b_float}")
                            return True
                    except:
                        pass  # Continue with regular processing
        except:
            pass  # Continue with regular processing
            
        # Special handling for determinant notation |A| = n
        det_pattern = r'\|([A-Z])\|\s*=\s*(\d+)'
        det_match_a = re.search(det_pattern, a_str)
        det_match_b = re.search(det_pattern, b_str)
        number_pattern = r'^[\d.]+$'
        
        # If one expression is determinant and the other is a number, compare directly
        if det_match_a and re.match(number_pattern, b_str.strip()):
            det_value = det_match_a.group(2)
            if det_value.strip() == b_str.strip():
                return True
        elif det_match_b and re.match(number_pattern, a_str.strip()):
            det_value = det_match_b.group(2)
            if det_value.strip() == a_str.strip():
                return True
        
        # First, extract units from both expressions
        a_stripped, a_units = extract_units(a_str)
        b_stripped, b_units = extract_units(b_str)
        
        # Check if both have units and they differ
        if a_units and b_units and a_units != b_units:
            print(f"Units differ: '{a_units}' vs '{b_units}'")
            return False
        
        # First, check for text-only answers with semantic matching
        a_proc = preprocess_math_expr(a_stripped).lower()
        b_proc = preprocess_math_expr(b_stripped).lower()
        
        # Define semantic categories with variations
        no_solution_texts = ["no solution", "no solutions", "empty set", "null set", "∅", "{}", "emptyset"]
        infinite_solution_texts = ["infinitely many", "infinite solutions", "infinite number of solutions", "all real numbers"]
        
        # Check if both belong to the same semantic category
        a_has_no_sol = any(term in a_proc for term in no_solution_texts)
        b_has_no_sol = any(term in b_proc for term in no_solution_texts)
        
        a_has_inf_sol = any(term in a_proc for term in infinite_solution_texts)
        b_has_inf_sol = any(term in b_proc for term in infinite_solution_texts)
        
        # If both answers are text-based and in the same category, they're equivalent
        if (a_has_no_sol and b_has_no_sol) or (a_has_inf_sol and b_has_inf_sol):
            return True
        
        # First try with solution set extraction
        a_name, a_vals = extract_solutions(a_stripped)
        b_name, b_vals = extract_solutions(b_stripped)

        if a_vals and b_vals:
            if a_name and b_name:
                return a_name == b_name and a_vals == b_vals
            return a_vals == b_vals
            
        # If extraction doesn't work, try direct parsing and comparison
        try:
            a = parse_expr_flex(a_stripped)
            b = parse_expr_flex(b_stripped)
            
            print(f"Parsed expressions:\nA: {a}\nB: {b}")
            
            # Handle lists of equations (multiple solutions)
            if isinstance(a, list) or isinstance(b, list):
                # Convert single equation to list for uniform handling
                a_list = a if isinstance(a, list) else [a]
                b_list = b if isinstance(b, list) else [b]
                
                # Convert equations to sets of solutions
                a_solutions = {eq.rhs if isinstance(eq, Relational) else eq for eq in a_list}
                b_solutions = {eq.rhs if isinstance(eq, Relational) else eq for eq in b_list}
                
                # Compare solution sets
                if a_solutions == b_solutions:
                    print("Solution sets match")
                    return True
                
                # Try numerical comparison for each solution
                try:
                    a_nums = {float(sol.evalf()) for sol in a_solutions}
                    b_nums = {float(sol.evalf()) for sol in b_solutions}
                    if a_nums == b_nums:
                        print("Numerical solution sets match")
                        return True
                except:
                    pass
                
                return False
            
            # Check for relational expressions (equations)
            if isinstance(a, Relational) and isinstance(b, Relational):
                if a.rel_op == '=' and b.rel_op == '=':
                    if a.lhs == b.lhs:
                        result = simplify(a.rhs - b.rhs) == 0
                        print(f"Comparing RHS: {a.rhs} == {b.rhs}, Result: {result}")
                        return result
                    elif a.rhs == b.rhs:
                        result = simplify(a.lhs - b.lhs) == 0
                        print(f"Comparing LHS: {a.lhs} == {b.lhs}, Result: {result}")
                        return result
                    else:
                        result1 = simplify((a.lhs - a.rhs) - (b.lhs - b.rhs)) == 0
                        result2 = simplify((a.lhs - b.lhs) - (a.rhs - b.rhs)) == 0
                        print(f"Comparing equations, Result: {result1 or result2}")
                        return result1 or result2
                return simplify(a.lhs - a.rhs - (b.lhs - b.rhs)) == 0
                
            # For regular expressions, just compare if they simplify to the same thing
            if isinstance(a, Relational) or isinstance(b, Relational):
                return False
                
            # Try direct comparison using simplify first - this should handle fractions vs decimals
            direct_comparison = simplify(a - b) == 0
            if direct_comparison:
                print("Direct comparison successful: expressions are equivalent")
                return True
                
            # If direct comparison doesn't work, try numerical comparison with tolerance
            try:
                from sympy import Float, N, Rational
                
                # Convert to float explicitly
                try:
                    a_float = float(N(a, 15))
                    b_float = float(N(b, 15))
                    
                    # Compare with appropriate tolerance
                    if abs(a_float - b_float) < 1e-10:
                        print(f"Numerical comparison: {a_float} ≈ {b_float}")
                        return True
                except Exception as e:
                    print(f"Numerical conversion error: {e}")
                    pass  # Continue with other methods if float conversion fails
                
                # Try rational conversion for fractions vs decimals
                try:
                    if hasattr(a, 'convert_to_Rational') and hasattr(b, 'convert_to_Rational'):
                        a_rational = a.convert_to_Rational()
                        b_rational = b.convert_to_Rational()
                        if a_rational == b_rational:
                            print(f"Rational comparison: {a_rational} == {b_rational}")
                            return True
                except Exception:
                    pass
            except Exception as e:
                print(f"Numerical comparison error: {e}")
                
            # Try alternative comparison approaches
            try:
                from sympy import expand, factor, trigsimp, powsimp
                
                methods = [
                    lambda expr: simplify(expr),
                    lambda expr: expand(expr),
                    lambda expr: factor(expr),
                    lambda expr: trigsimp(expr),
                    lambda expr: powsimp(expr),
                ]
                
                for method in methods:
                    result = method(a - b)
                    if result == 0:
                        print(f"Method {method.__name__} comparison successful")
                        return True
                    
                return False
            except Exception as e:
                print(f"Alternative comparison error: {e}")
                return direct_comparison
                
        except Exception as e:
            print(f"Expression comparison failed: {e}")
            return a_stripped.strip() == b_stripped.strip()

    except Exception as e:
        print(f"Comparison failed:\n{e}")
        return False

# 👇 Accept user input here
if __name__ == "__main__":
    print("🔢 Enter two math expressions or sets to compare.")
    a_input = input("First expression: ")
    b_input = input("Second expression: ")
    
    print("\n🔍 Processing expressions...")
    print(f"Input 1: '{a_input}'")
    print(f"Input 2: '{b_input}'")
    
    # Special handling for determinant notation |A| = n
    det_pattern = r'\|([A-Z])\|\s*=\s*(\d+)'
    det_match_a = re.search(det_pattern, a_input)
    number_pattern = r'^[\d.]+$'
    
    # If one expression is determinant and the other is a number, compare directly
    if det_match_a and re.match(number_pattern, b_input.strip()):
        det_value = det_match_a.group(2)
        if det_value.strip() == b_input.strip():
            print("\n✅ Equivalent (Determinant notation)")
            exit(0)
    
    # Extract units first
    a_stripped, a_units = extract_units(a_input)
    b_stripped, b_units = extract_units(b_input)
    
    if a_units:
        print(f"Units in expression 1: '{a_units}'")
    if b_units:
        print(f"Units in expression 2: '{b_units}'")
    
    # Preprocess
    a_processed = preprocess_math_expr(a_input)
    b_processed = preprocess_math_expr(b_input)
    print(f"\nPreprocessed 1: '{a_processed}'")
    print(f"Preprocessed 2: '{b_processed}'")
    
    try:
        # Check for text-only answers with semantic categories
        no_solution_texts = ["no solution", "no solutions", "empty set", "null set", "∅", "{}", "emptyset"]
        infinite_solution_texts = ["infinitely many", "infinite solutions", "infinite number of solutions", "all real numbers"]
        
        a_lower = a_processed.lower()
        b_lower = b_processed.lower()
        
        a_is_text = any(term in a_lower for term in no_solution_texts + infinite_solution_texts)
        b_is_text = any(term in b_lower for term in no_solution_texts + infinite_solution_texts)
        
        if a_is_text or b_is_text:
            print("\nDetected text-based answer - will compare semantically.")
            
            # Standardize the text answers for display
            if any(term in a_lower for term in no_solution_texts):
                a_parsed = "No solutions"
            elif any(term in a_lower for term in infinite_solution_texts):
                a_parsed = "Infinitely many solutions"
            else:
                a_parsed = parse_expr_flex(a_input)
                
            if any(term in b_lower for term in no_solution_texts):
                b_parsed = "No solutions"
            elif any(term in b_lower for term in infinite_solution_texts):
                b_parsed = "Infinitely many solutions"
            else:
                b_parsed = parse_expr_flex(b_input)
        else:
            # Attempt parsing before comparison
            print("\nAttempting to parse expressions...")
            
            # Special handling for derivative expressions
            if "dy/dx" in a_processed or "Derivative" in a_processed:
                from sympy import Symbol, Derivative, sympify
                # Process the first expression for derivatives
                if "=" in a_processed:
                    lhs, rhs = a_processed.split("=", 1)
                    if "dy/dx" in lhs:
                        x, y = symbols('x y')
                        lhs_expr = Derivative(y, x)
                        rhs_expr = sympify(rhs)
                        a_parsed = Eq(lhs_expr, rhs_expr)
                    else:
                        a_parsed = parse_expr_flex(a_input)
                else:
                    a_parsed = parse_expr_flex(a_input)
            else:
                a_parsed = parse_expr_flex(a_input)
                
            # Similar handling for second expression
            if "dy/dx" in b_processed or "Derivative" in b_processed:
                from sympy import Symbol, Derivative, sympify
                # Process the second expression for derivatives
                if "=" in b_processed:
                    lhs, rhs = b_processed.split("=", 1)
                    if "dy/dx" in lhs:
                        x, y = symbols('x y')
                        lhs_expr = Derivative(y, x)
                        rhs_expr = sympify(rhs)
                        b_parsed = Eq(lhs_expr, rhs_expr)
                    else:
                        b_parsed = parse_expr_flex(b_input)
                else:
                    b_parsed = parse_expr_flex(b_input)
            else:
                b_parsed = parse_expr_flex(b_input)
        
        print(f"Parsed 1: {a_parsed}")
        print(f"Parsed 2: {b_parsed}")
        
        # Compare
        print("\n🧮 Checking equivalence...")
        is_equivalent = are_math_equivalent_sets(a_input, b_input)
        
        # Show unit comparison result if applicable
        if is_equivalent and a_units and b_units:
            if a_units == b_units:
                print(f"✅ Units match: '{a_units}'")
            else:
                print(f"⚠️ Units differ: '{a_units}' vs '{b_units}'")
                is_equivalent = False
                
        result_text = "✅ Equivalent" if is_equivalent else "❌ Not Equivalent"
        print(f"\nResult: {result_text}")
    except Exception as e:
        print(f"\n❗ Error during processing: {e}")
        print("\nResult: ❌ Not Equivalent (due to processing error)")