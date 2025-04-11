import re
from typing import Dict, Any
from sympy import symbols, sympify, parse_expr
from sympy.parsing.latex import parse_latex
from sympy.parsing.mathematica import mathematica
from sympy.parsing.maxima import parse_maxima
from sympy.parsing.sympy_parser import parse_expr as sympy_parse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import numpy as np

class MathFormatConverter:
    def __init__(self):
        self.symbols = {}
        self._init_symbols()
    
    def _init_symbols(self):
        """Initialize common mathematical symbols and their SymPy equivalents"""
        # Greek letters
        greek_letters = {
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
            'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
            'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
            'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
            'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega'
        }
        
        # Mathematical operators
        operators = {
            '∑': 'Sum', '∫': 'Integral', '∏': 'Product',
            '∞': 'oo', '≠': '!=', '≤': '<=', '≥': '>=',
            '≈': '~=', '∝': '~', '∩': '&', '∪': '|',
            '∈': 'in', '∉': 'not in', '⊂': 'subset', '⊃': 'superset',
            '∀': 'ForAll', '∃': 'Exists', '∧': '&', '∨': '|',
            '¬': '~', '→': '>>', '↔': '<<', '⇒': '>>', '⇔': '<<'
        }
        
        # Special sets
        sets = {
            '∅': 'EmptySet', 'ℕ': 'Naturals', 'ℤ': 'Integers',
            'ℚ': 'Rationals', 'ℝ': 'Reals', 'ℂ': 'Complexes'
        }
        
        self.symbols = {**greek_letters, **operators, **sets}
    
    def convert_to_sympy(self, expr: str, input_format: str) -> Any:
        """Convert mathematical expression from various formats to SymPy format"""
        try:
            if input_format.lower() == 'latex':
                return self._convert_latex(expr)
            elif input_format.lower() == 'asciimath':
                return self._convert_asciimath(expr)
            elif input_format.lower() == 'mathml':
                return self._convert_mathml(expr)
            elif input_format.lower() == 'wolfram':
                return self._convert_wolfram(expr)
            elif input_format.lower() == 'mathematica':
                return self._convert_mathematica(expr)
            elif input_format.lower() == 'matlab':
                return self._convert_matlab(expr)
            elif input_format.lower() == 'r':
                return self._convert_r(expr)
            elif input_format.lower() == 'julia':
                return self._convert_julia(expr)
            elif input_format.lower() == 'python':
                return self._convert_python(expr)
            elif input_format.lower() == 'excel':
                return self._convert_excel(expr)
            elif input_format.lower() == 'desmos':
                return self._convert_desmos(expr)
            elif input_format.lower() == 'geogebra':
                return self._convert_geogebra(expr)
            elif input_format.lower() == 'unicode':
                return self._convert_unicode(expr)
            else:
                raise ValueError(f"Unsupported format: {input_format}")
        except Exception as e:
            raise ValueError(f"Conversion failed: {str(e)}")
    
    def _convert_latex(self, expr: str) -> Any:
        """Convert LaTeX to SymPy"""
        try:
            return parse_latex(expr)
        except:
            # Fallback to manual conversion
            expr = self._replace_symbols(expr)
            return parse_expr(expr)
    
    def _convert_asciimath(self, expr: str) -> Any:
        """Convert AsciiMath to SymPy"""
        # Convert AsciiMath specific syntax
        expr = expr.replace('sqrt', 'sqrt')
        expr = expr.replace('^', '**')
        expr = expr.replace('*', '*')
        expr = expr.replace('/', '/')
        return parse_expr(expr)
    
    def _convert_mathml(self, expr: str) -> Any:
        """Convert MathML to SymPy"""
        try:
            # Parse MathML
            soup = BeautifulSoup(expr, 'xml')
            # Convert to SymPy expression
            # This is a simplified version - would need more complex parsing for full MathML support
            return parse_expr(self._mathml_to_sympy(soup))
        except:
            raise ValueError("Invalid MathML format")
    
    def _convert_wolfram(self, expr: str) -> Any:
        """Convert Wolfram Alpha format to SymPy"""
        # Convert Wolfram specific syntax
        expr = expr.replace('[', '(')
        expr = expr.replace(']', ')')
        expr = expr.replace('^', '**')
        return parse_expr(expr)
    
    def _convert_mathematica(self, expr: str) -> Any:
        """Convert Mathematica format to SymPy"""
        try:
            return mathematica(expr)
        except:
            # Fallback to manual conversion
            expr = expr.replace('^', '**')
            expr = expr.replace('*', '*')
            return parse_expr(expr)
    
    def _convert_matlab(self, expr: str) -> Any:
        """Convert MATLAB format to SymPy"""
        # Convert MATLAB specific syntax
        expr = expr.replace('^', '**')
        expr = expr.replace('.*', '*')
        expr = expr.replace('./', '/')
        return parse_expr(expr)
    
    def _convert_r(self, expr: str) -> Any:
        """Convert R format to SymPy"""
        # Convert R specific syntax
        expr = expr.replace('^', '**')
        expr = expr.replace('%%', '%')
        return parse_expr(expr)
    
    def _convert_julia(self, expr: str) -> Any:
        """Convert Julia format to SymPy"""
        # Convert Julia specific syntax
        expr = expr.replace('^', '**')
        expr = expr.replace('.*', '*')
        return parse_expr(expr)
    
    def _convert_python(self, expr: str) -> Any:
        """Convert Python format to SymPy"""
        return parse_expr(expr)
    
    def _convert_excel(self, expr: str) -> Any:
        """Convert Excel formula to SymPy"""
        # Convert Excel specific syntax
        expr = expr.replace('^', '**')
        expr = expr.replace('*', '*')
        expr = expr.replace('/', '/')
        return parse_expr(expr)
    
    def _convert_desmos(self, expr: str) -> Any:
        """Convert Desmos format to SymPy"""
        # Convert Desmos specific syntax
        expr = expr.replace('^', '**')
        expr = expr.replace('*', '*')
        expr = expr.replace('/', '/')
        return parse_expr(expr)
    
    def _convert_geogebra(self, expr: str) -> Any:
        """Convert GeoGebra format to SymPy"""
        # Convert GeoGebra specific syntax
        expr = expr.replace('^', '**')
        expr = expr.replace('*', '*')
        expr = expr.replace('/', '/')
        return parse_expr(expr)
    
    def _convert_unicode(self, expr: str) -> Any:
        """Convert Unicode math to SymPy"""
        # Handle derivatives
        expr = re.sub(r'd([a-zA-Z])/d([a-zA-Z])', r'Derivative(\1, \2)', expr)
        
        # Handle fractions with Unicode fraction slash
        expr = re.sub(r'(\d+)[⁄](\d+)', r'(\1)/(\2)', expr)
        
        # Handle exponents with Unicode superscripts
        expr = re.sub(r'([a-zA-Z])ʸ', r'\1**y', expr)
        
        # Handle natural log
        expr = expr.replace('ln', 'log')
        
        # Replace Unicode operators
        expr = expr.replace('·', '*')  # Middle dot
        expr = expr.replace('⁄', '/')  # Fraction slash
        expr = expr.replace('−', '-')  # Minus sign
        expr = expr.replace('×', '*')  # Multiplication sign
        
        # Handle spaces
        expr = expr.replace(' ', ' ')  # Narrow no-break space
        expr = expr.replace(' ', ' ')  # En space
        expr = expr.replace(' ', ' ')  # Thin space
        expr = expr.replace(' ', '')   # Hair space
        
        # Handle brackets
        expr = expr.replace('[', '(')
        expr = expr.replace(']', ')')
        
        # Remove extra spaces around operators
        expr = re.sub(r'\s*([+\-*/=])\s*', r'\1', expr)
        
        return parse_expr(expr)
    
    def _replace_symbols(self, expr: str) -> str:
        """Replace Unicode symbols with their SymPy equivalents"""
        for symbol, replacement in self.symbols.items():
            expr = expr.replace(symbol, replacement)
        return expr
    
    def _mathml_to_sympy(self, soup: BeautifulSoup) -> str:
        """Convert MathML to SymPy expression string"""
        # This is a simplified version - would need more complex parsing for full MathML support
        result = []
        for element in soup.find_all():
            if element.name == 'mi':
                result.append(element.text)
            elif element.name == 'mo':
                result.append(element.text)
            elif element.name == 'mn':
                result.append(element.text)
            elif element.name == 'msup':
                base = self._mathml_to_sympy(element.find('mi') or element.find('mn'))
                exp = self._mathml_to_sympy(element.find('mn'))
                result.append(f"{base}**{exp}")
        return ' '.join(result) 