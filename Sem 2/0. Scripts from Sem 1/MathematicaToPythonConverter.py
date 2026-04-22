import re
import os

def mathematica_to_python(expr):
    """
    Convert Mathematica expression to Python.
    
    Args:
        expr (str): Mathematica expression
        
    Returns:
        str: Python expression
    """
    # Replace Greek letters FIRST (before removing backslashes)
    expr = expr.replace(r'\[Alpha]', 'alpha')
    expr = expr.replace(r'\[Theta]k', 'theta')
    expr = expr.replace(r'\[Pi]', 'np.pi')
    
    # Remove line continuation backslashes (backslash followed by newline)
    # This handles cases like: 12345\<newline>67890
    expr = re.sub(r'\\\s*\n\s*', '', expr)
    
    # Remove any remaining backslashes (from Mathematica line continuations)
    expr = expr.replace('\\', '')
    
    # Replace functions BEFORE power notation to handle Sin[x]^2 correctly
    expr = re.sub(r'Cos\[([^\]]+)\]', r'np.cos(\1)', expr)
    expr = re.sub(r'Sin\[([^\]]+)\]', r'np.sin(\1)', expr)
    expr = re.sub(r'ArcTan\[([^\]]+)\]', r'np.arctan(\1)', expr)
    
    # Replace power notation with parentheses: (...)^b -> (...)**b
    # This needs to handle deeply nested parentheses
    # Strategy: repeatedly replace )^X with )**X until no more matches
    while re.search(r'\)\^(\w+)', expr):
        expr = re.sub(r'\)\^(\w+)', r')**\1', expr)
    
    # Replace power notation: a^b -> a**b (after parentheses to avoid conflicts)
    expr = re.sub(r'(\w+)\^(\d+)', r'\1**\2', expr)
    expr = re.sub(r'(\w+)\^(\w+)', r'\1**\2', expr)
    
    # Replace variable names
    expr = expr.replace('GN', 'G_N')
    expr = expr.replace('rg', 'r_g')
    
    # Add explicit multiplication operators where needed
    # Between ) and number: ) 82944 -> ) * 82944
    expr = re.sub(r'\)\s+(\d)', r') * \1', expr)
    # Between number and letter: 1024 GN -> 1024 * G_N
    expr = re.sub(r'(\d)\s+([A-Za-z_])', r'\1 * \2', expr)
    # Between letter and number: N 2 -> N * 2  
    expr = re.sub(r'([A-Za-z_])\s+(\d)', r'\1 * \2', expr)
    # Between two variables: G_N N -> G_N * N
    expr = re.sub(r'([A-Za-z_]\w*)\s+([A-Za-z_])', r'\1 * \2', expr)
    # Between ) and (: )( -> ) * (
    expr = re.sub(r'\)\s*\(', r') * (', expr)
    # Between ) and letter: ) alpha -> ) * alpha
    expr = re.sub(r'\)\s+([A-Za-z_])', r') * \1', expr)
    # Between letter and (: alpha ( -> alpha * (
    expr = re.sub(r'([A-Za-z_\d])\s+\(', r'\1 * (', expr)
    # Between number and (: 2 ( -> 2 * (
    expr = re.sub(r'(\d)\s+\(', r'\1 * (', expr)
    
    # Remove N**2 terms (occupation number) - do this AFTER adding multiplication operators
    expr = re.sub(r'N\s*\*\*\s*2\s*\*\s*', '', expr)
    expr = re.sub(r'\*\s*N\s*\*\*\s*2\s*\*\s*', ' * ', expr)
    expr = re.sub(r'\*\s*N\s*\*\*\s*2(?=\s|\)|$)', '', expr)
    
    # Clean up spaces
    expr = re.sub(r'\s+', ' ', expr)
    expr = expr.strip()
    
    return expr

def extract_functions(content):
    """
    Extract all function definitions from the Mathematica text file.
    
    Args:
        content (str): File content
        
    Returns:
        tuple: (annihilations, transitions) where each is a list of tuples (function_name, expression)
    """
    annihilations = []
    transitions = []
    
    # Pattern to match: dPd\[CapitalOmega]<name> = <expression>
    # Expression continues until the next dPd or end of file
    pattern = r'dPd\\?\[CapitalOmega\](\w+)\s*=\s*(.*?)(?=\ndPd\\?\[CapitalOmega\]|\Z)'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        name = match.group(1)
        expr = match.group(2).strip()
        
        # Remove trailing semicolon if present
        expr = expr.rstrip(';').strip()
        
        # Remove any Series[], Print[], or other Mathematica commands that might follow
        # Split by newline and filter out lines that start with Series, Print, etc.
        lines = expr.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are Mathematica commands (not part of the expression)
            if stripped.startswith('Series[') or stripped.startswith('Print['):
                break  # Stop processing at first command
            cleaned_lines.append(line)
        expr = '\n'.join(cleaned_lines).strip()
        
        # Remove trailing semicolon again after cleaning
        expr = expr.rstrip(';').strip()
        
        # Skip if expression is empty
        if not expr:
            continue
        
        # Check if this is a transition (contains "to") or annihilation
        if 'to' in name.lower():
            transitions.append((name, expr))
        else:
            annihilations.append((name, expr))
    
    return annihilations, transitions

def create_annihilation_function(name, expr):
    """
    Create a complete Python function definition for annihilation.
    
    Args:
        name (str): Function identifier (e.g., "2p", "3d")
        expr (str): Mathematica expression
        
    Returns:
        str: Complete Python function definition
    """
    python_expr = mathematica_to_python(expr)
    
    # Create function name
    func_name = f"ann_{name.lower()}"
    
    # Build the function
    function_str = f'''def {func_name}(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for {name} annihilation.
    """
    return {python_expr}
'''
    
    return function_str

def create_transition_function(name, expr):
    """
    Create a complete Python function definition for transition.
    
    Args:
        name (str): Function identifier (e.g., "3pto2p", "4dto3d")
        expr (str): Mathematica expression
        
    Returns:
        str: Complete Python function definition
    """
    python_expr = mathematica_to_python(expr)
    
    # Create function name: trans_3p_2p from 3pto2p
    clean_name = name.lower().replace('to', '_')
    func_name = f"trans_{clean_name}"
    
    # Format display name with space: "3p 2p" from "3pto2p"
    display_name = name.lower().replace('to', ' ')
    
    # Build the function
    function_str = f'''def {func_name}(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for {display_name} transition.
    """
    return {python_expr}
'''
    
    return function_str

def main():
    """Main converter."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    text_file = os.path.join(script_dir, "MathematicaFunctions.txt")
    output_file = os.path.join(script_dir, "ConvertedFunctions.py")
    
    print("=" * 80)
    print("MATHEMATICA TO PYTHON CONVERTER")
    print("=" * 80)
    print(f"\nReading: {text_file}")
    print(f"Output: {output_file}\n")
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        annihilations, transitions = extract_functions(content)
        
        print(f"Found {len(annihilations)} annihilation(s)")
        print(f"Found {len(transitions)} transition(s)\n")
        
        # Build output
        output_lines = []
        output_lines.append("import numpy as np\n")
        output_lines.append('"""')
        output_lines.append("Differential power functions for gravitational atom processes.")
        output_lines.append("")
        output_lines.append("Automatically converted from Mathematica using MathematicaToPythonConverter.py")
        output_lines.append('"""\n\n')
        
        # Add annihilation functions
        if annihilations:
            output_lines.append("# Annihilation functions")
            for name, expr in annihilations:
                python_func = create_annihilation_function(name, expr)
                output_lines.append(python_func)
                output_lines.append("\n")
        
        # Add transition functions
        if transitions:
            output_lines.append("\n# Transition functions")
            for name, expr in transitions:
                python_func = create_transition_function(name, expr)
                output_lines.append(python_func)
                output_lines.append("\n")
        
        # Add annihilation dictionary
        if annihilations:
            output_lines.append("\ndiff_power_ann_dict = {")
            for i, (name, _) in enumerate(annihilations):
                comma = "," if i < len(annihilations) - 1 else ""
                output_lines.append(f'    "{name.lower()}": ann_{name.lower()}{comma}')
            output_lines.append("}\n")
        
        # Add transition dictionary
        if transitions:
            output_lines.append("\ndiff_power_trans_dict = {")
            for i, (name, _) in enumerate(transitions):
                comma = "," if i < len(transitions) - 1 else ""
                # Display name format: "3p 2p" from "3pto2p"
                display_name = name.lower().replace('to', ' ')
                # Function name format: trans_3p_2p from 3pto2p
                func_name = name.lower().replace('to', '_')
                output_lines.append(f'    "{display_name}": trans_{func_name}{comma}')
            output_lines.append("}\n")
        
        # Write to file
        output_text = '\n'.join(output_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        # Also print to console
        print("=" * 80)
        print("PYTHON CODE")
        print("=" * 80)
        print(output_text)
        
        print("\n" + "=" * 80)
        print(f"DONE - Functions saved to: {output_file}")
        print("=" * 80)
        print("\nNOTE: Please review the output and verify:")
        print("  - All parentheses are balanced")
        print("  - Multiplication operators (*) are correct")
        print("  - Angular terms (cos, sin) converted properly")
        print("  - Powers (**) are correct")
        print("  - N**2 terms removed (occupation number)")
        
    except FileNotFoundError:
        print(f"ERROR: File not found at {text_file}")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()