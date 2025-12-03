import re
import sys

def convert_latex_to_markdown(text):
    # Convert display math
    text = re.sub(r'\\\[', r'$$', text)
    text = re.sub(r'\\\]', r'$$', text)

    text = re.sub(r'\\begin{align}', r'$$\begin{align}', text)
    text = re.sub(r'\\begin{align*}', r'$$\begin{align*}', text)

    text = re.sub(r'\\end{align}', r'\\end{align}$$', text)
    text = re.sub(r'\\end{align*}', r'\\end{align*}$$', text)
    
    # Sections
    text = re.sub(r'\\section\{([^}]+)\}', r'# \1', text)
    text = re.sub(r'\\subsection\{([^}]+)\}', r'## \1', text)
    text = re.sub(r'\\subsubsection\{([^}]+)\}', r'### \1', text)
    
    # Text formatting
    text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', text)
    text = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', text)
    text = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', text)
    text = re.sub(r'\\underline\{([^}]+)\}', r'*\1*', text)
    
    # Environments
    text = re.sub(r'\\begin\{proof\}', r'**Proof:**', text)
    text = re.sub(r'\\end\{proof\}', r'', text)
    text = re.sub(r'\\begin\{itemize\}', '', text)
    text = re.sub(r'\\end\{itemize\}', '', text)
    text = re.sub(r'\\begin\{enumerate\}', '', text)
    text = re.sub(r'\\end\{enumerate\}', '', text)
    
    # Lists
    text = re.sub(r'^\s*\\item\s+', '- ', text, flags=re.MULTILINE)
    
    # Structure
    text = re.sub(r'\\noindent\s*', '', text)
    text = re.sub(r'\\bigskip\s*', '\n', text)
    text = re.sub(r'\\newpage\s*', '\n---\n', text)
    
    return text

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        content = f.read()
    
    print(convert_latex_to_markdown(content))