"""
Python Tokenizer/Untokenizer using PLY for Encoder-Decoder Models
All information encoded as numeric token IDs with vocabulary
"""

import ply.lex as lex
from typing import List, Dict
from dataclasses import dataclass
import json
import string


@dataclass
class Vocabulary:
    """Universal vocabulary for token encoding/decoding"""
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    special_tokens: Dict[str, int]

    def __init__(self, create_universal: bool = True):
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,  # Begin of sequence
            '<EOS>': 3,  # End of sequence
        }

        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)

        if create_universal:
            self._create_universal_vocabulary()
    
    def __len__(self):
        return len(self.token_to_id)

    def _create_universal_vocabulary(self):
        """Create a universal vocabulary with all Python tokens and printable characters"""

        # Add all Python fixed tokens (keywords, operators, delimiters)
        fixed_tokens = [
            # Keywords
            'FALSE', 'NONE', 'TRUE', 'AND', 'AS', 'ASSERT',
            'ASYNC', 'AWAIT', 'BREAK', 'CLASS', 'CONTINUE',
            'DEF', 'DEL', 'ELIF', 'ELSE', 'EXCEPT', 'FINALLY',
            'FOR', 'FROM', 'GLOBAL', 'IF', 'IMPORT', 'IN', 'IS',
            'LAMBDA', 'NONLOCAL', 'NOT', 'OR', 'PASS', 'RAISE',
            'RETURN', 'TRY', 'WHILE', 'WITH', 'YIELD',
            # Operators
            'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MODULO', 'POWER',
            'FLOORDIV', 'MATMUL', 'LSHIFT', 'RSHIFT',
            'BITOR', 'BITXOR', 'BITAND', 'BITNOT',
            'EQ', 'NE', 'LT', 'LE', 'GT', 'GE',
            'ASSIGN', 'PLUSASSIGN', 'MINUSASSIGN', 'TIMESASSIGN',
            'DIVIDEASSIGN', 'MODULOASSIGN', 'POWERASSIGN',
            'FLOORDIVASSIGN', 'MATMULASSIGN',
            'LSHIFTASSIGN', 'RSHIFTASSIGN',
            'ANDASSIGN', 'ORASSIGN', 'XORASSIGN', 'WALRUS',
            # Delimiters
            'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
            'LBRACE', 'RBRACE', 'COMMA', 'COLON', 'SEMICOLON',
            'DOT', 'ELLIPSIS', 'ARROW',
            # Structural tokens
            'NEWLINE', 'WHITESPACE', 'COMMENT_START',
        ]

        for token in fixed_tokens:
            self.add_token(token)

        # Add all printable ASCII characters (for names, numbers, strings, comments)
        # This includes: letters, digits, punctuation, and space
        for char in string.printable:
            if char == '\n':  # Skip newline as we have NEWLINE token
                continue
            char_token = f'CHAR:{repr(char)[1:-1]}'  # Use repr to escape special chars
            self.add_token(char_token)

    def add_token(self, token: str) -> int:
        """Add a token to vocabulary and return its ID"""
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.token_to_id[token]

    def get_id(self, token: str) -> int:
        """Get ID for a token (returns UNK if not found)"""
        return self.token_to_id.get(token, self.special_tokens['<UNK>'])

    def get_token(self, token_id: int) -> str:
        """Get token string from ID"""
        return self.id_to_token.get(token_id, '<UNK>')

    def save(self, filepath: str):
        """Save vocabulary to file"""
        data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens,
            'next_id': self.next_id
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        vocab = cls(create_universal=False)  # Don't recreate when loading
        vocab.token_to_id = data['token_to_id']
        # Swap keys and values: token_to_id has str->int, we need int->str
        vocab.id_to_token = {v: k for k, v in data['token_to_id'].items()}
        vocab.special_tokens = data['special_tokens']
        vocab.next_id = data['next_id']
        return vocab

    def __len__(self):
        return len(self.token_to_id)


class PythonLexer:
    """Lexer that tokenizes Python code"""
    
    # Reserved keywords
    reserved = {
        'False': 'FALSE',
        'None': 'NONE',
        'True': 'TRUE',
        'and': 'AND',
        'as': 'AS',
        'assert': 'ASSERT',
        'async': 'ASYNC',
        'await': 'AWAIT',
        'break': 'BREAK',
        'class': 'CLASS',
        'continue': 'CONTINUE',
        'def': 'DEF',
        'del': 'DEL',
        'elif': 'ELIF',
        'else': 'ELSE',
        'except': 'EXCEPT',
        'finally': 'FINALLY',
        'for': 'FOR',
        'from': 'FROM',
        'global': 'GLOBAL',
        'if': 'IF',
        'import': 'IMPORT',
        'in': 'IN',
        'is': 'IS',
        'lambda': 'LAMBDA',
        'nonlocal': 'NONLOCAL',
        'not': 'NOT',
        'or': 'OR',
        'pass': 'PASS',
        'raise': 'RAISE',
        'return': 'RETURN',
        'try': 'TRY',
        'while': 'WHILE',
        'with': 'WITH',
        'yield': 'YIELD',
    }
    
    # Token list
    tokens = [
        'NAME', 'NUMBER', 'STRING',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MODULO', 'POWER',
        'FLOORDIV', 'MATMUL',
        'LSHIFT', 'RSHIFT',
        'BITOR', 'BITXOR', 'BITAND', 'BITNOT',
        'EQ', 'NE', 'LT', 'LE', 'GT', 'GE',
        'ASSIGN', 'PLUSASSIGN', 'MINUSASSIGN', 'TIMESASSIGN',
        'DIVIDEASSIGN', 'MODULOASSIGN', 'POWERASSIGN',
        'FLOORDIVASSIGN', 'MATMULASSIGN',
        'LSHIFTASSIGN', 'RSHIFTASSIGN',
        'ANDASSIGN', 'ORASSIGN', 'XORASSIGN',
        'WALRUS',
        'LPAREN', 'RPAREN',
        'LBRACKET', 'RBRACKET',
        'LBRACE', 'RBRACE',
        'COMMA', 'COLON', 'SEMICOLON',
        'DOT', 'ELLIPSIS',
        'ARROW',
        'NEWLINE', 'WHITESPACE', 'COMMENT',
    ] + list(reserved.values())
    
    # Operators (order matters for multi-char operators!)
    t_ELLIPSIS = r'\.\.\.'
    t_ARROW = r'->'
    t_WALRUS = r':='
    
    t_POWERASSIGN = r'\*\*='
    t_FLOORDIVASSIGN = r'//='
    t_LSHIFTASSIGN = r'<<='
    t_RSHIFTASSIGN = r'>>='
    t_MATMULASSIGN = r'@='
    t_PLUSASSIGN = r'\+='
    t_MINUSASSIGN = r'-='
    t_TIMESASSIGN = r'\*='
    t_DIVIDEASSIGN = r'/='
    t_MODULOASSIGN = r'%='
    t_ANDASSIGN = r'&='
    t_ORASSIGN = r'\|='
    t_XORASSIGN = r'\^='
    
    t_POWER = r'\*\*'
    t_FLOORDIV = r'//'
    t_LSHIFT = r'<<'
    t_RSHIFT = r'>>'
    t_LE = r'<='
    t_GE = r'>='
    t_EQ = r'=='
    t_NE = r'!='
    t_MATMUL = r'@'
    
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_MODULO = r'%'
    t_BITOR = r'\|'
    t_BITXOR = r'\^'
    t_BITAND = r'&'
    t_BITNOT = r'~'
    t_LT = r'<'
    t_GT = r'>'
    t_ASSIGN = r'='
    
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_COMMA = r','
    t_COLON = r':'
    t_SEMICOLON = r';'
    t_DOT = r'\.'
    
    def t_STRING(self, t):
        r'([rRfFbBuU]?["\']([^"\'\\]|\\.)*["\']|[rRfFbBuU]?"""([^\\]|\\.)*?"""|[rRfFbBuU]?\'\'\'([^\\]|\\.)*?\'\'\')'
        return t
    
    def t_NUMBER(self, t):
        r'(\d+\.?\d*([eE][+-]?\d+)?[jJ]?|0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+)'
        return t
    
    def t_NAME(self, t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        t.type = self.reserved.get(t.value, 'NAME')
        return t
    
    def t_COMMENT(self, t):
        r'\#[^\n]*'
        return t
    
    def t_NEWLINE(self, t):
        r'(\r\n|\r|\n)+'
        t.lexer.lineno += t.value.count('\n') + t.value.count('\r')
        return t
    
    def t_WHITESPACE(self, t):
        r'[ \t]+'
        return t
    
    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
        t.lexer.skip(1)
    
    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)
        return self.lexer


# Global lexer and vocabulary
_lexer = None
_vocabulary = None


def get_lexer():
    """Get or create the lexer instance"""
    global _lexer
    if _lexer is None:
        lexer_builder = PythonLexer()
        _lexer = lexer_builder.build()
    return _lexer


def get_vocabulary():
    """Get or create the vocabulary instance"""
    global _vocabulary
    if _vocabulary is None:
        _vocabulary = Vocabulary()
    return _vocabulary


def is_fixed_token(token_type: str) -> bool:
    """Check if a token type is a fixed token (keyword, operator, delimiter)"""
    fixed_types = {
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MODULO', 'POWER',
        'FLOORDIV', 'MATMUL', 'LSHIFT', 'RSHIFT',
        'BITOR', 'BITXOR', 'BITAND', 'BITNOT',
        'EQ', 'NE', 'LT', 'LE', 'GT', 'GE',
        'ASSIGN', 'PLUSASSIGN', 'MINUSASSIGN', 'TIMESASSIGN',
        'DIVIDEASSIGN', 'MODULOASSIGN', 'POWERASSIGN',
        'FLOORDIVASSIGN', 'MATMULASSIGN',
        'LSHIFTASSIGN', 'RSHIFTASSIGN',
        'ANDASSIGN', 'ORASSIGN', 'XORASSIGN', 'WALRUS',
        'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
        'LBRACE', 'RBRACE', 'COMMA', 'COLON', 'SEMICOLON',
        'DOT', 'ELLIPSIS', 'ARROW',
        'FALSE', 'NONE', 'TRUE', 'AND', 'AS', 'ASSERT',
        'ASYNC', 'AWAIT', 'BREAK', 'CLASS', 'CONTINUE',
        'DEF', 'DEL', 'ELIF', 'ELSE', 'EXCEPT', 'FINALLY',
        'FOR', 'FROM', 'GLOBAL', 'IF', 'IMPORT', 'IN', 'IS',
        'LAMBDA', 'NONLOCAL', 'NOT', 'OR', 'PASS', 'RAISE',
        'RETURN', 'TRY', 'WHILE', 'WITH', 'YIELD',
    }
    return token_type in fixed_types


def encode_char(char: str) -> str:
    """Encode a single character as a CHAR token"""
    return f'CHAR:{repr(char)[1:-1]}'


def tokenize(code: str) -> List[int]:
    """
    Tokenize Python code into a list of integer token IDs.
    Uses a universal vocabulary with character-level encoding for names, numbers, and strings.

    Args:
        code: Python source code string

    Returns:
        List of integer token IDs
    """
    lexer = get_lexer()
    vocab = get_vocabulary()
    lexer.input(code)

    token_ids = [vocab.special_tokens['<BOS>']]  # Start with BOS token

    for tok in lexer:
        if is_fixed_token(tok.type):
            # Fixed token - use token type directly
            token_id = vocab.get_id(tok.type)
            token_ids.append(token_id)
        elif tok.type == 'NEWLINE':
            # Newline is a special fixed token
            token_id = vocab.get_id('NEWLINE')
            token_ids.append(token_id)
        elif tok.type == 'WHITESPACE':
            # Encode whitespace as WHITESPACE + individual space/tab characters
            token_id = vocab.get_id('WHITESPACE')
            token_ids.append(token_id)
            for char in tok.value:
                char_token = encode_char(char)
                char_id = vocab.get_id(char_token)
                token_ids.append(char_id)
        elif tok.type == 'COMMENT':
            # Encode comment as COMMENT_START + individual characters (excluding #)
            token_id = vocab.get_id('COMMENT_START')
            token_ids.append(token_id)
            # Skip the '#' and encode the rest
            for char in tok.value[1:]:
                char_token = encode_char(char)
                char_id = vocab.get_id(char_token)
                token_ids.append(char_id)
        elif tok.type in ['NAME', 'NUMBER', 'STRING']:
            # Decompose into individual characters
            for char in tok.value:
                char_token = encode_char(char)
                char_id = vocab.get_id(char_token)
                token_ids.append(char_id)
        else:
            # Unknown token type - shouldn't happen but handle gracefully
            for char in tok.value:
                char_token = encode_char(char)
                char_id = vocab.get_id(char_token)
                token_ids.append(char_id)

    token_ids.append(vocab.special_tokens['<EOS>'])  # End with EOS token

    return token_ids


def untokenize(token_ids: List[int]) -> str:
    """
    Convert a list of token IDs back to Python code.
    Reconstructs code from the universal vocabulary with character-level encoding.

    Args:
        token_ids: List of integer token IDs from tokenize()

    Returns:
        Original Python source code string
    """
    vocab = get_vocabulary()

    # Mapping from token types to their string values
    type_to_value = {
        'PLUS': '+', 'MINUS': '-', 'TIMES': '*', 'DIVIDE': '/',
        'MODULO': '%', 'POWER': '**', 'FLOORDIV': '//', 'MATMUL': '@',
        'LSHIFT': '<<', 'RSHIFT': '>>', 'BITOR': '|', 'BITXOR': '^',
        'BITAND': '&', 'BITNOT': '~',
        'EQ': '==', 'NE': '!=', 'LT': '<', 'LE': '<=', 'GT': '>', 'GE': '>=',
        'ASSIGN': '=', 'PLUSASSIGN': '+=', 'MINUSASSIGN': '-=',
        'TIMESASSIGN': '*=', 'DIVIDEASSIGN': '/=', 'MODULOASSIGN': '%=',
        'POWERASSIGN': '**=', 'FLOORDIVASSIGN': '//=', 'MATMULASSIGN': '@=',
        'LSHIFTASSIGN': '<<=', 'RSHIFTASSIGN': '>>=',
        'ANDASSIGN': '&=', 'ORASSIGN': '|=', 'XORASSIGN': '^=',
        'WALRUS': ':=',
        'LPAREN': '(', 'RPAREN': ')', 'LBRACKET': '[', 'RBRACKET': ']',
        'LBRACE': '{', 'RBRACE': '}', 'COMMA': ',', 'COLON': ':',
        'SEMICOLON': ';', 'DOT': '.', 'ELLIPSIS': '...', 'ARROW': '->',
        'FALSE': 'False', 'NONE': 'None', 'TRUE': 'True',
        'AND': 'and', 'AS': 'as', 'ASSERT': 'assert', 'ASYNC': 'async',
        'AWAIT': 'await', 'BREAK': 'break', 'CLASS': 'class',
        'CONTINUE': 'continue', 'DEF': 'def', 'DEL': 'del', 'ELIF': 'elif',
        'ELSE': 'else', 'EXCEPT': 'except', 'FINALLY': 'finally',
        'FOR': 'for', 'FROM': 'from', 'GLOBAL': 'global', 'IF': 'if',
        'IMPORT': 'import', 'IN': 'in', 'IS': 'is', 'LAMBDA': 'lambda',
        'NONLOCAL': 'nonlocal', 'NOT': 'not', 'OR': 'or', 'PASS': 'pass',
        'RAISE': 'raise', 'RETURN': 'return', 'TRY': 'try',
        'WHILE': 'while', 'WITH': 'with', 'YIELD': 'yield',
        'NEWLINE': '\n', 'COMMENT_START': '#',
    }

    code_parts = []
    i = 0
    while i < len(token_ids):
        token_id = token_ids[i]
        token_str = vocab.get_token(token_id)

        # Skip special tokens (BOS, EOS, PAD, UNK)
        if token_str in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
            i += 1
            continue

        # Handle character tokens
        if token_str.startswith('CHAR:'):
            # Extract the character
            char_repr = token_str[5:]  # Remove 'CHAR:' prefix
            # Unescape the character representation
            try:
                char = eval(f"'{char_repr}'")
                code_parts.append(char)
            except:
                # If eval fails, just use the string as-is
                code_parts.append(char_repr)
            i += 1
        # Handle WHITESPACE token (followed by character tokens)
        elif token_str == 'WHITESPACE':
            # Consume following character tokens
            i += 1
            while i < len(token_ids):
                next_token_str = vocab.get_token(token_ids[i])
                if next_token_str.startswith('CHAR:'):
                    char_repr = next_token_str[5:]
                    try:
                        char = eval(f"'{char_repr}'")
                        code_parts.append(char)
                    except:
                        code_parts.append(char_repr)
                    i += 1
                else:
                    break
        # Handle fixed tokens
        elif token_str in type_to_value:
            code_parts.append(type_to_value[token_str])
            i += 1
        else:
            # Unknown token - skip
            i += 1

    return ''.join(code_parts)


def get_vocab_size() -> int:
    """Get the current vocabulary size"""
    return len(get_vocabulary())


def save_vocabulary(filepath: str):
    """Save the vocabulary to a file"""
    get_vocabulary().save(filepath)


def load_vocabulary(filepath: str):
    """Load a vocabulary from a file"""
    global _vocabulary
    _vocabulary = Vocabulary.load(filepath)


# Example usage and tests
if __name__ == '__main__':
    test_code = """
def hello(name: str) -> None:
    print(f"Hello {name}")
"""

    print("Test code:")
    print(test_code)

    tokens = tokenize(test_code)
    print("Tokens:")
    # print(tokens)
    for i, token in enumerate(tokens, 1):
        print(f"{i:-3}. {token} -> {_vocabulary.get_token(token)}")
