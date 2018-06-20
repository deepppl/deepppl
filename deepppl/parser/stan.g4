/*
 * Copyright 2018 IBM Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

grammar stan;

/** Includes (section 2.2) */

// XXX TODO: #include file.stan XXX


/** Comments (section 2.3) */

COMMENT
    :   '/*' .*? '*/' -> channel(3) // COMMENTS
    ;

LINE_COMMENT
    :   '//' ~[\r\n]* -> channel(3) //COMMENTS
    ;


/** Whitespaces (section 2.4) */

WS  :  [ \t\r\n\u000C]+ -> channel(1) // WHITESPACES
    ;

/** Numeric literals (section 4.1) */

IntegerLiteral
    :  DecimalNumeral
    ;

RealLiteral:
      Digits '.' Digits? ExponentPart?
    | '.' Digits ExponentPart?
    | Digits ExponentPart
    ;

fragment
ExponentPart:
      ('e'|'E') ('+'|'-')? Digits
    ;

fragment
DecimalNumeral
    :   '0'
    |   NonZeroDigit Digits?
    ;

fragment
Digits
    :   Digit+
    ;

fragment
Digit
    :   '0'
    |   NonZeroDigit
    ;

fragment
NonZeroDigit
    :   [1-9]
    ;

StringLiteral
    : '"' ([a-z]|[A-Z]|[0-9]|Symbol)* '"'
    ;

fragment
Symbol
    : '~'
    | '@'
    | '#'
    | '$'
    | '%'
    | '^'
    | '&'
    | '*'
    | '_'
    | '\''
    | '`'
    | '-'
    | '+'
    | '='
    | '{'
    | ' '
    | '}'
    | '['
    | ']'
    | '('
    | ')'
    | '<'
    | '>'
    | '|'
    | '/'
    | '!'
    | '?'
    | '.'
    | ','
    | ';'
    | ':'
    ;

/** Variables and keywords (section 4.2) */

/* Keywords */
FOR: 'for';
IN: 'in';
WHILE: 'while';
REPEAT: 'repeat';
UNTIL: 'until';
IF: 'if';
THEN: 'then';
ELSE: 'else';
TRUE: 'true';
FALSE: 'false';

/* Type names */
INT: 'int';
REAL: 'real';
VECTOR: 'vector';
SIMPLEX: 'simplex';
ORDERED: 'ordered';
POSITIVE_ORDERED: 'positive_ordered';
ROW_VECTOR: 'row_vector';
UNIT_VECTOR: 'unit_vector';
MATRIX: 'matrix';
CHOLESKY_FACTOR_CORR: 'cholesky_factor_corr';
CHOLESKY_FACTOR_COV: 'cholesky_factor_cov';
CORR_MATRIX: 'corr_matrix';
COV_MATRIX: 'cov_matrix';

/* Block identifiers */
FUNCTIONS: 'functions';
MODEL: 'model';
DATA: 'data';
NETWORK: 'network';
PRIOR: 'prior';
GUIDE: 'guide';
PARAMETERS: 'parameters';
QUANTITIES: 'quantities';
TRANSFORMED: 'transformed';
GENERATED: 'generated';

/* Reserved Names from Stan Implementation */
VAR: 'var';
FVAR: 'fvar';
STAN_MAJOR: 'STAN_MAJOR';
STAN_MINOR: 'STAN_MINOR';
STAN_PATCH: 'STAN_PATCH';
STAN_MATH_MAJOR: 'STAN_MATH_MAJOR';
STAN_MATH_MINOR: 'STAN_MATH_MINOR';
STAN_MATH_PATCH: 'STAN_MATH_PATCH';

/* Reserved Names from C++ */
ALIGNAS: 'alignas';
ALIGNOF: 'alignof';
AND: 'and';
AND_EQ: 'and_eq';
ASM: 'asm';
AUTO: 'auto';
BITAND: 'bitand';
BITOR: 'bitor';
BOOL: 'bool';
BREAK: 'break';
CASE: 'case';
CATCH: 'catch';
CHAR: 'char';
CHAR16_T: 'char16_t';
CHAR32_T: 'char32_t';
CLASS: 'class';
COMPL: 'compl';
CONST: 'const';
CONSTEXPR: 'constexpr';
CONST_CAST: 'const_cast';
CONTINUE: 'continue';
DECLTYPE: 'decltype';
DEFAULT: 'default';
DELETE: 'delete';
DO: 'do';
DOUBLE: 'double';
DYNAMIC_CAST: 'dynamic_cast';
ENUM: 'enum';
EXPLICIT: 'explicit';
EXPORT: 'export';
EXTERN: 'extern';
FLOAT: 'float';
FRIEND: 'friend';
GOTO: 'goto';
INLINE: 'inline';
LONG: 'long';
MUTABLE: 'mutable';
NAMESPACE: 'namespace';
NEW: 'new';
NOEXCEPT: 'noexcept';
NOT: 'not';
NOT_EQ: 'not_eq';
NULLPTR: 'nullptr';
OPERATOR: 'operator';
OR: 'or';
OR_EQ: 'or_eq';
PRIVATE: 'private';
PROTECTED: 'protected';
PUBLIC: 'public';
REGISTER: 'register';
REINTERPRET_CAST: 'reinterpret_cast';
RETURN: 'return';
SHORT: 'short';
SIGNED: 'signed';
SIZEOF: 'sizeof';
STATIC: 'static';
STATIC_ASSERT: 'static_assert';
STATIC_CAST: 'static_cast';
STRUCT: 'struct';
SWITCH: 'switch';
TEMPLATE: 'template';
THIS: 'this';
THREAD_LOCAL: 'thread_local';
THROW: 'throw';
TRY: 'try';
TYPEDEF: 'typedef';
TYPEID: 'typeid';
TYPENAME: 'typename';
UNION: 'union';
UNSIGNED: 'unsigned';
USING: 'using';
VIRTUAL: 'virtual';
VOID: 'void';
VOLATILE: 'volatile';
WCHAR_T: 'wchar_t';
XOR: 'xor';
XOR_EQ: 'xor_eq';


/* Variables */

BAD_IDENTIFIER
    : [a-zA-Z] [a-zA-Z0-9_]* '__'
    ;

IDENTIFIER
    : [a-zA-Z] [a-zA-Z0-9_]*
    ;


/* Operators (figure 4.1) */

OR_OP: '||';
AND_OP: '&&';
EQ_OP: '==';
NEQ_OP: '!=';
LT_OP: '<';
LE_OP: '<=';
GT_OP: '>';
GE_OP: '>=';
PLUS_OP: '+';
MINUS_OP: '-';
MULT_OP: '*';
DIV_OP: '/';
MOD_OP: '%';
LEFT_DIV_OP: '\\';
DOT_MULT_OP: '.*';
DOT_DIV_OP: './';
NOT_OP: '!';
POW_OP: '^';
TRANSPOSE_OP: '\'';

PLUS_EQ: '+=';
MINUS_EQ: '-=';
MULT_EQ: '*=';
DIV_EQ: '/=';
DOT_MULT_EQ: '.*=';
DOT_DIV_EQ: './=';


/* Types (section 3.1) */

primitiveType
    : REAL
    | INT
    ;

vectorType
    : VECTOR
    | SIMPLEX
    | UNIT_VECTOR
    | ORDERED
    | POSITIVE_ORDERED
    | ROW_VECTOR
    ;

matrixType
    : MATRIX
    | CORR_MATRIX
    | COV_MATRIX
    | CHOLESKY_FACTOR_COV
    | CHOLESKY_FACTOR_CORR
    ;

type_
    : primitiveType typeConstraints? ('[' ']')?
    | vectorType typeConstraints? ('[' expression ']')?
    | matrixType typeConstraints? ('[' expression (',' expression)? ']')?
    ;

typeConstraints
    : '<' typeConstraintList '>'
    ;

typeConstraintList
    :  typeConstraint (',' typeConstraint)*
    ;

typeConstraint
    : IDENTIFIER '=' atom
    ;

variableDecl
    : type_ IDENTIFIER arrayDim? ';'
    | type_ IDENTIFIER arrayDim? '=' expression ';'
    ;

netVariableDecl
    : IDENTIFIER IDENTIFIER ';'
    ;

netLValue
    : netName ('.' IDENTIFIER)*
    ;

netName
    : IDENTIFIER
    ;

arrayDim
    : '[' expressionCommaList ']'
    ;

variableDeclsOpt
    : variableDecl*
    ;

netVariableDeclsOpt
    : netVariableDecl*
    ;

/** Numeric Litterals (section 4.1) */
constant
    : IntegerLiteral
    | RealLiteral
    ;

/** Variable (section 4.2) */
variable
    : IDENTIFIER
    ;

/** Vector, matrix and array expressions (section 4.2) */

vectorExpr
    : '[' expressionCommaList ']'
    ;

arrayExpr
    : '{' expressionCommaList '}'
    ;

atom
    : constant
    | variable
    | vectorExpr
    | arrayExpr
    | atom '[' indexExpression ']'
    | '(' expression ')'
    ;

callExpr
    : IDENTIFIER '(' expressionOrStringCommaList ')'
    ;

expression
    : atom
    | callExpr
    | expression TRANSPOSE_OP
    | <assoc=right> e1=expression POW_OP e2=expression
    | op=(NOT_OP|PLUS_OP|MINUS_OP) expression
    | e1=expression op=(DOT_MULT_OP|DOT_DIV_OP) e2=expression
    | e1=expression LEFT_DIV_OP e2=expression
    | e1=expression op=(MULT_OP|DIV_OP|MOD_OP) e2=expression
    | e1=expression op=(PLUS_OP|MINUS_OP) e2=expression
    | e1=expression op=(LT_OP|LE_OP|GT_OP|GE_OP) e2=expression
    | e1=expression op=(EQ_OP|NEQ_OP) e2=expression
    | e1=expression AND_OP e2=expression
    | e1=expression OR_OP e2=expression
    | <assoc=right> e1=expression '?' e2=expression ':' e3=expression
    ;

indexExpression
    : expressionCommaListOpt
    | e1=expression? ':' e2=expression?
    ;

expressionCommaList
    : expression (',' expression)*
    ;

expressionCommaListOpt
    : expressionCommaList?
    ;


/** Statements (section 5) */

/** Assignment (section 5.1) */

lvalue
    : IDENTIFIER
    | IDENTIFIER '[' expressionCommaList ']'
    ;
assignStmt
    : lvalue '=' expression ';'
    | lvalue op=(PLUS_EQ|MINUS_EQ|MULT_EQ|DIV_EQ|DOT_MULT_EQ|DOT_DIV_EQ) expression ';'
    ;

/** Sampling (section 5.3) */

lvalueSampling
    : lvalue
    | expression
    | netLValue
    ;

samplingStmt
    : lvalueSampling '~' IDENTIFIER '(' expressionCommaList ')' truncation? ';'
    | lvalueSampling PLUS_EQ IDENTIFIER '(' IDENTIFIER '|' expressionCommaList ')' ';'
    ;

truncation
    : 'T' '['  e1=expression? ',' e1=expression? ']'
    ;

/** For loops (section 5.4) */

forStmt
    : FOR '(' IDENTIFIER IN atom ':' atom ')' statement
    | FOR '(' IDENTIFIER IN atom ')' statement
    ;


/** Conditional statements (section 5.5) */

conditionalStmt
    : IF '(' expression ')' s1=statement (ELSE s2=statement)?
    ;


/** While loops (section 5.6) */

whileStmt
    : WHILE '(' expression ')' statement
    ;


/** Blocks (section 5.7) */
blockStmt
    : '{' variableDeclsOpt statementsOpt '}'
    ;


/** Functions calls (sections 5.9 and 5.10) */

callStmt
    : callExpr ';'
    ;

expressionOrString
    : expression
    | StringLiteral
    ;

expressionOrStringCommaList:
    | expressionOrString (',' expressionOrString)*
    ;

/** statements */

statement
    : assignStmt
    | samplingStmt
    | forStmt
    | conditionalStmt
    | whileStmt
    | blockStmt
    | callStmt
    | BREAK ';'
    | CONTINUE ';'
    ;

statementsOpt
    : statement*
    ;


/** Functions (section 7) */

functionType
    : type_ IDENTIFIER '(' parameterCommaListopt ')'
    | VOID IDENTIFIER '(' parameterCommaListopt ')'
    ;

parameterDecl
    : type_ IDENTIFIER
    ;

parameterCommaList
    : parameterDecl (',' parameterDecl)*
    ;

parameterCommaListopt
    : parameterCommaList?
    ;

functionStatement
    : statement
    | RETURN expression ';'
    | RETURN ';'
    ;

functionStatementsOpt
    : functionStatement*
    ;

functionDecl
    : functionType '{' variableDeclsOpt functionStatementsOpt '}'
    | functionType ';'
    ;

functionDeclsOpt
    : functionDecl*
    ;

/** Program blocks (section 6) */

functionBlock
    : FUNCTIONS '{'  functionDeclsOpt '}'
    ;

dataBlock
    : DATA '{' variableDeclsOpt '}'
    ;

networkBlock
    : NETWORK '{' netVariableDeclsOpt '}'
    ;

priorBlock
    : PRIOR '{' samplingStmt* '}'
    ;

guideBlock
    : GUIDE '{' variableDeclsOpt statementsOpt '}'
    ;

transformedDataBlock
    : TRANSFORMED DATA '{' variableDeclsOpt statementsOpt '}'
    ;

parametersBlock
    : PARAMETERS '{' variableDeclsOpt '}'
    ;

transformedParametersBlock
    : TRANSFORMED PARAMETERS '{' variableDeclsOpt statementsOpt '}'
    ;

modelBlock
    : MODEL '{' variableDeclsOpt statementsOpt '}'
    ;

generatedQuantitiesBlock
    : GENERATED QUANTITIES '{' variableDeclsOpt statementsOpt '}'
    ;

program
    : functionBlock?
        dataBlock?
        transformedDataBlock?
        parametersBlock?
        transformedParametersBlock?
        networkBlock?
        priorBlock?
        guideBlock?
        modelBlock?
        generatedQuantitiesBlock?
    ;
