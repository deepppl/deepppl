all: stanLexer.py stanParser.py

stanLexer.py stanParser.py: stan.g4
	antlr4 -Dlanguage=Python3 stan.g4

clean:
	-rm -f	stan.tokens stanLexer.tokens \
		stanLexer.py stanParser.py stanListener.py
	-rm -rf __pycache__

cleanall: clean
	-rm -f *~
