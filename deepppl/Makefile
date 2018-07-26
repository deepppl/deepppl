all: parser/stanLexer.py parser/stanParser.py parser/dppplLexer.py parser/dppplParser.py

parser/dppplLexer.py parser/dppplParser.py: parser/dpppl.g4
	antlr4 -Dlanguage=Python3 parser/dpppl.g4

parser/stanLexer.py parser/stanParser.py: parser/stan.g4
	antlr4 -Dlanguage=Python3 parser/stan.g4

test:
	python test.py tests/good/coin.stan

clean:
	-rm -f	parser/stan.tokens parser/stanLexer.tokens \
		parser/stanLexer.py parser/stanParser.py \
		parser/stanListener.py \
		parser/stan.interp parser/stanLexer.interp

	-rm -f	parser/dpppl.tokens parser/dppplLexer.tokens \
		parser/dppplLexer.py parser/dppplParser.py \
		parser/dppplListener.py \
		parser/dpppl.interp parser/dppplLexer.interp

	-rm -rf __pycache__

cleanall: clean
	-rm -f *~ */*~
