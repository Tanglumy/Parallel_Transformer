CC=/opt/homebrew/opt/llvm/bin/clang++
FLAGS=-g -Wall -std=c++17 -arch arm64 -lomp
LDFLAGS=-L/opt/homebrew/Cellar/libomp/19.1.5/lib
CPPFLAGS=-I/opt/homebrew/Cellar/libomp/19.1.5/include

transformer: ./MPI_transformer/transformer.cpp
	$(CC) -o MPI_transformer ./MPI_transformer/transformer.cpp $(FLAGS) $(LDFLAGS) $(CPPFLAGS)