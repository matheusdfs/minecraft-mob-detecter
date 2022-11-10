CC = g++
PROJECT = output
SRC = main.c
LIBS = `pkg-config --cflags --libs opencv4`
VER = -std=c++14
$(PROJECT) : $(SRC)
	$(CC) $(VER) $(SRC) -o $(PROJECT) $(LIBS)