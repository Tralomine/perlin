all:
	g++ -c -fPIC perlin.cpp -o libperlin.o
	g++ libperlin.o -shared -o libperlin.so
	g++ main.cpp libperlin.so -lsfml-graphics -lsfml-window -lsfml-system -o perlin -Wl,-rpath='$${ORIGIN}' -lpthread

debug:
	nvcc -g main.cpp perlin.cpp -lsfml-graphics -lsfml-system -lsfml-window -o perlin
	gdb ./perlin

clean:
	rm libperlin.o
