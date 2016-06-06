all:
	nvcc transpose.cu -o transpose
test:
	./transpose 2 16 16 0 1
	./transpose 2 16 16 0 2
	./transpose 2 16 16 1 2
	./transpose 2 1 16 0 1
	./transpose 2 1 16 0 2
	./transpose 2 1 16 1 2
	./transpose 1 1 1 0 1
	./transpose 1 1 1 0 2
	./transpose 1 1 1 1 2
	./transpose 256 32 16 0 1
	./transpose 256 32 16 0 2
	./transpose 256 32 16 1 2
	./transpose 286 372 167 0 1
	./transpose 286 372 167 0 2
	./transpose 286 372 167 1 2
clean:
	rm -f transpose
