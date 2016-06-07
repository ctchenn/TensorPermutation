all:
	nvcc transpose.cu -o transpose
test:
	./transpose -d 2 -s 96 32 -p 0 1
	./transpose -d 3 -s 2 16 16 -p 0 1
	./transpose -d 3 -s 2 16 16 -p 0 2
	./transpose -d 3 -s 2 16 16 -p 1 2
	./transpose -d 3 -s 2 1 16 -p 0 1
	./transpose -d 3 -s 2 1 16 -p 0 2
	./transpose -d 3 -s 2 1 16 -p 1 2
	./transpose -d 3 -s 1 1 1 -p 0 1
	./transpose -d 3 -s 1 1 1 -p 0 2
	./transpose -d 3 -s 1 1 1 -p 1 2
	./transpose -d 3 -s 256 32 16 -p 0 1
	./transpose -d 3 -s 256 32 16 -p 0 2
	./transpose -d 3 -s 256 32 16 -p 1 2
	./transpose -d 3 -s 286 372 167 -p 0 1
	./transpose -d 3 -s 286 372 167 -p 0 2
	./transpose -d 3 -s 286 372 167 -p 1 2
clean:
	rm -f transpose
