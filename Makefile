all:
	nvcc transpose.cu -o transpose
test:
		
#	./transpose -d 3 -s 64 64 64 -sf 0 1 2
#	./transpose -d 3 -s 64 64 64 -sf 0 2 1
#	./transpose -d 3 -s 64 64 64 -sf 1 0 2
#	./transpose -d 3 -s 64 64 64 -sf 1 2 0
#	./transpose -d 3 -s 64 64 64 -sf 2 0 1
#	./transpose -d 3 -s 64 64 64 -sf 2 1 0
#	./transpose -d 4 -s 100 100 100 100 -sf 0 3 2 1 
#	./transpose -d 4 -s 100 100 100 100 -sf 3 2 1 0 
#	./transpose -d 4 -s 100 100 100 100 -sf 2 1 3 0
	./transpose -d 4 -s 100 100 100 100 -sf 3 0 2 1
	./transpose -d 4 -s 100 100 100 100 -sf 0 1 3 2 
	./transpose -d 4 -s 100 100 100 100 -sf 1 3 0 2
#	./transpose -d 2 -s 96 32 -p 0 1
#	./transpose -d 4 -s 64 32 64 32 -p 0 1
#	./transpose -d 4 -s 64 32 64 32 -p 0 2
#	./transpose -d 4 -s 64 32 64 32 -p 0 3
#	./transpose -d 4 -s 64 32 64 32 -p 1 2
#	./transpose -d 4 -s 64 32 64 32 -p 1 3
#	./transpose -d 4 -s 64 32 64 32 -p 2 3
#	./transpose -d 3 -s 2 16 16 -p 0 1
#	./transpose -d 3 -s 2 16 16 -p 0 2
#	./transpose -d 3 -s 2 16 16 -p 1 2
#	./transpose -d 3 -s 2 1 16 -p 0 1
#	./transpose -d 3 -s 2 1 16 -p 0 2
#	./transpose -d 3 -s 2 1 16 -p 1 2
#	./transpose -d 3 -s 1 1 1 -p 0 1
#	./transpose -d 3 -s 1 1 1 -p 0 2
#	./transpose -d 3 -s 1 1 1 -p 1 2
#	./transpose -d 3 -s 256 32 16 -p 0 1
#	./transpose -d 3 -s 256 32 16 -p 0 2
#	./transpose -d 3 -s 256 32 16 -p 1 2
#	./transpose -d 3 -s 286 372 167 -p 0 1
#	./transpose -d 3 -s 286 372 167 -p 0 2
#	./transpose -d 3 -s 286 372 167 -p 1 2
clean:
	rm -f transpose
