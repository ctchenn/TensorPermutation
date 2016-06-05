all:
	nvcc transpose.cu -o transpose
	./transpose
clean:
	rm -f transpose
