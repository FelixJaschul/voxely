all: build run

allc: clean build run

build:
	mkdir -p cmake-build-debug
	cd cmake-build-debug && cmake .. && make

run:
	cmake-build-debug/voxely

clean:
	rm -rf cmake-build-debug
