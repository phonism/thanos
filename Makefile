.PHONY: lib, pybind, clean, format, all

all: lib


lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

format:
	python -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/thanos/backend_ndarray/ndarray_backend*.so
