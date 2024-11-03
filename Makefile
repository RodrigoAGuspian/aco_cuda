# Nombre del compilador CUDA
NVCC = nvcc

# Nombre del archivo fuente
SRC = aco.cu

# Nombre del ejecutable
EXE = aco

# Regla por defecto
all: $(EXE)

# Regla para compilar el ejecutable
$(EXE): $(SRC)
	$(NVCC) $(SRC) -o $(EXE)

# Regla para limpiar los archivos generados
clean:
	rm -f $(EXE)
