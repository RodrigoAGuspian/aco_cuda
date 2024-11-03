#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <limits>
#include <vector>
#include <cmath>
#include <chrono>

#define NUM_HORMIGAS 5
#define NUM_CIUDADES 5
#define ALPHA 1.0f
#define BETA 5.0f
#define EVAPORACION 0.5f
#define FEROMONA_INICIAL 1.0f
#define ITERACIONES 100
#define COSTO_PENALIZADO 100000.0f

__device__ float distancias[NUM_CIUDADES][NUM_CIUDADES] = {
    {0, 3, 9, 2, 5}, 
    {3, 0, 4, 8, 2}, 
    {9, 4, 0, 8, 1}, 
    {2, 8, 8, 0, 4}, 
    {5, 2, 1, 4, 0}  
};

__device__ float feromonas[NUM_CIUDADES][NUM_CIUDADES];

__global__ void inicializarFeromonas() {
    int i = threadIdx.x;
    int j = threadIdx.y;
    if (i < NUM_CIUDADES && j < NUM_CIUDADES) {
        feromonas[i][j] = FEROMONA_INICIAL;
    }
}

__device__ float calcularProbabilidad(int ciudad_actual, int ciudad_siguiente, bool* visitado) {
    if (visitado[ciudad_siguiente] || distancias[ciudad_actual][ciudad_siguiente] == 0) return 0.0f;
    return powf(feromonas[ciudad_actual][ciudad_siguiente], ALPHA) *
           powf(1.0f / distancias[ciudad_actual][ciudad_siguiente], BETA);
}

__global__ void construirSoluciones(int* rutas, float* costos, curandState* states, float* probabilidades_host) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= NUM_HORMIGAS) return;

    bool visitado[NUM_CIUDADES] = {false};
    int ruta[NUM_CIUDADES];
    float costo = 0.0f;

    curandState localState = states[id];
    int ciudad_actual = curand(&localState) % NUM_CIUDADES;
    ruta[0] = ciudad_actual;
    visitado[ciudad_actual] = true;

    for (int step = 1; step < NUM_CIUDADES; ++step) {
        float total_probabilidad = 0.0f;

        for (int j = 0; j < NUM_CIUDADES; ++j) {
            float prob = calcularProbabilidad(ciudad_actual, j, visitado);
            total_probabilidad += prob;
            probabilidades_host[id * NUM_CIUDADES + j] = prob;
        }

        float prob_seleccion = curand_uniform(&localState) * total_probabilidad;
        float suma_probabilidad = 0.0f;
        int siguiente_ciudad = -1;
        for (int j = 0; j < NUM_CIUDADES; ++j) {
            suma_probabilidad += probabilidades_host[id * NUM_CIUDADES + j];
            if (suma_probabilidad >= prob_seleccion) {
                siguiente_ciudad = j;
                break;
            }
        }

        ruta[step] = siguiente_ciudad;
        visitado[siguiente_ciudad] = true;
        costo += distancias[ciudad_actual][siguiente_ciudad];
        ciudad_actual = siguiente_ciudad;
    }
    costo += distancias[ciudad_actual][ruta[0]];

    if (costo == 0) {
        costo = COSTO_PENALIZADO;
    }

    for (int i = 0; i < NUM_CIUDADES; ++i) {
        rutas[id * NUM_CIUDADES + i] = ruta[i];
    }
    costos[id] = costo;
    states[id] = localState;
}

__global__ void actualizarFeromonas(int* rutas, float* costos) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    if (i < NUM_CIUDADES && j < NUM_CIUDADES) {
        feromonas[i][j] *= (1.0f - EVAPORACION);
    }

    __syncthreads();

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < NUM_HORMIGAS) {
        int* ruta = &rutas[id * NUM_CIUDADES];
        float costo = costos[id];
        if (costo != COSTO_PENALIZADO) {
            for (int k = 0; k < NUM_CIUDADES - 1; ++k) {
                int a = ruta[k];
                int b = ruta[k + 1];
                atomicAdd(&feromonas[a][b], 1.0f / costo);
                atomicAdd(&feromonas[b][a], 1.0f / costo);
            }
        }
    }
}

__global__ void inicializarEstados(curandState* states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &states[id]);
}

void imprimirInformacion(int* rutas, float* costos, float* probabilidades, int num_hormigas) {
    for (int i = 0; i < num_hormigas; ++i) {
        std::cout << "Hormiga " << i << ": Ruta = ";
        for (int j = 0; j < NUM_CIUDADES; ++j) {
            std::cout << rutas[i * NUM_CIUDADES + j] << " ";
        }
        std::cout << "| Costo = " << costos[i];

        std::cout << "| Probabilidades: ";
        for (int j = 0; j < NUM_CIUDADES; ++j) {
            std::cout << probabilidades[i * NUM_CIUDADES + j] << " ";
        }
        std::cout << std::endl;
    }
}


int main() {
    int* d_rutas;
    float* d_costos;
    curandState* d_states;
    float* d_probabilidades;

    cudaMalloc(&d_rutas, NUM_HORMIGAS * NUM_CIUDADES * sizeof(int));
    cudaMalloc(&d_costos, NUM_HORMIGAS * sizeof(float));
    cudaMalloc(&d_states, NUM_HORMIGAS * sizeof(curandState));
    cudaMalloc(&d_probabilidades, NUM_HORMIGAS * NUM_CIUDADES * sizeof(float));

    inicializarFeromonas<<<1, dim3(NUM_CIUDADES, NUM_CIUDADES)>>>();
    inicializarEstados<<<NUM_HORMIGAS / 32 + 1, 32>>>(d_states);

    int* h_rutas = new int[NUM_HORMIGAS * NUM_CIUDADES];
    float* h_costos = new float[NUM_HORMIGAS];
    float* h_probabilidades = new float[NUM_HORMIGAS * NUM_CIUDADES];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int iter = 0; iter < ITERACIONES; ++iter) {
        construirSoluciones<<<NUM_HORMIGAS / 32 + 1, 32>>>(d_rutas, d_costos, d_states, d_probabilidades);
        actualizarFeromonas<<<NUM_HORMIGAS / 32 + 1, 32>>>(d_rutas, d_costos);

        cudaMemcpy(h_rutas, d_rutas, NUM_HORMIGAS * NUM_CIUDADES * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_costos, d_costos, NUM_HORMIGAS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_probabilidades, d_probabilidades, NUM_HORMIGAS * NUM_CIUDADES * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Iteración " << iter + 1 << std::endl;
        imprimirInformacion(h_rutas, h_costos, h_probabilidades, NUM_HORMIGAS);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiempo total de ejecución: " << elapsedTime << " ms" << std::endl;

    cudaFree(d_rutas);
    cudaFree(d_costos);
    cudaFree(d_states);
    cudaFree(d_probabilidades);
    delete[] h_rutas;
    delete[] h_costos;
    delete[] h_probabilidades;

    std::cout << "Algoritmo ACO en CUDA completado." << std::endl;
    return 0;
}