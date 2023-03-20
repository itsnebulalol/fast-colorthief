
#include "constants.hpp"

#include <iostream>
#include <chrono>



__global__ void preprocess_image(uint8_t* data, int* histo, uint8_t* max_color_array, uint8_t* min_color_array,
                                 uint8_t* pixel_found_array, int* start_indices, int* stop_indices) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int pixel_index=start_indices[idx]; pixel_index<stop_indices[idx]; ++pixel_index) {

        bool alpha_valid = data[pixel_index * 4 + 3] >= 125;
        bool not_white = data[pixel_index * 4] <= 250 || data[pixel_index * 4 + 1] <= 250 || data[pixel_index * 4 + 2] <= 250;
        bool mask = alpha_valid && not_white;

        int histo_pixel_index = 0;

        for (int color_index=0; color_index<3; ++color_index) {
            uint8_t color_value = data[pixel_index * 4 + color_index] >> RSHIFT;

            max_color_array[idx * 3 + color_index] = max(max_color_array[idx * 3 + color_index], color_value * mask);
            min_color_array[idx * 3 + color_index] = min(min_color_array[idx * 3 + color_index], color_value * (1 - mask));

            histo_pixel_index += color_value << ((2 - color_index) * SIGBITS);
        }
        atomicAdd(histo + histo_pixel_index, int(mask));
        pixel_found_array[idx] = pixel_found_array[idx] || mask;
    }
}


std::tuple<std::vector<int>, color_t, color_t, bool> get_histo_cuda(uint8_t* data, int pixel_count, int quality) {
    std::vector<int> histo(std::pow(2, 3 * SIGBITS), 0);

    int num_threads = NUM_BLOCKS * THREADS_PER_BLOCK;
    int data_per_thread = std::ceil(double(pixel_count) / double(num_threads));

    std::vector<uint8_t> max_color_array(num_threads * 3, 0);
    std::vector<uint8_t> min_color_array(num_threads * 3, 0);
    std::vector<uint8_t> pixel_found_array(num_threads, 0);
    std::vector<int> start_indices;
    std::vector<int> stop_indices;

    start_indices.reserve(num_threads);
    stop_indices.reserve(num_threads);
    for (int i=0; i<num_threads; ++i) {
      start_indices.push_back(std::min(i * data_per_thread, pixel_count));
      stop_indices.push_back(std::min((i + 1) * data_per_thread, pixel_count));
    }

    if (start_indices.size() != num_threads) {
        std::cout << num_threads << " " << start_indices.size() << std::endl;
        throw std::runtime_error("Bug in preparation of data for cuda");
    }

    uint8_t *cuda_data, *cuda_max_color_array, *cuda_min_color_array;
    int *cuda_histo, *cuda_start_indices, *cuda_stop_indices;
    uint8_t *cuda_pixel_found_array;

    cudaMalloc(&cuda_data, pixel_count * 4);
    cudaMalloc(&cuda_max_color_array, max_color_array.size());
    cudaMalloc(&cuda_min_color_array, min_color_array.size());
    cudaMalloc(&cuda_histo, histo.size() * sizeof(int));
    cudaMalloc(&cuda_start_indices, start_indices.size() * sizeof(int));
    cudaMalloc(&cuda_stop_indices, stop_indices.size() * sizeof(int));
    cudaMalloc(&cuda_pixel_found_array, pixel_found_array.size() * sizeof(uint8_t));

    cudaMemcpyAsync(cuda_data, data, pixel_count * 4, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(cuda_max_color_array, max_color_array.data(), max_color_array.size(), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(cuda_min_color_array, min_color_array.data(), min_color_array.size(), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(cuda_histo, histo.data(), histo.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(cuda_start_indices, start_indices.data(), start_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(cuda_stop_indices, stop_indices.data(), stop_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(cuda_pixel_found_array, pixel_found_array.data(), pixel_found_array.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

    preprocess_image<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cuda_data, cuda_histo, cuda_max_color_array, cuda_min_color_array,
                                                        cuda_pixel_found_array, cuda_start_indices, cuda_stop_indices);

    cudaMemcpyAsync(histo.data(), cuda_histo, histo.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(max_color_array.data(), cuda_max_color_array, max_color_array.size(), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(min_color_array.data(), cuda_min_color_array, min_color_array.size(), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(pixel_found_array.data(), cuda_pixel_found_array, pixel_found_array.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(cuda_data);
    cudaFree(cuda_max_color_array);
    cudaFree(cuda_min_color_array);
    cudaFree(cuda_histo);
    cudaFree(cuda_start_indices);
    cudaFree(cuda_stop_indices);
    cudaFree(cuda_pixel_found_array);

    color_t max_color, min_color;
    bool pixel_found = false;
    for (int thread_index=0; thread_index<num_threads; ++thread_index) {
        for (int color_index=0; color_index<3; ++color_index) {
            max_color[color_index] = std::max(max_color[color_index], max_color_array[thread_index * 3 + color_index]);
            min_color[color_index] = std::min(min_color[color_index], min_color_array[thread_index * 3 + color_index]);
        }
        pixel_found = pixel_found_array[thread_index] || pixel_found;
    }

    return {histo, min_color, max_color, pixel_found};
}
