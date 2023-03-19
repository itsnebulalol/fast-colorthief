#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <cmath>
#include <chrono>


namespace py = pybind11;

//using color_t = std::tuple<uint8_t, uint8_t, uint8_t>;
using color_t = std::array<uint8_t, 3>;

const int SIGBITS = 5;
const int RSHIFT = 8 - SIGBITS;
const int MAX_ITERATION = 1000;
const double FRACT_BY_POPULATIONS = 0.75;

enum Color {RED, GREEN, BLUE};

int get_color_index(int r, int g, int b);


#include "cmap.hpp"

//CMap quantize(std::vector<int>& histo, VBox& vbox, int color_count, std::vector<color_t>& pixels);
std::vector<color_t> quantize(std::vector<int>& histo, VBox& vbox, int color_count);


//py::array::c_style remove strides (https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)
std::vector<color_t> get_palette(py::array_t<uint8_t,  py::array::c_style> image, int color_count, int quality) {
    auto start = std::chrono::system_clock::now();
    py::buffer_info image_buffer = image.request();

    if (image_buffer.ndim != 3) throw std::runtime_error("Image must be 3D matrix (height x width x color)");
    if (image_buffer.shape[2] != 4) throw std::runtime_error("Image must have 4 channels (red x green x blue x alpha)");

    uint8_t* data = (uint8_t*)image_buffer.ptr;
    
    std::vector<int> histo(std::pow(2, 3 * SIGBITS), 0);
    color_t max_colors{0, 0, 0};
    color_t min_colors{255, 255, 255};
    bool pixel_found = false;

    for (int pixel_index=0; pixel_index < image_buffer.shape[0] * image_buffer.shape[1]; pixel_index += quality) {
        // Alpha channel big enough
        if (data[pixel_index * 4 + 3] >= 125) {
            // Not white
            if (data[pixel_index * 4] <= 250 || data[pixel_index * 4 + 1] <= 250 || data[pixel_index * 4 + 2] <= 250) {

                int histo_pixel_index = 0;
                for (int color_index=0; color_index<3; ++color_index) {
                    uint8_t color_value = data[pixel_index * 4 + color_index] >> RSHIFT;
                    max_colors[color_index] = std::max(max_colors[color_index], color_value);
                    min_colors[color_index] = std::min(min_colors[color_index], color_value);
                    histo_pixel_index += color_value << ((2 - color_index) * SIGBITS);
                }
                histo[histo_pixel_index] += 1;
                pixel_found = true;
            }
        }
    }

    if (!pixel_found) {
        throw std::runtime_error("Empty pixels when quantize");
    }

    auto end = std::chrono::system_clock::now();

    auto start2 = std::chrono::system_clock::now();

    VBox vbox = VBox(min_colors[0], max_colors[0], min_colors[1], max_colors[1], min_colors[2], max_colors[2], histo);
    auto result = quantize(histo, vbox, color_count);

    auto end2 = std::chrono::system_clock::now();

    std::cout << "First time " << (end - start).count() << " second time: " << (end2 - start2).count() << std::endl; 
    return result;
}


int get_color_index(int r, int g, int b) {
    return (r << (2 * SIGBITS)) + (g << SIGBITS) + b;
}


std::tuple<std::unordered_map<int, int>, int> compute_partial_sums(std::vector<int>& histo, VBox vbox, Color color) {
    int total = 0;
    int sum = 0;
    std::unordered_map<int, int> partialsum;

    if (color == RED) {
        for (int i=vbox.r1; i<vbox.r2 + 1; ++i) {
            sum = 0;
            for (int j=vbox.g1; j<vbox.g2 + 1; j++) {
                for (int k=vbox.b1; k<vbox.b2 + 1; k++) {
                    int index = get_color_index(i, j, k);
                    sum += histo[index];
                }
            }
            total += sum;
            partialsum[i] = total;
        }
    } else if (color == GREEN) {
        for (int i=vbox.g1; i<vbox.g2 + 1; ++i) {
            sum = 0;
            for (int j=vbox.r1; j<vbox.r2 + 1; j++) {
                for (int k=vbox.b1; k<vbox.b2 + 1; k++) {
                    int index = get_color_index(j, i, k);
                    sum += histo[index];
                }
            }
            total += sum;
            partialsum[i] = total;
        }
    } else {
        for (int i=vbox.b1; i<vbox.b2 + 1; ++i) {
            sum = 0;
            for (int j=vbox.r1; j<vbox.r2 + 1; j++) {
                for (int k=vbox.g1; k<vbox.g2 + 1; k++) {
                    int index = get_color_index(j, k, i);
                    sum += histo[index];
                }
            }
            total += sum;
            partialsum[i] = total;
        }
    }

    return {partialsum, total};
}


std::tuple<std::optional<VBox>, std::optional<VBox>> median_cut_apply(std::vector<int>& histo, VBox vbox) {
    int rw = vbox.r2 - vbox.r1 + 1;
    int gw = vbox.g2 - vbox.g1 + 1;
    int bw = vbox.b2 - vbox.b1 + 1;
    int maxw = std::max(rw, std::max(gw, bw));

    Color do_cut_color = RED;
    if (rw == maxw) {
        do_cut_color = RED;
    } else if (gw == maxw) {
        do_cut_color = GREEN;
    } else {
        do_cut_color = BLUE;
    }

    if (vbox.count() == 1)
        return {{vbox.copy()}, {}};

    std::unordered_map<int, int> lookaheadsum;

    auto [partialsum, total] = compute_partial_sums(histo, vbox, do_cut_color);

    for (auto [i, d] : partialsum) {
        lookaheadsum[i] = total - d;
    }

    int dim1_val;
    int dim2_val;
    if (do_cut_color == RED) {
        dim1_val = vbox.r1;
        dim2_val = vbox.r2;
    } else if (do_cut_color == GREEN) {
        dim1_val = vbox.g1;
        dim2_val = vbox.g2;
    } else {
        dim1_val = vbox.b1;
        dim2_val = vbox.b2;
    }

    for (int i=dim1_val; i<dim2_val + 1; ++i) {
        if (partialsum[i] > total / 2) {
            VBox vbox1 = vbox.copy();
            VBox vbox2 = vbox.copy();
            int left = i - dim1_val;
            int right = dim2_val - i;
            int d2;
            if (left <= right) {
                d2 = std::min(dim2_val - 1, int(i + right / 2.0));
            } else {
                d2 = std::max(dim1_val, int(i - 1 - left / 2.0));
            }

            while (!(partialsum.count(d2) > 0 && partialsum[d2] > 0)) {
                d2 += 1;
            }

            int original_d2 = d2;
            int count2 = lookaheadsum[d2];
            while (count2 == 0 && partialsum.count(d2 - 1) > 0 && partialsum[d2 - 1] > 0) {
                d2 -= 1;
            }
    
            if (do_cut_color == RED) {
                vbox1.r2 = d2;
                vbox2.r1 = vbox1.r2 + 1;
            } else if (do_cut_color == GREEN) {
                vbox1.g2 = d2;
                vbox2.g1 = vbox1.g2 + 1;

            } else {
                vbox1.b2 = d2;
                vbox2.b1 = vbox1.b2 + 1;
            }

            return {vbox1, vbox2};
        }
    }
    return {{}, {}};
}
 

bool box_count_compare(VBox& a, VBox& b) {
    return a.count() < b.count();
}


bool box_count_volume_compare(VBox& a, VBox& b) {
    return uint64_t(a.count()) * uint64_t(a.volume()) < uint64_t(b.count()) * uint64_t(b.volume());
}


void iter(PQueue<VBox, decltype(box_count_compare)>& lh, int target, std::vector<int>& histo) {
    int n_color = 1;
    int n_iter = 0;
    while (n_iter < MAX_ITERATION) {
        VBox vbox = lh.pop();
        if (vbox.count() == 0) {
            lh.push(vbox);
            n_iter += 1;
            continue;
        }

        auto [vbox1, vbox2] = median_cut_apply(histo, vbox);

        if (!vbox1) {
            throw std::runtime_error("vbox1 not defined; shouldnt happen!");
        }

        lh.push(vbox1.value());
        if (vbox2) {
            lh.push(vbox2.value());
            n_color += 1;
        }
        if (n_color >= target || n_iter > MAX_ITERATION) {
            return;
        }
        n_iter += 1;
    }
}


std::vector<color_t> quantize(std::vector<int>& histo, VBox& vbox, int color_count) {
    if (color_count < 2 || color_count > 256)
        throw std::runtime_error("Wrong number of max colors when quantize.");

    //std::vector<int> orig_histo = get_histo(pixels);
    //VBox orig_vbox = vbox_from_pixels(pixels, histo);

    /*
    for (int i=0; i<orig_histo.size(); ++i) {
        if (orig_histo[i] != histo[i]) {
            std::cout << "Difference on index " << i << ", " << orig_histo[i] << " vs " << histo[i] << std::endl;
        }
    }
    */
    /*
    std::cout << "Orig vbox" << std::endl;
    
    std::cout << orig_vbox.r1 << std::endl;
    std::cout << orig_vbox.g1 << std::endl;
    std::cout << orig_vbox.b1 << std::endl;
    std::cout << orig_vbox.r2 << std::endl;
    std::cout << orig_vbox.g2 << std::endl;
    std::cout << orig_vbox.b2 << std::endl;

    std::cout << "New vbox" << std::endl;

    std::cout << vbox.r1 << std::endl;
    std::cout << vbox.g1 << std::endl;
    std::cout << vbox.b1 << std::endl;
    std::cout << vbox.r2 << std::endl;
    std::cout << vbox.g2 << std::endl;
    std::cout << vbox.b2 << std::endl;
    */
    PQueue<VBox, decltype(box_count_compare)> pq(box_count_compare);
    pq.push(vbox);

    iter(pq, std::ceil(FRACT_BY_POPULATIONS * color_count), histo);

    //std::cout <<FRACT_BY_POPULATIONS * (double)color_count << std::endl;

    PQueue<VBox, decltype(box_count_volume_compare)> pq2(box_count_volume_compare);
    while (pq.size() > 0) {
        pq2.push(pq.pop());
    }

    iter(pq2, color_count - pq2.size(), histo);
    pq2.sort();
    std::vector<color_t> final_colors;
    for (auto& vbox : pq2.get_contents()) {
        final_colors.push_back(vbox.avg());
    }

    // in the queue, boxes were sorted from smallest to biggest, now we want to return the most important color (=biggest box) first
    std::reverse(final_colors.begin(), final_colors.end());

    return final_colors;
}


PYBIND11_MODULE(fast_colorthief_backend, m) {
    m.def("get_palette", &get_palette, "Return color palette");
};

