#include <array>
#include <tuple>
#include <vector>
#include <stdint.h>  // needed for uint8_t
#include <stdexcept>  // needed for std::runtime_error

using color_t = std::array<uint8_t, 3>;

const int SIGBITS = 5;
const int RSHIFT = 8 - SIGBITS;
const int MAX_ITERATION = 1000;
const double FRACT_BY_POPULATIONS = 0.75;
const int NUM_BLOCKS = 16;
const int THREADS_PER_BLOCK = 256;
//const int NUM_BLOCKS = 1;
//const int THREADS_PER_BLOCK = 1;

std::tuple<std::vector<int>, color_t, color_t, bool> get_histo_cuda(uint8_t* data, int pixel_count, int quality);