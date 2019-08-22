#ifndef PY_UTILS_H
#define PY_UTILS_H

#include <torch/torch.h>
#include <vector>

torch::Tensor to_tensor(std::vector<std::vector<std::vector<double>>> vec, torch::Device dev);
std::vector<std::vector<std::vector<double>>> to_vec(torch::Tensor ten);
std::vector<double> to_vec_1d(torch::Tensor ten);

#endif
