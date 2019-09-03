#ifndef PYTORCH_RNN_H
#define PYTORCH_RNN_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include "network.hpp"
#include "error.hpp"
#include "pytorch_layers.hpp"

class Pytorch : public Network {
        public:
        Pytorch(Config conf, std::vector<std::vector<double>> raw_time_series);

        Error train(bool verbose = true);
        std::pair<std::vector<double>, Error> evaluate();
	void save(std::string filename);
	void load(std::string filename);

        private:
        LSTM model;
        double mi, ma;
	torch::Device device = torch::kCPU;
        torch::Tensor ttime_series;
};

#endif 
