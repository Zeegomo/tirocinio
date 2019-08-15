#ifndef PYTORCH_RNN_H
#define PYTORCH_RNN_H

#include <torch/torch.h>
#include <vector>
#include "network.hpp"

struct LSTM : torch::nn::Module {
        LSTM(int input_dim, int hidden_dim, int output_dim, int b_size);
        void set_batch_size(int b_size);
        void reset_hidden();
        torch::Tensor forward(torch::Tensor input);

 	private:
	torch::nn::LSTM lstm;
        torch::nn::Linear linear;
        torch::Tensor hidden;
        int batch_size;
        int input_dim;
        int output_dim;
        int hidden_dim;
};


struct GRU : torch::nn::Module {
        GRU(int input_dim, int hiddem_dim, int output_dim, int b_size);
        void reset_hidden();
	torch::Tensor forward(torch::Tensor input);

	private:
	torch::nn::GRU gru;
        torch::nn::Linear linear;
        torch::Tensor hidden;
        int batch_size;
        int input_dim;
        int output_dim;
        int hidden_dim;
};

class Pytorch : public Network {
        public:
        Pytorch(int num_epochs, int seq_len, int batch_size, int eval_batch_size, int n_sequences, float learning_rate, std::vector<double> raw_time_series, int input_dim, int hidden_dim, int output_dim);

        std::vector<double> train(bool verbose = true);
        std::pair<std::vector<double>, double> evaluate();

        private:
        LSTM model;
        double mi, ma;
	torch::Device device = torch::kCPU;
        torch::Tensor time_series;
};

#endif 
