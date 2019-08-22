#ifndef PYTORCH_LAYERS_H
#define PYTORCH_LAYERS_H

#include <torch/torch.h>

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

#endif
