#include "pytorch_layers.hpp"

using namespace std;

LSTM::LSTM(int input_dim, int hidden_dim, int output_dim, int b_size)
        :       lstm(register_module("lstm", torch::nn::LSTM(input_dim, hidden_dim))),
                linear(register_module("linear", torch::nn::Linear(hidden_dim, output_dim)))
        {
                this->batch_size = b_size;
                this->input_dim = input_dim;
                this->output_dim = output_dim;
                this->hidden_dim = hidden_dim;
        }

void LSTM::set_batch_size(int b_size){
                this->batch_size = b_size;
        }

void LSTM::reset_hidden(){
                hidden = torch::zeros({2, 1, batch_size, hidden_dim});
        }

torch::Tensor LSTM::forward(torch::Tensor input){
                auto out = lstm(input, hidden);
                hidden = out.state;
                hidden.detach_();
                hidden = hidden.detach();
                //out.output.permute({1, 0, 2});
                int dim = out.output.sizes()[0];
                return linear(out.output);
}

GRU::GRU(int input_dim, int hiddem_dim, int output_dim, int b_size)
        :       gru(register_module("gru", torch::nn::GRU(input_dim, hidden_dim))),
                linear(register_module("linear", torch::nn::Linear(hidden_dim, output_dim)))
        {
                this->batch_size = b_size;
                this->input_dim = input_dim;
                this->output_dim = output_dim;
        }

void GRU::reset_hidden(){
        hidden = torch::zeros({2, 1, batch_size, hidden_dim});
}

torch::Tensor GRU::forward(torch::Tensor input){
        auto out = gru(input, hidden);
        hidden = out.state;
        hidden.detach_();
        hidden = hidden.detach();
        return linear(out.output);
}


