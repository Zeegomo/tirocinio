#include <torch/torch.h>
#include <vector>
#include "network.hpp"
#include "pytorch.hpp"

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

torch::Tensor inverse_rescale(torch::Tensor base, double ma, double mi){
        auto res = base;
        for(int i = 0; i < base.sizes()[1]; i++){
                res[0][i] = (base[0][i]*(ma-mi) + mi);
        }
        return res;
}

Pytorch::Pytorch(int num_epochs, int seq_len, int batch_size, int eval_batch_size, int n_sequences, float learning_rate, vector<double> raw_time_series, int input_dim, int hidden_dim, int output_dim) : 
		Network(num_epochs, seq_len, batch_size, eval_batch_size, n_sequences, learning_rate, raw_time_series, input_dim, hidden_dim, output_dim), model(input_dim, hidden_dim, output_dim, batch_size){
		if (torch::cuda::is_available()) {
	                cout << "CUDA is available! Training on GPU." << endl;
        	        device = torch::kCUDA;
        	}	
		this->model.to(device);

		time_series = torch::zeros({seq_len, 1});

		mi = 1e9;
		ma = -1e9;

		for(auto d : raw_time_series){
			mi = min(mi, d);
			ma = max(ma, d);
		}
		for(int i = 0; i < seq_len; i++){
			time_series[i][0] = (raw_time_series[i] - mi) / (ma - mi);
 		}
	}

vector<double> Pytorch::train(bool verbose) {
	auto optimiser = torch::optim::Adam(model.parameters(), learning_rate);

	auto inputs = torch::zeros({n_sequences, batch_size, 1}, device);
        auto outputs = torch::zeros({n_sequences, batch_size, 1}, device);
	auto hist = torch::zeros(num_epochs, device);

        int sshift = 0;
        for ( size_t i = 0; i < n_sequences; i++ )
       	{
            for ( size_t j = 0; j < batch_size; j++ )
       	    {
               	inputs[i][j] = time_series[i + j+ sshift];
                outputs[i][j] = time_series[i + j + 1 + sshift];
       	    }
        }

       	for(int i = 0; i < n_sequences; i+=2){
               	auto tmp = inputs[i];
                auto tmp2 = outputs[i];
                inputs[i] = inputs[n_sequences-1-i];
       	        outputs[i] = outputs[n_sequences-1-i];
               	inputs[n_sequences-1-i] = tmp;
                outputs[n_sequences-1-i] = tmp2;
       	}

        model.train();
	for(int i = 0; i < num_epochs; i++){
               	model.zero_grad();

                model.reset_hidden();

       	        auto y_pred = model.forward(inputs);
               	auto loss = at::mse_loss(y_pred, outputs);

                if (i % 100 == 0 && verbose){
			cout << "Epoch: " << i << " | MSE: " << loss.item().to<float>() << endl;
                }
	        hist[i] = loss.item();

		optimiser.zero_grad();
               	loss.backward();
                optimiser.step();
       	}

	vector<double> res;
	for(int i = 0; i < num_epochs; i++){
		res.push_back(hist[i].item().to<double>());
	}
	return res;
}

pair<vector<double>, double> Pytorch::evaluate() {
	model.eval();
        model.reset_hidden();
       	model.set_batch_size(eval_batch_size);
        auto output = torch::zeros({1, seq_len, 1}, device);
       	auto net_output = torch::zeros({1, seq_len, 1}, device);
        for(int i = 0; i < seq_len/eval_batch_size; i++){
               	auto input = torch::zeros({1, eval_batch_size, 1}, device);
                int shift = i*eval_batch_size;
       	        for(int j = 0; j < eval_batch_size; j++){
               	        if(j + shift + 1 < seq_len){
                       	        input[0][j] = time_series[j + shift];
                               	output[0][j + shift] = time_series[j + 1 + shift];
                        }
       	        }
               	auto net = model.forward(input);
                for(int j = 0; j < eval_batch_size; j++){
       	                net_output[0][j + shift] = net[0][j];
               	}
        }

	auto rescaled = inverse_rescale(net_output, ma, mi);
	vector<double> res;
	for(int i = 0; i < seq_len; i++){
		res.push_back(rescaled[0][i].item().to<double>());
	}
	return {res, at::mse_loss(net_output, output).item().to<double>()};

}
