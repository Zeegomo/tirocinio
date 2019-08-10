#include <torch/torch.h>
#include <iostream>

using namespace std;

struct LSTM : torch::nn::Module {
	LSTM(int input_dim, int hidden_dim, int output_dim, int b_size)
	: 	lstm(register_module("lstm", torch::nn::LSTM(input_dim, hidden_dim))),
		linear(register_module("linear", torch::nn::Linear(hidden_dim, output_dim)))	
	{
		this->batch_size = b_size;	
		this->input_dim = input_dim;
		this->output_dim = output_dim;
	}

	torch::Tensor forward(torch::Tensor input){
		auto out = lstm(input, hidden);
		hidden = out.state;
		hidden.detach_();
		hidden = hidden.detach();
		//out.output.permute({1, 0, 2});
		int dim = out.output.sizes()[0];
		return linear(out.output);

	}
	torch::nn::LSTM lstm;
	torch::nn::Linear linear;
	torch::Tensor hidden;
	int batch_size;
	int input_dim;
	int output_dim;
	int hidden_dim;
};

struct GRU : torch::nn::Module {
	GRU(int input_dim, int hiddem_dim, int output_dim, int b_size)
	:	gru(register_module("gru", torch::nn::GRU(input_dim, hidden_dim))),
		linear(register_module("linear", torch::nn::Linear(hidden_dim, output_dim)))
	{
		this->batch_size = b_size;
		this->input_dim = input_dim;
		this->output_dim = output_dim;
	}

	torch::Tensor forward(torch::Tensor input){
		auto out = gru(input, hidden);
		hidden = out.state;
		hidden.detach_();
		hidden = hidden.detach();
		return linear(out.output);
	}
	torch::nn::GRU gru;
	torch::nn::Linear linear;
	torch::Tensor hidden;
	int batch_size;
	int input_dim;
	int output_dim;
	int hidden_dim;
};

int main() {

	int num_epochs = 1001;

	LSTM model(1, 30, 1, 10);
	//GRU model(1, 30, 1, 10);

	auto loss_fn = torch::mse_loss;
	auto optimiser = torch::optim::Adam(model.parameters(), 0.05f);

//	auto dataset = torch::data::datasets::

	auto time_series = torch::zeros({100, 1});

	for(int i = 0; i < 100; i++){
		double a;
		string c;
		cin /*>> c*/ >> a;
		time_series[i][0] = a;
	}

	auto tmp = time_series.chunk(2);
	auto inputs = torch::zeros({1, 10, 1});
	auto outputs = torch::zeros({1, 10, 1});

	int pos = 0;
	for ( size_t i = 0; i < 1; i++ )
        {
            for ( size_t j = 0; j < 10; j++ )
            {
                inputs[i][j] = time_series[i + j];
                outputs[i][j] = time_series[i + j + 1];
		pos++;
            }
        }
	
	
	cout << "Training..." << endl;	
	auto hist = torch::zeros(num_epochs);
	for(int i = 0; i < num_epochs; i++){
		model.zero_grad();

		auto y_pred = model.forward(inputs);
		auto loss = at::mse_loss(y_pred, outputs);

		if (i % 100 == 0){
			cout<< "Epoch: " << i << " | MSE: " << loss << endl;
		}
		hist[i] = loss.item();

		optimiser.zero_grad();
		loss.backward();
		optimiser.step();
	}

	cout << "Evaluation..." << endl;
	model.eval();
	
	auto input = torch::zeros({1, 10, 1});
	auto output = torch::zeros({1, 10, 1});
	int shift = 0;
	for(int i = 0; i < 10; i++){
		input[0][i] = time_series[i + shift];
		output[0][i] = time_series[i + 1 + shift];
	}

	auto net_output = model.forward(input);

	//net_output.permute({1, 0, 2});
	//output.permute({1, 0, 2});
	
	cout << "Input: " << input << endl;
	cout << "Real Data: " << output << endl;
	cout << "Prediction: " << net_output << endl;
	cout << "MSE: " << at::mse_loss(net_output, output);
}

