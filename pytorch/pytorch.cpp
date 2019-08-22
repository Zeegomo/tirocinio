#include <torch/torch.h>
#include <vector>
#include "network.hpp"
#include "pytorch.hpp"
#include "pytorch_utils.hpp"

using namespace std;

Pytorch::Pytorch(Config conf, vector<vector<double>> raw_time_series) : 
		Network(conf, raw_time_series), model(conf.input_dim, conf.hidden_dim, conf.output_dim, conf.batch_size){
		if (torch::cuda::is_available()) {
	                cout << "CUDA is available! Training on GPU." << endl;
        	        device = torch::kCUDA;
        	}
		cout << conf.n_sequences << endl;	
		this->model.to(device);
}

Error Pytorch::train(bool verbose) {
	auto optimiser = torch::optim::Adam(model.parameters(), conf.learning_rate);
	cout << "created samples" << endl;
	auto samples = create_training_samples();
	
	cout << "done" << endl;
	vector<vector<vector<double>>> in;
	vector<vector<vector<double>>> out;
	for(auto d : samples){
		in.push_back(d.first);
		out.push_back(d.second);
	}

	cout << "converting to tensor" << endl;
	auto inputs = to_tensor(in , device);
	auto outputs = to_tensor(out, device);

	/*auto inputs = 
		torch::zeros({n_sequences, batch_size, 1}, device);
        auto outputs = torch::zeros({n_sequences, batch_size, 1}, device);*/
	auto hist = torch::zeros(conf.num_epochs, device);
	cout << "training" << endl;
        model.train();
	for(int i = 0; i < conf.num_epochs; i++){
               	model.zero_grad();

                model.reset_hidden();

       	        auto y_pred = model.forward(inputs);
               	auto loss = at::mse_loss(y_pred, outputs);

                if (i % 100 == 0 && verbose){
			cout << "Epoch: " << i << " | MSE: " << loss.item().to<float>() << endl;
                }
	        hist[i] = loss.item();
		//err.add_record(to_vec(y_pred), to_vec(outputs));

		optimiser.zero_grad();
               	loss.backward();
                optimiser.step();
       	}

	return err;
}

pair<vector<double>, double> Pytorch::evaluate() {
	model.eval();
        model.reset_hidden();
        auto output = torch::zeros({1, (int)time_series.size(), 1}, device);
       	auto net_output = torch::zeros({1, (int)time_series.size(), 1}, device);
        for(int i = 0; i < time_series.size()/conf.batch_size; i++){
               	auto input = torch::zeros({1, conf.batch_size, conf.input_dim}, device);
                int shift = i*conf.batch_size;
       	        for(int j = 0; j < conf.batch_size; j++){
			if(j + shift + 1 < time_series.size()){
				for(int z = 0; z < conf.input_dim; z++){
                                	input[0][j][z] = time_series[j + shift][z];
                        	}
				if(time_series[0].size() > 1){
                               		output[0][j + shift] = time_series[j + shift][conf.target_column];
				}else{
					output[0][j + shift] = time_series[j + shift + 1][0];
				}

                        }
       	        }
               	auto net = model.forward(input);
                for(int j = 0; j < conf.batch_size; j++){
       	        	net_output[0][j + shift] = net[0][j];
               	}
        }

	auto rescaled = rescale(to_vec_1d(net_output));
	return {rescaled, at::mse_loss(net_output, output).item().to<double>()};

}
