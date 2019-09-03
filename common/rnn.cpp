#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "csv.hpp"
#include "network.hpp"
#include "rnn.hpp"

#define CONF "parameters.json"
#define SAVE_PATH "saved"

using namespace std;

void serialize(ofstream &fout, vector<double> &v, int len = -1){
	if(len == -1){
		len = v.size();
	}
	for(int i = 0; i < len; i++){
		fout << v[i] << " ";
	}
	fout << endl;
}	

Executor::Executor(){
        this->conf = load_config(CONF);
	this->raw_time_series = load_data(conf);
}

pair<Config, vector<vector<double>>> Executor::get_data(){
	return {conf, raw_time_series}; 	
}

void Executor::train(Network *model){
	cout << "Starting.." << endl << endl;
	cout << "Input dim: " << conf.input_dim << endl;
	cout << "Hidden dim: " << conf.hidden_dim << endl;
	cout << "Output dim: " << conf.output_dim << endl;
	cout << "Sequence length: " << raw_time_series.size() << endl;
	cout << "Number of epochs: " << conf.num_epochs << endl;
	cout << "Number of training sequences: " << conf.n_sequences << endl;
	cout << "Length of training sequences: " << conf.batch_size << endl;
	cout << "Learning rate: " << conf.learning_rate << endl << endl;


	
	cout << "Training..." << endl;	
	hist = model->train(true);	
	model->save(SAVE_PATH);
	cout << "Model saved to " << SAVE_PATH << endl;
}

void Executor::evaluate(Network *model){
	ofstream fout("out.data");

	cout << endl << "Evaluation..." << endl;
	auto res = model->evaluate();

	vector<double> output;
	for(int i = 1; i < raw_time_series.size()-1; i++){
		output.push_back(raw_time_series[i+1][conf.target_column]);
	}

	cout << "MSE: " << res.second.mse << endl;
	cout << "MAD: " << res.second.mad << endl;
	cout << "BIAS: " << res.second.bias << endl;
	cout << "MAPE: " << res.second.mape << endl;
	cout << "RMSE: " << res.second.rmse << endl;
	cout << endl << "Serializing to file" << endl;

	serialize(fout, output);
	serialize(fout, res.first, output.size());
	serialize(fout, hist.v_mse);
}

void Executor::evaluate(Network *model, string path){
	model->load(path);
	model->apply_diff();
	model->normalize_data();
	evaluate(model);
}
