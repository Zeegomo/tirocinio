#include <iostream>
#include <vector>
#include <fstream>
#include "csv.hpp"
#include "network.hpp"
#include "pytorch.hpp"

#define CONF "parameters.json"
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

int main() {
	ofstream fout("out.data");
	auto raw_time_series = load_data();
	Config conf = load_config(CONF, raw_time_series);

	cout << "Starting.." << endl << endl;
	cout << "Input dim: " << conf.input_dim << endl;
	cout << "Hidden dim: " << conf.hidden_dim << endl;
	cout << "Output dim: " << conf.output_dim << endl;
	cout << "Sequence length: " << raw_time_series.size() << endl;
	cout << "Number of epochs: " << conf.num_epochs << endl;
	cout << "Number of training sequences: " << conf.n_sequences << endl;
	cout << "Length of training sequences: " << conf.batch_size << endl;
	cout << "Learning rate: " << conf.learning_rate << endl << endl;


	vector<double> output;
	for(int i = 1; i < raw_time_series.size()-1; i++){
		output.push_back(raw_time_series[i+1][conf.target_column]);
	}


	Pytorch pytorch_model(conf, raw_time_series);
	
	cout << "Training..." << endl;	
	auto pytorch_hist = pytorch_model.train();
	
	cout << endl << "Evaluation..." << endl;
	auto pytorch_res = pytorch_model.evaluate();

	cout << "MSE: " << pytorch_res.second << endl;

	cout << "Serializing to file" << endl;

	serialize(fout, output);
	serialize(fout, pytorch_res.first, output.size());
	serialize(fout, pytorch_hist.v_mse);
}

