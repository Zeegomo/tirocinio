#include <iostream>
#include <vector>
#include <fstream>
#include "csv.hpp"
#include "network.hpp"
#include "pytorch.hpp"
#include "annt_net.hpp"

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

	int input_dim = 1;
	int output_dim = 1;
	int hidden_dim = 30;

	int num_epochs = 1001;
	int seq_len = 100;
	int batch_size = 10;
	int eval_batch_size = 10;
	int n_sequences = 50;
	float learning_rate = 0.1f;

	cout << "Starting.." << endl << endl;
	cout << "Input dim: " << input_dim << endl;
	cout << "Hidden dim: " << hidden_dim << endl;
	cout << "Output dim: " << output_dim << endl;
	cout << "Sequence length: " << seq_len << endl;
	cout << "Number of epochs: " << num_epochs << endl;
	cout << "Number of training sequences: " << n_sequences << endl;
	cout << "Length of training sequences: " << batch_size << endl;
	cout << "Learning rate: " << learning_rate << endl << endl;

	auto raw_time_series = loadFlatData();
	vector<double> output;
	for(int i = 1; i < raw_time_series.size(); i++){
		output.push_back(raw_time_series[i]);
	}


	Pytorch pytorch_model(num_epochs, seq_len, batch_size, eval_batch_size, n_sequences, learning_rate, raw_time_series, input_dim, hidden_dim, output_dim);
	ANNT_Net annt_model(num_epochs, seq_len, batch_size, eval_batch_size, n_sequences, learning_rate, raw_time_series, input_dim, hidden_dim, output_dim);	
	
	cout << "Training..." << endl;	
	auto pytorch_hist = pytorch_model.train();
	auto annt_hist = annt_model.train();
	
	cout << endl << "Evaluation..." << endl;
	auto pytorch_res = pytorch_model.evaluate();
	auto annt_res = annt_model.evaluate();

	cout << "MSE: " << endl;
	cout << "Pytorch: " << pytorch_res.second << endl;
	cout << "ANNT: " << annt_res.second << endl;

	cout << "Serializing to file" << endl;

	serialize(fout, output, seq_len-1);
	serialize(fout, pytorch_res.first, seq_len-1);
	serialize(fout, annt_res.first);
	serialize(fout, pytorch_hist);
	serialize(fout, annt_hist);
}

