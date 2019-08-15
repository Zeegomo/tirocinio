#include "network.hpp"
		
Network::Network(int num_epochs, int seq_len, int batch_size, int eval_batch_size, int n_sequences, float learning_rate, std::vector<double> raw_time_series, int input_dim, int hidden_dim, int output_dim){
	this->num_epochs = num_epochs;
        this->seq_len = seq_len;
        this->batch_size = batch_size;
        this->eval_batch_size = eval_batch_size;
        this->n_sequences = n_sequences;
        this->learning_rate = learning_rate;
        this->raw_time_series = raw_time_series;
        this->input_dim = input_dim;
        this->hidden_dim = hidden_dim;
	this->output_dim = output_dim;	
}
