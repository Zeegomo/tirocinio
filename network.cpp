#include "network.hpp"
using namespace std;
		
Network::Network(int num_epochs, int seq_len, int batch_size, int eval_batch_size, int n_sequences, float learning_rate, std::vector<double> raw_time_series, int input_dim, int hidden_dim, int output_dim){
	this->num_epochs = num_epochs;
        this->seq_len = seq_len;
        this->batch_size = batch_size;
        this->eval_batch_size = eval_batch_size;
        this->n_sequences = n_sequences;
        this->learning_rate = learning_rate;
        this->raw_time_series = raw_time_series;
        this->time_series = time_series;
        this->input_dim = input_dim;
        this->hidden_dim = hidden_dim;
	this->output_dim = output_dim;	
}


vector<pair<vector<vector<double>>, vector<vector<double>>>> Network::create_training_samples(bool normalize = true, bool diff = true){
        vector<pair<vector<vector<double>>, vector<vector<double>>>> samples(n_sequences);
        
        if(diff){
                apply_diff();
        }
        if(normalize){
                normalize_data();        
        }
        

        for (size_t i = 0; i < n_sequences; i++){
                vector<vector<double>> inputs(batch_size, vector<double>(input_dim));
                vector<vector<double>> outputs(batch_size, vector<double>(output_dim));
                for (size_t j = 0; j < batch_size; j++){
                        for(size_t z = 0; z < input_dim; j++){
                                inputs[j][z] = time_series[i + j][z];        
                        }        
                        outputs[i][j] = time_series[i + j + 1 + sshift];
                }
                samples[i].first = inputs;
                samples[i].second = outputs;
        }

        random_shuffle(samples.begin(), samples.end());
        return samples;

}

void Network::normalize_data(){
        mi = 1e9;
        ma = -1e9;

        for(auto d : time_series){
                for(auto x : d){
                        mi = min(mi, d);
                        ma = max(ma, d);
                }
        }

        for(auto &d : time_series){
                for(auto &x : d){
                        d = (d - mi)/ (m a- mi);
                }
        }
}

vector<vector<double>> rescale(vector<vector<double>> source){
        vector<vector<double>> res = source;
        for(auto &v : res){
                for(auto &d : v){
                        d = d*(ma-mi) + mi;
                }
        }
        return res;
}

void Network::apply_diff(){
        for(int i = 1; i < raw_time_series.size(); i++){
                for(int j = 1; j < raw_time_series[0].size(); j++){
                        time_series[i][j] -= raw_time_series[i-1][j];
                }
        }
}