#include <algorithm>
#include "network.hpp"
#include <iostream>

using namespace std;
		
Network::Network(Config config, std::vector<std::vector<double>> raw_time_series){
	this->raw_time_series = raw_time_series;
	this->time_series = raw_time_series;
	this->conf = config;
}


vector<pair<vector<vector<double>>, vector<vector<double>>>> Network::create_training_samples(bool normalize, bool diff){
	vector<pair<vector<vector<double>>, vector<vector<double>>>> samples(conf.n_sequences);
        if(diff){
                apply_diff();
        }
        if(normalize){
               normalize_data();        
        }
        for (size_t i = 0; i < conf.n_sequences; i++){
                vector<vector<double>> inputs(conf.batch_size);
                vector<vector<double>> outputs(conf.batch_size);
                for (size_t j = 0; j < conf.batch_size; j++){
                        for(size_t z = 0; z < conf.input_dim; z++){
                                	inputs[j].push_back(time_series[i + j][z]);        
                        }
			outputs[j].push_back(time_series[i + j + 1][conf.target_column]);
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
                        mi = min(mi, x);
                        ma = max(ma, x);
                }
        }

        for(auto &d : time_series){
                for(auto &x : d){
                        x = (x - mi) / (ma- mi);
                }
        }
}

vector<double> Network::rescale(vector<double> source){
        vector<double> res = source;
        for(auto &d : res){
        	d = d*(ma-mi) + mi;
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
