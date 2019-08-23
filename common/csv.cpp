#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include "csv.hpp"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

vector<vector<double>> load_data(Config conf){
	vector<vector<double>> data;

	//store already mappend strings
	map<int, map<string, double>> conv;
	//store next available double for conversion for each column
	map<int, double> avail_conv;
	
	char delimeter = ',';
	string line = "";
	int data_index;
        while(getline(cin, line)){
		vector<double> dest;
		string buf = "";
                int i = 0;
		data_index = 0;
                while(i < line.length()) {
                        if(line[i] != delimeter){
                                buf += line[i];
                        }else{
				if(conf.ignore.count(data_index) == 0){	
					try{
						double d = stod(buf);
						dest.push_back(d);
					} catch(const invalid_argument &ia){
						if(conv[data_index].count(buf) == 0){
								conv[data_index][buf] = avail_conv[data_index]++;
						}
						dest.push_back(conv[data_index][buf]);
					}
				}
                                buf = "";
				data_index++;
                        }
                        i++;
                }

                if(!buf.empty())
                        dest.push_back(stod(buf));

                data.push_back(dest);
        }

	 if(conf.n_sequences + conf.batch_size >= data.size()){
                cerr << "Invalid configuration: too many training samples" << endl;
                exit(EXIT_FAILURE);
        }
        if(conf.target_column >= data[0].size() && data[0].size() != 1){
                cerr << "Invalid column target: " << conf.target_column << ", found " << data[0].size() << " columns" << endl;
                exit(EXIT_FAILURE);
        }

        cout << "Training size: " << (conf.n_sequences + conf.batch_size) / (double)data.size() * 100 << "%" << endl;


        return data;
}

Config load_config(std::string path){
        ifstream fin(path);
        json j;
     	fin >> j;
	Config conf;

	conf.input_dim = j.at("input_dim").get<int>();
	conf.hidden_dim = j.at("hidden_dim").get<int>();
	conf.output_dim = j.at("output_dim").get<int>();

	conf.learning_rate = j.at("learning_rate").get<float>();
	conf.num_epochs = j.at("num_epochs").get<int>();
	conf.batch_size = j.at("batch_size").get<int>();
	conf.n_sequences = j.at("training_samples").get<int>();
	conf.target_column = j.at("target_column").get<int>();
	vector<int> ignore = j.at("ignore").get<vector<int>>();
	for(int i : ignore){
		conf.ignore.insert(i);
	}

	return conf;
}
