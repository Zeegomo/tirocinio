#include "pytorch_utils.hpp"
#include <vector>

using namespace std;

torch::Tensor to_tensor(vector<vector<vector<double>>> vec, torch::Device dev){
        auto res = torch::zeros({(int)vec.size(), (int)vec[0].size(), (int)vec[0][0].size()}, dev);

        for(int i = 0; i < vec.size(); i++){
                for(int j = 0; j < vec[0].size(); j++){
                        for(int z = 0; z < vec[0][0].size(); z++){
                                res[i][j][z] = vec[i][j][z];
                        }
                }
        }

        return res;
}

vector<vector<vector<double>>> to_vec(torch::Tensor ten){
	auto sizes = ten.sizes();
	vector<vector<vector<double>>> vec(sizes[0], 
			vector<vector<double>>(sizes[1], 
				vector<double> (sizes[2], 0.0)));

	for(int i = 0; i < sizes[0]; i++){
		for(int j = 0; j < sizes[1]; j++){
			for(int z = 0; z < sizes[2]; z++){
				vec[i][j][z] = ten[i][j][z].item().to<double>();
			}
		}
	}
	return vec;
}

vector<double> to_vec_1d(torch::Tensor ten){
        auto sizes = ten.sizes();
        vector<double> vec;
	
	for(int i = 0; i < sizes[0]; i++){
                for(int j = 0; j < sizes[1]; j++){
                        for(int z = 0; z < sizes[2]; z++){
                                vec.push_back(ten[i][j][z].item().to<double>());
                        }
                }
        }


        return vec;
}

