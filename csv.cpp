#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include "csv.hpp"

std::vector<std::vector<double>> loadData() {
	std::vector<std::vector<double>> data;
	char delimeter = ',';
	std::string line = "";
        while(getline(std::cin, line)){
		std::vector<double> dest;
		std::string buf = "";
                int i = 0;
                while(i < line.length()) {
                        if(line[i] != delimeter){
                                buf += line[i];
                        }else{
                                dest.push_back(stod(buf));
                                buf = "";
                        }
                        i++;
                }

                if(!buf.empty())
                        dest.push_back(stod(buf));

                data.push_back(dest);
        }
        return data;
}

std::vector<double> loadFlatData(){
	auto vec = loadData();
	std::vector<double> res;
	for(auto v : vec){
		for(auto d : v){
			res.push_back(d);
		}
	}
	return res;
}
