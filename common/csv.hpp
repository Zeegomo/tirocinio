#ifndef CSVReader_H
#define CSVReader_H

#include "config.hpp"

std::vector<std::vector<double>> load_data();
Config load_config(std::string path, std::vector<std::vector<double>> &data);

#endif
