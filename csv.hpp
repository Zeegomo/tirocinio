#ifndef CSVReader_H
#define CSVReader_H

#include "config.hpp"

std::vector<std::vector<double>> loadData();
std::vector<double> loadFlatData();
Config loadConfig(std::string path);

#endif
