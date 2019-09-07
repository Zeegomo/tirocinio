#include <string>
#include "rnn.hpp"
#include "pytorch.hpp"

using namespace std;

int main(int argc, char *argv[]){
	Executor ex;
	auto data = ex.get_data();
	Pytorch model(data.first, data.second);
	if(argc > 2){
		if(argv[1] == string("t")){
			ex.train(&model, argv[2]);
		}else if (argv[1] == string("e")){
			ex.evaluate(&model, argv[2]);
		}else{
			ex.train(&model, argv[2]);
			ex.evaluate(&model);
		}
	} else {
		ex.train(&model);
		ex.evaluate(&model);
	}
}
