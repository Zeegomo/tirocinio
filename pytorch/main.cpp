#include "rnn.hpp"
#include "pytorch.hpp"


int main(int argc, char *argv[]){
	Executor ex;
	auto data = ex.get_data();
	Pytorch model(data.first, data.second);
	if(argc > 1){
		ex.evaluate(&model, argv[1]);
	} else {
		ex.train(&model);
		ex.evaluate(&model);
	}
}
