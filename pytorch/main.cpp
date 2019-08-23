#include "rnn.hpp"
#include "pytorch.hpp"


int main(){
	Executor ex;
	auto data = ex.get_data();
	Pytorch model(data.first, data.second);
	ex.start(&model);
}
