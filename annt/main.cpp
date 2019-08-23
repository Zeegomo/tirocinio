#include "rnn.hpp"
#include "annt_net.hpp"


int main(){
	Executor ex;
	auto data = ex.get_data();
	ANNT_Net model(data.first, data.second);
	ex.start(&model);
}
