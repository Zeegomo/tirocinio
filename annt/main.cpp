#include <iostream>
#include "annt_net.hpp"
#include "rnn.hpp"

using namespace std;

int main() {
	Executor ex();
	auto data = ex.get_data();
	ANNT_Net model(data.first, data.second);
	ex.start(model);
}

