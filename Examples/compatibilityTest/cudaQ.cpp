#include <cudaq.h>

__qpu__ void kernel(int qubit_count) {
  cudaq::qvector qubits(qubit_count);
  h(qubits[0]);
  for (auto i = 1; i < qubit_count; ++i) {
    cx(qubits[0], qubits[i]);
  }
  mz(qubits);
}

void run()
{
  auto qubit_count = 3;
  auto result = cudaq::sample(kernel, qubit_count);
  result.dump(); // Example: { 11:500 00:500 }
}
