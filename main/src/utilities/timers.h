#ifndef TIMER_CLASSES_H
#define TIMER_CLASSES_H

#include <chrono>
#include <cuda.h>
#include "cuda_utilities/cuError.h"


class cudaTimer {
	cudaEvent_t _start{}, _end{};
	cudaStream_t stream{};
public:

	cudaTimer(cudaStream_t stream = 0) : stream(stream) {
		CUDA_ASSERT(cudaEventCreate(&_start));
		CUDA_ASSERT(cudaEventCreateWithFlags(&_end, cudaEventBlockingSync));
	}
	cudaTimer(cudaTimer&& t) {
		_start = t._start;
		_end = t._end;

		t._start = nullptr;
		t._end = nullptr;
	}
	~cudaTimer() {
		if (_start)
			CUDA_ASSERT(cudaEventDestroy(_start));
		if (_end)
			CUDA_ASSERT(cudaEventDestroy(_end));
	}

	void start() { CUDA_ASSERT(cudaEventRecord(_start, stream)); }
	void end() { CUDA_ASSERT(cudaEventRecord(_end, stream)); }
	float elapsedms() const {
		CUDA_ASSERT(cudaEventSynchronize(_end));
		float ms{};
		CUDA_ASSERT(cudaEventElapsedTime(&ms, _start, _end));
		return ms;
	}
};

class hostTimer {
	std::chrono::time_point<std::chrono::system_clock> _start{}, _end{};
public:

	hostTimer() = default;

	void start() { _start = std::chrono::system_clock::now(); }
	void end() { _end = std::chrono::system_clock::now(); }
	float elapsedms() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(_end - _start).count() * 1e-6f; }
};


#endif // TIMER_CLASSES_H //