#ifndef D_ABSTRACT_CLASSES_H
#define D_ABSTRACT_CLASSES_H

#include "cuda_utils.h"
#include "darray.cuh"
#include <concepts>


template <typename T>
class dAbstractArray {
	
	size_t length{ 0ull };
	darray<T*> ptrs;

	void _delete();

public:

	dAbstractArray(const dAbstractArray&) = delete;
	dAbstractArray& operator=(const dAbstractArray&) = delete;

	dAbstractArray(size_t size);
	~dAbstractArray();

	template <typename U> requires std::derived_from<U, T>
	dAbstractArray<T>(dAbstractArray<U>&& other);
	template <typename U> requires std::derived_from<U, T>
	dAbstractArray<T>& operator=(dAbstractArray<U>&& other);

	size_t getLength() const { return length; }
	const T** getDeviceArrayPtr() const { return ptrs.getPtr(); }
	T** getDeviceArrayPtr() { return ptrs.getPtr(); }
	std::vector<T*> getPtrVector();
	std::vector<const T*> getPtrVector() const;

	template <typename U, typename... Args> requires std::derived_from<U, T>
	void MakeOnDevice(size_t count, size_t offset, size_t input_offset, const Args*... args);

	template <typename DeviceFactoryType>
	void MakeOnDeviceFactory(size_t count, size_t offset, size_t input_offset, DeviceFactoryType* factory);

	void DeleteOnDevice(size_t count, size_t offset);
};

#include "dAbstracts.inl"

#endif // D_ABSTRACT_CLASSES_H //