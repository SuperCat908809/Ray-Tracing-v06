#ifndef D_ABSTRACT_CLASSES_H
#define D_ABSTRACT_CLASSES_H

#include "cuda_utils.h"
#include "darray.cuh"
#include <concepts>


template <typename T, bool destruct = false>
class dAbstractArray {
	
	darray<T*> ptrs;

	void _delete();

public:

	dAbstractArray(const dAbstractArray&) = delete;
	dAbstractArray& operator=(const dAbstractArray&) = delete;

	dAbstractArray(size_t size);
	~dAbstractArray();

	template <typename U> requires std::derived_from<U, T>
	dAbstractArray(dAbstractArray<U, destruct>&& other);
	template <typename U> requires std::derived_from<U, T>
	dAbstractArray& operator=(dAbstractArray<U, destruct>&& other);

	
	size_t getLength() const { return ptrs.getLength(); }

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