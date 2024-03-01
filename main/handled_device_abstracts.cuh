#ifndef HANDLED_DEVICE_ABSTRACT_CLASSES_H
#define HANDLED_DEVICE_ABSTRACT_CLASSES_H

#include "cuda_utils.h"


template <typename T>
class dAbstract {

	struct M {
		T* ptr;
		T** ptr2;

		static const M null;
	} m;

	// private constructor called by factory //
	dAbstract(M m) : m(std::move(m)) {}

public:

	// delete copy constructors //
	dAbstract(const dAbstract&) = delete;
	dAbstract& operator=(const dAbstract&) = delete;

	// factory //
	template <typename U = T, typename... Args>
	static dAbstract MakeAbstract(Args... args);

	// destructor
	~dAbstract();

	// move assignment and constructor //
	dAbstract<T>& operator=(dAbstract<T>&& other);
	dAbstract(dAbstract<T>&& other) : m(std::move(other.m)) {
		other.m = dAbstract<T>::M::null;
	}

	// accessors //
	T* getPtr() const { return m.ptr; }
	T** getPtr2() const { return m.ptr2; }

	// device side abstract constructor //
	template <typename T2, typename... Args>
	void MakeOnDevice(Args... args);

	// delete abstract from the device //
	void DeleteOnDevice();
};

template <typename T>
const dAbstract<T>::M dAbstract<T>::M::null{ nullptr, nullptr };



template <typename T>
class dAbstractArray {

	struct M {
		size_t length{};
		T** ptr2{};

		static const M null;
	} m;

	// private constructor called by factory //
	dAbstractArray(M m) : m(std::move(m)) {}

public:

	// delete copy constructors //
	dAbstractArray(const dAbstractArray&) = delete;
	dAbstractArray& operator=(const dAbstractArray&) = delete;

	// factory //
	static dAbstractArray<T> MakeArray(size_t count);

	// destructor
	~dAbstractArray();
	
	// move assignment and constructor //
	dAbstractArray<T>& operator=(dAbstractArray<T>&& other);
	dAbstractArray(dAbstractArray<T>&& other) : m(std::move(other.m)) {
		other.m = dAbstractArray<T>::M::null;
	}

	// accessors //
	size_t getLength() const { return m.length; }
	T** getDeviceArrayPtr() const { return m.ptr2; }
	std::vector<T*> getPtrVector() const;


	// device side abstract constructors //

	// make single abstract on device //
	template <typename T2, typename... Args>
	void MakeSingleOnDevice(size_t offset, Args... args);

	// make an array of abstracts using gpu array input //
	template <typename T2, typename... Args>
	void MakeOnDevice(size_t count, size_t array_offset, size_t input_offset, Args*... args);
	
	// make an array of abstracts by passing a vector to be copied to the device //
	template <typename T2, typename Arg>
	void MakeOnDeviceVector(size_t count, size_t array_offset, size_t input_offset, const std::vector<Arg>& varg);

	// make an array of abstracts by using a factory object which is already on the device //
	template <typename DeviceFactoryType>
	void MakeOnDeviceFactoryPtr(size_t count, size_t array_offset, size_t input_offset, DeviceFactoryType* d_factory);

	// make an array of abstracts by using a factory object to be copied to the device //
	template <typename DeviceFactoryType>
	void MakeOnDeviceFactory(size_t count, size_t array_offset, size_t input_offset, DeviceFactoryType factory);

	// delete a range of abstracts from the device //
	void DeleteOnDevice(size_t count, size_t offset);
};

template <typename T>
const dAbstractArray<T>::M dAbstractArray<T>::M::null{ 0ull, nullptr };

#include "handled_device_abstracts.inl"

#endif // HANDLED_DEVICE_ABSTRACT_CLASSES_H //