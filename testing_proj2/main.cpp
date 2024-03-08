//#include <iostream>

class Base {
public:

	int virtual func() {
		return 1;
	}
};

class Derived : public Base {
public:

	int func() override {
		return 2;
	}
};

int main() {

	int total = 0;

	Base base1;
	Base* base2 = new Base();
	Derived derived1;
	Derived* derived2 = new Derived();
	Base* derived3 = &derived1;

	total += base1.func();
	total += base2->func();
	total += derived1.func();
	total += derived2->func();
	total += derived3->func();

	delete base2;
	delete derived2;

	return total;
}