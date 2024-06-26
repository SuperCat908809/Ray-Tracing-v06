#include "BVH.cuh"

#include <vector>
#include <tuple>
#include <algorithm>

#include "cuError.h"
#include "cuda_utils.cuh"


void BVH_Handle::_delete() {
	CUDA_ASSERT(cudaFree(bvh));
	CUDA_ASSERT(cudaFree(bvh_nodes));
	CUDA_ASSERT(cudaFree(hittables));
}

BVH_Handle::~BVH_Handle() {
	_delete();
}

BVH_Handle::BVH_Handle(BVH_Handle&& bvhh) {
	bvh = bvhh.bvh;
	bvh_nodes = bvhh.bvh_nodes;
	hittables = bvhh.hittables;

	bvhh.bvh = nullptr;
	bvhh.bvh_nodes = nullptr;
	bvhh.hittables = nullptr;
}
BVH_Handle& BVH_Handle::operator=(BVH_Handle&& bvhh) {
	_delete();

	bvh = bvhh.bvh;
	bvh_nodes = bvhh.bvh_nodes;
	hittables = bvhh.hittables;

	bvhh.bvh = nullptr;
	bvhh.bvh_nodes = nullptr;
	bvhh.hittables = nullptr;

	return *this;
}


aabb BVH_Handle::Factory::_get_partition_bounds(int start, int end) {
	aabb partition_bounds{};
	for (int i = start; i < end; i++) {
		partition_bounds += std::get<0>(arr[i]);
	}
	return partition_bounds;
}

int BVH_Handle::Factory::_build_bvh_rec(int start, int end) {
	aabb bounds = _get_partition_bounds(start, end);
	int axis = bounds.longest_axis();
	//auto comparator = (axis == 0) ? box_x_compare : ((axis == 1) ? box_y_compare : box_z_compare);
	auto comparator = (axis == 0) ? box_x_compare : ((axis == 1) ? box_y_compare : box_z_compare);

	int object_span = end - start;
	if (object_span == 1) {
		BVH::Node node;
		node.bounds = bounds;
		node.left_child_idx = -1;
		node.right_chlid_hittable_idx = start;
		bvh_nodes.push_back(node);
		return bvh_nodes.size() - 1;
	}
	else {
		//std::sort(arr.begin() + start, arr.begin() + end,
		//	[comparator](const std::tuple<aabb, const Hittable*>& a, const std::tuple<aabb, const Hittable*>& b)
		//	{
		//		return comparator(std::get<0>(a), std::get<0>(b));
		//	});
		std::sort(arr.begin() + start, arr.begin() + end,
			[comparator](const std::tuple<aabb, const Hittable*>& a, const std::tuple<aabb, const Hittable*>& b) {
				return comparator(std::get<0>(a), std::get<0>(b));
			}
		);

		int mid = (start + end) / 2;

		BVH::Node node;
		node.bounds = bounds;
		node.left_child_idx = _build_bvh_rec(start, mid);
		node.right_chlid_hittable_idx = _build_bvh_rec(mid, end);
		bvh_nodes.push_back(node);
		return bvh_nodes.size() - 1;
	}
}


BVH_Handle::Factory::Factory(std::vector<std::tuple<aabb, const Hittable*>>& arr)
	: arr(arr) {}

BVH_Handle BVH_Handle::Factory::MakeBVH() {

	int root_idx = _build_bvh_rec(0, arr.size());
	aabb bounds = bvh_nodes[root_idx].bounds;

	std::vector<const Hittable*> hittables;
	hittables.reserve(arr.size());
	for (int i = 0; i < arr.size(); i++) {
		hittables.push_back(std::get<1>(arr[i]));
	}

	const Hittable** d_hittables;
	CUDA_ASSERT(cudaMalloc((void**)&d_hittables, sizeof(const Hittable*) * hittables.size()));
	CUDA_ASSERT(cudaMemcpy(d_hittables, hittables.data(), sizeof(const Hittable*) * hittables.size(), cudaMemcpyHostToDevice));

	BVH::Node* d_nodes;
	CUDA_ASSERT(cudaMalloc((void**)&d_nodes, sizeof(BVH::Node) * bvh_nodes.size()));
	CUDA_ASSERT(cudaMemcpy(d_nodes, bvh_nodes.data(), sizeof(BVH::Node) * bvh_nodes.size(), cudaMemcpyHostToDevice));

	BVH* bvh = newOnDevice<BVH>(d_nodes, d_hittables, root_idx);

	BVH_Handle handle{};
	handle.bvh = bvh;
	handle.bounds = bounds;
	handle.bvh_nodes = d_nodes;
	handle.hittables = d_hittables;
	return handle;
}