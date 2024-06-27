#include "BVH.cuh"

#include <vector>
#include <tuple>
#include <algorithm>

#include "cuError.h"
#include "cuda_utils.cuh"


void BVH_Handle::_delete() {
	CUDA_ASSERT(cudaFree(d_bvh));
	CUDA_ASSERT(cudaFree(d_bvh_nodes));
	CUDA_ASSERT(cudaFree(d_hittables));
}

BVH_Handle::~BVH_Handle() {
	_delete();
}

BVH_Handle::BVH_Handle(BVH_Handle&& bvhh) {
	d_bvh = bvhh.d_bvh;
	d_bvh_nodes = bvhh.d_bvh_nodes;
	d_hittables = bvhh.d_hittables;

	bvhh.d_bvh = nullptr;
	bvhh.d_bvh_nodes = nullptr;
	bvhh.d_hittables = nullptr;
}
BVH_Handle& BVH_Handle::operator=(BVH_Handle&& bvhh) {
	_delete();

	d_bvh = bvhh.d_bvh;
	d_bvh_nodes = bvhh.d_bvh_nodes;
	d_hittables = bvhh.d_hittables;

	bvhh.d_bvh = nullptr;
	bvhh.d_bvh_nodes = nullptr;
	bvhh.d_hittables = nullptr;

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


BVH_Handle::BVH_Handle(aabb bounds, int root_idx, std::vector<BVH::Node>& nodes, std::vector<const Hittable*>& hittables) {
	CUDA_ASSERT(cudaMalloc((void**)&d_hittables, sizeof(const Hittable*) * hittables.size()));
	CUDA_ASSERT(cudaMemcpy(d_hittables, hittables.data(), sizeof(const Hittable*) * hittables.size(), cudaMemcpyHostToDevice));

	CUDA_ASSERT(cudaMalloc((void**)&d_bvh_nodes, sizeof(BVH::Node) * nodes.size()));
	CUDA_ASSERT(cudaMemcpy(d_bvh_nodes, nodes.data(), sizeof(BVH::Node) * nodes.size(), cudaMemcpyHostToDevice));

	d_bvh = newOnDevice<BVH>(d_bvh_nodes, d_hittables, root_idx);
}

BVH_Handle::Factory::Factory(std::vector<std::tuple<aabb, const Hittable*>>& arr) 
	: arr(arr) {}

void BVH_Handle::Factory::BuildBVH_TopDown() {

	root_idx = _build_bvh_rec(0, arr.size());

	hittables.reserve(arr.size());
	for (int i = 0; i < arr.size(); i++) {
		hittables.push_back(std::get<1>(arr[i]));
	}
}

void BVH_Handle::Factory::_find_optimal_merge(int& a_idx, int& b_idx) {
	float best_cost = FLT_MAX;

	for (int a = 0; a < building_nodes.size(); a++) {
		for (int b = a + 1; b < building_nodes.size(); b++) {
			if (a == b) continue;

			auto& a_node = building_nodes[a];
			auto& b_node = building_nodes[b];

			aabb new_bounds = aabb(std::get<0>(a_node).bounds, std::get<0>(b_node).bounds);
			int hittable_count = std::get<1>(a_node) + std::get<1>(b_node);
			float new_cost = new_bounds.surface_area() * hittable_count;

			if (new_cost < best_cost) {
				a_idx = a;
				b_idx = b;
				best_cost = new_cost;
			}
		}
	}
}

void BVH_Handle::Factory::_merge_nodes(int a_idx, int b_idx) {
	auto& a = building_nodes[a_idx];
	auto& b = building_nodes[b_idx];

	int hittable_count = std::get<1>(a) + std::get<1>(b);
	aabb bounds = aabb(std::get<0>(a).bounds, std::get<0>(b).bounds);
	int a_list_idx = std::get<2>(a);
	int b_list_idx = std::get<2>(b);

	BVH::Node node;
	node.bounds = bounds;
	node.left_child_idx = a_list_idx;
	node.right_chlid_hittable_idx = b_list_idx;


	// delete the later index first otherwise everything shifts over and your index points to the wrong element
	if (a_idx > b_idx) std::swap(a_idx, b_idx);
	building_nodes.erase(building_nodes.begin() + b_idx);
	building_nodes.erase(building_nodes.begin() + a_idx);


	building_nodes.push_back(std::make_tuple(node, hittable_count, bvh_nodes.size()));
	bvh_nodes.push_back(node);
}

void BVH_Handle::Factory::BuildBVH_BottomUp() {
	
	for (int i = 0; i < arr.size(); i++) {

		BVH::Node node;
		node.bounds = std::get<0>(arr[i]);
		node.left_child_idx = -1;
		node.right_chlid_hittable_idx = hittables.size();
		hittables.push_back(std::get<1>(arr[i]));

		building_nodes.push_back(std::make_tuple(node, 1, bvh_nodes.size()));
		bvh_nodes.push_back(node);
	}

	while (building_nodes.size() > 1) {
		int a, b;
		_find_optimal_merge(a, b);
		_merge_nodes(a, b);
	}

	root_idx = std::get<2>(building_nodes[0]);
}

BVH_Handle BVH_Handle::Factory::MakeHandle() {

	aabb bounds = bvh_nodes[root_idx].bounds;

	return BVH_Handle(bounds, root_idx, bvh_nodes, hittables);
}