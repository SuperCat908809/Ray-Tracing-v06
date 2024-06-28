#include "BVH.cuh"

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>
#include <tuple>
#include <algorithm>

#include "cuError.h"
#include "cuda_utils.cuh"

#include "ray_data.cuh"
#include "hittable.cuh"
#include "aabb.cuh"


#define _PRIO_QUEUE_ELEM_COUNT (32)
#define _USE_PRIO_QUEUE false
struct BVH::priority_queue {
	int indices[_PRIO_QUEUE_ELEM_COUNT];
	int head = 0;
#if _USE_PRIO_QUEUE
	float distances[_PRIO_QUEUE_ELEM_COUNT];
#endif

	__device__ bool isEmpty() const { return head == 0; }
	__device__ int  dequeue()		{ return indices[--head]; }
	__device__ void enqueue(int idx, float dist);
};

__device__ void BVH::priority_queue::enqueue(int idx, float dist) {
#if !_USE_PRIO_QUEUE
	indices[head] = idx;
	head++;

#else
	indices[head] = idx;
	head++;
	distances[head] = dist;

	for (int i = head - 1; i >= 1; i--) {
		if (distances[i] > distances[i - 1]) {
			cuda_swap(distances[i], distances[i - 1]);
			cuda_swap(indices[i], indices[i - 1]);
		}
		else break;
	}
#endif
}

__device__ BVH::BVH(const Node* bvh_nodes, const Hittable** hittables, int root_idx)
	: bvh_nodes(bvh_nodes), hittables(hittables), root_idx(root_idx) {}

__device__ bool BVH::ClosestIntersection(const Ray& ray, RayPayload& rec) const {

	priority_queue q{};

	float root_dist;
	if (!bvh_nodes[root_idx].bounds.intersects(ray, rec.distance, root_dist)) return false;

	q.enqueue(root_idx, root_dist);

	bool hit_any = false;

	while (!q.isEmpty()) {
		int idx = q.dequeue();
		const Node& node = bvh_nodes[idx];

		if (node.isLeaf()) {
			auto ptr = hittables[node.right_child_hittable_idx];
			hit_any |= ptr->ClosestIntersection(ray, rec);
			continue;
		}
		else {
			float left_dist{ _MISS_DIST }, right_dist{ _MISS_DIST };
			int left_idx = node.left_child_idx;
			int right_idx = node.right_child_hittable_idx;

#if _USE_PRIO_QUEUE
			if (bvh_nodes[left_idx].bounds.intersects(ray, rec.distance, left_dist)) {
				q.enqueue(left_idx, left_dist);
			}
			if (bvh_nodes[right_idx].bounds.intersects(ray, rec.distance, right_dist)) {
				q.enqueue(right_idx, right_dist);
			}
#else
			bvh_nodes[left_idx].bounds.intersects(ray, rec.distance, left_dist);
			bvh_nodes[right_idx].bounds.intersects(ray, rec.distance, right_dist);

			// assert that left is closer for next step
			if (left_dist > right_dist) {
				cuda_swap(left_idx, right_idx);
				cuda_swap(left_dist, right_dist);
			}

			if (right_dist < rec.distance) q.enqueue(right_idx, right_dist);
			if (left_dist < rec.distance) q.enqueue(left_idx, left_dist); // push then increment
			// left is closer so it is pushed last to be popped first
#endif

			continue;
		}
	}

	return hit_any;
}



BVH_Handle::BVH_Handle(aabb bounds, int root_idx, std::vector<BVH::Node>& nodes, std::vector<const Hittable*>& hittables) {
	BVH_Handle::bounds = bounds;

	CUDA_ASSERT(cudaMalloc((void**)&d_hittables, sizeof(const Hittable*) * hittables.size()));
	CUDA_ASSERT(cudaMemcpy(d_hittables, hittables.data(), sizeof(const Hittable*) * hittables.size(), cudaMemcpyHostToDevice));

	CUDA_ASSERT(cudaMalloc((void**)&d_bvh_nodes, sizeof(BVH::Node) * nodes.size()));
	CUDA_ASSERT(cudaMemcpy(d_bvh_nodes, nodes.data(), sizeof(BVH::Node) * nodes.size(), cudaMemcpyHostToDevice));

	d_bvh = newOnDevice<BVH>(d_bvh_nodes, d_hittables, root_idx);
}

BVH_Handle::~BVH_Handle() {
	_delete();
}

void BVH_Handle::_delete() {
	CUDA_ASSERT(cudaFree(d_bvh));
	CUDA_ASSERT(cudaFree(d_bvh_nodes));
	CUDA_ASSERT(cudaFree(d_hittables));
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



BVH_Handle::Factory::Factory(std::vector<std::tuple<aabb, const Hittable*>>& arr)
	: arr(arr) {}

BVH_Handle BVH_Handle::Factory::MakeHandle() {
	aabb bounds = bvh_nodes[root_idx].bounds;
	return BVH_Handle(bounds, root_idx, bvh_nodes, hittables);
}


void BVH_Handle::Factory::BuildBVH_TopDown() {

#if 1
	root_idx = _build_bvh_rec1(0, arr.size());
#else
	root_idx = _build_bvh_rec2(0, arr.size());
#endif

	hittables.reserve(arr.size());
	for (int i = 0; i < arr.size(); i++) {
		hittables.push_back(std::get<1>(arr[i]));
	}
}

int BVH_Handle::Factory::_build_bvh_rec1(int start, int end) {
	aabb bounds = _get_partition_bounds(start, end);
	int axis = bounds.longest_axis();
	auto comparator = (axis == 0) ? box_x_compare : ((axis == 1) ? box_y_compare : box_z_compare);

	int object_span = end - start;
	if (object_span == 1) {
		BVH::Node node;
		node.bounds = bounds;
		node.left_child_idx = _IS_LEAF_CODE;
		node.right_child_hittable_idx = start;
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
		node.left_child_idx = _build_bvh_rec1(start, mid);
		node.right_child_hittable_idx = _build_bvh_rec1(mid, end);
		bvh_nodes.push_back(node);
		return bvh_nodes.size() - 1;
	}
}

int BVH_Handle::Factory::_build_bvh_rec2(int start, int end) {
	aabb bounds = _get_partition_bounds(start, end);
	int object_span = end - start;

	if (object_span == 1) {
		BVH::Node node;
		node.bounds = bounds;
		node.left_child_idx = _IS_LEAF_CODE;
		node.right_child_hittable_idx = start;
		bvh_nodes.push_back(node);
		return bvh_nodes.size() - 1;
	}

	int axis;
	float split;
	_find_optimal_split(start, end, bounds, axis, split);

	int mid_idx;
	_partition_by_split(start, end, axis, split, mid_idx);

	BVH::Node node;
	node.bounds = bounds;
	node.left_child_idx = _build_bvh_rec2(start, mid_idx);
	node.right_child_hittable_idx = _build_bvh_rec2(mid_idx, end);
	bvh_nodes.push_back(node);
	return bvh_nodes.size() - 1;
}

void BVH_Handle::Factory::_find_optimal_split(int start, int end, const aabb& bounds, int& best_axis, float& best_split) {
	const int split_points = 16;
	float best_cost = FLT_MAX;

	for (int axis = 0; axis < 3; axis++) {
		for (int split = 0; split < split_points; split++) {
			// create N points equally spaced between 0 and 1 but not including either
			// e.g. N == 1 : 0.50
			//		N == 2 : 0.33, 0.66
			//		N == 3 : 0.25, 0.50, 0.75
			float pos = (split + 1.0f) / (split_points + 1.0f);
			pos = glm::mix(bounds.getMin()[axis], bounds.getMax()[axis], pos);
			aabb left_bounds{}, right_bounds{};
			int left_count = 0, right_count = 0;

			for (int i = start; i < end; i++) {
				const aabb& b = std::get<0>(arr[i]);
				glm::vec3 centeroid = b.centeroid();

				// true if b belongs left of split
				if (centeroid[axis] < pos) {
					left_bounds += b;
					left_count++;
				}
				else {
					right_bounds += b;
					right_count++;
				}
			}


			float cost = left_bounds.surface_area() * left_count + right_bounds.surface_area() * right_count;

			if (cost < best_cost) {
				best_cost = cost;
				best_axis = axis;
				best_split = pos;
			}
		}
	}
}

void BVH_Handle::Factory::_partition_by_split(int start, int end, int axis, float split_pos, int& mid_idx) {
	// true if b belongs left of split
	auto comparator = [axis, split_pos](const aabb& b) {
		glm::vec3 centeroid = b.centeroid();
		return centeroid[axis] < split_pos;
	};

	int i = start;
	int j = end;
	while (i < j) {
		const aabb& b = std::get<0>(arr[i]);

		if (comparator(b)) {
			i++;
		}
		else {
			std::swap(arr[i], arr[--j]);
		}
	}

	//mid_idx = comparator(std::get<0>(arr[i])) ? i : i + 1;
	mid_idx = i;
}

aabb BVH_Handle::Factory::_get_partition_bounds(int start, int end) {
	aabb partition_bounds{};
	for (int i = start; i < end; i++) {
		partition_bounds += std::get<0>(arr[i]);
	}
	return partition_bounds;
}


void BVH_Handle::Factory::BuildBVH_BottomUp() {

	for (int i = 0; i < arr.size(); i++) {

		BVH::Node node;
		node.bounds = std::get<0>(arr[i]);
		node.left_child_idx = _IS_LEAF_CODE;
		node.right_child_hittable_idx = hittables.size();
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
	node.right_child_hittable_idx = b_list_idx;


	// delete the later index first otherwise everything shifts over and your index points to the wrong element
	if (a_idx > b_idx) std::swap(a_idx, b_idx);
	building_nodes.erase(building_nodes.begin() + b_idx);
	building_nodes.erase(building_nodes.begin() + a_idx);


	building_nodes.push_back(std::make_tuple(node, hittable_count, bvh_nodes.size()));
	bvh_nodes.push_back(node);
}