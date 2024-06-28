#ifndef BVH_CLASS_CUH
#define BVH_CLASS_CUH

#include <cuda_runtime.h>
#include <tuple>
#include <vector>

#include "ray_data.cuh"
#include "hittable.cuh"
#include "aabb.cuh"
#include "cuda_utils.cuh"


#define _IS_LEAF_CODE (-1)
class BVH : public Hittable {

	friend class BVH_Handle;
public:
	struct Node {
		aabb bounds;
		int left_child_idx;
		int right_child_hittable_idx;
		// if leaf node then right_child_hittable_idx holds the hittable idx
		// otherwise left_child_idx and right_child_hittable_idx hold the indices
		// for the left and right child nodes respectively.

		__host__ __device__ bool isLeaf() const { return left_child_idx == _IS_LEAF_CODE; }
	};

private:
	const Node* bvh_nodes;
	const Hittable** hittables;
	const int root_idx;


public:

	__device__ BVH(const Node* bvh_nodes, const Hittable** hittables, int root_idx)
		: bvh_nodes(bvh_nodes), hittables(hittables), root_idx(root_idx) {}

	__device__ virtual bool ClosestIntersection(const Ray& ray, RayPayload& rec) const {
		int node_idx_stack[32];
		int stack_head = 0;

		node_idx_stack[stack_head++] = root_idx; // push root node

		bool hit_any = false;

		// while stack is not empty
		while (stack_head > 0) {
			int idx = node_idx_stack[--stack_head]; // pop by decrementing and retrieveing at that index
			const Node& node = bvh_nodes[idx];

			//if (!node.bounds.intersects(ray, rec.distance)) continue;

			if (node.isLeaf()) {
				auto ptr = hittables[node.right_child_hittable_idx];
				hit_any |= ptr->ClosestIntersection(ray, rec);
				continue;
			}
			else {
				float left_dist{ _MISS_DIST }, right_dist{ _MISS_DIST };
				int left_idx = node.left_child_idx;
				int right_idx = node.right_child_hittable_idx;

				bvh_nodes[left_idx].bounds.intersects(ray, rec.distance, left_dist);
				bvh_nodes[right_idx].bounds.intersects(ray, rec.distance, right_dist);

				// assert that left is closer for next step
				if (left_dist > right_dist) {
					cuda_swap(left_idx, right_idx);
					cuda_swap(left_dist, right_dist);
				}

				if (right_dist < rec.distance) node_idx_stack[stack_head++] = right_idx;
				if (left_dist < rec.distance) node_idx_stack[stack_head++] = left_idx; // push then increment
				// left is closer so it is pushed last to be popped first

				continue;
			}
		}

		return hit_any;
	}

};

class BVH_Handle {

	BVH_Handle(aabb bounds, int root_idx, std::vector<BVH::Node>& nodes, std::vector<const Hittable*>& hittables);

	BVH_Handle(const BVH_Handle&) = delete;
	BVH_Handle& operator=(const BVH_Handle&) = delete;

	BVH* d_bvh;
	aabb bounds;
	BVH::Node* d_bvh_nodes;
	const Hittable** d_hittables;

	void _delete();

public:

	class Factory {

		std::vector<BVH::Node> bvh_nodes;
		std::vector<const Hittable*> hittables;
		int root_idx;

		std::vector<std::tuple<aabb, const Hittable*>>& arr;
		
		// top down
		aabb _get_partition_bounds(int start, int end);
		int _build_bvh_rec1(int start, int end);
		int _build_bvh_rec2(int start, int end);
		void _find_optimal_split(int start, int end, const aabb& bounds, int& axis, float& split);
		void _partition_by_split(int start, int end, int axis, float split, int& mid_idx);


		// bottom up
		// BHV::Node node, int hittable_count, int bvh_list_index
		std::vector<std::tuple<BVH::Node, int, int>> building_nodes;

		void _find_optimal_merge(int& a_idx, int& b_idx);
		void _merge_nodes(int a, int b);

	public:

		Factory(std::vector<std::tuple<aabb, const Hittable*>>& arr);
		void BuildBVH_TopDown();
		void BuildBVH_BottomUp();
		BVH_Handle MakeHandle();

	};

	~BVH_Handle();

	BVH_Handle(BVH_Handle&& bvhh);
	BVH_Handle& operator=(BVH_Handle&& bvhh);

	const BVH* getBVHPtr() const { return d_bvh; }
	aabb getBounds() const { return bounds; }

};


#endif // BVH_CLASS_CUH //