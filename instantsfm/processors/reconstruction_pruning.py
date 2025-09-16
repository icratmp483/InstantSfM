import numpy as np

from instantsfm.scene.defs import ImagePair, Ids2PairId, PairId2Ids, ViewGraph
from instantsfm.utils.union_find import UnionFind

'''
image_t ViewGraphManipulater::EstablishStrongClusters(
    ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    StrongClusterCriteria criteria,
    double min_thres,
    int min_num_images) {
  image_t num_img_before = view_graph.KeepLargestConnectedComponents(images);

  // Construct the initial cluster by keeping the pairs with weight > min_thres
  UnionFind<image_pair_t> uf;
  // Go through the edges, and add the edge with weight > min_thres
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    bool status = false;
    status = status ||
             (criteria == INLIER_NUM && image_pair.inliers.size() > min_thres);
    status = status || (criteria == WEIGHT && image_pair.weight > min_thres);
    if (status) {
      uf.Union(image_pair_t(image_pair.image_id1),
               image_pair_t(image_pair.image_id2));
    }
  }

  // For every two connected components, we check the number of slightly weaker
  // pairs (> 0.75 min_thres) between them Two clusters are concatenated if the
  // number of such pairs is larger than a threshold (2)
  bool status = true;
  int iteration = 0;
  while (status) {
    status = false;
    iteration++;

    if (iteration > 10) {
      break;
    }

    std::unordered_map<image_pair_t, std::unordered_map<image_pair_t, int>>
        num_pairs;
    for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
      if (image_pair.is_valid == false) continue;

      // If the number of inliers < 0.75 of the threshold, skip
      bool status = false;
      status = status || (criteria == INLIER_NUM &&
                          image_pair.inliers.size() < 0.75 * min_thres);
      status = status ||
               (criteria == WEIGHT && image_pair.weight < 0.75 * min_thres);
      if (status) continue;

      image_t image_id1 = image_pair.image_id1;
      image_t image_id2 = image_pair.image_id2;

      image_pair_t root1 = uf.Find(image_pair_t(image_id1));
      image_pair_t root2 = uf.Find(image_pair_t(image_id2));

      if (root1 == root2) {
        continue;
      }
      if (num_pairs.find(root1) == num_pairs.end())
        num_pairs.insert(
            std::make_pair(root1, std::unordered_map<image_pair_t, int>()));
      if (num_pairs.find(root2) == num_pairs.end())
        num_pairs.insert(
            std::make_pair(root2, std::unordered_map<image_pair_t, int>()));

      num_pairs[root1][root2]++;
      num_pairs[root2][root1]++;
    }
    // Connect the clusters progressively. If two clusters have more than 3
    // pairs, then connect them
    for (auto& [root1, counter] : num_pairs) {
      for (auto& [root2, count] : counter) {
        if (root1 <= root2) continue;

        if (count >= 2) {
          status = true;
          uf.Union(root1, root2);
        }
      }
    }
  }

  for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    image_t image_id1 = image_pair.image_id1;
    image_t image_id2 = image_pair.image_id2;

    if (uf.Find(image_pair_t(image_id1)) != uf.Find(image_pair_t(image_id2))) {
      image_pair.is_valid = false;
    }
  }
  int num_comp = view_graph.MarkConnectedComponents(images);

  LOG(INFO) << "Clustering take " << iteration << " iterations. "
            << "Images are grouped into " << num_comp
            << " clusters after strong-clustering";

  return num_comp;
}
'''
def EstablishStrongClusters(view_graph:ViewGraph, images, threshold, min_num_images):
    view_graph.keep_largest_connected_component(images)

    uf = UnionFind()
    for pair_id, image_pair in view_graph.image_pairs.items():
        if not image_pair.is_valid:
            continue
        if image_pair.weight > threshold:
            uf.Union(image_pair.image_id1, image_pair.image_id2)
    
    # For every two connected components, we check the number of slightly weaker
    # pairs (> 0.75 min_thres) between them Two clusters are concatenated if the
    # number of such pairs is larger than a threshold (2)
    status = True
    iteration = 0
    while status:
        status = False
        iteration += 1
        if iteration > 10:
            break

        num_pairs = {}
        for pair_id, image_pair in view_graph.image_pairs.items():
            if not image_pair.is_valid:
                continue
            if image_pair.weight < 0.75 * threshold:
                continue
            image_id1 = image_pair.image_id1
            image_id2 = image_pair.image_id2
            root1 = uf.Find(image_id1)
            root2 = uf.Find(image_id2)
            if root1 == root2:
                continue
            if root1 not in num_pairs:
                num_pairs[root1] = {}
            if root2 not in num_pairs:
                num_pairs[root2] = {}
            if root2 not in num_pairs[root1]:
                num_pairs[root1][root2] = 0
            if root1 not in num_pairs[root2]:
                num_pairs[root2][root1] = 0
            num_pairs[root1][root2] += 1
            num_pairs[root2][root1] += 1

        for root1, counter in num_pairs.items():
            for root2, count in counter.items():
                if root1 <= root2:
                    continue
                if count >= 2:
                    status = True
                    uf.Union(root1, root2)

    for pair_id, image_pair in view_graph.image_pairs.items():
        if not image_pair.is_valid:
            continue
        image_id1 = image_pair.image_id1
        image_id2 = image_pair.image_id2
        if uf.Find(image_id1) != uf.Find(image_id2):
            image_pair.is_valid = False
    
    num_comp = view_graph.mark_connected_components(images)
    print(f"Clustering take {iteration} iterations. Images are grouped into {num_comp} clusters after strong-clustering")

def PruneWeaklyConnectedImages(images, tracks, min_num_images=2):
    image_observation_count = {}

    for track in tracks.values():
        obs_count = track.observations.shape[0]
        if obs_count <= 2:
            continue
        for i in range(obs_count):
            for j in range(i+1, obs_count):
                image_id1 = track.observations[i][0]
                image_id2 = track.observations[j][0]
                if image_id1 == image_id2:
                    continue
                pair_id = Ids2PairId(image_id1, image_id2)
                if pair_id not in image_observation_count:
                    image_observation_count[pair_id] = 1
                else:
                    image_observation_count[pair_id] += 1

    image_observation_count = {pair_id: count for pair_id, count in image_observation_count.items() if count >= 5}
    print(f"Established visibility graph with {len(image_observation_count)} pairs")

    # sort the pair count
    pair_count = np.array(list(image_observation_count.values()))
    pair_count.sort()
    median_count = pair_count[len(pair_count) // 2]

    # calculate the MAD (median absolute deviation)
    pair_count_diff = np.abs(pair_count - median_count)
    pair_count_diff.sort()
    median_count_diff = pair_count_diff[len(pair_count_diff) // 2]
    print(f"Threshold for Strong Clustering: {median_count - median_count_diff}")

    view_graph = ViewGraph()
    for pair_id, count in image_observation_count.items():
        image_id1, image_id2 = PairId2Ids(pair_id)
        view_graph.image_pairs[pair_id] = ImagePair(image_id1=image_id1, image_id2=image_id2, weight=count)
    threshold = max(median_count - median_count_diff, 20)
    EstablishStrongClusters(view_graph, images, threshold, min_num_images)