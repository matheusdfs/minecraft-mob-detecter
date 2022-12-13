from typing import List, Tuple
from cv2 import DMatch
import numpy as np


def bf_match_descriptors(query_descs, train_descs, match_min_ratio) -> List[Tuple[DMatch]]:
    matches = []
    for query_idx, query_desc in enumerate(query_descs):
        best_dist_1 = 100000000
        best_dist_2 = 100000000
        train_idx_1 = 0

        for train_idx, train_desc in enumerate(train_descs):
            distance = np.linalg.norm(query_desc - train_desc)

            if distance < best_dist_1:
                best_dist_2 = best_dist_1

                best_dist_1 = distance
                train_idx_1 = train_idx

            elif distance < best_dist_2:
                best_dist_2 = distance

        if best_dist_1 < best_dist_2 * match_min_ratio:
            matches.append([DMatch(query_idx, train_idx_1, best_dist_1)])

    matches_by_train_idx = {}
    for match in matches:
        train_idx = match[0].trainIdx
        if train_idx not in matches_by_train_idx:
            matches_by_train_idx[train_idx] = []
            
        matches_by_train_idx[train_idx].append(match)

    filtered_matches = [matches_group[0] for matches_group in matches_by_train_idx.values() if len(matches_group) == 1]

    return filtered_matches
