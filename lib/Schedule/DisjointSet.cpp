#include "Schedule/DisjointSet.h"

/**
 * @brief Constructor initializes the parent vector.
 * @param sz Number of elements.
 */
DisjointSet::DisjointSet(int sz) : size_(sz), parent_(sz) {
    for (int i = 0; i < size_; ++i)
        parent_[i] = i; // Initialize each node's parent to itself
}

/**
 * @brief Finds the representative (root) of the set containing element x.
 *        Implements path compression for efficiency.
 * @param x Element to find.
 * @return Representative of the set containing x.
 */
int DisjointSet::find(int x) {
    if (parent_[x] != x)
        parent_[x] = find(parent_[x]); // Path compression
    return parent_[x];
}

/**
 * @brief Merges the set containing element x into the set containing element y.
 *        Note: x and y are not interchangeable.
 * @param x Element from the first set.
 * @param y Element from the second set.
 */
void DisjointSet::merge(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    if (rootX != rootY)
        parent_[rootX] = rootY;
}
