#ifndef DISJOINTSET_H
#define DISJOINTSET_H

#include <vector>

/**
 * @brief Disjoint Set (Union-Find) data structure for managing disjoint sets.
 */
class DisjointSet {
public:
    /**
     * @brief Constructor initializes the parent vector.
     * @param sz Number of elements.
     */
    DisjointSet(int sz);

    /**
     * @brief Finds the representative (root) of the set containing element x.
     *        Implements path compression for efficiency.
     * @param x Element to find.
     * @return Representative of the set containing x.
     */
    int find(int x);

    /**
     * @brief Merges the set containing element x into the set containing element y.
     *        Note: x and y are not interchangeable.
     * @param x Element from the first set.
     * @param y Element from the second set.
     */
    void merge(int x, int y);

private:
    int size_;                     // Number of elements
    std::vector<int> parent_;      // Parent pointers for each element
};

#endif // DISJOINTSET_H
