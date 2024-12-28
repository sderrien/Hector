#ifndef TIMEGRAPH_H
#define TIMEGRAPH_H

#include <string>
#include <vector>
#include <utility>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "DisjointSet.h" // Include the DisjointSet header

namespace scheduling {
    class ScheduleBase; // Forward declaration
}

namespace mlir {
    class MLIRContext;
}

class TimeGraph {
private:
    /**
     * @brief Represents an edge in the time graph.
     */
    struct Edge {
        std::string ds;    // Type or label of the edge
        int from;          // Source node
        int to;            // Destination node
        int length;        // Length or duration of the edge
        int tripcount;     // Number of trips or repetitions
        int II;            // Initiation Interval (for pipelining)
    };

    int numNode_;                                 // Total number of nodes
    std::vector<std::vector<Edge>> edge_;         // Adjacency list for edges
    std::vector<std::vector<Edge>> redge_;        // Reverse adjacency list
    std::vector<std::pair<int, int>> intvMap_;    // Interval mapping

public:
    /**
     * @brief Constructor initializes the graph with one node.
     */
    TimeGraph();

    /**
     * @brief Adds a new node to the graph connected from a previous node.
     * @param prev Previous node to connect from.
     * @param type Type or label of the edge.
     * @param length Length or duration of the edge.
     * @param II Initiation Interval (default -1).
     * @param tripcount Number of trips or repetitions (default -1).
     * @return The index of the newly added node.
     */
    int addNode(int prev, const std::string &type, int length, int II = -1, int tripcount = -1);

    /**
     * @brief Adds a directed edge to the graph.
     * @param from Source node.
     * @param to Destination node.
     * @param type Type or label of the edge.
     * @param length Length or duration of the edge.
     * @param II Initiation Interval (default -1).
     * @param tripcount Number of trips or repetitions (default -1).
     */
    void addEdge(int from, int to, const std::string &type, int length, int II = -1, int tripcount = -1);

    /**
     * @brief Creates an MLIR attribute from an Edge.
     * @param ctx MLIR context.
     * @param edge The edge to convert.
     * @return An MLIR DictionaryAttr representing the edge.
     */
    mlir::Attribute makeAttr(mlir::MLIRContext *ctx, const Edge &edge) const;

    /**
     * @brief Prints the current state of the time graph to stdout.
     */
    void print() const;

    /**
     * @brief Rewrites the MLIR region based on the time graph.
     * @param region The MLIR region to rewrite.
     * @param rewriter The pattern rewriter.
     */
    void rewrite(mlir::Region &region, mlir::PatternRewriter &rewriter) const;

    /**
     * @brief Canonicalizes the time graph by removing nodes with specific criteria.
     * @param newId Vector to store the new node IDs after canonicalization.
     */
    void canonicalize(std::vector<int> &newId);

    // Additional methods related to building the graph
    int buildTimeGraphBlock(mlir::Region &region,
                            std::vector<mlir::Operation *> &vec,
                            int prev,
                            scheduling::ScheduleBase *scheduler);

    int buildTimeGraph(mlir::Region &region,
                       int prev,
                       scheduling::ScheduleBase *scheduler);
};

#endif // TIMEGRAPH_H
