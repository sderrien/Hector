#include "Schedule/TimeGraph.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>

// Include any other necessary headers
#include "Schedule/SDCSchedule.h" // Adjust the path as per your project structure

/**
 * @brief Constructor initializes the graph with one node.
 */
TimeGraph::TimeGraph() : numNode_(1) {
    edge_.emplace_back();    // Initialize adjacency list for node 0
    redge_.emplace_back();   // Initialize reverse adjacency list for node 0
}

/**
 * @brief Adds a new node to the graph connected from a previous node.
 */
int TimeGraph::addNode(int prev, const std::string &type, int length, int II, int tripcount) {
    edge_.emplace_back();    // Add adjacency list for the new node
    redge_.emplace_back();   // Add reverse adjacency list for the new node
    numNode_ += 1;
    int newNode = numNode_ - 1;

    // Create and add the edge from prev to newNode
    Edge e = {type, prev, newNode, length, tripcount, II};
    edge_[prev].push_back(e);
    redge_[newNode].push_back(e);

    return newNode;
}

/**
 * @brief Adds a directed edge to the graph.
 */
void TimeGraph::addEdge(int from, int to, const std::string &type, int length, int II, int tripcount) {
    Edge e = {type, from, to, length, tripcount, II};
    edge_[from].push_back(e);
    redge_[to].push_back(e);
}

/**
 * @brief Creates an MLIR attribute from an Edge.
 */
mlir::Attribute TimeGraph::makeAttr(mlir::MLIRContext *ctx, const Edge &edge) const {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    std::string retStr(edge.ds);

    // Append length if greater than 0
    if (edge.length > 0)
        retStr += ":" + std::to_string(edge.length);

    // Add type attribute
    attrs.emplace_back(mlir::StringAttr::get(ctx, "type"), mlir::StringAttr::get(ctx, retStr));

    // Add pipeline and II attributes if applicable
    if (edge.II != -1) {
        attrs.emplace_back(mlir::StringAttr::get(ctx, "pipeline"),
                           mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 1));
        attrs.emplace_back(mlir::StringAttr::get(ctx, "II"),
                           mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), edge.II));
    }

    // Add tripcount attribute if applicable
    if (edge.tripcount != -1) {
        attrs.emplace_back(mlir::StringAttr::get(ctx, "times"),
                           mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), edge.tripcount));
    }

    // Create and return the dictionary attribute
    return mlir::DictionaryAttr::get(ctx, attrs);
}

/**
 * @brief Prints the current state of the time graph to stdout.
 */
void TimeGraph::print() const {
    llvm::outs() << "-----Here is Time Graph-----\n";
    for (int i = 0; i < numNode_; ++i) {
        llvm::outs() << i << ": ";
        for (const auto &e : edge_[i]) {
            llvm::outs() << e.to << "(" << e.length << ") ";
        }
        llvm::outs() << "\n";
    }
    llvm::outs() << "-----------------------------\n";
}

/**
 * @brief Rewrites the MLIR region based on the time graph.
 */
void TimeGraph::rewrite(mlir::Region &region, mlir::PatternRewriter &rewriter) const {
    std::vector<std::vector<mlir::Attribute>> froms(numNode_);
    std::vector<std::vector<mlir::Attribute>> attrs(numNode_);

    // Populate 'froms' and 'attrs' based on edges
    for (int i = 0; i < numNode_; ++i) {
        for (const auto &e : edge_[i]) {
            froms[e.to].push_back(
                    mlir::IntegerAttr::get(
                            mlir::IntegerType::get(region.getContext(), 32, mlir::IntegerType::Signless),
                            e.from));
            attrs[e.to].push_back(makeAttr(region.getContext(), e));
        }
    }

    mlir::Location loc = region.getLoc();
    rewriter.setInsertionPointToStart(&region.front());

    // Create a TimeGraphOp with start and end nodes
    auto timeGraphOp = rewriter.create<mlir::tor::TimeGraphOp>(loc, 0, numNode_ - 1);
    rewriter.createBlock(&timeGraphOp.getBodyRegion());
    rewriter.setInsertionPointToStart(timeGraphOp.getBody());

    // Assume start-time is 0 and create successor time operations
    for (int i = 1; i < numNode_; ++i) {
        rewriter.create<mlir::tor::SuccTimeOp>(
                loc, i,
                mlir::ArrayAttr::get(region.getContext(), froms[i]),
                mlir::ArrayAttr::get(region.getContext(), attrs[i])
        );
    }

    // Finish the TimeGraphOp
    rewriter.create<mlir::tor::FinishOp>(loc);
}

/**
 * @brief Canonicalizes the time graph by removing nodes with specific criteria.
 */
void TimeGraph::canonicalize(std::vector<int> &newId) {
    DisjointSet dset(numNode_);

    // Merge nodes based on reverse edges criteria
    for (int i = 1; i < numNode_; ++i) {
        if (redge_[i].size() > 1)
            continue;
        const auto &e = redge_[i][0];
        if (e.ds == "static" && e.length == 0) {
            llvm::outs() << i << " " << e.ds << " " << e.length << "\n";
            dset.merge(i, e.from);
        }
    }

    int reducedNum = 0;
    newId.resize(numNode_);

    // Assign new IDs to representative nodes
    for (int i = 0; i < numNode_; ++i)
        if (dset.find(i) == i)
            newId[i] = reducedNum++;

    // Assign new IDs to non-representative nodes
    for (int i = 0; i < numNode_; ++i)
        if (dset.find(i) != i)
            newId[i] = newId[dset.find(i)];

    llvm::outs() << numNode_ << " " << reducedNum << "\n";

    // Move existing edges to temporary storage
    std::vector<std::vector<Edge>> oldedges(std::move(edge_));
    std::vector<std::vector<Edge>> oldredges(std::move(redge_));

    // Resize the edge vectors to the reduced number of nodes
    edge_.resize(reducedNum);
    redge_.resize(reducedNum);

    // Reconstruct edges based on new IDs
    for (int i = 0; i < numNode_; ++i) {
        for (auto &e : oldedges[i]) {
            int u = dset.find(e.from);
            int v = dset.find(e.to);
            if (u == v)
                continue; // Skip self-loops

            addEdge(newId[u], newId[v], e.ds, e.length, e.II, e.tripcount);
        }
    }

    numNode_ = reducedNum;
}
