
// Include necessary headers
#include "TOR/PassDetail.h"
// #include "mlir/Analysis/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "TOR/TOR.h"
#include "TOR/TORDialect.h"
#include "TOR/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include <mlir/Transforms/DialectConversion.h>
#include "mlir/Transforms/Passes.h"
// #include "mlir/Transforms/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "Schedule/SDCSchedule.h"
#include "nlohmann/json.hpp"

#include "Schedule/DisjointSet.h"
#include "Schedule/TimeGraph.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <queue>
#include <vector>
#include <unordered_map>
#include <set>

// Define the debug type for LLVM debugging
#define DEBUG_TYPE "create-tor"

// Function to set interval attributes on an operation
void setIntvAttr(mlir::Operation *op, std::pair<int, int> intv) {
    op->setAttr("starttime",
                mlir::IntegerAttr::get(
                        mlir::IntegerType::get(op->getContext(), 32, mlir::IntegerType::Signless),
                        intv.first));

    op->setAttr("endtime",
                mlir::IntegerAttr::get(
                        mlir::IntegerType::get(op->getContext(), 32, mlir::IntegerType::Signless),
                        intv.second));
}

// Function to build a time graph block
int buildTimeGraphBlock(TimeGraph &tg,
                        std::vector<mlir::Operation *> &vec,
                        int prev,
                        scheduling::ScheduleBase *scheduler) {
    std::set<int> timeStamp;
    std::map<int, int> ts2Node;

    for (auto op : vec) {

        auto intv = scheduler->queryOp(op);
        // The operation runs in [intv.first, intv.second + 1)
        timeStamp.insert(intv.first);
        timeStamp.insert(intv.second);
        llvm::outs() << "building time graph block for " << *op << " -> ["<< intv.first<<":"<< intv.second<<"]\n";
    }

    int last = -1;
    int currentNode = prev;

    // Create nodes based on timestamps
    for (auto ts : timeStamp) {
        int node;

        if (last != -1)
            node = tg.addNode(currentNode, "static", ts - last);
        else
            node = tg.addNode(currentNode, "static", 0);

        ts2Node[ts] = node;
        currentNode = node;
        last = ts;
    }

    // Assign interval attributes to operations
    for (auto op : vec) {
        auto cycle = scheduler->queryOp(op);
        auto intv = std::make_pair(ts2Node[cycle.first], ts2Node[cycle.second]);
        setIntvAttr(op, intv);
        llvm::outs() << "setting attribute  [" << intv.first << ":" << intv.second << "] for " << *op << "\n";
    }

    vec.clear();
    llvm::outs() << "time graph block built\n";
    return currentNode;
}

// Function to build the entire time graph recursively
int buildTimeGraph(TimeGraph &tg,
                   mlir::Block &block,
                   int prev,
                   scheduling::ScheduleBase *scheduler) {
    int currentNode = prev;
    std::vector<mlir::Operation *> vec; // Buffer for sequential operations

    for (auto &op : block) {
        llvm::outs() << "building time graph for " << op << "\n";

        if (auto ifOp = llvm::dyn_cast<mlir::tor::IfOp>(op)) {
            // Handle IfOp by processing then and else regions
            currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);

            if (!ifOp.getElseRegion().empty()) {
                int thenNode = buildTimeGraph(tg, ifOp.getThenRegion().front(), currentNode, scheduler);
                int elseNode = buildTimeGraph(tg, ifOp.getElseRegion().front(), currentNode, scheduler);
                int nxtNode = tg.addNode(thenNode, "static", 0);

                // Connect else branch to the next node
                tg.addEdge(elseNode, nxtNode, "static", 0);
                setIntvAttr(&op, std::make_pair(currentNode, nxtNode));
                currentNode = nxtNode;
            } else {
                // Only then branch exists
                int thenNode = buildTimeGraph(tg, ifOp.getThenRegion().front(), currentNode, scheduler);
                setIntvAttr(&op, std::make_pair(currentNode, thenNode));
                currentNode = thenNode;
            }

        } else if (auto whileOp = llvm::dyn_cast<mlir::tor::WhileOp>(op)) {
            // Handle WhileOp by processing before and after regions
            currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);
            int beginNode = tg.addNode(currentNode, "static", 0);
            int condNode = buildTimeGraph(tg, whileOp.getBefore().front(), beginNode, scheduler);
            int endNode = buildTimeGraph(tg, whileOp.getAfter().front(), condNode, scheduler); // Body

            auto info = scheduler->queryLoop(&op);

            if (info.first) { // If pipelined
                op.setAttr("pipeline",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(op.getContext(), 32), 1));

                op.setAttr("II",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(op.getContext(), 32), info.second));
            }

            // Create a node representing the end of the loop
            int nxtNode = tg.addNode(beginNode, "static-while", 0, info.second);
            setIntvAttr(&op, std::make_pair(beginNode, endNode));
            currentNode = nxtNode;

        } else if (auto forOp = llvm::dyn_cast<mlir::tor::ForOp>(op)) {
            // Handle ForOp similarly to WhileOp
            currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);
            int beginNode = tg.addNode(currentNode, "static", 0);
            int endNode = buildTimeGraph(tg, *forOp.getBody(), beginNode, scheduler);

            auto info = scheduler->queryLoop(&op);

            if (info.first) { // If pipelined
                op.setAttr("pipeline",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(op.getContext(), 32), 1));

                op.setAttr("II",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(op.getContext(), 32), info.second));
            }

            // Create a node representing the end of the loop
            int nxtNode = tg.addNode(beginNode, "static-for", 0, info.second);
            setIntvAttr(&op, std::make_pair(beginNode, endNode));
            currentNode = nxtNode;

        } else {
            // Handle other operations
            if (llvm::isa<mlir::tor::YieldOp>(op) ||
                llvm::isa<mlir::tor::ConditionOp>(op) ||
                llvm::isa<mlir::tor::ReturnOp>(op) ||
                llvm::isa<mlir::arith::ConstantOp>(op)) {
                continue; // Skip certain operations
            }
            llvm::outs() << "adding " << op << " to be scheduled\n";
            vec.push_back(&op); // Buffer the operation for sequential processing
        }
    }

    llvm::outs() << "process any remaining buffered operations \n";
    // Process any remaining buffered operations
    if (!vec.empty())
        currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);

    llvm::outs() << "finished building time graph \n";
    return currentNode;
}

// Function to remove extra edges and update attributes
mlir::LogicalResult removeExtraEdges(mlir::tor::FuncOp funcOp, TimeGraph *tg) {
    std::vector<int> newId;
    tg->canonicalize(newId); // Canonicalize the time graph

    // Walk through the function operations to update attributes
    if (funcOp.walk([&](mlir::Operation *op) -> mlir::WalkResult {
        if (op->getDialect()->getNamespace() != mlir::tor::TORDialect::getDialectNamespace())
            return mlir::WalkResult::skip();

        // Update 'starttime' attribute
        if (auto starttime = op->getAttrOfType<mlir::IntegerAttr>("starttime")) {
            int t = starttime.getInt();
            op->setAttr("starttime",
                        mlir::IntegerAttr::get(
                                mlir::IntegerType::get(funcOp.getContext(), 32),
                                newId[t]));
        }

        // Update 'endtime' attribute
        if (auto endtime = op->getAttrOfType<mlir::IntegerAttr>("endtime")) {
            int t = endtime.getInt();
            op->setAttr("endtime",
                        mlir::IntegerAttr::get(
                                mlir::IntegerType::get(funcOp.getContext(), 32),
                                newId[t]));
        }

        return mlir::WalkResult::advance();
    }).wasInterrupted())
        return mlir::failure();

    return mlir::success();
}

// Function to schedule operations within a TOR function
mlir::LogicalResult scheduleOps(mlir::tor::FuncOp funcOp,
                                mlir::PatternRewriter &rewriter) {
    llvm::outs() << "scheduling function " << funcOp.getName() << "\n";

    using namespace scheduling;
    if (auto strategy = funcOp->getAttrOfType<mlir::StringAttr>("strategy")) {
        llvm::outs() << funcOp->getName() << " is dynamic. No static scheduling\n";
        if (strategy.getValue().str() == "dynamic")
            return mlir::success();
    }

    std::unique_ptr<SDCSchedule> scheduler =
            std::make_unique<SDCSchedule>(SDCSchedule(funcOp.getOperation()));

    if (mlir::succeeded(scheduler->runSchedule()))
        llvm::outs() << "Schedule Succeeded\n";
    else {
        llvm::outs() << "Schedule Failed\n";
        return mlir::failure();
    }

    llvm::outs() << "scheduling done " << funcOp.getName() << "\n";
    scheduler->printSchedule();

    TimeGraph tg; // Use the separated TimeGraph class

    buildTimeGraph(tg, funcOp.getRegion().front(), 0, scheduler.get());

    /*
    if (failed(removeExtraEdges(funcOp, &tg)))
        return mlir::failure();
    */

    tg.rewrite(funcOp.getBody(), rewriter);

    return mlir::success();
}

// ========================================
// MLIR Pattern Rewriting
// ========================================

namespace mlir {
    /**
     * @brief Pattern for lowering TOR functions.
     */
    struct FuncOpLowering : public OpRewritePattern<mlir::tor::FuncOp> {
        using OpRewritePattern<mlir::tor::FuncOp>::OpRewritePattern;

        /**
         * @brief Matches and rewrites TOR function operations.
         * @param funcOp The TOR function operation to match.
         * @param rewriter The pattern rewriter.
         * @return LogicalResult indicating success or failure.
         */
        LogicalResult
        matchAndRewrite(mlir::tor::FuncOp funcOp,
                        PatternRewriter &rewriter) const override {
            llvm::SmallVector<NamedAttribute, 4> attributes;
            // skip if already scheduled
            if (funcOp->getAttrOfType<BoolAttr>("scheduled")) {
                llvm::outs() << funcOp.getName() <<" is already scheduled\n";
                return failure();
            }

            // Exclude symbol and type attributes
            std::string typeAttrName = "function_type"; // Placeholder for actual type attr name
            for (const auto &attr : funcOp->getAttrs()) {
                if (attr.getName() == SymbolTable::getSymbolAttrName() ||
                    attr.getName() == typeAttrName) {
                    continue;
                }
                attributes.push_back(attr);
            }

            // Collect argument types
            llvm::SmallVector<mlir::Type, 8> argTypes;
            for (auto &arg : funcOp.getArguments()) {
                argTypes.push_back(arg.getType());
            }

            // Collect result types
            llvm::SmallVector<mlir::Type, 8> resTypes;
            for (auto &resultType : funcOp.getResultTypes()) {
                resTypes.push_back(resultType);
            }

            // Signature conversion for function arguments
            int arg_count = funcOp.getNumArguments() + 1; // Extra argument for control
            TypeConverter::SignatureConversion result(arg_count);
            for (unsigned idx = 0, e = argTypes.size(); idx < e; ++idx)
                result.addInputs(idx, argTypes[idx]);

            // Create the new function type
            auto func_type = rewriter.getFunctionType(argTypes, resTypes);

            // Create a new TOR function operation with the updated type and attributes
            auto newFuncOp = rewriter.create<mlir::tor::FuncOp>(
                    funcOp.getLoc(), funcOp.getName(), func_type, attributes);

            newFuncOp->setAttr("scheduled",mlir::BoolAttr::get(getContext(), true));

            // Inline the original function body into the new function
            rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                        newFuncOp.end());

            // Schedule the operations within the new function
            if (failed(scheduleOps(newFuncOp, rewriter)))
                return failure();

            llvm::outs() << "### scheduled \n"<< newFuncOp << "\n";
            // Erase the original function operation
            rewriter.eraseOp(funcOp);

            return success();
        }
    };

    /**
     * @brief Pass for scheduling TOR functions.
     */
    struct TORSchedulePass : public TORScheduleBase<TORSchedulePass> {
        /**
         * @brief Runs the pass on the operation.
         */
        void runOnOperation() override {
            mlir::tor::DesignOp designOp = getOperation();

            // Walk through all TOR functions within the design
            auto result = designOp.walk([&](tor::FuncOp op) {
                mlir::RewritePatternSet patterns(&getContext());
                patterns.insert<FuncOpLowering>(designOp.getContext());

                // Apply the rewriting patterns
                if (failed(applyOpPatternsAndFold({op}, std::move(patterns))))
                    WalkResult::interrupt(); // Interrupt if pattern application fails

                return WalkResult::advance();
            });

            // If the walk was interrupted, signal a pass failure
            if (result.wasInterrupted())
                signalPassFailure();

            llvm::outs() << designOp << "\n"       ;
        }
    };

    /**
     * @brief Factory function to create the TORSchedulePass.
     * @return A unique pointer to the newly created pass.
     */
    std::unique_ptr<mlir::OperationPass<mlir::tor::DesignOp>>
    createTORSchedulePass() {
        return std::make_unique<TORSchedulePass>();
    }

} // namespace mlir
